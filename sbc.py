#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SBC for Program B (freq-only, NO derivative likelihood, noise = sigma_y only)

Additions in this version:
  - Curve coverage metrics:
      * pointwise coverage: average across frequencies for 50% and 95%
      * simultaneous 95% band coverage using max standardized residuals
  - KS test for rank uniformity per parameter (against Uniform[0,1])
    with a SciPy-based path if available, and a built-in asymptotic fallback.

CLI example:
  python sbc_sigma_y_only.py --N 400 --R 6 --chains 4 --chain-method vectorized \
    --warmup 1000 --samples 4000 --threads 8 --procs 8 --seed 0 --freq-logspace 4 8 20
"""

import os
import argparse
import time
from multiprocessing import get_context, cpu_count

# ================= CLI =================
parser = argparse.ArgumentParser(description="SBC (no-derivative likelihood), sigma_y only")
parser.add_argument("--N", type=int, default=500, help="Number of SBC trials")
parser.add_argument("--R", type=int, default=6, help="Replicates per simulated dataset")
parser.add_argument("--chains", type=int, default=4, help="Num MCMC chains")
parser.add_argument("--chain-method", type=str, default="vectorized",
                    choices=["vectorized","sequential","parallel"])
parser.add_argument("--warmup", type=int, default=1000, help="Num warmup per chain")
parser.add_argument("--samples", type=int, default=4000, help="Num post samples per chain")
parser.add_argument("--target-accept", type=float, default=0.99)
parser.add_argument("--max-tree-depth", type=int, default=10)
parser.add_argument("--threads", type=int, default=None, help="OMP/MKL threads")
parser.add_argument("--procs", type=int, default=None, help="Processes for SBC outer loop")
parser.add_argument("--host-devices", type=int, default=None,
                    help="Host platform device count override for parallel chain method")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--freq-logspace", type=float, nargs=3, default=[4, 8, 20],
                    help="log10(f_min) log10(f_max) K  (default: 4 8 20)")
parser.add_argument("--freq-csv", type=str, default=None,
                    help="CSV file with frequency Hz (row/col). If set, overrides freq-logspace.")
args = parser.parse_args()

# ============ CPU / XLA ENV (set BEFORE importing jax) ============
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if args.threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(int(args.threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.threads))
if args.chain_method == "parallel" and args.host_devices is not None:
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS","") + \
        f" --xla_force_host_platform_device_count={int(args.host_devices)}"

# ============ Imports (after env) ============
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible
import numpyro.handlers as handlers

# Try SciPy for KS test; if not available, we will use a built-in approximation
try:
    from scipy.stats import kstest as scipy_kstest
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ============ Constants & Priors ============
eps0   = 8.854e-12
E      = 30000.0
D      = 1e-3
r      = 3e-6 + 55e-9      # total particle radius R
tmem   = 5.0e-9
tw     = 50e-9
t_shell = tmem + tw
ep_med = 80 * eps0
si_med = 0.1

# Priors (log-space Uniform)
log_eps_inner_low   = jnp.log(10*eps0);   log_eps_inner_high   = jnp.log(200*eps0)
log_sigma_inner_low = jnp.log(1e-6);      log_sigma_inner_high = jnp.log(1.0)
log_eps_outer_low   = jnp.log(10*eps0);   log_eps_outer_high   = jnp.log(200*eps0)
log_sigma_outer_low = jnp.log(1e-6);      log_sigma_outer_high = jnp.log(1.0)

# ============ Physics ============
@jax.jit
def eps_complex(f, eps_r, sigma):
    return eps_r - 1j * sigma / (2 * jnp.pi * f + 1e-300)

@jax.jit
def fun_innerouter(f, eps_inner, sigma_inner, eps_outer, sigma_outer):
    """
    Coated sphere effective permittivity with a = R / (R - t_shell).
    """
    eps_i = eps_complex(f, eps_inner, sigma_inner)
    eps_o = eps_complex(f, eps_outer, sigma_outer)
    a = r / (r - t_shell + 1e-300)
    num = (a**3 + 2.0 * (eps_i - eps_o) / (eps_i + 2.0 * eps_o + 1e-300))
    den = (a**3 -       (eps_i - eps_o) / (eps_i + 2.0 * eps_o + 1e-300))
    den = den + 1e-30j
    return eps_o * (num / den)

@jax.jit
def fun_fcm(f, eps_inner, sigma_inner, eps_outer, sigma_outer):
    ep_med_c = eps_complex(f, ep_med, si_med)
    ep_p     = fun_innerouter(f, eps_inner, sigma_inner, eps_outer, sigma_outer)
    denom = ep_p + 2.0 * ep_med_c
    denom = denom + (1e-300 + 1e-300j)
    return (ep_p - ep_med_c) / denom

@jax.jit
def theory_velocity(f, eps_inner, sigma_inner, eps_outer, sigma_outer):
    dep_const = ((r**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return dep_const * jnp.real(fun_fcm(f, eps_inner, sigma_inner, eps_outer, sigma_outer))

# ============ Model ============
def model_sigma_only(f_obs, V_obs, mask_obs):
    # dielectric parameters in log space
    log_eps_inner   = numpyro.sample("log_eps_inner",   dist.Uniform(log_eps_inner_low,   log_eps_inner_high))
    log_sigma_inner = numpyro.sample("log_sigma_inner", dist.Uniform(log_sigma_inner_low, log_sigma_inner_high))
    log_eps_outer   = numpyro.sample("log_eps_outer",   dist.Uniform(log_eps_outer_low,   log_eps_outer_high))
    log_sigma_outer = numpyro.sample("log_sigma_outer", dist.Uniform(log_sigma_outer_low, log_sigma_outer_high))

    p1 = jnp.exp(log_eps_inner)
    p2 = jnp.exp(log_sigma_inner)
    p3 = jnp.exp(log_eps_outer)
    p4 = jnp.exp(log_sigma_outer)

    v_pred = theory_velocity(f_obs, p1, p2, p3, p4)  # (K,)

    # noise: sigma_y only
    sigma_y = numpyro.sample("sigma_y", dist.HalfCauchy(5.0))

    with handlers.mask(mask=mask_obs):
        numpyro.sample("obs", dist.Normal(v_pred[None, :], sigma_y), obs=V_obs)

# ============ SBC utilities ============
def sample_true_params(key):
    k1, k2, k3, k4, ky = random.split(key, 5)
    return dict(
        log_eps_inner   = float(dist.Uniform(log_eps_inner_low,   log_eps_inner_high).sample(k1, ())),
        log_sigma_inner = float(dist.Uniform(log_sigma_inner_low, log_sigma_inner_high).sample(k2, ())),
        log_eps_outer   = float(dist.Uniform(log_eps_outer_low,   log_eps_outer_high).sample(k3, ())),
        log_sigma_outer = float(dist.Uniform(log_sigma_outer_low, log_sigma_outer_high).sample(k4, ())),
        sigma_y         = float(dist.HalfCauchy(5.0).sample(ky, ())),
    )

def simulate_dataset(true_params, f_obs, R=6, drop_prob=0.0, rng=None):
    """
    Simulate replicates:
      V[r,k] = v_true[k] + eps, eps ~ Normal(0, sigma_y)
    Returns V_filled, mask_obs (R,K)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    p1 = float(np.exp(true_params["log_eps_inner"]))
    p2 = float(np.exp(true_params["log_sigma_inner"]))
    p3 = float(np.exp(true_params["log_eps_outer"]))
    p4 = float(np.exp(true_params["log_sigma_outer"]))
    sigy = float(true_params["sigma_y"])

    v_true = np.array(theory_velocity(f_obs, p1, p2, p3, p4))
    K = v_true.shape[0]

    V = v_true[None, :] + rng.normal(0.0, sigy, size=(R, K))
    mask_obs = np.isfinite(V)
    if drop_prob > 0.0:
        drop = rng.uniform(size=V.shape) < drop_prob
        mask_obs = mask_obs & (~drop)
        V = np.where(mask_obs, V, 0.0)

    return V, mask_obs

def run_mcmc(f_obs, V_obs, mask_obs,
             num_warmup=800, num_samples=1200, n_chains=4,
             target_accept=0.99, max_tree_depth=10, seed=0, chain_method="vectorized"):
    kernel = NUTS(model_sigma_only, target_accept_prob=target_accept,
                  init_strategy=init_to_feasible(), max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=n_chains, chain_method=chain_method)
    key = random.PRNGKey(seed)
    mcmc.run(key,
             f_obs=jnp.array(f_obs),
             V_obs=jnp.array(V_obs),
             mask_obs=jnp.array(mask_obs))
    return mcmc.get_samples()

def rank_stat(post_samples, true_value, rng=None):
    """
    Randomized tie-breaking:
      rank = (k + u*(m+1)) / (S + 1)
      where k = # {s < t}, m = # {s == t}, u ~ U(0,1)
    """
    s = np.asarray(post_samples).ravel()
    S = s.size
    if rng is None:
        rng = np.random.default_rng()
    k = int(np.sum(s < true_value))
    m = int(np.sum(s == true_value))
    u = float(rng.uniform())
    return (k + u * (m + 1)) / (S + 1.0)

def in_credible_interval(post_samples, true_value, alpha=0.95):
    lo, hi = np.percentile(np.asarray(post_samples), [(1 - alpha) * 50, (1 + alpha) * 50])
    return (true_value >= lo) and (true_value <= hi)

def curves_from_post(post, f_obs_np):
    """
    Build deterministic theory curve samples (no observation noise) from posterior.
    Returns array of shape (S, K).
    """
    p1 = np.exp(np.asarray(post["log_eps_inner"]).ravel())
    p2 = np.exp(np.asarray(post["log_sigma_inner"]).ravel())
    p3 = np.exp(np.asarray(post["log_eps_outer"]).ravel())
    p4 = np.exp(np.asarray(post["log_sigma_outer"]).ravel())
    S = p1.size
    K = f_obs_np.size
    V = np.empty((S, K))
    # Loop over S samples; theory_velocity is JAX; call via np.array(...)
    for s in range(S):
        V[s] = np.array(theory_velocity(f_obs_np, p1[s], p2[s], p3[s], p4[s]))
    return V

# ---------- KS test utils ----------
def ks_test_uniform_01(x):
    """
    Kolmogorov-Smirnov test for Uniform(0,1).
    Returns (D, p_value). Uses SciPy if available; otherwise uses an asymptotic approximation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    # Clamp to [0,1] just in case of tiny numerical issues
    x = np.minimum(1.0, np.maximum(0.0, x))
    if SCIPY_AVAILABLE:
        res = scipy_kstest(x, "uniform")
        return float(res.statistic), float(res.pvalue)
    # Built-in path: compute D and asymptotic p-value
    xs = np.sort(x)
    i = np.arange(1, n+1, dtype=float)
    d_plus = np.max(i / n - xs)
    d_minus = np.max(xs - (i - 1) / n)
    D = float(max(d_plus, d_minus))
    # Asymptotic p-value using the Kolmogorov distribution approximation:
    # p = 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 lambda^2), where lambda = (sqrt(n)+0.12+0.11/sqrt(n))*D
    if D <= 0:
        return 0.0, 1.0
    sqn = np.sqrt(n)
    lam = (sqn + 0.12 + 0.11 / sqn) * D
    # Truncate the series when terms become negligible
    p = 0.0
    max_k = 100
    for k in range(1, max_k + 1):
        term = np.exp(-2.0 * (k*k) * (lam*lam))
        if k % 2 == 1:
            p += 2.0 * term
        else:
            p -= 2.0 * term
        if term < 1e-10:
            break
    p = max(0.0, min(1.0, p))
    return D, float(p)

def sbc_one_trial(task_args):
    (i, seed_i, f_obs_np, R, mcmc_cfg) = task_args
    f_obs = jnp.array(f_obs_np)
    key = random.PRNGKey(seed_i)
    true_p = sample_true_params(key)

    V_obs, mask_obs = simulate_dataset(true_p, f_obs=f_obs, R=R, rng=np.random.default_rng(seed_i+12345))
    post = run_mcmc(
        f_obs=f_obs,
        V_obs=V_obs, mask_obs=mask_obs,
        num_warmup=mcmc_cfg["warmup"], num_samples=mcmc_cfg["samples"],
        n_chains=mcmc_cfg["chains"], target_accept=mcmc_cfg["target_accept"],
        max_tree_depth=mcmc_cfg["max_tree_depth"], seed=seed_i,
        chain_method=mcmc_cfg["chain_method"]
    )

    out = {}
    params = ["log_eps_inner","log_sigma_inner","log_eps_outer","log_sigma_outer","sigma_y"]
    rng_local = np.random.default_rng(seed_i ^ 0xABCDEF)

    # parameter-level SBC stats
    for p in params:
        if p not in post:
            continue
        out[p + "_rank"]  = rank_stat(post[p], true_p[p], rng=rng_local)
        out[p + "_hit95"] = int(in_credible_interval(post[p], true_p[p], 0.95))
        out[p + "_hit50"] = int(in_credible_interval(post[p], true_p[p], 0.50))

    # curve-level coverage stats (deterministic theory curves, no obs noise)
    p1 = float(np.exp(true_p["log_eps_inner"]))
    p2 = float(np.exp(true_p["log_sigma_inner"]))
    p3 = float(np.exp(true_p["log_eps_outer"]))
    p4 = float(np.exp(true_p["log_sigma_outer"]))
    v_true = np.array(theory_velocity(f_obs, p1, p2, p3, p4))  # (K,)
    V_post = curves_from_post(post, np.asarray(f_obs))         # (S,K)

    # pointwise coverage
    q_lo95, q_hi95 = np.percentile(V_post, [2.5, 97.5], axis=0)
    q_lo50, q_hi50 = np.percentile(V_post, [25.0, 75.0], axis=0)
    hit95_pointwise = float(np.mean((v_true >= q_lo95) & (v_true <= q_hi95)))
    hit50_pointwise = float(np.mean((v_true >= q_lo50) & (v_true <= q_hi50)))

    # simultaneous 95% band via max standardized residuals
    mu = V_post.mean(axis=0)
    sd = V_post.std(axis=0, ddof=1) + 1e-12
    z = (V_post - mu) / sd                      # (S,K)
    z_true = (v_true - mu) / sd                 # (K,)
    M_s = np.max(np.abs(z), axis=1)             # per-sample max |z|
    thr = float(np.quantile(M_s, 0.95))         # 95% threshold
    hit95_simul = float(np.max(np.abs(z_true)) <= thr)

    out["curve_hit95_pointwise"] = hit95_pointwise
    out["curve_hit50_pointwise"] = hit50_pointwise
    out["curve_hit95_simul"] = hit95_simul

    return out

def sbc_loop_parallel(N, f_obs, R, seed, procs, mcmc_cfg):
    rng = np.random.default_rng(seed)
    tasks = [(i, int(rng.integers(1_000_000_000)), np.asarray(f_obs), R, mcmc_cfg) for i in range(N)]
    if procs is None:
        procs = min(cpu_count(), 26)
    ctx = get_context("spawn")
    with ctx.Pool(processes=procs) as pool:
        results = list(pool.map(sbc_one_trial, tasks))

    params = ["log_eps_inner","log_sigma_inner","log_eps_outer","log_sigma_outer","sigma_y"]
    ranks  = {p: [] for p in params}
    cover95 = {p: 0 for p in params}
    cover50 = {p: 0 for p in params}

    curve_stats_sum = {
        "curve_hit95_pointwise": 0.0,
        "curve_hit50_pointwise": 0.0,
        "curve_hit95_simul": 0.0,
    }

    for r in results:
        # parameter-level
        for p in params:
            key_rank = p + "_rank"
            key95 = p + "_hit95"
            key50 = p + "_hit50"
            if key_rank in r:
                ranks[p].append(r[key_rank])
            if key95 in r:
                cover95[p] += r[key95]
            if key50 in r:
                cover50[p] += r[key50]
        # curve-level
        for k in curve_stats_sum.keys():
            if k in r:
                curve_stats_sum[k] += float(r[k])

    for p in params:
        cover95[p] = cover95[p] / max(1, N)
        cover50[p] = cover50[p] / max(1, N)

    curve_stats_avg = {k: v / max(1, N) for k, v in curve_stats_sum.items()}
    return ranks, cover50, cover95, curve_stats_avg

def plot_rank_hist(ranks, out_prefix="sbc_rank", bins=20):
    for p, arr in ranks.items():
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        plt.figure(figsize=(4, 3))
        plt.hist(a, bins=bins, range=(0, 1), density=True, alpha=0.85, edgecolor="k")
        plt.axhline(1.0, linestyle="--", label="uniform baseline")
        plt.title(f"SBC rank: {p}")
        plt.xlabel("rank")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{p}.png", dpi=150)
        plt.close()

def ks_report(ranks):
    """
    Compute KS stats (D and p-value) for each parameter's rank list.
    Returns dict mapping param -> dict(D=..., p=..., n=...).
    """
    report = {}
    for p, arr in ranks.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        n = int(a.size)
        if n == 0:
            report[p] = {"D": float("nan"), "p": float("nan"), "n": 0}
            continue
        D, pval = ks_test_uniform_01(a)
        report[p] = {"D": float(D), "p": float(pval), "n": n}
    return report

# ============ Main ============
def main():
    # Frequency grid
    if args.freq_csv is not None:
        vals = np.loadtxt(args.freq_csv, delimiter=",")
        vals = np.asarray(vals).reshape(-1)
        f_obs = jnp.array(vals[np.isfinite(vals)])
    else:
        log_lo, log_hi, K = args.freq_logspace
        f_obs = jnp.logspace(float(log_lo), float(log_hi), int(K))

    # MCMC config
    mcmc_cfg = dict(
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        target_accept=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        chain_method=args.chain_method
    )

    print("=== CPU / JAX settings ===")
    print(f"JAX_PLATFORM_NAME={os.environ.get('JAX_PLATFORM_NAME')}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS','')}")
    print(f"Detected CPU cores: {cpu_count()}")
    print("=== SBC run plan ===")
    print(f"N={args.N}, R={args.R}, chains={args.chains}, chain_method={args.chain_method}, procs={args.procs}")
    print(f"warmup={args.warmup}, samples={args.samples}, target_accept={args.target_accept}, max_tree_depth={args.max_tree_depth}")
    print(f"freq: {len(np.asarray(f_obs))} points from "
          f"{float(np.min(np.asarray(f_obs))):.3e} to {float(np.max(np.asarray(f_obs))):.3e} Hz")
    print(f"SciPy available for KS: {SCIPY_AVAILABLE}")

    t0 = time.perf_counter()
    ranks, cover50, cover95, curve_stats = sbc_loop_parallel(
        N=args.N, f_obs=f_obs, R=args.R, seed=args.seed,
        procs=args.procs, mcmc_cfg=mcmc_cfg
    )
    t1 = time.perf_counter()

    print("\n=== SBC coverage (parameter, empirical) ===")
    def _fmt(d):
        return {k: f"{v:.2f}" for k, v in d.items()}
    print("50%:", _fmt(cover50))
    print("95%:", _fmt(cover95))

    print("\n=== SBC coverage (curve, empirical) ===")
    print(f"pointwise 50%: {curve_stats['curve_hit50_pointwise']:.3f}  (avg across frequencies)")
    print(f"pointwise 95%: {curve_stats['curve_hit95_pointwise']:.3f}  (avg across frequencies)")
    print(f"simultaneous 95% band: {curve_stats['curve_hit95_simul']:.3f}  (fraction of trials where whole curve is inside)")

    # KS tests for rank uniformity
    ks = ks_report(ranks)
    print("\n=== KS test for rank uniformity U(0,1) ===")
    for p in ["log_eps_inner","log_sigma_inner","log_eps_outer","log_sigma_outer","sigma_y"]:
        if p in ks:
            D = ks[p]["D"]; pv = ks[p]["p"]; n = ks[p]["n"]
            print(f"{p:>16s}: n={n:4d}, D={D:.4f}, p={pv:.4g}")

    print(f"\nTotal time: {t1 - t0:.1f} s")

    out_prefix = "sbc_rank_sigma_y_only"
    plot_rank_hist(ranks, out_prefix=out_prefix, bins=20)
    print(f"Saved rank histograms: {out_prefix}_*.png")

if __name__ == "__main__":
    main()
