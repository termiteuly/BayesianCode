#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEP synthetic: random mode = FIM-only double vs single-equivalent (supports --pairs averaging);
anchor mode optional MCMC (no derivative anywhere).

Changes for anchor mode (single):
- By default (anchor_noise=1), observational noise uses sigma_y:
    v_obs = v_true + Normal(0, sigma_y)
- You can turn it off with --anchor-noise 0

Dump levels:
- --dump-level 1 : print each case (pair) result, then a COMPLETE aggregated block
- --dump-level 0 : print only the COMPLETE aggregated block

Additions in this version:
- When using MCMC in anchor mode (--true-mode anchor --do-mcmc 1),
  plot posterior distributions for parameters on normal scale (exp of log params),
  and print posterior summaries (mean, std, 2.5%, 50%, 97.5%) on normal scale.
"""

import os
import argparse

# ---------- CLI ----------
p = argparse.ArgumentParser(description="DEP synthetic FIM-only (random, supports averaging) and optional MCMC (anchor)")
# modes
p.add_argument("--true-mode", type=str, default="anchor", choices=["anchor","random"])
p.add_argument("--model", type=str, default="single", choices=["single","double"],
               help="used only when true-mode=random; anchor forces single")
# anchor truth (single-shell + sigma_y)
p.add_argument("--anchor-eps-cyto", type=float, default=9.3384e-10)
p.add_argument("--anchor-sig-cyto", type=float, default=1.7548e-01)
p.add_argument("--anchor-eps-mem",  type=float, default=2.7917e-10)
p.add_argument("--anchor-sig-mem",  type=float, default=5.8656e-04)
p.add_argument("--anchor-sigma-y",  type=float, default=1.3303e+01)
# NEW: anchor noise control (default = use sigma_y)
p.add_argument("--anchor-noise", type=int, default=1,
               help="1: in anchor mode, add Normal(0, sigma_y) noise; 0: noise-free")

# HMC config (anchor mode only, and only if --do-mcmc=1)
p.add_argument("--do-mcmc", type=int, default=1, help="anchor mode only; random mode ignores this")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--chains", type=int, default=3)
p.add_argument("--chain-method", type=str, default="sequential", choices=["parallel","vectorized","sequential"])
p.add_argument("--warmup", type=int, default=10000)
p.add_argument("--samples", type=int, default=20000)
p.add_argument("--target-accept", type=float, default=0.99)
p.add_argument("--max-tree-depth", type=int, default=10)

# FIM screening (random truths)
p.add_argument("--fim-filter", type=str, default="none", choices=["none","min_eig","cond"])
p.add_argument("--min-eig-thresh", type=float, default=1e-3)
p.add_argument("--cond-thresh", type=float, default=1e4)
p.add_argument("--random-tries", type=int, default=30)

# FIM whitening scale for random mode (std-based)
p.add_argument("--eps-scale", type=float, default=1e-3, help="eps_tol = max(eps_scale*std(v), 0.02*std(v)) in random mode")

# equivalence options (random double only)
p.add_argument("--do-equivalence", type=int, default=1, help="1: run double->single equivalence + FIM compare (random double)")
p.add_argument("--pairs", type=int, default=500, help="number of random repeats for averaging (random mode)")

# output control (only screen printing)
p.add_argument("--dump-level", type=int, default=0, choices=[0,1], help="1: per-case + aggregated; 0: aggregated only")

# plotting (only first pair shows spectrum alignment if model=double and do-equivalence=1)
p.add_argument("--plot", type=int, default=1)
args = p.parse_args()

# ---------- Setup ----------
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, jacfwd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

USE_NUMPYRO = (args.true_mode == "anchor" and int(args.do_mcmc) == 1)
if USE_NUMPYRO:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import arviz as az

jax.config.update("jax_enable_x64", True)

# ---------- Small utilities for numerical robustness ----------
def is_all_finite(x):
    x = np.asarray(x)
    return np.all(np.isfinite(x))

def safe_std(x):
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return 0.0
    return float(np.nanstd(x[finite]))

# ---------- Helpers ----------
def mean_std(arr):
    a = np.asarray(arr, float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float("nan")
    return float(np.mean(a)), float(np.std(a, ddof=1))

def print_sens_and_cos_from_Jw(Jw, labels):
    Jw = np.asarray(Jw)
    sens = np.sqrt(np.sum(Jw**2, axis=0))
    Jn = Jw / (np.linalg.norm(Jw, axis=0, keepdims=True) + 1e-30)
    cosM = Jn.T @ Jn
    print("\n-- Sensitivity norms (||d v / d log(theta)|| / eps_tol) --")
    for k, n in zip(labels, sens):
        print(f"{k:16s}: {n: .3e}")
    print("\n-- Column direction cosine (J) --")
    hdr = " " * 22 + " ".join([f"{k:>12s}" for k in labels])
    print(hdr)
    for i, ki in enumerate(labels):
        row = " ".join([f"{cosM[i,j]:12.3f}" for j in range(len(labels))])
        print(f"{ki:>22s} {row}")
    return sens, cosM

def print_pretty_sens_cos(title, sens_vec, cosM, labels):
    print(f"\n-- Sensitivity & cosine ({title}) --\n")
    print("-- Sensitivity norms (||d v / d log(theta)|| / eps_tol) --")
    for k, n in zip(labels, np.asarray(sens_vec)):
        print(f"{k:16s}: {n: .3e}")
    print("\n-- Column direction cosine (J) --")
    header = " " * 22 + " ".join([f"{k:>12s}" for k in labels])
    print(header)
    for i, ki in enumerate(labels):
        row = " ".join([f"{cosM[i,j]:12.3f}" for j in range(len(labels))])
        print(f"{ki:>22s} {row}")

# ---------- Constants ----------
eps0   = 8.854e-12
E      = 30000.0
D      = 1e-3
ep_med = 80.0 * eps0
si_med = 0.1

# Geometry aligned with MCMC code
cyto_r = 3.0e-6          # cytoplasm radius (inner-most)
tmem   = 5.0e-9          # membrane thickness
tw     = 50.0e-9         # wall (outer shell) thickness
t_shell = tmem + tw      # total shell thickness used by single-shell
r_total = cyto_r + tmem + tw  # total particle radius R

# ---------- Frequency grid ----------
FREQ = jnp.array([
    5e4, 7.5e4,
    1e5, 2e5, 3e5, 4e5, 5e5, 7.5e5,
    1e6, 2e6, 3e6, 4e6, 5e6, 7.5e6,
    1e7, 2e7, 3e7, 4e7, 5e7, 7.5e7,
    1e8, 2e8, 3e8, 4e8, 5e8
])

# ---------- Physics (geometry aligned to MCMC) ----------
@jit
def eps_complex(f, eps_abs, sigma):
    return eps_abs - 1j * sigma / (2.0 * jnp.pi * f + 1e-300)

@jit
def eff_perm_two_layer_totalR(f, R, t, eps_core, sig_core, eps_shell, sig_shell):
    """
    Two-layer coated sphere using total outer radius R and shell thickness t.
    a = R / (R - t), aligned with MCMC fun_innerouter.
    """
    eps_c = eps_complex(f, eps_core, sig_core)
    eps_s = eps_complex(f, eps_shell, sig_shell)
    Rin = R - t
    a = R / (Rin + 1e-300)
    num = (a**3 + 2.0 * (eps_c - eps_s) / (eps_c + 2.0 * eps_s + 1e-300))
    den = (a**3 -       (eps_c - eps_s) / (eps_c + 2.0 * eps_s + 1e-300))
    den = den + 1e-30j
    return eps_s * (num / den)

@jit
def eff_perm_three_layer_totalR(f, R, t1, t2,
                                eps_core, sig_core,
                                eps_s1, sig_s1,
                                eps_s2, sig_s2):
    """
    Three-layer coated sphere built as two successive two-layer steps.
    Step 1: core + shell1 up to Rmid = R - t2, with a1 = Rmid / (Rmid - t1)
    Step 2: (core+shell1)_eff + shell2 up to R, with a2 = R / Rmid
    """
    Rmid = R - t2          # after adding shell1
    Rin  = Rmid - t1       # pure core radius

    # Step 1: core with shell1 to Rmid
    eps_c = eps_complex(f, eps_core, sig_core)
    eps_s1_c = eps_complex(f, eps_s1, sig_s1)
    a1 = Rmid / (Rin + 1e-300)
    num1 = (a1**3 + 2.0 * (eps_c - eps_s1_c) / (eps_c + 2.0 * eps_s1_c + 1e-300))
    den1 = (a1**3 -       (eps_c - eps_s1_c) / (eps_c + 2.0 * eps_s1_c + 1e-300))
    den1 = den1 + 1e-30j
    eps_eff_1 = eps_s1_c * (num1 / den1)  # effective core after step 1

    # Step 2: (core+shell1)_eff with shell2 to R
    eps_s2_c = eps_complex(f, eps_s2, sig_s2)
    a2 = R / (Rmid + 1e-300)
    num2 = (a2**3 + 2.0 * (eps_eff_1 - eps_s2_c) / (eps_eff_1 + 2.0 * eps_s2_c + 1e-300))
    den2 = (a2**3 -       (eps_eff_1 - eps_s2_c) / (eps_eff_1 + 2.0 * eps_s2_c + 1e-300))
    den2 = den2 + 1e-30j
    return eps_s2_c * (num2 / den2)

@jit
def fcm_from_ep(f, ep_particle):
    ep_med_c = eps_complex(f, ep_med, si_med)
    denom = (ep_particle + 2.0 * ep_med_c)
    # Extra tiny bumps on both real and imaginary parts for stability
    denom = denom + (1e-300 + 1e-300j)
    return (ep_particle - ep_med_c) / denom

@jit
def v_single(f, eps_cyto, sig_cyto, eps_shell, sig_shell):
    # single-shell built at total radius R = r_total with shell thickness t_shell
    ep_p = eff_perm_two_layer_totalR(f, r_total, t_shell, eps_cyto, sig_cyto, eps_shell, sig_shell)
    pref = ((r_total**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return pref * jnp.real(fcm_from_ep(f, ep_p))

@jit
def v_double(f, eps_cyto, sig_cyto, eps_mem, sig_mem, eps_wall, sig_wall):
    # double-shell: stepwise to total radius r_total with t1=tmem, t2=tw
    ep_p = eff_perm_three_layer_totalR(f, r_total, tmem, tw,
                                       eps_cyto, sig_cyto,
                                       eps_mem, sig_mem,
                                       eps_wall, sig_wall)
    pref = ((r_total**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return pref * jnp.real(fcm_from_ep(f, ep_p))

# ---------- Priors (log-uniform) for random truth ----------
def prior_sample_param(key, name):
    if "eps" in name:
        return jnp.exp(jax.random.uniform(key, (), minval=jnp.log(10*eps0), maxval=jnp.log(200*eps0)))
    else:
        return jnp.exp(jax.random.uniform(key, (), minval=jnp.log(1e-6),   maxval=jnp.log(1.0)))

def sample_truth_from_priors(key, names):
    out = {}
    subs = random.split(key, len(names))
    for sk, nm in zip(subs, names):
        out[nm] = float(prior_sample_param(sk, nm))
    return out

# ---------- Equivalence fit: double -> single outer ----------
def fit_single_equiv_outer(f_vec, eps_cyto, sig_cyto, eps_mem, sig_mem, eps_wall, sig_wall,
                           bounds_log=((np.log(10*eps0), np.log(200*eps0)),
                                       (np.log(1e-6),   np.log(1.0))),
                           x0_log=None, weight=None, verbose=False):
    f = np.asarray(f_vec, float)
    vD = np.asarray(v_double(f, eps_cyto, sig_cyto, eps_mem, sig_mem, eps_wall, sig_wall))
    if weight is None:
        weight = np.ones_like(vD)
    if x0_log is None:
        x0_log = np.array([np.log(eps_mem), np.log(sig_mem)], float)
    (l_eps, u_eps), (l_sig, u_sig) = bounds_log

    def obj(xlog):
        em = np.exp(xlog[0]); sm = np.exp(xlog[1])
        vS = np.asarray(v_single(f, eps_cyto, sig_cyto, em, sm))
        r = vS - vD
        return float(np.sum(weight * r * r))

    res = minimize(obj, x0_log, method="L-BFGS-B", bounds=[(l_eps,u_eps),(l_sig,u_sig)])
    em_eq = float(np.exp(res.x[0])); sm_eq = float(np.exp(res.x[1]))
    vS = np.asarray(v_single(f, eps_cyto, sig_cyto, em_eq, sm_eq))
    resid = vS - vD
    rmse = float(np.sqrt(np.mean(resid**2)))
    nrmse = rmse / (np.ptp(vD) + 1e-12)
    max_rel = float(np.max(np.abs(resid)) / (np.max(np.abs(vD)) + 1e-12))
    info = dict(success=bool(res.success), nrmse=nrmse, max_rel=max_rel, iters=int(res.nit), msg=str(res.message))
    if verbose:
        print("[equiv-fit] success:", info["success"], "nrmse=", info["nrmse"], "max_rel=", info["max_rel"])
    return em_eq, sm_eq, info

# ---------- FIM (log-parameter, epsilon-whitened) ----------
def fim_logspace(model_kind, f_vec, params_abs_list, eps_tol):
    f = jnp.array(f_vec)
    phi0 = jnp.log(jnp.array(params_abs_list))
    if model_kind == "single":
        def m(phi): th = jnp.exp(phi); return v_single(f, th[0], th[1], th[2], th[3])
    else:
        def m(phi): th = jnp.exp(phi); return v_double(f, th[0], th[1], th[2], th[3], th[4], th[5])
    J = jacfwd(m)(phi0)
    Jw = J / (eps_tol + 1e-30)
    F = Jw.T @ Jw
    F = 0.5 * (F + F.T)
    s = jnp.linalg.svd(Jw, compute_uv=False)
    cond = float(s.max() / jnp.maximum(s.min(), 1e-300))
    sign, logdet = jnp.linalg.slogdet(F)
    evals = jnp.linalg.eigvalsh(F)
    return onp.asarray(F), float(cond), int(sign), float(logdet), onp.asarray(evals), onp.asarray(Jw)

# ---------- Random mode: one draw ----------
def random_one_draw(rng_key, mode, dump_each=False, show_plot=False, is_first_pair=False):
    if mode == "single":
        names = ["eps_cyto","sigma_cyto","eps_mem","sigma_mem"]
    else:
        names = ["eps_cyto","sigma_cyto","eps_mem","sigma_mem","eps_wall","sigma_wall"]
    rng_key, sk = random.split(rng_key)
    attempts = 0
    while True:
        attempts += 1
        truth = sample_truth_from_priors(sk, names)

        # forward model
        if mode == "double":
            v_tmp = v_double(FREQ, truth["eps_cyto"], truth["sigma_cyto"],
                             truth["eps_mem"], truth["sigma_mem"], truth["eps_wall"], truth["sigma_wall"])
        else:
            v_tmp = v_single(FREQ, truth["eps_cyto"], truth["sigma_cyto"], truth["eps_mem"], truth["sigma_mem"])

        # finite check and robust std
        ok = True
        if not is_all_finite(v_tmp):
            ok = False
        else:
            y_std = safe_std(v_tmp) + 1e-12
            eps_tol = max(args.eps_scale * y_std, 0.02 * y_std)
            if not np.isfinite(eps_tol) or eps_tol <= 0.0:
                ok = False
            else:
                F, cond, sign, logdetF, evals, Jw = fim_logspace(
                    mode, FREQ, [truth[k] for k in names], eps_tol
                )
                if args.fim_filter == "min_eig":
                    ok = bool(float(evals.min()) >= float(args.min_eig_thresh))
                elif args.fim_filter == "cond":
                    ok = bool(float(cond) <= float(args.cond_thresh))

        if ok or attempts >= args.random_tries:
            break
        rng_key, sk = random.split(rng_key)

    # If still not ok (exceeded tries), set safe fallbacks to avoid crashes
    if not ok:
        diag = dict(cond=float("nan"), min_eig=float("nan"), logdet=float("nan"),
                    attempts=int(attempts), eps_tol=float("nan"))
        result = {"truth": truth, "diag": diag, "Jw": np.full((len(FREQ), len(names)), np.nan)}
        return rng_key, result

    diag = dict(cond=float(cond), min_eig=float(evals.min()), logdet=float(logdetF),
                attempts=int(attempts), eps_tol=float(eps_tol))

    if dump_each:
        print("\n=== Random truth ({}): attempts={} ===".format(mode.upper(), attempts))
        for k in names:
            print(f"{k:16s} = {truth[k]:.6e}")
        print("FIM cond={:.3e}, lambda_min={:.3e}, logdet={:.3f}".format(diag["cond"], diag["min_eig"], diag["logdet"]))
        _ = print_sens_and_cos_from_Jw(Jw, names)

    result = {
        "truth": truth, "diag": diag,
        "Jw": Jw
    }

    # double -> single equivalence block (only for double)
    if mode == "double" and int(args.do_equivalence) == 1:
        # fit equivalent single
        em_eq, sm_eq, info = fit_single_equiv_outer(
            FREQ, truth["eps_cyto"], truth["sigma_cyto"],
            truth["eps_mem"], truth["sigma_mem"], truth["eps_wall"], truth["sigma_wall"],
            verbose=dump_each
        )

        vD = np.asarray(v_double(FREQ, truth["eps_cyto"], truth["sigma_cyto"],
                                 truth["eps_mem"], truth["sigma_mem"], truth["eps_wall"], truth["sigma_wall"]))
        if is_all_finite(vD):
            y_std_d = safe_std(vD) + 1e-12
            eps_tol_d = max(args.eps_scale * y_std_d, 0.02 * y_std_d)

            Fd, cond_d, _, logdet_d, evals_d, Jw_d = fim_logspace(
                "double", FREQ,
                [truth["eps_cyto"], truth["sigma_cyto"], truth["eps_mem"], truth["sigma_mem"], truth["eps_wall"], truth["sigma_wall"]],
                eps_tol_d
            )
            Fs, cond_s, _, logdet_s, evals_s, Jw_s = fim_logspace(
                "single", FREQ,
                [truth["eps_cyto"], truth["sigma_cyto"], em_eq, sm_eq],
                eps_tol_d
            )

            if dump_each:
                print("\n-- Single-equivalent outer (from double truth) --")
                print(f"eps_shell_eq = {em_eq:.6e}")
                print(f"sigma_shell_eq = {sm_eq:.6e}")
                print(f"alignment nrmse={info['nrmse']:.3e}, max_rel={info['max_rel']:.3e}")
                print("\n=== FIM comparison ===")
                print(f"DOUBLE : cond={cond_d:.3e}, lambda_min={float(evals_d.min()):.3e}, logdet={logdet_d:.3f}")
                print(f"SINGLE*: cond={cond_s:.3e}, lambda_min={float(evals_s.min()):.3e}, logdet={logdet_s:.3f}  (*equivalent)")
                print_sens_and_cos_from_Jw(Jw_d, ["eps_cyto","sigma_cyto","eps_mem","sigma_mem","eps_wall","sigma_wall"])
                print_sens_and_cos_from_Jw(Jw_s, ["eps_cyto","sigma_cyto","eps_shell_eq","sigma_shell_eq"])

                if show_plot and is_first_pair:
                    vS = np.asarray(v_single(FREQ, truth["eps_cyto"], truth["sigma_cyto"], em_eq, sm_eq))
                    plt.figure(figsize=(8,4.5))
                    plt.semilogx(np.asarray(FREQ), vD, "o-", label="Double truth")
                    plt.semilogx(np.asarray(FREQ), vS, "--", label="Single equiv")
                    plt.xlabel("Frequency (Hz)"); plt.ylabel("DEP velocity")
                    plt.title("Spectra alignment (double vs single-equivalent)")
                    plt.grid(True, which="both", ls=":")
                    plt.legend(); plt.tight_layout(); plt.show()

            result.update({
                "double": {"cond": float(cond_d), "lmin": float(evals_d.min()), "logdet": float(logdet_d), "Jw": np.asarray(Jw_d)},
                "single_equiv": {"cond": float(cond_s), "lmin": float(evals_s.min()), "logdet": float(logdet_s), "Jw": np.asarray(Jw_s)},
                "equiv_params": {"eps_shell_eq": float(em_eq), "sigma_shell_eq": float(sm_eq)},
                "align": {"nrmse": float(info["nrmse"]), "max_rel": float(info["max_rel"])}
            })
        else:
            # mark as skipped equivalence due to non-finite spectrum
            result.update({"equiv_skipped": True})

    return rng_key, result

# ---------- Random mode: averaging over --pairs ----------
def run_random_fim_only():
    mode = args.model.lower()
    if mode == "single" and args.do_equivalence == 1:
        print("[warn] do-equivalence is ignored for model=single (needs double truth).")

    rng_key = random.PRNGKey(args.seed)

    # scalars
    cond_double, lmin_double, logdet_double = [], [], []
    cond_single, lmin_single, logdet_single = [], [], []
    nrmse_list, maxrel_list = [], []

    # Jw collectors for averaged sensitivity and cosine
    Jw_double_list, Jw_single_list = [], []
    eq_eps_list, eq_sig_list = [], []

    for i in range(1, args.pairs + 1):
        dump_each = (args.dump_level == 1)
        show_plot = (args.plot == 1)
        rng_key, res = random_one_draw(
            rng_key, mode,
            dump_each=dump_each,
            show_plot=show_plot,
            is_first_pair=(i == 1)
        )

        if mode == "double" and int(args.do_equivalence) == 1 and ("equiv_skipped" not in res):
            cond_double.append(res["double"]["cond"])
            lmin_double.append(res["double"]["lmin"])
            logdet_double.append(res["double"]["logdet"])
            Jw_double_list.append(res["double"]["Jw"])

            cond_single.append(res["single_equiv"]["cond"])
            lmin_single.append(res["single_equiv"]["lmin"])
            logdet_single.append(res["single_equiv"]["logdet"])
            Jw_single_list.append(res["single_equiv"]["Jw"])

            nrmse_list.append(res["align"]["nrmse"])
            maxrel_list.append(res["align"]["max_rel"])

            eq_eps_list.append(res["equiv_params"]["eps_shell_eq"])
            eq_sig_list.append(res["equiv_params"]["sigma_shell_eq"])
        else:
            cond_single.append(res["diag"]["cond"])
            lmin_single.append(res["diag"]["min_eig"])
            logdet_single.append(res["diag"]["logdet"])
            Jw_single_list.append(res["Jw"])

    # Aggregated
    print("\n=== Aggregated over pairs (summary) ===")

    def avg_sens_cos_from_Jw_list(Jw_list):
        Jw_list = [np.asarray(x) for x in Jw_list if x is not None and np.all(np.isfinite(x))]
        if len(Jw_list) == 0:
            return None, None
        sens_list = []
        cos_list = []
        for Jw in Jw_list:
            s = np.sqrt(np.sum(Jw**2, axis=0))
            Jn = Jw / (np.linalg.norm(Jw, axis=0, keepdims=True) + 1e-30)
            c = Jn.T @ Jn
            sens_list.append(s)
            cos_list.append(c)
        sens_mean = np.mean(np.stack(sens_list, axis=0), axis=0)
        cos_mean  = np.mean(np.stack(cos_list, axis=0), axis=0)
        return sens_mean, cos_mean

    if mode == "double" and int(args.do_equivalence) == 1:
        if len(cond_double) > 0:
            m_cd, s_cd = mean_std(cond_double)
            m_ldmin_d, s_ldmin_d = mean_std(lmin_double)
            m_logdet_d, s_logdet_d = mean_std(logdet_double)
            m_cs, s_cs = mean_std(cond_single)
            m_ldmin_s, s_ldmin_s = mean_std(lmin_single)
            m_logdet_s, s_logdet_s = mean_std(logdet_single)
            m_nrmse, s_nrmse = mean_std(nrmse_list)
            m_mr, s_mr = mean_std(maxrel_list)

            print("DOUBLE : cond={:.3e} +/- {:.3e}, lambda_min={:.3e} +/- {:.3e}, logdet={:.3f} +/- {:.3f}".format(
                m_cd, s_cd, m_ldmin_d, s_ldmin_d, m_logdet_d, s_logdet_d))
            print("SINGLE*: cond={:.3e} +/- {:.3e}, lambda_min={:.3e} +/- {:.3e}, logdet={:.3f} +/- {:.3f}  (*equivalent)".format(
                m_cs, s_cs, m_ldmin_s, s_ldmin_s, m_logdet_s, s_logdet_s))
            print("Alignment (double->single): NRMSE={:.3e} +/- {:.3e}, MaxRel={:.3e} +/- {:.3e}".format(
                m_nrmse, s_nrmse, m_mr, s_mr))

            if len(eq_eps_list) > 0:
                m_eqe, s_eqe = mean_std(eq_eps_list)
                m_eqs, s_eqs = mean_std(eq_sig_list)
                print("Equiv params: eps_shell_eq={:.6e} +/- {:.6e}, sigma_shell_eq={:.6e} +/- {:.6e}".format(
                    m_eqe, (s_eqe if not np.isnan(s_eqe) else 0.0), m_eqs, (s_eqs if not np.isnan(s_eqs) else 0.0)
                ))

            sens_d_mean, cos_d_mean = avg_sens_cos_from_Jw_list(Jw_double_list)
            sens_s_mean, cos_s_mean = avg_sens_cos_from_Jw_list(Jw_single_list)

            if sens_d_mean is not None and cos_d_mean is not None:
                print_pretty_sens_cos("DOUBLE, aggregated", sens_d_mean, cos_d_mean,
                                      ["eps_cyto","sigma_cyto","eps_mem","sigma_mem","eps_wall","sigma_wall"])
            if sens_s_mean is not None and cos_s_mean is not None:
                print_pretty_sens_cos("SINGLE-EQUIV, aggregated", sens_s_mean, cos_s_mean,
                                      ["eps_cyto","sigma_cyto","eps_shell_eq","sigma_shell_eq"])
        else:
            print("[note] All double->single equivalence cases were skipped due to non-finite spectra.")
    else:
        m_cs, s_cs = mean_std(cond_single)
        m_ldmin_s, s_ldmin_s = mean_std(lmin_single)
        m_logdet_s, s_logdet_s = mean_std(logdet_single)
        print("SINGLE : cond={:.3e} +/- {:.3e}, lambda_min={:.3e} +/- {:.3e}, logdet={:.3f} +/- {:.3f}".format(
            m_cs, s_cs, m_ldmin_s, s_ldmin_s, m_logdet_s, s_logdet_s))

        sens_s_mean, cos_s_mean = avg_sens_cos_from_Jw_list(Jw_single_list)
        if sens_s_mean is not None and cos_s_mean is not None:
            print_pretty_sens_cos("SINGLE, aggregated", sens_s_mean, cos_s_mean,
                                  ["eps_cyto","sigma_cyto","eps_mem","sigma_mem"])

# ---------- Anchor mode: single truth (+sigma_y), FIM uses sigma_y; optional MCMC ----------
def run_anchor_optional_mcmc():
    truth = {
        "eps_cyto": float(args.anchor_eps_cyto),
        "sigma_cyto": float(args.anchor_sig_cyto),
        "eps_mem": float(args.anchor_eps_mem),
        "sigma_mem": float(args.anchor_sig_mem),
        "sigma_y": float(args.anchor_sigma_y),
    }
    print("\n=== Truth (ANCHOR, SINGLE) ===")
    for k in ["eps_cyto","sigma_cyto","eps_mem","sigma_mem","sigma_y"]:
        print(f"{k:16s} = {truth[k]:.6e}")

    # true clean spectrum
    v_true = v_single(FREQ, truth["eps_cyto"], truth["sigma_cyto"], truth["eps_mem"], truth["sigma_mem"])

    # observational noise (default use sigma_y)
    if int(args.anchor_noise) == 1:
        rng_key = random.PRNGKey(args.seed)
        v_obs = v_true + random.normal(rng_key, shape=v_true.shape) * float(truth["sigma_y"])
        print("[anchor] observational noise: Normal(0, sigma_y)")
    else:
        v_obs = v_true
        print("[anchor] observational noise: OFF (noise-free)")

    # FIM whiten scale = sigma_y (aligned with observational std in anchor mode)
    eps_tol = max(float(truth["sigma_y"]), 1e-12)
    _, cond, _, logdetF, evals, Jw = fim_logspace("single", FREQ,
        [truth["eps_cyto"], truth["sigma_cyto"], truth["eps_mem"], truth["sigma_mem"]], eps_tol)
    print("\n=== FIM at anchor truth (single) ===")
    print("cond={:.3e}, lambda_min={:.3e}, logdet={:.3f}".format(
        cond, float(evals.min()), logdetF))
    _ = print_sens_and_cos_from_Jw(Jw, ["eps_cyto","sigma_cyto","eps_mem","sigma_mem"])

    if not USE_NUMPYRO:
        print("\n[MCMC skipped: do-mcmc=0 or true-mode!=anchor]")
        return

    # optional MCMC (anchor) to double-check identifiability under this clean setup
    def model_numpyro(f_obs, v_obs_in):
        log_eps_cyto = numpyro.sample("log_eps_cyto", dist.Uniform(jnp.log(10*eps0), jnp.log(200*eps0)))
        log_sig_cyto = numpyro.sample("log_sigma_cyto", dist.Uniform(jnp.log(1e-6), jnp.log(1.0)))
        log_eps_mem  = numpyro.sample("log_eps_mem",  dist.Uniform(jnp.log(10*eps0), jnp.log(200*eps0)))
        log_sig_mem  = numpyro.sample("log_sigma_mem",dist.Uniform(jnp.log(1e-6), jnp.log(1.0)))
        v_pred = v_single(f_obs, jnp.exp(log_eps_cyto), jnp.exp(log_sig_cyto), jnp.exp(log_eps_mem), jnp.exp(log_sig_mem))
        sigma_y = numpyro.sample("sigma_y", dist.HalfCauchy(5.0))
        numpyro.sample("obs", dist.Normal(v_pred, sigma_y), obs=v_obs_in)

    kernel = NUTS(model_numpyro, target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, chain_method=args.chain_method)
    mcmc.run(random.PRNGKey(args.seed), f_obs=FREQ, v_obs_in=v_obs, extra_fields=("num_steps","accept_prob","diverging","energy"))
    idata = az.from_numpyro(mcmc)
    print("\n[MCMC summary (NumPyro/ArviZ, raw)]")
    print(az.summary(idata, round_to=4))

    # -------- Posterior extraction on NORMAL SCALE and plotting --------
    samples = mcmc.get_samples(group_by_chain=False)

    # Convert to numpy arrays and transform logs to normal scale
    eps_cyto_post  = np.asarray(np.exp(onp.array(samples["log_eps_cyto"])), dtype=float)
    sigma_cyto_post= np.asarray(np.exp(onp.array(samples["log_sigma_cyto"])), dtype=float)
    eps_mem_post   = np.asarray(np.exp(onp.array(samples["log_eps_mem"])), dtype=float)
    sigma_mem_post = np.asarray(np.exp(onp.array(samples["log_sigma_mem"])), dtype=float)
    sigma_y_post   = np.asarray(onp.array(samples["sigma_y"]), dtype=float)

    # Print posterior summaries on normal scale
    def summarize(name, arr):
        arr = np.asarray(arr, float)
        q = np.quantile(arr, [0.025, 0.5, 0.975])
        print(f"{name:16s} mean={arr.mean():.6e}  std={arr.std(ddof=1):.6e}  "
              f"q2.5={q[0]:.6e}  q50={q[1]:.6e}  q97.5={q[2]:.6e}")

    print("\n[Posterior summary on NORMAL scale]")
    summarize("eps_cyto",   eps_cyto_post)
    summarize("sigma_cyto", sigma_cyto_post)
    summarize("eps_mem",    eps_mem_post)
    summarize("sigma_mem",  sigma_mem_post)
    summarize("sigma_y",    sigma_y_post)

    # Plot posterior histograms on normal scale
    if int(args.plot) == 1:
        params = [
            ("eps_cyto",   eps_cyto_post),
            ("sigma_cyto", sigma_cyto_post),
            ("eps_mem",    eps_mem_post),
            ("sigma_mem",  sigma_mem_post),
            ("sigma_y",    sigma_y_post),
        ]
        n = len(params)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        plt.figure(figsize=(4.5*ncols, 3.5*nrows))
        for i, (name, arr) in enumerate(params, 1):
            plt.subplot(nrows, ncols, i)
            plt.hist(arr, bins=50, density=True)
            plt.xlabel(name + " (normal scale)")
            plt.ylabel("density")
            plt.title("Posterior: " + name)
        plt.tight_layout()
        plt.show()

    print("[Anchor sigma_y truth] {:.6e}".format(truth["sigma_y"]))

# ---------- Main ----------
if args.true_mode == "random":
    run_random_fim_only()
else:
    run_anchor_optional_mcmc()

print("\n=== Run config ===")
print("true_mode={} | model={}".format(args.true_mode, args.model))
print("anchor_noise={} | eps_scale={} | pairs={}".format(int(args.anchor_noise), args.eps_scale, args.pairs))
print("do_mcmc={} (ignored in random mode) | dump_level={}".format(int(args.do_mcmc), args.dump_level))
