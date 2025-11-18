# -*- coding: utf-8 -*-
# ==========================================================
# DEP Bayesian pipeline (freq-only, single-shell, NO tau, NO derivative likelihood)
# Geometry switch removed: use a = r / (r - t_shell)
# With cyto/shell naming
# ==========================================================

import os, time
os.environ["JAX_PLATFORM_NAME"] = os.getenv("JAX_PLATFORM_NAME", "cpu")

# ---------- Reproducibility & Config ----------
import numpy as onp
SEED = int(os.getenv("SEED", "0")); onp.random.seed(SEED)

NCHAINS        = int(os.getenv("NCHAINS", "3"))
CHAIN_METHOD   = os.getenv("CHAIN_METHOD", "sequential")  # "sequential" | "vectorized" | "parallel"
TARGET_ACCEPT  = float(os.getenv("TARGET_ACCEPT", "0.99"))
NUM_WARMUP     = int(os.getenv("NUM_WARMUP", "15000"))
NUM_SAMPLES    = int(os.getenv("NUM_SAMPLES", "45000"))
MAX_TREE_DEPTH = int(os.getenv("MAX_TREE_DEPTH", "10"))

PPC_MAX_SAMPLES   = int(os.getenv("PPC_MAX_SAMPLES", "4000"))
M_NOISE_DRAWS     = int(os.getenv("M_NOISE_DRAWS", "10"))

IDATA_NC_PATH     = os.getenv("IDATA_NC", "dep_single_shell_notau_idata.nc")
SUMMARY_CSV_PATH  = os.getenv("SUMMARY_CSV", "dep_posterior_summary_notau.csv")
DATA_XLSX         = os.getenv("DATA_XLSX", "/home/cjli/Python/MCMCpy/DEPvelocityRAW_Control.xlsx")

# Convergence gates
RHAT_MAX = float(os.getenv("RHAT_MAX", "1.01"))
ESS_MIN  = int(os.getenv("ESS_MIN", "1000"))
MCSE_SD_RATIO_MAX = float(os.getenv("MCSE_SD_RATIO_MAX", "0.05"))

# ---------- Packages ----------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible
import numpyro.handlers as handlers
import arviz as az

# ---------- Constants ----------
eps0   = 8.854e-12
E      = 30000.0
D      = 1e-3
r      = 3e-6 + 55e-9
tmem   = 5.0e-9
tw     = 50.0e-9
t_shell = tmem + tw
ep_med = 80 * eps0
si_med = 0.1

# ---------- Priors (log-space Uniform) ----------
log_eps_cyto_low   = jnp.log(10*eps0);   log_eps_cyto_high   = jnp.log(200*eps0)
log_sigma_cyto_low = jnp.log(1e-6);      log_sigma_cyto_high = jnp.log(1.0)
log_eps_shell_low  = jnp.log(10*eps0);   log_eps_shell_high  = jnp.log(200*eps0)
log_sigma_shell_low= jnp.log(1e-6);      log_sigma_shell_high= jnp.log(1.0)

# ---------- Data loading ----------
def load_one_xlsx(path):
    df = pd.read_excel(path, header=None)
    freq_raw = onp.asarray(df.iloc[0].values, dtype=float)
    V_raw    = onp.asarray(df.iloc[1:].values, dtype=float)
    col_ok = onp.isfinite(freq_raw) & onp.any(onp.isfinite(V_raw), axis=0)
    freq = freq_raw[col_ok]
    V    = V_raw[:, col_ok]
    ord_idx = onp.argsort(freq)
    freq = freq[ord_idx]; V = V[:, ord_idx]
    mask_obs_np = onp.isfinite(V)
    V_filled_np = onp.where(mask_obs_np, V, 0.0)
    return freq, V, mask_obs_np, V_filled_np

t0_data = time.perf_counter()
freq, V, mask_obs_np, V_filled_np = load_one_xlsx(DATA_XLSX)
Exp_Frequency = jnp.array(freq)
V_obs         = jnp.array(V_filled_np)
mask_obs      = jnp.array(mask_obs_np)

with onp.errstate(invalid="ignore"):
    Exp_Result_mean_np = onp.nanmean(onp.where(mask_obs_np, V, onp.nan), axis=0)
Exp_Result_mean = jnp.array(Exp_Result_mean_np)

t1_data = time.perf_counter()

# ---------- Electrical model ----------
def eps_complex(f, eps_r, sigma):
    return eps_r - 1j * sigma / (2 * jnp.pi * f + 1e-300)

def fun_cytoshell(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    eps_i = eps_complex(f, eps_cyto,  sigma_cyto)
    eps_o = eps_complex(f, eps_shell, sigma_shell)
    a = (r / (r - t_shell))
    num = (a**3 + 2.0 * (eps_i - eps_o) / (eps_i + 2.0 * eps_o + 1e-300))
    den = (a**3 -       (eps_i - eps_o) / (eps_i + 2.0 * eps_o + 1e-300))
    den = den + 1e-30j
    return eps_o * (num / den)

def fun_fcm(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    ep_med_c = eps_complex(f, ep_med, si_med)
    ep_p     = fun_cytoshell(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell)
    return (ep_p - ep_med_c) / (ep_p + 2.0 * ep_med_c + 1e-300)

def theory_velocity(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    dep_const = ((r**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return dep_const * jnp.real(fun_fcm(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell))

# ---------- Model ----------
def model_freq_only_notau(f_obs, V_obs, mask_obs):
    log_eps_cyto   = numpyro.sample("log_eps_cyto",   dist.Uniform(log_eps_cyto_low,   log_eps_cyto_high))
    log_sigma_cyto = numpyro.sample("log_sigma_cyto", dist.Uniform(log_sigma_cyto_low, log_sigma_cyto_high))
    log_eps_shell  = numpyro.sample("log_eps_shell",  dist.Uniform(log_eps_shell_low,  log_eps_shell_high))
    log_sigma_shell= numpyro.sample("log_sigma_shell",dist.Uniform(log_sigma_shell_low,log_sigma_shell_high))

    p1 = jnp.exp(log_eps_cyto)
    p2 = jnp.exp(log_sigma_cyto)
    p3 = jnp.exp(log_eps_shell)
    p4 = jnp.exp(log_sigma_shell)

    v_pred = theory_velocity(f_obs, p1, p2, p3, p4)

    sigma_y = numpyro.sample("sigma_y", dist.HalfCauchy(5.0))
    sigma_total = sigma_y

    with handlers.mask(mask=mask_obs):
        numpyro.sample("obs", dist.Normal(v_pred[None, :], sigma_total), obs=V_obs)

# ---------- HMC fit ----------
t0_hmc = time.perf_counter()
master_key = random.PRNGKey(SEED)
kernel = NUTS(
    model_freq_only_notau,
    target_accept_prob=TARGET_ACCEPT,
    init_strategy=init_to_feasible(),
    max_tree_depth=MAX_TREE_DEPTH
)
mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
            num_chains=NCHAINS, chain_method=CHAIN_METHOD)

mcmc.run(
    master_key,
    f_obs=Exp_Frequency,
    V_obs=V_obs,
    mask_obs=mask_obs,
    extra_fields=("num_steps", "accept_prob", "diverging", "energy"),
)

print("\n[Geometry] coated-sphere a = r / (r - t_shell)\n")
mcmc.print_summary()
posterior_samples = mcmc.get_samples()
idata = az.from_numpyro(mcmc)
t1_hmc = time.perf_counter()

# ---------- Diagnostics ----------
summ = az.summary(idata, round_to=4)
print("\n[Convergence gates & summary]")
print(summ)
rhat_ok = (summ["r_hat"] < RHAT_MAX).all()
ess_ok  = (summ["ess_bulk"] > ESS_MIN).all() & (summ["ess_tail"] > ESS_MIN).all()
mcse_ok = (summ["mcse_mean"] / summ["sd"] < MCSE_SD_RATIO_MAX).fillna(False).all()
num_total = NCHAINS * NUM_SAMPLES
ess_ratio_bulk = summ["ess_bulk"] / num_total
ess_ratio_tail = summ["ess_tail"] / num_total
ess_ok_ratio = (ess_ratio_bulk > 0.1).all() & (ess_ratio_tail > 0.1).all()
print(f"rhat_ok={rhat_ok}, ess_ok={ess_ok}, mcse_ok={mcse_ok}")
print(f"ESS ratio bulk min={ess_ratio_bulk.min():.3f}, tail min={ess_ratio_tail.min():.3f}")
print(f"ess_ok (absolute)={ess_ok}, ess_ok (ratio)={ess_ok_ratio}")

def _fetch_num_steps_flat(idata, mcmc):
    try:
        ss = idata.sample_stats
        if "num_steps" in ss:
            arr = onp.asarray(ss["num_steps"]).reshape(-1)
            if arr.size > 0:
                return arr.astype(float)
        if "tree_depth" in ss:
            td = onp.asarray(ss["tree_depth"]).reshape(-1).astype(float)
            return onp.power(2.0, td) - 1.0
    except Exception:
        pass
    try:
        ef = mcmc.get_extra_fields()
        if "num_steps" in ef:
            arr = onp.asarray(ef["num_steps"]).reshape(-1)
            if arr.size > 0:
                return arr.astype(float)
    except Exception:
        pass
    return None

# Extra NUTS diagnostics
try:
    ss = idata.sample_stats

    div = onp.asarray(ss["diverging"]).reshape(-1).astype(bool) if "diverging" in ss else onp.array([])
    if div.size > 0:
        div_count = int(div.sum())
        div_total = int(div.size)
        div_rate = float(div.mean())
        div_ok = (div_rate <= 0.005)
        print(f"Divergences: count={div_count} / draws={div_total}  rate={div_rate:.4%}  (ok={div_ok})")
    else:
        print("Divergences: not recorded")

    ns_flat = _fetch_num_steps_flat(idata, mcmc)
    if ns_flat is not None and ns_flat.size > 0:
        max_steps_theory = (2 ** MAX_TREE_DEPTH) - 1
        ns_max = int(onp.nanmax(ns_flat))
        p95 = float(onp.nanpercentile(ns_flat, 95))
        p99 = float(onp.nanpercentile(ns_flat, 99))
        hits = (ns_flat >= max_steps_theory - 1e-9)
        hit_rate = float(onp.mean(hits))
        nearly = (ns_flat >= 0.9 * max_steps_theory)
        near_rate = float(onp.mean(nearly))
        tree_ok = (hit_rate <= 0.005) and (near_rate <= 0.05)
        print("Tree depth (approx via num_steps):")
        print(f"  max_steps_theory={max_steps_theory}, ns_max={ns_max}, p95={p95:.1f}, p99={p99:.1f}")
        print(f"  hits==max: rate={hit_rate:.4%}  near(>=90% max): rate={near_rate:.4%}  (ok={tree_ok})")
    else:
        print("Tree depth: 'num_steps' unavailable.")

    try:
        bfmi = az.bfmi(idata)
        bfmi_min = float(onp.asarray(bfmi).min())
        print(f"E-BFMI (min across chains): {bfmi_min:.3f}  (rule-of-thumb: > 0.3 is ok)")
    except Exception:
        print("E-BFMI: skipped")

    if "acceptance_rate" in ss:
        acc = onp.asarray(ss["acceptance_rate"]).reshape(-1)
    elif "accept_prob" in ss:
        acc = onp.asarray(ss["accept_prob"]).reshape(-1)
    else:
        acc = onp.array([])
    if acc.size > 0:
        acc_med = float(onp.nanmedian(acc))
        acc_q25 = float(onp.nanpercentile(acc, 25))
        acc_q75 = float(onp.nanpercentile(acc, 75))
        print(f"Accept stat: median={acc_med:.3f}, IQR=({acc_q25:.3f},{acc_q75:.3f})")
    else:
        print("Accept stat: not recorded")

    try:
        last_state = getattr(mcmc, "last_state", None) or getattr(mcmc, "_last_state", None)
        step_list = []
        if last_state is not None:
            adapt_state = getattr(last_state, "adapt_state", None)
            if adapt_state is not None and hasattr(adapt_state, "step_size"):
                ss_val = onp.asarray(adapt_state.step_size)
                step_list = ss_val.ravel().tolist()
        if len(step_list) > 0:
            print("Step size (per-chain): median={:.3e}, min={:.3e}, max={:.3e}".format(
                onp.nanmedian(step_list), onp.nanmin(step_list), onp.nanmax(step_list)))
        else:
            print("Step size: not available.")
    except Exception:
        print("Step size: retrieval failed.")
except Exception as e:
    print(f"[Extra diagnostics skipped] {e}")

# ---------- Posterior mean curve on dense & observed grid ----------
f_dense = jnp.logspace(4, 8, 1000)
post_mean = {k: posterior_samples[k].mean() for k in posterior_samples}
p1 = jnp.exp(post_mean["log_eps_cyto"])
p2 = jnp.exp(post_mean["log_sigma_cyto"])
p3 = jnp.exp(post_mean["log_eps_shell"])
p4 = jnp.exp(post_mean["log_sigma_shell"])
v_fit_dense   = theory_velocity(f_dense, p1, p2, p3, p4)
v_fit_obs     = theory_velocity(Exp_Frequency, p1, p2, p3, p4)

# ---------- Latent bands ----------
def latent_bands_on_obs(posterior_samples, Exp_Frequency, keep_max, seed):
    p1_all = jnp.exp(posterior_samples["log_eps_cyto"])
    p2_all = jnp.exp(posterior_samples["log_sigma_cyto"])
    p3_all = jnp.exp(posterior_samples["log_eps_shell"])
    p4_all = jnp.exp(posterior_samples["log_sigma_shell"])
    S = int(p1_all.shape[0])
    keep = min(keep_max, S)
    sel = onp.random.default_rng(seed).choice(S, size=keep, replace=False)
    def _vel_on_obs(p1, p2, p3, p4):
        return theory_velocity(Exp_Frequency, p1, p2, p3, p4)
    v_all = jnp.stack(
        [
            _vel_on_obs(
                p1_all[sel][i],
                p2_all[sel][i],
                p3_all[sel][i],
                p4_all[sel][i],
            )
            for i in range(min(keep, S))
        ],
        axis=0,
    )
    v_mean  = jnp.mean(v_all, axis=0)
    v_lo95  = jnp.percentile(v_all, 2.5,  axis=0)
    v_hi95  = jnp.percentile(v_all, 97.5, axis=0)
    v_q25   = jnp.percentile(v_all, 25.0, axis=0)
    v_q75   = jnp.percentile(v_all, 75.0, axis=0)
    return sel, v_all, v_mean, v_lo95, v_hi95, v_q25, v_q75

idx_ss, v_lat_all, v_lat_mean, v_lat_lo95, v_lat_hi95, v_lat_q25, v_lat_q75 = \
    latent_bands_on_obs(posterior_samples, Exp_Frequency, PPC_MAX_SAMPLES, SEED)

def latent_bands_on_dense(posterior_samples, f_dense, idx_sel):
    p1s = jnp.exp(posterior_samples["log_eps_cyto"])[idx_sel]
    p2s = jnp.exp(posterior_samples["log_sigma_cyto"])[idx_sel]
    p3s = jnp.exp(posterior_samples["log_eps_shell"])[idx_sel]
    p4s = jnp.exp(posterior_samples["log_sigma_shell"])[idx_sel]
    def _vel_on_dense(p1, p2, p3, p4):
        return theory_velocity(f_dense, p1, p2, p3, p4)
    v_all_dense = jnp.stack(
        [_vel_on_dense(p1s[i], p2s[i], p3s[i], p4s[i]) for i in range(len(idx_sel))],
        axis=0,
    )
    v_lo = jnp.percentile(v_all_dense, 2.5,  axis=0)
    v_hi = jnp.percentile(v_all_dense, 97.5, axis=0)
    v_mean = jnp.mean(v_all_dense, axis=0)
    return v_mean, v_lo, v_hi, v_all_dense

v_lat_mean_dense, v_lat_lo_dense, v_lat_hi_dense, v_lat_all_dense = latent_bands_on_dense(
    posterior_samples, f_dense, idx_ss
)

# ---------- Full predictive (observed grid) ----------
rng_np = onp.random.default_rng(SEED + 1234)
K = Exp_Frequency.shape[0]
rep_counts = mask_obs_np.sum(axis=0).astype(int)
n_k = onp.maximum(rep_counts, 1)
S_keep = len(idx_ss)

sigma_y_all = onp.asarray(posterior_samples["sigma_y"]).reshape(-1)[idx_ss]
sigma_total_all = sigma_y_all

v_lat_all_np = onp.asarray(v_lat_all)
M = max(1, M_NOISE_DRAWS)
if M == 1:
    eps_nf = rng_np.standard_normal(size=(S_keep, K))
    y_mm_all = v_lat_all_np + sigma_total_all[:, None] * eps_nf
else:
    eps_nf = rng_np.standard_normal(size=(S_keep * M, K))
    v_rep = onp.repeat(v_lat_all_np, M, axis=0)
    sig_rep = onp.repeat(sigma_total_all, M)[:, None]
    y_mm_all = v_rep + sig_rep * eps_nf

y_lo95_mm = onp.percentile(y_mm_all, 2.5,  axis=0)
y_hi95_mm = onp.percentile(y_mm_all, 97.5, axis=0)
y_lo50_mm = onp.percentile(y_mm_all, 25.0, axis=0)
y_hi50_mm = onp.percentile(y_mm_all, 75.0, axis=0)

# mean-level predictive bands (observed grid)
if M == 1:
    eps_nm = rng_np.standard_normal(size=(S_keep, K))
    y_mean_all = v_lat_all_np + (sigma_total_all[:, None] / onp.sqrt(n_k)[None, :]) * eps_nm
else:
    eps_nm = rng_np.standard_normal(size=(S_keep * M, K))
    v_rep = onp.repeat(v_lat_all_np, M, axis=0)
    sig_rep = onp.repeat(sigma_total_all, M)[:, None] / onp.sqrt(n_k)[None, :]
    y_mean_all = v_rep + sig_rep * eps_nm

ybar_lo95 = onp.percentile(y_mean_all, 2.5,  axis=0)
ybar_hi95 = onp.percentile(y_mean_all, 97.5, axis=0)
ybar_lo50 = onp.percentile(y_mean_all, 25.0, axis=0)
ybar_hi50 = onp.percentile(y_mean_all, 75.0, axis=0)

# ---------- Coverage metrics (observed grid) ----------
def _multiple_measurement_level_coverage(V_np, mask_np, lo, hi):
    m = mask_np & onp.isfinite(V_np)
    if not onp.any(m):
        return onp.nan
    inside = (V_np >= lo[None, :]) & (V_np <= hi[None, :])
    return float(onp.mean(inside[m]))

def _mean_level_weighted_coverage(v_mean_np, lo, hi, weights):
    m = (
        onp.isfinite(v_mean_np)
        & onp.isfinite(lo)
        & onp.isfinite(hi)
        & (weights > 0)
    )
    if not onp.any(m):
        return onp.nan
    w = weights[m].astype(float)
    w = w / onp.sum(w)
    inside = ((v_mean_np[m] >= lo[m]) & (v_mean_np[m] <= hi[m])).astype(float)
    return float(onp.sum(w * inside))

def _latent_level_coverage(v_mean_np, lo, hi):
    m = onp.isfinite(v_mean_np) & onp.isfinite(lo) & onp.isfinite(hi)
    if not onp.any(m):
        return onp.nan
    inside = ((v_mean_np[m] >= lo[m]) & (v_mean_np[m] <= hi[m])).astype(float)
    return float(onp.mean(inside))

cov50_mm = _multiple_measurement_level_coverage(V, mask_obs_np, y_lo50_mm, y_hi50_mm)
cov95_mm = _multiple_measurement_level_coverage(V, mask_obs_np, y_lo95_mm, y_hi95_mm)
cov50_mean_w = _mean_level_weighted_coverage(
    Exp_Result_mean_np, ybar_lo50, ybar_hi50, rep_counts
)
cov95_mean_w = _mean_level_weighted_coverage(
    Exp_Result_mean_np, ybar_lo95, ybar_hi95, rep_counts
)
cov50_latent = _latent_level_coverage(
    Exp_Result_mean_np, onp.asarray(v_lat_q25), onp.asarray(v_lat_q75)
)
cov95_latent = _latent_level_coverage(
    Exp_Result_mean_np, onp.asarray(v_lat_lo95), onp.asarray(v_lat_hi95)
)

# ---------- Error metrics ----------
def _rmse_r2_vs_mean(y_true_mean_np, y_pred_np):
    y = onp.asarray(y_true_mean_np).copy()
    yp = onp.asarray(y_pred_np).copy()
    m = onp.isfinite(y) & onp.isfinite(yp)
    y = y[m]
    yp = yp[m]
    if y.size == 0:
        return onp.nan, onp.nan
    rmse = float(onp.sqrt(onp.mean((y - yp) ** 2)))
    sse = float(onp.sum((y - yp) ** 2))
    sst = float(onp.sum((y - onp.mean(y)) ** 2))
    r2 = float("nan") if sst <= 0 else (1.0 - sse / sst)
    return rmse, r2

def _rmse_r2_all_measurements(V_matrix_np, yhat_vec_np, mask_np):
    diff = yhat_vec_np[None, :] - V_matrix_np
    resid = diff[mask_np]
    if resid.size == 0:
        return onp.nan, onp.nan
    sse = float(onp.sum(resid**2))
    rmse = float(onp.sqrt(onp.mean(resid**2)))
    y = V_matrix_np[mask_np]
    y_bar = float(onp.mean(y))
    sst = float(onp.sum((y - y_bar) ** 2))
    r2 = float("nan") if sst <= 0 else (1.0 - sse / sst)
    return rmse, r2

# ---------- MAP & metrics ----------
def compute_log_prob(sample_dict):
    conditioned = numpyro.handlers.condition(model_freq_only_notau, sample_dict)
    trace = numpyro.handlers.trace(conditioned).get_trace(
        f_obs=Exp_Frequency, V_obs=V_obs, mask_obs=mask_obs
    )
    return sum(
        site["log_prob"].sum() for site in trace.values() if "log_prob" in site
    )

def _ith_sample(i):
    return {k: posterior_samples[k][i] for k in posterior_samples}

num_draws = len(posterior_samples["log_eps_cyto"])
rng = onp.random.default_rng(SEED)
cand = rng.choice(num_draws, size=min(num_draws, 5000), replace=False)
log_probs = jnp.array([compute_log_prob(_ith_sample(i)) for i in cand])
map_idx = int(cand[int(jnp.argmax(log_probs))])
map_sample = _ith_sample(map_idx)

map_p1 = jnp.exp(map_sample["log_eps_cyto"])
map_p2 = jnp.exp(map_sample["log_sigma_cyto"])
map_p3 = jnp.exp(map_sample["log_eps_shell"])
map_p4 = jnp.exp(map_sample["log_sigma_shell"])
v_map_obs   = theory_velocity(Exp_Frequency, map_p1, map_p2, map_p3, map_p4)
v_map_dense = theory_velocity(f_dense,       map_p1, map_p2, map_p3, map_p4)

rmse_all_latent, r2_all_latent = _rmse_r2_all_measurements(
    V, onp.asarray(v_lat_mean), mask_obs_np
)
rmse_all_pmean,  r2_all_pmean  = _rmse_r2_all_measurements(
    V, onp.asarray(v_fit_obs),  mask_obs_np
)
rmse_latent_vs_mean, r2_latent_vs_mean = _rmse_r2_vs_mean(Exp_Result_mean_np, v_lat_mean)
rmse_pmean_vs_mean,  r2_pmean_vs_mean  = _rmse_r2_vs_mean(Exp_Result_mean_np, v_fit_obs)
rmse_all_map, r2_all_map = _rmse_r2_all_measurements(
    V, onp.asarray(v_map_obs), mask_obs_np
)
rmse_map_vs_mean, r2_map_vs_mean = _rmse_r2_vs_mean(Exp_Result_mean_np, v_map_obs)

print("\n[Predictive coverage (marginal, single numbers)]")
print(f"Latent-level (vs measurement-mean): 50%={cov50_latent:.3f}, 95%={cov95_latent:.3f}")
print(f"Multiple-measurement level       : 50%={cov50_mm:.3f}, 95%={cov95_mm:.3f}")
print(f"Measurement-mean level (weighted): 50%={cov50_mean_w:.3f}, 95%={cov95_mean_w:.3f}")

print("\n[Fit metrics (RMSE and R^2 grouped)]")
print(f"Latent mean (all measurements)         : RMSE={rmse_all_latent:.6e}, R^2={r2_all_latent:.6f}")
print(f"Param mean  (all measurements)         : RMSE={rmse_all_pmean:.6e},  R^2={r2_all_pmean:.6f}")
print(f"Latent mean vs measurement-mean        : RMSE={rmse_latent_vs_mean:.6e}, R^2={r2_latent_vs_mean:.6f}")
print(f"Param mean  vs measurement-mean        : RMSE={rmse_pmean_vs_mean:.6e}, R^2={r2_pmean_vs_mean:.6f}")
print(f"MAP        (all measurements)          : RMSE={rmse_all_map:.6e},     R^2={r2_all_map:.6f}")
print(f"MAP        vs measurement-mean         : RMSE={rmse_map_vs_mean:.6e}, R^2={r2_map_vs_mean:.6f}")

# ---------- Posterior (median [2.5,97.5]%, real scale) ----------
post = az.extract(idata)
param_list = ["log_eps_cyto","log_sigma_cyto","log_eps_shell","log_sigma_shell","sigma_y"]
print("\n[Posterior (median [2.5,97.5]%, real scale)]")
for p in param_list:
    if p not in idata.posterior.data_vars:
        continue
    arr = onp.asarray(idata.posterior[p]).reshape(
        -1, *onp.asarray(idata.posterior[p]).shape[2:]
    )
    name = p
    if p.startswith("log_"):
        arr = onp.exp(arr)
        name = p.replace("log_", "")
    if arr.ndim == 1:
        q2p5, q50, q97p5 = onp.percentile(arr, [2.5, 50, 97.5])
        print(f"{name:20s}: {q50:.4e}  [{q2p5:.4e}, {q97p5:.4e}]")
    else:
        flat = arr.reshape(arr.shape[0], -1)
        q2p5, q50, q97p5 = onp.percentile(flat, [2.5, 50, 97.5], axis=0)
        print(
            f"{name:20s}: vector (len={flat.shape[1]})  median/CI not expanded here"
        )

sigma_total_samples = onp.asarray(idata.posterior["sigma_y"]).reshape(-1)
q2p5, q50, q97p5 = onp.percentile(sigma_total_samples, [2.5, 50, 97.5])
print(f"\n{'sigma_total (== sigma_y)':20s}: {q50:.4e}  [{q2p5:.4e}, {q97p5:.4e}]")

# ---------- Posterior histograms ----------
try:
    hist_keys = ["log_eps_cyto","log_sigma_cyto","log_eps_shell","log_sigma_shell","sigma_y"]
    vals = []; names = []
    for k in hist_keys:
        if k not in idata.posterior.data_vars:
            continue
        arr = onp.asarray(idata.posterior[k]).reshape(-1)
        if k.startswith("log_"):
            arr = onp.exp(arr)
            nm = k.replace("log_", "")
        else:
            nm = k
        vals.append(arr); names.append(nm)

    if len(vals) > 0:
        n_params = len(vals)
        n_cols = min(3, n_params)
        n_rows = int(onp.ceil(n_params / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows))
        axs = onp.array(axs).reshape(-1) if n_params > 1 else [axs]

        for i in range(n_params):
            ax = axs[i]
            arr_real = vals[i]
            name = names[i]

            ax.hist(arr_real, bins=60, density=True, alpha=0.85, edgecolor="k")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))

            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.set_title("Posterior " + name)

        for j in range(i+1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"[Posterior histogram plotting skipped] {e}")

# ---------- Trace plots and pair plots ----------
try:
    trace_vars = ["log_eps_cyto", "log_sigma_cyto", "log_eps_shell", "log_sigma_shell", "sigma_y"]
    present_vars = [v for v in trace_vars if v in idata.posterior.data_vars]
    if len(present_vars) > 0:
        az.plot_trace(idata, var_names=present_vars)
        plt.tight_layout()
        plt.show()
    else:
        print("[Trace plots] No matching variables.")
except Exception as e:
    print(f"[Trace plots skipped] {e}")

try:
    pair_vars = ["log_eps_cyto", "log_sigma_cyto", "log_eps_shell", "log_sigma_shell"]
    pair_vars_present = [v for v in pair_vars if v in idata.posterior.data_vars]
    if len(pair_vars_present) > 1:
        az.plot_pair(idata, var_names=pair_vars_present, kind="kde", marginals=True)
        plt.tight_layout()
        plt.show()
    else:
        print("[Pair plots] Not enough variables.")
except Exception as e:
    print(f"[Pair plots skipped] {e}")

# ---------- Plots: latent / multiple-measurement / measurement-mean (dense bands) ----------
try:
    freq_obs_np   = onp.asarray(Exp_Frequency)
    freq_dense_np = onp.asarray(f_dense)

    # Dense predictive bands for multiple-measurement and measurement-mean
    v_lat_all_dense_np = onp.asarray(v_lat_all_dense)  # (S_keep, K_dense)
    K_dense = v_lat_all_dense_np.shape[1]
    S_keep_dense = v_lat_all_dense_np.shape[0]

    rng_plot = onp.random.default_rng(SEED + 9999)
    M_dense = max(1, M_NOISE_DRAWS)

    # Multiple-measurement level predictive (dense)
    if M_dense == 1:
        eps_nf_dense = rng_plot.standard_normal(size=(S_keep_dense, K_dense))
        y_mm_all_dense = v_lat_all_dense_np + sigma_total_all[:, None] * eps_nf_dense
    else:
        eps_nf_dense = rng_plot.standard_normal(size=(S_keep_dense * M_dense, K_dense))
        v_rep_dense = onp.repeat(v_lat_all_dense_np, M_dense, axis=0)
        sig_rep_dense = onp.repeat(sigma_total_all, M_dense)[:, None]
        y_mm_all_dense = v_rep_dense + sig_rep_dense * eps_nf_dense

    y_lo95_mm_dense = onp.percentile(y_mm_all_dense, 2.5, axis=0)
    y_hi95_mm_dense = onp.percentile(y_mm_all_dense, 97.5, axis=0)

    # Measurement-mean predictive bands (dense)
    n_eff = float(onp.mean(rep_counts))
    if not onp.isfinite(n_eff) or n_eff <= 0:
        n_eff = 1.0
    denom = onp.sqrt(n_eff)

    if M_dense == 1:
        eps_nm_dense = rng_plot.standard_normal(size=(S_keep_dense, K_dense))
        y_mean_all_dense = v_lat_all_dense_np + (sigma_total_all[:, None] / denom) * eps_nm_dense
    else:
        eps_nm_dense = rng_plot.standard_normal(size=(S_keep_dense * M_dense, K_dense))
        v_rep_dense = onp.repeat(v_lat_all_dense_np, M_dense, axis=0)
        sig_rep_dense = onp.repeat(sigma_total_all, M_dense)[:, None] / denom
        y_mean_all_dense = v_rep_dense + sig_rep_dense * eps_nm_dense

    ybar_lo95_dense = onp.percentile(y_mean_all_dense, 2.5, axis=0)
    ybar_hi95_dense = onp.percentile(y_mean_all_dense, 97.5, axis=0)
    ybar_lo50_dense = onp.percentile(y_mean_all_dense, 25.0, axis=0)
    ybar_hi50_dense = onp.percentile(y_mean_all_dense, 75.0, axis=0)

    # 1) Latent CI (dense) + posterior mean (dense) + MAP (dense) + individual measurements + measurement mean
    plt.figure(figsize=(10,6))
    plt.fill_between(
        freq_dense_np,
        onp.asarray(v_lat_lo_dense),
        onp.asarray(v_lat_hi_dense),
        alpha=0.35,
        label="95% latent CI (dense)",
    )
    plt.semilogx(
        freq_dense_np,
        onp.asarray(v_fit_dense),
        "-",
        lw=2,
        label="Posterior mean",
    )
    plt.semilogx(
        freq_dense_np,
        onp.asarray(v_map_dense),
        "--",
        lw=2,
        label="MAP fit",
    )
    Rplot = V_obs.shape[0]
    for r_i in range(Rplot):
        plt.scatter(
            freq_obs_np,
            onp.asarray(onp.where(mask_obs_np[r_i, :], V[r_i, :], onp.nan)),
            s=12,
            alpha=0.5,
            label="_nolegend_",
        )
    plt.plot(
        freq_obs_np,
        onp.asarray(Exp_Result_mean_np),
        "o",
        ms=5,
        label="Measurement mean",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title("Latent CI (dense) + posterior mean + MAP + individual measurements")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Latent band (dense) + measurement mean (points)
    plt.figure(figsize=(9,5))
    plt.fill_between(
        freq_dense_np,
        onp.asarray(v_lat_lo_dense),
        onp.asarray(v_lat_hi_dense),
        alpha=0.35,
        label="95% latent (dense)",
    )
    plt.plot(
        freq_dense_np,
        onp.asarray(v_lat_mean_dense),
        "-",
        lw=2,
        label="Latent mean (dense)",
    )
    plt.scatter(
        freq_obs_np,
        onp.asarray(Exp_Result_mean_np),
        s=35,
        marker="o",
        label="Measurement mean",
    )
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title("Latent band (dense) with measurement mean")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Multiple-measurement level predictive band (dense) + posterior mean (dense) + measurements
    plt.figure(figsize=(10,6))
    plt.fill_between(
        freq_dense_np,
        y_lo95_mm_dense,
        y_hi95_mm_dense,
        alpha=0.25,
        label="95% multiple-measurement level (dense)",
    )
    plt.plot(
        freq_dense_np,
        onp.asarray(v_fit_dense),
        "-",
        lw=2,
        label="Posterior mean (dense)",
    )
    for r_i in range(Rplot):
        plt.scatter(
            freq_obs_np,
            onp.asarray(onp.where(mask_obs_np[r_i, :], V[r_i, :], onp.nan)),
            s=12,
            alpha=0.5,
            label="_nolegend_",
        )
    plt.plot(
        freq_obs_np,
        onp.asarray(Exp_Result_mean_np),
        "o",
        ms=5,
        label="Measurement mean",
    )
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title("Multiple-measurement predictive band (dense) + posterior mean + measurements")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Measurement-mean predictive bands (dense) + measurement mean (points)
    plt.figure(figsize=(9,5))
    plt.fill_between(
        freq_dense_np,
        ybar_lo95_dense,
        ybar_hi95_dense,
        alpha=0.25,
        label="95% measurement-mean level (dense)",
    )
    plt.fill_between(
        freq_dense_np,
        ybar_lo50_dense,
        ybar_hi50_dense,
        alpha=0.25,
        label="50% measurement-mean level (dense)",
    )
    plt.scatter(
        freq_obs_np,
        onp.asarray(Exp_Result_mean_np),
        s=35,
        marker="o",
        label="Measurement mean",
    )
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title("Measurement-mean predictive bands (dense)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[Band plots skipped] {e}")

# ---------- MAP parameters (real scale) ----------
print("\n[MAP parameters (real scale)]")
def _to_numpy(x):
    try:
        return onp.asarray(x)
    except Exception:
        return onp.asarray(jnp.array(x))
for k, v in map_sample.items():
    if k == "obs":
        continue
    arr = _to_numpy(v)
    is_log = k.startswith("log_")
    if arr.size == 1:
        val = float(onp.exp(arr)[()] if is_log else arr[()])
        print(f"{k:20s} = {val:.4e}")
    else:
        if is_log:
            arr = onp.exp(arr)
        flat = arr.reshape(-1)
        head = ", ".join(f"{x:.3e}" for x in flat[:6])
        ellipsis = "..." if flat.size > 6 else ""
        print(f"{k:20s} = [{head}{ellipsis}]  (len={flat.size})")

# ---------- Coverage report ----------
def _binom_ci(p_hat, n, alpha=0.05):
    import numpy as _np
    if n <= 0 or not _np.isfinite(p_hat):
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    se = _np.sqrt(max(p_hat * (1 - p_hat) / n, 0.0))
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    return (float(lo), float(hi))

def _marginal_bool_rate(b):
    import numpy as _np
    b = _np.asarray(b).astype(bool)
    if b.size == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    p = float(b.mean())
    lo, hi = _binom_ci(p, b.size)
    return p, (lo, hi), b.size

def _per_freq_mm_inside(lo, hi):
    import numpy as _np
    m = mask_obs_np & _np.isfinite(V)
    inside = (V >= lo[None, :]) & (V <= hi[None, :]) & m
    n_r = m.sum(axis=0).astype(int)
    hits = _np.where(
        n_r > 0,
        inside.sum(axis=0) / _np.maximum(n_r, 1),
        _np.nan,
    )
    return hits, n_r

def _freq_bands_log10_tertiles(freq_vec):
    import numpy as _np
    logf = _np.log10(_np.asarray(freq_vec))
    q1, q2 = _np.nanpercentile(logf, [33.3333, 66.6667])
    fmin, fmax = float(logf.min()), float(logf.max())
    bands = {
        "low":  {"mask": logf <= q1,             "range": (fmin, q1)},
        "mid":  {"mask": (logf > q1) & (logf <= q2), "range": (q1, q2)},
        "high": {"mask": logf > q2,             "range": (q2, fmax)},
    }
    return bands, logf

def _print_band_header(bname, brange):
    a, b = brange
    print(f"{bname} band (log10 f in [{a:.3f}, {b:.3f}])")

def report_marginal_and_freqband_coverage():
    import numpy as _np
    mm50_all = ((V >= y_lo50_mm[None, :]) & (V <= y_hi50_mm[None, :]) & mask_obs_np)
    mm95_all = ((V >= y_lo95_mm[None, :]) & (V <= y_hi95_mm[None, :]) & mask_obs_np)
    p_mm50, (mm50_lo, mm50_hi), n_mm = _marginal_bool_rate(mm50_all[mask_obs_np])
    p_mm95, (mm95_lo, mm95_hi), n_mm95 = _marginal_bool_rate(mm95_all[mask_obs_np])

    mean50_hit = (Exp_Result_mean_np >= ybar_lo50) & (Exp_Result_mean_np <= ybar_hi50)
    mean95_hit = (Exp_Result_mean_np >= ybar_lo95) & (Exp_Result_mean_np <= ybar_hi95)
    p_mean50, (mean50_lo, mean50_hi), n_mean = _marginal_bool_rate(mean50_hit)
    p_mean95, (mean95_lo, mean95_hi), n_mean95 = _marginal_bool_rate(mean95_hit)

    lat50_hit = (Exp_Result_mean_np >= _np.asarray(v_lat_q25)) & (
        Exp_Result_mean_np <= _np.asarray(v_lat_q75)
    )
    lat95_hit = (Exp_Result_mean_np >= _np.asarray(v_lat_lo95)) & (
        Exp_Result_mean_np <= _np.asarray(v_lat_hi95)
    )
    p_lat50, (lat50_lo, lat50_hi), n_lat = _marginal_bool_rate(lat50_hit)
    p_lat95, (lat95_lo, lat95_hi), n_lat95 = _marginal_bool_rate(lat95_hit)

    print("\n[Marginal coverage] overall")
    print(
        f"Multiple-measurement level 50% : {p_mm50:.3f} [{mm50_lo:.3f}, {mm50_hi:.3f}]  (n={n_mm})"
    )
    print(
        f"Multiple-measurement level 95% : {p_mm95:.3f} [{mm95_lo:.3f}, {mm95_hi:.3f}]  (n={n_mm95})"
    )
    print(
        f"Measurement-mean level 50%     : {p_mean50:.3f} [{mean50_lo:.3f}, {mean50_hi:.3f}]  (K={n_mean})"
    )
    print(
        f"Measurement-mean level 95%     : {p_mean95:.3f} [{mean95_lo:.3f}, {mean95_hi:.3f}]  (K={n_mean95})"
    )
    print(
        f"Latent-level 50%               : {p_lat50:.3f} [{lat50_lo:.3f}, {lat50_hi:.3f}]  (K={n_lat})"
    )
    print(
        f"Latent-level 95%               : {p_lat95:.3f} [{lat95_lo:.3f}, {lat95_hi:.3f}]  (K={n_lat95})"
    )

    bands, logf = _freq_bands_log10_tertiles(Exp_Frequency)
    mm50_per_k, n_r_per_k = _per_freq_mm_inside(y_lo50_mm, y_hi50_mm)
    mm95_per_k, _         = _per_freq_mm_inside(y_lo95_mm, y_hi95_mm)

    print("\n[Conditional coverage] by frequency bands (log10 tertiles)")
    for name in ["low", "mid", "high"]:
        bmask = _np.asarray(bands[name]["mask"])
        lo_r, hi_r = bands[name]["range"]
        K_band = int(bmask.sum())

        mm50_vals = mm50_per_k[bmask]
        mm95_vals = mm95_per_k[bmask]
        mm50_vals = mm50_vals[_np.isfinite(mm50_vals)]
        mm95_vals = mm95_vals[_np.isfinite(mm95_vals)]

        def _avg_ci(vec):
            if vec.size == 0:
                return float("nan"), (float("nan"), float("nan")), 0
            p = float(_np.mean(vec))
            lo, hi = _binom_ci(p, vec.size)
            return p, (lo, hi), int(vec.size)

        mm50_p, (mm50_lo_b, mm50_hi_b), n_mm50_b = _avg_ci(mm50_vals)
        mm95_p, (mm95_lo_b, mm95_hi_b), n_mm95_b = _avg_ci(mm95_vals)

        mean50_hit = (Exp_Result_mean_np >= ybar_lo50) & (
            Exp_Result_mean_np <= ybar_hi50
        )
        mean95_hit = (Exp_Result_mean_np >= ybar_lo95) & (
            Exp_Result_mean_np <= ybar_hi95
        )
        mean50_p, (mean50_lo_b, mean50_hi_b), n_mean_b = _marginal_bool_rate(
            mean50_hit[bmask]
        )
        mean95_p, (mean95_lo_b, mean95_hi_b), n_mean95_b = _marginal_bool_rate(
            mean95_hit[bmask]
        )

        lat50_hit = (Exp_Result_mean_np >= _np.asarray(v_lat_q25)) & (
            Exp_Result_mean_np <= _np.asarray(v_lat_q75)
        )
        lat95_hit = (Exp_Result_mean_np >= _np.asarray(v_lat_lo95)) & (
            Exp_Result_mean_np <= _np.asarray(v_lat_hi95)
        )
        lat50_p, (lat50_lo_b, lat50_hi_b), n_lat_b = _marginal_bool_rate(
            lat50_hit[bmask]
        )
        lat95_p, (lat95_lo_b, lat95_hi_b), n_lat95_b = _marginal_bool_rate(
            lat95_hit[bmask]
        )

        _print_band_header(name, (lo_r, hi_r))
        print(f"  K in band: {K_band}")
        print(
            f"  Multiple-measurement 50% : {mm50_p:.3f} [{mm50_lo_b:.3f}, {mm50_hi_b:.3f}]"
        )
        print(
            f"  Multiple-measurement 95% : {mm95_p:.3f} [{mm95_lo_b:.3f}, {mm95_hi_b:.3f}]"
        )
        print(
            f"  Measurement-mean 50%     : {mean50_p:.3f} [{mean50_lo_b:.3f}, {mean50_hi_b:.3f}]"
        )
        print(
            f"  Measurement-mean 95%     : {mean95_p:.3f} [{mean95_lo_b:.3f}, {mean95_hi_b:.3f}]"
        )
        print(
            f"  Latent-level 50%         : {lat50_p:.3f} [{lat50_lo_b:.3f}, {lat50_hi_b:.3f}]"
        )
        print(
            f"  Latent-level 95%         : {lat95_p:.3f} [{lat95_lo_b:.3f}, {lat95_hi_b:.3f}]"
        )
        print("")

report_marginal_and_freqband_coverage()

# ---------- Save artifacts ----------
try:
    idata.to_netcdf(IDATA_NC_PATH)
    az.summary(idata).to_csv(SUMMARY_CSV_PATH)
    print(f"\nSaved: {IDATA_NC_PATH}, {SUMMARY_CSV_PATH}")
except Exception as e:
    print(f"[Save artifacts failed] {e}")

# ---------- Timing ----------
print("\n=== Timing (seconds) ===")
print(f"Data loading:          {t1_data - t0_data:8.3f}")
print(f"HMC (NumPyro MCMC):    {t1_hmc - t0_hmc:8.3f}")
print("========================")
