# -*- coding: utf-8 -*-
"""
GA baseline for DEP single-shell (multi-core) + metrics (R2, RMSE) + plots
- Parallel modes: runs | population | off
- Metrics computed consistently with FIT_TO_MEAN choice
- Figures saved:
    fit_best.png            (observed freq, replicates + mean + GA best)
    fit_best_dense.png      (dense freq, replicates + mean + GA best, for comparison with Bayesian plot)
    hist_metrics.png
    parity_best.png
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ----------------------------
# Reproducibility and config
# ----------------------------
SEED = int(os.getenv("SEED", "0"))
rng = np.random.default_rng(SEED)

DATA_XLSX = os.getenv("DATA_XLSX", "/home/cjli/Python/MCMCpy/DEPvelocityRAW_Control.xlsx")

LAMBDA_W = float(os.getenv("LAMBDA_W", "0.2"))
USE_ALT_A = bool(int(os.getenv("USE_ALT_A", "1")))
USE_DERIV = False

# Fit target
FIT_TO_MEAN = True
WEIGHT_BY_COUNT = True  # only used when FIT_TO_MEAN=True (weighted SSE/RMSE/R2 by number of replicates)

# GA hyperparameters
POP_SIZE         = int(os.getenv("GA_POP_SIZE", "200"))
N_GENERATIONS    = int(os.getenv("GA_N_GEN", "1000"))
ELITE_FRAC       = float(os.getenv("GA_ELITE_FRAC", "0.05"))
TOURNAMENT_K     = int(os.getenv("GA_TOUR_K", "3"))
CROSSOVER_RATE   = float(os.getenv("GA_CX_RATE", "0.9"))
MUTATION_RATE    = float(os.getenv("GA_MUT_RATE", "0.3"))
LOG_MUTATION_SD  = float(os.getenv("GA_LOG_MUT_SD", "0.15"))
N_RUNS           = int(os.getenv("GA_N_RUNS", "100"))
EARLY_STOP_ROUNDS= int(os.getenv("GA_EARLY_STOP", "100"))
VERBOSE_EVERY    = int(os.getenv("GA_VERBOSE_EVERY", "50"))

# Parallel settings
PARALLEL_MODE = os.getenv("PARALLEL_MODE", "runs").lower()  # "runs" | "population" | "off"
N_JOBS_ENV = int(os.getenv("N_JOBS", "-1"))
CPU_COUNT = os.cpu_count() or 1
N_WORKERS = (CPU_COUNT if N_JOBS_ENV in (-1, 0) else max(1, N_JOBS_ENV))

# Output paths
GA_RESULTS_CSV      = os.getenv("GA_RESULTS_CSV", "ga_baseline_runs.csv")
FIG_FIT_BEST        = os.getenv("FIG_FIT_BEST",   "fit_best.png")
FIG_FIT_BEST_DENSE  = os.getenv("FIG_FIT_BEST_DENSE", "fit_best_dense.png")
FIG_HIST_MET        = os.getenv("FIG_HIST_MET",   "hist_metrics.png")
FIG_PARITY          = os.getenv("FIG_PARITY",     "parity_best.png")

# ----------------------------
# Physical constants (match Bayesian)
# ----------------------------
eps0   = 8.854e-12
E      = 25000.0
D      = 1e-3
r      = 3e-6 + 55e-9
tmem   = 5.0e-9
tw     = 50e-9
t_shell = tmem + tw
ep_med = 80 * eps0
si_med = 0.1

# ----------------------------
# Parameter bounds (log-space = prior support)
# ----------------------------
log_eps_cyto_low,   log_eps_cyto_high   = np.log(10 * eps0), np.log(200 * eps0)
log_sigma_cyto_low, log_sigma_cyto_high = np.log(1e-6),      np.log(1.0)
log_eps_mem_low,    log_eps_mem_high    = np.log(10 * eps0), np.log(200 * eps0)
log_sigma_mem_low,  log_sigma_mem_high  = np.log(1e-6),      np.log(1.0)

LB = np.array([log_eps_cyto_low, log_sigma_cyto_low, log_eps_mem_low, log_sigma_mem_low])
UB = np.array([log_eps_cyto_high, log_sigma_cyto_high, log_eps_mem_high, log_sigma_mem_high])

# ----------------------------
# Data loading and cleaning (same as Bayesian)
# ----------------------------
t0_data = time.perf_counter()
df = pd.read_excel(DATA_XLSX, header=None)

freq_raw = np.asarray(df.iloc[0].values, dtype=float)
V_raw    = np.asarray(df.iloc[1:].values, dtype=float)

col_ok = np.isfinite(freq_raw) & np.any(np.isfinite(V_raw), axis=0)
freq = freq_raw[col_ok]
V    = V_raw[:, col_ok]

ord_idx = np.argsort(freq)
freq = freq[ord_idx]
V    = V[:, ord_idx]

mask_obs = np.isfinite(V)             # shape: (R, K)
V_filled = np.where(mask_obs, V, 0.0)

with np.errstate(invalid="ignore"):
    V_mean = np.nanmean(np.where(mask_obs, V, np.nan), axis=0)

USE_DERIV = USE_DERIV and (freq.size >= 2)
if USE_DERIV:
    mask_dv = mask_obs[:, 1:] & mask_obs[:, :-1]
else:
    mask_dv = None
t1_data = time.perf_counter()

# ----------------------------
# Model (NumPy; same equations as Bayesian)
# ----------------------------
def eps_complex(f, eps_r, sigma):
    return eps_r - 1j * sigma / (2.0 * np.pi * f + 1e-300)

def fun_cymem(f, eps_cyto, sigma_cyto, eps_mem, sigma_mem):
    eps_c = eps_complex(f, eps_cyto, sigma_cyto)
    eps_m = eps_complex(f, eps_mem,  sigma_mem)
    a = (r / (r - t_shell)) if USE_ALT_A else ((r - t_shell) / r)
    num = (a**3 + 2.0 * (eps_c - eps_m) / (eps_c + 2.0 * eps_m + 1e-300))
    den = (a**3 -       (eps_c - eps_m) / (eps_c + 2.0 * eps_m + 1e-300))
    den = den + 1e-30j
    return eps_m * (num / den)

def fun_fcm(f, eps_cyto, sigma_cyto, eps_mem, sigma_mem):
    ep_med_c = eps_complex(f, ep_med, si_med)
    ep_p     = fun_cymem(f, eps_cyto, sigma_cyto, eps_mem, sigma_mem)
    return (ep_p - ep_med_c) / (ep_p + 2.0 * ep_med_c + 1e-300)

def theory_velocity(f, eps_cyto, sigma_cyto, eps_mem, sigma_mem):
    dep_const = ((r**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return dep_const * np.real(fun_fcm(f, eps_cyto, sigma_cyto, eps_mem, sigma_mem))

# ----------------------------
# Metrics helpers
# ----------------------------
def metrics_mean(v_pred):
    """
    R2 and RMSE vs replicate mean; mask NaNs; optional weighting by replicate count.
    """
    y = V_mean.copy()
    valid = np.isfinite(y)
    y = y[valid]
    yp = v_pred[valid]
    if y.size == 0:
        return np.nan, np.nan, np.nan  # SSE, RMSE, R2

    if WEIGHT_BY_COUNT:
        w_full = mask_obs.sum(axis=0).astype(float)
        w = w_full[valid]
        # SSE and RMSE (weighted)
        sse = np.sum(w * (y - yp) ** 2)
        mse = sse / np.sum(w)
        rmse = np.sqrt(mse)
        # SST (weighted to mean of y with the same weights)
        y_bar = np.sum(w * y) / np.sum(w)
        sst = np.sum(w * (y - y_bar) ** 2)
    else:
        resid = (y - yp)
        sse = np.sum(resid ** 2)
        rmse = np.sqrt(np.mean(resid ** 2))
        y_bar = np.mean(y)
        sst = np.sum((y - y_bar) ** 2)

    r2 = np.nan if sst <= 0 else (1.0 - sse / sst)
    return sse, rmse, r2

def metrics_all(v_pred):
    """
    R2 and RMSE vs all replicates (masked).
    """
    diff = (v_pred[None, :] - V_filled)
    mask = mask_obs
    resid = diff[mask]
    if resid.size == 0:
        return np.nan, np.nan, np.nan
    sse = np.sum(resid ** 2)
    rmse = np.sqrt(np.mean(resid ** 2))
    y = V[mask]
    y_bar = np.mean(y)
    sst = np.sum((y - y_bar) ** 2)
    r2 = np.nan if sst <= 0 else (1.0 - sse / sst)
    return sse, rmse, r2

def evaluate_metrics(log_params):
    p = np.exp(log_params)
    v_pred = theory_velocity(freq, p[0], p[1], p[2], p[3])
    if FIT_TO_MEAN:
        sse, rmse, r2 = metrics_mean(v_pred)
    else:
        sse, rmse, r2 = metrics_all(v_pred)
    return v_pred, sse, rmse, r2

# ----------------------------
# Objective function (SSE plus optional derivative term)
# ----------------------------
def objective_logparam(log_params):
    p = np.exp(log_params)
    v_pred = theory_velocity(freq, p[0], p[1], p[2], p[3])

    # primary term
    if FIT_TO_MEAN:
        y = V_mean.copy()
        valid = np.isfinite(y)
        resid = v_pred[valid] - y[valid]
        if WEIGHT_BY_COUNT:
            w = mask_obs.sum(axis=0).astype(float)[valid]
            sse_mean = np.sum(w * (resid ** 2))
        else:
            sse_mean = np.sum(resid ** 2)
    else:
        diff = (v_pred[None, :] - V_filled)
        sse_mean = np.sum((diff ** 2)[mask_obs])

    # derivative term
    sse_deriv = 0.0
    if USE_DERIV and freq.size >= 2:
        logf  = np.log10(freq)
        dlogf = np.diff(logf)
        dlogf[dlogf == 0.0] = 1e-12
        dv_pred = np.diff(v_pred) / dlogf

        if FIT_TO_MEAN:
            V_masked = np.where(mask_obs, V, np.nan)
            with np.errstate(invalid="ignore"):
                V_mean_each = np.nanmean(V_masked, axis=0)
            dv_obs = np.diff(V_mean_each) / dlogf
            valid_d = np.isfinite(dv_obs)

            if WEIGHT_BY_COUNT and (mask_dv is not None):
                w_d_full = mask_dv.sum(axis=0).astype(float)  # shape: (K-1,)
                w_d = w_d_full[valid_d]
                resid_d = LAMBDA_W * (dv_pred[valid_d] - dv_obs[valid_d])
                sse_deriv = np.sum(w_d * resid_d ** 2)
            else:
                resid_d = LAMBDA_W * (dv_pred[valid_d] - dv_obs[valid_d])
                sse_deriv = np.sum(resid_d ** 2)
        else:
            V_masked = np.where(mask_obs, V, 0.0)
            dv_obs_each = (np.diff(V_masked, axis=1) / dlogf[None, :])
            resid_d_mat = LAMBDA_W * (dv_pred[None, :] - dv_obs_each)
            sse_deriv = np.sum((resid_d_mat ** 2)[mask_dv])

    return float(sse_mean + sse_deriv)

# ----------------------------
# GA operators (log-space)
# ----------------------------
def tournament_select(pop, fitness, k=3, local_rng=None):
    rr = local_rng if local_rng is not None else rng
    idx = rr.integers(0, pop.shape[0], size=k)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()

def crossover(parent1, parent2, local_rng=None):
    rr = local_rng if local_rng is not None else rng
    if rr.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    mask = rr.random(size=parent1.shape) < 0.5
    mid = (parent1 + parent2) / 2.0
    child1 = np.where(mask, parent1, mid)
    child2 = np.where(~mask, parent2, mid)
    return child1, child2

def mutate(ind, local_rng=None):
    rr = local_rng if local_rng is not None else rng
    if rr.random() < MUTATION_RATE:
        noise = rr.normal(0.0, LOG_MUTATION_SD, size=ind.shape)
        ind = ind + noise
    return np.minimum(np.maximum(ind, LB), UB)

def init_population(pop_size, local_rng=None):
    rr = local_rng if local_rng is not None else rng
    return rr.uniform(LB, UB, size=(pop_size, 4))

# ----------------------------
# Fitness evaluation (serial or parallel per population)
# ----------------------------
def evaluate_population(pop, executor=None):
    if executor is None:
        return np.array([objective_logparam(ind) for ind in pop], dtype=float)
    else:
        futures = [executor.submit(objective_logparam, pop[i]) for i in range(pop.shape[0])]
        out = np.empty(pop.shape[0], dtype=float)
        for j, f in enumerate(futures):
            out[j] = f.result()
        return out

# ----------------------------
# Single GA run
# ----------------------------
def run_ga_once(seed=None, parallel_population=False):
    local_rng = np.random.default_rng(seed if seed is not None else rng.integers(1 << 31))
    pop = init_population(POP_SIZE, local_rng=local_rng)

    pop_executor = None
    if parallel_population and PARALLEL_MODE == "population" and N_WORKERS > 1:
        pop_executor = ThreadPoolExecutor(max_workers=N_WORKERS)

    try:
        fitness = evaluate_population(pop, executor=pop_executor)
        best_fit = float(np.min(fitness))
        best_ind = pop[np.argmin(fitness)].copy()
        stall = 0

        for gen in range(1, N_GENERATIONS + 1):
            n_elite = max(1, int(ELITE_FRAC * POP_SIZE))
            elite_idx = np.argsort(fitness)[:n_elite]
            elites = pop[elite_idx].copy()

            off = []
            while len(off) < POP_SIZE - n_elite:
                p1 = tournament_select(pop, fitness, TOURNAMENT_K, local_rng)
                p2 = tournament_select(pop, fitness, TOURNAMENT_K, local_rng)
                c1, c2 = crossover(p1, p2, local_rng)
                c1 = mutate(c1, local_rng)
                c2 = mutate(c2, local_rng)
                off.append(c1)
                if len(off) < POP_SIZE - n_elite:
                    off.append(c2)
            offspring = np.vstack(off)

            pop = np.vstack([elites, offspring])
            fitness = evaluate_population(pop, executor=pop_executor)

            current_best = float(np.min(fitness))
            if current_best + 1e-12 < best_fit:
                best_fit = current_best
                best_ind = pop[np.argmin(fitness)].copy()
                stall = 0
            else:
                stall += 1

            if (gen % VERBOSE_EVERY == 0) or gen == 1:
                print(f"[GA] gen {gen:4d} | best SSE={best_fit:.6e}")

            if stall >= EARLY_STOP_ROUNDS:
                print(f"[GA] Early stop at gen {gen} (stall {stall} rounds).")
                break

        # metrics for best
        v_pred, sse_m, rmse_m, r2_m = evaluate_metrics(best_ind)
        return best_ind, best_fit, v_pred, sse_m, rmse_m, r2_m
    finally:
        if pop_executor is not None:
            pop_executor.shutdown(wait=True)

# ----------------------------
# Multi-run (parallel over runs)
# ----------------------------
def _run_wrapper(run_idx, base_seed):
    s = base_seed + run_idx * 9973
    t0 = time.perf_counter()
    best_log, best_sse, v_pred, sse_m, rmse_m, r2_m = run_ga_once(
        seed=s, parallel_population=(PARALLEL_MODE == "population")
    )
    t1 = time.perf_counter()
    best_real = np.exp(best_log)
    return {
        "run": run_idx,
        "seed": s,
        "best_sse_obj": best_sse,   # objective with derivative term if enabled
        "sse_metric": sse_m,        # metric SSE (without derivative term)
        "rmse": rmse_m,
        "r2": r2_m,
        "eps_cyto":   best_real[0],
        "sigma_cyto": best_real[1],
        "eps_mem":    best_real[2],
        "sigma_mem":  best_real[3],
        "time_sec": t1 - t0,
    }, v_pred

def multi_run(n_runs=N_RUNS, base_seed=SEED, verbose=False):
    preds = {}  # store prediction curve of each run if needed
    if PARALLEL_MODE == "runs" and N_WORKERS > 1:
        print(f"[Parallel] Running {n_runs} GA runs with {N_WORKERS} workers ...")
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_run_wrapper, i, base_seed): i for i in range(n_runs)}
            for fut in as_completed(futs):
                res, v_pred = fut.result()
                results.append(res)
                preds[res["run"]] = v_pred
                if verbose:
                    print(
                        f"[GA] run {res['run']:03d} | R2={res['r2']:.4f} | "
                        f"RMSE={res['rmse']:.4e} | time={res['time_sec']:.2f}s"
                    )
        results.sort(key=lambda d: d["run"])
        return pd.DataFrame(results), preds
    else:
        results = []
        for i in range(n_runs):
            res, v_pred = _run_wrapper(i, base_seed)
            results.append(res)
            preds[res["run"]] = v_pred
            if verbose:
                print(
                    f"[GA] run {res['run']:03d} | R2={res['r2']:.4f} | "
                    f"RMSE={res['rmse']:.4e} | time={res['time_sec']:.2f}s"
                )
        return pd.DataFrame(results), preds

# ----------------------------
# Plot helpers
# ----------------------------
def plot_best_fit(freq, V, mask_obs, V_mean, v_pred_best, r2, rmse, path):
    plt.figure(figsize=(9, 6))
    # replicates scatter
    R = V.shape[0]
    for r_i in range(R):
        y = np.where(mask_obs[r_i, :], V[r_i, :], np.nan)
        plt.scatter(freq, y, s=12, alpha=0.5, label="_nolegend_")
    # replicate mean
    plt.plot(freq, V_mean, "o", label="Mean of replicates")
    # best fit
    plt.plot(freq, v_pred_best, "-", linewidth=2, label="GA best fit")
    plt.xscale("log")
    plt.xlim(1e4, 1e8)  # enforce x-axis from 1e4 to 1e8
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title(f"GA Best Fit  (R2={r2:.4f}, RMSE={rmse:.3e})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_best_fit_dense(freq, V, mask_obs, V_mean, best_params_real, r2, rmse, path):
    """
    Dense-frequency version of GA best fit, for comparison with Bayesian plot.
    - freq_dense: logspace from 1e4 to 1e8
    - plots: replicates (at observed freq), replicate mean, GA best curve (dense)
    """
    # dense frequency grid (aligned with Bayesian style: 10^4 to 10^8)
    log_lo = 4.0
    log_hi = 8.0
    freq_dense = np.logspace(log_lo, log_hi, 1000)

    eps_cyto, sigma_cyto, eps_mem, sigma_mem = best_params_real
    v_dense = theory_velocity(freq_dense, eps_cyto, sigma_cyto, eps_mem, sigma_mem)

    plt.figure(figsize=(10, 6))
    # GA best curve (dense)
    plt.semilogx(freq_dense, v_dense, "-", linewidth=2, label="GA best fit (dense)")
    # replicates scatter (observed frequencies)
    R = V.shape[0]
    for r_i in range(R):
        y = np.where(mask_obs[r_i, :], V[r_i, :], np.nan)
        plt.scatter(freq, y, s=12, alpha=0.5, label="_nolegend_")
    # replicate mean (observed frequency grid)
    plt.scatter(freq, V_mean, s=40, marker="o", label="Replicate mean")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DEP Velocity")
    plt.title(f"GA Best Fit (dense)  (R2={r2:.4f}, RMSE={rmse:.3e})")
    plt.grid(True, which="both", linestyle=":")
    plt.xlim(1e4, 1e8)  # enforce x-axis from 1e4 to 1e8
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_hist_metrics(df, path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df["r2"].dropna(), bins=20, edgecolor="k")
    plt.xlabel("R2")
    plt.ylabel("Count")
    plt.title("Distribution of R2 (across GA runs)")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(df["rmse"].dropna(), bins=20, edgecolor="k")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.title("Distribution of RMSE (across GA runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_parity_mean(freq, V_mean, v_pred_best, path):
    valid = np.isfinite(V_mean)
    y = V_mean[valid]
    yp = v_pred_best[valid]
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y, yp, s=20, alpha=0.7)
    lo = np.nanmin([y.min(), yp.min()])
    hi = np.nanmax([y.max(), yp.max()])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)):
        lo, hi = -1.0, 1.0
    pad = 0.05 * (hi - lo) if np.isfinite(hi - lo) else 1.0
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", linewidth=1)
    plt.xlabel("Observed mean")
    plt.ylabel("Predicted")
    plt.title("Parity plot (replicate mean)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("=== GA baseline for DEP single-shell (multi-core + metrics) ===")
    print(f"Data loading: {t1_data - t0_data:.3f}s, K={freq.size}, R={V.shape[0]}, USE_DERIV={USE_DERIV}")
    print(f"Parallel mode: {PARALLEL_MODE} | Workers: {N_WORKERS} | FIT_TO_MEAN={FIT_TO_MEAN}")

    t0 = time.perf_counter()
    df_res, preds = multi_run(n_runs=N_RUNS, base_seed=SEED, verbose=True)
    t1 = time.perf_counter()

    # summary table
    print("\n[GA reproducibility summary over runs]")
    print(df_res[["best_sse_obj", "sse_metric", "rmse", "r2", "time_sec"]].describe())

    # best run (by metric R2 then by sse_metric)
    idx_best = int(((-df_res["r2"]).rank(method="min") + (df_res["sse_metric"]).rank(method="min")).idxmin())
    row = df_res.loc[idx_best]

    # print best params
    print("\n[GA best run parameters (real scale)]")
    for k in ["eps_cyto", "sigma_cyto", "eps_mem", "sigma_mem"]:
        print(f"{k:12s} = {row[k]:.6e}")
    print(
        f"Best (by R2/SSE) -> R2={row['r2']:.4f}, RMSE={row['rmse']:.4e}, "
        f"SSE_metric={row['sse_metric']:.6e}, SSE_obj={row['best_sse_obj']:.6e} "
        f"(run {int(row['run'])}, seed {int(row['seed'])})"
    )

    # save CSV
    df_res.to_csv(GA_RESULTS_CSV, index=False)
    print(f"\nSaved GA run table: {GA_RESULTS_CSV}")

    # plots
    v_pred_best = preds[int(row["run"])]
    best_params_real = np.array(
        [row["eps_cyto"], row["sigma_cyto"], row["eps_mem"], row["sigma_mem"]]
    )

    plot_best_fit(freq, V, mask_obs, V_mean, v_pred_best, row["r2"], row["rmse"], FIG_FIT_BEST)
    plot_best_fit_dense(freq, V, mask_obs, V_mean, best_params_real, row["r2"], row["rmse"], FIG_FIT_BEST_DENSE)
    plot_hist_metrics(df_res, FIG_HIST_MET)
    plot_parity_mean(freq, V_mean, v_pred_best, FIG_PARITY)
    print(f"Saved figures: {FIG_FIT_BEST}, {FIG_FIT_BEST_DENSE}, {FIG_HIST_MET}, {FIG_PARITY}")

    print(f"\nTotal wall time: {t1 - t0:.2f}s")
