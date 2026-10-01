"""E7 + E8 — Phase 6: Rung 1 (confounding) and Rung 2 (do-calculus).

Estimators (observational, 500 rows, OBSERVABLE variables only — UP-25):
  naive OLS        mu_obs(t)  = E[Y | T = t], OLS on num_prb alone
  adjusted OLS     mu_adj(t)  = E_L[E[Y | T=t, L]] under a linear model
                                (the paper's "adjusted OLS", eq. 13)
  adjusted GBM     same functional, nonparametric — fails under Scenario A
                                (positivity; see the diagnostic below)

Reference oracles:
  n=30 regimes            -> paper reproduction (Table III)
  high-N paired oracle    -> method validation (data/oracle_paired.csv.gz,
                             SE ~0.4 Mbps vs ~6 Mbps at n=30)

All estimators get 95% bootstrap CIs, and the naive-vs-adjusted absolute
error reduction is bootstrapped as a PAIRED statistic (same resample for
both estimators), so the improvement itself carries uncertainty.

Usage: python experiments/step6_rung12.py
Outputs: results/table3_top.csv, results/rung2_dose_response.csv,
         results/figures/step6_rung12.png
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 42
N_BOOT = 2000
GBM = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
T_GRID = [10, 50, 100]


def naive_ols(obs, t):
    b, a = np.polyfit(obs.num_prb, obs.goodput, 1)
    return a + b * t


def adjusted_ols(obs, t):
    """Backdoor adjustment E_L[E[Y|T=t,L]] under a linear outcome model."""
    X = np.column_stack([np.ones(len(obs)), obs.num_prb, obs.cell_load])
    beta, *_ = np.linalg.lstsq(X, obs.goodput.values, rcond=None)
    return beta[0] + beta[1] * t + beta[2] * obs.cell_load.mean()


def adjusted_gbm(obs, t, model=None):
    if model is None:
        model = GradientBoostingRegressor(**GBM).fit(
            obs[["num_prb", "cell_load"]].values, obs.goodput.values)
    X = np.column_stack([np.full(len(obs), t), obs.cell_load.values])
    return float(model.predict(X).mean()), model


def paired_bootstrap(obs, t, truth, n_boot=N_BOOT, seed=SEED):
    """Resample units once per replicate; recompute BOTH estimators on the
    same resample so the error-reduction statistic is properly paired."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    naive, adj, red = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = obs.iloc[idx]
        mn, ma = naive_ols(b, t), adjusted_ols(b, t)
        naive.append(mn)
        adj.append(ma)
        red.append(100 * (abs(mn - truth) - abs(ma - truth))
                   / max(abs(mn - truth), 1e-9))
    q = lambda v: np.percentile(v, [2.5, 97.5])  # noqa: E731
    return {"naive_ci": q(naive), "adj_ci": q(adj), "reduction_ci": q(red),
            "reduction_median": float(np.median(red))}


def main():
    suffix = ""
    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    intv = pd.read_csv(os.path.join(ROOT, "data", "interventional.csv"))

    print(f"r(cell_load, num_prb) = "
          f"{np.corrcoef(obs.cell_load, obs.num_prb)[0, 1]:+.3f}\n")

    # oracles
    small = {t: intv.loc[intv.regime == f"do(num_prb={t})", "goodput"]
             for t in T_GRID}
    op = os.path.join(ROOT, "data", f"oracle_paired{suffix}.csv.gz")
    have_hiN = os.path.exists(op)
    if have_hiN:
        oracle_df = pd.read_csv(op)
        big = {t: oracle_df.loc[oracle_df.arm == f"do(num_prb={t})", "goodput"]
               for t in T_GRID}
    else:
        print(f"NOTE: {op} missing — run build_oracle_validation.py; "
              "falling back to the n=30 oracle.\n")
        big = small

    gbm_model = None
    rows = []
    for t in T_GRID:
        mu_n, mu_a = naive_ols(obs, t), adjusted_ols(obs, t)
        mu_g, gbm_model = adjusted_gbm(obs, t, gbm_model)
        truth = big[t].mean()
        rows.append({"t": t, "naive_ols": mu_n, "adjusted_ols": mu_a,
                     "adjusted_gbm": mu_g,
                     "oracle_n30": small[t].mean(), "oracle_n30_std": small[t].std(),
                     "oracle_hiN": truth,
                     "oracle_hiN_se": big[t].std() / np.sqrt(len(big[t])),
                     "abs_err_naive": abs(mu_n - truth),
                     "abs_err_adj": abs(mu_a - truth),
                     "bias_naive_pct": 100 * (mu_n - truth) / truth,
                     "bias_adj_pct": 100 * (mu_a - truth) / truth})
    dose = pd.DataFrame(rows)
    dose.to_csv(os.path.join(ROOT, "results",
                             f"rung2_dose_response{suffix}.csv"), index=False)

    t = 100
    r = dose[dose.t == t].iloc[0]
    bs = paired_bootstrap(obs, t, r.oracle_hiN)
    _, gm = adjusted_gbm(obs, t, gbm_model)

    print(f"Rung 1 & 2 at t = {t}  (paper: naive 34.8, adjusted 24.2, "
          f"oracle 24.7 +- 30.0)")
    print(f"  naive OLS      : {r.naive_ols:6.2f} Mbps  95% CI "
          f"[{bs['naive_ci'][0]:6.2f}, {bs['naive_ci'][1]:6.2f}]")
    print(f"  adjusted OLS   : {r.adjusted_ols:6.2f} Mbps  95% CI "
          f"[{bs['adj_ci'][0]:6.2f}, {bs['adj_ci'][1]:6.2f}]")
    print(f"  adjusted GBM   : {r.adjusted_gbm:6.2f} Mbps  "
          f"(positivity-sensitive)")
    print(f"  oracle n=30    : {r.oracle_n30:6.2f} +- {r.oracle_n30_std:.2f} "
          f"(SE {r.oracle_n30_std / np.sqrt(30):.2f})   <- paper reproduction")
    if have_hiN:
        print(f"  oracle high-N  : {r.oracle_hiN:6.2f} (SE "
              f"{r.oracle_hiN_se:.3f}, N={len(big[t])})   <- method validation")

    print(f"\n  |error| naive    : {r.abs_err_naive:5.2f} Mbps "
          f"({r.bias_naive_pct:+.1f}%)")
    print(f"  |error| adjusted : {r.abs_err_adj:5.2f} Mbps "
          f"({r.bias_adj_pct:+.1f}%)")
    print(f"  error reduction  : {bs['reduction_median']:.1f}%  95% CI "
          f"[{bs['reduction_ci'][0]:.1f}%, {bs['reduction_ci'][1]:.1f}%] "
          f"(paired bootstrap, {N_BOOT} reps)")

    # positivity diagnostic
    print("\npositivity / overlap diagnostic:")
    for tt in T_GRID:
        near = obs[(obs.num_prb - tt).abs() <= 5]
        rng_ = (f"L in [{near.cell_load.min():.2f}, {near.cell_load.max():.2f}]"
                if len(near) else "NO SUPPORT")
        print(f"  |T-{tt:3d}| <= 5: n={len(near):3d}  {rng_}"
              f"   (adjustment needs the full L range)")

    print("\ndose-response (naive / adjOLS / adjGBM / oracle high-N):")
    for rr in dose.itertuples(index=False):
        print(f"  t={rr.t:3d}: {rr.naive_ols:6.1f} / {rr.adjusted_ols:6.1f} "
              f"/ {rr.adjusted_gbm:6.1f} / {rr.oracle_hiN:6.2f}")

    t3 = pd.DataFrame([
        {"estimand": "Rung 1 naive obs", "n": 500,
         "mean_mbps": round(r.naive_ols, 1),
         "ci95": f"[{bs['naive_ci'][0]:.1f}, {bs['naive_ci'][1]:.1f}]",
         "abs_err_vs_oracle": round(r.abs_err_naive, 2),
         "note": f"{r.bias_naive_pct:+.0f}% confounding bias"},
        {"estimand": "Rung 1 adjusted obs (OLS)", "n": 500,
         "mean_mbps": round(r.adjusted_ols, 1),
         "ci95": f"[{bs['adj_ci'][0]:.1f}, {bs['adj_ci'][1]:.1f}]",
         "abs_err_vs_oracle": round(r.abs_err_adj, 2),
         "note": (f"error -{bs['reduction_median']:.0f}% "
                  f"[{bs['reduction_ci'][0]:.0f}, {bs['reduction_ci'][1]:.0f}]%")},
        {"estimand": "Rung 1 adjusted obs (GBM)", "n": 500,
         "mean_mbps": round(r.adjusted_gbm, 1), "ci95": "",
         "abs_err_vs_oracle": round(abs(r.adjusted_gbm - r.oracle_hiN), 2),
         "note": "nonparametric; positivity-sensitive"},
        {"estimand": "Rung 2 interventional (n=30)", "n": 30,
         "mean_mbps": round(r.oracle_n30, 1),
         "ci95": f"+-{r.oracle_n30_std:.1f} sd",
         "abs_err_vs_oracle": "", "note": "paper reproduction"},
        {"estimand": "Rung 2 oracle (high-N paired)", "n": len(big[t]),
         "mean_mbps": round(r.oracle_hiN, 2),
         "ci95": f"SE {r.oracle_hiN_se:.3f}",
         "abs_err_vs_oracle": 0.0, "note": "method validation ground truth"},
    ])
    t3.to_csv(os.path.join(ROOT, "results", f"table3_top{suffix}.csv"),
              index=False)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    names = ["naive\nOLS", "adjusted\nOLS", "adjusted\nGBM", "oracle\n(high-N)"]
    vals = [r.naive_ols, r.adjusted_ols, r.adjusted_gbm, r.oracle_hiN]
    err = [[r.naive_ols - bs["naive_ci"][0], r.adjusted_ols - bs["adj_ci"][0], 0, 0],
           [bs["naive_ci"][1] - r.naive_ols, bs["adj_ci"][1] - r.adjusted_ols, 0, 0]]
    ax[0].bar(names, vals, color=["#C0392B", "#2E86C1", "#BDC3C7", "#27AE60"],
              yerr=err, capsize=5)
    ax[0].axhline(r.oracle_hiN, color="#27AE60", ls="--", lw=0.9)
    ax[0].set_ylabel(f"E[goodput | T={t}] (Mbps)")
    ax[0].set_title(f"(a) t={t}: naive {r.bias_naive_pct:+.0f}%, "
                    f"adjusted {r.bias_adj_pct:+.0f}%")
    tg = np.linspace(10, 100, 40)
    b1, a1 = np.polyfit(obs.num_prb, obs.goodput, 1)
    ax[1].scatter(obs.num_prb, obs.goodput, s=5, alpha=0.15, color="grey",
                  label="observational")
    ax[1].plot(tg, a1 + b1 * tg, color="#C0392B", label="naive OLS")
    ax[1].plot(tg, [adjusted_ols(obs, x) for x in tg], color="#2E86C1",
               label="adjusted OLS")
    ax[1].plot(tg, [adjusted_gbm(obs, x, gm)[0] for x in tg], color="#BDC3C7",
               label="adjusted GBM")
    ax[1].errorbar(T_GRID, dose.oracle_hiN, yerr=1.96 * dose.oracle_hiN_se,
                   fmt="o", color="#27AE60", capsize=4, label="oracle high-N")
    ax[1].set(xlabel="num_prb (T)", ylabel="goodput (Mbps)",
              title="(b) dose-response")
    ax[1].legend(fontsize=8)
    xb = np.arange(len(T_GRID))
    ax[2].bar(xb - 0.2, dose.naive_ols - dose.oracle_hiN, 0.2, color="#C0392B",
              label="naive OLS")
    ax[2].bar(xb, dose.adjusted_ols - dose.oracle_hiN, 0.2, color="#2E86C1",
              label="adjusted OLS")
    ax[2].bar(xb + 0.2, dose.adjusted_gbm - dose.oracle_hiN, 0.2,
              color="#BDC3C7", label="adjusted GBM")
    ax[2].axhline(0, color="black", lw=0.8)
    ax[2].set_xticks(xb)
    ax[2].set_xticklabels([f"t={x}" for x in T_GRID])
    ax[2].set(ylabel="bias vs high-N oracle (Mbps)",
              title="(c) bias before/after adjustment")
    ax[2].legend(fontsize=8)
    fig.suptitle("Rungs 1 & 2: naive vs adjusted vs ground truth")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures",
                       f"step6_rung12{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
