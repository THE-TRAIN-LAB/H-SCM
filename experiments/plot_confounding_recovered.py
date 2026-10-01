"""confounding_analysis.png, but with the adjustment set DERIVED, not assumed.

The companion figure (step8_paper_figures.fig_confounding) demonstrates
confounding while taking the true DAG as given: its backdoor schematic is
hardcoded and its adjusted estimator hardcodes cell_load as the adjustment
set. That is a fair illustration of the problem, but it presupposes the answer.

This version reads the graph step11 actually RECOVERED
(results/step11_recovered_dag.yaml, the majority graph over 20 seeds), derives
the adjustment set from it as the observable parents of the treatment, and
estimates with THAT. Panels (b) and (c) draw both estimators so the reader can
see whether they agree rather than being told they do; the two curves coincide
only because the recovery succeeded, and would visibly separate if it had not.

Usage: python experiments/plot_confounding_recovered.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.fitting import LATENT_NODES                       # noqa: E402
import print_sizes                                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TREATMENT, OUTCOME = "num_prb", "goodput"
# The two adjusted curves COINCIDE -- that is the result -- so the recovered
# series is drawn thin-dashed on top of the thick true-DAG line and must
# contrast with it. It was #009E73, a green that both hid against the blue
# underneath and repeated the oracle's green. Okabe-Ito reddish purple is
# distinct from the red, blue and green already in this figure, and stays
# distinguishable under the common forms of colour blindness.
TRUE, REC = "#0072B2", "#CC79A7"
NAIVE, ORACLE, INK = "#C0392B", "#27AE60", "#8E44AD"
# Drawn at 5 in and printed at \\columnwidth; every text size comes from
# print_sizes so it is chosen for the PAGE, not for this canvas.
FIG_SIZE = (5.0, 4.2)
S = print_sizes.sizes(FIG_SIZE[0])
# panel (a): room below zero for the legend (goodput is never negative)
A_YMIN, A_HANDLE = -55, 1.6


def backdoor_set(edges):
    """Observable parents of the treatment in whichever graph is passed."""
    return sorted({a for a, b in edges if b == TREATMENT
                   and a not in LATENT_NODES})


def adjusted_ols(obs, t, Z):
    """E_L[E[Y | T=t, Z]] under a linear outcome model. Empty Z -> naive."""
    if not Z:
        b, a = np.polyfit(obs[TREATMENT], obs[OUTCOME], 1)
        return a + b * t
    X = np.column_stack([np.ones(len(obs)), obs[TREATMENT].values,
                         obs[Z].values.astype(float)])
    beta, *_ = np.linalg.lstsq(X, obs[OUTCOME].values, rcond=None)
    return float(beta[0] + beta[1] * t
                 + obs[Z].values.astype(float).mean(0) @ beta[2:])


def kde(ax, data, label, color, ls="-", lo=None, hi=None):
    data = np.asarray(data); data = data[np.isfinite(data)]
    if len(data) < 3 or data.std() < 1e-9:
        return
    lo = np.percentile(data, 0.5) if lo is None else lo
    hi = np.percentile(data, 99.5) if hi is None else hi
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, stats.gaussian_kde(data)(xs), color=color, ls=ls, lw=1.6,
            label=label)


def main():
    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    oracle = pd.read_csv(os.path.join(ROOT, "data", "oracle_paired.csv.gz"))
    dose = pd.read_csv(os.path.join(ROOT, "results",
                                    "rung2_dose_response.csv"))
    truth = [tuple(e) for e in yaml.safe_load(
        open(os.path.join(ROOT, "config", "dag_edges.yaml")))["edges"]]
    recdoc = yaml.safe_load(open(os.path.join(ROOT, "results",
                                              "step11_recovered_dag.yaml")))
    rec = [tuple(e) for e in recdoc["recovered_edges"]]
    Zt, Zr = backdoor_set(truth), backdoor_set(rec)
    seeds = recdoc["metrics"]["seeds"]
    print(f"adjustment set  true DAG: {Zt}   recovered: {Zr}"
          + ("   (identical)" if Zt == Zr else "   (DIFFERENT)"))

    print_sizes.apply(plt, FIG_SIZE[0])
    fig, ax = plt.subplots(2, 2, figsize=FIG_SIZE)


    # (b) dose-response: BOTH adjustment sets -------------------------------
    b = ax[0, 0]
    tg = np.linspace(10, 100, 40)
    m, c = np.polyfit(obs[TREATMENT], obs[OUTCOME], 1)
    b.scatter(obs[TREATMENT], obs[OUTCOME], s=3, alpha=0.12, color="grey")
    b.plot(tg, c + m * tg, color=NAIVE, lw=1.6, label="naive")
    b.plot(tg, [adjusted_ols(obs, x, Zt) for x in tg], color=TRUE, lw=3.2,
           label="adj (true)")
    b.plot(tg, [adjusted_ols(obs, x, Zr) for x in tg], color=REC, lw=1.9,
           ls=(0, (4, 3)), label="adj (rec.)")
    b.errorbar(dose.t, dose.oracle_hiN, yerr=1.96 * dose.oracle_hiN_se,
               fmt="o", ms=5, color=ORACLE, capsize=3, label="oracle")
    b.set(xlabel="PRBs ($T$)", ylabel="goodput (Mbps)", ylim=(A_YMIN, 60),
          title="(a)")
    # lower right is structurally empty: goodput >= 0, and the naive line is
    # above zero for T > ~47, so nothing is ever drawn there
    b.legend(fontsize=S["legend"], loc="lower right", handlelength=A_HANDLE,
             borderpad=0.3, labelspacing=0.25, framealpha=0.9)

    # (c) three-rung bars at t = 100 ---------------------------------------
    r = dose[dose.t == 100].iloc[0]
    at, ar = adjusted_ols(obs, 100, Zt), adjusted_ols(obs, 100, Zr)
    c3 = ax[0, 1]
    vals = [r.naive_ols, at, ar, r.oracle_hiN]
    c3.bar(["naive", "adj\n(true)", "adj\n(rec.)", "oracle"], vals,
           color=[NAIVE, TRUE, REC, ORACLE])
    c3.axhline(r.oracle_hiN, color=ORACLE, ls="--", lw=0.9)
    for i, v in enumerate(vals):
        # light backing: the oracle's dashed reference line runs through the
        # value it equals, and would otherwise cut the digits
        c3.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=S["note"],
                bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    c3.set(ylabel="$E[Y\\,|\\,T{=}100]$ (Mbps)", ylim=(0, 52),
           title="(b)")

    # raw data, independent of either graph -----------------------------

    e = ax[1, 0]
    kde(e, obs[OUTCOME], "observational", NAIVE)
    kde(e, oracle.loc[oracle.arm == "do(num_prb=100)", OUTCOME],
        "do(prb=100)", ORACLE)
    kde(e, oracle.loc[oracle.arm == "do(num_prb=50)", OUTCOME],
        "do(prb=50)", TRUE, ls="--")
    e.set(xlabel="goodput (Mbps)", ylabel="probability density", xlim=(-2, 80),
          title="(c)")
    e.legend(fontsize=S["legend"], handlelength=1.6, borderpad=0.3,
             labelspacing=0.25)

    f = ax[1, 1]
    bins = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
    names = ["<.25", ".25-.5", ".5-.75", ">.75"]
    ofact = oracle[oracle.arm == "factual"].reset_index(drop=True)
    o100 = oracle[oracle.arm == "do(num_prb=100)"].reset_index(drop=True)
    obs_m, do_m = [], []
    for lo, hi in bins:
        s = obs[(obs.cell_load >= lo) & (obs.cell_load < hi)]
        obs_m.append(s[OUTCOME].mean() if len(s) else np.nan)
        sel = (ofact.cell_load >= lo) & (ofact.cell_load < hi)
        do_m.append(o100[OUTCOME][sel].mean() if sel.any() else np.nan)
    x = np.arange(4)
    f.bar(x - 0.2, obs_m, 0.4, color=NAIVE, label="observed")
    f.bar(x + 0.2, do_m, 0.4, color=ORACLE, label="do(prb=100)")
    f.set_xticks(x); f.set_xticklabels(names, fontsize=S["tick"])
    f.set(xlabel="cell-load range", ylabel="goodput (Mbps)",
          title="(d)")
    f.legend(fontsize=S["legend"], handlelength=1.2, borderpad=0.3,
             labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    p = os.path.join(ROOT, "results", "figures", "fig2_confounding.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", p)
    print(f"  naive {r.naive_ols:.2f} | adjusted true {at:.2f} | "
          f"adjusted recovered {ar:.2f} | oracle {r.oracle_hiN:.2f}")
    print(f"  |adj_true - adj_rec| = {abs(at-ar):.2e} Mbps "
          f"(majority graph over {seeds} seeds)")
    return fig


if __name__ == "__main__":
    main()
