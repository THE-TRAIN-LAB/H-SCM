"""Allocation figure for Sec. IV: oracle regret of each policy, by environment.

Draws the numbers of the allocation paragraph -- step15 -- as one column-width
panel: for each of the four evaluation environments, the regret of the realized
cell goodput against the oracle policy, one marker per policy, on a log axis
because the policies differ by two orders of magnitude (0.3% .. 49%).

Form: dot + whisker rather than bars. A bar's length is meaningless on a log
axis (it depends on where the axis is cut), a dot's position is not. The whisker
is the 95% confidence interval of the mean over the 50 seeds (each seed is the
mean over its 25 cells) -- the uncertainty of the quantity the text quotes; the
MLP pools its five initialisations (250 values), since the text quotes the range
over initialisations.

Colour follows the ROLE it has in Fig. 2 so the two figures read the same way:
red = correlational estimate/policy, blue = the causal method, green = oracle.
Orange and grey are Okabe-Ito. Marker SHAPE carries identity independently of
hue, so the figure survives greyscale printing. (The dataviz palette validator
is a node script and node is not installed here, so it was not run.)

Reads results/step15_alloc_by_env.csv. Writes results/figures/fig4_scheduling.png:
(a) loss against the oracle allocation, (b) realized total cell goodput.

Usage: python experiments/plot_alloc_by_env.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import print_sizes                                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#DDDDDD"
ORACLE = "#27AE60"

# (column key, printed label, colour, marker) -- fixed order, worst to best
POLICIES = [
    ("equal",         "equal split",          "#9A9A9A", "s"),
    ("proportional",  "proportional to goodput",       "#C0392B", "^"),
    ("noncausal_mlp", "MLP",                            "#E69F00", "D"),
    ("hscm_struct",   "H-SCM (proposed)",              "#0072B2", "o"),
]
MLP_INITS = ("", "_rs43", "_rs44", "_rs45", "_rs46")
ENVS = [("E0_nominal", "nominal"), ("E1_high_load", "high load"),
        ("E2_poor_channel", "poor channel"), ("E3_heterogeneous", "heterog.")]
SHORT = ["nominal", "high\nload", "poor\nchannel", "diverse\nUEs"]   # two-panel x labels, spelled out

FIG_1 = (5.0, 2.55)          # single panel, printed at \columnwidth
FIG_2 = (6.0, 3.05)          # two panels side by side (Fig. 3 width)
YLIM = (-0.03, 100.0)        # regret %; symlog. Slightly below 0 so the oracle's
                             # zero line is not hidden under the x-axis spine
LINTHRESH = 0.1              # so the oracle's TRUE zero has a position on the axis


def regret_samples(df, env, key):
    """Per-seed regret (%) of one policy in one environment."""
    e = df[df.env == env]
    if key == "noncausal_mlp":
        return np.concatenate([100 * e[f"regret_noncausal_mlp{t}"].values
                               for t in MLP_INITS])
    return 100 * e[f"regret_{key}"].values


def style(ax, S, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=S["title"], color=INK, loc="left", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=S["label"], labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=S["label"], labelpad=2)
    ax.grid(True, axis="y", which="major", color=GRID, lw=.5, alpha=.8)
    ax.grid(True, axis="y", which="minor", color=GRID, lw=.4, alpha=.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=S["tick"], pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_regret(ax, df, S, label_proposed=True):
    n = len(POLICIES)
    off = np.linspace(-0.3, 0.3, n)
    for k, (key, lab, col, mk) in enumerate(POLICIES):
        xs, med, lo, hi = [], [], [], []
        for i, (env, _) in enumerate(ENVS):
            r = regret_samples(df, env, key)
            xs.append(i + off[k])
            m, ci = r.mean(), 1.96 * r.std(ddof=1) / np.sqrt(len(r))
            med.append(m)
            lo.append(max(m - ci, 0.0))
            hi.append(m + ci)
        med, lo, hi = map(np.asarray, (med, lo, hi))
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="none",
                    ecolor=col, elinewidth=1.0, capsize=1.6, capthick=1.0,
                    zorder=3)
        ax.scatter(xs, med, s=26, marker=mk, color=col, edgecolor="white",
                   linewidth=0.6, zorder=4, label=lab)
        if label_proposed and key == "hscm_struct":
            for x, m in zip(xs, med):
                ax.annotate(f"{m:.2f}", xy=(x, m), xytext=(4.0, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=S["note"], color=MUTED)
    # oracle: regret 0 is off a log axis, so it is the floor, stated once
    # the oracle has zero loss. A pure log axis has no position for zero, so
    # the axis is symlog: linear on [0, LINTHRESH], logarithmic above. The
    # oracle line is then drawn AT zero, not at a stand-in value.
    ax.axhline(0.0, color=ORACLE, lw=1.4, ls="--", zorder=2)
    ax.text(-0.52, 0.012, "ground truth", color=ORACLE,
            fontsize=S["note"], ha="left", va="bottom")
    # light separators between environment groups
    for i in range(1, len(ENVS)):
        ax.axvline(i - 0.5, color=GRID, lw=.6, zorder=1)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.6)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
    # minor ticks at 2..9 within each decade of the log region, so the scale
    # is visibly logarithmic on the axis itself (none in the linear strip)
    ax.yaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(
        base=10, linthresh=LINTHRESH, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2.0, width=0.6)
    ax.set_xlim(-0.55, len(ENVS) - 0.45)
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([lab for _, lab in ENVS], fontsize=S["tick"])
    style(ax, S, ylabel="loss vs. ground truth (%)")


def draw_goodput(ax, df, S):
    """Realized cell goodput (Mbps): oracle beside each policy; whiskers are
    the 95% CI of the mean over seeds, as in the regret panel."""
    series = [("oracle", "ground truth", ORACLE, "o")] + POLICIES[::-1]
    n = len(series)
    w = 0.8 / n
    for k, (key, lab, col, mk) in enumerate(series):
        vals, err = [], []
        for env, _ in ENVS:
            e = df[df.env == env]
            j = e[f"J_{key}"].values
            vals.append(j.mean())
            err.append(1.96 * j.std(ddof=1) / np.sqrt(len(j)))
        x = np.arange(len(ENVS)) - 0.4 + w * (k + 0.5)
        ax.bar(x, vals, width=w * 0.92, color=col, zorder=3, label=lab,
               yerr=err, error_kw=dict(ecolor=INK, elinewidth=.6, capsize=1.2))
    ax.set_xlim(-0.55, len(ENVS) - 0.45)
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([lab for _, lab in ENVS], fontsize=S["tick"])
    ax.set_ylim(0, 42)
    style(ax, S, ylabel="total cell goodput (Mbps)")



def build_single(df, S):
    fig, ax = plt.subplots(figsize=FIG_1)
    draw_regret(ax, df, S)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=S["legend"], ncol=2, loc="upper center",
               bbox_to_anchor=(0.55, 1.0), frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.80)
    return fig


def build_two_panel(df, S):
    fig, ax = plt.subplots(1, 2, figsize=FIG_2, gridspec_kw=dict(wspace=0.55))
    draw_regret(ax[0], df, S)
    draw_goodput(ax[1], df, S)
    for a, letter in zip(ax, ("(a)", "(b)")):
        a.set_xticklabels(SHORT, fontsize=S["tick"])
        a.set_title(letter, fontsize=S["title"], loc="left", pad=3)
    # legend BELOW both panels, in a band reserved by explicit margins:
    # tight_layout ignores figure-level legends, so it cannot be trusted here
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=S["legend"], ncol=3, loc="lower center",
               bbox_to_anchor=(0.53, 0.0), frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.33, top=0.93,
                        wspace=0.55)
    return fig


def save_sized(build, df, canvas_w, paths, passes=2):
    """Size the type for the width that is actually SAVED, not the canvas.

    bbox_inches="tight" crops (or, with a legend overhanging the axes,
    extends) the canvas, and the paper scales whatever is saved to
    \columnwidth. Figs. 2 and 3 crop to within 2% of their canvas, so their
    type prints at print_sizes.ON_PAGE; a figure whose tight box is 15%
    wider would print 15% smaller. Measure, re-size, and render again.
    """
    w = canvas_w
    for _ in range(passes):
        S = print_sizes.apply(plt, w)
        fig = build(df, S)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w_new = bb.width + 2 * 0.02          # + pad_inches on both sides
        if abs(w_new - w) < 0.02:
            break
        plt.close(fig)
        w = w_new
    for pth in paths:
        fig.savefig(pth, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {paths[0]}  (type sized for {w:.2f} in; printed at 3.5 in)")


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "step15_alloc_by_env.csv"))
    n_seeds = df.groupby("env").seed.nunique().min()
    n_cells = int(df.n_cells.iloc[0])

    save_sized(build_two_panel, df, FIG_2[0],
               [os.path.join(ROOT, "results", "figures", "fig4_scheduling.png")])

    # the numbers the figure encodes, for the caption
    print(f"\n{n_seeds} seeds x {n_cells} cells per environment; whiskers = "
          f"95% CI of the mean over seeds (MLP pools 5 initialisations)")
    print(f"{'policy':<16}" + "".join(f"{lab:>14}" for _, lab in ENVS))
    for key, lab, _, _ in POLICIES:
        print(f"{key:<16}" + "".join(
            f"{regret_samples(df, env, key).mean():>13.2f}%" for env, _ in ENVS))
    print(f"{'oracle J (Mbps)':<16}" + "".join(
        f"{df[df.env == env].J_oracle.mean():>14.1f}" for env, _ in ENVS))
    print(f"{'equal J (Mbps)':<16}" + "".join(
        f"{df[df.env == env].J_equal.mean():>14.1f}" for env, _ in ENVS))


if __name__ == "__main__":
    main()
