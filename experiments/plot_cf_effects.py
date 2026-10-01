"""Rung-3 figure, rebuilt around the PER-UNIT effect. Recovered graph only.

The previous version drew, per query, a KDE of the factual, interventional and
counterfactual goodput. Three problems:

  * These are PAIRED within-unit counterfactuals; a marginal density discards
    the pairing, which is the entire content of Rung 3.
  * The "interventional" curve is the population do() distribution over a
    DIFFERENT set of units than the query's subpopulation, so placing it beside
    the counterfactual curve compares two things that are not comparable.
  * CF-2 shifts the mean by 1.6 Mbps on a distribution that is 64% zeros, so
    its three curves were visually identical and said nothing.

The paper's Rung-3 claim is that a population effect does not tell you which UE
benefits. The top row states exactly that: units are sorted by their own
predicted effect and drawn as a curve, with the population mean -- everything
Rung 2 can give you -- as a flat line through it. The gap between curve and
line IS the claim. The bottom row then shows that the abducted latent predicts
where on that curve a UE falls.

Everything is from the RECOVERED graph (results/step13_*), matching Sec. V.

Usage: python experiments/plot_cf_effects.py
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
# Okabe-Ito; hue is decorative here, position and sign carry the meaning.
POS, NEG, MEAN, INK, GRID = "#0072B2", "#D55E00", "#E69F00", "#1A1A1A", "#DDDDDD"
BAR = "#009E73"
# A 2x3 grid at \\columnwidth prints each panel ~1 in wide, so what fits is
# set by the ON-PAGE type size, not by this canvas: titles and labels below
# are kept short for that reason, and the caption carries the explanation.
FIG_SIZE = (6.0, 4.4)
S = print_sizes.sizes(FIG_SIZE[0])

QUERIES = [
    ("CF-1", r"$do(T{=}100)$", "moderate-goodput UEs", r"$\hat{\xi}_{sf}$ (dB)"),
    ("CF-2", r"$do(\xi_{hw}{=}0)$", "peer-underperformers", r"$\hat{\xi}_{hw}$"),
    ("CF-3", r"$do(T{=}100)$", "underperforming UEs", r"$\hat{\xi}_{sf}$ (dB)"),
]
# CF-1 and CF-2 stratify a continuous latent into five bins. Their ticks sit
# on the bin BOUNDARIES, between bars, as a histogram's would: exact, with no
# open-tail notation, and narrow enough for five bars in a ~1 in panel (named
# per-bar labels such as "<-1.5" overlapped their neighbours by up to 4 pt).
# Plain text with a true minus (U+2212), not mathtext: mathtext spaces "-" as
# a binary operator, which alone made "-1.5" and "-.5" collide.
EDGES = {"CF-1": ["\u22127", "\u22123", "3", "7"],
         "CF-2": ["\u22121.5", "\u2212.5", ".5", "1.5"]}
# Boundary label sets too wide to sit on one row. CF-2's four labels measured
# 0.23 pt apart on the page even after trimming the bar margins, and widening
# its column pushed the CF-1/CF-3 legends onto their curves -- so alternate
# labels drop to a second row instead.
STAGGER = {"CF-2"}
# CF-3's three strata are named regimes, so they stay labelled per bar.
SHORT = {"xi_sf<-5 (channel-limited)": "<\u22125\nchan.",
         "-5..5": "\u22125\u20265", ">+5 (PRB-limited)": ">+5\nPRB"}


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=S["title"], color=INK, loc="left", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=S["label"], labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=S["label"], labelpad=2)
    ax.grid(True, color=GRID, lw=.5, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=S["tick"], pad=2)


def _line_points(ax):
    """Every plotted line in display pixels, densified. Uses each line's own
    transform, so axhline (x in axes fraction) lands where it is drawn."""
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_transform().transform(
            np.column_stack(ln.get_data()).astype(float))
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy) > 1:
            xy = np.vstack([np.linspace(xy[k], xy[k + 1], 60)
                            for k in range(len(xy) - 1)])
        pts.append(xy)
    return np.vstack(pts) if pts else np.zeros((0, 2))


def place_notes(ax, notes, candidates):
    """Place each (text, colour) at the first candidate (x, y, ha) -- axes
    fraction, top-aligned -- where it covers no line, no legend and no note
    placed before it. A sorted-effect curve splits its panel differently for
    every query, so a fixed spot that clears one curve sits on another."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = _line_points(ax)
    taken = [ax.get_legend().get_window_extent(r)] if ax.get_legend() else []
    for text, colour in notes:
        t = ax.text(0, 0, text, transform=ax.transAxes, va="top",
                    fontsize=S["note"], color=colour)
        for x, y, ha in candidates:
            t.set_position((x, y)); t.set_ha(ha)
            e = t.get_window_extent(r).padded(3)
            on_line = ((pts[:, 0] > e.x0) & (pts[:, 0] < e.x1)
                       & (pts[:, 1] > e.y0) & (pts[:, 1] < e.y1)).any()
            if not on_line and not any(e.overlaps(b) for b in taken):
                break
        taken.append(t.get_window_extent(r).padded(3))


def main():
    z = np.load(os.path.join(ROOT, "results", "step13_cf_arrays.npz"))
    st = pd.read_csv(os.path.join(ROOT, "results", "step13_strata.csv"))
    st = st[st.graph == "recovered"]

    print_sizes.apply(plt, FIG_SIZE[0])
    fig, ax = plt.subplots(2, 3, figsize=FIG_SIZE)

    panel_notes = []
    for k, (q, act, sub, xlab) in enumerate(QUERIES):
        d = np.sort(z[f"rec_{q}_cf"] - z[f"rec_{q}_fact"])
        n, mu = len(d), d.mean()
        x = np.arange(1, n + 1)

        # ---- top: the per-unit effect curve against the population mean ----
        a = ax[0, k]
        a.fill_between(x, 0, np.maximum(d, 0), color=POS, alpha=.30, lw=0)
        if (d < 0).any():
            a.fill_between(x, np.minimum(d, 0), 0, color=NEG, alpha=.45, lw=0)
        a.plot(x, d, color=POS, lw=1.5)
        a.axhline(0, color=INK, lw=.8)
        a.axhline(mu, color=MEAN, ls="--", lw=1.8, label=f"mean {mu:+.2f}")
        frac0 = 100 * (np.abs(d) < 0.05).mean()
        # K in the title: "number of UEs (K=175)" is wider than a ~1 in panel
        style(a, f"({chr(97+k)}) {q} ($n$={n})",
              xlabel="number of UEs",
              ylabel="gain (Mbps)" if k == 0 else None)
        a.legend(fontsize=S["legend"], loc="upper left", frameon=False,
                 borderpad=0, handlelength=1.3, handletextpad=0.4,
                 bbox_to_anchor=(0.0, 0.97))
        # At readable type a panel prints ~1 in wide: "bottom 10% +0.7" alone
        # spans two-thirds of it, so legend + three notes + a rising curve do
        # not fit. Kept: the population mean (legend) and the share of UEs the
        # action leaves unchanged -- CF-2's 64% and CF-3's 54% ARE the
        # heterogeneity. The decile extremes are read off the curve's ends and
        # are quoted in the caption.
        if frac0 >= 1:
            panel_notes.append((a, [(f"{frac0:.0f}% $|\\Delta|{{<}}0.05$",
                                     "#5A5A5A")]))

        # ---- bottom: the abducted latent predicts where a UE lands ---------
        b = ax[1, k]
        g = st[st["query"] == q]
        lab = [SHORT.get(s, s) for s in g.stratum]
        v = g.mean_effect.values
        b.bar(range(len(v)), v, color=[BAR if t >= 0 else NEG for t in v],
              width=.68)
        b.axhline(0, color=INK, lw=.8)
        b.axhline(mu, color=MEAN, ls="--", lw=1.4)
        for i, (val, nn) in enumerate(zip(v, g.n.values)):
            # offset in points, not data units, so the gap is the same on every
            # axis scale; a light backing keeps the dashed mean line off digits
            b.annotate(f"{nn}", xy=(i, val), xytext=(0, 1.2 if val >= 0 else -1.2),
                       textcoords="offset points", ha="center",
                       va="bottom" if val >= 0 else "top",
                       fontsize=S["note"], color="#5A5A5A",
                       bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
        lo, hi = min(0.0, v.min()), max(0.0, v.max())
        b.set_ylim(lo - 0.18 * (hi - lo), hi + 0.28 * (hi - lo))
        # bars fill the axis: the default outer margin cost each of five
        # slots room its boundary labels need
        b.set_xlim(-0.52, len(v) - 0.48)
        if q in EDGES:
            pos = [i + 0.5 for i in range(len(v) - 1)]
            if q in STAGGER:
                # Alternate labels go on MINOR ticks with a larger pad. Setting
                # a pad on individual major ticks does not survive drawing --
                # matplotlib regenerates those Tick objects -- whereas
                # tick_params(which="minor") persists.
                b.set_xticks(pos[0::2]); b.set_xticklabels(EDGES[q][0::2], fontsize=S["tick"])
                b.set_xticks(pos[1::2], minor=True)
                b.set_xticklabels(EDGES[q][1::2], minor=True, fontsize=S["tick"])
                b.tick_params(axis="x", which="minor", labelsize=S["tick"],
                              pad=2 + 1.55 * S["tick"],
                              length=plt.rcParams["xtick.major.size"],
                              width=plt.rcParams["xtick.major.width"])
            else:
                b.set_xticks(pos)
                b.set_xticklabels(EDGES[q], fontsize=S["tick"])
        else:
            b.set_xticks(range(len(v)))
            b.set_xticklabels(lab, fontsize=S["tick"])
        style(b, f"({chr(100+k)}) {q}",
              xlabel=xlab, ylabel="gain (Mbps)" if k == 0 else None)

    # no suptitle: the manuscript caption states the claim, and at column
    # width a figure-level title prints at ~3 pt while costing a row of height
    fig.tight_layout(pad=0.3, w_pad=0.35, h_pad=0.7)
    # notes are placed AFTER layout, against the final axes geometry
    cands = [(.03, .80, "left"), (.03, .67, "left"), (.03, .54, "left"),
             (.97, .40, "right"), (.97, .29, "right"), (.97, .18, "right"),
             (.50, .80, "left"), (.50, .67, "left")]
    for a, notes in panel_notes:
        place_notes(a, notes, cands)
    p = os.path.join(ROOT, "results", "figures", "fig3_counterfactual_gains.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", p)
    for q, _, _, _ in QUERIES:
        d = z[f"rec_{q}_cf"] - z[f"rec_{q}_fact"]
        dec = max(1, len(d) // 10)
        s = np.sort(d)
        print(f"  {q}: mean {d.mean():+.2f}  median {np.median(d):+.2f}  "
              f"bottom decile {s[:dec].mean():+.2f}  top decile {s[-dec:].mean():+.2f}"
              f"  ratio {s[-dec:].mean()/max(abs(s[:dec].mean()),1e-9):.0f}x")
    return fig


if __name__ == "__main__":
    main()
