"""Render the H-SCM ground-truth DAG (paper Fig. 1) from config/dag_edges.yaml.

The edge set is read from the config — the figure cannot drift out of sync
with the ground truth (the script fails if positions and YAML disagree).
Visual conventions follow the paper's Fig. 1 caption: protocol layers as
coloured bands; treatment T (num_prb) and outcome Y (goodput) with bold
borders; the two confounding edges L->T and L->sinr_eff in red; effect-
modifier edges dashed purple; exogenous nodes grey.

Outputs results/figures/fig1_dag.png (+ .pdf for LaTeX).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ---------------------------------------------------------------- layout
POS = {
    # exogenous, top row (feed L1)
    "snr": (2.76, 10.57), "shadow_fading": (4.21, 10.57), "doppler": (5.66, 10.57),
    "delay_spread": (7.11, 10.57), "eps_sinr": (8.56, 10.57),
    # L1 PHY band
    "sinr_eff": (5.66, 8.55), "mcs_idx": (8.3, 9.0),
    "mod_order": (10.5, 9.05), "code_rate": (10.5, 7.6),
    "bler": (8.3, 7.25), "se_phy": (12.6, 8.3),
    # confounder + L2 exogenous, left margin
    "cell_load": (2.55, 7.31), "eps_sched": (2.25, 5.25),
    # L2 MAC band
    "num_prb": (4.35, 5.6),
    "harq_retx": (6.75, 5.89), "mac_tput": (9.23, 5.07),
    "hw_impairment": (12.0, 6.3), "traffic_burst": (12.95, 4.95),
    # L3 network band. Ordered so that each node sits BELOW-LEFT of its own
    # parents: mac_tput/queue_dep -> pkt_loss -> goodput reads top-right to
    # bottom-left without the long leftward sweep goodput used to need.
    "eps_queue": (12.95, 3.09), "queue_dep": (11.13, 2.82),
    "pkt_loss": (8.85, 2.2), "rtt_ms": (11.89, 0.84), "goodput": (8.09, 0.28),
}
# Every node is drawn with the symbol the manuscript's system model defines
# for it, so the figure and the equations name each variable the same way.
LABEL = {
    "cell_load": r"$L$",
    "snr": r"$\gamma_0$",
    "doppler": r"$f_{\mathrm{D}}$",
    "delay_spread": r"$\tau_{\mathrm{rms}}$",
    "shadow_fading": r"$\xi_{\mathrm{sf}}$",
    "eps_sinr": r"$\varepsilon_{\gamma}$",
    "sinr_eff": r"$\gamma_{\mathrm{eff}}$",
    "mcs_idx": r"$j^{\star}$",
    "mod_order": r"$b_{\mathrm{mod}}$",   # NOT Q_m: Q is the queue backlog
    "code_rate": r"$R$",
    "bler": r"$p$",
    "se_phy": r"$\eta_{\mathrm{PHY}}$",
    "num_prb": r"$T$",
    "eps_sched": r"$\varepsilon_T$",
    "harq_retx": r"$N_{\mathrm{retx}}$",
    "hw_impairment": r"$\xi_{\mathrm{hw}}$",
    "mac_tput": r"$\mu$",
    "traffic_burst": r"$\xi_{\mathrm{tb}}$",
    "eps_queue": r"$\varepsilon_3$",
    "queue_dep": r"$Q$",
    "rtt_ms": r"$D_{\mathrm{RTT}}$",
    "pkt_loss": r"$P_{\mathrm{loss}}$",
    "goodput": r"$Y$",
}
# Rendered width in glyph units. Hand-set because mathtext width cannot be
# read off the string: "$\eta_{\mathrm{PHY}}$" is 21 characters but renders
# about 3 glyphs wide, and a subscript counts roughly 0.6 of a base glyph.
GLYPHS = {
    "cell_load": 1.0, "snr": 2.0, "doppler": 2.0, "delay_spread": 2.9,
    "shadow_fading": 2.4, "eps_sinr": 2.4, "sinr_eff": 2.9, "mcs_idx": 1.8,
    "mod_order": 2.8, "code_rate": 1.0, "bler": 1.0, "se_phy": 3.3,
    "num_prb": 1.0, "eps_sched": 2.2, "harq_retx": 3.3, "hw_impairment": 2.6,
    "mac_tput": 1.0, "traffic_burst": 2.4, "eps_queue": 2.0, "queue_dep": 1.0,
    "rtt_ms": 3.3, "pkt_loss": 3.3, "goodput": 1.0,
}
# The paper places this figure at \columnwidth (~3.5 in). FIG_SIZE is drawn at
# 9 in wide, so the page shows it at ~0.39x; the sizes below are picked so
# that node labels land near 9 pt and legend text near 7 pt ON THE PAGE.
FIG_SIZE = (9.0, 7.2)
NODE_FONTSIZE = 23.0
LEGEND_FONTSIZE = 14.5
BAND_FONTSIZE = 17.0
XLIM, YLIM = (1.45, 14.70), (-0.45, 11.25)
EDGE_LW = {"red": 3.6, "purple": 2.8, "grey": 2.1}
ARROW = "-|>,head_length=11,head_width=5.5"
HEAD_LEN, HEAD_WID = 13, 6.5
HEAD_ONLY = f"-|>,head_length={HEAD_LEN},head_width={HEAD_WID}"
BOX_PAD = 0.06   # FancyBboxPatch pad, added on every side


def box_size(name):
    """Node box geometry in data coordinates, sized to the rendered label."""
    return 0.50 + 0.235 * GLYPHS.get(name, 3.0), 1.00
EXOGENOUS_STYLE = {"snr", "shadow_fading", "doppler", "delay_spread",
                   "eps_sinr", "eps_sched", "eps_queue", "hw_impairment",
                   "traffic_burst"}
RED_EDGES = {("cell_load", "num_prb"), ("cell_load", "sinr_eff")}
PURPLE_EDGES = {("shadow_fading", "sinr_eff"), ("hw_impairment", "mac_tput"),
                ("traffic_burst", "queue_dep")}
# per-edge arc curvature (positive = counter-clockwise bow)
# per-edge arc curvature (positive = counter-clockwise bow). The L1/L2 -> L3
# edges are the crowded ones: bler crosses the MAC band to reach pkt_loss and
# harq_retx to reach rtt_ms, so they are bowed apart deliberately rather than
# left to overlap.
RAD = {
    ("cell_load", "sinr_eff"): -0.25, ("cell_load", "num_prb"): 0.15,
    ("num_prb", "mac_tput"): -0.18,
    ("bler", "pkt_loss"): -0.44,
    ("harq_retx", "rtt_ms"): 0.34, ("mac_tput", "rtt_ms"): 0.16,
    ("mac_tput", "goodput"): -0.40, ("pkt_loss", "goodput"): 0.08,
    ("queue_dep", "pkt_loss"): 0.14, ("queue_dep", "rtt_ms"): -0.14,
    ("traffic_burst", "queue_dep"): 0.18,
    ("se_phy", "mac_tput"): 0.12, ("hw_impairment", "mac_tput"): -0.10,
    ("snr", "sinr_eff"): -0.08,
    ("mac_tput", "pkt_loss"): 0.28, ("mac_tput", "queue_dep"): -0.10,
}
# Curvature overrides for THIS layout, found by searching each edge for the
# arc that comes within no box it does not connect. At column width a line
# passing behind a node reads as an edge into it, so this is a correctness
# property of the figure, not a cosmetic one. Re-derive after moving nodes.
RAD.update({("snr", "sinr_eff"): 0.10, ("eps_sinr", "sinr_eff"): -0.10,
            ("num_prb", "mac_tput"): 0.10, ("se_phy", "mac_tput"): 0.05,
            ("harq_retx", "rtt_ms"): 0.00, ("mac_tput", "goodput"): 0.30,
            ("bler", "pkt_loss"): 0.00})
BANDS = [  # (ymin, ymax, colour, label)
    (6.91, 9.77, "#E9F1FB", "L1 PHY"),
    (3.65, 6.91, "#EAF7EB", "L2 MAC"),
    (-0.39, 3.65, "#FFF3E2", "L3 Network"),
]


def add_edge(ax, a, b, patchA, patchB, color, lw, ls="-", z=2.0, rad=0.06):
    """Draw one edge and return the patch that carries its full path.

    A dashed linestyle is applied to the WHOLE patch outline, arrowhead
    included, so a dashed FancyArrowPatch renders its head as a notched,
    ragged blob. A dashed edge is therefore drawn in two parts: a dashed
    shaft with no head, and a separate head that is filled and has no
    outline -- lw=0 also hides that second patch's own shaft, which would
    otherwise paint a solid line straight over the dashes.
    """
    common = dict(patchA=patchA, patchB=patchB, shrinkA=2, shrinkB=2,
                  connectionstyle=f"arc3,rad={rad}")
    if ls in ("-", "solid"):
        return ax.add_patch(FancyArrowPatch(
            POS[a], POS[b], arrowstyle=ARROW, color=color, lw=lw,
            zorder=z, **common))
    # A headless shaft runs all the way to the box, so its last dash would
    # poke out past the tip; a solid "-|>" instead stops its shaft at the base
    # of the head. End the dashed shaft inside the head to match.
    shaft = ax.add_patch(FancyArrowPatch(
        POS[a], POS[b], arrowstyle="-", color=color, lw=lw, linestyle=ls,
        capstyle="butt", zorder=z,
        **{**common, "shrinkB": 2 + 0.75 * HEAD_LEN}))
    ax.add_patch(FancyArrowPatch(
        POS[a], POS[b], arrowstyle=HEAD_ONLY, color=color, lw=0,
        joinstyle="miter", zorder=z + 0.01, **common))
    return shaft


def build(edges):
    """Draw the ground-truth DAG. Returns (fig, ax, boxes, texts)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for ymin, ymax, c, lab in BANDS:
        ax.axhspan(ymin, ymax, xmin=0.09, color=c, zorder=0)
        ax.text(14.05, (ymin + ymax) / 2, lab, rotation=90, va="center",
                ha="center", fontsize=BAND_FONTSIZE, color="#555",
                fontweight="bold")

    boxes, texts, arrows = {}, {}, {}
    for name, (x, y) in POS.items():
        label = LABEL.get(name, name)
        bold = name in ("num_prb", "goodput")
        if name == "cell_load":
            fc, ec, lw = "#FADBD8", "#C0392B", 3.0
        elif name in EXOGENOUS_STYLE:
            fc, ec, lw = "#F4F4F4", "#909497", 1.6
        else:
            fc, ec, lw = "white", "#34495E", 3.4 if bold else 1.9
        w, h = box_size(name)
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle=f"round,pad={BOX_PAD},rounding_size=0.14",
                             fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(box)
        boxes[name] = box
        texts[name] = ax.text(
            x, y, label, ha="center", va="center", zorder=4,
            fontsize=NODE_FONTSIZE, fontweight="bold" if bold else "normal",
            style="italic" if name in EXOGENOUS_STYLE else "normal")

    for (a, b) in edges:
        if (a, b) in RED_EDGES:
            color, lw, ls, z = "#C0392B", EDGE_LW["red"], "-", 2.5
        elif (a, b) in PURPLE_EDGES:
            color, lw, ls, z = "#7D3C98", EDGE_LW["purple"], (0, (4, 2.5)), 2
        else:
            color, lw, ls, z = "#5D6D7E", EDGE_LW["grey"], "-", 1.5
        arrows[(a, b)] = add_edge(ax, a, b, boxes[a], boxes[b], color, lw,
                                  ls, z, RAD.get((a, b), 0.06))

    # Short labels: the caption carries the explanation, and at column width
    # a long legend entry is both unreadable and wider than the empty corner
    # it sits in.
    handles = [
        plt.Line2D([], [], color="#C0392B", lw=EDGE_LW["red"],
                   label="backdoor edge"),
        plt.Line2D([], [], color="#7D3C98", lw=EDGE_LW["purple"],
                   ls=(0, (4, 2.5)), label="effect modifier"),
        plt.Line2D([], [], color="#5D6D7E", lw=EDGE_LW["grey"],
                   label="structural edge"),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor="#F4F4F4",
                   markeredgecolor="#909497", markersize=13,
                   markeredgewidth=1.6, label="exogenous"),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor="white",
                   markeredgecolor="#34495E", markersize=13,
                   markeredgewidth=3.4, label=r"treatment / outcome"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=LEGEND_FONTSIZE,
              framealpha=0.96, borderpad=0.5, handlelength=1.8,
              handletextpad=0.5, labelspacing=0.35, edgecolor="#BBBBBB",
              bbox_to_anchor=(-0.005, -0.005))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig, ax, boxes, texts, arrows


def main():
    with open(os.path.join(ROOT, "config", "dag_edges.yaml")) as f:
        edges = [tuple(e) for e in yaml.safe_load(f)["edges"]]
    nodes = {n for e in edges for n in e}
    missing = nodes ^ set(POS)
    if missing:
        raise SystemExit(f"layout out of sync with dag_edges.yaml: {missing}")
    # No title: the manuscript caption names the figure, and a title only
    # spends vertical space that the column-width rendering cannot afford.
    fig, *_ = build(edges)
    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "results", "figures", f"fig1_dag.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("saved", out)


if __name__ == "__main__":
    main()
