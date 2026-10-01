"""Type sizes for figures placed at \\columnwidth in the manuscript.

Matplotlib sizes are points on the CANVAS, but the paper scales every figure
to the column (~3.5 in). A 10 pt label on a 9.6 in canvas therefore prints at
3.6 pt. Figures pick their sizes from here so that what is chosen is the size
on the PAGE, and so that all figures in the paper share one type hierarchy.

The legend target matches the DAG figures (14.5 pt on a 9 in canvas).
"""
COLUMN_IN = 3.5

# points as printed in the two-column paper
ON_PAGE = {
    "title": 7.0,    # panel letters / short panel titles
    "label": 6.6,    # axis labels
    "tick": 6.0,     # tick labels
    "legend": 5.6,   # legend entries -- same as the DAG figures
    "note": 5.4,     # in-plot annotations (bar values, n=, deciles)
}


def sizes(fig_width_in):
    """Canvas point sizes that print at ON_PAGE when scaled to the column."""
    scale = COLUMN_IN / fig_width_in
    return {k: v / scale for k, v in ON_PAGE.items()}


def apply(plt, fig_width_in):
    """Set rcParams so every default text in the figure follows ON_PAGE."""
    s = sizes(fig_width_in)
    plt.rcParams.update({
        "axes.titlesize": s["title"], "axes.labelsize": s["label"],
        "xtick.labelsize": s["tick"], "ytick.labelsize": s["tick"],
        "legend.fontsize": s["legend"],
    })
    return s
