"""Interventional mediator blocking: is X -> Y direct, or routed through M?

The augmentation stage of causal.discovery controls mediators STATISTICALLY --
it regresses Y on X plus every topological predecessor of Y. When the mediating
chain is multiplicative (it is: mac_tput is a product of num_prb, se_phy, a
hardware term and a HARQ term), a linear coefficient cannot absorb it, and the
residual leaks through as an apparent direct effect. That is the sole cause of
the systematic long-range false positives -- bler -> goodput, sinr_eff ->
goodput, num_prb -> queue_dep -- each of which appears in every seed.

This module replaces that with an EXPERIMENTAL control. Rather than
conditioning on the mediator in a regression, hold it fixed by intervention
and re-run the contrast:

    total    D_tot = E[Y | do(X=x1)]           - E[Y | do(X=x0)]
    direct   D_dir = E[Y | do(X=x1), do(M=m)]  - E[Y | do(X=x0), do(M=m)]

    D_tot ~ 0                 -> no effect at all       -> reject X -> Y
    D_tot != 0 and D_dir ~ 0  -> fully mediated         -> reject X -> Y
    D_dir != 0                -> direct effect remains  -> keep   X -> Y

Both arms of every contrast reuse the SAME exogenous draw, so the comparison is
within-unit (the fixed-unit mode of RANSimulator.propagate). In a deterministic
SCM that makes the test effectively exact rather than statistical: with all
mediators blocked, a non-edge's direct effect is 0 to floating-point, not
merely insignificant.

Mediator sets are read from the RECOVERED graph and the protocol order, never
from the ground-truth edge list. For a candidate X -> Y, M qualifies when
  * X precedes M precedes Y in protocol order,
  * M responds to do(X), and
  * M is an ancestor of Y in the recovered graph, or an already-confirmed
    direct child of X.
Qualifying mediators are blocked jointly.

Search order is outward: for each X its candidate children are tested
nearest-first in protocol order, and a child confirmed direct becomes eligible
as a mediator when testing X against more distant variables.

CAVEAT for deployment. This requires do() on internal nodes -- forcing se_phy,
mac_tput or queue_dep to a chosen value. That is available in simulation and
generally is not on a live RAN. The method establishes that interventional
mediator control strictly dominates statistical control; realising it on
hardware needs interventions that are actually actuable.
"""
import numpy as np
from scipy import stats

from .protocol_knowledge import protocol_position

# do() targets the simulator requires as integers
_INT_NODES = {"mcs_idx"}


def _ancestors(edges, y):
    """Ancestors of y in the RECOVERED graph."""
    par = {}
    for a, b in edges:
        par.setdefault(b, []).append(a)
    seen, stack = set(), list(par.get(y, []))
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(par.get(n, []))
    return seen


def paired_contrast(sim, rng, x, x_low, x_high, hold, columns, n=2000):
    """Paired do() contrast. Returns {column: (mean difference, p-value)}.

    One call yields the response of EVERY variable to do(X), so the screening
    pass over candidate children costs a single contrast per parent.
    """
    u = sim.sample_exogenous(n, rng)

    def arm(v):
        d = {x: int(round(v)) if x in _INT_NODES else float(v)}
        for m, mv in hold.items():
            d[m] = int(round(mv)) if m in _INT_NODES else float(mv)
        return sim.propagate(u, do=d)

    lo, hi = arm(x_low), arm(x_high)
    out = {}
    for c in columns:
        diff = hi[c].values - lo[c].values
        if diff.std() < 1e-12:                  # deterministic response
            out[c] = (float(diff.mean()),
                      0.0 if abs(diff.mean()) > 1e-12 else 1.0)
        else:
            out[c] = (float(diff.mean()),
                      float(stats.ttest_1samp(diff, 0.0).pvalue))
    return out


def mediator_blocking_prune(edges, sim, columns, levels, seed=0, alpha=0.01,
                            tau=0.0, n=2000):
    """Prune indirect edges from a recovered graph.

    levels : {node: {0.1: low, 0.5: mid, 0.9: high}} intervention levels, fixed
             a priori from a reference sample -- never per-seed.
    tau    : direct effect must satisfy |D_dir| >= tau * |D_tot|. Retained as a
             tunable, but on a deterministic SCM the decision is insensitive to
             it (measured identical across 0.00-0.20) because a blocked
             non-edge gives exactly zero.

    Returns (kept_edges, decisions) where decisions is a list of
    (parent, child, verdict, d_total, d_direct).
    """
    rng = np.random.default_rng(seed * 9187 + 13)
    kept, decisions = set(), []
    by_parent = {}
    for a, b in edges:
        by_parent.setdefault(a, []).append(b)

    for x in sorted(by_parent, key=protocol_position):
        lo, hi = levels[x][0.1], levels[x][0.9]
        if abs(hi - lo) < 1e-12:                       # not manipulable
            kept.update((x, y) for y in by_parent[x])
            for y in by_parent[x]:
                decisions.append((x, y, "not manipulable", np.nan, np.nan))
            continue

        resp = paired_contrast(sim, rng, x, lo, hi, {}, columns, n)
        responders = {c for c in columns
                      if resp[c][1] < alpha and abs(resp[c][0]) > 1e-9}
        confirmed = set()
        for y in sorted(by_parent[x], key=protocol_position):   # nearest first
            d_tot, p_tot = resp[y]
            if p_tot >= alpha or abs(d_tot) < 1e-9:
                decisions.append((x, y, "rejected: no total effect", d_tot, 0.0))
                continue
            anc = _ancestors(edges, y)
            meds = [m for m in columns
                    if m not in (x, y)
                    and protocol_position(x) < protocol_position(m)
                    < protocol_position(y)
                    and m in responders and (m in anc or m in confirmed)]
            if not meds:
                kept.add((x, y))
                confirmed.add(y)
                decisions.append((x, y, "kept: no mediator", d_tot, d_tot))
                continue
            hold = {m: levels[m][0.5] for m in meds}
            d_dir, p_dir = paired_contrast(sim, rng, x, lo, hi, hold,
                                           columns, n)[y]
            if p_dir < alpha and abs(d_dir) >= tau * abs(d_tot):
                kept.add((x, y))
                confirmed.add(y)
                decisions.append((x, y, f"kept: direct (blocked {len(meds)})",
                                  d_tot, d_dir))
            else:
                decisions.append((x, y, "rejected: mediated via "
                                  + ",".join(meds), d_tot, d_dir))
    return kept, decisions
