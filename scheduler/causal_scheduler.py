"""Counterfactual PRB scheduling (paper eq. 5).

A cell serves K UEs and must divide a PRB budget B between them. The paper
poses this as

    max_{t}  sum_u w_u * Yhat_t^(u)     s.t.  sum_u t^(u) <= B,

where Yhat_t^(u) is the UE's *counterfactual* goodput under allocation t —
what that specific UE would achieve, given its own latent channel, device
and traffic state. This module turns that statement into a working
scheduler and provides the baselines it must be compared against.

Policies
--------
equal         B/K to everyone. The reference point.
proportional  allocate in proportion to each UE's *observed* goodput. This
              is what a KPI-driven xApp does, and it is exactly the
              correlational mistake the paper warns about: a UE looks good
              partly because its neighbour load is low, not because it
              converts PRBs efficiently.
rung2         allocate using the population dose-response E[Y | do(prb=k)],
              identical for every UE. NOTE: with one shared concave curve
              the marginal gain is the same for all UEs at equal
              allocation, so this provably reduces to `equal`. Rung 2
              simply has no per-UE information to act on.
rung3         allocate using per-UE counterfactuals from the fitted SCM.
oracle        same, but using the simulator's exact counterfactuals — the
              attainable upper bound.

Allocation rule
---------------
Goodput is concave in PRBs (sub-linear: the bottleneck shifts from
bandwidth to channel quality), so greedy marginal allocation is optimal for
this separable budget problem: hand each successive PRB to whichever UE has
the largest marginal counterfactual gain. Concavity is asserted, not
assumed — `marginal_gain_table` reports any violations.
"""
import numpy as np

K_MIN, K_MAX = 1, 100


def jain(x):
    """Jain's fairness index in [1/K, 1]; 1 = perfectly equal."""
    x = np.asarray(x, dtype=float)
    s = x.sum()
    return float(s * s / (len(x) * np.sum(x * x))) if np.any(x) else 0.0


def greedy_allocate(utility, budget, k_min=K_MIN, k_max=K_MAX):
    """Maximise sum_u utility[u, alloc_u] subject to sum(alloc) <= budget.

    utility: (K, k_max+1) array; utility[u, k] = predicted outcome for UE u
    given k PRBs. Optimal when each row is concave in k.
    """
    K = utility.shape[0]
    alloc = np.full(K, k_min, dtype=int)
    remaining = int(budget) - K * k_min
    if remaining < 0:
        raise ValueError(f"budget {budget} below the floor of {K} x {k_min}")
    # marginal gain of giving UE u its next PRB
    nxt = np.where(alloc + 1 <= k_max,
                   utility[np.arange(K), np.minimum(alloc + 1, k_max)]
                   - utility[np.arange(K), alloc], -np.inf)
    for _ in range(remaining):
        u = int(np.argmax(nxt))
        if not np.isfinite(nxt[u]):
            break                      # everyone at k_max
        alloc[u] += 1
        nxt[u] = (utility[u, alloc[u] + 1] - utility[u, alloc[u]]
                  if alloc[u] + 1 <= k_max else -np.inf)
    return alloc


def proportional_allocate(observed, budget, k_min=K_MIN, k_max=K_MAX):
    """Allocate proportionally to an observed KPI (the correlational policy)."""
    w = np.clip(np.asarray(observed, dtype=float), 1e-9, None)
    raw = k_min + (budget - len(w) * k_min) * w / w.sum()
    alloc = np.clip(np.round(raw), k_min, k_max).astype(int)
    return _repair(alloc, budget, k_min, k_max)


def equal_allocate(K, budget, k_min=K_MIN, k_max=K_MAX):
    alloc = np.full(K, min(max(budget // K, k_min), k_max), dtype=int)
    return _repair(alloc, budget, k_min, k_max)


def _repair(alloc, budget, k_min, k_max):
    """Nudge an allocation until it exactly meets the budget."""
    alloc = alloc.astype(int).copy()
    while alloc.sum() > budget:
        cand = np.where(alloc > k_min)[0]
        if not len(cand):
            break
        alloc[cand[np.argmax(alloc[cand])]] -= 1
    while alloc.sum() < budget:
        cand = np.where(alloc < k_max)[0]
        if not len(cand):
            break
        alloc[cand[np.argmin(alloc[cand])]] += 1
    return alloc


def dp_allocate(utility, budget, k_min=K_MIN, k_max=K_MAX):
    """EXACT optimum of  max sum_u utility[u, k_u]  s.t.  sum_u k_u <= budget.

    Multiple-choice knapsack by dynamic programming — O(K * B * k_max), which
    the paper quotes as O(KB). Unlike greedy this needs no concavity
    assumption, so it is the correct optimizer for a true oracle: the oracle
    must maximise the *exact* simulator utility, not a smoothed stand-in.

    Using the same routine for the learned utility keeps the comparison
    optimizer-matched: any gap is then attributable to the causal model,
    not to the search.
    """
    K = utility.shape[0]
    NEG = -np.inf
    dp = np.full((K + 1, budget + 1), NEG)
    dp[0, 0] = 0.0
    choice = np.zeros((K, budget + 1), dtype=np.int32)
    for u in range(K):
        for k in range(k_min, min(k_max, budget) + 1):
            if not np.isfinite(utility[u, k]):
                continue
            cand = dp[u, :budget + 1 - k] + utility[u, k]
            better = cand > dp[u + 1, k:]
            dp[u + 1, k:][better] = cand[better]
            choice[u, k:][better] = k
    b = int(np.nanargmax(np.where(np.isfinite(dp[K]), dp[K], NEG)))
    if not np.isfinite(dp[K, b]):
        raise ValueError(f"no feasible allocation: {K} UEs, budget {budget}, "
                         f"k in [{k_min}, {k_max}]")
    alloc = np.zeros(K, dtype=int)
    for u in range(K - 1, -1, -1):
        alloc[u] = choice[u, b]
        b -= alloc[u]
    return alloc


def smooth_utility(predict_fn, probes, k_min=K_MIN, k_max=K_MAX, degree=1):
    """Utility table from a SMOOTH surrogate fitted to counterfactual probes.

    Why this is needed. A gradient-boosted structural equation predicts
    Yhat_u(k) accurately at any single k, but as a function of k it is
    piecewise constant: its slope is zero almost everywhere and jumps at
    split points. Greedy allocation consumes exactly that slope, so raw tree
    curves make the scheduler behave close to randomly.

    The fix uses the quantity the model estimates *reliably* — the
    wide-range counterfactual contrast Yhat_u(k2) - Yhat_u(k1), which tracks
    truth at r ~ 0.97 — rather than local derivatives: evaluate the model at
    a handful of well-supported probe allocations and fit a low-order
    polynomial in k through them, per UE.

    probes must lie inside the observational support, or the model is being
    asked to extrapolate (see the positivity discussion).
    """
    probes = np.asarray(probes, dtype=float)
    Y = np.column_stack([predict_fn(int(k)) for k in probes])   # (K, |probes|)
    coef = np.polyfit(probes, Y.T, degree)                      # (deg+1, K)
    ks = np.arange(k_max + 1, dtype=float)
    fitted = np.polyval(coef, ks[:, None]).T                    # (K, k_max+1)
    util = np.full_like(fitted, -np.inf)
    util[:, k_min:] = fitted[:, k_min:]
    return util


def marginal_gain_table(predict_fn, k_grid):
    """Build utility[u, k] from a callable and flag concavity violations.

    predict_fn(k) -> array of per-UE predicted outcomes at allocation k.
    Returns (utility over 0..K_MAX by interpolation, n_concavity_violations).
    """
    vals = np.column_stack([predict_fn(k) for k in k_grid])   # (K, |grid|)
    full = np.zeros((vals.shape[0], K_MAX + 1))
    for u in range(vals.shape[0]):
        full[u] = np.interp(np.arange(K_MAX + 1), k_grid, vals[u])
    second = np.diff(full, n=2, axis=1)
    return full, int(np.sum(second > 1e-6))


def evaluate(goodput, rtt):
    """Scheduler-level metrics."""
    return {"total_goodput": float(np.sum(goodput)),
            "mean_goodput": float(np.mean(goodput)),
            "min_goodput": float(np.min(goodput)),
            "mean_rtt": float(np.mean(rtt)),
            "p90_rtt": float(np.percentile(rtt, 90)),
            "jain_fairness": jain(goodput)}
