"""E17 — Phases 6 and 7 regenerated on the RECOVERED graph, side by side.

step6 (Rungs 1-2) hardcodes the backdoor adjustment set to {cell_load}; step7
(Rung 3) fits and propagates over config/dag_edges.yaml. Both therefore assume
the true graph. Here each is rerun with the graph discovery actually produced,
so the adjustment set is DERIVED from that graph rather than assumed:

    Z_hat = observable parents of the treatment in the recovered graph

and the SCM is fitted and propagated over the recovered topology. Everything
else -- estimators, unit filters, strata edges, oracle -- is byte-identical to
step6/step7, so any difference is attributable to the graph alone.

Reported quantities match those experiments exactly:
  Phase 6   naive OLS, adjusted OLS, adjusted GBM, |error| vs the high-N
            oracle, and bias% at t in {10,50,100}
  Phase 7   the learned-vs-oracle validation table, and CF-1/2/3 with the
            same unit filters and stratification

SEED CAVEAT. This runs ONE realization (default seed 42), because it exists to
reproduce the Phase 6/7 report tables line for line; step12 carries the 20-seed
version of the same comparison. Two consequences:
  * CF-2 is now do(xi_hw = 0) (UP-29), applied to mac_tput's abducted
    multiplicative residual. On the RECOVERED graph this is a real structural
    test: the hardware CF can only reach goodput/rtt/pkt_loss through the
    edges discovery actually found, so a missed mac_tput -> * edge shows up
    as a descendant that never gets recomputed. That is the failure mode this
    script exists to expose, not to hide.
  * The CF-3 stratum ratio (98x true vs 39x recovered) is NOT safe to quote
    from one seed. That statistic is already known to be seed-sensitive
    (median 12x, range 6-127x over 20 seeds), so 98 vs 39 sits well inside its
    existing spread and should not be read as a graph effect.

Usage: python experiments/step13_phase67_recovered.py [--seed 42]
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from causal.counterfactual import (CounterfactualEngine,             # noqa: E402
                                   cf_statistics, latent_fingerprints,
                                   peer_underperformers, stratify)
from causal.discovery import evaluate
from causal.fitting import observable_edges  # noqa: E402                                # noqa: E402
from causal.fitting import LATENT_NODES, StructuralModel             # noqa: E402
from simulator.ran_simulator import COLUMNS, ENDOGENOUS, RANSimulator  # noqa: E402
from step11_protocol_discovery import (discover, reference_levels,   # noqa: E402
                                       targeted_regimes)

ROOT = os.path.join(os.path.dirname(__file__), "..")
HW_EDGES = [(-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)]
HW_LABELS = ["xi_hw<-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]
GBM = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
T_GRID = [10, 50, 100]
TREATMENT, OUTCOME = "num_prb", "goodput"


def backdoor_set(edges, treatment=TREATMENT):
    """Observable parents of the treatment IN THIS GRAPH.

    For the H-SCM the parents of T are exactly the backdoor admissible set:
    every backdoor path from T to Y starts with an edge into T, so blocking
    T's observable parents blocks all of them. On the true DAG this yields
    {cell_load} (eps_sched is latent); on a recovered graph it yields whatever
    that graph believes -- which is the point of the experiment.
    """
    if edges is None:
        with open(os.path.join(ROOT, "config", "dag_edges.yaml")) as f:
            edges = [tuple(e) for e in yaml.safe_load(f)["edges"]]
    return sorted({a for a, b in edges
                   if b == treatment and a not in LATENT_NODES})


def naive_ols(obs, t):
    b, a = np.polyfit(obs[TREATMENT], obs[OUTCOME], 1)
    return a + b * t


def adjusted_ols(obs, t, Z):
    if not Z:                                   # no adjustment set -> naive
        return naive_ols(obs, t)
    X = np.column_stack([np.ones(len(obs)), obs[TREATMENT].values,
                         obs[Z].values.astype(float)])
    beta, *_ = np.linalg.lstsq(X, obs[OUTCOME].values, rcond=None)
    return float(beta[0] + beta[1] * t
                 + obs[Z].values.astype(float).mean(axis=0) @ beta[2:])


def adjusted_gbm(obs, t, Z):
    if not Z:
        return naive_ols(obs, t)
    m = GradientBoostingRegressor(**GBM).fit(
        obs[[TREATMENT] + Z].values.astype(float), obs[OUTCOME].values)
    X = np.column_stack([np.full(len(obs), t), obs[Z].values.astype(float)])
    return float(m.predict(X).mean())


def phase6(obs, big, Z):
    rows = []
    for t in T_GRID:
        mn, ma = naive_ols(obs, t), adjusted_ols(obs, t, Z)
        mg = adjusted_gbm(obs, t, Z)
        truth = big[t].mean()
        rows.append({"t": t, "naive_ols": mn, "adjusted_ols": ma,
                     "adjusted_gbm": mg, "oracle": truth,
                     "abs_err_naive": abs(mn - truth),
                     "abs_err_adj": abs(ma - truth),
                     "bias_naive_pct": 100 * (mn - truth) / truth,
                     "bias_adj_pct": 100 * (ma - truth) / truth})
    return pd.DataFrame(rows)


def phase7(obs, oracle, edges, kappa_hw):
    # step7 uses TWO models, and so does this: a full-data fit for the
    # held-out oracle validation, and a cross-fitted one for CF-1/2/3 so no
    # unit is abducted by a model that saw it.
    sm_full = StructuralModel(ROOT, mode="telemetry", edges=edges,
                              nodes=ENDOGENOUS).fit(obs, cv_folds=3)
    eng_full = CounterfactualEngine(sm_full, ROOT, edges=edges, nodes=COLUMNS)
    sm = StructuralModel(ROOT, mode="telemetry", edges=edges,
                         nodes=ENDOGENOUS).fit_crossfit(obs, folds=5)
    eng = CounterfactualEngine(sm, ROOT, edges=edges, nodes=COLUMNS)
    fold = sm.fold_of
    res = eng.abduct(obs, fold_of=fold)
    fps = latent_fingerprints(res, obs, kappa_hw)
    sf_hat = fps["xi_sf_hat"]
    hw_hat = fps["xi_hw_hat"]         # CF-2's stratifier (UP-29)

    # -- learned vs oracle on the 5000 held-out UEs --------------------
    fact = oracle[oracle.arm == "factual"].reset_index(drop=True)
    ores = eng_full.abduct(fact)
    val = []
    for arm, (do, col) in {"do(num_prb=10)": ({"num_prb": 10}, "goodput"),
                           "do(num_prb=50)": ({"num_prb": 50}, "goodput"),
                           "do(num_prb=100)": ({"num_prb": 100}, "goodput"),
                           "do(hw_impairment=0)": ({"hw_impairment": 0.0},
                                                   "goodput"),
                           "do(traffic_burst=0.5)": ({"traffic_burst": 0.5},
                                                     "rtt_ms")}.items():
        truth = oracle[oracle.arm == arm].reset_index(drop=True)[col].values
        try:
            pred = eng_full.predict(fact, do, ores)[col].values
        except ValueError:        # latent do() with no observable proxy
            val.append({"arm": arm, "outcome": col, "r2": np.nan,
                        "corr": np.nan, "rmse": np.nan,
                        "oracle_effect": float(np.mean(truth)
                                               - fact[col].mean())})
            continue
        err = pred - truth
        val.append({"arm": arm, "outcome": col,
                    "r2": 1 - np.sum(err**2) / np.sum((truth - truth.mean())**2),
                    "corr": float(np.corrcoef(pred, truth)[0, 1]),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "oracle_effect": float(np.mean(truth) - fact[col].mean())})

    # -- CF-1/2/3, identical filters and strata to step7 ---------------
    out, strata, arrays = [], [], {}
    def cf(mask, do, col, name, key, edges_, labels):
        if key is None:               # stratification latent unavailable
            out.append({"query": name, "outcome": col, "n": int(mask.sum()),
                        "factual_mean": obs.loc[mask, col].mean(),
                        "cf_mean": np.nan, "ks_stat": np.nan, "ks_p": np.nan,
                        "w1": np.nan, "note": "latent not abducible (UP-28)"})
            return
        try:
            c = eng.predict(obs[mask], do,
                            {k: v[mask.values] for k, v in res.items()},
                            fold_of=fold[mask.values])
        except ValueError:            # do() on a latent with no proxy
            out.append({"query": name, "outcome": col, "n": int(mask.sum()),
                        "factual_mean": obs.loc[mask, col].mean(),
                        "cf_mean": np.nan, "ks_stat": np.nan, "ks_p": np.nan,
                        "w1": np.nan,
                        "note": "not realisable in telemetry mode (UP-28)"})
            return
        s = cf_statistics(obs.loc[mask, col].values, c[col].values, name)
        s["outcome"] = col          # the figure reads the unit from here
        out.append(s)
        arrays[f"{name}_fact"] = obs.loc[mask, col].values
        arrays[f"{name}_cf"] = c[col].values
        delta = c[col].values - obs.loc[mask, col].values
        st = stratify(key[mask.values], delta, edges_, labels)
        st["query"] = name
        strata.append(st)

    cf((obs.goodput >= 5) & (obs.goodput <= 40), {"num_prb": 100}, "goodput",
       "CF-1", sf_hat, [(-99, -7), (-7, -3), (-3, 3), (3, 7), (7, 99)],
       ["xi_sf<-7", "-7..-3", "-3..3", "3..7", ">7"])
    cf(peer_underperformers(obs), {"hw_impairment": 0.0},
       "goodput", "CF-2", hw_hat, HW_EDGES, HW_LABELS)
    cf((obs.snr >= 10) & (obs.snr <= 20) & (obs.goodput < 15),
       {"num_prb": 100}, "goodput", "CF-3", sf_hat,
       [(-99, -5), (-5, 5), (5, 99)],
       ["xi_sf<-5 (channel-limited)", "-5..5", ">+5 (PRB-limited)"])
    # held-out oracle scatter at do(prb=100), for panel (a)
    arrays["val_pred"] = eng_full.predict(
        fact, {"num_prb": 100}, ores)["goodput"].values
    arrays["val_true"] = oracle[oracle.arm == "do(num_prb=100)"] \
        .reset_index(drop=True)["goodput"].values
    return (pd.DataFrame(val), pd.DataFrame(out),
            pd.concat(strata, ignore_index=True), arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    # scored against the observable subgraph, as step11 is (UP-33)
    truth = observable_edges([tuple(e) for e in yaml.safe_load(
        open(os.path.join(ROOT, "config", "dag_edges.yaml")))["edges"]])
    sim = RANSimulator(cfg)
    kappa_hw = float(cfg["mac"]["kappa_hw"])

    lv = reference_levels(sim)
    rec = sorted(discover(sim, cfg, a.seed, lv, targeted_regimes(lv), 60)[0])
    m = evaluate(rec, truth)
    print(f"recovered graph (seed {a.seed}): {len(rec)} edges, "
          f"{len(set(rec) & set(truth))} correct, "
          f"{len(set(rec) - set(truth))} spurious "
          f"(P {m['directed']['precision']:.3f} / R {m['directed']['recall']:.3f})")

    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    intv = pd.read_csv(os.path.join(ROOT, "data", "interventional.csv"))
    oracle = pd.read_csv(os.path.join(ROOT, "data", "oracle_paired.csv.gz"))
    big = {t: oracle.loc[oracle.arm == f"do(num_prb={t})", "goodput"]
           for t in T_GRID}

    Zt, Zr = backdoor_set(None), backdoor_set(rec)
    print(f"\nPHASE 6 — backdoor adjustment set DERIVED from each graph")
    print(f"  true DAG        Z = {Zt}")
    print(f"  recovered graph Z = {Zr}"
          + ("   (identical -- both backdoor edges recovered)"
             if Zt == Zr else "   (DIFFERENT)"))

    p6t, p6r = phase6(obs, big, Zt), phase6(obs, big, Zr)
    print(f"\n{'t':>5}{'naive':>9}{'adj (true)':>12}{'adj (rec)':>11}"
          f"{'oracle':>9}{'|err| naive':>13}{'|err| true':>12}{'|err| rec':>11}")
    print("-" * 82)
    for (_, x), (_, y) in zip(p6t.iterrows(), p6r.iterrows()):
        print(f"{int(x.t):>5}{x.naive_ols:>9.2f}{x.adjusted_ols:>12.2f}"
              f"{y.adjusted_ols:>11.2f}{x.oracle:>9.2f}"
              f"{x.abs_err_naive:>13.2f}{x.abs_err_adj:>12.2f}"
              f"{y.abs_err_adj:>11.2f}")
    r100t = p6t[p6t.t == 100].iloc[0]
    r100r = p6r[p6r.t == 100].iloc[0]
    print(f"\n  confounding bias at t=100: naive {r100t.bias_naive_pct:+.1f}% "
          f"-> adjusted {r100t.bias_adj_pct:+.1f}% (true DAG) / "
          f"{r100r.bias_adj_pct:+.1f}% (recovered)")

    print(f"\nPHASE 7 — Rung 3")
    vt, ct, st, at = phase7(obs, oracle, None, kappa_hw)
    vr, cr, sr, ar = phase7(obs, oracle, rec, kappa_hw)
    np.savez(os.path.join(ROOT, "results", "step13_cf_arrays.npz"),
             **{f"true_{k}": v for k, v in at.items()},
             **{f"rec_{k}": v for k, v in ar.items()})

    print(f"\n  learned vs oracle (5000 held-out UEs)")
    print(f"  {'arm':<24}{'R2 true':>10}{'R2 rec':>10}{'corr true':>11}"
          f"{'corr rec':>10}{'RMSE true':>11}{'RMSE rec':>10}")
    print("  " + "-" * 86)
    for (_, x), (_, y) in zip(vt.iterrows(), vr.iterrows()):
        f = lambda v, w: (f"{v:>{w}.3f}" if v == v else f"{'n/a':>{w}}")
        print(f"  {x['arm']:<24}{f(x['r2'],10)}{f(y['r2'],10)}"
              f"{f(x['corr'],11)}{f(y['corr'],10)}"
              f"{f(x['rmse'],11)}{f(y['rmse'],10)}"
              + (f"   oracle effect {x['oracle_effect']:+.3f}"
                 if x['r2'] != x['r2'] else ""))

    print(f"\n  Table III bottom — CF-1/2/3 on the canonical 500")
    print(f"  {'query':<7}{'n':>5}{'factual':>10}{'CF true':>10}{'CF rec':>10}"
          f"{'delta true':>12}{'delta rec':>11}{'KS true':>9}{'KS rec':>8}")
    print("  " + "-" * 84)
    for (_, x), (_, y) in zip(ct.iterrows(), cr.iterrows()):
        unit = "Mbps"
        g = lambda v, w, sign=False: (
            (f"{v:>+{w}.2f}" if sign else f"{v:>{w}.2f}") if v == v
            else f"{'n/a':>{w}}")
        print(f"  {x['query']:<7}{int(x['n']):>5}{x['factual_mean']:>10.2f}"
              f"{g(x['cf_mean'],10)}{g(y['cf_mean'],10)}"
              f"{g(x['cf_mean']-x['factual_mean'],12,True)}"
              f"{g(y['cf_mean']-y['factual_mean'],11,True)}"
              f"{g(x['ks_stat'],9)}{g(y['ks_stat'],8)}   {unit}")

    print(f"\n  CF-3 stratum ratio (PRB-limited vs channel-limited gain)")
    for lab, s in (("true DAG", st), ("recovered", sr)):
        d = s[s['query'] == "CF-3"]
        hi = d[d.stratum.str.contains("PRB-limited")].mean_effect.values
        lo = d[d.stratum.str.contains("channel-limited")].mean_effect.values
        if len(hi) and len(lo) and abs(lo[0]) > 1e-9:
            print(f"    {lab:<12} {hi[0]:.2f} vs {lo[0]:.2f} Mbps "
                  f"-> {hi[0]/lo[0]:.0f}x")

    pd.concat([vt.assign(graph="true"), vr.assign(graph="recovered")]).to_csv(
        os.path.join(ROOT, "results", "step13_val.csv"), index=False)
    pd.concat([p6t.assign(graph="true"), p6r.assign(graph="recovered")]).to_csv(
        os.path.join(ROOT, "results", "step13_phase6.csv"), index=False)
    pd.concat([ct.assign(graph="true"), cr.assign(graph="recovered")]).to_csv(
        os.path.join(ROOT, "results", "step13_phase7.csv"), index=False)
    pd.concat([st.assign(graph="true"), sr.assign(graph="recovered")]).to_csv(
        os.path.join(ROOT, "results", "step13_strata.csv"), index=False)
    print(f"\nwrote results/step13_phase6.csv, step13_phase7.csv, "
          f"step13_strata.csv")


if __name__ == "__main__":
    main()
