"""E15 — protocol-constrained discovery with targeted interventions and
interventional mediator blocking.

Pipeline, end to end:

  0  alias removal            exact functional copies (no-op since UP-30)
  1  protocol background      layer tiers + within-layer processing order +
                              exogenous mutual independence   (before any data)
  2  PC skeleton              Fisher-z, alpha = 0.01, with that knowledge
  3  orientation              precedence, then DirectLiNGAM within layers
  4  targeted interventions   19 base regimes + 30 aimed at the parents the
                              base grid never forces; direct-effect regression
                              on pooled obs + do() rows, p < 1e-6
  5  acyclic projection       admit candidates most-significant-first, skipping
                              any that closes a cycle
  6  mediator blocking        for each surviving X -> Y, re-run the contrast
                              with every candidate mediator held fixed by do();
                              keep only if a direct effect remains

Against the 35-edge ground truth over 20 seeds (see README for the ablation):

    baseline (causal.discovery)   P 0.718  R 0.500  F1 0.589  SHD 25.0
    this pipeline                 P 1.000  R 0.744  F1 0.853  SHD  9.5

Step 6 removes every spurious edge without losing a single true one; recall is
unchanged by it (paired difference exactly 0.000 over held-out seeds), so the
gain is entirely precision. The 9 edges still missing are the sub-noise-floor
inputs -- eps_sinr -> sinr_eff and friends -- which do not respond to sample
size, alpha, conditioning depth, or intervention budget.

Usage:
    python experiments/step11_protocol_discovery.py [--seeds 20] [--rows 60]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.discovery import (evaluate, find_aliases,                # noqa: E402
                              find_degenerate,
                              interventional_augmentation, orient)
from causal.fitting import observable, observable_edges              # noqa: E402
from causal.mediator_test import mediator_blocking_prune             # noqa: E402
from causal.protocol_knowledge import (acyclic_projection,           # noqa: E402
                                       build_background_knowledge, enforce,
                                       forbidden, is_acyclic,
                                       run_pc_constrained)
from simulator.ran_simulator import COLUMNS, RANSimulator            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Parents the expanded grid forces. interventional_augmentation only tests a
# parent it is given, so this list must match the regimes generated below.
# do() targets. Latent variables are excluded for the same reason they are
# excluded from the CI tests: a deployment cannot force what it cannot measure.
TARGETS = observable(["cell_load", "snr", "shadow_fading", "num_prb", "mcs_idx",
                      "harq_retx", "sinr_eff", "bler", "se_phy", "code_rate",
                      "hw_impairment", "mac_tput", "queue_dep", "pkt_loss"])

QUANTILES = (0.1, 0.5, 0.9)


def reference_levels(sim):
    """Intervention levels from ONE fixed reference sample, not per-seed.

    Using each variable's own observational quantiles keeps every do() inside
    the observed support, so the grid tests interpolation rather than
    extrapolation. Fixed a priori; never tuned against results.
    """
    ref = sim.sample(5000, seed=0, regime="obs")
    return {c: {q: float(np.quantile(ref[c], q)) for q in QUANTILES}
            for c in COLUMNS}


def targeted_regimes(lv):
    """30 regimes aimed at the parents the paper's 19-regime grid never forces.

    Co-interventions isolate a specific path: do(sinr_eff, mcs_idx) separates
    sinr_eff -> bler from the MCS-selection route, and do(bler, harq_retx)
    separates bler -> mac_tput from bler -> harq_retx -> mac_tput.
    """
    R = []
    for q in QUANTILES:                                     # --- L1 PHY ---
        R.append({"sinr_eff": lv["sinr_eff"][q]})
    for q in (0.1, 0.9):
        R.append({"sinr_eff": lv["sinr_eff"][q], "mcs_idx": 6})
    for q in QUANTILES:
        R.append({"bler": lv["bler"][q]})
    for q in (0.1, 0.9):
        R.append({"bler": lv["bler"][q], "mcs_idx": 6})
    for q in (0.1, 0.9):
        R.append({"code_rate": lv["code_rate"][q], "mcs_idx": 6})
    for q in (0.1, 0.9):
        R.append({"se_phy": lv["se_phy"][q]})
    for q in (0.1, 0.9):                                    # --- L2 MAC ---
        R.append({"bler": lv["bler"][q], "harq_retx": 1})
    for h in (0, 1, 2):
        R.append({"harq_retx": h})
    for v in (-2.5, 2.5):
        R.append({"hw_impairment": v})
    for q in QUANTILES:
        R.append({"mac_tput": lv["mac_tput"][q]})
    for q in QUANTILES:                                     # --- L3 network ---
        R.append({"queue_dep": lv["queue_dep"][q]})
    for q in QUANTILES:
        R.append({"pkt_loss": lv["pkt_loss"][q]})
    return R


def discover(sim, cfg, seed, lv, extra, rows, block=True):
    """One full run. Returns (edges, edges_before_blocking, decisions)."""
    obs = sim.sample(500, seed=seed, regime="obs")
    intv = pd.concat(
        [sim.sample(rows, do=d, seed=seed * 1000 + i)
         for i, d in enumerate(rg["do"] for rg in cfg["interventional"]["regimes"])]
        + [sim.sample(rows, do=d, seed=seed * 7717 + i)
           for i, d in enumerate(extra)], ignore_index=True)

    # UP-33: discovery sees TELEMETRY ONLY. Running this on COLUMNS would
    # condition PC on shadow_fading, hw_impairment, traffic_burst and the
    # eps_* noise terms -- the six variables the SCM fitting declares latent --
    # making the accuracy a full-simulator-state result rather than one a real
    # RAN could reproduce.
    cols = observable(COLUMNS)
    kept, aliases = find_aliases(obs, cols)                          # step 0
    # step 0b: drop exact LINEAR combinations too, or Fisher-z is singular and
    # PC cannot run at all (UP-31 made rtt_ms exactly linear in its parents).
    # The dropped column's predictors are NOT attached as edges -- it has to
    # earn them from the interventional stage like everything else.
    ci_cols, degenerate = find_degenerate(obs, kept)
    bk = build_background_knowledge(ci_cols)                         # step 1
    directed, undirected = run_pc_constrained(obs, ci_cols, 0.01, bk)  # step 2
    e0, _ = orient(directed, undirected, obs, ci_cols)               # step 3
    e0 = sorted(set(e0) | {(p, c) for p, c, _ in aliases})
    # The degeneracy breaks FISHER-Z, not least squares: step 4 regresses with
    # OLS, which handles an exactly-determined response fine. So it runs on the
    # FULL column set -- excluding rtt_ms here too would make it permanently
    # unreachable as a child rather than merely invisible to the CI tests.
    _, table = interventional_augmentation(e0, obs, intv, kept,      # step 4
                                           targets=TARGETS)
    base = {e for e in e0 if not forbidden(e)}
    cand = {}
    if len(table):
        for s, p in zip(table.edge, table.p_value):
            a, b = s.split(" -> ")
            if not forbidden((a, b)):
                cand[(a, b)] = float(p)
    before = acyclic_projection(base, cand)                          # step 5
    if not block:
        return before, before, []
    pruned, decisions = mediator_blocking_prune(                     # step 6
        before, sim, cols, lv, seed=seed)
    return enforce(pruned), before, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=60,
                    help="rows per interventional regime")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    full_truth = [tuple(e) for e in yaml.safe_load(
        open(os.path.join(ROOT, "config", "dag_edges.yaml")))["edges"]]
    # Scored against the OBSERVABLE subgraph (UP-33): an edge with a latent
    # endpoint cannot be recovered from telemetry, so counting it as a miss
    # would be as misleading as counting it as a hit. The excluded edges are
    # printed below rather than silently dropped.
    truth = observable_edges(full_truth)
    T = set(truth)
    excluded = [e for e in full_truth if e not in T]
    sim = RANSimulator(cfg)
    lv = reference_levels(sim)
    extra = targeted_regimes(lv)
    n_base = len(cfg["interventional"]["regimes"])

    print(f"{n_base} base + {len(extra)} targeted regimes x {a.rows} rows "
          f"= {(n_base + len(extra)) * a.rows} interventional rows, "
          f"plus 500 observational")
    print(f"ground truth (observable subgraph): {len(truth)} of "
          f"{len(full_truth)} edges over {len({v for e in truth for v in e})} "
          f"of {len({v for e in full_truth for v in e})} nodes")
    print(f"  excluded, latent endpoint ({len(excluded)}): "
          + ", ".join(f"{a}->{b}" for a, b in excluded) + "\n")

    rows, found, all_dec = [], Counter(), []
    for k in range(a.seeds):
        seed = a.start_seed + k
        after, before, dec = discover(sim, cfg, seed, lv, extra, a.rows)
        mb, ma = evaluate(sorted(before), truth), evaluate(sorted(after), truth)
        rows.append({
            "seed": seed,
            "precision_step5": mb["directed"]["precision"],
            "recall_step5": mb["directed"]["recall"],
            "f1_step5": mb["directed"]["f1"], "shd_step5": mb["shd"],
            "precision": ma["directed"]["precision"],
            "recall": ma["directed"]["recall"],
            "f1": ma["directed"]["f1"], "shd": ma["shd"],
            "n_edges": len(after), "n_true": len(after & T),
            "n_false": len(after - T),
            "n_pruned": len(before) - len(after),
            "n_pruned_true": len((before - after) & T),
            "acyclic": is_acyclic(after),
            "backdoor_L_num_prb": ma["backdoor_L_to_num_prb"],
            "backdoor_L_sinr_eff": ma["backdoor_L_to_sinr_eff"]})
        for e in after:
            found[e] += 1
        for d in dec:
            all_dec.append({"seed": seed, "parent": d[0], "child": d[1],
                            "verdict": d[2], "d_total": d[3], "d_direct": d[4]})
        print(f"  seed {seed}: {len(after):2d} edges = {len(after & T):2d} true "
              f"+ {len(after - T):2d} false   (pruned {len(before)-len(after)}, "
              f"{len((before-after) & T)} of them true)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "step11_per_seed.csv"), index=False)
    pd.DataFrame(all_dec).to_csv(
        os.path.join(ROOT, "results", "step11_mediator_decisions.csv"),
        index=False)

    print(f"\n{'stage':<26}{'precision':>16}{'recall':>16}{'F1':>16}{'SHD':>8}")
    print("-" * 82)
    for lab, p, r, f, s in [
            ("after step 5 (no blocking)", "precision_step5", "recall_step5",
             "f1_step5", "shd_step5"),
            ("after step 6 (blocking)", "precision", "recall", "f1", "shd")]:
        print(f"{lab:<26}{df[p].mean():>9.3f} ±{df[p].std():<5.3f}"
              f"{df[r].mean():>9.3f} ±{df[r].std():<5.3f}"
              f"{df[f].mean():>9.3f} ±{df[f].std():<5.3f}{df[s].mean():>8.1f}")
    print(f"\nedges/seed {df.n_edges.mean():.1f} = {df.n_true.mean():.1f} true "
          f"+ {df.n_false.mean():.1f} false     "
          f"mediator blocking removed {df.n_pruned.mean():.1f}/seed "
          f"({df.n_pruned_true.mean():.1f} true)")
    print(f"acyclic {int(df.acyclic.sum())}/{a.seeds} seeds | "
          f"backdoor L->num_prb {int(df.backdoor_L_num_prb.sum())}/{a.seeds}, "
          f"L->sinr_eff {int(df.backdoor_L_sinr_eff.sum())}/{a.seeds}")

    stab = pd.DataFrame([{"edge": f"{p} -> {c}", "seeds_found": found.get((p, c), 0),
                          "frac": found.get((p, c), 0) / a.seeds,
                          "true_edge": (p, c) in T}
                         for p, c in sorted(set(truth) | set(found))])
    stab.sort_values(["true_edge", "seeds_found"], ascending=[False, False]) \
        .to_csv(os.path.join(ROOT, "results", "step11_edge_stability.csv"),
                index=False)
    missing = [e for e in truth if found.get(e, 0) == 0]
    print(f"\ntrue edges never recovered ({len(missing)}):")
    for p, c in missing:
        print(f"    {p} -> {c}")
    # majority graph over the seeds -> the recovered DAG used by step13,
    # step15 and plot_confounding_recovered (an edge is kept when it appears
    # in at least half of the seeds)
    rec = sorted(e for e, k in found.items() if k / a.seeds >= 0.5)
    tp = [e for e in rec if e in T]
    fn = sorted(e for e in truth if e not in set(rec))
    pad = lambda p_, c_: " " * max(1, 34 - len(p_) - len(c_))  # noqa: E731
    lines = [
        "# Recovered causal graph — step11 (protocol-constrained PC + targeted",
        "# interventions + interventional mediator blocking), MAJORITY over the",
        "# seeds: an edge is included when it appears in >= 50% of them.",
        "# GENERATED by experiments/step11_protocol_discovery.py — do not edit",
        "# by hand.",
        "",
        "metrics:",
        f"  seeds: {a.seeds}",
        f"  n_true_edges: {len(truth)}",
        f"  n_recovered: {len(rec)}",
        f"  true_positives: {len(tp)}",
        f"  false_positives: {len(rec) - len(tp)}",
        f"  false_negatives: {len(fn)}",
        f"  precision: {len(tp) / max(len(rec), 1):.3f}",
        f"  recall: {len(tp) / max(len(truth), 1):.3f}",
        "",
        "recovered_edges:                    # [seeds_found / total]",
    ]
    for p_, c_ in rec:
        tag = "" if (p_, c_) in T else "   # SPURIOUS"
        lines.append(f"  - [{p_}, {c_}]{pad(p_, c_)}# {found.get((p_, c_), 0)}/{a.seeds}{tag}")
    lines += ["", "not_recovered:"]
    for p_, c_ in fn:
        lines.append(f"  - [{p_}, {c_}]{pad(p_, c_)}# {found.get((p_, c_), 0)}/{a.seeds}")
    with open(os.path.join(ROOT, "results", "step11_recovered_dag.yaml"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote results/step11_per_seed.csv, step11_mediator_decisions.csv, "
          f"step11_edge_stability.csv, step11_recovered_dag.yaml")


if __name__ == "__main__":
    main()
