"""E9-E11 — Phase 7: Rung 3, unit-level counterfactual inference.

Part 1 (decisive): LEARNED-vs-ORACLE validation. The telemetry-mode SCM is
fitted on the 500 observational rows, then applied to the 5000 *disjoint*
paired-oracle units: abduct from their factual arm, predict under each do()
arm, compare per unit against the simulator's exact counterfactual. Zero
leakage — the model never saw these units.

Part 2 (paper reproduction): CF-1/2/3 on the canonical 500 observational
rows with CROSS-FITTED abduction, reporting means, KS, Wasserstein-1 and
per-stratum spreads. Strata use ABDUCTED latents (xi_sf_hat, xi_hw_hat),
never the simulator's true values (UP-25).

CF-2 (UP-29). The paper's CF-2 was do(xi_tb = 0.5) on RTT. xi_tb has no
telemetry realisation -- no observable inverts it -- so that query is reported
here as unavailable (its oracle effect is still printed, for the record). The
query used instead is

    CF-2: what would this UE achieve if its device RF quality were
          normalised to the nominal hardware condition?   do(xi_hw = 0)

which needs no proxy: xi_hw is the whole of mac_tput's multiplicative noise
term, so the action is applied to that node's own abducted residual
(LATENT_RESIDUAL). P_2 is defined from telemetry — UEs below 0.95x the
median service rate of their radio-state peers — never from the abducted
latent, which would select partly on estimation error.

Outputs: results/table3_bottom.csv, results/cf_oracle_validation.csv,
         results/cf_strata.csv, results/figures/step7_counterfactuals.png
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.counterfactual import (CounterfactualEngine, cf_statistics,   # noqa: E402
                                   latent_fingerprints,
                                   peer_underperformers, stratify)
from causal.fitting import StructuralModel                                # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
# CF-2 strata, in units of xi_hw (sd 1.5). The middle bin is the nominal-
# hardware control: its effect must be ~0 while the outer bins are large and
# of OPPOSITE sign — normalising repairs a bad device and degrades a good one.
HW_EDGES = [(-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)]
HW_LABELS = ["xi_hw<-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]


def validate_against_oracle(engine, oracle):
    """Per-unit learned-CF vs simulator-oracle-CF on disjoint units."""
    fact = oracle[oracle.arm == "factual"].reset_index(drop=True)
    res = engine.abduct(fact)
    rows, preds = [], {}
    arms = {"do(num_prb=10)": ({"num_prb": 10}, "goodput"),
            "do(num_prb=50)": ({"num_prb": 50}, "goodput"),
            "do(num_prb=100)": ({"num_prb": 100}, "goodput"),
            "do(hw_impairment=0)": ({"hw_impairment": 0.0}, "goodput"),
            "do(traffic_burst=0.5)": ({"traffic_burst": 0.5}, "rtt_ms")}
    for arm, (do, col) in arms.items():
        truth = oracle[oracle.arm == arm].reset_index(drop=True)[col].values
        try:
            pred = engine.predict(fact, do, res)[col].values
        except ValueError as e:      # latent do() with no observable proxy
            fac = fact[col].values
            rows.append({"arm": arm, "outcome": col, "n": len(truth),
                         "oracle_mean": np.mean(truth),
                         "learned_mean": np.nan, "bias": np.nan,
                         "mae": np.nan, "rmse": np.nan, "nrmse_pct": np.nan,
                         "corr": np.nan, "r2": np.nan,
                         "r2_effect": np.nan, "corr_effect": np.nan,
                         "note": "NOT REALISABLE in telemetry mode (UP-28); "
                                 f"oracle effect on {col}: "
                                 f"{np.mean(truth) - np.mean(fac):+.3f}"})
            preds[arm] = (None, truth, col)
            continue
        preds[arm] = (pred, truth, col)
        err = pred - truth
        denom = max(np.mean(np.abs(truth)), 1e-9)
        # effect-level agreement, scored on dY = Y_cf - Y_factual. For arms
        # whose effect is small relative to var(Y) — do(xi_hw=0) above all —
        # the outcome-level R2 is nearly free (predicting "nothing changes"
        # already scores ~0.99), so r2_effect is the number that means
        # something. Both are reported; neither is hidden.
        dp = pred - fact[col].values
        dt = truth - fact[col].values
        rows.append({
            "arm": arm, "outcome": col, "n": len(truth),
            "oracle_mean": np.mean(truth), "learned_mean": np.mean(pred),
            "bias": np.mean(err), "mae": np.mean(np.abs(err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "nrmse_pct": 100 * float(np.sqrt(np.mean(err ** 2))) / denom,
            "corr": float(np.corrcoef(pred, truth)[0, 1]),
            "r2": 1 - np.sum(err ** 2) / np.sum((truth - truth.mean()) ** 2),
            "r2_effect": 1 - np.sum((dp - dt) ** 2)
                         / max(np.sum((dt - dt.mean()) ** 2), 1e-12),
            "corr_effect": float(np.corrcoef(dp, dt)[0, 1])
                           if dt.std() > 1e-12 else np.nan})
    return pd.DataFrame(rows), preds, fact, res


def main():
    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        kappa_hw = float(yaml.safe_load(f)["mac"]["kappa_hw"])
    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    oracle = pd.read_csv(os.path.join(ROOT, "data", "oracle_paired.csv.gz"))

    # fit on the canonical 500 (telemetry mode), full-data model
    model = StructuralModel(ROOT, mode="telemetry").fit(obs, cv_folds=5)
    engine = CounterfactualEngine(model, ROOT)

    # ---------------- Part 1: learned vs oracle ----------------
    val, preds, ofact, ores = validate_against_oracle(engine, oracle)
    val.to_csv(os.path.join(ROOT, "results", "cf_oracle_validation.csv"),
               index=False)
    print("PART 1 — learned SCM vs simulator oracle, per unit "
          "(N=5000 units never seen in training):")
    print(f"{'arm':>22} {'outcome':>8} {'oracle':>8} {'learned':>8} "
          f"{'bias':>7} {'MAE':>7} {'RMSE':>7} {'corr':>6} {'R2':>6} "
          f"{'R2(dY)':>7}")
    for r in val.itertuples(index=False):
        print(f"{r.arm:>22} {r.outcome:>8} {r.oracle_mean:8.2f} "
              f"{r.learned_mean:8.2f} {r.bias:+7.2f} {r.mae:7.2f} "
              f"{r.rmse:7.2f} {r.corr:6.3f} {r.r2:6.3f} {r.r2_effect:7.3f}")
    print("  R2 is on the OUTCOME Y_cf, R2(dY) on the EFFECT Y_cf - Y_fact. "
          "They differ most\n  where the effect is small next to var(Y) "
          "(do(xi_hw=0)): there only R2(dY) is evidence.")

    # treatment support in the TRAINING data — the counterfactual echo of the
    # Phase-6 positivity finding: no training unit is near T = 10, so the
    # fitted equations must extrapolate there and do(prb=10) degrades.
    print(f"\n  training-set treatment support: num_prb in "
          f"[{obs.num_prb.min():.0f}, {obs.num_prb.max():.0f}]; "
          f"n(|T-10|<=5) = {int(((obs.num_prb - 10).abs() <= 5).sum())}, "
          f"n(|T-50|<=5) = {int(((obs.num_prb - 50).abs() <= 5).sum())}, "
          f"n(|T-100|<=5) = {int(((obs.num_prb - 100).abs() <= 5).sum())}")
    print("  => do(prb=10) is OUT OF SUPPORT (extrapolation); "
          "do(prb=50/100) are in support.")

    # per-unit effect recovery (the ITE test), in- and out-of-support
    p100, t100, _ = preds["do(num_prb=100)"]
    p50, t50, _ = preds["do(num_prb=50)"]
    p10, t10, _ = preds["do(num_prb=10)"]
    for lab, (pp, tt) in {"Y(100)-Y(50)  [in support]": (p100 - p50, t100 - t50),
                          "Y(100)-Y(10)  [extrapolated]": (p100 - p10,
                                                           t100 - t10)}.items():
        print(f"  per-unit effect {lab}: corr = "
              f"{np.corrcoef(pp, tt)[0, 1]:.3f}, MAE "
              f"{np.mean(np.abs(pp - tt)):.2f} Mbps "
              f"(oracle mean {tt.mean():.2f}, sd {tt.std():.2f})")
    ite_pred, ite_true = p100 - p50, t100 - t50

    # latent recovery on the held-out units
    fp = latent_fingerprints(ores, ofact, kappa_hw)
    print(f"  abducted latents vs truth: xi_sf r="
          f"{np.corrcoef(fp['xi_sf_hat'], ofact.shadow_fading)[0, 1]:.3f}, "
          f"xi_tb n/a (not invertible from telemetry), "
          f"xi_hw r={np.corrcoef(fp['xi_hw_hat'], ofact.hw_impairment)[0, 1]:.3f}"
          f"  [xi_hw is what CF-2 now intervenes on]")

    # CF-2 calibration on the held-out units: is the effect recovered, and is
    # the sign reversal there? Stratifying on the TRUE latent is the honest
    # check — stratifying on xi_hw_hat selects partly on f_hat's own error.
    hw_arm = "do(hw_impairment=0)"
    if preds.get(hw_arm, (None,))[0] is not None:
        dl = preds[hw_arm][0] - ofact.goodput.values
        dt = preds[hw_arm][1] - ofact.goodput.values
        print(f"  CF-2 effect by TRUE xi_hw stratum (oracle | learned), Mbps:")
        so = stratify(ofact.hw_impairment.values, dt, HW_EDGES, HW_LABELS)
        sl = stratify(ofact.hw_impairment.values, dl, HW_EDGES, HW_LABELS)
        print("    " + "  ".join(
            f"{a.stratum}: {a.mean_effect:+.2f}|{b.mean_effect:+.2f}"
            for a, b in zip(so.itertuples(index=False),
                            sl.itertuples(index=False))))

    # ---------------- Part 2: CF-1/2/3 on the canonical 500 ----------------
    xmodel = StructuralModel(ROOT, mode="telemetry").fit_crossfit(obs, folds=5)
    xeng = CounterfactualEngine(xmodel, ROOT)
    fold = xmodel.fold_of
    res = xeng.abduct(obs, fold_of=fold)
    fps = latent_fingerprints(res, obs, kappa_hw)
    sf_hat = fps["xi_sf_hat"]
    hw_hat = fps["xi_hw_hat"]         # CF-2's stratifier (UP-29)

    stats_rows, strata_rows = [], []

    # CF-1: moderate-goodput UEs, do(prb=100), stratified by abducted xi_sf
    p1 = (obs.goodput >= 5) & (obs.goodput <= 40)
    cf1 = xeng.predict(obs[p1], {"num_prb": 100},
                       {k: v[p1.values] for k, v in res.items()},
                       fold_of=fold[p1.values])
    s = cf_statistics(obs.loc[p1, "goodput"].values, cf1.goodput.values, "CF-1")
    stats_rows.append(s)
    gain1 = cf1.goodput.values - obs.loc[p1, "goodput"].values
    st1 = stratify(sf_hat[p1.values], gain1,
                   [(-99, -7), (-7, -3), (-3, 3), (3, 7), (7, 99)],
                   ["xi_sf<-7", "-7..-3", "-3..3", "3..7", ">7"])
    st1["query"] = "CF-1"
    strata_rows.append(st1)

    # CF-2 (UP-29): "what would this UE achieve on nominal hardware?"
    # P_2 is OBSERVABLE — UEs delivering < 0.95x the median service rate of
    # peers in the same radio state — because §II requires subpopulations to
    # be defined by telemetry constraints, and because selecting on the
    # abducted xi_hw_hat would select partly on f_hat's estimation error and
    # inflate the measured gain by ~50%.
    p2 = peer_underperformers(obs)
    cf2 = xeng.predict(obs[p2], {"hw_impairment": 0.0},
                       {k: v[p2.values] for k, v in res.items()},
                       fold_of=fold[p2.values])
    s = cf_statistics(obs.loc[p2, "goodput"].values, cf2.goodput.values, "CF-2")
    stats_rows.append(s)
    gain2 = cf2.goodput.values - obs.loc[p2, "goodput"].values
    st2 = stratify(hw_hat[p2.values], gain2, HW_EDGES, HW_LABELS)
    st2["query"] = "CF-2"
    strata_rows.append(st2)
    print(f"\n  CF-2 P_2 (observable peer rule): n={int(p2.sum())} "
          f"({100 * p2.mean():.0f}% of the cell); RTT "
          f"{obs.loc[p2, 'rtt_ms'].mean():.2f} -> {cf2.rtt_ms.mean():.2f} ms "
          f"(the queue is near-idle at this load, so goodput is the live "
          f"outcome)")

    # the RETIRED CF-2 stays measurable as an oracle quantity even though it
    # is no longer predictable — record that it was retired, not forgotten
    old_p2 = (obs.rtt_ms >= 5) & (obs.rtt_ms <= 150)
    try:
        xeng.predict(obs[old_p2], {"traffic_burst": 0.5},
                     {k: v[old_p2.values] for k, v in res.items()},
                     fold_of=fold[old_p2.values])
        print("  WARNING: do(traffic_burst) unexpectedly succeeded")
    except ValueError:
        print(f"  retired CF-2 do(xi_tb=0.5) on RTT: still NOT realisable in "
              f"telemetry mode (UP-28); oracle effect on the held-out units "
              f"{preds['do(traffic_burst=0.5)'][1].mean() - ofact.rtt_ms.mean():+.3f} ms")

    # CF-3: underperformers at moderate SNR, do(prb=100)
    p3 = (obs.snr >= 10) & (obs.snr <= 20) & (obs.goodput < 15)
    cf3 = xeng.predict(obs[p3], {"num_prb": 100},
                       {k: v[p3.values] for k, v in res.items()},
                       fold_of=fold[p3.values])
    s = cf_statistics(obs.loc[p3, "goodput"].values, cf3.goodput.values, "CF-3")
    stats_rows.append(s)
    gain3 = cf3.goodput.values - obs.loc[p3, "goodput"].values
    st3 = stratify(sf_hat[p3.values], gain3,
                   [(-99, -5), (-5, 5), (5, 99)],
                   ["xi_sf<-5 (channel-limited)", "-5..5", ">+5 (PRB-limited)"])
    st3["query"] = "CF-3"
    strata_rows.append(st3)

    # persist per-unit arrays for the paper figures (E12)
    np.savez(os.path.join(ROOT, "results", "cf_arrays.npz"),
             cf1_fact=obs.loc[p1, "goodput"].values, cf1_cf=cf1.goodput.values,
             cf1_sf=sf_hat[p1.values],
             cf2_fact=obs.loc[p2, "goodput"].values, cf2_cf=cf2.goodput.values,
             cf2_hw=hw_hat[p2.values],
             cf3_fact=obs.loc[p3, "goodput"].values, cf3_cf=cf3.goodput.values,
             cf3_sf=sf_hat[p3.values],
             val_pred=preds["do(num_prb=100)"][0],
             val_true=preds["do(num_prb=100)"][1],
             ite_pred=ite_pred, ite_true=ite_true,
             sf_true=ofact.shadow_fading.values, sf_hat=fp["xi_sf_hat"])

    t3b = pd.DataFrame(stats_rows)
    t3b.to_csv(os.path.join(ROOT, "results", "table3_bottom.csv"), index=False)
    strata = pd.concat(strata_rows, ignore_index=True)
    strata.to_csv(os.path.join(ROOT, "results", "cf_strata.csv"), index=False)

    print("\nPART 2 — Table III bottom (canonical 500, cross-fitted abduction)")
    # [.] = the paper's printed value. CF-2 has NO paper counterpart any more:
    # it is a different query on a different outcome (UP-29), so comparing it
    # to the draft's RTT numbers would be meaningless.
    paper = {"CF-1": (129, 36.0, 0.50, 19.2), "CF-3": (104, 24.3, 0.20, 14.3)}
    for r in t3b.itertuples(index=False):
        unit = "Mbps"
        if r.query not in paper:
            print(f"  {r.query}: n={r.n:3d}       factual {r.factual_mean:6.2f} "
                  f"-> CF {r.cf_mean:6.2f} {unit}          "
                  f"KS {r.ks_stat:.2f} (p={r.ks_p:.1e})          "
                  f"W1 {r.wasserstein_1:5.2f}        "
                  f"[replaced query, no paper counterpart]")
            continue
        pn, pm, pks, pw = paper[r.query]
        print(f"  {r.query}: n={r.n:3d} [{pn}]  factual {r.factual_mean:6.2f} "
              f"-> CF {r.cf_mean:6.2f} {unit} [{pm}]   "
              f"KS {r.ks_stat:.2f} (p={r.ks_p:.1e}) [{pks}]   "
              f"W1 {r.wasserstein_1:5.2f} [{pw}]")

    print("\nper-stratum effects (strata from ABDUCTED latents):")
    for q, grp in strata.groupby("query"):
        print(f"  {q}:")
        for r in grp.itertuples(index=False):
            print(f"    {r.stratum:>26}: n={r.n:3d}  "
                  f"{r.mean_effect:+7.2f}")
        v = grp.dropna(subset=["mean_effect"])
        v = v[v.n >= 3]
        if len(v) >= 2 and v.mean_effect.min() > 0:
            print(f"    spread ratio (max/min): "
                  f"{v.mean_effect.max() / v.mean_effect.min():.1f}x")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.8))
    # (1a) learned vs oracle scatter
    p, t, _ = preds["do(num_prb=100)"]
    ax[0, 0].scatter(t, p, s=4, alpha=0.25, color="#2E86C1")
    lim = [0, max(t.max(), p.max())]
    ax[0, 0].plot(lim, lim, "k--", lw=0.9)
    r2 = val.loc[val.arm == "do(num_prb=100)", "r2"].iloc[0]
    ax[0, 0].set(xlabel="oracle goodput (Mbps)", ylabel="learned CF (Mbps)",
                 title=f"(a) unit-level CF vs oracle, do(prb=100)\nR²={r2:.3f}, "
                       f"N=5000 held-out units")
    # (1b) ITE recovery
    ax[0, 1].scatter(ite_true, ite_pred, s=4, alpha=0.25, color="#8E44AD")
    lim = [0, ite_true.max()]
    ax[0, 1].plot(lim, lim, "k--", lw=0.9)
    ax[0, 1].set(xlabel="oracle effect Y(100)-Y(50)",
                 ylabel="learned effect",
                 title=f"(b) per-unit treatment effects (in support)\n"
                       f"r={np.corrcoef(ite_pred, ite_true)[0,1]:.3f}")
    # (1c) latent recovery
    ax[0, 2].scatter(ofact.shadow_fading, fp["xi_sf_hat"], s=4, alpha=0.25,
                     color="#E67E22")
    ax[0, 2].set(xlabel="true ξ_sf (dB, latent)", ylabel="abducted ξ̂_sf",
                 title=f"(c) latent recovered by abduction\n"
                       f"ξ_sf r={np.corrcoef(fp['xi_sf_hat'], ofact.shadow_fading)[0,1]:.3f}, "
                       f"ξ_hw r={np.corrcoef(fp['xi_hw_hat'], ofact.hw_impairment)[0,1]:.3f} "
                       f"(CF-2)")
    # (2a-c) CF distributions
    panels = [("CF-1", obs.loc[p1, "goodput"].values, cf1.goodput.values, "Mbps"),
              ("CF-2", obs.loc[p2, "goodput"].values, cf2.goodput.values, "Mbps"),
              ("CF-3", obs.loc[p3, "goodput"].values, cf3.goodput.values,
               "Mbps")]
    for k, (q, fac, cfv, unit) in enumerate(panels):
        a = ax[1, k]
        row = t3b[t3b["query"] == q].iloc[0]
        if q == "CF-2":
            # CF-2 is a PAIRED within-unit contrast with a small mean shift on
            # a distribution that is 64% zeros, so overlaid marginals show two
            # indistinguishable curves under one spike at 0 and communicate
            # nothing. The per-unit effect is the honest display, and it makes
            # the real structure visible: hardware normalisation cannot help a
            # UE whose packets are already all lost to the link.
            d = cfv - fac
            live = fac > 1.0
            bins = np.linspace(0, np.percentile(d, 99), 30)
            a.hist(d[~live], bins=bins, alpha=.75, color="grey",
                   label=f"link-dead UEs, goodput<1 ({(~live).sum()})")
            a.hist(d[live], bins=bins, alpha=.75, color="#E67E22",
                   label=f"live UEs ({live.sum()}): mean {d[live].mean():+.2f}")
            a.set_yscale("log")
            a.set(xlabel=f"per-unit Δ {unit}", ylabel="count (log)",
                  title=f"({chr(100+k)}) {q} per-unit effect: mean "
                        f"{d.mean():+.2f}, median {np.median(d):+.2f} {unit}\n"
                        f"KS={row.ks_stat:.2f}, W₁={row.wasserstein_1:.1f}, "
                        f"n={row.n} (marginals overlap; see per-unit)")
            a.legend(fontsize=7.5)
            continue
        bins = np.linspace(min(fac.min(), cfv.min()),
                           max(np.percentile(fac, 99), np.percentile(cfv, 99)), 30)
        a.hist(fac, bins=bins, alpha=0.6, label="factual", color="grey")
        a.hist(cfv, bins=bins, alpha=0.6, label="counterfactual",
               color="#E67E22")
        a.set(xlabel=unit, ylabel="count",
              title=f"({chr(100+k)}) {q}: {row.factual_mean:.1f} → "
                    f"{row.cf_mean:.1f} {unit}\nKS={row.ks_stat:.2f}, "
                    f"W₁={row.wasserstein_1:.1f}, n={row.n}")
        a.legend(fontsize=8)
    fig.suptitle("Phase 7 — Rung 3: unit-level counterfactuals "
                 "(top: learned-vs-oracle validation; bottom: CF-1/2/3)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "step7_counterfactuals.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
