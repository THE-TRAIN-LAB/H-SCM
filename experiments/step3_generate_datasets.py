"""E3 + E4 — Phase 3: generate the canonical datasets and checkpoint tables.

Outputs
-------
data/observational.csv    500 rows x 26 cols   (seed 42)
data/interventional.csv   570 rows x 26 cols   (19 regimes x 30, seeds 1000+i)
data/hscm_dataset.csv     1070 rows x 26 cols  (concatenation, the paper's shape)
results/table2_prb_buckets.csv        Table II (observational PRB-bucket stats)
results/table_rung2_oracle.csv        interventional oracle means (Rung 2 inputs)
results/crosstab_load_prb.csv         load-quartile x PRB-quartile counts
results/figures/step3_crosstab.png    anti-diagonal confounding signature

Checkpoint quantities printed: dataset shape, r(L, prb), Table II,
high-load->low-PRB / low-load->high-PRB percentages (paper: 99.2% / 78.4%),
oracle E[Y | do(prb)] +- std at n=30.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.ran_simulator import RANSimulator, COLUMNS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED_OBS = 42
SEED_REGIME_BASE = 1000   # regime i uses seed 1000 + i


def generate(cfg):
    sim = RANSimulator(cfg)
    obs = sim.sample(int(cfg["observational"]["n_samples"]), seed=SEED_OBS,
                     regime="obs")
    frames = []
    for i, regime in enumerate(cfg["interventional"]["regimes"]):
        frames.append(sim.sample(int(cfg["interventional"]["samples_per_regime"]),
                                 do=regime["do"], seed=SEED_REGIME_BASE + i))
    intv = pd.concat(frames, ignore_index=True)
    return obs, intv


def table2(obs):
    edges = [0, 50, 60, 70, 80, 101]
    names = ["<50", "50-59", "60-69", "70-79", "80-100"]
    rows = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        s = obs[(obs.num_prb >= lo) & (obs.num_prb < hi)]
        rows.append({"prb_range": nm, "n": len(s),
                     "mean_goodput_mbps": round(s.goodput.mean(), 1),
                     "mean_load": round(s.cell_load.mean(), 2),
                     "mean_sinr_db": round(s.sinr_eff.mean(), 1)})
    return pd.DataFrame(rows)


def crosstab(obs):
    prb_q = pd.qcut(obs.num_prb, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    load_q = pd.qcut(obs.cell_load, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    ct = pd.crosstab(load_q, prb_q, rownames=["load"], colnames=["prb"])
    hi_load = obs[obs.cell_load > 0.75]
    lo_load = obs[obs.cell_load < 0.25]
    prb_q1_edge = obs.num_prb.quantile(0.25)
    prb_q4_edge = obs.num_prb.quantile(0.75)
    pct_hi = 100 * (hi_load.num_prb <= prb_q1_edge).mean()
    pct_lo = 100 * (lo_load.num_prb >= prb_q4_edge).mean()
    return ct, pct_hi, pct_lo, len(hi_load), len(lo_load)


def main():
    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)

    obs, intv = generate(cfg)
    full = pd.concat([obs, intv], ignore_index=True)
    assert list(full.columns) == COLUMNS + ["regime"]
    W = len(COLUMNS) + 1                      # nodes + regime label
    assert full.shape == (1070, W), full.shape
    assert obs.shape == (500, W) and intv.shape == (570, W)
    assert intv.regime.nunique() == len(cfg["interventional"]["regimes"]) == 19
    assert not full.isna().any().any()

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    obs.to_csv(os.path.join(ROOT, "data", "observational.csv"), index=False)
    intv.to_csv(os.path.join(ROOT, "data", "interventional.csv"), index=False)
    full.to_csv(os.path.join(ROOT, "data", "hscm_dataset.csv"), index=False)
    print(f"dataset: {full.shape[0]} rows x {full.shape[1]} cols "
          f"({len(obs)} obs + {len(intv)} intv, {intv.regime.nunique()} regimes)")

    r = np.corrcoef(obs.cell_load, obs.num_prb)[0, 1]
    print(f"\nr(cell_load, num_prb) on the 500 obs rows: {r:+.3f}")

    t2 = table2(obs)
    t2.to_csv(os.path.join(ROOT, "results", "table2_prb_buckets.csv"), index=False)
    print("\nTable II — observational PRB buckets (paper values in brackets):")
    paper = [(19, 2.3, 0.91, -1.9), (65, 12.1, 0.77, 6.9), (114, 16.0, 0.62, 8.2),
             (130, 22.1, 0.45, 10.2), (172, 28.0, 0.25, 10.7)]
    for row, p in zip(t2.itertuples(index=False), paper):
        print(f"  {row.prb_range:>7}: n={row.n:3d} [{p[0]:3d}]  "
              f"goodput {row.mean_goodput_mbps:5.1f} [{p[1]:4.1f}] Mbps  "
              f"load {row.mean_load:.2f} [{p[2]:.2f}]  "
              f"SINR {row.mean_sinr_db:5.1f} [{p[3]:5.1f}] dB")

    ct, pct_hi, pct_lo, n_hi, n_lo = crosstab(obs)
    ct.to_csv(os.path.join(ROOT, "results", "crosstab_load_prb.csv"))
    print(f"\nload x PRB quartile cross-tab:\n{ct}")
    print(f"\nhigh-load (L>0.75, n={n_hi}) in lowest PRB quartile:  "
          f"{pct_hi:5.1f}%   [paper: 99.2%]")
    print(f"low-load  (L<0.25, n={n_lo}) in highest PRB quartile: "
          f"{pct_lo:5.1f}%   [paper: 78.4%]")

    # Rung-2 oracle (marginal do(prb) regimes, n=30 each)
    rows = []
    for t in [10, 50, 100]:
        s = intv[intv.regime == f"do(num_prb={t})"]
        assert len(s) == 30
        rows.append({"treatment": t, "n": len(s),
                     "mean_goodput": round(s.goodput.mean(), 1),
                     "std_goodput": round(s.goodput.std(), 1)})
    oracle = pd.DataFrame(rows)
    oracle.to_csv(os.path.join(ROOT, "results", "table_rung2_oracle.csv"),
                  index=False)
    print("\nRung-2 oracle, E[Y | do(prb=t)] (n=30 each; paper: 1.8/13.4/24.7,"
          " 24.7 +- 30.0 at t=100):")
    for row in oracle.itertuples(index=False):
        print(f"  do(prb={row.treatment:3d}): {row.mean_goodput:5.1f} "
              f"+- {row.std_goodput:4.1f} Mbps")

    # checkpoint figure: anti-diagonal signature
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].imshow(ct.values, cmap="Blues")
    ax[0].set(xticks=range(4), yticks=range(4),
              xticklabels=ct.columns, yticklabels=ct.index,
              xlabel="num_prb quartile", ylabel="cell_load quartile",
              title="(i) load x PRB counts (anti-diagonal)")
    for i in range(4):
        for j in range(4):
            ax[0].text(j, i, ct.values[i, j], ha="center", va="center",
                       color="white" if ct.values[i, j] > ct.values.max() / 2
                       else "black")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].scatter(obs.cell_load, obs.num_prb, s=8, alpha=0.5)
    for q in obs.num_prb.quantile([0.25, 0.5, 0.75]):
        ax[1].axhline(q, color="grey", lw=0.6, ls="--")
    for q in obs.cell_load.quantile([0.25, 0.5, 0.75]):
        ax[1].axvline(q, color="grey", lw=0.6, ls="--")
    ax[1].set(xlabel="cell load L", ylabel="num_prb",
              title=f"(ii) confounding scatter, r={r:.2f}")
    fig.suptitle("Phase 3 checkpoint: observational confounding structure (n=500)")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "step3_crosstab.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
