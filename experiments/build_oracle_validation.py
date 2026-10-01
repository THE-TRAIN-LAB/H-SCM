"""High-N paired interventional oracle — the scientific ground truth.

The canonical interventional file (19 regimes x 30) reproduces the paper's
Table III; with std 33.6 at do(prb=100) its SE is ~6 Mbps, far too noisy to
validate a method against. This script builds a separate, much larger
PAIRED oracle: N fixed units (one exogenous draw U^(u) each), propagated
through the SAME U under every treatment arm.

Paired => per-unit treatment effects Y_t(u) - Y_t'(u) are exact (no
between-sample noise), which is what Phase 7's learned-vs-oracle
counterfactual test and Claims 3/4 require.

Roles (do not mix):
  data/interventional.csv   n=30 per regime  -> paper reproduction
  data/oracle_paired.csv.gz N units x arms   -> method validation

Usage: python experiments/build_oracle_validation.py [--n 5000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.ran_simulator import RANSimulator, EXOGENOUS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 20250721          # distinct from the canonical seeds (42, 1000+i)
# Arms must cover every Phase-7 counterfactual query:
#   CF-1 (goodput | do(prb=100)), CF-3 (root cause | do(prb=100)) -> prb arms
#   CF-2 (old, RTT | do(xi_tb=0.5))                               -> tb arm
#   CF-2 (new, goodput | do(xi_hw=0))                             -> hw arm
#
# The hw arm is the oracle for CF-2. xi_tb is not realisable in telemetry
# mode (no observable inverts it), but xi_hw is: it enters the graph at exactly
# one node, multiplicatively, as mac_tput = f(pa) * (1 + kappa_hw * xi_hw),
# so the abducted multiplicative residual of mac_tput IS the hardware state
# and do(xi_hw = 0) is realised by setting that residual to 1. No fake
# observable proxy is introduced. The tb arm is kept so the old CF-2 stays
# measurable as an oracle quantity even though it is not predictable.
ARMS = {"factual": None,
        "do(num_prb=10)": {"num_prb": 10},
        "do(num_prb=50)": {"num_prb": 50},
        "do(num_prb=100)": {"num_prb": 100},
        "do(traffic_burst=0.5)": {"traffic_burst": 0.5},
        "do(hw_impairment=0)": {"hw_impairment": 0.0}}
# outcome each arm is primarily used to validate
ARM_OUTCOME = {"do(traffic_burst=0.5)": "rtt_ms"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)
    suffix = ""

    sim = RANSimulator(cfg)
    rng = np.random.default_rng(SEED)
    u = sim.sample_exogenous(args.n, rng)             # ONE draw, shared by all arms

    frames = []
    for name, do in ARMS.items():
        df = sim.propagate(u, do=do)
        df.insert(0, "unit", np.arange(args.n))
        df.insert(1, "arm", name)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    out = os.path.join(ROOT, "data", f"oracle_paired{suffix}.csv.gz")
    full.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out}: {len(full)} rows = {args.n} units x {len(ARMS)} arms")

    # pairing check: exogenous columns identical across arms
    piv = full.pivot(index="unit", columns="arm", values="shadow_fading")
    assert np.allclose(piv.std(axis=1), 0), "arms are not paired!"
    print("pairing verified: identical U across arms")

    print(f"\noracle means (N={args.n}, SE in brackets) vs the n=30 "
          f"paper-reproduction file:")
    small = pd.read_csv(os.path.join(ROOT, "data", "interventional.csv"))
    for name in ARMS:
        col = ARM_OUTCOME.get(name, "goodput")
        g = full.loc[full.arm == name, col]
        se = g.std() / np.sqrt(len(g))
        line = (f"  {name:>21} [{col:>7}]: {g.mean():7.2f} +- {g.std():6.2f} "
                f"[SE {se:.3f}]")
        s = small.loc[small.regime == name, col]
        if len(s):
            line += f"   n=30: {s.mean():6.2f} [SE {s.std()/np.sqrt(30):.2f}]"
        elif name != "factual":
            line += "   (no n=30 counterpart — CF-2 arm is oracle-only)"
        print(line)

    # per-unit treatment effects (exact, paired)
    y = {n: full.loc[full.arm == n, "goodput"].values for n in ARMS}
    ite = y["do(num_prb=100)"] - y["do(num_prb=10)"]
    print(f"\npaired per-unit effect Y(100) - Y(10): mean {ite.mean():.2f}, "
          f"sd {ite.std():.2f}, min {ite.min():.2f}, max {ite.max():.2f}")
    sf = u["shadow_fading"]
    print(f"  channel-limited (xi_sf < -7 dB, n={(sf < -7).sum()}): "
          f"{ite[sf < -7].mean():.2f} Mbps")
    print(f"  PRB-limited     (xi_sf > +3 dB, n={(sf > 3).sum()}): "
          f"{ite[sf > 3].mean():.2f} Mbps")

    # CF-2 ground truth: per-unit RTT change under do(xi_tb = 0.5)
    rtt_f = full.loc[full.arm == "factual", "rtt_ms"].values
    rtt_cf = full.loc[full.arm == "do(traffic_burst=0.5)", "rtt_ms"].values
    d = rtt_f - rtt_cf
    p2 = (rtt_f >= 5) & (rtt_f <= 150)      # the paper's P_2 subpopulation
    print(f"\npaired per-unit RTT reduction under do(xi_tb=0.5): "
          f"mean {d.mean():.2f} ms, max {d.max():.2f} ms")
    print(f"  P_2 (factual RTT in [5,150] ms, n={p2.sum()} = "
          f"{100 * p2.mean():.1f}%): factual {rtt_f[p2].mean():.2f} -> "
          f"CF {rtt_cf[p2].mean():.2f} ms")
    tb = u["traffic_burst"]
    smooth, bursty = tb < 0.8, tb > 1.3
    print(f"  smooth UEs (xi_tb < 0.8, n={smooth.sum()}): "
          f"{d[smooth].mean():+.2f} ms")
    print(f"  bursty UEs (xi_tb > 1.3, n={bursty.sum()}): "
          f"{d[bursty].mean():+.2f} ms")

    # NEW CF-2 ground truth: per-unit goodput change under do(xi_hw = 0),
    # i.e. "normalise this device's RF quality to nominal". Note the SIGN
    # CONVENTION: xi_hw is a hardware-QUALITY offset, not an impairment
    # magnitude (se_eff = se_phy * (1 + kappa_hw * xi_hw)), so normalising a
    # good device to nominal makes it worse. The gain must therefore be
    # monotone DECREASING in xi_hw and cross zero at xi_hw = 0.
    gp_f = full.loc[full.arm == "factual", "goodput"].values
    gp_cf = full.loc[full.arm == "do(hw_impairment=0)", "goodput"].values
    dg = gp_cf - gp_f
    hw = u["hw_impairment"]
    print(f"\npaired per-unit goodput change under do(xi_hw=0): "
          f"mean {dg.mean():+.3f} Mbps, sd {dg.std():.3f}")
    for lo, hi, lab in [(-99, -1.5, "xi_hw < -1.5"),
                        (-1.5, -0.5, "-1.5..-0.5"),
                        (-0.5, 0.5, "-0.5..0.5"),
                        (0.5, 1.5, "0.5..1.5"),
                        (1.5, 99, "> 1.5")]:
        sel = (hw >= lo) & (hw < hi)
        print(f"  {lab:>14} (n={sel.sum():4d}): {dg[sel].mean():+7.3f} Mbps")


if __name__ == "__main__":
    main()
