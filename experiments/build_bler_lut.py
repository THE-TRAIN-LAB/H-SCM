"""Build the cached Sionna BLER lookup table (data/bler_lut.npz).

One-time artifact: for each of the 12 MCS entries, sweeps the Sionna link
(LDPC5G k=512, OFDM 76sc/15kHz/14sym, CDL-C, batch 32) over the MCS's
waterfall window (analytical threshold +-10 dB, 0.5 dB steps, 4 batches =
128 blocks per point) and stores monotone BLER curves. Config-fingerprinted:
refuses to load under a changed PHY/MCS config.
"""
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.phy import SionnaPHY  # noqa: E402


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)
    phy = SionnaPHY(cfg)
    if phy._lut is not None:
        print("LUT already present and fingerprint matches — nothing to do.")
        return
    t0 = time.time()
    phy.build_lut(n_batches=4)
    thr = phy.selection_thresholds()
    print(f"\nBuilt {phy.lut_path} in {time.time() - t0:.0f} s")
    print("MCS selection thresholds (SINR dB for BLER<=0.1):")
    for i, ((m, r), t) in enumerate(zip(phy.entries, thr)):
        print(f"  mcs={i:2d}  m={m}  r={r:.2f}  thr={t:+6.2f} dB")


if __name__ == "__main__":
    main()
