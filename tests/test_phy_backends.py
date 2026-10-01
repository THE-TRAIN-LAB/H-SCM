"""PHYBackend interface tests (analytical backend: fast, no TF needed)."""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.phy import AnalyticalPHY, make_backend, sinr_eff_db  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config", "simulator.yaml")) as f:
        return yaml.safe_load(f)


def test_analytical_bler_monotone_in_sinr(cfg):
    phy = AnalyticalPHY(cfg)
    for mcs in range(len(phy.entries)):
        blers = [phy.bler(s, mcs) for s in np.arange(-10, 35, 1.0)]
        assert all(b1 >= b2 for b1, b2 in zip(blers, blers[1:]))


def test_analytical_bler_monotone_in_mcs(cfg):
    """At fixed SINR, higher MCS (more aggressive) has higher or equal BLER
    whenever spectral efficiency increases."""
    phy = AnalyticalPHY(cfg)
    se = [m * r for m, r in phy.entries]
    for s in [0.0, 10.0, 20.0]:
        for i in range(len(se) - 1):
            if se[i + 1] > se[i]:
                assert phy.bler(s, i + 1) >= phy.bler(s, i) - 1e-12


def test_mcs_selection_respects_target(cfg):
    phy = AnalyticalPHY(cfg)
    for s in np.arange(-5, 32, 1.0):
        idx = phy.select_mcs(s)
        if idx > 0:
            assert phy.bler(s, idx) <= phy.target_bler + 1e-12


def test_mcs_selection_monotone_in_sinr(cfg):
    phy = AnalyticalPHY(cfg)
    idxs = [phy.select_mcs(s) for s in np.arange(-5, 32, 0.5)]
    assert all(i1 <= i2 for i1, i2 in zip(idxs, idxs[1:]))
    assert idxs[0] == 0 and idxs[-1] == len(phy.entries) - 1


def test_sinr_eff_eq7_terms(cfg):
    # more load, doppler, delay -> lower sinr; shadow fading adds directly
    base = sinr_eff_db(15, 0, 0.2, 50, 100, 0, cfg)
    assert sinr_eff_db(15, 0, 0.8, 50, 100, 0, cfg) < base
    assert sinr_eff_db(15, 3, 0.2, 50, 100, 0, cfg) == pytest.approx(base + 3)
    assert sinr_eff_db(15, 0, 0.2, 150, 100, 0, cfg) < base
    assert sinr_eff_db(15, 0, 0.2, 50, 250, 0, cfg) < base


def test_factory(cfg):
    assert isinstance(make_backend({**cfg, "phy": {**cfg["phy"], "backend": "analytical"}}),
                      AnalyticalPHY)
    with pytest.raises(ValueError):
        make_backend({**cfg, "phy": {**cfg["phy"], "backend": "nope"}})
