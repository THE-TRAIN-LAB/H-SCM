"""H-SCM simulator tests (analytical backend: fast, no TF/Sionna needed)."""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.phy import AnalyticalPHY                       # noqa: E402
ROOT = os.path.join(os.path.dirname(__file__), "..")
from simulator.ran_simulator import (RANSimulator, COLUMNS,   # noqa: E402
                                     EXOGENOUS, ENDOGENOUS)


@pytest.fixture(scope="module")
def cfg():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config", "simulator.yaml")) as f:
        c = yaml.safe_load(f)
    c["phy"]["backend"] = "analytical"
    return c


@pytest.fixture(scope="module")
def sim(cfg):
    return RANSimulator(cfg, backend=AnalyticalPHY(cfg))


def test_columns_and_shape(sim):
    df = sim.sample(100, seed=1)
    assert list(df.columns) == COLUMNS + ["regime"]
    assert len(df) == 100 and df["regime"].eq("obs").all()
    # The exogenous set U has exactly 10 members (paper §V) -- that count is a
    # CLAIM ABOUT THE PAPER and stays pinned. The endogenous count tracks the
    # equations and is pinned to the config so the two cannot drift apart.
    assert len(EXOGENOUS) == 10
    with open(os.path.join(ROOT, "config", "variables.yaml")) as f:
        declared = yaml.safe_load(f)
    assert len(ENDOGENOUS) == len(declared["endogenous"])
    assert set(ENDOGENOUS) == set(declared["endogenous"])
    assert set(EXOGENOUS) == set(declared["exogenous"])


def test_determinism(sim):
    a = sim.sample(50, seed=7)
    b = sim.sample(50, seed=7)
    assert np.allclose(a[COLUMNS].values, b[COLUMNS].values)


def test_physical_ranges(sim):
    df = sim.sample(2000, seed=2)
    assert df["num_prb"].between(1, 100).all()
    assert df["bler"].between(0, 1).all()
    assert df["pkt_loss"].between(0, 1).all()
    assert df["harq_retx"].between(0, 3).all()
    assert (df["goodput"] >= 0).all() and (df["goodput"] <= df["mac_tput"]).all()
    assert (df["rtt_ms"] > 1).all()
    assert (df["queue_dep"] >= 0).all()
    # C0 calibrated to Table II bucket means puts the extreme tail (top MCS,
    # full PRBs, favourable xi_hw) near ~110 Mbps for a few % of units
    assert df["goodput"].max() < 150
    assert df["mcs_idx"].between(0, 11).all()


def test_confounding_correlation(sim):
    """Load-bearing invariant: r(L, num_prb) ~ -0.97 observationally."""
    df = sim.sample(5000, seed=3)
    r = np.corrcoef(df["cell_load"], df["num_prb"])[0, 1]
    assert -0.985 < r < -0.955, r


def test_do_prb_severs_L_to_prb_but_keeps_L_to_sinr(sim):
    """Graph mutilation: do(prb) removes L->prb, preserves L->sinr_eff."""
    df = sim.sample(3000, do={"num_prb": 50}, seed=4)
    assert (df["num_prb"] == 50).all()
    r_load_sinr = np.corrcoef(df["cell_load"], df["sinr_eff"])[0, 1]
    assert r_load_sinr < -0.3   # interference environment stays active


def test_do_exogenous(sim):
    df = sim.sample(200, do={"cell_load": 0.9, "snr": 20.0}, seed=5)
    assert (df["cell_load"] == 0.9).all() and (df["snr"] == 20.0).all()
    # downstream responds: high load -> few PRBs
    assert df["num_prb"].mean() < 55


def test_effect_modifiers_do_not_touch_num_prb(sim):
    """xi_sf, xi_hw, xi_tb must have no edge into num_prb."""
    rng = np.random.default_rng(6)
    u = sim.sample_exogenous(500, rng)
    base = sim.propagate(u)
    for name in ["shadow_fading", "hw_impairment", "traffic_burst"]:
        u2 = {k: v.copy() for k, v in u.items()}
        u2[name] = u2[name] * 0.0 + (0.5 if name == "traffic_burst" else 0.0)
        alt = sim.propagate(u2)
        assert np.array_equal(base["num_prb"].values, alt["num_prb"].values), name


def test_consistency_axiom(sim):
    """T = t observed  =>  Y_t = Y (potential outcome equals factual)."""
    rng = np.random.default_rng(8)
    u = sim.sample_exogenous(300, rng)
    fact = sim.propagate(u)
    for i in [0, 17, 123]:
        u_i = {k: v[i:i + 1] for k, v in u.items()}
        cf = sim.propagate(u_i, do={"num_prb": float(fact["num_prb"].iloc[i])})
        assert np.isclose(cf["goodput"].iloc[0], fact["goodput"].iloc[i])
        assert np.isclose(cf["rtt_ms"].iloc[0], fact["rtt_ms"].iloc[i])


def test_fixed_unit_counterfactual_oracle(sim):
    """Same unit (fixed U), different treatments -> only T-descendants change,
    and more PRBs never hurt goodput for the same unit."""
    rng = np.random.default_rng(9)
    u = sim.sample_exogenous(400, rng)
    y10 = sim.propagate(u, do={"num_prb": 10})
    y100 = sim.propagate(u, do={"num_prb": 100})
    # non-descendants of num_prb are identical
    for col in ["sinr_eff", "mcs_idx", "bler", "se_phy", "harq_retx",
                ]:
        assert np.array_equal(y10[col].values, y100[col].values), col
    # monotone unit-level treatment response
    assert (y100["goodput"].values >= y10["goodput"].values - 1e-9).all()


def test_unit_heterogeneity_from_shadow_fading(sim):
    """Effect modification: same measured conditions, different xi_sf ->
    very different counterfactual gains (the paper's Rung-3 phenomenon)."""
    rng = np.random.default_rng(10)
    u = sim.sample_exogenous(2000, rng)
    gain = (sim.propagate(u, do={"num_prb": 100})["goodput"].values
            - sim.propagate(u, do={"num_prb": 10})["goodput"].values)
    lo = gain[u["shadow_fading"] < -7].mean()
    hi = gain[u["shadow_fading"] > 3].mean()
    assert hi > 4 * lo   # PRB-limited units gain far more


def test_regime_labels(sim):
    df = sim.sample(10, do={"num_prb": 100, "cell_load": 0.5}, seed=11)
    assert df["regime"].iloc[0] == "do(cell_load=0.5,num_prb=100)"


def test_unknown_do_target_raises(sim):
    """Misspelled intervention names must fail loudly, not silently no-op."""
    with pytest.raises(ValueError, match="num_prbb"):
        sim.sample(5, do={"num_prbb": 100}, seed=12)
    with pytest.raises(ValueError):
        sim.propagate(sim.sample_exogenous(5, np.random.default_rng(0)),
                      do={"goodputt": 1.0})


def test_service_limit_applied_once(cfg):
    """The delivered rate must carry the service limit ONCE (UP-34).

    pkt_loss used to include a served fraction min(1, mac_tput/lambda) while
    goodput is mac_tput * (1 - pkt_loss), so the limit was applied twice and
    the composed rate became mac_tput^2/lambda under overload. At
    mac_tput = lambda/2 with no other loss that returns lambda/4, half the
    correct value.
    """
    from simulator import network
    arr = float(cfg["network"]["arrival_mbps"])
    lossless = dict(bler=np.zeros(3), harq_retx=np.zeros(3),
                    queue_dep=np.zeros(3))
    p = network.packet_loss(**lossless, cfg=cfg)
    assert np.allclose(p, 0.0)
    for mu in (arr / 2, arr, 4 * arr):
        y = network.goodput(mu, network.packet_loss(
            bler=0.0, harq_retx=0.0, queue_dep=0.0, cfg=cfg))
        assert np.isclose(y, mu), f"mac_tput={mu} -> {y}, expected {mu}"


def test_pkt_loss_does_not_depend_on_mac_tput(cfg):
    """mac_tput is no longer a parent of pkt_loss (UP-34), so the loss
    probability must be identical across service rates spanning deep overload
    to heavy underload."""
    from simulator import network
    arr = float(cfg["network"]["arrival_mbps"])
    common = dict(bler=np.full(4, 0.1), harq_retx=np.full(4, 0.11),
                  queue_dep=np.full(4, 10.0))
    l = network.packet_loss(**common, cfg=cfg)
    assert np.allclose(l, l[0])
    import inspect
    assert "mac_tput" not in inspect.signature(network.packet_loss).parameters


def test_queue_overflow_still_graduated(cfg):
    """Removing the served fraction must not remove the queue-overflow term:
    a deeper backlog still costs strictly more loss."""
    from simulator import network
    l = network.packet_loss(bler=np.full(3, 0.1), harq_retx=np.full(3, 0.11),
                            queue_dep=np.array([0.0, 10.0, 60.0]), cfg=cfg)
    assert l[0] < l[1] < l[2]


def test_rtt_tail_bounded(sim):
    """The MGBR floor (UP-24) keeps the outage tail at a few hundred ms —
    no artificial multi-second cluster."""
    df = sim.sample(5000, seed=13)
    assert df["rtt_ms"].max() < 600
    # and the congested population still exists for CF-2's P2 window
    frac_p2 = df["rtt_ms"].between(5, 150).mean()
    assert 0.05 < frac_p2 < 0.45
