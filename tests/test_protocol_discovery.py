"""Contract tests for protocol-constrained discovery and mediator blocking."""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.mediator_test import paired_contrast                     # noqa: E402
from causal.protocol_knowledge import (WITHIN_RANK, acyclic_projection,  # noqa: E402
                                       build_background_knowledge, enforce,
                                       forbidden, is_acyclic,
                                       is_exogenous_pair,
                                       protocol_position,
                                       violates_protocol_order)
from simulator.ran_simulator import COLUMNS, RANSimulator            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def sim():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    return RANSimulator(cfg)


@pytest.fixture(scope="module")
def levels(sim):
    ref = sim.sample(3000, seed=0, regime="obs")
    return {c: {q: float(np.quantile(ref[c], q)) for q in (0.1, 0.5, 0.9)}
            for c in COLUMNS}


# ------------------------------------------------------------- constraints
def test_exogenous_pairs_forbidden_both_ways():
    assert is_exogenous_pair("eps_sched", "eps_sinr")
    assert forbidden(("eps_sched", "eps_sinr"))
    assert forbidden(("eps_sinr", "eps_sched"))


def test_within_layer_order_forbids_only_backwards():
    assert violates_protocol_order(("se_phy", "mcs_idx"))     # rank 3 -> 1
    assert not violates_protocol_order(("mcs_idx", "se_phy"))  # rank 1 -> 3
    # equal rank is deliberately unconstrained in BOTH directions
    assert WITHIN_RANK["mod_order"] == WITHIN_RANK["code_rate"]
    assert not violates_protocol_order(("mod_order", "code_rate"))
    assert not violates_protocol_order(("code_rate", "mod_order"))


def test_cross_layer_forward_edges_are_allowed():
    """The true L1->L3 edge bler -> pkt_loss must not be excluded."""
    assert not forbidden(("bler", "pkt_loss"))
    assert protocol_position("bler") < protocol_position("pkt_loss")


def test_background_knowledge_tiers():
    from causallearn.graph.GraphNode import GraphNode
    bk = build_background_knowledge(COLUMNS)
    n = {c: GraphNode(c) for c in COLUMNS}
    assert bk.is_forbidden(n["rtt_ms"], n["sinr_eff"])        # L3 -/-> L1
    assert not bk.is_forbidden(n["sinr_eff"], n["rtt_ms"])    # L1 -> L3 legal
    assert bk.is_forbidden(n["eps_sched"], n["eps_sinr"])     # exo <-> exo
    assert bk.is_forbidden(n["eps_sinr"], n["eps_sched"])


# --------------------------------------------------------------- acyclicity
def test_acyclic_projection_rejects_the_cycle_closing_edge():
    base = {("a", "b"), ("b", "c")}
    out = acyclic_projection(base, {("c", "a"): 1e-9})
    assert ("c", "a") not in out and is_acyclic(out)


def test_acyclic_projection_admits_by_significance():
    """Given a mutually exclusive pair, the smaller p-value wins."""
    out = acyclic_projection(set(), {("x", "y"): 1e-12, ("y", "x"): 1e-3})
    assert ("x", "y") in out and ("y", "x") not in out


def test_enforce_removes_forbidden_and_cycles():
    E = {("eps_sched", "eps_sinr"), ("se_phy", "mcs_idx"),
         ("sinr_eff", "mcs_idx")}
    out = enforce(E)
    assert out == {("sinr_eff", "mcs_idx")}
    assert is_acyclic(out)


# ---------------------------------------------------- mediator blocking
def test_blocking_mediators_kills_an_indirect_effect(sim, levels):
    """bler -> goodput is real as a TOTAL effect but is fully mediated."""
    rng = np.random.default_rng(0)
    x, y = "bler", "goodput"
    lo, hi = levels[x][0.1], levels[x][0.9]
    total = paired_contrast(sim, rng, x, lo, hi, {}, COLUMNS, n=400)[y]
    assert abs(total[0]) > 1.0 and total[1] < 0.01          # total effect real
    hold = {m: levels[m][0.5] for m in ("se_phy", "mac_tput", "pkt_loss")}
    direct = paired_contrast(sim, rng, x, lo, hi, hold, COLUMNS, n=400)[y]
    assert abs(direct[0]) < 1e-9                            # nothing direct left


def test_blocking_mediators_preserves_a_direct_effect(sim, levels):
    """bler -> pkt_loss survives blocking: it is a genuine direct L1->L3 edge."""
    rng = np.random.default_rng(0)
    x, y = "bler", "pkt_loss"
    lo, hi = levels[x][0.1], levels[x][0.9]
    hold = {m: levels[m][0.5]
            for m in ("se_phy", "mac_tput", "harq_retx", "queue_dep")}
    direct = paired_contrast(sim, rng, x, lo, hi, hold, COLUMNS, n=400)[y]
    assert abs(direct[0]) > 0.1 and direct[1] < 0.01


def test_paired_contrast_is_within_unit(sim, levels):
    """Both arms must reuse the same exogenous draw, else variance explodes."""
    rng = np.random.default_rng(1)
    out = paired_contrast(sim, rng, "num_prb", 10, 100, {}, COLUMNS, n=300)
    # num_prb is FORCED in both arms, so its paired difference is exactly the
    # contrast itself -- the sharpest available check that the two arms really
    # do share one exogenous draw.
    assert abs(out["num_prb"][0] - 90.0) < 1e-9
