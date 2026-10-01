"""Contract tests for the high-N paired oracle (Phase-7 ground truth).

Guards the properties Phase 7 depends on: every counterfactual query has a
matching arm, arms are exactly paired on U, and interventions actually took.
Skips (rather than fails) if the artifact has not been built yet.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.ran_simulator import EXOGENOUS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ORACLE = os.path.join(ROOT, "data", "oracle_paired.csv.gz")
REQUIRED_ARMS = {"factual", "do(num_prb=10)", "do(num_prb=50)",
                 "do(num_prb=100)", "do(hw_impairment=0)",
                 "do(traffic_burst=0.5)"}


@pytest.fixture(scope="module")
def oracle():
    if not os.path.exists(ORACLE):
        pytest.skip("run experiments/build_oracle_validation.py first")
    return pd.read_csv(ORACLE)


def test_all_phase7_queries_have_an_arm(oracle):
    """CF-1/CF-3 need do(prb=100); CF-2 needs do(hw_impairment=0) (UP-29).
    The retired do(traffic_burst=0.5) arm is kept so the old CF-2 stays
    measurable as an oracle quantity even though it is no longer
    predictable in telemetry mode (UP-28)."""
    assert REQUIRED_ARMS <= set(oracle.arm.unique())


def test_arms_are_paired_on_exogenous_state(oracle):
    """Same unit id => same U in every arm, except the intervened variable.

    The exemption is DERIVED from each arm's name rather than hardcoded, so
    adding an exogenous do() arm cannot silently break this test (nor be
    silently exempted by editing a literal)."""
    arms = list(oracle.arm.unique())
    for col in EXOGENOUS:
        forced = [a for a in arms if a.startswith(f"do({col}=")]
        piv = oracle.pivot(index="unit", columns="arm", values=col)
        assert np.allclose(piv[forced].std(axis=0), 0), \
            f"{col} not constant within its own do() arm"
        assert np.allclose(piv.drop(columns=forced).std(axis=1), 0), col


def test_interventions_took_effect(oracle):
    for t in [10, 50, 100]:
        arm = oracle[oracle.arm == f"do(num_prb={t})"]
        assert (arm.num_prb == t).all()
    tb = oracle[oracle.arm == "do(traffic_burst=0.5)"]
    assert (tb.traffic_burst == 0.5).all()
    hw = oracle[oracle.arm == "do(hw_impairment=0)"]
    assert (hw.hw_impairment == 0.0).all()


def test_oracle_precision_beats_the_n30_file(oracle):
    """The whole point: SE must be far below the n=30 reproduction file."""
    g = oracle.loc[oracle.arm == "do(num_prb=100)", "goodput"]
    se = g.std() / np.sqrt(len(g))
    assert len(g) >= 5000 and se < 1.0

    small = pd.read_csv(os.path.join(ROOT, "data", "interventional.csv"))
    s = small.loc[small.regime == "do(num_prb=100)", "goodput"]
    assert se < s.std() / np.sqrt(len(s)) / 5


def test_per_unit_effects_are_exact_and_ordered(oracle):
    """Paired => per-unit effects computable; more PRBs never hurt a unit."""
    y = {t: oracle.loc[oracle.arm == f"do(num_prb={t})", "goodput"].values
         for t in [10, 50, 100]}
    assert (y[100] >= y[10] - 1e-9).all()
    assert (y[50] >= y[10] - 1e-9).all()
