"""Algorithm 1 tests: axioms, noise persistence, latent interventions."""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.counterfactual import (CounterfactualEngine,      # noqa: E402
                                   peer_underperformers)
from causal.fitting import StructuralModel                   # noqa: E402
from simulator.phy import AnalyticalPHY                      # noqa: E402
from simulator.ran_simulator import RANSimulator             # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def setup():
    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["phy"]["backend"] = "analytical"
    sim = RANSimulator(cfg, backend=AnalyticalPHY(cfg))
    obs = sim.sample(400, seed=11)
    model = StructuralModel(ROOT, mode="telemetry").fit(obs, cv_folds=3)
    return sim, obs, CounterfactualEngine(model, ROOT)


def test_consistency_axiom(setup):
    """do(T = observed t) must return the factual outcome (Y_t = Y)."""
    _, obs, eng = setup
    res = eng.abduct(obs)
    sub = obs.iloc[:50]
    r = {k: v[:50] for k, v in res.items()}
    for i in range(0, 50, 10):
        one = sub.iloc[[i]]
        cf = eng.predict(one, {"num_prb": float(one.num_prb.iloc[0])},
                         {k: v[i:i + 1] for k, v in r.items()})
        assert np.isclose(cf.goodput.iloc[0], one.goodput.iloc[0], rtol=1e-6)


def test_noise_persistence_non_descendants_unchanged(setup):
    """Non-descendants of T keep their factual values."""
    _, obs, eng = setup
    cf = eng.query(obs, {"num_prb": 100})
    for col in ["sinr_eff", "mcs_idx", "bler", "se_phy", "harq_retx",
                "cell_load", "snr"]:
        assert np.allclose(cf[col].values, obs[col].values), col


def test_intervention_applied_and_descendants_change(setup):
    _, obs, eng = setup
    cf = eng.query(obs, {"num_prb": 100})
    assert (cf.num_prb == 100).all()
    assert not np.allclose(cf.mac_tput.values, obs.mac_tput.values)
    assert not np.allclose(cf.goodput.values, obs.goodput.values)


def test_latent_without_realisation_rejected(setup):
    """do() on a latent with neither a proxy nor a residual realisation must
    raise, not silently no-op: no telemetry model reads shadow_fading, so
    'succeeding' would return the factual world unchanged.

    hw_impairment is deliberately NOT in this list any more — it has a
    residual realisation (UP-29) and is the new CF-2."""
    _, obs, eng = setup
    for bad in ["shadow_fading", "traffic_burst", "eps_sinr", "nonsense"]:
        with pytest.raises(ValueError):
            eng.query(obs.iloc[:5], {bad: 0.0})


def test_latent_without_proxy_is_rejected():
    """No observable inverts a latent, so LATENT_PROXY must stay empty:
    do(traffic_burst) has no realisation in telemetry mode and must raise
    rather than silently return the factual world."""
    import causal.counterfactual as C
    assert C.LATENT_PROXY == {}


def test_counterfactual_tracks_oracle_in_support(setup):
    """Learned CF vs simulator oracle on the SAME units, in-support arm."""
    sim, obs, eng = setup
    rng = np.random.default_rng(5)
    u = sim.sample_exogenous(400, rng)
    fact = sim.propagate(u)
    truth = sim.propagate(u, do={"num_prb": 80}).goodput.values
    pred = eng.query(fact, {"num_prb": 80}).goodput.values
    r2 = 1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
    assert r2 > 0.9, r2


# --------------------------------------------------------------------------
# CF-2 (UP-29): do(xi_hw = 0) on mac_tput's abducted multiplicative residual
# --------------------------------------------------------------------------
def test_hw_residual_intervention_is_consistent(setup):
    """Consistency again, but for a RESIDUAL action: do(xi_hw = xi_hw_hat(u))
    must return the factual world, because it substitutes the residual the
    unit already had. This is the check that the residual path uses the same
    equations as abduction and has not silently drifted."""
    import causal.counterfactual as C
    _, obs, eng = setup
    kappa = float(eng.cfg["mac"]["kappa_hw"])
    res = eng.abduct(obs)
    hw_hat = (res["mac_tput"] - 1.0) / kappa
    cf = obs.copy()
    # per-unit target, applied one unit at a time (do() takes a scalar)
    for i in (0, 7, 23, 99):
        one = obs.iloc[[i]]
        r1 = {k: v[i:i + 1] for k, v in res.items()}
        out = eng.predict(one, {"hw_impairment": float(hw_hat[i])}, r1)
        assert np.isclose(out.mac_tput.iloc[0], one.mac_tput.iloc[0], rtol=1e-6)
        assert np.isclose(out.goodput.iloc[0], one.goodput.iloc[0], rtol=1e-6)


def test_hw_intervention_leaves_non_descendants_alone(setup):
    """The action is on mac_tput's noise, so nothing upstream may move —
    including mac_tput's own parents. A regression here would mean the
    residual action is being applied as a value intervention."""
    _, obs, eng = setup
    cf = eng.query(obs, {"hw_impairment": 0.0})
    for col in ["sinr_eff", "mcs_idx", "bler", "se_phy", "harq_retx",
                "num_prb", "cell_load", "snr"]:
        assert np.allclose(cf[col].values, obs[col].values), col
    assert not np.allclose(cf.mac_tput.values, obs.mac_tput.values)
    assert not np.allclose(cf.goodput.values, obs.goodput.values)


def test_hw_intervention_sign_reverses(setup):
    """xi_hw is a hardware-QUALITY offset, not an impairment magnitude, so
    normalising must HELP below-nominal devices and HURT above-nominal ones.
    No constant offset or scaling bug can produce a sign reversal."""
    _, obs, eng = setup
    kappa = float(eng.cfg["mac"]["kappa_hw"])
    res = eng.abduct(obs)
    hw_hat = (res["mac_tput"] - 1.0) / kappa
    gain = eng.predict(obs, {"hw_impairment": 0.0}, res).goodput.values \
        - obs.goodput.values
    assert gain[hw_hat < -1.0].mean() > 0.1, gain[hw_hat < -1.0].mean()
    assert gain[hw_hat > 1.0].mean() < -0.1, gain[hw_hat > 1.0].mean()
    assert np.corrcoef(hw_hat, gain)[0, 1] < -0.3


def test_hw_counterfactual_tracks_oracle(setup):
    """Learned CF vs the simulator's own do(hw_impairment=0) arm, same units.
    Scored on the EFFECT, not the outcome: var(Y) >> var(dY) here, so an
    outcome-level R2 would pass even for a model that predicted no change."""
    sim, obs, eng = setup
    u = sim.sample_exogenous(400, np.random.default_rng(7))
    fact = sim.propagate(u)
    truth = sim.propagate(u, do={"hw_impairment": 0.0}).goodput.values
    pred = eng.query(fact, {"hw_impairment": 0.0}).goodput.values
    dt, dp = truth - fact.goodput.values, pred - fact.goodput.values
    assert abs(dp.mean() - dt.mean()) < 0.25              # unbiased on average
    assert np.corrcoef(dp, dt)[0, 1] > 0.6                # and per unit
    # the do-nothing baseline must be beaten outright
    assert np.sqrt(np.mean((dp - dt) ** 2)) < np.sqrt(np.mean(dt ** 2))


def test_peer_subpopulation_is_observable_and_selects_bad_hardware(setup):
    """P_2 must be computable from telemetry alone AND must actually select
    device-limited UEs -- a rule that just picks slow UEs is useless."""
    _, obs, eng = setup
    sel = peer_underperformers(obs)
    assert 0.05 < sel.mean() < 0.5, sel.mean()
    # uses only telemetry: dropping the latent columns must not change it
    tele = obs.drop(columns=["hw_impairment", "shadow_fading", "traffic_burst"])
    assert np.array_equal(sel.values, peer_underperformers(tele).values)
    # and it finds the right units
    assert obs.hw_impairment[sel].mean() < -0.5
    assert (obs.hw_impairment[sel] < 0).mean() > 0.8


def test_queue_residual_rescales_with_service_rate(setup):
    """UP-35. Q = [a(mu)*xi_tb + eps_Q]_+, so the queue residual carries the
    traffic term at the FACTUAL utilization. Carrying it unchanged through an
    action that moves mu leaves that term scaled by the old a(mu).

    The reported case: mu 6 -> 12 Mbps at lambda = 5.5 gives a(6) = 5.0417 and
    a(12) = 0.1939. With eps_Q = 0 the correct counterfactual is a(12)*xi_tb;
    the additive rule returns a(12)*m + a(6)*(xi_tb - m), which is -2.627 for
    xi_tb = 0.5 and 4.936 for xi_tb = 2.0. The second is a wrong POSITIVE
    prediction, so clipping cannot repair it.
    """
    _, obs, eng = setup
    lam, rmax = 5.5, 0.95
    a = lambda mu: (min(lam / mu, rmax) ** 2) / (2 * (1 - min(lam / mu, rmax)))
    a1, a2, m = a(6.0), a(12.0), np.exp(0.12 / 2)
    for xi, additive_would_give in ((0.5, -2.627), (2.0, 4.936)):
        correct = a2 * xi
        u = a1 * (xi - m)                      # residual at the factual mu
        assert a2 * m + u == pytest.approx(additive_would_give, abs=1e-3)
        # w = 1 (no flat noise) is the pure-scaling limit of the posterior
        rescaled = a2 * m + (a2 / a1) * u
        assert rescaled == pytest.approx(correct, abs=1e-9)
        assert abs(rescaled - correct) < abs(additive_would_give - correct)


def test_counterfactual_queue_beats_additive_against_oracle(setup):
    """End-to-end: the scaled rule must track the simulator's own queue under
    an action that changes the service rate, where the additive rule does not.
    """
    sim, obs, eng = setup
    res = eng.abduct(obs)
    pred = eng.predict(obs, {"num_prb": 100}, res)
    truth = sim.sample(len(obs), seed=11, regime="do", do={"num_prb": 100})
    err_scaled = np.sqrt(np.mean((pred.queue_dep.values
                                  - truth.queue_dep.values) ** 2))
    # the additive rule, reconstructed from the same fit and the same residual
    f0 = eng.model.predict_node("queue_dep", obs)
    f1 = eng.model.predict_node("queue_dep", pred)
    err_add = np.sqrt(np.mean((np.maximum(f1 + res["queue_dep"], 0.0)
                               - truth.queue_dep.values) ** 2))
    assert err_scaled < err_add


def test_counterfactual_queue_never_negative(setup):
    """queue_dep ends in [.]_+ , so no counterfactual may fall below zero."""
    _, obs, eng = setup
    res = eng.abduct(obs)
    for t in (10, 50, 100):
        cf = eng.predict(obs, {"num_prb": t}, res)
        assert (cf.queue_dep.values >= 0.0).all()


ARMS = [{"num_prb": 10}, {"num_prb": 50}, {"num_prb": 100},
        {"hw_impairment": 0.0}]


@pytest.mark.parametrize("do", ARMS)
def test_counterfactual_respects_physical_support(setup, do):
    """UP-36. Every counterfactual must be a physically possible world.

    A high outcome R^2 does not imply this: a model can explain most of the
    variation in Y and still hand back a negative goodput or a loss
    probability above 1 for individual UEs, and those units are exactly the
    ones a scheduler would act on.
    """
    _, obs, eng = setup
    cf = eng.predict(obs, do, eng.abduct(obs))
    tau0 = float(eng.cfg["network"]["tau0_ms"])
    assert (cf.queue_dep.values >= 0.0).all()
    assert (cf.pkt_loss.values >= 0.0).all() and (cf.pkt_loss.values <= 1.0).all()
    assert (cf.bler.values >= 0.0).all() and (cf.bler.values <= 1.0).all()
    assert (cf.mac_tput.values >= 0.0).all()
    assert (cf.rtt_ms.values >= tau0 - 1e-9).all()
    # 0 <= Y <= mu, the pair the separate goodput fit used to break
    assert (cf.goodput.values >= 0.0).all()
    assert (cf.goodput.values <= cf.mac_tput.values + 1e-9).all()


@pytest.mark.parametrize("do", ARMS)
def test_goodput_identity_holds_in_counterfactual(setup, do):
    """Y = mu (1 - p_loss) is a definition, so it must hold exactly after an
    intervention too -- not merely to within the goodput model's error."""
    _, obs, eng = setup
    cf = eng.predict(obs, do, eng.abduct(obs))
    lhs = cf.goodput.values
    rhs = cf.mac_tput.values * (1.0 - cf.pkt_loss.values)
    assert np.abs(lhs - rhs).max() < 1e-9


def test_identity_node_abducts_no_noise(setup):
    """An exact identity has nothing to abduct: its residual must be zero, or
    the definition would be perturbed by model error on replay."""
    _, obs, eng = setup
    assert np.abs(eng.abduct(obs)["goodput"]).max() == 0.0
