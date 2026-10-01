"""StructuralModel tests: observability split, residual kinds, cross-fitting.

Uses the analytical backend and a small node subset to stay fast.
"""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.fitting import (LATENT_NODES, StructuralModel,      # noqa: E402
                            fit_scale_params, load_parent_sets,
                            residual_kind)
from simulator.phy import AnalyticalPHY                          # noqa: E402
from simulator.ran_simulator import RANSimulator                 # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
NODES = ["sinr_eff", "num_prb", "mac_tput"]


@pytest.fixture(scope="module")
def obs():
    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["phy"]["backend"] = "analytical"
    return RANSimulator(cfg, backend=AnalyticalPHY(cfg)).sample(400, seed=7)


def test_telemetry_parents_exclude_all_latents():
    tele = load_parent_sets(ROOT, "telemetry")
    for node, pa in tele.items():
        assert not (set(pa) & LATENT_NODES), (node, pa)
    # oracle mode keeps the physical latents but not the eps noises
    ora = load_parent_sets(ROOT, "oracle")
    assert "shadow_fading" in ora["sinr_eff"]
    assert "shadow_fading" not in tele["sinr_eff"]
    # queue_dep keeps only its observable parent; xi_tb and eps_3 are latent
    assert tele["queue_dep"] == ["mac_tput"]


def test_residual_kinds():
    assert residual_kind("mac_tput") == "multiplicative"
    # queue_dep is 'scaled', not 'additive': its residual is
    # a(pa)*(xi_tb - m) + eps_Q, and a(pa) moves when an action moves the
    # service rate, so the residual cannot be carried unchanged (UP-35).
    assert residual_kind("queue_dep") == "scaled"
    assert residual_kind("sinr_eff") == "additive"


def test_scale_params_recover_the_variance_split():
    """fit_scale_params identifies the SCALING variance from u^2 ~ f_hat^2.

    zeta2 is the parameter that matters and the one that is well identified.
    sigma2_eps is the intercept of a regression whose response has variance
    growing as f^4, so large-f rows dominate it and it is recovered only to an
    order of magnitude. That asymmetry is harmless here: sigma2_eps only sets
    where the posterior weight rolls off, and w saturates near 1 wherever the
    queue is large enough for the rescaling to matter.
    """
    rng = np.random.default_rng(0)
    n = 4000
    f = rng.uniform(0.2, 8.0, n)                 # stands in for a(pa) * m
    zeta2, s2 = 0.13, 0.02
    u = f * rng.normal(0, np.sqrt(zeta2), n) + rng.normal(0, np.sqrt(s2), n)
    z_hat, s_hat = fit_scale_params(f, u, np.zeros(n, dtype=bool))
    assert z_hat == pytest.approx(zeta2, rel=0.15)
    assert 0.0 < s_hat < 20 * s2


def test_scale_params_ignore_censored_rows():
    """Clipped rows carry u = -f_hat exactly and must not inflate zeta2."""
    rng = np.random.default_rng(1)
    n = 2000
    f = rng.uniform(0.2, 8.0, n)
    u = f * rng.normal(0, np.sqrt(0.13), n)
    cens = np.zeros(n, dtype=bool)
    cens[:400] = True
    u[:400] = -f[:400] * 4.0                     # gross censored residuals
    z_clean, _ = fit_scale_params(f, u, cens)
    z_dirty, _ = fit_scale_params(f, u, np.zeros(n, dtype=bool))
    assert z_clean == pytest.approx(0.13, rel=0.2)
    assert z_dirty > 3 * z_clean                 # they really would poison it


def test_scale_weight_endpoints():
    """w -> 1 when flat noise vanishes, w -> 0 when the scaling term does."""
    m = StructuralModel(ROOT, mode="telemetry")
    m.scale["queue_dep"] = (0.13, 1e-12)
    assert m.scale_weight("queue_dep", np.array([5.0]))[0] > 0.999
    m.scale["queue_dep"] = (0.0, 0.02)
    assert m.scale_weight("queue_dep", np.array([5.0]))[0] == 0.0
    # and it self-regularizes: as f_hat -> 0 the rescaling ratio is unstable,
    # and the weight that would use it goes to zero on its own.
    m.scale["queue_dep"] = (0.13, 0.02)
    assert m.scale_weight("queue_dep", np.array([1e-6]))[0] < 1e-6


def test_abduction_recovers_latents_out_of_fold(obs):
    m = StructuralModel(ROOT, mode="telemetry").fit_crossfit(
        obs, folds=3, nodes=NODES)
    r = m.residuals(obs, fold_of=m.fold_of)
    # sinr_eff residual tracks the true (never seen) shadow fading
    assert np.corrcoef(r["sinr_eff"], obs["shadow_fading"])[0, 1] > 0.8
    # mac_tput multiplicative residual tracks (1 + kappa*xi_hw) — WEAKLY:
    # the xi_hw effect is only +-7.5% (kappa_hw*sigma) against ~10% OOF model
    # error, so r ~ 0.4 is the honest recoverability of the device
    # fingerprint (documented in UP-25; quantified vs oracle in Phase 7)
    assert np.corrcoef(r["mac_tput"], obs["hw_impairment"])[0, 1] > 0.2
    # xi_tb is NOT recoverable: it survives only inside the queue_dep
    # residual, entangled with rho and eps_3, and no observable inverts it.
    import causal.counterfactual as C
    assert "xi_tb_hat" not in C.latent_fingerprints(r, obs, 0.05)


def test_crossfit_residuals_not_optimistic(obs):
    tele = StructuralModel(ROOT, mode="telemetry").fit(
        obs, cv_folds=3, nodes=["sinr_eff"])
    telx = StructuralModel(ROOT, mode="telemetry").fit_crossfit(
        obs, folds=3, nodes=["sinr_eff"])
    sd_in = np.std(tele.residuals(obs)["sinr_eff"])
    sd_oof = np.std(telx.residuals(obs, fold_of=telx.fold_of)["sinr_eff"])
    assert sd_oof > sd_in            # in-sample optimism removed
    assert sd_oof > 6.0              # ~ sqrt(var(xi_sf)=49) + model error
