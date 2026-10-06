"""L2 MAC layer (paper Section IV-A). All functions vectorized over units.

Structural equations owned by this module:

    num_prb  = clip(floor(base_prb * (1 - alpha_L*L)) + eps_T, 1, base_prb)   (eq. 8)
    harq_retx = sum_{k=1..K} bler^k          (expected HARQ-IR retransmissions)
    bler_res  = bler^(1 + K)                 (residual decoding failure after the
                                             initial transmission and K retransmissions;
                                             consumed ONLY by network.packet_loss)
    se_eff   = se_phy * (1 + kappa_hw * xi_hw)
    mac_tput = C0 * num_prb * se_eff / (1 + kappa_harq * harq_retx)

HARQ model — expected-retransmission surrogate (UP-21, UP-37). Under an
independent-failure reading (every attempt fails with probability bler, up
to K retransmissions):
  - harq_retx = sum_{k=1..K} bler^k is the EXPECTED retransmission count —
    a fractional quantity (0.37, 1.42, ...), not an integer round count;
  - the residual failure probability after all 1+K attempts is bler^(1+K).
    Both follow from the same assumption, so bler_res depends on bler and
    the fixed limit K only — NOT on harq_retx. (UP-37: the earlier surrogate
    bler^(1 + harq_retx) charged 7.7% residual loss at the 0.1 BLER target,
    where independent attempts give 0.01%.)
Near-deterministic given bler, consistent with the paper's Table I
(harq_retx R^2 = 0.999).

Note (paper §II-B): the scheduler never observes L directly — eq. (8) is a
congestion response. L -> num_prb is the edge that creates confounding and
the edge severed by do(prb = t).
"""
import numpy as np


def scheduler_num_prb(load, eps_T, cfg):
    """Eq. (8), observational mode. do(prb=t) is applied by the orchestrator
    (constant replaces this equation; this function is never called then)."""
    base = int(cfg["mac"]["base_prb"])
    alpha = float(cfg["mac"]["alpha_L"])
    raw = np.floor(base * (1.0 - alpha * np.asarray(load))) + np.asarray(eps_T)
    return np.clip(raw, 1, base)


def harq_expected_retx(bler, cfg):
    """Expected HARQ-IR retransmissions: sum_{k=1..K} bler^k, in [0, K]."""
    K = int(cfg["mac"]["harq_max_rounds"])
    bler = np.asarray(bler, dtype=float)
    return sum(bler ** k for k in range(1, K + 1))


def residual_bler(bler, cfg):
    """Residual decoding failure after the initial transmission and K
    retransmissions that each fail independently with probability bler:
    bler^(1 + K), K = mac.harq_max_rounds (UP-37)."""
    K = int(cfg["mac"]["harq_max_rounds"])
    bler = np.asarray(bler, dtype=float)
    return np.clip(bler, 1e-12, 1.0) ** (1.0 + K)


def se_eff(se_phy, xi_hw, cfg):
    """Device-impairment-modified spectral efficiency (effect modifier xi_hw)."""
    k = float(cfg["mac"]["kappa_hw"])
    return np.asarray(se_phy) * np.maximum(1.0 + k * np.asarray(xi_hw), 0.05)


def mac_throughput(num_prb, se_phy, xi_hw, harq_retx, cfg):
    """Available MAC service rate, Mbps.

    RELIABILITY IS NOT APPLIED HERE. This is the rate the MAC can serve after
    retransmission overhead; whether a served block decodes is settled once,
    in network.packet_loss.

    harq_retx still enters, through 1 / (1 + kappa_harq * harq_retx): a
    retransmission consumes airtime whether or not it eventually succeeds.
    That is an overhead term, not a reliability term.
    """
    C0 = float(cfg["mac"]["tput_scale_mbps_per_prb_se"])
    kh = float(cfg["mac"]["kappa_harq"])
    tput = (C0 * np.asarray(num_prb) * se_eff(se_phy, xi_hw, cfg)
            / (1.0 + kh * np.asarray(harq_retx)))
    floor = float(cfg["mac"].get("tput_floor_mbps", 0.0))
    return np.maximum(tput, floor)


