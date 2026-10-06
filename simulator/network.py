"""L3 network layer (paper Section IV-A). All functions vectorized over units.

Structural equations owned by this module:

    rho_raw   = ARR_CONST / mac_tput                (unclamped offered intensity)
    rho       = min(rho_raw, rho_max)               (P-K validity clamp, eq. 9 input)
    queue_dep = [ rho^2 * xi_tb / (2*(1 - rho)) + eps3 ]_+                     (eq. 9)
    rtt_ms    = tau0 + c_q * queue_dep / arr + tau_harq * harq_retx   (UP-31:
                Little's law W_q = L_q / lambda; mac_tput is NOT a parent)
    pkt_loss  = 1 - (1 - bler_res) * exp(-max(0, queue_dep - q_thr) / q_scale) (UP-8)
    goodput   = mac_tput * (1 - pkt_loss)          <- outcome Y

arr is a FIXED simulation constant (network.arrival_mbps; UP-9) — NOT an SCM
node.

GOODPUT IS A SERVED RATE, NOT A SHARE OF OFFERED LOAD (UP-34). Y is what this
UE's elastic traffic achieves: the service rate reduced by the losses that
apply to what is served. `arr` is the loading parameter of the QUEUEING model
— it sets rho, hence backlog and delay — and is NOT the offered rate of the
flow whose goodput is reported; Y is not bounded by it.

pkt_loss therefore has NO mac_tput parent. Multiplying by a served fraction
min(1, mac_tput/arr) applied the service limit a SECOND time, because
Y = mac_tput * (1 - pkt_loss) already carries it once: composed, the delivered
rate became mac_tput^2/arr under overload — half the true value at
mac_tput = arr/2. rho_max = 0.95 remains a numerical validity clamp for the
queue term and nothing else.
"""
import numpy as np

from .mac import residual_bler


def raw_traffic_intensity(mac_tput, cfg):
    """Unclamped offered intensity rho_raw = arr / mac_tput."""
    arr = float(cfg["network"]["arrival_mbps"])
    return arr / np.asarray(mac_tput)


def traffic_intensity(mac_tput, cfg):
    """rho for eq. (9): rho_raw clamped to [0, rho_max] for P-K validity."""
    rho_max = float(cfg["network"]["rho_max"])
    return np.clip(raw_traffic_intensity(mac_tput, cfg), 0.0, rho_max)


def queue_depth(mac_tput, xi_tb, eps3, cfg):
    """Eq. (9): mean queue BACKLOG L_q, from the GI/G/1 approximation.

        L_q = [ rho^2 * C_a^2 / (2 (1 - rho)) + eps3 ]_+     with xi_tb = C_a^2

    This is Sakasegawa's mean queue-length approximation with C_s^2 = 0. The
    zero service-time variability is deliberate, not an omission: once
    mac_tput is fixed for a unit the SCM treats its service capability as
    fixed at that operating point, so a separate C_s^2 would inject
    randomness the simulator does not otherwise have.

    WAITING, NOT IN SYSTEM. The term is L_q, so the "+ rho" that would give
    total occupancy L = L_q + rho is deliberately absent: the variable is
    called queue_dep and feeds a queueing delay, so the backlog reading is
    the one that keeps rtt interpretable.

ARRIVAL VARIABILITY ENTERS ONCE, as C_a^2 = xi_tb in the numerator. Buffer
    occupancy is a description of queue state rather than an independent cause
    of it, so no occupancy term appears here; no GI/G/1 result calls for one.

    rho = min(rho_raw, rho_max) is OUR clamp, not part of the approximation,
    which requires rho < 1 for a finite steady-state queue. Physical overload
    (rho_raw > 1) is handled separately, in packet_loss.
    """
    rho = traffic_intensity(mac_tput, cfg)
    qd = rho ** 2 * np.asarray(xi_tb) / (2.0 * (1.0 - rho)) + np.asarray(eps3)
    return np.maximum(qd, 0.0)


def rtt(queue_dep, harq_retx, cfg):
    """RTT in ms.  tau0 + c_q * queue_dep / arr + tau_harq * harq_retx  (UP-31)

    LITTLE'S LAW, NOT THE SERVICE RATE. queue_dep is Sakasegawa's mean number
    WAITING, L_q (UP-27). Little applied to the queue gives the waiting time
    experienced by an arriving packet as

        W_q = L_q / lambda        lambda = arrival rate, NOT mu = mac_tput

    The previous form divided by mac_tput, i.e. computed L_q / mu. That is not
    a waiting time: substituting L_q = lambda W_q gives

        L_q / mu = (lambda / mu) W_q = rho * W_q

    so it understated queueing delay by exactly the utilization factor. It is
    the time to DRAIN the backlog at the service rate, which is a different
    quantity from the time an arriving packet waits.

    mac_tput IS NOT A PARENT any more. lambda is a fixed simulation constant
    (UP-9), so nothing about the service rate enters this equation directly.
    mac_tput still reaches rtt_ms, but along the physically correct path
        mac_tput -> queue_dep -> rtt_ms
    (lower service rate -> higher utilization -> longer queue -> more delay),
    which is mediated, not direct.

    c_q carries the packet size and is the SAME constant as before: with 1 kB
    packets, 8 kbit / 1 Mbps = 8 ms, so c_q = 8.0 ms*Mbps/packet whether the
    divisor is lambda or mu. The correction therefore introduces no new free
    parameter -- it only puts the right rate in the denominator.
    """
    n = cfg["network"]
    arr = float(n["arrival_mbps"])
    tau0 = float(n.get("tau0_ms", 1.0))
    c_q = float(n.get("c_q_ms_mbps", 8.0))
    t_h = float(n.get("tau_harq_ms", 8.0))
    return (tau0 + c_q * np.asarray(queue_dep) / arr
            + t_h * np.asarray(harq_retx))


def packet_loss(bler, queue_dep, cfg):
    """UP-8 loss model: residual link loss x queue-overflow loss.

    harq_retx is NOT a parent (UP-37): the residual decoding failure after
    the initial transmission and K retransmissions is bler^(1 + K), which
    depends on bler and the fixed limit K only.

    Both factors are fractions OF WHAT IS SERVED, which is what a loss
    probability should be. mac_tput is NOT a parent -- see the module note on
    UP-34 for why the served-fraction term double-counted the service limit.
    """
    q_thr = float(cfg["network"]["q_thr"])
    q_scale = float(cfg["network"]["q_scale"])
    b_res = residual_bler(bler, cfg)
    overflow = np.exp(-np.maximum(0.0, np.asarray(queue_dep) - q_thr) / q_scale)
    return np.clip(1.0 - (1.0 - b_res) * overflow, 0.0, 1.0)


def goodput(mac_tput, pkt_loss):
    """Outcome Y: goodput = tput * (1 - loss), Mbps."""
    return np.asarray(mac_tput) * (1.0 - np.asarray(pkt_loss))
