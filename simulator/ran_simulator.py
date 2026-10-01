"""H-SCM orchestrator: samples the full 25-node SCM in topological order.

Modes
-----
observational : draw all exogenous U from config distributions, propagate eqs.
do(...)       : graph mutilation (paper Definition 3) — the structural equation
                of each intervened node is replaced by a constant (its incoming
                edges are thereby deleted); descendants recompute, everything
                else untouched. do(prb=t) severs L -> num_prb while keeping
                L -> sinr_eff active (noise persistence).
fixed-unit    : propagate(U, do=...) with a supplied exogenous vector — the
                counterfactual ORACLE for validating Algorithm 1.

Intervenable names = any exogenous variable (cell_load, snr, shadow_fading,
traffic_burst, ...) or endogenous variable computed here (num_prb, mcs_idx,
harq_retx, ...).

Every returned frame has the 25 node columns (10 exogenous + 15 endogenous)
plus a `regime` label — 24 columns (config/variables.yaml).
"""
import numpy as np
import pandas as pd

from . import mac, network
from .phy import make_backend, sinr_eff_db

EXOGENOUS = ["cell_load", "shadow_fading", "hw_impairment", "traffic_burst",
             "snr", "doppler", "delay_spread", "eps_sinr", "eps_sched",
             "eps_queue"]
ENDOGENOUS = ["sinr_eff", "mcs_idx", "mod_order", "code_rate", "bler",
              "se_phy", "num_prb", "harq_retx", "mac_tput",
              "queue_dep", "pkt_loss", "rtt_ms", "goodput"]
COLUMNS = EXOGENOUS + ENDOGENOUS


class RANSimulator:
    """Ground-truth H-SCM over a PHYBackend (analytical | sionna)."""

    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend if backend is not None else make_backend(cfg)
        m_r = np.array(self.backend.entries, dtype=float)
        self._mod_orders = m_r[:, 0]
        self._code_rates = m_r[:, 1]

    # ------------------------------------------------------------------
    # exogenous sampling (paper §V + UP-9/UP-10)
    # ------------------------------------------------------------------
    def sample_exogenous(self, n, rng):
        o = self.cfg["observational"]
        u = {
            "cell_load": rng.beta(o["cell_load"]["a"], o["cell_load"]["b"], n),
            "shadow_fading": rng.normal(o["shadow_fading_db"]["mean"],
                                        np.sqrt(o["shadow_fading_db"]["var"]), n),
            "hw_impairment": rng.normal(o["hw_impairment"]["mean"],
                                        np.sqrt(o["hw_impairment"]["var"]), n),
            "traffic_burst": rng.lognormal(o["traffic_burst"]["mu"],
                                           np.sqrt(o["traffic_burst"]["sigma2"]), n),
            "snr": rng.uniform(o["snr_db"]["low"], o["snr_db"]["high"], n),
            "doppler": rng.uniform(o["doppler_hz"]["low"],
                                   o["doppler_hz"]["high"], n),
            "delay_spread": rng.uniform(o["delay_spread_ns"]["low"],
                                        o["delay_spread_ns"]["high"], n),
            "eps_sinr": rng.normal(0.0, np.sqrt(o["eps_sinr"]["var"]), n),
            "eps_sched": np.round(rng.normal(0.0, np.sqrt(o["eps_sched_prb"]["var"]), n)),
            "eps_queue": rng.normal(0.0, np.sqrt(o["eps_queue"]["var"]), n),
        }
        return u

    # ------------------------------------------------------------------
    # structural propagation (topological order; do = graph mutilation)
    # ------------------------------------------------------------------
    def propagate(self, u, do=None):
        """Propagate exogenous dict u through the (possibly mutilated) SCM.

        u: dict name -> array (n,) for all 10 exogenous variables.
        do: dict node -> constant. Exogenous do-targets override u; endogenous
        do-targets replace the node's structural equation.
        Returns a DataFrame with the 25 node columns.
        """
        do = dict(do or {})
        unknown = set(do) - set(EXOGENOUS) - set(ENDOGENOUS)
        if unknown:
            raise ValueError(
                f"unknown do() target(s) {sorted(unknown)}: valid names are "
                f"the 25 SCM nodes ({', '.join(EXOGENOUS + ENDOGENOUS)})")
        u = {k: np.asarray(v, dtype=float).copy() for k, v in u.items()}
        n = len(next(iter(u.values())))

        # do() on exogenous variables
        for name in list(do):
            if name in EXOGENOUS:
                u[name] = np.full(n, float(do.pop(name)))

        def forced(name):
            return np.full(n, float(do[name])) if name in do else None

        v = {}

        # --- L1 PHY ---
        v["sinr_eff"] = (forced("sinr_eff") if "sinr_eff" in do else
                         sinr_eff_db(u["snr"], u["shadow_fading"], u["cell_load"],
                                     u["doppler"], u["delay_spread"],
                                     u["eps_sinr"], self.cfg))
        if "mcs_idx" in do:
            v["mcs_idx"] = np.full(n, int(do["mcs_idx"]), dtype=int)
        else:
            v["mcs_idx"] = self.backend.select_mcs_vec(v["sinr_eff"])
        v["mod_order"] = (forced("mod_order") if "mod_order" in do else
                          self._mod_orders[v["mcs_idx"]])
        v["code_rate"] = (forced("code_rate") if "code_rate" in do else
                          self._code_rates[v["mcs_idx"]])
        v["bler"] = (forced("bler") if "bler" in do else
                     self.backend.bler_vec(v["sinr_eff"], v["mcs_idx"]))
        # nominal MCS spectral efficiency Qm * R. Reliability is NOT applied
        # here -- residual decoding failure is settled once, in packet_loss.
        v["se_phy"] = (forced("se_phy") if "se_phy" in do else
                       v["mod_order"] * v["code_rate"])

        # --- L2 MAC ---
        v["num_prb"] = (forced("num_prb") if "num_prb" in do else
                        mac.scheduler_num_prb(u["cell_load"], u["eps_sched"],
                                              self.cfg))
        v["harq_retx"] = (forced("harq_retx") if "harq_retx" in do else
                          mac.harq_expected_retx(v["bler"], self.cfg))
        v["mac_tput"] = (forced("mac_tput") if "mac_tput" in do else
                         mac.mac_throughput(v["num_prb"], v["se_phy"],
                                            u["hw_impairment"],
                                            v["harq_retx"], self.cfg))

        # --- L3 network ---
        v["queue_dep"] = (forced("queue_dep") if "queue_dep" in do else
                          network.queue_depth(v["mac_tput"], u["traffic_burst"],
                                              u["eps_queue"], self.cfg))
        v["pkt_loss"] = (forced("pkt_loss") if "pkt_loss" in do else
                         network.packet_loss(v["bler"], v["harq_retx"],
                                             v["queue_dep"], self.cfg))
        v["rtt_ms"] = (forced("rtt_ms") if "rtt_ms" in do else
                       network.rtt(v["queue_dep"], v["harq_retx"], self.cfg))
        v["goodput"] = (forced("goodput") if "goodput" in do else
                        network.goodput(v["mac_tput"], v["pkt_loss"]))

        df = pd.DataFrame({**u, **v})
        return df[COLUMNS]

    # ------------------------------------------------------------------
    # public sampling API
    # ------------------------------------------------------------------
    def sample(self, n, do=None, seed=None, regime=None):
        """Sample n units (observational if do is None/empty)."""
        rng = np.random.default_rng(seed)
        df = self.propagate(self.sample_exogenous(n, rng), do=do)
        if regime is None:
            regime = "obs" if not do else "do(" + ",".join(
                f"{k}={v:g}" for k, v in sorted(do.items())) + ")"
        df["regime"] = regime
        return df
