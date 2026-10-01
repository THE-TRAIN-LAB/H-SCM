"""L1 PHY layer (paper Section IV-A).

sinr_eff convention (project-wide, FIXED)
-----------------------------------------
``sinr_eff`` is **Es/N0 per resource element at the receiver input**, in dB:
the ratio of average data-RE symbol energy to noise power spectral density,
*before* equalization.

In the Sionna backend this is exact by construction:
  - ``Mapper("qam", m)`` uses unit-energy constellations (Es = 1 per RE),
  - ``OFDMChannel(..., normalize_channel=True)`` enforces E[|h|^2] = 1 per RE,
  - AWGN is added with variance ``no`` per complex RE,
hence ``no = 10**(-sinr_eff/10)`` with no rate/modulation/overhead correction.
Code rate, modulation order, pilot overhead and channel-estimation loss affect
``BLER(sinr_eff, mcs)`` — never the conversion itself. (It is NOT Eb/N0 and
NOT post-equalization SINR.)

Backends
--------
Two interchangeable implementations of :class:`PHYBackend` (the causal
pipeline must only ever touch this interface, never Sionna directly):

  - :class:`AnalyticalPHY` — AWGN logistic approximation (the paper's
    "analytical fallback"): fast, deterministic; for unit tests, debugging
    causal logic, and analytical-vs-Sionna ablations.
  - :class:`SionnaPHY`     — Sionna 1.2 end-to-end link: LDPC5G (512 info
    bits), OFDM (76 sc, 15 kHz, 14 symbols), CDL-C, BLER from the forward
    pass (batch 32). Ground truth for the paper's results.

Structural equations owned by this module:
    sinr_eff = snr + xi_sf - beta_L*L - dop_pen(dop) - ds_pen(ds) + eps1   (eq. 7)
    mcs_idx  = highest MCS with BLER(sinr_eff, mcs) <= target_bler
"""
import abc
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


# --------------------------------------------------------------------------
# eq. (7) terms
# --------------------------------------------------------------------------

def doppler_penalty_db(dop_hz, a, f0):
    """Monotone Doppler penalty Delta_dop (UP-6). No defaults: coefficients
    come from config/simulator.yaml (sinr.doppler_penalty) — stale defaults
    here once diverged from the calibrated values."""
    return a * np.log2(1.0 + np.asarray(dop_hz) / f0)


def delay_penalty_db(ds_ns, a, d0):
    """Monotone delay-spread penalty Delta_ds (UP-7). No defaults; see
    config/simulator.yaml (sinr.delay_penalty)."""
    return a * np.log2(1.0 + np.asarray(ds_ns) / d0)


def sinr_eff_db(snr, xi_sf, load, dop, ds, eps1, cfg):
    """Effective SINR (Es/N0 per RE, dB), eq. (7)."""
    dp = cfg["sinr"]["doppler_penalty"]
    sp = cfg["sinr"]["delay_penalty"]
    return (snr + xi_sf - cfg["sinr"]["beta_L"] * load
            - doppler_penalty_db(dop, dp["a"], dp["f0"])
            - delay_penalty_db(ds, sp["a"], sp["d0"])
            + eps1)


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------

class PHYBackend(abc.ABC):
    """BLER provider + MCS logic. The only PHY surface the SCM pipeline sees."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = [(int(m), float(r)) for m, r in cfg["mcs_table"]]
        self.target_bler = float(cfg["phy"]["target_bler"])

    @abc.abstractmethod
    def bler(self, sinr_db, mcs_idx):
        """Block error rate at a given effective SINR (Es/N0 per RE, dB)."""

    @abc.abstractmethod
    def selection_thresholds(self):
        """Array (n_mcs,): SINR (dB) above which BLER(mcs) <= target_bler.
        Monotone increasing in MCS index (higher MCS needs more SINR)."""

    def bler_vec(self, sinr_db, mcs_idx):
        """Vectorized BLER for per-sample (sinr, mcs) pairs."""
        sinr_db = np.asarray(sinr_db, dtype=float)
        mcs_idx = np.asarray(mcs_idx, dtype=int)
        out = np.empty_like(sinr_db)
        for idx in np.unique(mcs_idx):
            sel = mcs_idx == idx
            out[sel] = [self.bler(s, int(idx)) for s in np.atleast_1d(sinr_db[sel])]
        return out

    def select_mcs_vec(self, sinr_db):
        """Vectorized MCS selection: highest MCS with BLER <= target; 0 if none."""
        thr = self.selection_thresholds()
        sinr_db = np.atleast_1d(np.asarray(sinr_db, dtype=float))
        idx = (sinr_db[:, None] >= thr[None, :]).sum(axis=1) - 1
        return np.clip(idx, 0, len(self.entries) - 1)

    def select_mcs(self, sinr_db):
        return int(self.select_mcs_vec([float(sinr_db)])[0])

class AnalyticalPHY(PHYBackend):
    """AWGN logistic approximation (paper §IV-A "analytical fallback").

    BLER(sinr, mcs) = sigmoid((thr_mcs - sinr)/slope), where the waterfall
    midpoint uses a Shannon-gap threshold:
        thr_mcs = 10*log10(2**(m*r) - 1) + gap_db.
    gap_db and slope_db live in cfg["phy"]["analytical"]; defaults are
    roughly calibrated to the Sionna CDL-C link (Phase 2 refines this).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        a = cfg["phy"]["analytical"]
        self.gap_db = float(a["gap_db"])
        self.slope_db = float(a["slope_db"])
        self._thr = np.array([10.0 * np.log10(2.0 ** (m * r) - 1.0) + self.gap_db
                              for m, r in self.entries])

    def bler(self, sinr_db, mcs_idx):
        z = (self._thr[mcs_idx] - np.asarray(sinr_db)) / self.slope_db
        return float(1.0 / (1.0 + np.exp(-z)))

    def bler_vec(self, sinr_db, mcs_idx):
        sinr_db = np.asarray(sinr_db, dtype=float)
        mcs_idx = np.asarray(mcs_idx, dtype=int)
        z = (self._thr[mcs_idx] - sinr_db) / self.slope_db
        return 1.0 / (1.0 + np.exp(-z))

    def selection_thresholds(self):
        # logistic BLER = t  at  thr + slope*ln((1-t)/t)
        t = self.target_bler
        return self._thr + self.slope_db * np.log((1.0 - t) / t)


class SionnaPHY(PHYBackend):
    """Sionna 1.2 forward-pass backend (ground truth for reported results).

    BLER comes from a cached (sinr x mcs) lookup table built once by running
    the Sionna link over each MCS's waterfall region (data/bler_lut.npz);
    dataset generation interpolates instead of re-running Sionna per sample.
    Curves are made monotone (cumulative min over increasing SINR) to remove
    Monte-Carlo jitter, keeping bler a deterministic function of (sinr, mcs)
    as the SCM requires.
    """

    LUT_STEP_DB = 0.5
    LUT_HALF_WINDOW_DB = 10.0

    def __init__(self, cfg, lut_path=None, build_if_missing=False):
        super().__init__(cfg)
        self._links = {}
        root = os.path.join(os.path.dirname(__file__), "..")
        self.lut_path = lut_path or os.path.join(root, "data", "bler_lut.npz")
        self._grid = None      # (n_mcs, n_pts) SINR grid
        self._lut = None       # (n_mcs, n_pts) monotone BLER
        if os.path.exists(self.lut_path):
            self._load_lut()
        elif build_if_missing:
            self.build_lut()

    # -- config fingerprint: LUT is invalid if PHY/MCS config changed --
    def _fingerprint(self):
        import json
        keys = ["ldpc_info_bits", "fft_size", "subcarrier_spacing",
                "num_ofdm_symbols", "channel_model", "batch_size",
                "carrier_frequency", "cdl_delay_spread", "ue_speed_mps",
                "pilot_pattern", "pilot_ofdm_symbol_indices",
                "cyclic_prefix_length"]
        return json.dumps({k: self.cfg["phy"][k] for k in keys}
                          | {"mcs": self.entries}, sort_keys=True)

    def _link(self, mcs_idx):
        if mcs_idx not in self._links:
            m, r = self.entries[mcs_idx]
            self._links[mcs_idx] = SionnaLink(m, r, self.cfg)
        return self._links[mcs_idx]

    def build_lut(self, n_batches=4, verbose=True):
        """Run the Sionna link over each MCS's waterfall window and cache."""
        ana = AnalyticalPHY(self.cfg)   # centers the sweep window per MCS
        n_pts = int(2 * self.LUT_HALF_WINDOW_DB / self.LUT_STEP_DB) + 1
        grid = np.empty((len(self.entries), n_pts))
        lut = np.empty_like(grid)
        for idx in range(len(self.entries)):
            center = ana._thr[idx]
            grid[idx] = center + np.linspace(-self.LUT_HALF_WINDOW_DB,
                                             self.LUT_HALF_WINDOW_DB, n_pts)
            link = self._link(idx)
            for j, s in enumerate(grid[idx]):
                lut[idx, j] = link.bler(float(s), n_batches=n_batches)
            # enforce monotone non-increasing BLER in SINR
            lut[idx] = np.minimum.accumulate(lut[idx])
            if verbose:
                m, r = self.entries[idx]
                print(f"LUT mcs={idx} (m={m}, r={r:.2f}): waterfall "
                      f"{grid[idx][np.searchsorted(-lut[idx], -0.5)]:.1f} dB "
                      f"(bler=0.5)", flush=True)
            self._links.clear()   # free graph memory per link
        os.makedirs(os.path.dirname(self.lut_path), exist_ok=True)
        np.savez(self.lut_path, grid=grid, lut=lut,
                 fingerprint=np.array(self._fingerprint()))
        self._grid, self._lut = grid, lut

    def _load_lut(self):
        z = np.load(self.lut_path, allow_pickle=False)
        if str(z["fingerprint"]) != self._fingerprint():
            raise RuntimeError(
                f"{self.lut_path} was built with a different PHY/MCS config; "
                "rebuild with SionnaPHY(cfg).build_lut()")
        self._grid, self._lut = z["grid"], z["lut"]

    def _require_lut(self):
        if self._lut is None:
            raise RuntimeError("BLER LUT missing — run "
                               "experiments/build_bler_lut.py first")

    def bler(self, sinr_db, mcs_idx):
        self._require_lut()
        g, b = self._grid[mcs_idx], self._lut[mcs_idx]
        return float(np.interp(sinr_db, g, b, left=1.0, right=b[-1]))

    def bler_vec(self, sinr_db, mcs_idx):
        self._require_lut()
        sinr_db = np.asarray(sinr_db, dtype=float)
        mcs_idx = np.asarray(mcs_idx, dtype=int)
        out = np.empty_like(sinr_db)
        for idx in np.unique(mcs_idx):
            sel = mcs_idx == idx
            g, b = self._grid[idx], self._lut[idx]
            out[sel] = np.interp(sinr_db[sel], g, b, left=1.0, right=b[-1])
        return out

    def selection_thresholds(self):
        self._require_lut()
        thr = np.empty(len(self.entries))
        for idx in range(len(self.entries)):
            g, b = self._grid[idx], self._lut[idx]
            below = np.nonzero(b <= self.target_bler)[0]
            if len(below) == 0:
                thr[idx] = g[-1] + self.LUT_STEP_DB   # never selectable
                continue
            j = below[0]
            if j == 0:
                thr[idx] = g[0]
            else:   # linear interpolation of the crossing point
                f = (b[j - 1] - self.target_bler) / max(b[j - 1] - b[j], 1e-12)
                thr[idx] = g[j - 1] + f * (g[j] - g[j - 1])
        # select_mcs_vec's counting rule needs thresholds monotone in MCS;
        # enforce against residual Monte-Carlo jitter
        return np.maximum.accumulate(thr)

    def bler_forward_pass(self, sinr_db, mcs_idx, n_batches=1):
        """Direct Sionna forward pass (bypasses LUT) — for validation only."""
        return self._link(mcs_idx).bler(sinr_db, n_batches=n_batches)


def make_backend(cfg):
    """Factory honouring cfg['phy']['backend']: 'sionna' | 'analytical'."""
    kind = cfg["phy"]["backend"]
    if kind == "analytical":
        return AnalyticalPHY(cfg)
    if kind == "sionna":
        return SionnaPHY(cfg)
    raise ValueError(f"unknown PHY backend: {kind}")


# --------------------------------------------------------------------------
# Sionna link (only Sionna-touching code in the project)
# --------------------------------------------------------------------------

class SionnaLink:
    """One end-to-end Sionna link for a fixed MCS (mod order m, code rate r).

    LDPC info bits are fixed at k=512 (paper); n = k/r rounded to an integer
    number of QAM symbols that fits the 76x14 resource grid.
    Grid parameters marked UP-16..UP-20 in config/simulator.yaml are assumed,
    not paper-specified.
    """

    def __init__(self, mod_order, code_rate, cfg):
        import tensorflow as tf
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.phy.mapping import BinarySource, Mapper, Demapper
        from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper,
                                     LSChannelEstimator, LMMSEEqualizer)
        from sionna.phy.channel.tr38901 import CDL, Antenna
        from sionna.phy.channel import OFDMChannel
        from sionna.phy.mimo import StreamManagement
        self._tf = tf

        p = cfg["phy"]
        self.m = int(mod_order)
        self.r = float(code_rate)
        self.batch = int(p["batch_size"])
        self.k = int(p["ldpc_info_bits"])
        n_sym = int(np.ceil(self.k / self.r / self.m))
        self.n = n_sym * self.m

        self.rg = ResourceGrid(
            num_ofdm_symbols=int(p["num_ofdm_symbols"]),
            fft_size=int(p["fft_size"]),
            subcarrier_spacing=float(p["subcarrier_spacing"]),
            num_tx=1, num_streams_per_tx=1,
            cyclic_prefix_length=int(p["cyclic_prefix_length"]),      # UP-20
            pilot_pattern=p["pilot_pattern"],                          # UP-19
            pilot_ofdm_symbol_indices=list(p["pilot_ofdm_symbol_indices"]))
        if n_sym > int(self.rg.num_data_symbols):
            raise ValueError(f"MCS (m={self.m}, r={self.r}): {n_sym} symbols "
                             f"exceed grid capacity {self.rg.num_data_symbols}")

        fc = float(p["carrier_frequency"])                             # UP-16
        ut = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="omni", carrier_frequency=fc)
        bs = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="omni", carrier_frequency=fc)
        cdl = CDL("C", float(p["cdl_delay_spread"]), fc, ut, bs,       # UP-17
                  "downlink", min_speed=float(p["ue_speed_mps"]))      # UP-18

        self.source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.m)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.channel = OFDMChannel(cdl, self.rg, add_awgn=True,
                                   normalize_channel=True, return_channel=True)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.equalizer = LMMSEEqualizer(self.rg, StreamManagement(np.array([[1]]), 1))
        self.demapper = Demapper("app", "qam", self.m)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self._n_data_re = int(self.rg.num_data_symbols)
        self._run = tf.function(self._forward)

    def _forward(self, no):
        tf = self._tf
        b = self.source([self.batch, 1, 1, self.k])
        c = self.encoder(b)
        x = self.mapper(c)
        # zero-pad unused data REs
        pad = self._n_data_re - self.n // self.m
        x_full = tf.pad(tf.reshape(x, [self.batch, 1, 1, -1]),
                        [[0, 0], [0, 0], [0, 0], [0, pad]])
        x_rg = self.rg_mapper(x_full)
        y, h = self.channel(x_rg, no)
        h_hat, err_var = self.ls_est(y, no)
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        llr = tf.reshape(llr, [self.batch, 1, 1, -1])[..., :self.n]
        b_hat = self.decoder(llr)
        block_err = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
        return tf.reduce_mean(tf.cast(block_err, tf.float32))

    def bler(self, sinr_db, n_batches=1):
        """BLER at sinr_eff (= Es/N0 per RE, dB): no = 10^(-sinr/10)."""
        no = self._tf.constant(10.0 ** (-sinr_db / 10.0), self._tf.float32)
        return float(np.mean([self._run(no).numpy() for _ in range(n_batches)]))
