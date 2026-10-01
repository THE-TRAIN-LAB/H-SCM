"""Structural-equation fitting (paper §IV-C, eq. 16) and Table I.

Observability interpretation (UP-25, binding)
---------------------------------------------
The paper states (§II-B/II-C) that the per-unit effect modifiers do NOT
appear in standard telemetry. We therefore distinguish:

  observable telemetry : the 15 endogenous nodes + cell_load (O-RAN PRB
                         occupancy per TTI, paper's R1) + snr, doppler,
                         delay_spread (channel-estimator statistics)
  latent               : shadow_fading, hw_impairment, traffic_burst,
                         eps_sinr, eps_sched, eps_queue

Two fitting modes:

  mode="oracle"    : features = ground-truth parents minus the eps_* noise
                     nodes. This is what the paper's Table I must have used —
                     sinr_eff R^2 = 0.999 is only achievable with xi_sf as a
                     feature (var(xi_sf) = 49 of ~135 total). Used ONLY to
                     reproduce Table-I-style fit quality.
  mode="telemetry" : features = ground-truth parents minus ALL latents. This
                     is what the causal pipeline (Phases 6-7) is allowed to
                     use. The per-node residuals now CONTAIN the latent
                     fingerprint (e.g. sinr_eff residual = xi_sf + eps_1) —
                     abduction recovers them instead of reading them.

Residual kinds (Algorithm 1 abduction is NOT blindly additive)
--------------------------------------------------------------
additive       x = f(pa) + u        -> u_hat = x - f_hat(pa)
multiplicative x = f(pa) * u        -> u_hat = x / f_hat(pa)
               (mac_tput: the latent enters as (1 + kappa_hw * xi_hw))
carry          no telemetry parents: the factual value is itself the
               abducted state and counterfactuals carry it unchanged.
               CURRENTLY UNUSED — every endogenous node in the ground-truth
               graph has at least one observable parent. Kept because the kind
               is a property of the residual convention, not of one node: a
               RECOVERED graph can leave an endogenous node parentless.

Deterministic lookups (mcs/mod/cr/bler/se_phy/harq) have residuals
~ 0 under either convention; they are treated as additive.

Cross-fitting
-------------
fit_crossfit() trains K per-fold models so that every unit's abducted
residual (and its counterfactual prediction in Phase 7) comes from a model
that never saw that unit — removing the optimism visible in
num_prb (in-sample R^2 0.973 vs CV 0.929).
"""
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

NOISE_NODES = {"eps_sinr", "eps_sched", "eps_queue"}
LATENT_NODES = {"shadow_fading", "hw_impairment", "traffic_burst"} | NOISE_NODES


def observable(columns):
    """The telemetry-visible subset of `columns` (UP-25, UP-33).

    THE single definition of observability, consumed by fitting AND by
    discovery. Before UP-33 only the fitting side honoured it: discovery ran
    PC and the interventional augmentation on all 23 simulator nodes,
    including the six declared latent here, so its accuracy was a
    full-simulator-state result rather than one obtainable from a real RAN.
    """
    return [c for c in columns if c not in LATENT_NODES]


def observable_edges(edges):
    """Ground-truth edges whose BOTH endpoints are telemetry-visible.

    Discovery is scored against this subgraph. Scoring against the full DAG
    would count edges as missed that no deployment could measure; the excluded
    edges are reported separately rather than silently dropped.
    """
    return [e for e in edges if e[0] not in LATENT_NODES
            and e[1] not in LATENT_NODES]
GBM_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  random_state=42)
NODE_LAYER_NAME = {1: "L1", 2: "L2", 3: "L3"}

RESIDUAL_KIND = {"mac_tput": "multiplicative",
                 "queue_dep": "scaled",
                 "goodput": "identity"}
DEFAULT_KIND = "additive"

# Nodes whose structural equation is an EXACT identity in observable parents:
# no noise term and no free constants, so there is nothing to fit and nothing
# to abduct. goodput is a definition -- delivered rate = service rate x success
# fraction -- and it holds in the data to 4e-14.
#
# Fitting it separately and then re-adding a residual is what let Y escape
# [0, mu]: the residual is pure model error, and model error added to a
# definition is not a counterfactual, it is a broken identity. Computing it
# instead makes 0 <= Y <= mac_tput automatic whenever pkt_loss is in [0, 1].
IDENTITY = {
    "goodput": lambda d: (np.asarray(d["mac_tput"], dtype=float)
                          * (1.0 - np.asarray(d["pkt_loss"], dtype=float))),
}

# Mechanisms whose structural form is PROPORTIONAL to the treatment, and the
# parent that carries it. From the MAC service rate of the system model,
#
#   mac_tput = C0 * num_prb * se_phy * (1 + kappa_hw xi_hw)
#                                    / (1 + kappa_HARQ N_retx)
#
# the treatment enters as a pure multiplicative factor, so
#
#   mac_tput / num_prb = g(se_phy, N_retx, ...)
#
# is free of T. Fitting g and reconstructing T * g_hat(X) forces the treatment
# response to obey the physically justified shape while leaving the
# context-dependent slope g(X) fully data-driven -- and it keeps the slope
# UE-SPECIFIC, which a single global linear-in-T term would not.
#
# Opt-in (StructuralModel(..., factored=True)): the default path is unchanged,
# so every previously reported result stands.
TREATMENT_FACTOR = {"mac_tput": "num_prb"}

# Physical support of each variable, as (lo, hi); None means unbounded.
# rtt_ms's floor is tau0 and comes from config, so the engine supplies it.
#
# This is a GUARD, not a mechanism. Where a projection actually binds, the
# prediction was already wrong and clipping only hides how wrong -- so the
# engine can report what bound, and the tests assert it does not.
SUPPORT = {"queue_dep": (0.0, None),      # [.]_+ in eq. (9)
           "pkt_loss": (0.0, 1.0),        # a probability
           "bler": (0.0, 1.0),            # a probability
           "harq_retx": (0.0, None),      # a count of retransmissions
           "se_phy": (0.0, None),         # bits/symbol
           "mac_tput": (0.0, None),       # a rate
           "goodput": (0.0, None)}        # upper bound is mac_tput, via IDENTITY

# what the telemetry-mode residual of each node physically abducts
ABDUCTS = {
    "sinr_eff": "xi_sf + eps_1 (the shadow-fading fingerprint)",
    "num_prb": "eps_T (scheduler jitter)",
    "mac_tput": "(1 + kappa_hw * xi_hw) (device impairment, multiplicative)",
    "queue_dep": "eps_3 AND a(pa)*(xi_tb - m) (both latent; UP-28). The two "
                 "are not separately identified from Q and pa, so they are "
                 "split by posterior share, not assumed away (UP-35).",
}


class _ConstantModel:
    """Mean predictor for nodes whose telemetry parent set is empty."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        ss_res = np.sum((y - self.mean_) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-12)


def load_parent_sets(root, mode, edges=None, nodes=None):
    """Parent sets for the SCM.

    edges=None reads the ground-truth DAG, which is what Table I and the
    oracle-validation experiments want. Passing a RECOVERED edge list instead
    fits the SCM on the graph discovery actually produced -- the honest
    end-to-end setting, where a missed edge shows up as a missing feature and
    is absorbed into the abducted residual rather than modelled.

    nodes, when given, guarantees every listed node has an entry (possibly an
    empty parent set). A recovered graph need not give every endogenous node a
    parent, and a node absent from this dict is never fitted and never
    recomputed during counterfactual propagation.
    """
    if edges is None:
        with open(os.path.join(root, "config", "dag_edges.yaml")) as f:
            edges = [tuple(e) for e in yaml.safe_load(f)["edges"]]
    drop = NOISE_NODES if mode == "oracle" else LATENT_NODES
    parents = {}
    for a, b in edges:
        parents.setdefault(b, []).append(a)
    out = {c: sorted(p for p in ps if p not in drop)
           for c, ps in parents.items()}
    if nodes is not None:
        for c in nodes:
            out.setdefault(c, [])
    return out


def residual_kind(node):
    return RESIDUAL_KIND.get(node, DEFAULT_KIND)


def fit_scale_params(f, u, censored):
    """Split Var(u | pa) into a mean-scaling part and a flat part (UP-35).

    The queue mechanism is Q = [a(pa) * xi_tb + eps_Q]_+ , so its residual is

        u = a(pa) * (xi_tb - m) + eps_Q ,     a(pa) = E[Q | pa] / m

    -- one component proportional to the fitted mean and one that is not.
    Carrying u unchanged through an intervention (the additive rule) moves
    a(pa) but leaves the traffic term scaled by the OLD utilization, which is
    wrong in both directions and can flip the sign of the prediction.

    Because a(pa) is proportional to f_hat, regressing u^2 on f_hat^2
    identifies both variances without ever naming a(pa) or lambda:

        E[u^2 | pa] = zeta2 * f_hat^2 + sigma2_eps

    Returns (zeta2, sigma2_eps). Censored rows -- Q clipped at 0, where
    u = -f_hat exactly and is not a noise draw -- are excluded rather than
    allowed to inflate zeta2.
    """
    keep = ~np.asarray(censored, dtype=bool)
    f, u = np.asarray(f, float), np.asarray(u, float)
    if int(keep.sum()) < 8:                    # too little to identify a split
        return 0.0, float(np.var(u)) + 1e-12
    A = np.column_stack([f[keep] ** 2, np.ones(int(keep.sum()))])
    (zeta2, s2), *_ = np.linalg.lstsq(A, u[keep] ** 2, rcond=None)
    return max(float(zeta2), 0.0), max(float(s2), 1e-12)


def _new_model(parents):
    """`parents` may be a list or a feature count; both mean "any features?"."""
    n = parents if isinstance(parents, int) else len(parents or [])
    return GradientBoostingRegressor(**GBM_PARAMS) if n else _ConstantModel()


class StructuralModel:
    """Fitted SCM {f_hat_i} in 'oracle' or 'telemetry' mode.

    fit()          full-data models + in-sample and K-fold-CV R^2 (Table I)
    fit_crossfit() per-fold models; every unit gets out-of-fold abduction
                   and prediction (Phase 7 uses this)
    residuals()    kind-aware abduction (additive / multiplicative / carry)
    predict_node() f_hat_node on a frame (full-data or per-fold model)
    """

    def __init__(self, root, mode="telemetry", edges=None, nodes=None,
                 factored=False):
        assert mode in ("oracle", "telemetry")
        self.mode = mode
        # factored=True imposes mac_tput = num_prb * g(X); see TREATMENT_FACTOR
        self.factored = factored
        self.edges = edges
        self.parents = load_parent_sets(root, mode, edges=edges, nodes=nodes)
        self.models = {}
        self.fold_models = {}
        self.fold_of = None
        self.r2_in = {}
        self.r2_cv = {}
        # (zeta2, sigma2_eps) per 'scaled' node -- see fit_scale_params
        self.scale = {}
        self.fold_scale = {}

    def _factor_of(self, node):
        """The treatment parent factored out of `node`, or None."""
        if not self.factored:
            return None
        f = TREATMENT_FACTOR.get(node)
        return f if f is not None and f in self.parents[node] else None

    def _design(self, node, frame):
        """(X, factor_values) for `node`: the factor is removed from X."""
        pa = self.parents[node]
        f = self._factor_of(node)
        cols = [c for c in pa if c != f] if f else pa
        X = (frame[cols].values.astype(float) if cols
             else np.zeros((len(frame), 1)))
        return X, (frame[f].values.astype(float) if f else None)

    # -- full-data fitting (Table I) ------------------------------------
    def fit(self, obs, cv_folds=5, nodes=None):
        for node, pa in self.parents.items():
            if nodes is not None and node not in nodes:
                continue
            X, fac = self._design(node, obs)
            y = obs[node].values.astype(float)
            if fac is not None:                 # fit g = y / T, not y
                y = y / np.maximum(fac, 1e-9)
            m = _new_model(X.shape[1] if X.size else 0).fit(X, y)
            self.models[node] = m
            self.r2_in[node] = m.score(X, y)
            press, ss = 0.0, np.sum((y - y.mean()) ** 2)
            oof = np.empty(len(y))
            for tr, te in KFold(cv_folds, shuffle=True,
                                random_state=42).split(X):
                mm = _new_model(X.shape[1] if X.size else 0).fit(X[tr], y[tr])
                oof[te] = mm.predict(X[te])
                press += np.sum((y[te] - oof[te]) ** 2)
            self.r2_cv[node] = 1.0 - press / max(ss, 1e-12)
            if residual_kind(node) == "scaled":
                # OUT-OF-FOLD, not in-sample. A 200-tree GBM on 500 rows fits
                # queue_dep to R^2 0.977 in sample against 0.756 cross-
                # validated, so in-sample residuals are mostly memorization
                # and understate zeta2 by ~8x. The variance split has to be
                # estimated from residuals the model did not fit.
                self.scale[node] = fit_scale_params(oof, y - oof, y <= 1e-12)
        return self

    # -- cross-fitting (honest abduction, Phase 7) ----------------------
    def fit_crossfit(self, obs, folds=5, nodes=None):
        kf = KFold(folds, shuffle=True, random_state=42)
        self.fold_of = np.empty(len(obs), dtype=int)
        splits = list(kf.split(obs))
        for k, (_, te) in enumerate(splits):
            self.fold_of[te] = k
        for node, pa in self.parents.items():
            if nodes is not None and node not in nodes:
                continue
            X, fac = self._design(node, obs)
            y = obs[node].values.astype(float)
            if fac is not None:
                y = y / np.maximum(fac, 1e-9)
            self.fold_models[node] = [
                _new_model(X.shape[1] if X.size else 0).fit(X[tr], y[tr])
                for tr, _ in splits]
            if residual_kind(node) == "scaled":
                # Estimated once, from the out-of-fold residuals the fold
                # models already produce -- every unit predicted by a model
                # that did not train on it. A per-fold split would be noisier
                # (two scalars from N/5 rows) than the leakage it avoids.
                oof = np.empty(len(y))
                for mdl, (_, te) in zip(self.fold_models[node], splits):
                    oof[te] = mdl.predict(X[te])
                par = fit_scale_params(oof, y - oof, y <= 1e-12)
                self.fold_scale[node] = [par] * len(splits)
        return self

    # -- residual variance split (UP-35) ---------------------------------
    def scale_weight(self, node, f, fold_of=None):
        """Posterior share of u that scales with the fitted mean, given f_hat.

            w = zeta2 f^2 / (zeta2 f^2 + sigma2_eps)

        w -> 1 where the mechanism's scaling term dominates and the residual
        should be rescaled by an intervention; w -> 0 where flat noise
        dominates and it should be carried unchanged. This also regularizes
        itself: as f_hat -> 0 the rescaling ratio f'/f becomes unstable, and w
        goes to 0 there on its own, so the unstable term is never used.
        """
        f = np.asarray(f, float)
        if fold_of is None:
            zeta2, s2 = self.scale.get(node, (0.0, 1.0))
            v = zeta2 * f ** 2
            return v / (v + s2)
        out = np.zeros(len(f))
        for k, (zeta2, s2) in enumerate(self.fold_scale[node]):
            sel = np.asarray(fold_of) == k
            if sel.any():
                v = zeta2 * f[sel] ** 2
                out[sel] = v / (v + s2)
        return out

    # -- prediction ------------------------------------------------------
    def predict_node(self, node, frame, fold_of=None):
        X, fac = self._design(node, frame)
        if fold_of is None:
            out = self.models[node].predict(X)
        else:
            out = np.empty(len(frame))
            for k, m in enumerate(self.fold_models[node]):
                sel = fold_of == k
                if sel.any():
                    out[sel] = m.predict(X[sel])
        # reconstruct mac_tput = T * g_hat(X): the treatment response is
        # forced to the structural shape, the slope stays UE-specific
        return out * fac if fac is not None else out

    # -- abduction (Algorithm 1 step 1), kind-aware ----------------------
    def residuals(self, frame, fold_of=None):
        """u_hat per node. fold_of != None -> out-of-fold (cross-fitted)."""
        out = {}
        nodes = self.fold_models if fold_of is not None else self.models
        for node in nodes:
            y = frame[node].values.astype(float)
            kind = residual_kind(node)
            if kind == "identity":
                # An exact definition has no noise to abduct. Returning zeros
                # rather than dropping the key keeps every consumer's residual
                # dict the same shape.
                out[node] = np.zeros(len(y))
                continue
            if kind == "carry":
                out[node] = y.copy()
                continue
            f = self.predict_node(node, frame, fold_of=fold_of)
            if kind == "multiplicative":
                out[node] = y / np.maximum(f, 1e-9)
            else:
                out[node] = y - f
        return out

    # -- Table I ----------------------------------------------------------
    def table1(self, node_layers):
        rows = []
        for node in self.models:
            rows.append({"variable": node,
                         "layer": NODE_LAYER_NAME[node_layers[node]],
                         "n_parents": len(self.parents[node]),
                         "residual_kind": residual_kind(node),
                         "r2_insample": round(self.r2_in[node], 3),
                         "r2_cv5": round(self.r2_cv[node], 3)})
        return (pd.DataFrame(rows)
                .sort_values(["layer", "variable"])
                .reset_index(drop=True))
