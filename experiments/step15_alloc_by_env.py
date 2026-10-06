"""E15 — allocation quality of a FROZEN H-SCM under environment shift (Phase 9b)

Separate model from Phase 1 (step14), and the distinction is load-bearing:

  step14 (generalization)  500 observational          -> zero-shot transfer
  step15 (allocation)      500 observational + 570 interventional

Both are fitted ONCE on NOMINAL-regime data and held fixed across all four
evaluation environments. Neither is refitted, re-tuned or re-discovered per
environment; the graph is read from results/step11_recovered_dag.yaml.

Why the allocation model gets the interventional rows. The scheduler evaluates
allocations near 10-30 PRB, while the 500 observational rows span 37-100 (1st
percentile 43). Scoring an observational-only fit there would confound TWO
failures -- environment shift and treatment-support violation -- so a bad
number could not be attributed. The 570 pre-existing interventional rows carry
PRB in {10, 50, 100} and bracket the scheduler's domain. They are part of the
original experimental design and were NOT generated after seeing any shift
result. --support-ablation reproduces the observational-only failure as a
deliberate, separately reported control.

FAIRNESS. Every learned method is fitted on exactly the same nominal pool and
reads the same telemetry at decision time. The MLP baseline (noncausal_mlp) is
a conditional-expectation predictor E[Y | T=k, X] over the observable
non-descendants of T, SMOOTH IN T so it can represent a UE-specific treatment
response, and it is CALIBRATED TO THE UE'S FACTUAL OUTCOME: its predicted
curve is rescaled so that the prediction at the factual allocation equals the
factual goodput. It therefore has access to everything the H-SCM abducts
from. The claim under test is a causal-structure advantage, not a
better-regressor or more-information advantage.

What the structural decomposition actually buys (and the claim to make): it
localizes where protocol knowledge can be imposed. mac_tput is proportional to
the treatment, so that ONE mechanism is fitted as T * g_hat(X) while every
other mechanism stays a flexible GBM. A direct T -> Y predictor has no such
seam: it must learn the whole treatment response end to end. This is NOT a
claim that a noncausal model cannot be made smooth in T -- it can, and
noncausal_mlp is.

hscm_gbm is retained as the function-class ablation: same graph, same data,
same propagation, tree-based MAC mechanism.

noncausal_mlp is one model of E[Y | T, X], queried at every candidate T and
rescaled by the UE's factual-to-predicted goodput ratio at its factual
allocation. Its single initialisation is re-fitted at four more seeds
(noncausal_mlp_rs43..rs46) so the reported interval covers training noise, not
only sampling noise -- "the baseline was one lucky or unlucky net" is then
answerable from the table.

flat_tprop closes the 2x2 that the claim above needs. hscm_struct differs from
the MLP in TWO ways at once: it has the SCM decomposition AND the
proportional-in-T form. flat_tprop has the form WITHOUT the structure: it fits
goodput = T * h(X) directly, with h fitted exactly as the SCM fits g for
mac_tput (unweighted GBM on the ratio, same GBM_PARAMS, T removed from X).

                     proportional in T      unrestricted in T
    SCM (structure)  hscm_struct            hscm_gbm
    flat (no graph)  flat_tprop             noncausal_gbm

The form is only exact where the physics puts it -- mac_tput. Goodput is NOT
proportional to T, because queueing loss depends on T through the service
rate. So if flat_tprop matches hscm_struct, the decomposition bought nothing
beyond the functional form; if hscm_struct wins, the gain is from imposing the
form only on the mechanism where it holds. Under the sum-goodput objective a
model linear in T gives the whole surplus above the floor to argmax h(X), so
flat_tprop can only allocate winner-take-all.

The oracle is never fitted: it queries the ground-truth simulator and is the
upper bound only.
"""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.counterfactual import CounterfactualEngine            # noqa: E402
from causal.fitting import GBM_PARAMS, StructuralModel, observable  # noqa: E402
from scheduler.causal_scheduler import (dp_allocate,              # noqa: E402
                                        equal_allocate,
                                        proportional_allocate)
from simulator.ran_simulator import COLUMNS, RANSimulator         # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Evaluation environments: each is a pure covariate shift of the root-variable
# distributions in config/simulator.yaml; no structural constant is touched.
# Each override is "<section>.<key>[.<subkey>] : value". Nominal values are in
# the comment so the size of every shift is visible in one place.
ENVIRONMENTS = {
    "E0_nominal": {},
    "E1_high_load": {                       # more congestion, burstier traffic
        "observational.cell_load.a": 5.0,           # Beta(2,2) -> Beta(5,2)
        "observational.cell_load.b": 2.0,           #   mean 0.50 -> 0.71
        "observational.traffic_burst.sigma2": 0.30,  # 0.12 -> 0.30 (burstier)
    },
    "E2_poor_channel": {                    # same supports, degraded half
        "observational.snr_db.high": 16.0,           # U(2,28) -> U(2,16)
        "observational.shadow_fading_db.var": 81.0,  # sd 7 -> 9 dB
        "observational.doppler_hz.low": 100.0,       # U(5,200) -> U(100,200)
        "observational.delay_spread_ns.low": 165.0,  # U(30,300) -> U(165,300)
    },
    "E3_heterogeneous": {                   # same means, much wider spread
        "observational.hw_impairment.var": 9.0,      # sd 1.5 -> 3.0
        "observational.shadow_fading_db.var": 100.0,  # sd 7 -> 10 dB
        "observational.traffic_burst.sigma2": 0.40,  # 0.12 -> 0.40
    },
}


def env_cfg(base, overrides):
    """Config overlay. Only root-variable distributions and lambda move."""
    cfg = copy.deepcopy(base)
    for path, val in overrides.items():
        node, *rest = path.split(".")
        d = cfg[node]
        for k in rest[:-1]:
            d = d[k]
        d[rest[-1]] = val
    return cfg

# Observable non-descendants of T, plus T. Nothing downstream of the treatment
# is used as a feature: conditioning on a mediator would be a different (and
# indefensible) error from the one this baseline is meant to represent.
NC_FEATURES = ["cell_load", "snr", "doppler", "delay_spread", "sinr_eff",
               "mcs_idx", "mod_order", "code_rate", "bler", "se_phy",
               "harq_retx", "num_prb"]
NC_ND = [c for c in NC_FEATURES if c != "num_prb"]   # T factored out
MLP_INITS = (43, 44, 45, 46)            # extra initialisations of the MLP
MLP_ARMS = ["noncausal_mlp"] + [f"noncausal_mlp_rs{r}" for r in MLP_INITS]
METHODS = (["oracle", "hscm_struct", "hscm_gbm", "noncausal_mlp",
            "noncausal_gbm", "flat_tprop", "proportional", "equal", "random"]
           + [f"noncausal_mlp_rs{r}" for r in MLP_INITS])


class TorchMLP:
    """MLP regressor in PyTorch, mirroring scikit-learn's MLPRegressor recipe:
    standardized inputs, two hidden layers of 64 ReLU units, Adam (lr 1e-3),
    L2 penalty alpha = 1e-4 on the weights (not the biases), mini-batches of
    200, at most 3000 epochs, and the same plateau rule (stop when the epoch
    loss has not improved by tol = 1e-4 for 10 consecutive epochs). Weights
    and biases start Glorot-uniform; `seed` fixes initialisation and shuffling."""

    def __init__(self, hidden=(64, 64), lr=1e-3, alpha=1e-4, batch_size=200,
                 max_iter=3000, tol=1e-4, n_iter_no_change=10, seed=42):
        self.hidden, self.lr, self.alpha = hidden, lr, alpha
        self.batch_size, self.max_iter = batch_size, max_iter
        self.tol, self.n_iter_no_change, self.seed = tol, n_iter_no_change, seed

    def fit(self, X, y):
        torch.set_num_threads(2)
        g = torch.Generator().manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xt = torch.from_numpy((X - self.mu_) / self.sd_); yt = torch.from_numpy(y)
        sizes = [X.shape[1], *self.hidden, 1]; layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lin = torch.nn.Linear(a, b); bound = float(np.sqrt(6.0 / (a + b)))
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.uniform_(-bound, bound, generator=g)
            layers.append(lin)
            if i < len(self.hidden): layers.append(torch.nn.ReLU())
        self.net_ = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        n = len(Xt); bs = min(self.batch_size, n); best, stale = np.inf, 0
        weights = [m.weight for m in self.net_ if isinstance(m, torch.nn.Linear)]
        for _ in range(self.max_iter):
            perm = torch.randperm(n, generator=g); total = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                pred = self.net_(Xt[idx]).squeeze(1)
                loss = 0.5 * torch.mean((pred - yt[idx]) ** 2) \
                    + 0.5 * self.alpha * sum((w ** 2).sum() for w in weights) / n
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            epoch_loss = total / n
            if epoch_loss > best - self.tol: stale += 1
            else: stale = 0
            best = min(best, epoch_loss)
            if stale >= self.n_iter_no_change: break
        self.n_iter_ = _ + 1
        return self

    def predict(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        with torch.no_grad():
            return self.net_(torch.from_numpy(X)).squeeze(1).numpy().astype(float)


class TProportional:
    """goodput = T * h(X), fitted the way the SCM fits g for mac_tput."""

    def fit(self, Xnd, T, y):
        self.h = GradientBoostingRegressor(**GBM_PARAMS).fit(
            Xnd, y / np.maximum(T, 1e-9))
        return self

    def predict(self, Xnd, T):
        return np.asarray(T, dtype=float) * self.h.predict(Xnd)


def training_pool(with_interventional):
    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    if not with_interventional:
        return obs, obs
    intv = pd.read_csv(os.path.join(ROOT, "data", "interventional.csv"))
    return obs, pd.concat([obs, intv], ignore_index=True)


def frozen_models(with_interventional=True):
    """Fit every learned method ONCE, on the same nominal pool."""
    obs, pool = training_pool(with_interventional)
    edges = [tuple(e) for e in yaml.safe_load(open(os.path.join(
        ROOT, "results", "step11_recovered_dag.yaml")))["recovered_edges"]]
    X = pool[NC_FEATURES].values.astype(float)
    y = pool.goodput.values

    def scm(factored):
        sm = StructuralModel(ROOT, mode="telemetry", edges=edges,
                             nodes=observable(COLUMNS),
                             factored=factored).fit(pool, cv_folds=5)
        return CounterfactualEngine(sm, ROOT, edges=edges, nodes=COLUMNS)

    return {
        "hscm_struct": scm(True),          # MAC mechanism = T * g_hat(X)
        "hscm_gbm": scm(False),            # function-class ablation
        # smooth in T, UE-specific response, no structural knowledge
        "noncausal_mlp": TorchMLP(seed=42).fit(X, y),
        "noncausal_gbm": GradientBoostingRegressor(**GBM_PARAMS).fit(X, y),
        # the proportional form WITHOUT the graph: the missing 2x2 cell
        "flat_tprop": TProportional().fit(
            pool[NC_ND].values.astype(float), pool.num_prb.values, y),
        # the same MLP at four more initialisations
        **{f"noncausal_mlp_rs{r}": TorchMLP(seed=r).fit(X, y) for r in MLP_INITS},
    }, pool


def random_allocate(rng, K, budget, k_min, k_max):
    a = np.full(K, k_min, dtype=int)
    for _ in range(int(budget) - K * k_min):
        j = rng.integers(K)
        if a[j] < k_max:
            a[j] += 1
    return a


def one_cell(sim, M, cfg, rng, n_ues, k_min, budget_mode="load"):
    o = cfg["observational"]["cell_load"]
    u = sim.sample_exogenous(n_ues, rng)
    L = float(rng.beta(o["a"], o["b"]))
    u["cell_load"] = np.full(n_ues, L)          # one cell -> one load state

    base = int(cfg["mac"]["base_prb"])
    if budget_mode == "fixed":
        budget = base                      # whole carrier, independent of L
    else:                                  # eq. (2) without the scheduling noise
        budget = int(np.floor(base * (1.0 - float(cfg["mac"]["alpha_L"]) * L)))
    budget = max(budget, n_ues * k_min)
    k_max = min(budget - (n_ues - 1) * k_min, base)
    if k_max < k_min:
        return None

    # the incumbent allocation the cell is actually running
    fact_alloc = equal_allocate(n_ues, budget, k_min, k_max)
    fact = pd.concat(
        [sim.propagate({k: v[[i]] for k, v in u.items()},
                       do={"num_prb": int(fact_alloc[i])})
         for i in range(n_ues)], ignore_index=True)
    def util(fn):
        t = np.full((n_ues, k_max + 1), -np.inf)
        for k in range(k_min, k_max + 1):
            t[:, k] = fn(k)
        return t

    def flat(m, k):
        f = fact.copy(); f["num_prb"] = k
        return m.predict(f[NC_FEATURES].values.astype(float))

    tables = {"oracle": util(
        lambda k: sim.propagate(u, do={"num_prb": int(k)}).goodput.values)}
    for name in ("hscm_struct", "hscm_gbm"):
        eng = M[name]
        res = eng.abduct(fact)
        tables[name] = util(
            lambda k, e=eng, r=res: e.predict(
                fact, {"num_prb": int(k)}, r).goodput.values)
    tables["noncausal_gbm"] = util(lambda k, m=M["noncausal_gbm"]: flat(m, k))
    # the MLP is calibrated to the UE's factual outcome: its predicted curve is
    # rescaled so that the prediction at the factual allocation equals the
    # factual goodput, which gives it the same post-allocation information the
    # H-SCM abducts from (without a graph or mechanism-wise residuals)
    y_fact = fact.goodput.values
    for name in MLP_ARMS:
        yhat_fact = M[name].predict(fact[NC_FEATURES].values.astype(float))
        ratio = y_fact / np.maximum(yhat_fact, 0.5)
        tables[name] = util(lambda k, m=M[name], r=ratio: flat(m, k) * r)
    hx = M["flat_tprop"].h.predict(fact[NC_ND].values.astype(float))
    tables["flat_tprop"] = util(lambda k: k * hx)

    alloc = {n: dp_allocate(t, budget, k_min, k_max) for n, t in tables.items()}
    alloc["proportional"] = proportional_allocate(fact.goodput.values, budget,
                                                  k_min, k_max)
    alloc["equal"] = fact_alloc
    alloc["random"] = random_allocate(rng, n_ues, budget, k_min, k_max)
    # every policy is EXECUTED in the ground-truth simulator; J is realized,
    # never the utility the policy believed it would get
    out = {"budget": budget, "cell_load": L}
    for name, a in alloc.items():
        real = pd.concat(
            [sim.propagate({k: v[[i]] for k, v in u.items()},
                           do={"num_prb": int(a[i])})
             for i in range(n_ues)], ignore_index=True)
        out[f"J_{name}"] = float(real.goodput.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--ues", type=int, default=4)
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--budget", choices=["load", "fixed"], default="load",
                    help="cell PRB budget: floor(N_PRB(1 - alpha_L L)) from "
                         "eq. (2) (default) or the whole carrier N_PRB")
    ap.add_argument("--tag", default="",
                    help="suffix for results/step15_alloc_by_env<tag>.csv")
    ap.add_argument("--support-ablation", action="store_true",
                    help="fit on the 500 observational rows ONLY, to expose "
                         "the low-PRB treatment-support failure")
    a = ap.parse_args()

    M, pool = frozen_models(not a.support_ablation)
    print(f"FROZEN once on nominal data: {len(pool)} rows "
          f"({'500 obs + 570 interventional' if not a.support_ablation else '500 obs ONLY'})")
    print(f"  PRB support in the pool: {pool.num_prb.min()}-{pool.num_prb.max()}"
          f"; scheduler domain: {a.kmin}-...\n")

    base = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    rows = []
    for env, ov in ENVIRONMENTS.items():
        cfg = env_cfg(base, ov)
        sim = RANSimulator(cfg)
        for s in range(a.seeds):
            rng = np.random.default_rng(58_000 + 131 * s)   # final seeds,
            # disjoint from the pilot stream that guided estimator choice
            per = [one_cell(sim, M, cfg, rng, a.ues, a.kmin, a.budget)
                   for _ in range(a.cells)]
            per = [p for p in per if p]
            d = pd.DataFrame(per)
            r = {"env": env, "seed": s, "n_cells": len(d)}
            for m in METHODS:
                r[f"J_{m}"] = d[f"J_{m}"].mean()
                r[f"regret_{m}"] = float(
                    ((d.J_oracle - d[f"J_{m}"]) / d.J_oracle).mean())
            rows.append(r)
    df = pd.DataFrame(rows)
    tag = a.tag or ("_obsonly" if a.support_ablation else "")
    out = os.path.join(ROOT, "results", f"step15_alloc_by_env{tag}.csv")
    df.to_csv(out, index=False)

    print(f"realized cell goodput J (Mbps, sum over {a.ues} UEs) and oracle "
          f"regret, {a.seeds} seeds x {a.cells} cells")
    for env in ENVIRONMENTS:
        e = df[df.env == env]
        print(f"\n  {env}")
        for m in METHODS:
            if m.startswith("noncausal_mlp_rs"):
                continue
            print(f"    {m:<13} J {e[f'J_{m}'].mean():6.2f} +/- "
                  f"{e[f'J_{m}'].std():4.2f}    regret "
                  f"{100*e[f'regret_{m}'].mean():6.2f}% +/- "
                  f"{100*e[f'regret_{m}'].std():4.2f}%")
        inits = [100 * e[f"regret_noncausal_mlp{t}"].mean()
                 for t in [""] + [f"_rs{r}" for r in MLP_INITS]]
        print(f"    MLP (factual-calibrated) over {len(inits)} initialisations: regret "
              f"{min(inits):.2f}% .. {max(inits):.2f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
