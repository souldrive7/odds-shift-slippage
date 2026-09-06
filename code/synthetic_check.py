"""Synthetic verification of route F theory against ground truth.

Called from run_experiment.py --part synthetic.

Generator: x ~ N(0, I_d); true logit z_j = a_j + b_j . x with per-label intercepts spanning
`prev_decades` orders of magnitude of prevalence; y_j ~ Bernoulli(sigmoid(z_j)).

Checks:
 C1 (identity, realizable): weighted logistic regression (class weight w_j on positives) recovers
     logit(q) = logit(p) + ln w_j up to estimation error; Elkan inversion restores the unweighted
     ranking; ranking by raw weighted scores loses MAP@K relative to ranking by true p, and the
     loss grows with the spread of ln w_j (weights applied with exponent e on r_j).
 C2 (saturation, capacity-limited): LightGBM with small capacity under the same weights; report
     distinct scores per label, fraction saturated, oracle ceiling, and whether inversion still works.
 C3 (Bayes optimality): MAP@K of ranking by true p is the maximum over all tested scorers.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-15


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def map_at_k(y, s, k):
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    return float(((prec * rel).sum(1) / np.minimum(y.sum(1), k)).mean())


def oracle_ceiling(q, y):
    out = np.zeros_like(q)
    for j in range(q.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            out[:, j] = IsotonicRegression(out_of_bounds="clip").fit(q[:, j], y[:, j].astype(float)).predict(q[:, j])
        else:
            out[:, j] = y[:, j].mean()
    return out


def gen_params(L, d, prev_decades, rng):
    b = rng.normal(0, 1, size=(L, d)) / np.sqrt(d)
    prev = 10 ** rng.uniform(-prev_decades, 0, size=L) * 0.3  # prevalence between 0.3*10^-decades and 0.3
    a = logit(prev) - 0.5 * (b**2).sum(1)  # rough centering so mean sigmoid ~ prev
    return a, b


def gen(n, params, d, rng):
    """Sample (x, true p, y) from a FIXED label model `params` (train and test share it)."""
    a, b = params
    x = rng.normal(0, 1, size=(n, d))
    z = a + x @ b.T
    p = sigmoid(z)
    y = (rng.uniform(size=p.shape) < p).astype(np.int8)
    return x, p, y


def fit_weighted_logreg(x_tr, y_tr, x_te, w):
    L = y_tr.shape[1]
    q = np.zeros((len(x_te), L))
    for j in range(L):
        yj = y_tr[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            q[:, j] = yj.mean()
            continue
        sw = np.where(yj == 1, w[j], 1.0)
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(x_tr, yj, sample_weight=sw)
        q[:, j] = clf.predict_proba(x_te)[:, 1]
    return q


def fit_weighted_lgbm(x_tr, y_tr, x_te, w, n_estimators, num_leaves, min_child):
    from lightgbm import LGBMClassifier

    L = y_tr.shape[1]
    q = np.zeros((len(x_te), L))
    for j in range(L):
        yj = y_tr[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            q[:, j] = yj.mean()
            continue
        clf = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.1, num_leaves=num_leaves, min_child_samples=min_child,
            scale_pos_weight=float(w[j]), random_state=42, verbosity=-1,
        ).fit(x_tr, yj)
        q[:, j] = clf.predict_proba(x_te)[:, 1]
    return q


def run(seeds=(0, 1, 2), n_tr=20000, n_te=20000, L=20, d=8, k=5, prev_decades_list=(1.0, 2.0, 3.0)):
    from learners import mlp_posweight, xgb_ovr

    t0 = time.time()
    exps = [0.0, 0.5, 1.0, 1.5]
    out = {"config": dict(seeds=list(seeds), n_tr=n_tr, n_te=n_te, L=L, d=d, k=k, prev_decades_list=list(prev_decades_list), weight_exponents=exps), "cells": []}
    for prev_decades in prev_decades_list:
      for seed in seeds:
        rng = np.random.default_rng(seed)
        params = gen_params(L, d, prev_decades, rng)
        x_tr, p_tr, y_tr = gen(n_tr, params, d, rng)
        x_te, p_te, y_te = gen(n_te, params, d, rng)
        n_pos = y_tr.sum(0).astype(float)
        r = (len(y_tr) - n_pos) / np.maximum(n_pos, 1)
        base = {"seed": seed, "prev_decades": prev_decades, "max_r": float(r.max()), "map_true_p": map_at_k(y_te, p_te, k), "map_popularity": map_at_k(y_te, np.tile(y_tr.mean(0), (n_te, 1)), k)}
        for e in exps:
            w = r**e
            cell = dict(base, weight_exponent=e, ln_w_spread=float(np.log(w).max() - np.log(w).min()))
            # extra learners (odds shift is learner-agnostic; saturation is learner-specific)
            for lname, fn in {"xgb": lambda: xgb_ovr(x_tr, y_tr, [x_te], w)[0], "mlp_posweight": lambda: mlp_posweight(x_tr, y_tr, [x_te], w, epochs=40)[0]}.items():
                q3 = fn()
                qc3 = np.clip(q3, EPS, 1 - EPS)
                cell[lname] = {
                    "map_raw": map_at_k(y_te, q3, k),
                    "map_elkan_inversion": map_at_k(y_te, qc3 / (qc3 + w * (1 - qc3)), k),
                    "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q3, y_te), k),
                    "distinct_scores_per_label": float(np.mean([len(np.unique(q3[:, j])) for j in range(L)])),
                    "sat_ge_1m1e9": float((q3 >= 1 - 1e-9).mean()),
                }
            # C1 realizable learner
            q = fit_weighted_logreg(x_tr, y_tr, x_te, w)
            qc = np.clip(q, EPS, 1 - EPS)
            inv = qc / (qc + w * (1 - qc))
            cell["logreg"] = {
                "map_raw": map_at_k(y_te, q, k),
                "map_elkan_inversion": map_at_k(y_te, inv, k),
                "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q, y_te), k),
                "mean_abs_logit_shift_error": float(np.mean(np.abs((logit(q) - logit(p_te)) - np.log(w)))),
                "sat_ge_1m1e9": float((q >= 1 - 1e-9).mean()),
            }
            # C2 capacity-limited learner (small trees, few rounds)
            for cap_name, cap in {"lgbm_small": (30, 4, 20), "lgbm_large": (300, 31, 20)}.items():
                q2 = fit_weighted_lgbm(x_tr, y_tr, x_te, w, *cap)
                qc2 = np.clip(q2, EPS, 1 - EPS)
                cell[cap_name] = {
                    "map_raw": map_at_k(y_te, q2, k),
                    "map_elkan_inversion": map_at_k(y_te, qc2 / (qc2 + w * (1 - qc2)), k),
                    "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q2, y_te), k),
                    "distinct_scores_per_label": float(np.mean([len(np.unique(q2[:, j])) for j in range(L)])),
                    "sat_ge_1m1e9": float((q2 >= 1 - 1e-9).mean()),
                    "sat_le_1e9": float((q2 <= 1e-9).mean()),
                }
            out["cells"].append(cell)
            print(f"seed {seed} e={e}: true {base['map_true_p']:.3f} | logreg raw {cell['logreg']['map_raw']:.3f} inv {cell['logreg']['map_elkan_inversion']:.3f} | small raw {cell['lgbm_small']['map_raw']:.3f} inv {cell['lgbm_small']['map_elkan_inversion']:.3f} ceil {cell['lgbm_small']['map_oracle_ceiling']:.3f} distinct {cell['lgbm_small']['distinct_scores_per_label']:.0f}", flush=True)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    return out
