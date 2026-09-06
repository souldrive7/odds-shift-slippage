"""MULAN dose-response across learners: how per-label positive-class weight magnitude damages
top-K, what the ideal odds-shift theory predicts (what-if from the unweighted model), and what repairs it.

Called from run_experiment.py --part mulan. ARFF fetch/parse helpers live in mulan_io.py.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import mulan_io
from learners import LEARNERS

EPS = 1e-15
CAL_SEED = 42
CAL_FRAC = 0.3
DATASETS = ["emotions", "scene", "flags", "birds", "yeast", "genbase", "enron", "medical"]
WEIGHT_SCHEMES = {"w=1": (0.0, 1.0), "w=sqrt(r)": (0.5, 1.0), "w=r": (1.0, 1.0), "w=10r": (1.0, 10.0)}


def _load_p4():
    """Kept for call-site compatibility: the MULAN I/O module (fetch, parse, provenance)."""
    return mulan_io


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def map_at_k(y, s, k):
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    return float(((prec * rel).sum(1) / np.minimum(y.sum(1), k)).mean())


def per_label_iso(q_cal, y_cal, q_ev, prior, policy):
    out = q_ev.copy()
    n_dead = 0
    for j in range(q_ev.shape[1]):
        yv = y_cal[:, j]
        if yv.sum() == 0 or yv.sum() == len(yv):
            n_dead += 1
            if policy == "prior":
                out[:, j] = prior[j]
            elif policy == "exclude":
                out[:, j] = -np.inf
            continue
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9).fit(q_cal[:, j], yv.astype(float))
        out[:, j] = m.predict(q_ev[:, j])
    return out, n_dead


def pooled_iso(q_cal, y_cal, q_ev):
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9).fit(q_cal.ravel(), y_cal.ravel().astype(float))
    return m.predict(q_ev.ravel()).reshape(q_ev.shape)


def oracle_ceiling(q, y):
    out = np.zeros_like(q)
    for j in range(q.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            out[:, j] = IsotonicRegression(out_of_bounds="clip").fit(q[:, j], y[:, j].astype(float)).predict(q[:, j])
        else:
            out[:, j] = y[:, j].mean()
    return out


def prior_match_shift(q_ev, q_ref, pi_train, hi=40.0):
    z_ref = logit(q_ref)
    b = np.zeros(q_ev.shape[1])
    for j in range(q_ev.shape[1]):
        lo, up = -hi, hi
        for _ in range(60):
            mid = 0.5 * (lo + up)
            if sigmoid(z_ref[:, j] - mid).mean() > pi_train[j]:
                lo = mid
            else:
                up = mid
        b[j] = 0.5 * (lo + up)
    return sigmoid(logit(q_ev) - b), b


def supervised_offset(q_cal, y_cal, q_ev, fallback_b, hi=40.0):
    z_cal = logit(q_cal)
    b = fallback_b.copy()
    for j in range(q_ev.shape[1]):
        yv = y_cal[:, j].astype(float)
        if yv.sum() == 0 or yv.sum() == len(yv):
            continue
        lo, up = -hi, hi
        for _ in range(60):
            mid = 0.5 * (lo + up)
            if sigmoid(z_cal[:, j] - mid).mean() > yv.mean():
                lo = mid
            else:
                up = mid
        b[j] = 0.5 * (lo + up)
    return sigmoid(logit(q_ev) - b)


def within_label_auc(q, y):
    a = [roc_auc_score(y[:, j], q[:, j]) for j in range(q.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


def distinct_scores_per_label(q):
    return float(np.mean([len(np.unique(np.round(q[:, j], 12))) for j in range(q.shape[1])]))


def load_split(p4, name):
    labels = p4.parse_labels_xml(p4._fetch(f"{name}/{name}.xml"))
    x_tr, y_tr = p4.parse_mulan_arff(p4._fetch(f"{name}/{name}-train.arff"), labels)
    x_te, y_te = p4.parse_mulan_arff(p4._fetch(f"{name}/{name}-test.arff"), labels)
    y_tr = y_tr.astype(np.int8)
    y_te = y_te.astype(np.int8)
    rng = np.random.default_rng(CAL_SEED)
    perm = rng.permutation(len(x_tr))
    ncal = max(int(len(x_tr) * CAL_FRAC), 50)
    ci, ti = perm[:ncal], perm[ncal:]
    return x_tr[ti], y_tr[ti], x_tr[ci], y_tr[ci], x_te, y_te


def cell_metrics(q_cal, q_te, y_cal, y_te, w, pi_train, kk, q_te_w1=None):
    c = {
        "w_median": float(np.median(w)),
        "w_max": float(w.max()),
        "sat_ge_1m1e9": float((q_te >= 1 - 1e-9).mean()),
        "sat_le_1e9": float((q_te <= 1e-9).mean()),
        "distinct_scores_per_label_test": distinct_scores_per_label(q_te),
        "within_label_auc": within_label_auc(q_te, y_te),
        "map_raw": map_at_k(y_te, q_te, kk),
        "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q_te, y_te), kk),
        "map_pooled_iso": map_at_k(y_te, pooled_iso(q_cal, y_cal, q_te), kk),
    }
    qc = np.clip(q_te, EPS, 1 - EPS)
    c["map_elkan_inversion_known_w"] = map_at_k(y_te, qc / (qc + w * (1 - qc)), kk)
    for pol in ("identity", "prior", "exclude"):
        p, nd = per_label_iso(q_cal, y_cal, q_te, pi_train, pol)
        c[f"map_per_label_iso_{pol}"] = map_at_k(y_te, p, kk)
    c["n_dead_in_cal"] = nd
    pm, b = prior_match_shift(q_te, q_cal, pi_train)
    c["map_prior_match_label_free"] = map_at_k(y_te, pm, kk)
    c["map_supervised_offset_cal"] = map_at_k(y_te, supervised_offset(q_cal, y_cal, q_te, b), kk)
    if q_te_w1 is not None:  # what-if: ideal odds shift applied to the unweighted model of the same learner
        c["map_whatif_unweighted_plus_lnw"] = map_at_k(y_te, sigmoid(logit(q_te_w1) + np.log(w)), kk)
    return c


def calibration_stats(datasets=None):
    """Per-dataset statistics of the calibration split: positives per label (no training)."""
    p4 = _load_p4()
    out = {}
    for name in datasets or DATASETS:
        x_fit, y_fit, x_cal, y_cal, x_te, y_te = load_split(p4, name)
        pos = y_cal.sum(0)
        out[name] = {
            "n_cal": int(len(y_cal)),
            "L": int(y_cal.shape[1]),
            "cal_pos_per_label_median": float(np.median(pos)),
            "cal_pos_per_label_min": int(pos.min()),
            "n_labels_lt5_cal_pos": int((pos < 5).sum()),
            "n_labels_lt10_cal_pos": int((pos < 10).sum()),
            "n_labels_zero_cal_pos": int((pos == 0).sum()),
        }
    return out


def run(datasets=None, k=7, learners=("lgbm_default", "lgbm_strong", "xgb", "logreg", "mlp_posweight")):
    p4 = _load_p4()
    datasets = datasets or DATASETS
    out = {"datasets": {}, "fetch_provenance": {}, "learners": list(learners), "weight_schemes": list(WEIGHT_SCHEMES)}
    for name in datasets:
        t0 = time.time()
        x_fit, y_fit, x_cal, y_cal, x_te, y_te = load_split(p4, name)
        n_pos = y_fit.sum(0).astype(float)
        pi_train = n_pos / len(y_fit)
        r = np.where(n_pos > 0, (len(y_fit) - n_pos) / np.maximum(n_pos, 1), 1.0)
        kk = min(k, y_te.shape[1])
        d = {
            "L": int(y_te.shape[1]),
            "n_fit": int(len(y_fit)),
            "n_cal": int(len(y_cal)),
            "n_test": int(len(y_te)),
            "d_features": int(x_te.shape[1]),
            "k": kk,
            "n_labels_zero_pos_fit": int((n_pos == 0).sum()),
            "n_labels_zero_pos_cal": int((y_cal.sum(0) == 0).sum()),
            "n_labels_zero_pos_test": int((y_te.sum(0) == 0).sum()),
            "r_neg_over_pos": {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())},
            "popularity_map": map_at_k(y_te, np.tile(pi_train, (len(y_te), 1)), kk),
            "cells": {},
        }
        for lname in learners:
            fit = LEARNERS[lname]
            q_te_w1 = None
            for w_name, (expo, mult) in WEIGHT_SCHEMES.items():
                w = r**expo * mult
                try:
                    q_cal, q_te = fit(x_fit, y_fit, [x_cal, x_te], w)
                except Exception as e:  # noqa: BLE001 - record and continue
                    d["cells"][f"{lname}|{w_name}"] = {"error": repr(e)}
                    continue
                if w_name == "w=1":
                    q_te_w1 = q_te
                d["cells"][f"{lname}|{w_name}"] = cell_metrics(q_cal, q_te, y_cal, y_te, w, pi_train, kk, q_te_w1)
            print(f"  [{name}] {lname} done", flush=True)
        d["elapsed_sec"] = round(time.time() - t0, 1)
        out["datasets"][name] = d
        print(f"[{name}] done in {d['elapsed_sec']}s", flush=True)
    out["fetch_provenance"] = dict(p4._FETCH_PROV)
    return out
