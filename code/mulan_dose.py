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
# Every MULAN dataset that ships a standard train/test split, except tmc2007 (61 MB) and Corel16k
# (366 MB), which we leave out for size. Ordered by label count so the table reads as a ladder.
DATASETS = [
    "emotions",
    "scene",
    "flags",
    "yeast",
    "birds",
    "genbase",
    "medical",
    "enron",
    "bibtex",
    "corel5k",
    "delicious",
]
FILE_STEM = {"corel5k": "Corel5k"}
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


def per_label_iso(q_cal, y_cal, q_ev, prior, policy, tau=0):
    """Per-label isotonic fitted on the calibration split. A label whose calibration positives
    are <= tau (or that has no negatives) is 'dead' and handled by `policy`; tau=0 is the plain
    zero-positive rule used throughout the dose-response sweep."""
    out = q_ev.copy()
    n_dead = 0
    for j in range(q_ev.shape[1]):
        yv = y_cal[:, j]
        if yv.sum() <= tau or yv.sum() == len(yv):
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
    stem = FILE_STEM.get(name, name)  # the mirror's directory and file stem differ for Corel5k
    labels = p4.parse_labels_xml(p4._fetch(f"{name}/{stem}.xml"))
    x_tr, y_tr = p4.parse_mulan_arff(p4._fetch(f"{name}/{stem}-train.arff"), labels)
    x_te, y_te = p4.parse_mulan_arff(p4._fetch(f"{name}/{stem}-test.arff"), labels)
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


TAUS = (0, 1, 2, 5, 10)


def select_tau_cv(q_cal, y_cal, prior, kk, taus=TAUS, n_folds=5, seed=42):
    """Choose the positives threshold tau and per-label-vs-shared on the calibration split only.

    K-fold inside the calibration split: fit on K-1 folds, score the held-out fold, pool the
    out-of-fold scores and evaluate MAP@K once. Candidates are per-label isotonic with prior
    fallback at each tau, plus the shared (pooled) isotonic map. Ties go to the smaller tau
    (and to per-label over shared) so that the selection is deterministic.
    Returns (choice, tau, oof_map_by_candidate).
    """
    n = len(y_cal)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    cands = [f"per_label_tau{t}" for t in taus] + ["shared", "raw"]
    oof = {c: np.zeros_like(q_cal, dtype=np.float64) for c in cands}
    for f in range(n_folds):
        ho = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        for t in taus:
            oof[f"per_label_tau{t}"][ho], _ = per_label_iso(q_cal[tr], y_cal[tr], q_cal[ho], prior, "prior", tau=t)
        oof["shared"][ho] = pooled_iso(q_cal[tr], y_cal[tr], q_cal[ho])
    oof["raw"] = q_cal.astype(np.float64)  # no calibration is always a candidate
    scores = {c: map_at_k(y_cal, oof[c], kk) for c in cands}
    best = max(cands, key=lambda c: (scores[c], -cands.index(c)))
    if best in ("shared", "raw"):
        return best, None, scores
    return "per_label", int(best.replace("per_label_tau", "")), scores


def _row_ap(y, s, k):
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    return (prec * rel).sum(1) / np.minimum(y.sum(1), k)


def _boot_diff(y, s_a, s_b, k, B=2000, seed=42):
    """Row-bootstrap 95% CI of MAP(s_a) - MAP(s_b) on the rows with positives (paired)."""
    a = _row_ap(y, s_a, k)
    b = _row_ap(y, s_b, k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return {"point": float((a - b).mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def run_tau_selection(datasets=None, k=7, learners=("lgbm_default", "lgbm_strong", "xgb", "logreg", "mlp_posweight"), weights=("w=1", "w=r"), B=2000):
    """Recipe check: choose tau (and per-label vs shared) inside the calibration split, then
    report the test MAP@K of the chosen calibrator against raw and against tau=0. Adds
    row-bootstrap CIs for the default LightGBM cells (the ones quoted in the paper)."""
    p4 = _load_p4()
    datasets = datasets or DATASETS
    out = {"datasets": {}, "taus": list(TAUS), "learners": list(learners), "weights": list(weights), "cv": {"n_folds": 5, "seed": 42}, "bootstrap": {"B": B, "seed": 42}}
    for name in datasets:
        t0 = time.time()
        x_fit, y_fit, x_cal, y_cal, x_te, y_te = load_split(p4, name)
        n_pos = y_fit.sum(0).astype(float)
        pi_train = n_pos / len(y_fit)
        r = np.where(n_pos > 0, (len(y_fit) - n_pos) / np.maximum(n_pos, 1), 1.0)
        kk = min(k, y_te.shape[1])
        d = {"L": int(y_te.shape[1]), "n_cal": int(len(y_cal)), "n_test": int(len(y_te)), "k": kk, "cells": {}}
        for lname in learners:
            fit = LEARNERS[lname]
            for w_name in weights:
                expo, mult = WEIGHT_SCHEMES[w_name]
                w = r**expo * mult
                try:
                    q_cal, q_te = fit(x_fit, y_fit, [x_cal, x_te], w)
                except Exception as e:  # noqa: BLE001
                    d["cells"][f"{lname}|{w_name}"] = {"error": repr(e)}
                    continue
                choice, tau, oof = select_tau_cv(q_cal, y_cal, pi_train, kk)
                p0, _ = per_label_iso(q_cal, y_cal, q_te, pi_train, "prior", tau=0)
                if choice == "shared":
                    p_sel = pooled_iso(q_cal, y_cal, q_te)
                elif choice == "raw":
                    p_sel = q_te
                else:
                    p_sel = per_label_iso(q_cal, y_cal, q_te, pi_train, "prior", tau=tau)[0]
                c = {
                    "choice": choice,
                    "tau_selected": tau,
                    "oof_map_by_candidate": oof,
                    "map_raw": map_at_k(y_te, q_te, kk),
                    "map_per_label_iso_prior_tau0": map_at_k(y_te, p0, kk),
                    "map_selected": map_at_k(y_te, p_sel, kk),
                    "map_shared": map_at_k(y_te, pooled_iso(q_cal, y_cal, q_te), kk),
                    "map_per_label_iso_prior_by_tau": {str(t): map_at_k(y_te, per_label_iso(q_cal, y_cal, q_te, pi_train, "prior", tau=t)[0], kk) for t in TAUS},
                    "n_dead_by_tau": {str(t): int(per_label_iso(q_cal, y_cal, q_te, pi_train, "prior", tau=t)[1]) for t in TAUS},
                }
                if lname == "lgbm_default":
                    c["ci_tau0_minus_raw"] = _boot_diff(y_te, p0, q_te, kk, B)
                    c["ci_selected_minus_raw"] = _boot_diff(y_te, p_sel, q_te, kk, B)
                    if w_name == "w=r":
                        q_w1 = d["cells"].get(f"{lname}|w=1", {}).get("_q_te")
                        if q_w1 is not None:
                            c["ci_whatif_minus_raw"] = _boot_diff(y_te, sigmoid(logit(q_w1) + np.log(w)), q_te, kk, B)
                    if w_name == "w=1":
                        c["_q_te"] = q_te
                d["cells"][f"{lname}|{w_name}"] = c
            print(f"  [{name}] {lname} tau-selection done", flush=True)
        for c in d["cells"].values():
            c.pop("_q_te", None)
        d["elapsed_sec"] = round(time.time() - t0, 1)
        out["datasets"][name] = d
        print(f"[{name}] done in {d['elapsed_sec']}s", flush=True)
    out["fetch_provenance"] = dict(p4._FETCH_PROV)
    return out


def dead_label_frequency(datasets=None, fracs=(0.05, 0.1, 0.2, 0.3, 0.5), n_seeds=10):
    """How often does a random calibration split of a given size leave a label with no
    positives? Labels only, no training. Per dataset and fraction: mean/max dead labels over seeds."""
    p4 = _load_p4()
    out = {}
    for name in datasets or DATASETS:
        labels = p4.parse_labels_xml(p4._fetch(f"{name}/{name}.xml"))
        _, y_tr = p4.parse_mulan_arff(p4._fetch(f"{name}/{name}-train.arff"), labels)
        y_tr = y_tr.astype(np.int8)
        n = len(y_tr)
        rec = {"L": int(y_tr.shape[1]), "n_train_file": int(n), "train_pos_min": int(y_tr.sum(0).min()), "by_frac": {}}
        for fr in fracs:
            ncal = max(int(n * fr), 1)
            dead = []
            for s in range(n_seeds):
                rng = np.random.default_rng(1000 + s)
                ci = rng.permutation(n)[:ncal]
                dead.append(int((y_tr[ci].sum(0) == 0).sum()))
            rec["by_frac"][str(fr)] = {"n_cal": ncal, "dead_mean": float(np.mean(dead)), "dead_max": int(max(dead)), "dead_min": int(min(dead)), "frac_seeds_any_dead": float(np.mean([x > 0 for x in dead]))}
        out[name] = rec
    return out


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
