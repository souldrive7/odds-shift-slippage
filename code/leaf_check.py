"""Leaf-saturation lemma check (route F, theorem T2).

Lemma: for weighted log-loss, the optimal constant in a leaf with n+ positives and n- negatives is
the weighted empirical rate  w n+ / (w n+ + n-).  A leaf containing at least one positive is pushed to
probability ~1 as soon as w >> n-, so all cells falling in such leaves tie near 1 and lose their order.

Exact check: sklearn DecisionTreeClassifier with class_weight={1: w} returns exactly the weighted leaf
rate, so the predicted saturation fraction (share of test cells landing in leaves with n+ >= 1 and
w n+ / (w n+ + n-) >= 1 - 1e-3) must equal the measured share of predictions >= 1 - 1e-3.
Empirical check: LightGBM with min_child_samples in {5, 20, 50, 200} x weight multipliers; report
saturation fraction, distinct scores, raw MAP and oracle ceiling.

Called from run_experiment.py --part leaf.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mulan_dose import DATASETS, _load_p4, distinct_scores_per_label, load_split, map_at_k, oracle_ceiling

MIN_CHILD = [5, 20, 50, 200]
W_MULT = {"w=1": 0.0, "w=r": 1.0, "w=10r": 10.0, "w=100r": 100.0}


def tree_exact(x_fit, y_fit, x_te, y_te, w, kk, min_leaf=20):
    """Single tree per label: predicted vs measured saturation from the lemma."""
    L = y_fit.shape[1]
    q = np.zeros((len(x_te), L))
    pred_sat = 0.0
    for j in range(L):
        yj = y_fit[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            q[:, j] = yj.mean()
            continue
        t = DecisionTreeClassifier(min_samples_leaf=min_leaf, class_weight={0: 1.0, 1: float(w[j])}, random_state=0).fit(x_fit, yj)
        q[:, j] = t.predict_proba(x_te)[:, 1]
        leaf_tr = t.apply(x_fit)
        leaf_te = t.apply(x_te)
        npos = {lf: int(yj[leaf_tr == lf].sum()) for lf in np.unique(leaf_tr)}
        nneg = {lf: int((yj[leaf_tr == lf] == 0).sum()) for lf in np.unique(leaf_tr)}
        rate = {lf: (w[j] * npos[lf]) / (w[j] * npos[lf] + nneg[lf]) if (w[j] * npos[lf] + nneg[lf]) > 0 else 0.0 for lf in npos}
        pred_sat += float(np.mean([rate.get(lf, 0.0) >= 1 - 1e-3 for lf in leaf_te]))
    pred_sat /= L
    meas_sat = float((q >= 1 - 1e-3).mean())
    return {
        "predicted_sat_from_lemma": pred_sat,
        "measured_sat": meas_sat,
        "map_raw": map_at_k(y_te, q, kk),
        "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q, y_te), kk),
        "distinct_scores_per_label": distinct_scores_per_label(q),
    }


def run(datasets=("enron", "medical", "scene"), k=7):
    from learners import lgbm_ovr

    p4 = _load_p4()
    out = {"datasets": {}, "min_child": MIN_CHILD, "weights": list(W_MULT)}
    for name in datasets:
        t0 = time.time()
        x_fit, y_fit, x_cal, y_cal, x_te, y_te = load_split(p4, name)
        n_pos = y_fit.sum(0).astype(float)
        r = np.where(n_pos > 0, (len(y_fit) - n_pos) / np.maximum(n_pos, 1), 1.0)
        kk = min(k, y_te.shape[1])
        d = {"L": int(y_te.shape[1]), "n_fit": int(len(y_fit)), "k": kk, "tree_exact": {}, "lgbm": {}}
        for w_name, mult in W_MULT.items():
            w = np.ones_like(r) if mult == 0 else r * mult
            d["tree_exact"][w_name] = tree_exact(x_fit, y_fit, x_te, y_te, w, kk)
            for mc in MIN_CHILD:
                params = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=mc)
                (q_te,) = lgbm_ovr(x_fit, y_fit, [x_te], w, params)
                d["lgbm"][f"min_child={mc}|{w_name}"] = {
                    "sat_ge_1m1e3": float((q_te >= 1 - 1e-3).mean()),
                    "sat_ge_1m1e9": float((q_te >= 1 - 1e-9).mean()),
                    "distinct_scores_per_label": distinct_scores_per_label(q_te),
                    "map_raw": map_at_k(y_te, q_te, kk),
                    "map_oracle_ceiling": map_at_k(y_te, oracle_ceiling(q_te, y_te), kk),
                    "w_median": float(np.median(w)),
                }
        d["elapsed_sec"] = round(time.time() - t0, 1)
        out["datasets"][name] = d
        print(f"[leaf:{name}] done in {d['elapsed_sec']}s", flush=True)
    out["fetch_provenance"] = dict(p4._FETCH_PROV)
    _ = DATASETS
    return out
