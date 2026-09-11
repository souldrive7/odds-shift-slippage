"""Evidence recomputed from fixed prediction arrays only (no retraining), for every
dataset in datasets.py (Santander / Instacart, LightGBM / MLP) with one JSON schema.

Usage:
    uv run python code/run_experiment.py --dataset santander --part ladder
    uv run python code/run_experiment.py --dataset instacart --part capsweep

Parts (results/<dataset>_<part>.json unless noted):
  ladder      raw / saturation / within-label AUC / per-label iso (identity, prior, exclude) /
              pooled iso / offset, Platt, beta / oracle ceiling / Elkan inversion / label-free
              prevalence-matching shift, for S, N, every cap and weight dose
  whatif      loss decomposition: N raw -> N logit + a ln w -> S actual
  bootstrap   row-bootstrap CIs (B=2000) of the ladder entries on the te split
  deploy      calibrators fitted on the validation period, evaluated on all test rows
  calsize     calibration-split size sweep
  dead_freq   dead-label frequency vs split size (merged into dead_label_frequency.json)
  tie         tie attribution: MAP@7 when the cells at exactly 1.0 are re-ordered by the
              unweighted model / at random / by the labels                                (new)
  capsweep    realized shift b_j vs the step budget T*eta*c and ln w_j for every cap      (new)
  seeds       mean +- SD over the three 90% train-row draws                                (new)
    mulan, mulan_stats, mulan_tau, leaf, synthetic   dataset-independent studies

Every number the manuscript cites must come from the JSON written here. The dead-label policy of
every per-label calibrator is explicit (identity / prior / exclude).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from datasets import Dataset, get_dataset, sha256  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "results" / "canonical"
K = 7
CAL_SEED = 42
CAL_FRAC = 0.3
EPS = 1e-15
N_JOBS = 8


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # LF only: pre-commit rejects CRLF
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# metric and calibration helpers (per-label loops threaded)
# --------------------------------------------------------------------------------------


def map_at_k(y: np.ndarray, s: np.ndarray, k: int = K) -> float:
    """Kaggle MAP@K: rows without positives are excluded; stable argsort."""
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    ap = (prec * rel).sum(1) / np.minimum(y.sum(1), k)
    return float(ap.mean())


def _row_ap(y: np.ndarray, s: np.ndarray, k: int = K) -> np.ndarray:
    """Per-row AP@K on rows with positives (same definition as map_at_k)."""
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    return (prec * rel).sum(1) / np.minimum(y.sum(1), k)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def cal_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(CAL_SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * CAL_FRAC)
    return idx[:n_val], idx[n_val:]


def _columns(fn, L: int) -> list:
    """Apply fn(j) to every column, threaded (each column is independent, so the result does
    not depend on the schedule)."""
    if N_JOBS <= 1 or L <= 64:
        return [fn(j) for j in range(L)]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(fn)(j) for j in range(L))


def per_label_iso_multi(q: np.ndarray, y: np.ndarray, va: np.ndarray, prior: np.ndarray, policies=("identity", "prior", "exclude")) -> tuple[dict[str, np.ndarray], int, list[int]]:
    """Per-label isotonic fitted on `va` once; the three dead-label policies differ only on dead
    labels. Returns {policy: calibrated (n, L)}, n_dead, dead label indices."""
    L = q.shape[1]
    dead = [j for j in range(L) if y[va, j].sum() in (0, len(va))]
    dead_set = set(dead)

    def fit(j: int) -> np.ndarray | None:
        if j in dead_set:
            return None
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
        m.fit(q[va, j].astype(np.float64), y[va, j].astype(np.float64))
        return m.predict(q[:, j].astype(np.float64))

    cols = _columns(fit, L)
    out: dict[str, np.ndarray] = {}
    for pol in policies:
        o = q.copy()
        for j in range(L):
            if cols[j] is not None:
                o[:, j] = cols[j]
            elif pol == "identity":
                continue
            elif pol == "prior":
                o[:, j] = prior[j]
            elif pol == "exclude":
                o[:, j] = -np.inf
            else:
                raise ValueError(pol)
        out[pol] = o
    return out, len(dead), dead


def per_label_iso(q: np.ndarray, y: np.ndarray, va: np.ndarray, prior: np.ndarray, policy: str) -> np.ndarray:
    out, n_dead, _ = per_label_iso_multi(q, y, va, prior, (policy,))
    per_label_iso.n_dead = n_dead  # type: ignore[attr-defined]
    return out[policy]


def per_label_parametric(q: np.ndarray, y: np.ndarray, va: np.ndarray, prior: np.ndarray, kind: str) -> np.ndarray:
    """Per-label intercept-only logit shift ('offset'), Platt or beta calibration; prior fallback on dead labels."""
    from sklearn.linear_model import LogisticRegression

    L = q.shape[1]

    def fit(j: int) -> np.ndarray:
        yv = y[va, j]
        if yv.sum() == 0 or yv.sum() == len(yv):
            return np.full(q.shape[0], prior[j])
        pc = np.clip(q[:, j], 1e-12, 1 - 1e-12)
        if kind == "offset":  # intercept-only logit shift (slope fixed at 1), max likelihood by bisection
            z = logit(pc)
            lo, up = -40.0, 40.0
            for _ in range(60):
                mid = 0.5 * (lo + up)
                if sigmoid(z[va] - mid).mean() > yv.mean():
                    lo = mid
                else:
                    up = mid
            return sigmoid(z - 0.5 * (lo + up))
        X = logit(pc).reshape(-1, 1) if kind == "platt" else np.column_stack([np.log(pc), -np.log1p(-pc)])
        clf = LogisticRegression(C=1e6, max_iter=500).fit(X[va], yv)
        return clf.predict_proba(X)[:, 1]

    cols = _columns(fit, L)
    out = q.copy()
    for j in range(L):
        out[:, j] = cols[j]
    return out


def pooled_iso(q: np.ndarray, y: np.ndarray, va: np.ndarray, max_cells: int = 50_000_000, seed: int = 0) -> np.ndarray:
    """One shared isotonic map on all (cell, label) pairs of the calibration rows. For very large
    label sets the fit uses a seeded subsample of the calibration cells (recorded in _meta)."""
    qv = q[va].ravel().astype(np.float64)
    yv = y[va].ravel().astype(np.float64)
    if len(qv) > max_cells:
        sel = np.random.default_rng(seed).choice(len(qv), max_cells, replace=False)
        qv, yv = qv[sel], yv[sel]
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
    m.fit(qv, yv)
    return m.predict(q.ravel().astype(np.float64)).reshape(q.shape)


def oracle_ceiling(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Best per-label monotone map in squared loss, fitted in-sample on test (unreachable ceiling)."""
    L = q.shape[1]

    def fit(j: int) -> np.ndarray:
        if y[:, j].sum() == 0:
            return np.zeros(q.shape[0])
        m = IsotonicRegression(out_of_bounds="clip").fit(q[:, j].astype(np.float64), y[:, j].astype(np.float64))
        return m.predict(q[:, j].astype(np.float64))

    cols = _columns(fit, L)
    out = np.zeros_like(q, dtype=np.float64)
    for j in range(L):
        out[:, j] = cols[j]
    return out


def prior_match_shift(q: np.ndarray, pi_train: np.ndarray, hi: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Label-free repair: per-label logit shift b_j so that mean sigmoid(logit(q)-b_j) == pi_train_j."""
    z = logit(q)
    L = q.shape[1]

    def solve(j: int) -> float:
        lo, up = -hi, hi
        zj = z[:, j]
        for _ in range(60):
            mid = 0.5 * (lo + up)
            if sigmoid(zj - mid).mean() > pi_train[j]:
                lo = mid
            else:
                up = mid
        return 0.5 * (lo + up)

    b = np.array(_columns(solve, L), dtype=np.float64)
    return sigmoid(z - b), b


def within_label_auc(q: np.ndarray, y: np.ndarray, n_sub: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    aucs = []
    for j in range(q.shape[1]):
        ys = y[sub, j]
        if 0 < ys.sum() < len(ys):
            aucs.append(roc_auc_score(ys, q[sub, j]))
    return float(np.mean(aucs))


def elkan_inversion(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    qc = np.clip(q, EPS, 1 - EPS)
    return qc / (qc + w * (1 - qc))


# --------------------------------------------------------------------------------------
# Santander legacy scorers retained as non-reproducible robustness rows: the earlier weighted run,
# WGBoost, and the 100-tree unweighted model.
# --------------------------------------------------------------------------------------


def _santander_legacy(ds: Dataset, y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    if ds.key != "santander":
        return {}, {}, {}
    base = ds.arrays_dir.parent
    uq_p = base / "outputs_uq_regen" / "uq_arrays.npz"
    lg_p = base / "outputs_lgbm_baseline" / "predictions.npz"
    scores: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    inputs: dict[str, str] = {}
    if uq_p.exists():
        uq = np.load(uq_p)
        assert np.array_equal(uq["Y_te"].astype(np.int8), y), "uq_arrays Y_te differs from the matched pair"
        scores["S_orig_lgbm_spw"] = uq["p_te_br"].astype(np.float64)
        scores["W_wgboost"] = uq["p_te_wg"].astype(np.float64)
        notes["S_orig_lgbm_spw"] = "earlier run of the same weighted configuration (uq_arrays.npz p_te_br)"
        notes["W_wgboost"] = "WGBoost (uq_arrays.npz p_te_wg)"
        inputs["uq_arrays.npz"] = sha256(uq_p)
    if lg_p.exists():
        scores["Nh_lgbm_unweighted_100"] = np.load(lg_p)["p_te_lgbm"].astype(np.float64)
        notes["Nh_lgbm_unweighted_100"] = "earlier run: lgb.train 100 trees, feature_fraction 0.9, bagging 0.8/5, seed 42+j"
        inputs["predictions.npz"] = sha256(lg_p)
    return scores, notes, inputs


def _meta(ds: Dataset, t0: float, **extra) -> dict:
    inputs = ds.inputs_meta()
    inputs.update(extra.pop("inputs", {}))
    return {"generated_at": datetime.now(UTC).isoformat(), "dataset": ds.key, "learner": ds.learner, "inputs": inputs, "k": K, "elapsed_sec": round(time.time() - t0, 1), **extra}


# --------------------------------------------------------------------------------------
# parts
# --------------------------------------------------------------------------------------


def ladder_block(name: str, q: np.ndarray, y: np.ndarray, va: np.ndarray, te: np.ndarray, pi_train: np.ndarray, w_used: np.ndarray | None) -> dict:
    """The v2 per-model block, unchanged in keys. ``w_used`` (the weight the model was trained
    with) enables the Elkan inversion and the label-free shift for weighted models."""
    r: dict = {}
    r["raw"] = {"map7_all": map_at_k(y, q), "map7_te": map_at_k(y[te], q[te])}
    r["saturation"] = {
        "frac_ge_1m1e9": float((q >= 1 - 1e-9).mean()),
        "frac_le_1e9": float((q <= 1e-9).mean()),
        "frac_exact_1": float((q >= 1.0).mean()),
        "frac_exact_0": float((q <= 0.0).mean()),
        "frac_exact_1_per_label": (q >= 1.0).mean(0).tolist(),
    }
    r["within_label_auc_mean_sub200k"] = within_label_auc(q, y)
    cal, n_dead, dead = per_label_iso_multi(q, y, va, pi_train)
    for pol in ("identity", "prior", "exclude"):
        p = cal[pol]
        r[f"per_label_iso_{pol}"] = {"map7_te": map_at_k(y[te], p[te]), "n_dead_in_cal": n_dead}
        if pol == "identity" and dead:
            top = np.argsort(-p[te], axis=1, kind="stable")[:, :K]
            in_top = np.isin(top, dead).sum(1)
            alive = [j for j in range(q.shape[1]) if j not in set(dead)]
            r["per_label_iso_identity"]["dead_label_diagnostics"] = {
                "dead_label_idx": dead,
                "mean_dead_labels_in_top7": float(in_top.mean()),
                "frac_rows_all_dead_in_top7": float((in_top == len(dead)).mean()),
                "mean_raw_score_dead_labels": float(q[te][:, dead].mean()),
                "max_mean_calibrated_score_alive_labels": float(p[te][:, alive].mean(0).max()),
            }
    del cal
    r["pooled_iso"] = {"map7_te": map_at_k(y[te], pooled_iso(q, y, va)[te])}
    for kind in ("offset", "platt", "beta"):
        r[f"per_label_{kind}_prior"] = {"map7_te": map_at_k(y[te], per_label_parametric(q, y, va, pi_train, kind)[te])}
    oc = oracle_ceiling(q, y)
    r["oracle_in_sample_per_label_iso"] = {"map7_all": map_at_k(y, oc), "map7_te": map_at_k(y[te], oc[te])}
    del oc
    if w_used is not None:
        inv = elkan_inversion(q, w_used)
        r["elkan_inversion_known_w"] = {"map7_all": map_at_k(y, inv), "map7_te": map_at_k(y[te], inv[te])}
        del inv
        pm, b = prior_match_shift(q, pi_train)
        r["prior_match_shift_label_free"] = {"map7_all": map_at_k(y, pm), "map7_te": map_at_k(y[te], pm[te]), "shift_b": b.tolist(), "ln_w": np.log(w_used).tolist()}
    return r


def part_ladder(ds: Dataset) -> dict:
    t0 = time.time()
    y = ds.y("te")
    n = len(y)
    va, te = cal_split(n)
    pi_train = ds.pi_train
    res: dict = {}
    pop = np.tile(pi_train, (n, 1))
    res["popularity_train_prevalence"] = {"map7_all": map_at_k(y, pop), "map7_te": map_at_k(y[te], pop[te])}
    del pop
    cal_pos = y[va].sum(0).astype(int)
    te_pos = y[te].sum(0).astype(int)
    alive = cal_pos > 0
    res["calibration_split_positives"] = {
        "cal_pos_per_label": cal_pos.tolist(),
        "test_pos_per_label": te_pos.tolist(),
        "dead_in_cal_idx": [int(j) for j in np.flatnonzero(~alive)],
        "dead_in_test_idx": [int(j) for j in np.flatnonzero(te_pos == 0)],
        "cal_pos_min_alive": int(cal_pos[alive].min()),
        "cal_pos_median_alive": float(np.median(cal_pos[alive])),
        "cal_pos_max": int(cal_pos.max()),
    }
    notes: dict[str, str] = {}
    weights: dict[str, list] = {}
    order = ds.ladder_models()
    legacy, legacy_notes, legacy_inputs = _santander_legacy(ds, y)
    for spec in order:
        _log(f"[ladder:{ds.key}] {spec.name} ({spec.config})")
        ds.check_n_pos(spec.config)
        q = ds.scores(spec.config)
        w_used = ds.spw(spec.config) if spec.name.startswith("S_") else None
        res[spec.name] = ladder_block(spec.name, q, y, va, te, pi_train, w_used)
        res[spec.name]["params"] = ds.lgbm_params(spec.config) if ds.learner == "lightgbm" else {k: ds.meta(spec.config).get(k) for k in ("hidden", "epochs", "batch_size", "lr", "weight_decay", "random_state")}
        notes[spec.name] = spec.note
        if w_used is not None:
            weights[spec.name] = w_used.tolist()
        del q
    for name, q in legacy.items():
        _log(f"[ladder:{ds.key}] legacy {name}")
        res[name] = ladder_block(name, q, y, va, te, pi_train, ds.w_true if name.startswith("S_") else None)
        notes[name] = legacy_notes[name]
    res["_meta"] = _meta(
        ds,
        t0,
        inputs=legacy_inputs,
        n_test=int(n),
        n_train=ds.n_train,
        n_pos_train=ds.n_pos.astype(int).tolist(),
        scale_pos_weight_S=ds.w_true.tolist(),
        weights_used=weights,
        label_names=ds.label_names,
        cal_split={"seed": CAL_SEED, "frac": CAL_FRAC, "n_val": int(len(va))},
        dead_label_policies=["identity", "prior(train prevalence)", "exclude(-inf)"],
        map_definition="Kaggle MAP@7 over the model's label set, rows with no positives excluded, stable argsort",
        models=notes,
    )
    return res


def part_whatif(ds: Dataset) -> dict:
    """Decompose S's loss: N raw -> N logit + a*ln w (ideal odds shift, no saturation) -> actual S."""
    t0 = time.time()
    y = ds.y("te")
    S = ds.scores(ds.S.config)
    N = ds.scores(ds.N.config)
    w = ds.spw(ds.S.config)
    zN = logit(N)
    res: dict = {"N_raw": map_at_k(y, N), "S_actual": map_at_k(y, S)}
    res["whatif_N_plus_a_lnw"] = {str(a): map_at_k(y, sigmoid(zN + a * np.log(w))) for a in (0.25, 0.5, 0.75, 1.0)}
    m = min(500_000, len(y))
    tN = np.argsort(-N[:m], axis=1, kind="stable")[:, :K]
    tI = np.argsort(-(zN[:m] + np.log(w)), axis=1, kind="stable")[:, :K]
    ov = np.array([len(set(a) & set(b)) for a, b in zip(tN, tI, strict=True)])
    res["top7_overlap_N_vs_whatif"] = {"mean": float(ov.mean()), "frac_rows_changed": float((ov < K).mean()), "n_rows": int(m)}
    tS = np.argsort(-S[:m], axis=1, kind="stable")[:, :K]
    ovS = np.array([len(set(a) & set(b)) for a, b in zip(tN, tS, strict=True)])
    res["top7_overlap_N_vs_S_actual"] = {"mean": float(ovS.mean()), "frac_rows_changed": float((ovS < K).mean()), "n_rows": int(m)}
    loss = res["N_raw"] - res["S_actual"]
    res["loss_share"] = {
        "odds_shift_part": float((res["N_raw"] - res["whatif_N_plus_a_lnw"]["1.0"]) / loss) if loss > 0 else None,
        "residual_saturation_part": float((res["whatif_N_plus_a_lnw"]["1.0"] - res["S_actual"]) / loss) if loss > 0 else None,
    }
    res["_meta"] = _meta(ds, t0, weight="scale_pos_weight of the S configuration (run_meta per_label)")
    return res


def part_bootstrap(ds: Dataset, B: int = 2000, seed: int = 42) -> dict:
    """Row-bootstrap CIs on the te split for the ladder entries (per-row AP resampled)."""
    t0 = time.time()
    y = ds.y("te")
    pi_train = ds.pi_train
    va, te = cal_split(len(y))
    entries: dict[str, np.ndarray] = {}
    models: dict[str, str] = {"S": ds.S.config, "N": ds.N.config}
    legacy, _, legacy_inputs = _santander_legacy(ds, y)
    for name, cfg in models.items():
        q = ds.scores(cfg)
        entries[f"{name}_raw"] = _row_ap(y[te], q[te])
        cal, _, _ = per_label_iso_multi(q, y, va, pi_train, ("identity", "prior"))
        for pol in ("identity", "prior"):
            entries[f"{name}_per_label_iso_{pol}"] = _row_ap(y[te], cal[pol][te])
        del cal
        entries[f"{name}_pooled_iso"] = _row_ap(y[te], pooled_iso(q, y, va)[te])
        if name == "S":
            entries["S_prior_match_label_free"] = _row_ap(y[te], prior_match_shift(q, pi_train)[0][te])
            entries["S_elkan_inversion_known_w"] = _row_ap(y[te], elkan_inversion(q, ds.spw(cfg))[te])
        del q
    for name, q in legacy.items():
        short = {"W_wgboost": "W", "Nh_lgbm_unweighted_100": "Nh"}.get(name)
        if short is None:
            continue
        entries[f"{short}_raw"] = _row_ap(y[te], q[te])
        cal, _, _ = per_label_iso_multi(q, y, va, pi_train, ("identity", "prior"))
        for pol in ("identity", "prior"):
            entries[f"{short}_per_label_iso_{pol}"] = _row_ap(y[te], cal[pol][te])
        entries[f"{short}_pooled_iso"] = _row_ap(y[te], pooled_iso(q, y, va)[te])
    entries["popularity"] = _row_ap(y[te], np.tile(pi_train, (len(te), 1)))
    n = len(next(iter(entries.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = {k_: np.array([v[i].mean() for i in idx]) for k_, v in entries.items()}
    res: dict = {k_: {"point": float(v.mean()), "ci95": [float(np.percentile(boots[k_], 2.5)), float(np.percentile(boots[k_], 97.5))]} for k_, v in entries.items()}
    diffs = {
        "S_per_label_iso_prior_minus_popularity": ("S_per_label_iso_prior", "popularity"),
        "S_per_label_iso_identity_minus_popularity": ("S_per_label_iso_identity", "popularity"),
        "N_per_label_iso_prior_minus_N_raw": ("N_per_label_iso_prior", "N_raw"),
        "N_raw_minus_S_per_label_iso_prior": ("N_raw", "S_per_label_iso_prior"),
        "N_raw_minus_S_raw": ("N_raw", "S_raw"),
        "S_per_label_iso_prior_minus_S_elkan_inversion_known_w": ("S_per_label_iso_prior", "S_elkan_inversion_known_w"),
        "W_per_label_iso_prior_minus_W_raw": ("W_per_label_iso_prior", "W_raw"),
    }
    res["differences"] = {k_: {"point": float((entries[a] - entries[b]).mean()), "ci95": [float(np.percentile(boots[a] - boots[b], 2.5)), float(np.percentile(boots[a] - boots[b], 97.5))]} for k_, (a, b) in diffs.items() if a in entries and b in entries}
    res["_meta"] = _meta(ds, t0, inputs=legacy_inputs, B=B, seed=seed, n_rows_with_positives_te=int(n))
    return res


def part_deploy(ds: Dataset) -> dict:
    """Deployment protocol: calibrators fitted on the validation period (Santander: last 15% of the
    training months; Instacart: each user's order n_u-1) and evaluated on ALL test rows."""
    t0 = time.time()
    pi_train = ds.pi_train
    res: dict = {}
    for spec in ds.ladder_models():
        _log(f"[deploy:{ds.key}] {spec.name}")
        q_te, q_va = ds.scores(spec.config, "te"), ds.scores(spec.config, "va")
        y_te, y_va = ds.y("te"), ds.y("va")
        q = np.vstack([q_va, q_te])
        yy = np.vstack([y_va, y_te])
        fit_idx = np.arange(len(q_va))
        ev = slice(len(q_va), None)
        r: dict = {"model": spec.name, "n_val": int(len(q_va)), "n_test": int(len(q_te)), "raw": map_at_k(y_te, q_te)}
        r["val_pos_per_label"] = y_va.sum(0).astype(int).tolist()
        cal, n_dead, _ = per_label_iso_multi(q, yy, fit_idx, pi_train)
        for pol in ("identity", "prior", "exclude"):
            r[f"per_label_iso_{pol}"] = {"map7": map_at_k(y_te, cal[pol][ev]), "n_dead_in_val": n_dead}
        del cal
        r["pooled_iso"] = map_at_k(y_te, pooled_iso(q, yy, fit_idx)[ev])
        for kind in ("offset", "platt", "beta"):
            r[f"per_label_{kind}_prior"] = map_at_k(y_te, per_label_parametric(q, yy, fit_idx, pi_train, kind)[ev])
        r["oracle_in_sample_per_label_iso"] = map_at_k(y_te, oracle_ceiling(q_te, y_te))
        r["popularity"] = map_at_k(y_te, np.tile(pi_train, (len(y_te), 1)))
        if spec.name.startswith("S_"):
            r["elkan_inversion_known_w"] = map_at_k(y_te, elkan_inversion(q_te, ds.spw(spec.config)))
        res[spec.config] = r
        del q, yy, q_te, q_va
    res["_meta"] = _meta(ds, t0, protocol="calibrators fitted on the validation period (p_va, Y_va), evaluated on all test rows")
    return res


def part_calsize(ds: Dataset, fracs=(0.3, 0.1, 0.03, 0.01, 0.003), seeds=(42, 1, 2, 3, 4)) -> dict:
    """Calibration-split size sweep: when does per-label isotonic (prior fallback) stop beating the shared map?"""
    t0 = time.time()
    y = ds.y("te")
    pi_train = ds.pi_train
    res: dict = {"fracs": list(fracs), "seeds": list(seeds), "models": {}}
    legacy, _, legacy_inputs = _santander_legacy(ds, y)
    models = [("S", ds.scores(ds.S.config)), ("N", ds.scores(ds.N.config))]
    if "W_wgboost" in legacy:
        models.append(("W", legacy["W_wgboost"]))
    for name, q in models:
        rows = []
        for fr in fracs:
            for sd in seeds:
                rng = np.random.default_rng(sd)
                idx = np.arange(len(y))
                rng.shuffle(idx)
                nv = int(len(y) * fr)
                va, te = idx[:nv], idx[nv:]
                cal_pos = y[va].sum(0)
                rec = {"frac": fr, "seed": sd, "n_cal": int(nv), "n_dead": int((cal_pos == 0).sum()), "cal_pos_median": float(np.median(cal_pos)), "cal_pos_min": int(cal_pos.min())}
                rec["raw"] = map_at_k(y[te], q[te])
                cal, _, _ = per_label_iso_multi(q, y, va, pi_train, ("prior", "identity"))
                rec["per_label_iso_prior"] = map_at_k(y[te], cal["prior"][te])
                rec["per_label_iso_identity"] = map_at_k(y[te], cal["identity"][te])
                del cal
                rec["pooled_iso"] = map_at_k(y[te], pooled_iso(q, y, va)[te])
                rows.append(rec)
                _log(f"  [calsize:{ds.key}:{name}] frac={fr} seed={sd} dead={rec['n_dead']} pl={rec['per_label_iso_prior']:.4f} pooled={rec['pooled_iso']:.4f}")
        res["models"][name] = rows
    res["_meta"] = _meta(ds, t0, inputs=legacy_inputs)
    return res


def _dead_by_frac(y, n: int, fracs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3), n_seeds: int = 10) -> dict:
    out = {}
    for fr in fracs:
        ncal = int(n * fr)
        dead = []
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            ci = rng.permutation(n)[:ncal]
            dead.append(int((np.asarray(y[ci].sum(0)).ravel() == 0).sum()))
        out[str(fr)] = {"n_cal": ncal, "dead_mean": float(np.mean(dead)), "dead_max": int(max(dead)), "dead_min": int(min(dead)), "frac_seeds_any_dead": float(np.mean([x > 0 for x in dead]))}
    return out


def part_dead_freq(ds: Dataset) -> dict:
    """Frequency of dead labels (no positives in the calibration split) as a function of the split
    size. Labels only, no training. MULAN block recomputed; dataset blocks merged into one file."""
    t0 = time.time()
    path = OUT_DIR / "dead_label_frequency.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "mulan" not in res:
        import mulan_dose

        res["mulan"] = mulan_dose.dead_label_frequency()
    y = ds.y("te")
    n = len(y)
    res[ds.key] = {"L": int(y.shape[1]), "n_test": int(n), "test_pos_per_label": y.sum(0).astype(int).tolist(), "by_frac": _dead_by_frac(y, n)}
    if ds.key == "instacart":
        all_bundle = ds.labels_bundle.parent / "labels_all.npz" if ds.labels_bundle else None
        if all_bundle is not None and all_bundle.exists():
            import scipy.sparse as sp

            with np.load(all_bundle) as z:
                Y = sp.csr_matrix((z["Y_te_data"], z["Y_te_indices"], z["Y_te_indptr"]), shape=tuple(z["Y_te_shape"]))
            pos = np.asarray(Y.sum(0)).ravel().astype(int)
            res["instacart_all"] = {"L": int(Y.shape[1]), "n_test": int(Y.shape[0]), "n_labels_zero_test_pos": int((pos == 0).sum()), "by_frac": _dead_by_frac(Y, Y.shape[0])}
    res["_meta"] = {"generated_at": datetime.now(UTC).isoformat(), "n_seeds": 10, "last_dataset": ds.key, "elapsed_sec": round(time.time() - t0, 1)}
    return res


def part_tie(ds: Dataset, n_random_seeds: int = 5) -> dict:
    """Tie attribution: cells at exactly 1.0 have lost their order. Re-break those ties with the
    unweighted model's order (what a separable map cannot do), at random, or with the labels
    (oracle), keeping every unsaturated cell where it is; the MAP recovered by the unweighted
    order is the part of the residue that ties, not distortion, explain."""
    t0 = time.time()
    y = ds.y("te")
    va, te = cal_split(len(y))
    N = ds.scores(ds.N.config)
    N_raw = {"map7_all": map_at_k(y, N), "map7_te": map_at_k(y[te], N[te])}
    res: dict = {"N_raw": N_raw, "models": {}}
    for spec in [ds.S] + ds.role("cap"):
        if not ds.available(spec.config):
            continue
        q = ds.scores(spec.config)
        sat = q >= 1.0
        per_row = sat.sum(1)
        r: dict = {
            "cap": spec.cap,
            "saturated_cells_per_row": {"mean": float(per_row.mean()), "p50": float(np.median(per_row)), "p90": float(np.percentile(per_row, 90)), "frac_rows_ge1": float((per_row >= 1).mean()), "frac_rows_ge_k": float((per_row >= K).mean())},
            "frac_cells_exact_1": float(sat.mean()),
        }
        r["column_order"] = {"map7_all": map_at_k(y, q), "map7_te": map_at_k(y[te], q[te])}
        if sat.any():
            rs = []
            for s in range(n_random_seeds):
                s2 = q.copy()
                s2[sat] = 1.0 + np.random.default_rng(s).random(int(sat.sum()))
                rs.append((map_at_k(y, s2), map_at_k(y[te], s2[te])))
            r["random"] = {"map7_all_mean": float(np.mean([a for a, _ in rs])), "map7_all_sd": float(np.std([a for a, _ in rs], ddof=1)) if len(rs) > 1 else 0.0, "map7_te_mean": float(np.mean([b for _, b in rs])), "n_seeds": n_random_seeds}
            s3 = q.copy()
            s3[sat] = 1.0 + N[sat]
            r["unweighted_order"] = {"map7_all": map_at_k(y, s3), "map7_te": map_at_k(y[te], s3[te])}
            s4 = q.copy()
            s4[sat] = 1.0 + y[sat]
            r["oracle"] = {"map7_all": map_at_k(y, s4), "map7_te": map_at_k(y[te], s4[te])}
            del s2, s3, s4
            den = N_raw["map7_all"] - r["column_order"]["map7_all"]
            r["recovered_share_unweighted_order"] = float((r["unweighted_order"]["map7_all"] - r["column_order"]["map7_all"]) / den) if den > 0 else None
            r["recovered_share_oracle"] = float((r["oracle"]["map7_all"] - r["column_order"]["map7_all"]) / den) if den > 0 else None
        else:
            r["note"] = "no cell at exactly 1.0"
        res["models"][spec.name] = r
        del q, sat
    res["_meta"] = _meta(ds, t0, definition="column_order = raw stable argsort (ties resolved by column index, which is arbitrary w.r.t. popularity); other rules re-order only the cells at exactly 1.0")
    return res


def part_capsweep(ds: Dataset) -> dict:
    """Proposition 3 check: with a leaf cap c, the realized per-label shift is bounded by T*eta*c.
    Two estimates of the realized shift per label: the label-free prevalence-matching shift b_j
    (v2 estimator; search bound +-40) and the direct median of logit S - logit N over cells with
    both scores strictly inside (1e-6, 1-1e-6)."""
    t0 = time.time()
    y = ds.y("te")
    va, te = cal_split(len(y))
    pi_train = ds.pi_train
    N = ds.scores(ds.N.config)
    zN = logit(N)
    inner_N = (N > 1e-6) & (N < 1 - 1e-6)
    N_raw = map_at_k(y, N)
    res: dict = {"N_raw": N_raw, "models": {}}
    for spec in [ds.S] + ds.role("cap"):
        if not ds.available(spec.config):
            continue
        _log(f"[capsweep:{ds.key}] {spec.name}")
        q = ds.scores(spec.config)
        params = ds.lgbm_params(spec.config)
        T, eta = params.get("n_trees"), params.get("learning_rate")
        c = spec.cap
        tec = float(T * eta * c) if (c is not None and T and eta) else None
        w = ds.spw(spec.config)
        ln_w = np.log(w)
        pm, b_prior = prior_match_shift(q, pi_train)
        zq = logit(q)
        inner = inner_N & (q > 1e-6) & (q < 1 - 1e-6)
        diff = np.where(inner, zq - zN, np.nan)
        b_direct = np.nanmedian(diff, axis=0)
        n_inner = inner.sum(0)
        del diff, zq
        bound = ln_w if tec is None else np.minimum(ln_w, tec)
        r: dict = {
            "cap": c,
            "T": T,
            "eta": eta,
            "T_eta_c": tec,
            "ln_w": {"min": float(ln_w.min()), "median": float(np.median(ln_w)), "max": float(ln_w.max())},
            "b_prior": {"max_abs": float(np.abs(b_prior).max()), "median": float(np.median(b_prior)), "n_at_search_bound": int((np.abs(b_prior) >= 39.9).sum()), "per_label": b_prior.tolist()},
            "b_direct": {"max_abs": float(np.nanmax(np.abs(b_direct))), "median": float(np.nanmedian(b_direct)), "per_label": [None if np.isnan(v) else float(v) for v in b_direct], "n_inner_cells_per_label": n_inner.astype(int).tolist()},
            "ln_w_per_label": ln_w.tolist(),
            "saturation": {"frac_exact_1": float((q >= 1.0).mean()), "frac_ge_1m1e9": float((q >= 1 - 1e-9).mean()), "frac_exact_0": float((q <= 0.0).mean())},
            "map_raw": {"map7_all": map_at_k(y, q), "map7_te": map_at_k(y[te], q[te])},
            "map_prior_match_label_free": {"map7_all": map_at_k(y, pm), "map7_te": map_at_k(y[te], pm[te])},
            "map_elkan_inversion_known_w": {"map7_te": map_at_k(y[te], elkan_inversion(q, w)[te])},
            "map_whatif_N_plus_bound": {"map7_all": map_at_k(y, sigmoid(zN + bound)), "bound": "min(ln w_j, T eta c)" if tec is not None else "ln w_j"},
        }
        if tec is not None:
            r["bound_check"] = {
                "frac_labels_abs_b_prior_le_T_eta_c": float((np.abs(b_prior) <= tec + 1e-9).mean()),
                "frac_labels_abs_b_prior_le_T_eta_c_plus_0.1": float((np.abs(b_prior) <= tec + 0.1).mean()),
                "max_excess_b_prior": float((np.abs(b_prior) - tec).max()),
                # both boosters move by at most T*eta*c from the common start, so their difference is bounded by 2*T*eta*c
                "frac_labels_abs_b_direct_le_2T_eta_c": float(np.nanmean(np.abs(b_direct) <= 2 * tec + 1e-9)),
                "max_excess_b_direct_vs_2T_eta_c": float(np.nanmax(np.abs(b_direct) - 2 * tec)),
                "frac_labels_ln_w_gt_T_eta_c": float((ln_w > tec).mean()),
            }
        cal, _, _ = per_label_iso_multi(q, y, va, pi_train, ("prior",))
        r["map_per_label_iso_prior"] = {"map7_te": map_at_k(y[te], cal["prior"][te])}
        # The unweighted twin with the same cap (added after the pre-submission audit): the control
        # that says whether a capped-and-calibrated weighted arm gains from the weight or from the cap.
        twin_cfg = f"unweighted_mds{c:g}" if c is not None else None
        if twin_cfg is not None and ds.available(twin_cfg):
            _log(f"[capsweep:{ds.key}]   twin {twin_cfg}")
            qn = ds.scores(twin_cfg)
            twin_params = ds.lgbm_params(twin_cfg)
            assert float(twin_params.get("max_delta_step") or 0.0) == float(c), (twin_cfg, twin_params)
            cal_n, _, _ = per_label_iso_multi(qn, y, va, pi_train, ("prior",))
            ceil_n = oracle_ceiling(qn, y)
            whatif = sigmoid(logit(qn) + ln_w)  # the ideal odds shift applied to the capped unweighted twin
            n_raw_all, n_raw_te = map_at_k(y, qn), map_at_k(y[te], qn[te])
            s_raw_te = r["map_raw"]["map7_te"]
            loss_te = n_raw_te - s_raw_te
            r["unweighted_twin"] = {
                "config": twin_cfg,
                "map_raw": {"map7_all": n_raw_all, "map7_te": n_raw_te},
                "map_per_label_iso_prior": {"map7_te": map_at_k(y[te], cal_n["prior"][te])},
                "oracle_in_sample_per_label_iso": {"map7_all": map_at_k(y, ceil_n)},
                "map_whatif_N_plus_lnw": {"map7_te": map_at_k(y[te], whatif[te])},
                "loss_te": loss_te,
                "odds_shift_share_te": float((n_raw_te - map_at_k(y[te], whatif[te])) / loss_te) if loss_te > 0 else None,
                "saturation": {"frac_exact_1": float((qn >= 1.0).mean())},
                "within_label_auc_mean_sub200k": within_label_auc(qn, y),
            }
            del qn, cal_n, ceil_n, whatif
        del cal, pm, q
        res["models"][spec.name] = r
    res["_meta"] = _meta(ds, t0, note="b_prior: label-free prevalence-matching shift (bisection on [-40, 40]); b_direct: per-label median of logit S - logit N over cells strictly inside (1e-6, 1-1e-6) in both models")
    return res


def part_seeds(ds: Dataset) -> dict:
    """Mean +- SD over the three 90% train-row draws of the matched pair (LightGBM's seed is a
    no-op without subsampling on Instacart), for the headline ladder entries."""
    t0 = time.time()
    y = ds.y("te")
    va, te = cal_split(len(y))
    pi_train = ds.pi_train
    draws = []
    for s in range(3):
        sS, sN = ds.spec(f"weighted_sub{s}"), ds.spec(f"unweighted_sub{s}")
        if not (ds.available(sS.config) and ds.available(sN.config)):
            continue
        _log(f"[seeds:{ds.key}] draw {s}")
        S = ds.scores(sS.config)
        N = ds.scores(sN.config)
        w = ds.spw(sS.config)
        rec: dict = {"draw": s, "n_train": int(ds.meta(sS.config)["n_train"])}
        rec["S_raw"] = {"map7_all": map_at_k(y, S), "map7_te": map_at_k(y[te], S[te])}
        rec["N_raw"] = {"map7_all": map_at_k(y, N), "map7_te": map_at_k(y[te], N[te])}
        for name, q in (("S", S), ("N", N)):
            cal, n_dead, _ = per_label_iso_multi(q, y, va, pi_train, ("prior",))
            rec[f"{name}_per_label_iso_prior"] = {"map7_te": map_at_k(y[te], cal["prior"][te]), "n_dead_in_cal": n_dead}
            rec[f"{name}_pooled_iso"] = {"map7_te": map_at_k(y[te], pooled_iso(q, y, va)[te])}
            oc = oracle_ceiling(q, y)
            rec[f"{name}_oracle"] = {"map7_te": map_at_k(y[te], oc[te])}
            rec[f"{name}_saturation_frac_exact_1"] = float((q >= 1.0).mean())
            del cal, oc
        rec["S_elkan_inversion_known_w"] = {"map7_te": map_at_k(y[te], elkan_inversion(S, w)[te])}
        whatif = map_at_k(y, sigmoid(logit(N) + np.log(w)))
        loss = rec["N_raw"]["map7_all"] - rec["S_raw"]["map7_all"]
        rec["whatif_N_plus_lnw"] = {"map7_all": whatif}
        rec["loss_share_odds_shift"] = float((rec["N_raw"]["map7_all"] - whatif) / loss) if loss > 0 else None
        draws.append(rec)
        del S, N
    res: dict = {"draws": draws, "summary": {}}
    if draws:
        def collect(getter):
            vals = np.array([getter(d) for d in draws], dtype=np.float64)
            return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None, "values": vals.tolist()}

        keys = {
            "S_raw_te": lambda d: d["S_raw"]["map7_te"],
            "N_raw_te": lambda d: d["N_raw"]["map7_te"],
            "S_per_label_iso_prior_te": lambda d: d["S_per_label_iso_prior"]["map7_te"],
            "N_per_label_iso_prior_te": lambda d: d["N_per_label_iso_prior"]["map7_te"],
            "S_pooled_iso_te": lambda d: d["S_pooled_iso"]["map7_te"],
            "S_oracle_te": lambda d: d["S_oracle"]["map7_te"],
            "N_oracle_te": lambda d: d["N_oracle"]["map7_te"],
            "S_elkan_inversion_known_w_te": lambda d: d["S_elkan_inversion_known_w"]["map7_te"],
            "S_saturation_frac_exact_1": lambda d: d["S_saturation_frac_exact_1"],
            "loss_share_odds_shift": lambda d: d["loss_share_odds_shift"] if d["loss_share_odds_shift"] is not None else np.nan,
        }
        res["summary"] = {k: collect(g) for k, g in keys.items()}
    res["_meta"] = _meta(ds, t0, protocol="90% train-row draws, seeds 0/1/2, LightGBM configuration unchanged", n_draws=len(draws))
    return res


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

DATASET_PARTS = ("ladder", "santander", "whatif", "bootstrap", "deploy", "calsize", "dead_freq", "tie", "capsweep", "seeds")
GLOBAL_PARTS = ("mulan", "mulan_stats", "mulan_tau", "leaf", "synthetic")


def main() -> None:
    global N_JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="santander", help="santander | instacart | santander_mlp | instacart_mlp")
    ap.add_argument("--part", choices=DATASET_PARTS + GLOBAL_PARTS, default="ladder")
    ap.add_argument("--datasets", nargs="*", default=None, help="MULAN dataset subset (mulan parts only)")
    ap.add_argument("--arrays-dir", type=Path, default=None, help="override the prediction-array directory of --dataset")
    ap.add_argument("--labels-bundle", type=Path, default=None, help="override the label bundle used for the consistency asserts")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--out-dir", type=Path, default=None, help="override results/ (smoke tests)")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()
    N_JOBS = int(args.n_jobs)
    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(args.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    part = "ladder" if args.part == "santander" else args.part
    if part in GLOBAL_PARTS:
        import leaf_check
        import mulan_dose
        import synthetic_check

        if part == "mulan":
            res, out = mulan_dose.run(args.datasets), OUT_DIR / "mulan_dose_response.json"
        elif part == "mulan_stats":
            # keep the v2 envelope: make_tables.py reads ["datasets"], and the cal split is
            # what makes these counts meaningful, so it is recorded next to them
            res = {"datasets": mulan_dose.calibration_stats(args.datasets),
                   "_meta": {"generated_at": datetime.now(UTC).isoformat(),
                             "cal_split": {"seed": mulan_dose.CAL_SEED, "frac": mulan_dose.CAL_FRAC}}}
            out = OUT_DIR / "mulan_cal_stats.json"
        elif part == "mulan_tau":
            res, out = mulan_dose.run_tau_selection(args.datasets), OUT_DIR / "mulan_tau_select.json"
        elif part == "leaf":
            res, out = leaf_check.run(), OUT_DIR / "leaf_check.json"
        else:
            res, out = synthetic_check.run(), OUT_DIR / "synthetic_check.json"
    else:
        ds = get_dataset(args.dataset, args.arrays_dir)
        if args.labels_bundle is not None:
            ds.labels_bundle = args.labels_bundle
        _log(f"dataset {ds.key}: arrays {ds.arrays_dir}; available models {[s.name for s in ds.specs if ds.available(s.config)]}")
        fn = {"ladder": part_ladder, "whatif": part_whatif, "bootstrap": part_bootstrap, "deploy": part_deploy, "calsize": part_calsize, "dead_freq": part_dead_freq, "tie": part_tie, "capsweep": part_capsweep, "seeds": part_seeds}[part]
        res = fn(ds)
        out = OUT_DIR / ("dead_label_frequency.json" if part == "dead_freq" else f"{ds.key}_{part}.json")
    atomic_write_json(out, res)
    _log(f"wrote {out}")


if __name__ == "__main__":
    main()
