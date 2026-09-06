"""Route F evidence, recomputed from primary arrays only (no retraining).

Usage:
    uv run python experiments/arxiv_v2_rederive/run_experiment.py --part santander

Every number the manuscript cites must come from the JSON written here.
Dead-label policy for per-label calibration is explicit (identity / prior / exclude).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def _arrays_dir() -> Path:
    """Where the two fixed Santander prediction arrays live.

    Resolution order: --arrays-dir (set by main), $DRC_ARRAYS_DIR, ../../data (release layout),
    then the thesis-repository layout in which the arrays were produced.
    """
    env = os.environ.get("DRC_ARRAYS_DIR")
    if env:
        return Path(env)
    release = HERE.parent / "data"
    if (release / "uq_arrays.npz").exists():
        return release
    # thesis-repository layout: some experiment directory holds outputs_uq_regen/uq_arrays.npz
    for hit in sorted((HERE.parents[1] / "experiments").glob("*/outputs_uq_regen/uq_arrays.npz")):
        return hit.parents[1]
    return release


ARRAYS = _arrays_dir()


def _find(name: str, *subdirs: str) -> Path:
    """First existing candidate among ARRAYS/<subdir>/name (thesis layout) and ARRAYS/name (release)."""
    for sub in subdirs:
        cand = ARRAYS / sub / name
        if cand.exists():
            return cand
    return ARRAYS / name


UQ_NPZ = _find("uq_arrays.npz", "outputs_uq_regen")  # keys Y_te, p_te_br (LGBM-w), p_te_wg (WGB)
LGBM_NPZ = _find("predictions.npz", "outputs_lgbm_baseline")  # key p_te_lgbm (LGBM-0)
META_JSON = _find("santander_run_meta.json") if (ARRAYS / "santander_run_meta.json").exists() else _find("run_meta.json", "outputs_lgbm_baseline")
OUT_DIR = HERE / "results"  # committed (outputs/ is gitignored)
K = 7
CAL_SEED = 42
CAL_FRAC = 0.3
EPS = 1e-15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # LF only: pre-commit rejects CRLF
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


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


def per_label_iso(q: np.ndarray, y: np.ndarray, va: np.ndarray, prior: np.ndarray, policy: str) -> np.ndarray:
    """Per-label isotonic fitted on `va`; dead labels handled by `policy`."""
    out = q.copy()
    n_dead = 0
    for j in range(q.shape[1]):
        yv = y[va, j]
        if yv.sum() == 0 or yv.sum() == len(yv):
            n_dead += 1
            if policy == "identity":
                continue
            if policy == "prior":
                out[:, j] = prior[j]
            elif policy == "exclude":
                out[:, j] = -np.inf
            else:
                raise ValueError(policy)
            continue
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
        m.fit(q[va, j].astype(np.float64), yv.astype(np.float64))
        out[:, j] = m.predict(q[:, j].astype(np.float64))
    per_label_iso.n_dead = n_dead  # type: ignore[attr-defined]
    return out


def per_label_parametric(q: np.ndarray, y: np.ndarray, va: np.ndarray, prior: np.ndarray, kind: str) -> np.ndarray:
    """Per-label Platt (logit slope+intercept) or Beta (3-parameter) calibration, prior fallback on dead labels."""
    from sklearn.linear_model import LogisticRegression

    out = q.copy()
    for j in range(q.shape[1]):
        yv = y[va, j]
        if yv.sum() == 0 or yv.sum() == len(yv):
            out[:, j] = prior[j]
            continue
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
            out[:, j] = sigmoid(z - 0.5 * (lo + up))
            continue
        if kind == "platt":
            X = logit(pc).reshape(-1, 1)
        else:  # beta calibration features [log p, -log(1-p)]
            X = np.column_stack([np.log(pc), -np.log1p(-pc)])
        clf = LogisticRegression(C=1e6, max_iter=500).fit(X[va], yv)
        out[:, j] = clf.predict_proba(X)[:, 1]
    return out


def pooled_iso(q: np.ndarray, y: np.ndarray, va: np.ndarray) -> np.ndarray:
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
    m.fit(q[va].ravel().astype(np.float64), y[va].ravel().astype(np.float64))
    return m.predict(q.ravel().astype(np.float64)).reshape(q.shape)


def oracle_ceiling(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Best per-label monotone map in squared loss, fitted in-sample on test (unreachable ceiling)."""
    out = np.zeros_like(q, dtype=np.float64)
    for j in range(q.shape[1]):
        if y[:, j].sum() == 0:
            continue
        m = IsotonicRegression(out_of_bounds="clip").fit(q[:, j].astype(np.float64), y[:, j].astype(np.float64))
        out[:, j] = m.predict(q[:, j].astype(np.float64))
    return out


def prior_match_shift(q: np.ndarray, pi_train: np.ndarray, hi: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Label-free repair: per-label logit shift b_j so that mean sigmoid(logit(q)-b_j) == pi_train_j."""
    z = logit(q)
    b = np.zeros(q.shape[1])
    for j in range(q.shape[1]):
        lo, up = -hi, hi
        for _ in range(60):
            mid = 0.5 * (lo + up)
            if sigmoid(z[:, j] - mid).mean() > pi_train[j]:
                lo = mid
            else:
                up = mid
        b[j] = 0.5 * (lo + up)
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


def part_santander() -> dict:
    t0 = time.time()
    meta = json.load(open(META_JSON, encoding="utf-8"))
    n_train = int(meta["n_train"])
    n_pos = np.array([d["n_pos"] for d in meta["per_label"]], dtype=np.float64)
    pi_train = n_pos / n_train
    w = (n_train - n_pos) / n_pos
    uq = np.load(UQ_NPZ)
    y = uq["Y_te"].astype(np.int8)
    scores = {
        "S_lgbm_spw": uq["p_te_br"].astype(np.float64),
        "W_wgboost": uq["p_te_wg"].astype(np.float64),
        "N_lgbm_unweighted": np.load(LGBM_NPZ)["p_te_lgbm"].astype(np.float64),
    }
    n = len(y)
    va, te = cal_split(n)
    res: dict = {}
    pop = np.tile(pi_train, (n, 1))
    res["popularity_train_prevalence"] = {"map7_all": map_at_k(y, pop), "map7_te": map_at_k(y[te], pop[te])}
    for name, q in scores.items():
        r: dict = {}
        r["raw"] = {"map7_all": map_at_k(y, q), "map7_te": map_at_k(y[te], q[te])}
        r["saturation"] = {
            "frac_ge_1m1e9": float((q >= 1 - 1e-9).mean()),
            "frac_le_1e9": float((q <= 1e-9).mean()),
            "frac_exact_1": float((q >= 1.0).mean()),
            "frac_exact_0": float((q <= 0.0).mean()),
        }
        r["within_label_auc_mean_sub200k"] = within_label_auc(q, y)
        dead = [j for j in range(q.shape[1]) if y[va, j].sum() == 0]
        for pol in ("identity", "prior", "exclude"):
            p = per_label_iso(q, y, va, pi_train, pol)
            r[f"per_label_iso_{pol}"] = {"map7_te": map_at_k(y[te], p[te]), "n_dead_in_cal": per_label_iso.n_dead}
            if pol == "identity" and dead:
                top = np.argsort(-p[te], axis=1, kind="stable")[:, :K]
                in_top = np.isin(top, dead).sum(1)
                alive = [j for j in range(q.shape[1]) if j not in dead]
                r["per_label_iso_identity"]["dead_label_diagnostics"] = {
                    "dead_label_idx": dead,
                    "mean_dead_labels_in_top7": float(in_top.mean()),
                    "frac_rows_all_dead_in_top7": float((in_top == len(dead)).mean()),
                    "mean_raw_score_dead_labels": float(q[te][:, dead].mean()),
                    "max_mean_calibrated_score_alive_labels": float(p[te][:, alive].mean(0).max()),
                }
        r["pooled_iso"] = {"map7_te": map_at_k(y[te], pooled_iso(q, y, va)[te])}
        for kind in ("offset", "platt", "beta"):
            r[f"per_label_{kind}_prior"] = {"map7_te": map_at_k(y[te], per_label_parametric(q, y, va, pi_train, kind)[te])}
        oc = oracle_ceiling(q, y)
        r["oracle_in_sample_per_label_iso"] = {"map7_all": map_at_k(y, oc), "map7_te": map_at_k(y[te], oc[te])}
        if name == "S_lgbm_spw":
            qc = np.clip(q, EPS, 1 - EPS)
            r["elkan_inversion_known_w"] = {"map7_all": map_at_k(y, qc / (qc + w * (1 - qc)))}
            pm, b = prior_match_shift(q, pi_train)
            r["prior_match_shift_label_free"] = {
                "map7_all": map_at_k(y, pm),
                "map7_te": map_at_k(y[te], pm[te]),
                "shift_b": b.tolist(),
                "ln_w": np.log(w).tolist(),
            }
        res[name] = r
    res["_meta"] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": _inputs_meta(),
        "n_test": int(n),
        "n_train": n_train,
        "n_pos_train": n_pos.astype(int).tolist(),
        "scale_pos_weight_S": w.tolist(),
        "k": K,
        "cal_split": {"seed": CAL_SEED, "frac": CAL_FRAC, "n_val": int(len(va))},
        "dead_label_policies": ["identity", "prior(train prevalence)", "exclude(-inf)"],
        "map_definition": "Kaggle MAP@7, rows with no positives excluded, stable argsort",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return res


def _inputs_meta() -> dict[str, str]:
    """File name -> SHA-256 of the two primary arrays (no local path is recorded)."""
    return {UQ_NPZ.name: sha256(UQ_NPZ), LGBM_NPZ.name: sha256(LGBM_NPZ)}


def _load_santander():
    meta = json.load(open(META_JSON, encoding="utf-8"))
    n_pos = np.array([d["n_pos"] for d in meta["per_label"]], dtype=np.float64)
    n_train = int(meta["n_train"])
    uq = np.load(UQ_NPZ)
    y = uq["Y_te"].astype(np.int8)
    S = uq["p_te_br"].astype(np.float64)
    W = uq["p_te_wg"].astype(np.float64)
    N = np.load(LGBM_NPZ)["p_te_lgbm"].astype(np.float64)
    return y, S, W, N, n_pos, n_train


def part_whatif() -> dict:
    """Decompose S's loss: N raw -> N logit + a*ln w (ideal odds shift, no saturation) -> actual S."""
    t0 = time.time()
    y, S, W, N, n_pos, n_train = _load_santander()
    w = (n_train - n_pos) / n_pos
    zN = logit(N)
    res: dict = {"N_raw": map_at_k(y, N), "S_actual": map_at_k(y, S)}
    res["whatif_N_plus_a_lnw"] = {str(a): map_at_k(y, sigmoid(zN + a * np.log(w))) for a in (0.25, 0.5, 0.75, 1.0)}
    # flip statistics predicted by theorem T1 on the first 500k rows
    m = 500_000
    tN = np.argsort(-N[:m], axis=1, kind="stable")[:, :K]
    tI = np.argsort(-(zN[:m] + np.log(w)), axis=1, kind="stable")[:, :K]
    ov = np.array([len(set(a) & set(b)) for a, b in zip(tN, tI, strict=True)])
    res["top7_overlap_N_vs_whatif"] = {"mean": float(ov.mean()), "frac_rows_changed": float((ov < K).mean()), "n_rows": m}
    tS = np.argsort(-S[:m], axis=1, kind="stable")[:, :K]
    ovS = np.array([len(set(a) & set(b)) for a, b in zip(tN, tS, strict=True)])
    res["top7_overlap_N_vs_S_actual"] = {"mean": float(ovS.mean()), "frac_rows_changed": float((ovS < K).mean()), "n_rows": m}
    res["loss_share"] = {
        "odds_shift_part": float((res["N_raw"] - res["whatif_N_plus_a_lnw"]["1.0"]) / (res["N_raw"] - res["S_actual"])),
        "residual_saturation_part": float((res["whatif_N_plus_a_lnw"]["1.0"] - res["S_actual"]) / (res["N_raw"] - res["S_actual"])),
    }
    res["_meta"] = {"generated_at": datetime.now(UTC).isoformat(), "inputs": _inputs_meta(), "k": K, "elapsed_sec": round(time.time() - t0, 1)}
    return res


def _row_ap(y: np.ndarray, s: np.ndarray, k: int = K) -> np.ndarray:
    """Per-row AP@K on rows with positives (same definition as map_at_k)."""
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    return (prec * rel).sum(1) / np.minimum(y.sum(1), k)


def part_bootstrap(B: int = 2000, seed: int = 42) -> dict:
    """Row-bootstrap CIs on the te split for the ladder entries (per-row AP resampled)."""
    t0 = time.time()
    y, S, W, N, n_pos, n_train = _load_santander()
    pi_train = n_pos / n_train
    va, te = cal_split(len(y))
    entries: dict[str, np.ndarray] = {}
    for name, q in {"S": S, "W": W, "N": N}.items():
        entries[f"{name}_raw"] = _row_ap(y[te], q[te])
        for pol in ("identity", "prior"):
            p = per_label_iso(q, y, va, pi_train, pol)
            entries[f"{name}_per_label_iso_{pol}"] = _row_ap(y[te], p[te])
        entries[f"{name}_pooled_iso"] = _row_ap(y[te], pooled_iso(q, y, va)[te])
    entries["S_prior_match_label_free"] = _row_ap(y[te], prior_match_shift(S, pi_train)[0][te])
    entries["popularity"] = _row_ap(y[te], np.tile(pi_train, (len(te), 1)))
    n = len(next(iter(entries.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = {k_: np.array([v[i].mean() for i in idx]) for k_, v in entries.items()}
    res: dict = {k_: {"point": float(v.mean()), "ci95": [float(np.percentile(boots[k_], 2.5)), float(np.percentile(boots[k_], 97.5))]} for k_, v in entries.items()}
    diffs = {
        "S_per_label_iso_prior_minus_popularity": ("S_per_label_iso_prior", "popularity"),
        "S_per_label_iso_identity_minus_popularity": ("S_per_label_iso_identity", "popularity"),
        "W_per_label_iso_prior_minus_W_raw": ("W_per_label_iso_prior", "W_raw"),
        "N_per_label_iso_prior_minus_N_raw": ("N_per_label_iso_prior", "N_raw"),
        "N_raw_minus_S_per_label_iso_prior": ("N_raw", "S_per_label_iso_prior"),
    }
    res["differences"] = {k_: {"point": float((entries[a] - entries[b]).mean()), "ci95": [float(np.percentile(boots[a] - boots[b], 2.5)), float(np.percentile(boots[a] - boots[b], 97.5))]} for k_, (a, b) in diffs.items()}
    res["_meta"] = {"generated_at": datetime.now(UTC).isoformat(), "B": B, "seed": seed, "n_rows_with_positives_te": int(n), "elapsed_sec": round(time.time() - t0, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["santander", "mulan", "mulan_stats", "synthetic", "whatif", "bootstrap", "leaf"], default="santander")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--arrays-dir", default=None, help="directory holding uq_arrays.npz, predictions.npz, santander_run_meta.json (else $DRC_ARRAYS_DIR)")
    args = ap.parse_args()
    if args.arrays_dir:
        global ARRAYS, UQ_NPZ, LGBM_NPZ, META_JSON
        ARRAYS = Path(args.arrays_dir)
        UQ_NPZ = _find("uq_arrays.npz", "outputs_uq_regen")
        LGBM_NPZ = _find("predictions.npz", "outputs_lgbm_baseline")
        META_JSON = _find("santander_run_meta.json") if (ARRAYS / "santander_run_meta.json").exists() else _find("run_meta.json", "outputs_lgbm_baseline")
    if args.part == "santander":
        out = part_santander()
        path = OUT_DIR / "santander_ladder.json"
        msg = f"wrote {path} in {out['_meta']['elapsed_sec']}s"
    elif args.part == "synthetic":
        import synthetic_check

        out = synthetic_check.run()
        out["_meta"] = {"generated_at": datetime.now(UTC).isoformat()}
        path = OUT_DIR / "synthetic_check.json"
        msg = f"wrote {path} in {out['elapsed_sec']}s"
    elif args.part == "whatif":
        out = part_whatif()
        path = OUT_DIR / "santander_whatif.json"
        msg = f"wrote {path}"
    elif args.part == "bootstrap":
        out = part_bootstrap()
        path = OUT_DIR / "santander_bootstrap.json"
        msg = f"wrote {path}"
    elif args.part == "mulan_stats":
        import mulan_dose

        out = {"datasets": mulan_dose.calibration_stats(args.datasets or None), "_meta": {"generated_at": datetime.now(UTC).isoformat(), "cal_split": {"seed": CAL_SEED, "frac": CAL_FRAC}}}
        path = OUT_DIR / "mulan_cal_stats.json"
        msg = f"wrote {path}"
    elif args.part == "leaf":
        import leaf_check

        out = leaf_check.run()
        out["_meta"] = {"generated_at": datetime.now(UTC).isoformat()}
        path = OUT_DIR / "leaf_check.json"
        msg = f"wrote {path}"
    else:
        import mulan_dose

        out = mulan_dose.run(args.datasets or None)  # None -> mulan_dose.DATASETS (8 datasets)
        out["_meta"] = {"generated_at": datetime.now(UTC).isoformat(), "cal_split": {"seed": CAL_SEED, "frac": CAL_FRAC}}
        path = OUT_DIR / "mulan_dose_response.json"
        msg = f"wrote {path}"
    atomic_write_json(path, out)
    print(msg)
    if args.log:
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


if __name__ == "__main__":
    main()
