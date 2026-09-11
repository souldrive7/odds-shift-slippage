#!/usr/bin/env python3
"""Matched LightGBM pair: identical configuration with and without
per-label ``scale_pos_weight``, plus ``max_delta_step`` variants.

Why: the earlier unweighted baseline (``train_lightgbm_baseline.py``: 100 trees, feature and
bagging subsampling, per-label seeds, ``lgb.train``) and the weighted model of ``run_uq.py``
(``PerLabelScalePosLGBM``: 60 trees, no subsampling, one seed, ``LGBMClassifier``) differ in
more than the weights, which confounds the loss decomposition. Here every config goes
through ``PerLabelScalePosLGBM`` so that only the named arguments differ.

Data loading, time split and preprocessing mirror ``train_lightgbm_baseline.py`` /
``run_uq.py`` exactly (same rows, same ``Y_te`` order), so the outputs are cell-for-cell
comparable with ``outputs_uq_regen/uq_arrays.npz`` and ``outputs_lgbm_baseline/predictions.npz``.

Outputs (one directory per config, each written atomically once that config is done):
  <output-dir>/<config>/predictions.npz   p_te, p_va (float32), Y_te, Y_va (int8)
  <output-dir>/<config>/run_meta.json     full configuration, per-label n_pos / fit_sec

Configs:
  unweighted        LGBMClassifier(n_estimators, lr, num_leaves, random_state)      -- matched LGBM-0
  weighted          + scale_pos_weight = n_-/n_+ per label                           -- LGBM-w (re-run)
  weighted_mds0.7   weighted + max_delta_step=0.7
  weighted_mds2     weighted + max_delta_step=2.0

Launch at full scale only through pythonw.exe + Task Scheduler (see scripts/run_matched_pair.bat).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from io_data import build_xy_pairs_fast, featurize, load_panel_csv
from train_models import PerLabelScalePosLGBM, build_preprocess, split_cat_num

def _cfg(weighted: bool, mds: float = 0.0, power: float = 1.0, mult: float = 1.0, sub: float | None = None, sub_seed: int | None = None) -> dict:
    return {"weighted": weighted, "max_delta_step": float(mds), "weight_power": float(power), "weight_multiplier": float(mult), "train_subsample": sub, "subsample_seed": sub_seed}


CONFIGS: dict[str, dict] = {
    # Core matched-pair configurations (default --configs)
    "unweighted": _cfg(False),
    "weighted": _cfg(True),
    "weighted_mds0.7": _cfg(True, 0.7),
    "weighted_mds2": _cfg(True, 2.0),
    # Additional cap, weight-dose, and 90% train-row-draw configurations
    "weighted_mds0.3": _cfg(True, 0.3),
    "weighted_mds1": _cfg(True, 1.0),
    "weighted_mds5": _cfg(True, 5.0),
    "weighted_sqrt": _cfg(True, power=0.5),
    "weighted_10r": _cfg(True, mult=10.0),
    "weighted_sub0": _cfg(True, sub=0.9, sub_seed=0),
    "weighted_sub1": _cfg(True, sub=0.9, sub_seed=1),
    "weighted_sub2": _cfg(True, sub=0.9, sub_seed=2),
    "unweighted_sub0": _cfg(False, sub=0.9, sub_seed=0),
    "unweighted_sub1": _cfg(False, sub=0.9, sub_seed=1),
    "unweighted_sub2": _cfg(False, sub=0.9, sub_seed=2),
    # Unweighted twins of the cap sweep (the control the paper's c=5 observation lacked): same
    # max_delta_step, no scale_pos_weight. Added 2026-09-12 after the pre-submission audit.
    "unweighted_mds0.3": _cfg(False, 0.3),
    "unweighted_mds0.7": _cfg(False, 0.7),
    "unweighted_mds1": _cfg(False, 1.0),
    "unweighted_mds2": _cfg(False, 2.0),
    "unweighted_mds5": _cfg(False, 5.0),
}
V2_CONFIGS = ("unweighted", "weighted", "weighted_mds0.7", "weighted_mds2")
V3_CONFIGS = tuple(c for c in CONFIGS if c not in V2_CONFIGS)


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _time_split_idx(
    fecha: pd.Series, test_frac: float = 0.2, val_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(fecha.values)
    n = len(order)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_tr = n - n_test - n_val
    return order[:n_tr], order[n_tr : n_tr + n_val], order[n_tr + n_val :]


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    # The temp name must itself end in .npz -- np.savez_compressed appends .npz otherwise.
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Matched LightGBM pair (with / without per-label weights)")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--csv", type=str, default="train_ver2.csv")
    p.add_argument("--output-dir", type=Path, default=Path("outputs_matched_pair"))
    p.add_argument("--configs", type=str, default=",".join(V2_CONFIGS), help=f"comma list; v3 additions: {','.join(V3_CONFIGS)}")
    p.add_argument(
        "--save-features-cache",
        type=Path,
        default=None,
        help="write X_tr/X_va/X_te.npy (float32) and Y_*.npy (int8) here once (input for train_mlp_posweight.py --dataset santander)",
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trees", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="write progress here instead of stderr. Required under pythonw.exe, which "
        "has no console (see scripts/run_matched_pair.bat for the launch pattern).",
    )
    args = p.parse_args()

    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(args.log, "w", encoding="utf-8", buffering=1)  # noqa: SIM115

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs: {unknown}; choose from {list(CONFIGS)}")

    csv_path = args.data_dir / args.csv
    if not csv_path.is_file():
        sys.exit(f"Missing {csv_path}. Run scripts/fetch_santander.ps1 first.")

    _log(f"loading panel: {csv_path}")
    df = load_panel_csv(csv_path, max_rows=args.max_rows)
    _log(f"raw rows={len(df):,}")
    X_raw, Y, prod_cols = build_xy_pairs_fast(df)
    X_df, feat_names = featurize(X_raw, prod_cols)
    _log(f"feature rows={X_df.shape[0]:,} feature cols={X_df.shape[1]:,} labels={Y.shape[1]}")

    tr_idx, va_idx, te_idx = _time_split_idx(
        pd.to_datetime(X_raw["fecha_dato"]), args.test_frac, args.val_frac
    )
    _log(f"split: train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}")

    X_tr, X_va, X_te = X_df.iloc[tr_idx], X_df.iloc[va_idx], X_df.iloc[te_idx]
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    cat_cols, num_cols = split_cat_num(feat_names, prod_cols)
    pre = build_preprocess(cat_cols, num_cols)
    t0 = time.time()
    _log("fitting preprocessor on train ...")
    pre.fit(X_tr)
    _log(f"  preprocessor fit in {time.time() - t0:.1f}s")
    for name, X in (("train", X_tr), ("val", X_va), ("test", X_te)):
        t0 = time.time()
        _log(f"transforming {name} ...")
        arr = np.ascontiguousarray(pre.transform(X), dtype=np.float32)
        _log(f"  {name} shape={arr.shape} in {time.time() - t0:.1f}s")
        if name == "train":
            X_tr_t = arr
        elif name == "val":
            X_va_t = arr
        else:
            X_te_t = arr
    del X_df, X_raw, df

    if args.save_features_cache is not None:
        cache = args.save_features_cache
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / "X_te.npy").exists():
            _log(f"writing features cache to {cache} ...")
            for name, arr in (("X_tr", X_tr_t), ("X_va", X_va_t), ("X_te", X_te_t), ("Y_tr", Y_tr.astype(np.int8)), ("Y_va", Y_va.astype(np.int8)), ("Y_te", Y_te.astype(np.int8))):
                tmp = cache / f"{name}.partial.npy"
                np.save(tmp, arr)
                tmp.replace(cache / f"{name}.npy")
            (cache / "meta.json").write_text(
                json.dumps({"label_names": prod_cols, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)), "n_features": int(X_tr_t.shape[1]), "csv": csv_path.name, "split_strategy": "time-based via fecha_dato", "generated_at": datetime.now(UTC).isoformat()}, indent=2),
                encoding="utf-8",
            )
            _log("  features cache written")
        else:
            _log(f"features cache already present at {cache}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_labels = Y.shape[1]

    for cname in configs:
        cfg = CONFIGS[cname]
        out_dir = args.output_dir / cname
        out_npz = out_dir / "predictions.npz"
        if out_npz.exists():
            _log(f"[{cname}] already done ({out_npz}), skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        # 90% train-row draw (v3 "three seeds": LightGBM's seed only re-samples bin boundaries here)
        if cfg["train_subsample"] is not None:
            mask = np.random.default_rng(int(cfg["subsample_seed"])).random(len(tr_idx)) < float(cfg["train_subsample"])
            X_fit, Y_fit = X_tr_t[mask], Y_tr[mask]
        else:
            X_fit, Y_fit = X_tr_t, Y_tr
        y_tr_pos = Y_fit.sum(0).astype(int)
        _log(
            f"[{cname}] training {n_labels} LightGBM models on {len(X_fit):,} rows "
            f"(n_estimators={args.n_trees}, lr={args.learning_rate}, num_leaves={args.num_leaves}, "
            f"random_state={args.seed}, weighted={cfg['weighted']}, max_delta_step={cfg['max_delta_step']}, "
            f"weight_power={cfg['weight_power']}, weight_multiplier={cfg['weight_multiplier']}, train_subsample={cfg['train_subsample']})"
        )
        model = PerLabelScalePosLGBM(
            n_estimators=args.n_trees,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            random_state=args.seed,
            weighted=cfg["weighted"],
            max_delta_step=cfg["max_delta_step"],
            weight_power=cfg["weight_power"],
            weight_multiplier=cfg["weight_multiplier"],
        )
        # Fit label by label (same loop as PerLabelScalePosLGBM.fit) so that progress is logged.
        t_total = time.time()
        per_label: list[dict] = []
        model.models_ = []
        model.scale_pos_ = []
        model.n_labels_ = n_labels
        for j in range(n_labels):
            t0 = time.time()
            sub = PerLabelScalePosLGBM(
                n_estimators=args.n_trees,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                random_state=args.seed,
                weighted=cfg["weighted"],
                max_delta_step=cfg["max_delta_step"],
                weight_power=cfg["weight_power"],
                weight_multiplier=cfg["weight_multiplier"],
            )
            sub.fit(X_fit, Y_fit[:, j : j + 1])
            model.models_.append(sub.models_[0])
            model.scale_pos_.append(sub.scale_pos_[0])
            dt = time.time() - t0
            per_label.append(
                {
                    "label": prod_cols[j],
                    "n_pos": int(y_tr_pos[j]),
                    "scale_pos_weight": float(sub.scale_pos_[0]),
                    "fit_sec": round(dt, 2),
                }
            )
            _log(
                f"  [{cname}] label {j + 1:2d}/{n_labels} ({prod_cols[j]:24s}) "
                f"n_pos={int(y_tr_pos[j]):>8d} spw={sub.scale_pos_[0]:>12.1f} fit {dt:5.1f}s"
            )
        fit_sec = time.time() - t_total
        _log(f"[{cname}] total fit {fit_sec:.1f}s; predicting val/test ...")
        p_va = model.predict_proba_matrix(X_va_t).astype(np.float32)
        p_te = model.predict_proba_matrix(X_te_t).astype(np.float32)
        _atomic_savez(
            out_npz,
            p_te=p_te,
            p_va=p_va,
            Y_te=Y_te.astype(np.int8),
            Y_va=Y_va.astype(np.int8),
        )
        meta = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": cname,
            "csv": csv_path.name,
            "n_train": int(len(X_fit)),
            "n_train_full": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "n_labels": int(n_labels),
            "n_trees": int(args.n_trees),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "random_state": int(args.seed),
            "weighted": bool(cfg["weighted"]),
            "scale_pos_weight": (f"{cfg['weight_multiplier']} * (n_neg/n_pos) ** {cfg['weight_power']} per label" if cfg["weighted"] else None),
            "weight_power": float(cfg["weight_power"]),
            "weight_multiplier": float(cfg["weight_multiplier"]),
            "train_subsample": cfg["train_subsample"],
            "subsample_seed": cfg["subsample_seed"],
            "max_delta_step": float(cfg["max_delta_step"]),
            "subsampling": "none (feature_fraction=1.0, bagging off; LightGBM defaults)",
            "api": "lightgbm.LGBMClassifier via PerLabelScalePosLGBM, force_row_wise=True",
            "split_strategy": "time-based via fecha_dato (matches run_uq.py / train_lightgbm_baseline.py)",
            "max_rows": args.max_rows,
            "label_names": prod_cols,
            "total_fit_sec": round(fit_sec, 1),
            "per_label": per_label,
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log(f"[{cname}] wrote {out_npz}")
    _log("done")


if __name__ == "__main__":
    main()
