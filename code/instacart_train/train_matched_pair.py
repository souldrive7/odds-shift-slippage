"""Matched LightGBM pair on Instacart top-N products (arXiv v3): identical configuration with and
without per-label ``scale_pos_weight``, plus ``max_delta_step`` caps and 90% train-row draws.

Port of santander_train/train_matched_pair.py. Differences:
  * input is the sparse feature/label bundle written by build_features.py (CSR), not a CSV;
  * labels are fitted in parallel worker processes (one LightGBM ``Dataset`` per worker,
    ``set_label`` per label, ``lgb.train``); the parameter mapping is 1:1 with
    ``PerLabelScalePosLGBM`` (train_models.py) and eda.py records the parity check;
  * 50-label blocks are checkpointed atomically so an interrupted config resumes.

Outputs (one directory per config):
  <output-dir>/<config>/predictions.npz   p_te, p_va (float32), Y_te, Y_va (int8)   [gitignored]
  <output-dir>/<config>/run_meta.json     configuration, per-label n_pos / scale_pos_weight / fit_sec  [tracked]

The rederive code recomputes w_j from run_meta ``per_label[].n_pos`` and ``n_train``; keep them.
Launch at full scale only through pythonw.exe + Task Scheduler (scripts/run_instacart_matched_pair.bat).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_features import load_csr_bundle  # noqa: E402
from instacart_io import atomic_savez, atomic_write_json, log, redirect_log, sha256  # noqa: E402


def _cfg(weighted: bool, mds: float = 0.0, sub: float | None = None, sub_seed: int | None = None) -> dict:
    return {"weighted": weighted, "max_delta_step": float(mds), "train_subsample": sub, "subsample_seed": sub_seed}


CONFIGS: dict[str, dict] = {
    "unweighted": _cfg(False),
    "weighted": _cfg(True),
    "weighted_mds0.3": _cfg(True, 0.3),
    "weighted_mds0.7": _cfg(True, 0.7),
    "weighted_mds1": _cfg(True, 1.0),
    "weighted_mds2": _cfg(True, 2.0),
    "weighted_mds5": _cfg(True, 5.0),
    "weighted_sub0": _cfg(True, 0.0, 0.9, 0),
    "weighted_sub1": _cfg(True, 0.0, 0.9, 1),
    "weighted_sub2": _cfg(True, 0.0, 0.9, 2),
    "unweighted_sub0": _cfg(False, 0.0, 0.9, 0),
    "unweighted_sub1": _cfg(False, 0.0, 0.9, 1),
    "unweighted_sub2": _cfg(False, 0.0, 0.9, 2),
}
DEFAULT_CONFIGS = ("unweighted", "weighted", "weighted_mds0.3", "weighted_mds0.7", "weighted_mds1", "weighted_mds2", "weighted_mds5")
SUBSAMPLE_CONFIGS = tuple(c for c in CONFIGS if "_sub" in c)


def base_params(args: argparse.Namespace) -> dict:
    """Training parameters shared by every label and config (1:1 with PerLabelScalePosLGBM /
    LGBMClassifier defaults: objective binary, no subsampling, force_row_wise)."""
    return {
        "objective": "binary",
        "num_leaves": int(args.num_leaves),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "verbose": -1,
        "force_row_wise": True,
        "num_threads": int(args.num_threads),
    }


def label_params(base: dict, cfg: dict, spw: float) -> dict:
    params = dict(base)
    if cfg["weighted"]:
        params["scale_pos_weight"] = float(spw)
    if cfg["max_delta_step"] > 0.0:
        params["max_delta_step"] = float(cfg["max_delta_step"])
    return params


def row_mask_for(cfg: dict, n_train: int) -> np.ndarray | None:
    if cfg["train_subsample"] is None:
        return None
    rng = np.random.default_rng(int(cfg["subsample_seed"]))
    return rng.random(n_train) < float(cfg["train_subsample"])


# --------------------------------------------------------------------------------------
# worker process
# --------------------------------------------------------------------------------------

_G: dict = {}


def _worker_init(features_path: Path, labels_path: Path, row_mask: np.ndarray | None, base: dict, n_trees: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(base.get("num_threads", 1))
    import lightgbm as lgb

    f = load_csr_bundle(features_path)
    X_tr, X_va, X_te = f["X_tr"], f["X_va"], f["X_te"]
    Y_tr = load_csr_bundle(labels_path, ("Y_tr",))["Y_tr"]
    if row_mask is not None:
        X_tr = X_tr[row_mask]
        Y_tr = Y_tr[row_mask]
    ds = lgb.Dataset(X_tr, label=np.zeros(X_tr.shape[0], dtype=np.float32), params=dict(base), free_raw_data=False).construct()
    _G.update(lgb=lgb, X_va=X_va, X_te=X_te, Y_tr=Y_tr.tocsc(), ds=ds, base=dict(base), n_trees=int(n_trees), n_train=int(X_tr.shape[0]))


def _fit_block(task: tuple) -> dict:
    cname, cfg, j0, j1, label_cols, block_path = task
    lgb, ds, base = _G["lgb"], _G["ds"], _G["base"]
    n = _G["n_train"]
    m = len(label_cols)
    p_va = np.empty((_G["X_va"].shape[0], m), np.float32)
    p_te = np.empty((_G["X_te"].shape[0], m), np.float32)
    n_pos = np.empty(m, np.int64)
    spw = np.empty(m, np.float64)
    fit_sec = np.empty(m, np.float64)
    t_block = time.time()
    for jj, col in enumerate(label_cols):
        t0 = time.time()
        y = np.asarray(_G["Y_tr"][:, int(col)].todense()).ravel().astype(np.float32)
        npos = int(y.sum())
        w = (n - npos) / max(npos, 1) if cfg["weighted"] else 1.0
        ds.set_label(y)
        bst = lgb.train(label_params(base, cfg, w), ds, num_boost_round=_G["n_trees"])
        p_va[:, jj] = bst.predict(_G["X_va"], num_threads=base["num_threads"])
        p_te[:, jj] = bst.predict(_G["X_te"], num_threads=base["num_threads"])
        n_pos[jj], spw[jj], fit_sec[jj] = npos, w, time.time() - t0
    atomic_savez(block_path, p_va=p_va, p_te=p_te, n_pos=n_pos, spw=spw, fit_sec=fit_sec, label_cols=np.asarray(label_cols, np.int64))
    return {"config": cname, "j0": int(j0), "j1": int(j1), "block_sec": round(time.time() - t_block, 1), "mean_fit_sec": round(float(fit_sec.mean()), 2)}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def fit_config_parallel(
    *,
    cname: str,
    cfg: dict,
    features_path: Path,
    labels_path: Path,
    label_cols: np.ndarray,
    work_dir: Path,
    args: argparse.Namespace,
    row_mask: np.ndarray | None,
) -> dict:
    """Fit every column in ``label_cols`` for one config; block checkpoints under work_dir/blocks.
    Returns p_va, p_te (n, len(label_cols)) float32, per-label arrays and timing."""
    block_dir = work_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    label_cols = np.asarray(label_cols, dtype=np.int64)
    blocks: list[tuple[int, int, Path]] = []
    for j0 in range(0, len(label_cols), args.block_size):
        j1 = min(j0 + args.block_size, len(label_cols))
        blocks.append((j0, j1, block_dir / f"block_{j0:06d}_{j1:06d}.npz"))
    tasks = [(cname, cfg, j0, j1, label_cols[j0:j1], bp) for j0, j1, bp in blocks if not bp.exists()]
    log(f"[{cname}] {len(label_cols)} labels in {len(blocks)} blocks; {len(blocks) - len(tasks)} already done; n_jobs={args.n_jobs} num_threads={args.num_threads}")
    base = base_params(args)
    t0 = time.time()
    if tasks:
        with Pool(args.n_jobs, initializer=_worker_init, initargs=(features_path, labels_path, row_mask, base, args.n_trees)) as pool:
            for i, r in enumerate(pool.imap_unordered(_fit_block, tasks), 1):
                log(f"  [{cname}] block {r['j0']}-{r['j1']} done ({i}/{len(tasks)}) block {r['block_sec']}s mean fit {r['mean_fit_sec']}s elapsed {time.time() - t0:.0f}s")
    wall = time.time() - t0
    m = len(label_cols)
    p_va = p_te = None
    n_pos = np.empty(m, np.int64)
    spw = np.empty(m, np.float64)
    fit_sec = np.empty(m, np.float64)
    for j0, j1, bp in blocks:
        with np.load(bp) as z:
            assert np.array_equal(z["label_cols"], label_cols[j0:j1]), f"block {bp} label mismatch"
            if p_va is None:
                p_va = np.empty((z["p_va"].shape[0], m), np.float32)
                p_te = np.empty((z["p_te"].shape[0], m), np.float32)
            p_va[:, j0:j1] = z["p_va"]
            p_te[:, j0:j1] = z["p_te"]
            n_pos[j0:j1], spw[j0:j1], fit_sec[j0:j1] = z["n_pos"], z["spw"], z["fit_sec"]
    return {"p_va": p_va, "p_te": p_te, "n_pos": n_pos, "spw": spw, "fit_sec": fit_sec, "wall_sec": wall, "n_blocks_fitted": len(tasks), "block_dir": block_dir}


def main() -> None:
    p = argparse.ArgumentParser(description="Matched LightGBM pair on Instacart top-N products")
    p.add_argument("--processed-dir", type=Path, default=HERE / "data" / "processed")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs_matched_pair")
    p.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--num-threads", type=int, default=1, help="LightGBM threads per worker")
    p.add_argument("--block-size", type=int, default=50)
    p.add_argument("--n-trees", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-labels", type=int, default=None, help="smoke: first k top-N columns only")
    p.add_argument("--keep-blocks", action="store_true")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    redirect_log(args.log)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs: {unknown}; choose from {list(CONFIGS)}")
    features_path = args.processed_dir / "features.npz"
    labels_path = args.processed_dir / "labels_topN.npz"
    label_meta = json.loads((args.processed_dir / "label_meta.json").read_text(encoding="utf-8"))
    split = json.loads((args.processed_dir / "split.json").read_text(encoding="utf-8"))
    product_ids = np.asarray(label_meta["top_n"]["product_ids"], dtype=np.int64)
    n_labels_total = len(product_ids)
    label_cols = np.arange(n_labels_total if args.n_labels is None else min(args.n_labels, n_labels_total), dtype=np.int64)
    lab = load_csr_bundle(labels_path)
    Y_tr, Y_va, Y_te = lab["Y_tr"], lab["Y_va"], lab["Y_te"]
    n_train_full = int(Y_tr.shape[0])
    log(f"processed: {args.processed_dir}; {n_labels_total} top-N labels; train/val/test rows {Y_tr.shape[0]}/{Y_va.shape[0]}/{Y_te.shape[0]}; fitting {len(label_cols)} labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cname in configs:
        cfg = CONFIGS[cname]
        out_dir = args.output_dir / cname
        out_npz = out_dir / "predictions.npz"
        if out_npz.exists():
            log(f"[{cname}] already done ({out_npz}), skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = row_mask_for(cfg, n_train_full)
        n_train = n_train_full if mask is None else int(mask.sum())
        log(f"[{cname}] weighted={cfg['weighted']} max_delta_step={cfg['max_delta_step']} train_subsample={cfg['train_subsample']} n_train={n_train:,}")
        res = fit_config_parallel(cname=cname, cfg=cfg, features_path=features_path, labels_path=labels_path, label_cols=label_cols, work_dir=out_dir, args=args, row_mask=mask)
        Y_te_d = np.asarray(Y_te[:, label_cols].todense()).astype(np.int8)
        Y_va_d = np.asarray(Y_va[:, label_cols].todense()).astype(np.int8)
        atomic_savez(out_npz, compressed=True, p_te=res["p_te"], p_va=res["p_va"], Y_te=Y_te_d, Y_va=Y_va_d)
        y_tr_pos = np.asarray((Y_tr if mask is None else Y_tr[mask])[:, label_cols].sum(0)).ravel().astype(np.int64)
        assert np.array_equal(y_tr_pos, res["n_pos"]), "n_pos from workers differs from the label bundle"
        per_label = [
            {"label": int(product_ids[c]), "n_pos": int(res["n_pos"][i]), "scale_pos_weight": float(res["spw"][i]), "fit_sec": round(float(res["fit_sec"][i]), 2)}
            for i, c in enumerate(label_cols)
        ]
        meta = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": "instacart",
            "config": cname,
            "n_train": n_train,
            "n_train_full": n_train_full,
            "n_val": int(Y_va.shape[0]),
            "n_test": int(Y_te.shape[0]),
            "n_labels": int(len(label_cols)),
            "n_trees": int(args.n_trees),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "random_state": int(args.seed),
            "weighted": bool(cfg["weighted"]),
            "scale_pos_weight": "n_neg/n_pos per label" if cfg["weighted"] else None,
            "max_delta_step": float(cfg["max_delta_step"]),
            "train_subsample": cfg["train_subsample"],
            "subsample_seed": cfg["subsample_seed"],
            "subsampling": "none (feature_fraction=1.0, bagging off; LightGBM defaults)",
            "api": "lightgbm.train on one shared Dataset per worker, set_label per label; parameters 1:1 with LGBMClassifier via PerLabelScalePosLGBM (parity recorded in results/eda.json)",
            "split_strategy": split["basis_rule"],
            "label_set": label_meta["top_n"]["info"],
            "label_names": [int(product_ids[c]) for c in label_cols],
            "label_cols": label_cols.tolist(),
            "n_jobs": int(args.n_jobs),
            "num_threads": int(args.num_threads),
            "block_size": int(args.block_size),
            "inputs": {"features.npz": sha256(features_path), "labels_topN.npz": sha256(labels_path)},
            "total_fit_sec": round(float(res["fit_sec"].sum()), 1),
            "wall_sec": round(res["wall_sec"], 1),
            "per_label": per_label,
        }
        atomic_write_json(out_dir / "run_meta.json", meta)
        if not args.keep_blocks:
            for bp in res["block_dir"].glob("block_*.npz"):
                bp.unlink()
            try:
                res["block_dir"].rmdir()
            except OSError:
                pass
        log(f"[{cname}] wrote {out_npz} (wall {res['wall_sec']:.0f}s, fit {meta['total_fit_sec']}s)")
    log("done")


if __name__ == "__main__":
    main()
