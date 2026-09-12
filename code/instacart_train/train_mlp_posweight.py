"""Shared-trunk MLP with per-label ``pos_weight`` (second learner family).

One network scores every label: Linear(d, h) - ReLU - Linear(h, h) - ReLU - Linear(h, L), trained
with BCE-with-logits and per-label ``pos_weight`` = w_j (config ``wr``: w_j = n_-/n_+ on the fit
rows; config ``w1``: 1). Same recipe as learners.py:mlp_posweight,
scaled up: sparse inputs, minibatches with dense per-batch targets, width 256, fixed 20 epochs,
no early stopping, seed fixed, CPU only.

Datasets:
  --dataset instacart   features.npz (CSR, top-N product columns + dense block) and labels_all.npz
                        (all products); the top-N columns are also saved as predictions.npz so the
                        rederive code processes them exactly like the LightGBM pair
  --dataset santander   the dense features cache written by santander_train/train_matched_pair.py
                        --save-features-cache (X_*.npy, Y_*.npy, 24 labels)

Outputs (<output-dir>/<config>/):
  predictions.npz   p_te, p_va (float32, top-N columns), Y_te, Y_va (int8)           [gitignored]
  p_te_all.npy, p_va_all.npy   float32 label-major (L, n) memmaps, all labels (instacart) [gitignored]
  run_meta.json     configuration, per-label n_pos / pos_weight (top-N), n_pos of every label [tracked]

Launch at full scale through pythonw.exe + Task Scheduler (scripts/run_mlp_posweight.bat).
Smoke: python code/instacart_train/train_mlp_posweight.py --dataset instacart --processed-dir data/processed_smoke --n-labels 50 --epochs 1 --output-dir outputs_smoke_mlp
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
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_features import load_csr_bundle  # noqa: E402
from instacart_io import atomic_savez, atomic_write_json, log, redirect_log, sha256  # noqa: E402

CONFIGS = {"w1": {"weighted": False}, "wr": {"weighted": True}}


def dense_rows(X, idx: np.ndarray) -> np.ndarray:
    sub = X[idx]
    return (sub.toarray() if sp.issparse(sub) else np.asarray(sub)).astype(np.float32, copy=False)


def load_instacart(processed_dir: Path, n_labels: int | None, max_rows: int | None) -> dict:
    f = load_csr_bundle(processed_dir / "features.npz")
    lab = load_csr_bundle(processed_dir / "labels_all.npz")
    meta = json.loads((processed_dir / "label_meta.json").read_text(encoding="utf-8"))
    product_id = np.asarray(meta["product_id"], dtype=np.int64)
    top_idx = np.asarray(meta["top_n"]["idx"], dtype=np.int64)
    Y = {k: lab[k] for k in ("Y_tr", "Y_va", "Y_te")}
    if n_labels is not None:  # smoke: the all-label set becomes the first n_labels top-N products
        cols = top_idx[:n_labels]
        Y = {k: v[:, cols].tocsr() for k, v in Y.items()}
        names_all = product_id[cols]
        top_pos = np.arange(len(cols))
    else:
        names_all = product_id
        top_pos = top_idx
    d = {"X_tr": f["X_tr"], "X_va": f["X_va"], "X_te": f["X_te"], **Y, "label_names_all": names_all, "top_pos": top_pos, "n_features": int(f["X_tr"].shape[1])}
    if max_rows is not None:
        for k in ("X_tr", "X_va", "X_te", "Y_tr", "Y_va", "Y_te"):
            d[k] = d[k][:max_rows]
    d["inputs"] = {"features.npz": sha256(processed_dir / "features.npz"), "labels_all.npz": sha256(processed_dir / "labels_all.npz")}
    d["split_strategy"] = json.loads((processed_dir / "split.json").read_text(encoding="utf-8"))["basis_rule"]
    return d


def load_santander(cache_dir: Path, max_rows: int | None) -> dict:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    d: dict = {}
    for k in ("X_tr", "X_va", "X_te"):
        d[k] = np.load(cache_dir / f"{k}.npy", mmap_mode="r" if max_rows else None)
    for k in ("Y_tr", "Y_va", "Y_te"):
        d[k] = sp.csr_matrix(np.load(cache_dir / f"{k}.npy"))
    if max_rows is not None:
        for k in ("X_tr", "X_va", "X_te", "Y_tr", "Y_va", "Y_te"):
            d[k] = np.ascontiguousarray(d[k][:max_rows]) if k.startswith("X") else d[k][:max_rows]
    d["label_names_all"] = np.asarray(meta["label_names"])
    d["top_pos"] = np.arange(len(meta["label_names"]))
    d["n_features"] = int(d["X_tr"].shape[1])
    d["inputs"] = {f"{k}.npy": sha256(cache_dir / f"{k}.npy") for k in ("X_tr", "Y_tr")}
    d["split_strategy"] = meta.get("split_strategy", "time-based via fecha_dato")
    return d


def fit_predict(data: dict, cfg: dict, args: argparse.Namespace, out_dir: Path) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    n, d = X_tr.shape
    L = Y_tr.shape[1]
    n_pos = np.asarray(Y_tr.sum(0)).ravel().astype(np.int64)
    dead = np.flatnonzero(n_pos == 0)
    w = np.ones(L, dtype=np.float32)
    if cfg["weighted"]:
        alive = n_pos > 0
        w[alive] = ((n - n_pos[alive]) / n_pos[alive]).astype(np.float32)
    scaler = StandardScaler(with_mean=False).fit(X_tr)
    inv_scale = np.where(scaler.scale_ > 0, 1.0 / scaler.scale_, 1.0).astype(np.float32)

    net = torch.nn.Sequential(
        torch.nn.Linear(d, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, L)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(w))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    bs = min(args.batch_size, n)
    epoch_losses = []
    t0 = time.time()
    net.train()
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for b0 in range(0, n, bs):
            idx = np.sort(perm[b0 : b0 + bs])
            xb = torch.from_numpy(dense_rows(X_tr, idx) * inv_scale)
            yb = torch.from_numpy(dense_rows(Y_tr, idx))
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        epoch_losses.append(tot / cnt)
        log(f"  epoch {ep + 1}/{args.epochs} loss {tot / cnt:.5f} elapsed {time.time() - t0:.0f}s")
    fit_sec = time.time() - t0

    net.eval()
    top_pos = np.asarray(data["top_pos"], dtype=np.int64)
    out: dict = {"n_pos": n_pos, "w": w, "dead": dead, "epoch_losses": epoch_losses, "fit_sec": fit_sec}
    for split, key in (("va", "X_va"), ("te", "X_te")):
        X = data[key]
        m = X.shape[0]
        p_top = np.empty((m, len(top_pos)), np.float32)
        mm = None
        if args.save_all and args.dataset == "instacart":
            mm = np.lib.format.open_memmap(out_dir / f"p_{split}_all.partial.npy", mode="w+", dtype=np.float32, shape=(L, m))
        t1 = time.time()
        with torch.no_grad():
            for r0 in range(0, m, args.predict_batch):
                r1 = min(r0 + args.predict_batch, m)
                xb = torch.from_numpy(dense_rows(X, np.arange(r0, r1)) * inv_scale)
                pb = torch.sigmoid(net(xb)).numpy().astype(np.float32)
                p_top[r0:r1] = pb[:, top_pos]
                if mm is not None:
                    mm[:, r0:r1] = pb.T
        if mm is not None:
            mm.flush()
            del mm
            os.replace(out_dir / f"p_{split}_all.partial.npy", out_dir / f"p_{split}_all.npy")
        out[f"p_{split}"] = p_top
        log(f"  predicted {split}: {m:,} rows in {time.time() - t1:.0f}s")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Shared-trunk MLP with per-label pos_weight")
    p.add_argument("--dataset", choices=("instacart", "santander"), default="instacart")
    p.add_argument("--processed-dir", type=Path, default=HERE / "data" / "processed")
    p.add_argument("--features-cache", type=Path, default=HERE.parents[1] / "data" / "santander" / "outputs_matched_pair" / "features_cache")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--configs", type=str, default="w1,wr")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--predict-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--max-rows", type=int, default=None, help="smoke: first rows of every split")
    p.add_argument("--n-labels", type=int, default=None, help="smoke (instacart): first k top-N products as the label set")
    p.add_argument("--no-save-all", dest="save_all", action="store_false", help="skip the all-label memmaps")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    redirect_log(args.log)
    if args.output_dir is None:
        args.output_dir = HERE / "outputs_mlp" if args.dataset == "instacart" else HERE.parents[1] / "data" / "santander" / "outputs_mlp"
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs: {unknown}; choose from {list(CONFIGS)}")

    t0 = time.time()
    data = load_instacart(args.processed_dir, args.n_labels, args.max_rows) if args.dataset == "instacart" else load_santander(args.features_cache, args.max_rows)
    L = data["Y_tr"].shape[1]
    top_pos = np.asarray(data["top_pos"], dtype=np.int64)
    names_all = np.asarray(data["label_names_all"])
    log(f"{args.dataset}: train/val/test rows {data['X_tr'].shape[0]:,}/{data['X_va'].shape[0]:,}/{data['X_te'].shape[0]:,}, features {data['n_features']}, labels {L} (top-N columns {len(top_pos)}), loaded in {time.time() - t0:.0f}s")
    Y_va_top = np.asarray(data["Y_va"][:, top_pos].todense()).astype(np.int8)
    Y_te_top = np.asarray(data["Y_te"][:, top_pos].todense()).astype(np.int8)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cname in configs:
        cfg = CONFIGS[cname]
        out_dir = args.output_dir / cname
        out_npz = out_dir / "predictions.npz"
        if out_npz.exists():
            log(f"[{cname}] already done ({out_npz}), skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[{cname}] weighted={cfg['weighted']} hidden={args.hidden} epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.weight_decay} seed={args.seed} threads={args.threads}")
        r = fit_predict(data, cfg, args, out_dir)
        atomic_savez(out_npz, compressed=True, p_te=r["p_te"], p_va=r["p_va"], Y_te=Y_te_top, Y_va=Y_va_top)
        per_label = [{"label": names_all[j].item() if hasattr(names_all[j], "item") else names_all[j], "n_pos": int(r["n_pos"][j]), "pos_weight": float(r["w"][j])} for j in top_pos]
        meta = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": args.dataset,
            "config": cname,
            "learner": "shared-trunk MLP, BCEWithLogitsLoss(pos_weight per label)",
            "weighted": bool(cfg["weighted"]),
            "pos_weight": "n_neg/n_pos per label on the fit rows (1 for labels without positives)" if cfg["weighted"] else "1 for every label",
            "n_train": int(data["X_tr"].shape[0]),
            "n_val": int(data["X_va"].shape[0]),
            "n_test": int(data["X_te"].shape[0]),
            "n_features": int(data["n_features"]),
            "n_labels": int(len(top_pos)),
            "n_labels_all": int(L),
            "top_label_pos": top_pos.tolist(),
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stopping": False,
            "random_state": int(args.seed),
            "threads": int(args.threads),
            "scaler": "StandardScaler(with_mean=False) fitted on the train rows",
            "split_strategy": data["split_strategy"],
            "max_rows": args.max_rows,
            "n_labels_smoke": args.n_labels,
            "all_label_arrays": (args.save_all and args.dataset == "instacart"),
            "inputs": data["inputs"],
            "epoch_losses": [round(x, 6) for x in r["epoch_losses"]],
            "total_fit_sec": round(r["fit_sec"], 1),
            "dead_in_train_idx": r["dead"].tolist(),
            "label_names": [d["label"] for d in per_label],
            "per_label": per_label,
            "n_pos_per_label_all": r["n_pos"].tolist(),
            "label_names_all": [x.item() if hasattr(x, "item") else x for x in names_all],
        }
        atomic_write_json(out_dir / "run_meta.json", meta)
        log(f"[{cname}] wrote {out_npz} (fit {r['fit_sec']:.0f}s)")
    log("done")


if __name__ == "__main__":
    main()
