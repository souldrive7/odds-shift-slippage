"""All-product evaluation of the Instacart MLP (49,688 labels) from the label-major memmaps
written by train_mlp_posweight.py (p_te_all.npy / p_va_all.npy, float32, shape (L, n)).

The full score matrix (131k x 49,688 float64 = 52 GB) never exists in memory: labels are
processed in chunks of --chunk labels, and the top-K per row is maintained by a merge that
reproduces the stable-argsort order of the full matrix (ties broken by the lower column index).

Usage:
    uv run python code/chunked.py --config wr --part ladder
Writes results/instacart_mlp_all_<part>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as rx  # noqa: E402
from datasets import EXP, sha256  # noqa: E402

K = rx.K


class TopKAccumulator:
    """Running top-K (score, column) per row over column chunks, with the tie order of a stable
    argsort on -score over the full matrix: larger score first, then lower column index."""

    def __init__(self, n_rows: int, k: int = K) -> None:
        self.n = n_rows
        self.k = k
        self.score = np.full((n_rows, k), -np.inf, dtype=np.float64)
        self.col = np.full((n_rows, k), np.iinfo(np.int64).max, dtype=np.int64)

    def update(self, block: np.ndarray, j0: int) -> None:
        m = block.shape[1]
        cols = np.arange(j0, j0 + m, dtype=np.int64)
        s = np.concatenate([self.score, block.astype(np.float64)], axis=1)
        c = np.concatenate([self.col, np.tile(cols, (self.n, 1))], axis=1)
        # lexsort: primary key -score (descending score), secondary key column index ascending
        order = np.lexsort((c, -s), axis=1)[:, : self.k]
        self.score = np.take_along_axis(s, order, 1)
        self.col = np.take_along_axis(c, order, 1)

    def row_ap(self, y_csr) -> tuple[np.ndarray, np.ndarray]:
        """Per-row AP@K and the row mask (rows with at least one positive), from a CSR label matrix."""
        n_pos = np.asarray(y_csr.sum(1)).ravel()
        keep = n_pos > 0
        rel = np.zeros((self.n, self.k), dtype=np.float64)
        for kk in range(self.k):
            rel[:, kk] = np.asarray(y_csr[np.arange(self.n), self.col[:, kk]]).ravel()
        prec = np.cumsum(rel, 1) / np.arange(1, self.k + 1)
        ap = (prec * rel).sum(1) / np.maximum(np.minimum(n_pos, self.k), 1)
        return ap[keep], keep


def load_labels_all(processed: Path):
    import scipy.sparse as sp

    with np.load(processed / "labels_all.npz") as z:
        out = {}
        for name in ("Y_tr", "Y_va", "Y_te"):
            out[name] = sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=tuple(z[f"{name}_shape"]))
    return out


def part_ladder(config: str, out_dir: Path, processed: Path, chunk: int, n_jobs: int) -> dict:
    """All-label ladder for one MLP config: raw, per-label iso (prior fallback), label-free shift,
    Elkan inversion, saturation and within-label AUC over all products; MAP@7 via the accumulator."""
    t0 = time.time()
    rx.N_JOBS = n_jobs
    meta = json.loads((out_dir / config / "run_meta.json").read_text(encoding="utf-8"))
    labs = load_labels_all(processed)
    Y_te, Y_va = labs["Y_te"].tocsc(), labs["Y_va"].tocsc()
    n_te, L = Y_te.shape
    n_va = Y_va.shape[0]
    n_train = int(meta["n_train"])
    n_pos = np.asarray(meta["n_pos_per_label_all"], dtype=np.float64)
    pi_train = n_pos / n_train
    w_true = (n_train - n_pos) / np.maximum(n_pos, 1)
    p_te = np.load(out_dir / config / "p_te_all.npy", mmap_mode="r")
    p_va = np.load(out_dir / config / "p_va_all.npy", mmap_mode="r")
    assert p_te.shape == (L, n_te) and p_va.shape == (L, n_va)
    va_idx, te_idx = rx.cal_split(n_te)  # test-split protocol on the test rows (calibrate on 30%)
    acc = {name: TopKAccumulator(n_te) for name in ("raw", "per_label_iso_prior", "per_label_iso_identity", "prior_match_label_free", "elkan_inversion_known_w", "whatif_from_pi", "popularity")}
    acc_dep = {name: TopKAccumulator(n_te) for name in ("raw", "per_label_iso_prior")}  # deployment protocol: calibrate on the validation rows
    sat1 = np.zeros(L)
    sat0 = np.zeros(L)
    auc = np.full(L, np.nan)
    b_prior = np.zeros(L)
    dead_te = np.zeros(L, dtype=bool)
    dead_va = np.zeros(L, dtype=bool)
    y_te_sub = None
    for j0 in range(0, L, chunk):
        j1 = min(j0 + chunk, L)
        q = np.ascontiguousarray(p_te[j0:j1].T).astype(np.float64)  # (n_te, m)
        qv = np.ascontiguousarray(p_va[j0:j1].T).astype(np.float64)
        y = np.asarray(Y_te[:, j0:j1].todense()).astype(np.int8)
        yv = np.asarray(Y_va[:, j0:j1].todense()).astype(np.int8)
        pi = pi_train[j0:j1]
        w = w_true[j0:j1]
        sat1[j0:j1] = (q >= 1.0).mean(0)
        sat0[j0:j1] = (q <= 0.0).mean(0)
        if y_te_sub is None:
            y_te_sub = np.random.default_rng(0).choice(n_te, min(200_000, n_te), replace=False)
        for jj in range(j1 - j0):
            ys = y[y_te_sub, jj]
            if 0 < ys.sum() < len(ys):
                from sklearn.metrics import roc_auc_score

                auc[j0 + jj] = roc_auc_score(ys, q[y_te_sub, jj])
        acc["raw"].update(q, j0)
        acc["popularity"].update(np.tile(pi, (n_te, 1)), j0)
        cal, _, dead = rx.per_label_iso_multi(q, y, va_idx, pi, ("prior", "identity"))
        dead_te[j0 + np.asarray(dead, dtype=np.int64)] = True
        acc["per_label_iso_prior"].update(cal["prior"], j0)
        acc["per_label_iso_identity"].update(cal["identity"], j0)
        del cal
        pm, b = rx.prior_match_shift(q, pi)
        b_prior[j0:j1] = b
        acc["prior_match_label_free"].update(pm, j0)
        del pm
        acc["elkan_inversion_known_w"].update(rx.elkan_inversion(q, w), j0)
        acc["whatif_from_pi"].update(rx.sigmoid(rx.logit(np.tile(pi, (n_te, 1))) + np.log(w)), j0)
        # deployment protocol: fit on the validation rows, apply to all test rows
        stack = np.vstack([qv, q])
        ystack = np.vstack([yv, y])
        cal, _, dead_v = rx.per_label_iso_multi(stack, ystack, np.arange(n_va), pi, ("prior",))
        dead_va[j0 + np.asarray(dead_v, dtype=np.int64)] = True
        acc_dep["raw"].update(q, j0)
        acc_dep["per_label_iso_prior"].update(cal["prior"][n_va:], j0)
        del cal, stack, ystack, q, qv, y, yv
        rx._log(f"[mlp_all:{config}] labels {j1}/{L} elapsed {time.time() - t0:.0f}s")
    y_te_csr = labs["Y_te"]
    res: dict = {"config": config, "L": int(L), "n_test": int(n_te), "n_val": int(n_va), "n_train": n_train}
    maps = {}
    for name, a in acc.items():
        ap, keep = a.row_ap(y_te_csr)
        maps[name] = {"map7_all": float(ap.mean()), "map7_te": float(ap[keep[te_idx] if False else np.isin(np.flatnonzero(keep), te_idx)].mean())}
    res["map7"] = maps
    res["deploy"] = {name: {"map7_all": float(a.row_ap(y_te_csr)[0].mean())} for name, a in acc_dep.items()}
    res["saturation"] = {"frac_exact_1": float(sat1.mean()), "frac_exact_0": float(sat0.mean()), "frac_exact_1_per_label": sat1.tolist()}
    res["within_label_auc_mean_sub200k"] = float(np.nanmean(auc))
    res["n_labels_with_auc"] = int(np.isfinite(auc).sum())
    res["dead_labels"] = {"n_dead_in_cal_split": int(dead_te.sum()), "n_dead_in_val": int(dead_va.sum()), "n_labels_zero_train_pos": int((n_pos == 0).sum())}
    res["prior_match_shift_label_free"] = {"shift_b": b_prior.tolist(), "ln_w": np.log(w_true).tolist(), "n_at_search_bound": int((np.abs(b_prior) >= 39.9).sum())}
    res["w_true"] = {"min": float(w_true[n_pos > 0].min()), "max": float(w_true[n_pos > 0].max()), "median": float(np.median(w_true[n_pos > 0]))}
    res["_meta"] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "instacart_mlp_all",
        "inputs": {f"{config}/p_te_all.npy": sha256(out_dir / config / "p_te_all.npy"), f"{config}/p_va_all.npy": sha256(out_dir / config / "p_va_all.npy")},
        "chunk": chunk,
        "k": K,
        "cal_split": {"seed": rx.CAL_SEED, "frac": rx.CAL_FRAC},
        "map_definition": "Kaggle MAP@7 over all products, rows with no positives excluded, ties by lower column index (same as a stable argsort of the full matrix)",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="wr")
    ap.add_argument("--part", choices=("ladder",), default="ladder")
    ap.add_argument("--outputs-dir", type=Path, default=EXP / "data" / "instacart" / "outputs_mlp")
    ap.add_argument("--processed-dir", type=Path, default=EXP / "data" / "instacart" / "data" / "processed")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=rx.OUT_DIR)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(args.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    res = part_ladder(args.config, args.outputs_dir, args.processed_dir, args.chunk, args.n_jobs)
    out = args.out_dir / f"instacart_mlp_all_{args.part}_{args.config}.json"
    rx.atomic_write_json(out, res)
    rx._log(f"wrote {out}")


if __name__ == "__main__":
    main()
