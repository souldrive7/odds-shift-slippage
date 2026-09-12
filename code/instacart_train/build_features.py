"""Feature and label matrices for the Instacart product-level experiment.

Row = (user, basis order t). Features use orders 1..t-1 only (leak rule, asserted); the label
row is the content of order t. Two label sets: the top-N products (train positives >= --min-pos,
capped at --max-labels; LightGBM one-vs-rest) and all products (shared-trunk MLP).

Feature matrix (CSR float32, common to every label, as on Santander):
  dense block (320)   n_prior_orders, n_items_total, mean/std basket size, mean/std days since
                      prior, reorder_frac, n_unique_products, mode_dow, mode_hour,
                      aisle_cnt[134], aisle_frac[134], dept_cnt[21], dept_frac[21]
  product block (3N)  for each top-N product: prior purchase count, orders since last purchase,
                      purchase-order ratio  (block-major: [cnt | since | ratio])

Outputs (data/processed/, gitignored):
  features.npz, labels_topN.npz, labels_all.npz   CSR bundles (X_tr/X_va/X_te, Y_tr/Y_va/Y_te)
  split.npz + split.json, feature_names.json, label_meta.json

Usage:
  python code/instacart_train/build_features.py --max-users 5000 --self-check --out-dir data/processed_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from instacart_io import (  # noqa: E402
    Bases,
    atomic_savez,
    atomic_write_json,
    load_or_build_cache,
    log,
    make_all_prior_bases,
    make_default_bases,
    redirect_log,
    save_bases,
    select_users,
    subset_cache,
)

DENSE_SCALARS = (
    "n_prior_orders",
    "n_items_total",
    "mean_basket_size",
    "std_basket_size",
    "mean_days_since_prior",
    "std_days_since_prior",
    "reorder_frac",
    "n_unique_products",
    "mode_dow",
    "mode_hour",
)


# --------------------------------------------------------------------------------------
# index maps
# --------------------------------------------------------------------------------------


def order_index_map(cache: dict) -> np.ndarray:
    """om[u, t] = row of order (u, t) in the sorted orders table, -1 if absent. Shape (n_users, T_max+2)."""
    n_users = len(cache["n_u"])
    t_max = int(cache["n_u"].max())
    om = np.full((n_users, t_max + 2), -1, dtype=np.int64)
    om[cache["orders_user_idx"], cache["orders_order_number"]] = np.arange(len(cache["orders_user_idx"]))
    return om


def basis_row_map(cache: dict, bases: Bases) -> np.ndarray:
    """rm[u, t] = row index of basis (u, t), -1 if that pair is not a row."""
    n_users = len(cache["n_u"])
    t_max = int(cache["n_u"].max())
    rm = np.full((n_users, t_max + 2), -1, dtype=np.int64)
    rm[bases.user_idx, bases.t] = np.arange(bases.n_rows)
    return rm


def basis_slots(cache: dict, bases: Bases) -> np.ndarray:
    """slots[u, s] = the s-th basis t of user u (ascending), -1 padding."""
    n_users = len(cache["n_u"])
    order = np.lexsort((bases.t, bases.user_idx))
    u = bases.user_idx[order]
    t = bases.t[order]
    first = np.concatenate([[True], u[1:] != u[:-1]])
    grp_start = np.flatnonzero(first)
    grp_id = np.cumsum(first) - 1
    slot = np.arange(len(u)) - grp_start[grp_id]
    n_slots = int(slot.max()) + 1
    slots = np.full((n_users, n_slots), -1, dtype=np.int16)
    slots[u, slot] = t
    return slots


# --------------------------------------------------------------------------------------
# dense block: per-user aggregates over orders 1..t-1
# --------------------------------------------------------------------------------------


def build_dense_block(cache: dict, bases: Bases, om: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    n_orders = len(cache["orders_user_idx"])
    n_aisles = len(cache["aisle_id"])
    n_depts = len(cache["department_id"])
    irow = om[cache["items_user_idx"], cache["items_order_number"]]
    assert (irow >= 0).all()
    n_items = np.bincount(irow, minlength=n_orders).astype(np.int32)
    n_reord = np.bincount(irow, weights=cache["items_reordered"].astype(np.float64), minlength=n_orders).astype(np.int32)
    n_new = n_items - n_reord  # reordered == 0 <=> first purchase of that product by the user
    aisle_cnt = sp.coo_matrix(
        (np.ones(len(irow), np.int32), (irow, cache["product_aisle_idx"][cache["items_product_idx"]].astype(np.int64))),
        shape=(n_orders, n_aisles),
    ).toarray()
    dept_cnt = sp.coo_matrix(
        (np.ones(len(irow), np.int32), (irow, cache["product_dept_idx"][cache["items_product_idx"]].astype(np.int64))),
        shape=(n_orders, n_depts),
    ).toarray()
    onum = cache["orders_order_number"]
    has_days = onum >= 2
    d = np.where(has_days, cache["orders_days_since"], np.float32(0)).astype(np.float64)
    dow = np.zeros((n_orders, 7), np.int32)
    dow[np.arange(n_orders), cache["orders_dow"].astype(np.int64)] = 1
    hour = np.zeros((n_orders, 24), np.int32)
    hour[np.arange(n_orders), cache["orders_hour"].astype(np.int64)] = 1

    first = om[bases.user_idx, 1]
    last = om[bases.user_idx, bases.t - 1]
    assert (bases.t >= 2).all(), "a basis order needs at least one earlier order"
    assert (last >= first).all()
    max_order_used = int((bases.t - 1).max())

    def prefix(stat: np.ndarray) -> np.ndarray:
        """sum of stat over orders 1..t-1 of the row's user (orders are contiguous and sorted)."""
        c = np.cumsum(stat, axis=0)
        return c[last] - c[first] + stat[first]

    n_prior = (bases.t - 1).astype(np.float64)
    s_items = prefix(n_items.astype(np.int64)).astype(np.float64)
    s_items2 = prefix((n_items.astype(np.int64)) ** 2).astype(np.float64)
    s_reord = prefix(n_reord.astype(np.int64)).astype(np.float64)
    s_new = prefix(n_new.astype(np.int64)).astype(np.float64)
    s_days = prefix(d)
    s_days2 = prefix(d * d)
    n_days = np.maximum(bases.t.astype(np.float64) - 2, 0)  # orders 2..t-1 carry days_since
    mean_b = s_items / n_prior
    std_b = np.sqrt(np.maximum(s_items2 / n_prior - mean_b**2, 0))
    mean_d = np.where(n_days > 0, s_days / np.maximum(n_days, 1), 0)
    std_d = np.sqrt(np.maximum(np.where(n_days > 0, s_days2 / np.maximum(n_days, 1), 0) - mean_d**2, 0))
    reorder_frac = s_reord / np.maximum(s_items, 1)
    mode_dow = np.argmax(prefix(dow), axis=1).astype(np.float64)
    mode_hour = np.argmax(prefix(hour), axis=1).astype(np.float64)
    a_cnt = prefix(aisle_cnt).astype(np.float64)
    dp_cnt = prefix(dept_cnt).astype(np.float64)
    a_frac = a_cnt / np.maximum(s_items, 1)[:, None]
    dp_frac = dp_cnt / np.maximum(s_items, 1)[:, None]
    X = np.column_stack(
        [n_prior, s_items, mean_b, std_b, mean_d, std_d, reorder_frac, s_new, mode_dow, mode_hour, a_cnt, a_frac, dp_cnt, dp_frac]
    ).astype(np.float32)
    names = list(DENSE_SCALARS)
    names += [f"aisle_cnt[{int(a)}]" for a in cache["aisle_id"]]
    names += [f"aisle_frac[{int(a)}]" for a in cache["aisle_id"]]
    names += [f"dept_cnt[{int(a)}]" for a in cache["department_id"]]
    names += [f"dept_frac[{int(a)}]" for a in cache["department_id"]]
    assert X.shape[1] == len(names)
    return X, names, max_order_used


# --------------------------------------------------------------------------------------
# product block: per (row, product) history from purchase events
# --------------------------------------------------------------------------------------


def purchase_events(cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (user, product, order_number) triples sorted by (user, product, order)."""
    u = cache["items_user_idx"].astype(np.int64)
    j = cache["items_product_idx"].astype(np.int64)
    o = cache["items_order_number"].astype(np.int64)
    order = np.lexsort((o, j, u))
    u, j, o = u[order], j[order], o[order]
    dup = np.concatenate([[False], (u[1:] == u[:-1]) & (j[1:] == j[:-1]) & (o[1:] == o[:-1])])
    return u[~dup], j[~dup], o[~dup]


PRODUCT_BLOCKS = ("cnt", "since", "ratio")


def build_product_block(
    cache: dict, bases: Bases, rm: np.ndarray, product_sel: np.ndarray | None, blocks: tuple[str, ...] = PRODUCT_BLOCKS
) -> tuple[sp.csr_matrix, int]:
    """CSR (n_rows, len(blocks)*P) with blocks from [cnt | since | ratio] over ``product_sel``
    (sorted product indices; None = all products). Entry (row, j) exists iff the user bought j
    before order t. ``blocks`` drops columns to trade LightGBM split-finding time for features."""
    u, j, o = purchase_events(cache)
    if product_sel is not None:
        keep = np.isin(j, product_sel)
        u, j, o = u[keep], j[keep], o[keep]
        j = np.searchsorted(product_sel, j)
        P = int(len(product_sel))
    else:
        P = int(len(cache["product_id"]))
    n = len(u)
    same_prev = np.concatenate([[False], (u[1:] == u[:-1]) & (j[1:] == j[:-1])])
    grp_start = np.flatnonzero(~same_prev)
    grp_id = np.cumsum(~same_prev) - 1
    k = np.arange(n) - grp_start[grp_id]  # 0-based purchase index within (u, j)
    same_next = np.concatenate([same_prev[1:], [False]])
    o_next = np.where(same_next, np.concatenate([o[1:], [0]]), cache["n_u"][u].astype(np.int64) + 1)
    slots = basis_slots(cache, bases)
    rows, cols, cnt, since, ratio = [], [], [], [], []
    max_order_used = 0
    for s in range(slots.shape[1]):
        t_u = slots[u, s].astype(np.int64)
        m = (t_u >= 2) & (o < t_u) & (t_u <= o_next)
        if not m.any():
            continue
        r = rm[u[m], t_u[m]]
        assert (r >= 0).all()
        rows.append(r)
        cols.append(j[m])
        cnt.append((k[m] + 1).astype(np.float32))
        since.append((t_u[m] - o[m]).astype(np.float32))
        ratio.append(((k[m] + 1) / (t_u[m] - 1)).astype(np.float32))
        max_order_used = max(max_order_used, int(o[m].max()))
    values = {"cnt": cnt, "since": since, "ratio": ratio}
    nb = len(blocks)
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        rr = np.concatenate([r] * nb)
        cc = np.concatenate([c + i * P for i in range(nb)])
        vv = np.concatenate([np.concatenate(values[b]) for b in blocks])
        X = sp.coo_matrix((vv, (rr, cc)), shape=(bases.n_rows, nb * P)).tocsr()
    else:
        X = sp.csr_matrix((bases.n_rows, nb * P), dtype=np.float32)
    X.sum_duplicates()
    return X.astype(np.float32), max_order_used


def product_block_names(cache: dict, product_sel: np.ndarray | None, blocks: tuple[str, ...] = PRODUCT_BLOCKS) -> list[str]:
    pid = cache["product_id"] if product_sel is None else cache["product_id"][product_sel]
    return [f"prod_{blk}[{int(p)}]" for blk in blocks for p in pid]


# --------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------


def build_label_block(cache: dict, bases: Bases, rm: np.ndarray) -> sp.csr_matrix:
    """CSR int8 (n_rows, n_products): y = 1 iff product in the basis order t of the row."""
    r = rm[cache["items_user_idx"], cache["items_order_number"]]
    m = r >= 0
    P = int(len(cache["product_id"]))
    Y = sp.coo_matrix(
        (np.ones(int(m.sum()), np.int8), (r[m], cache["items_product_idx"][m].astype(np.int64))),
        shape=(bases.n_rows, P),
    ).tocsr()
    Y.sum_duplicates()
    Y.data[:] = 1
    return Y.astype(np.int8)


def select_top_n(n_pos_train: np.ndarray, min_pos: int, max_labels: int | None) -> tuple[np.ndarray, dict]:
    """Products with >= min_pos train positives, capped at max_labels by frequency. Returned in
    ascending product index order so that column order carries no popularity information."""
    order = np.lexsort((np.arange(len(n_pos_train)), -n_pos_train))  # by n_pos desc, then index
    n_ge = int((n_pos_train >= min_pos).sum())
    n = n_ge if max_labels is None else min(n_ge, int(max_labels))
    cap_applied = max_labels is not None and n_ge > max_labels
    top = np.sort(order[:n]).astype(np.int64)
    info = {
        "rule": "products with >= min_pos positives in the train basis rows; capped at max_labels by frequency",
        "min_pos": int(min_pos),
        "n_products_ge_min_pos": n_ge,
        "max_labels": max_labels,
        "N": int(n),
        "cap_applied": bool(cap_applied),
        "reason": (
            f"compute budget: {n_ge} products satisfy min_pos={min_pos}, kept the {n} most frequent"
            if cap_applied
            else f"all {n} products with >= {min_pos} train positives"
        ),
        "column_order": "ascending product index (arbitrary w.r.t. popularity)",
        "min_pos_in_topN": int(n_pos_train[top].min()) if n else None,
        "max_pos_in_topN": int(n_pos_train[top].max()) if n else None,
    }
    return top, info


# --------------------------------------------------------------------------------------
# CSR bundles
# --------------------------------------------------------------------------------------


def save_csr_bundle(path: Path, **mats: sp.csr_matrix) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, m in mats.items():
        m = sp.csr_matrix(m)
        m.sort_indices()
        arrays[f"{name}_data"] = m.data
        arrays[f"{name}_indices"] = m.indices
        arrays[f"{name}_indptr"] = m.indptr
        arrays[f"{name}_shape"] = np.asarray(m.shape, dtype=np.int64)
    atomic_savez(path, **arrays)


def load_csr_bundle(path: Path, names: tuple[str, ...] | None = None) -> dict[str, sp.csr_matrix]:
    out: dict[str, sp.csr_matrix] = {}
    with np.load(path) as z:
        keys = [k[: -len("_shape")] for k in z.files if k.endswith("_shape")]
        for name in keys:
            if names is not None and name not in names:
                continue
            out[name] = sp.csr_matrix(
                (z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=tuple(z[f"{name}_shape"])
            )
    return out


# --------------------------------------------------------------------------------------
# end-to-end build
# --------------------------------------------------------------------------------------


def build_all(cache: dict, bases: Bases, min_pos: int, max_labels: int | None, blocks: tuple[str, ...] = PRODUCT_BLOCKS) -> dict:
    """Return everything build_features writes, in memory."""
    t0 = time.time()
    for b in blocks:
        assert b in PRODUCT_BLOCKS, b
    om = order_index_map(cache)
    rm = basis_row_map(cache, bases)
    log(f"rows: train/val/test = {bases.counts()}")
    Y_all = build_label_block(cache, bases, rm)
    tr, va, te = bases.rows(0), bases.rows(1), bases.rows(2)
    n_pos_train = np.asarray(Y_all[tr].sum(0)).ravel().astype(np.int64)
    n_pos_val = np.asarray(Y_all[va].sum(0)).ravel().astype(np.int64)
    n_pos_test = np.asarray(Y_all[te].sum(0)).ravel().astype(np.int64)
    top_idx, top_info = select_top_n(n_pos_train, min_pos, max_labels)
    log(f"labels: {top_info['N']} top-N products ({top_info['reason']}); {len(n_pos_train)} products in total")
    X_dense, dense_names, mou_dense = build_dense_block(cache, bases, om)
    log(f"dense block {X_dense.shape} in {time.time() - t0:.0f}s")
    X_prod, mou_prod = build_product_block(cache, bases, rm, top_idx, blocks)
    log(f"product block {X_prod.shape} ({'+'.join(blocks)}) nnz={X_prod.nnz:,} in {time.time() - t0:.0f}s")
    X = sp.hstack([sp.csr_matrix(X_dense), X_prod], format="csr").astype(np.float32)
    names = dense_names + product_block_names(cache, top_idx, blocks)
    max_order_used = max(mou_dense, mou_prod)
    assert max_order_used < int(bases.t.max())
    assert (bases.t - 1 >= 1).all()
    return {
        "X": X,
        "Y_all": Y_all,
        "Y_top": Y_all[:, top_idx].tocsr(),
        "top_idx": top_idx,
        "top_info": top_info,
        "product_blocks": list(blocks),
        "n_pos": {"train": n_pos_train, "val": n_pos_val, "test": n_pos_test},
        "feature_names": names,
        "max_order_used": max_order_used,
        "build_sec": round(time.time() - t0, 1),
    }


def self_check(cache: dict, bases: Bases, built: dict, seed: int = 0) -> dict:
    """Leak test: scramble the products of every item at or after the user's train basis order
    and rebuild; train-row features must be identical, train labels must change."""
    tr = bases.rows(0)
    t_train = np.full(len(cache["n_u"]), 10**6, dtype=np.int64)
    np.minimum.at(t_train, bases.user_idx[tr], bases.t[tr].astype(np.int64))
    pert = dict(cache)
    rng = np.random.default_rng(seed)
    m = cache["items_order_number"].astype(np.int64) >= t_train[cache["items_user_idx"]]
    pidx = cache["items_product_idx"].copy()
    pidx[m] = rng.integers(0, len(cache["product_id"]), size=int(m.sum()), dtype=np.int32)
    pert["items_product_idx"] = pidx
    om = order_index_map(pert)
    rm = basis_row_map(pert, bases)
    Xd, _, _ = build_dense_block(pert, bases, om)
    Xp, _ = build_product_block(pert, bases, rm, built["top_idx"], tuple(built["product_blocks"]))
    X2 = sp.hstack([sp.csr_matrix(Xd), Xp], format="csr").astype(np.float32)
    Y2 = build_label_block(pert, bases, rm)
    x_same = (built["X"][tr] != X2[tr]).nnz == 0
    y_changed = (built["Y_all"][tr] != Y2[tr]).nnz > 0
    return {"train_features_unchanged": bool(x_same), "train_labels_changed": bool(y_changed), "n_items_scrambled": int(m.sum())}


def write_processed(out_dir: Path, cache: dict, bases: Bases, built: dict, args_dict: dict, extra_split: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = bases.rows(0), bases.rows(1), bases.rows(2)
    X, Y_all, Y_top = built["X"], built["Y_all"], built["Y_top"]
    save_csr_bundle(out_dir / "features.npz", X_tr=X[tr], X_va=X[va], X_te=X[te])
    save_csr_bundle(out_dir / "labels_topN.npz", Y_tr=Y_top[tr], Y_va=Y_top[va], Y_te=Y_top[te])
    save_csr_bundle(out_dir / "labels_all.npz", Y_tr=Y_all[tr], Y_va=Y_all[va], Y_te=Y_all[te])
    save_bases(out_dir / "split.npz", bases)
    atomic_write_json(out_dir / "feature_names.json", {"names": built["feature_names"], "n_dense": len(DENSE_SCALARS) + 2 * (len(cache["aisle_id"]) + len(cache["department_id"]))})
    meta = cache["_meta"]
    pid = cache["product_id"]
    label_meta = {
        "n_products": int(len(pid)),
        "product_id": pid.astype(int).tolist(),
        "product_name": meta["product_names"],
        "aisle_id": cache["aisle_id"][cache["product_aisle_idx"]].astype(int).tolist(),
        "department_id": cache["department_id"][cache["product_dept_idx"]].astype(int).tolist(),
        "n_pos_train": built["n_pos"]["train"].tolist(),
        "n_pos_val": built["n_pos"]["val"].tolist(),
        "n_pos_test": built["n_pos"]["test"].tolist(),
        "top_n": {
            "idx": built["top_idx"].tolist(),
            "product_ids": pid[built["top_idx"]].astype(int).tolist(),
            "info": built["top_info"],
        },
    }
    atomic_write_json(out_dir / "label_meta.json", label_meta)
    split = {
        "generated_at": datetime.now(UTC).isoformat(),
        "basis_rule": "within-user temporal: train t=n_u-2, val t=n_u-1, test t=n_u (eval_set=='train' order)"
        + ("; train rows for every 2<=t<=n_u-2" if args_dict.get("train_all_prior") else ""),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_users": int(len(cache["user_id"])),
        "n_features": int(X.shape[1]),
        "product_blocks": built["product_blocks"],
        "n_labels_topN": int(Y_top.shape[1]),
        "n_labels_all": int(Y_all.shape[1]),
        "args": args_dict,
        "csv_sha256": meta["csv_sha256"],
        "leak_check": {"features_use_orders_lt_t": True, "max_order_used_in_features": int(built["max_order_used"]), "min_basis_t": int(bases.t.min())},
        "build_sec": built["build_sec"],
    }
    if extra_split:
        split.update(extra_split)
    atomic_write_json(out_dir / "split.json", split)
    log(f"wrote {out_dir}: features {X.shape}, top-N labels {Y_top.shape}, all labels {Y_all.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description="Instacart product-level features and labels")
    p.add_argument("--raw-dir", type=Path, default=HERE / "data" / "raw")
    p.add_argument("--cache-dir", type=Path, default=HERE / "data" / "processed")
    p.add_argument("--out-dir", type=Path, default=HERE / "data" / "processed")
    p.add_argument("--min-pos", type=int, default=20)
    p.add_argument("--max-labels", type=int, default=5000)
    p.add_argument("--max-users", type=int, default=None, help="random user subset (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-all-prior", action="store_true")
    p.add_argument("--product-blocks", type=str, default=",".join(PRODUCT_BLOCKS), help="per-product feature blocks among cnt,since,ratio")
    p.add_argument("--self-check", action="store_true", help="leak test by scrambling orders >= train basis")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    redirect_log(args.log)
    cache = load_or_build_cache(args.raw_dir, args.cache_dir, force=args.force_cache)
    sel = select_users(cache, args.max_users, args.seed)
    cache = subset_cache(cache, sel)
    if sel is not None:
        log(f"user subset: {len(sel):,} users (seed {args.seed})")
    bases = make_all_prior_bases(cache["n_u"]) if args.train_all_prior else make_default_bases(cache["n_u"])
    blocks = tuple(b.strip() for b in args.product_blocks.split(",") if b.strip())
    built = build_all(cache, bases, args.min_pos, args.max_labels, blocks)
    extra = None
    if args.self_check:
        chk = self_check(cache, bases, built)
        log(f"self-check: {chk}")
        assert chk["train_features_unchanged"] and chk["train_labels_changed"], chk
        extra = {"leak_self_check": chk}
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_processed(args.out_dir, cache, bases, built, args_dict, extra)
    log("done")


if __name__ == "__main__":
    main()
