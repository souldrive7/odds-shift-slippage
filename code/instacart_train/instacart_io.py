"""Shared I/O for the Instacart product-level experiment (arXiv v3).

One pass over the Kaggle CSVs builds a compact NumPy cache (``data/processed/cache_items.npz``)
that ``eda.py``, ``build_features.py`` and ``train_mlp_posweight.py`` all consume, so the
32M-row ``order_products__prior.csv`` is read once per CSV version.

Row unit of every downstream matrix: one (user, basis order t) pair. Features for basis t use
orders 1..t-1 only; the label is the content of order t. Users whose last order is unlabelled
(``eval_set == 'test'``) are dropped here (README section 3.2).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = (
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
)
KAGGLE_ZIP = "instacart-market-basket-analysis.zip"
CACHE_NPZ = "cache_items.npz"
CACHE_META = "cache_items.meta.json"
SHA_FILE = "SHA256SUMS.txt"


def log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def redirect_log(path: Path | None) -> None:
    """Under pythonw.exe there is no console: send stdout/stderr to a line-buffered file."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115


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


def atomic_savez(path: Path, compressed: bool = False, **arrays: np.ndarray) -> None:
    # The temp name must itself end in .npz -- np.savez appends .npz otherwise.
    tmp = path.with_name(path.stem + ".partial.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# raw data: zip extraction and checksums
# --------------------------------------------------------------------------------------


def _extract_member(data_src, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".partial")
    with open(tmp, "wb") as dst:
        shutil.copyfileobj(data_src, dst)
    os.replace(tmp, target)


def extract_zip(zip_path: Path, raw_dir: Path, force: bool = False) -> dict:
    """Extract the Kaggle competition zip. The competition file is a zip of ``*.csv.zip``
    members; plain ``*.csv`` members are handled too. Existing CSVs are kept unless ``force``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"zip": zip_path.name, "members": [], "nested_zip_members": 0, "extracted": []}
    with zipfile.ZipFile(zip_path) as outer:
        for m in outer.namelist():
            base = Path(m).name
            if m.startswith("__MACOSX") or not base:
                continue
            info["members"].append(m)
            if base.endswith(".csv"):
                target = raw_dir / base
                if target.exists() and not force:
                    continue
                with outer.open(m) as src:
                    _extract_member(src, target)
                info["extracted"].append(base)
            elif base.endswith(".zip"):
                info["nested_zip_members"] += 1
                with zipfile.ZipFile(io.BytesIO(outer.read(m))) as inner:
                    for im in inner.namelist():
                        ibase = Path(im).name
                        if im.startswith("__MACOSX") or not ibase.endswith(".csv"):
                            continue
                        target = raw_dir / ibase
                        if target.exists() and not force:
                            continue
                        with inner.open(im) as src:
                            _extract_member(src, target)
                        info["extracted"].append(ibase)
    missing = [f for f in RAW_FILES if not (raw_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"after extracting {zip_path}: missing {missing}")
    return info


def write_sha256sums(raw_dir: Path, names: tuple[str, ...] = RAW_FILES + (KAGGLE_ZIP,)) -> dict[str, dict]:
    """SHA-256 + byte size of the raw files; also written to data/raw/SHA256SUMS.txt (LF)."""
    out: dict[str, dict] = {}
    lines = []
    for name in names:
        p = raw_dir / name
        if not p.is_file():
            continue
        h = sha256(p)
        out[name] = {"sha256": h, "bytes": p.stat().st_size}
        lines.append(f"{h}  {name}\n")
    tmp = raw_dir / (SHA_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    os.replace(tmp, raw_dir / SHA_FILE)
    return out


# --------------------------------------------------------------------------------------
# cache: one pass over the CSVs
# --------------------------------------------------------------------------------------

CACHE_KEYS = (
    "user_id",
    "n_u",
    "train_order_id",
    "orders_user_idx",
    "orders_order_number",
    "orders_dow",
    "orders_hour",
    "orders_days_since",
    "items_user_idx",
    "items_order_number",
    "items_product_idx",
    "items_reordered",
    "product_id",
    "product_aisle_idx",
    "product_dept_idx",
    "aisle_id",
    "department_id",
)


def _csv_hashes(raw_dir: Path) -> dict[str, str]:
    return {f: sha256(raw_dir / f) for f in RAW_FILES}


def load_or_build_cache(raw_dir: Path, cache_dir: Path, force: bool = False) -> dict:
    """Return the cache dict (arrays under CACHE_KEYS plus ``_meta``). Rebuilds when the CSV
    hashes differ from the recorded ones."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / CACHE_NPZ
    meta_path = cache_dir / CACHE_META
    hashes = _csv_hashes(raw_dir)
    if npz_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("csv_sha256") == hashes:
            log(f"cache hit: {npz_path}")
            with np.load(npz_path) as z:
                cache = {k: z[k] for k in CACHE_KEYS}
            cache["_meta"] = meta
            return cache
        log("cache stale (CSV hashes differ); rebuilding")
    return _build_cache(raw_dir, cache_dir, hashes)


def _build_cache(raw_dir: Path, cache_dir: Path, hashes: dict[str, str]) -> dict:
    t0 = time.time()
    log("loading products / aisles / departments")
    products = pd.read_csv(raw_dir / "products.csv")
    aisles = pd.read_csv(raw_dir / "aisles.csv")
    departments = pd.read_csv(raw_dir / "departments.csv")
    product_id = np.sort(products["product_id"].to_numpy(np.int32))
    aisle_id = np.sort(aisles["aisle_id"].to_numpy(np.int16))
    department_id = np.sort(departments["department_id"].to_numpy(np.int8))
    prod_sorted = products.set_index("product_id").loc[product_id]
    product_aisle_idx = np.searchsorted(aisle_id, prod_sorted["aisle_id"].to_numpy(np.int16)).astype(np.int16)
    product_dept_idx = np.searchsorted(department_id, prod_sorted["department_id"].to_numpy(np.int8)).astype(np.int8)
    product_names = prod_sorted["product_name"].astype(str).tolist()
    aisle_names = aisles.set_index("aisle_id").loc[aisle_id]["aisle"].astype(str).tolist()
    dept_names = departments.set_index("department_id").loc[department_id]["department"].astype(str).tolist()

    log("loading orders.csv")
    orders = pd.read_csv(
        raw_dir / "orders.csv",
        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "order_number": "int16",
            "order_dow": "int8",
            "order_hour_of_day": "int8",
            "days_since_prior_order": "float32",
        },
    )
    n_orders_all = len(orders)
    n_users_all = int(orders["user_id"].nunique())
    eval_counts = orders["eval_set"].value_counts().to_dict()
    train_rows = orders[orders["eval_set"] == "train"]
    assert train_rows["user_id"].is_unique, "a user has more than one eval_set=='train' order"
    users = np.sort(train_rows["user_id"].to_numpy(np.int32))
    n_users = len(users)
    log(f"users with a labelled last order: {n_users:,} (of {n_users_all:,}); eval_set counts {eval_counts}")

    sel = orders[np.isin(orders["user_id"].to_numpy(), users)]
    sel = sel.sort_values(["user_id", "order_number"], kind="stable")
    orders_user_idx = np.searchsorted(users, sel["user_id"].to_numpy(np.int32)).astype(np.int32)
    orders_order_number = sel["order_number"].to_numpy(np.int16)
    orders_dow = sel["order_dow"].to_numpy(np.int8)
    orders_hour = sel["order_hour_of_day"].to_numpy(np.int8)
    days = sel["days_since_prior_order"].to_numpy(np.float32)
    orders_days_since = np.where(np.isnan(days), np.float32(-1.0), days).astype(np.float32)
    # n_u = order_number of the labelled (eval_set == 'train') order; orders 1..n_u must all exist
    n_u = np.zeros(n_users, dtype=np.int16)
    tr_user_idx = np.searchsorted(users, train_rows["user_id"].to_numpy(np.int32))
    n_u[tr_user_idx] = train_rows["order_number"].to_numpy(np.int16)
    train_order_id = np.zeros(n_users, dtype=np.int32)
    train_order_id[tr_user_idx] = train_rows["order_id"].to_numpy(np.int32)
    counts = np.bincount(orders_user_idx, minlength=n_users)
    assert np.array_equal(counts, n_u.astype(np.int64)), "orders 1..n_u are not contiguous for every user"
    assert n_u.min() >= 4, f"min orders per user is {n_u.min()}, expected >= 4 (Kaggle guarantee)"
    del sel

    oid_max = int(orders["order_id"].max())
    oid2user = np.full(oid_max + 1, -1, dtype=np.int32)
    oid2onum = np.zeros(oid_max + 1, dtype=np.int16)
    sel_oid = orders.loc[np.isin(orders["user_id"].to_numpy(), users), ["order_id", "user_id", "order_number"]]
    oid2user[sel_oid["order_id"].to_numpy()] = np.searchsorted(users, sel_oid["user_id"].to_numpy(np.int32))
    oid2onum[sel_oid["order_id"].to_numpy()] = sel_oid["order_number"].to_numpy(np.int16)
    del orders, sel_oid

    def load_items(name: str) -> tuple[np.ndarray, ...]:
        log(f"loading {name}")
        df = pd.read_csv(
            raw_dir / name,
            usecols=["order_id", "product_id", "reordered"],
            dtype={"order_id": "int32", "product_id": "int32", "reordered": "int8"},
        )
        oid = df["order_id"].to_numpy()
        u = oid2user[oid]
        keep = u >= 0
        pidx = np.searchsorted(product_id, df["product_id"].to_numpy(np.int32)[keep]).astype(np.int32)
        log(f"  {name}: {len(df):,} line items, {int(keep.sum()):,} belong to the selected users")
        return len(df), u[keep].astype(np.int32), oid2onum[oid[keep]], pidx, df["reordered"].to_numpy(np.int8)[keep]

    n_prior_all, pu, po, pp, pr = load_items("order_products__prior.csv")
    n_train_all, tu, to, tp, tr = load_items("order_products__train.csv")
    items_user_idx = np.concatenate([pu, tu])
    items_order_number = np.concatenate([po, to])
    items_product_idx = np.concatenate([pp, tp])
    items_reordered = np.concatenate([pr, tr])
    order = np.lexsort((items_order_number, items_user_idx))
    items_user_idx = items_user_idx[order]
    items_order_number = items_order_number[order]
    items_product_idx = items_product_idx[order]
    items_reordered = items_reordered[order]
    del pu, po, pp, pr, tu, to, tp, tr, order
    # every labelled order must have at least one line item; every item order_number within 1..n_u
    assert items_order_number.min() >= 1
    assert (items_order_number <= n_u[items_user_idx]).all()

    cache = {
        "user_id": users,
        "n_u": n_u,
        "train_order_id": train_order_id,
        "orders_user_idx": orders_user_idx,
        "orders_order_number": orders_order_number,
        "orders_dow": orders_dow,
        "orders_hour": orders_hour,
        "orders_days_since": orders_days_since,
        "items_user_idx": items_user_idx,
        "items_order_number": items_order_number,
        "items_product_idx": items_product_idx,
        "items_reordered": items_reordered,
        "product_id": product_id,
        "product_aisle_idx": product_aisle_idx,
        "product_dept_idx": product_dept_idx,
        "aisle_id": aisle_id,
        "department_id": department_id,
    }
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "csv_sha256": hashes,
        "n_users_all": n_users_all,
        "n_users_selected": int(n_users),
        "eval_set_order_counts": {str(k): int(v) for k, v in eval_counts.items()},
        "n_orders_all": int(n_orders_all),
        "n_orders_selected": int(len(orders_user_idx)),
        "n_prior_items_all": int(n_prior_all),
        "n_train_items_all": int(n_train_all),
        "n_items_selected": int(len(items_user_idx)),
        "n_products": int(len(product_id)),
        "n_aisles": int(len(aisle_id)),
        "n_departments": int(len(department_id)),
        "product_names": product_names,
        "aisle_names": aisle_names,
        "department_names": dept_names,
        "build_sec": round(time.time() - t0, 1),
    }
    atomic_savez(cache_dir / CACHE_NPZ, **cache)
    atomic_write_json(cache_dir / CACHE_META, meta)
    log(f"cache written: {cache_dir / CACHE_NPZ} in {time.time() - t0:.0f}s")
    cache["_meta"] = meta
    return cache


# --------------------------------------------------------------------------------------
# user subsets (smoke tests) and basis rows
# --------------------------------------------------------------------------------------


def select_users(cache: dict, max_users: int | None, seed: int = 42) -> np.ndarray | None:
    n = len(cache["user_id"])
    if max_users is None or max_users >= n:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_users, replace=False)).astype(np.int32)


def subset_cache(cache: dict, sel: np.ndarray | None) -> dict:
    """Restrict the cache to the user positions in ``sel`` (sorted); user_idx is re-indexed."""
    if sel is None:
        return cache
    out = dict(cache)
    for k in ("user_id", "n_u", "train_order_id"):
        out[k] = cache[k][sel]
    for prefix in ("orders", "items"):
        u = cache[f"{prefix}_user_idx"]
        m = np.isin(u, sel)
        for k in CACHE_KEYS:
            if k.startswith(prefix + "_"):
                out[k] = cache[k][m]
        out[f"{prefix}_user_idx"] = np.searchsorted(sel, u[m]).astype(np.int32)
    return out


@dataclass
class Bases:
    """Row definition: row r is (user ``user_idx[r]``, basis order ``t[r]``), split 0/1/2 = train/val/test.
    Rows are ordered [all train | all val | all test], each block sorted by (user, t)."""

    user_idx: np.ndarray
    t: np.ndarray
    split: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(len(self.t))

    def rows(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def counts(self) -> tuple[int, int, int]:
        return tuple(int((self.split == s).sum()) for s in (0, 1, 2))  # type: ignore[return-value]


def make_default_bases(n_u: np.ndarray) -> Bases:
    """One row per user and split: train t = n_u-2, validation t = n_u-1, test t = n_u."""
    n = len(n_u)
    users = np.arange(n, dtype=np.int32)
    user_idx = np.concatenate([users, users, users])
    t = np.concatenate([n_u - 2, n_u - 1, n_u]).astype(np.int16)
    split = np.repeat(np.array([0, 1, 2], dtype=np.int8), n)
    return Bases(user_idx, t, split)


def make_all_prior_bases(n_u: np.ndarray) -> Bases:
    """Sensitivity variant: every 2 <= t <= n_u-2 is a train row; val/test as in the default."""
    n = len(n_u)
    n_train_rows = np.maximum(n_u.astype(np.int64) - 3, 0)  # t = 2..n_u-2
    tr_user = np.repeat(np.arange(n, dtype=np.int32), n_train_rows)
    starts = np.repeat(np.cumsum(n_train_rows) - n_train_rows, n_train_rows)
    tr_t = (np.arange(len(tr_user)) - starts + 2).astype(np.int16)
    users = np.arange(n, dtype=np.int32)
    user_idx = np.concatenate([tr_user, users, users])
    t = np.concatenate([tr_t, n_u - 1, n_u]).astype(np.int16)
    split = np.concatenate([np.zeros(len(tr_user), np.int8), np.full(n, 1, np.int8), np.full(n, 2, np.int8)])
    return Bases(user_idx, t, split)


def save_bases(path: Path, bases: Bases) -> None:
    atomic_savez(path, user_idx=bases.user_idx, t=bases.t, split=bases.split)


def load_bases(path: Path) -> Bases:
    with np.load(path) as z:
        return Bases(z["user_idx"], z["t"], z["split"])
