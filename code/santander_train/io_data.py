"""Santander Product Recommendation 形式のパネルを読み込み、(t)→(t+1) の新規商品追加をラベル化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def detect_product_columns(df: pd.DataFrame) -> list[str]:
    """Kaggle Santander の商品列を検出．

    実データには `ind_nomina_ult1` / `ind_nom_pens_ult1` / `ind_recibo_ult1` のように
    `fin` を含まない 3 列も含まれるため，`fin` 要件は外し，
    `ind_` 始まり + `ult1` 終わりかつ顧客属性ではないことで判定する．
    """
    customer_attr_exclude = {
        "ind_empleado",
        "ind_nuevo",
        "ind_actividad_cliente",
    }
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("ind_") and cl.endswith("ult1") and cl not in customer_attr_exclude:
            cols.append(c)
    return sorted(cols)


def load_panel_csv(csv_path: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    if "fecha_dato" not in df.columns:
        raise ValueError("fecha_dato column required")
    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"])
    if "ncodpers" not in df.columns:
        raise ValueError("ncodpers column required")
    return df


def build_xy_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """各行は「月 t の状態」から「月 t+1 で新規に 1 となった商品」を予測する多ラベル問題。"""
    prod_cols = detect_product_columns(df)
    if not prod_cols:
        raise ValueError("No product columns ind_*_fin_*ult1 found")

    df = df.sort_values(["ncodpers", "fecha_dato"])
    feat_rows: list[pd.Series] = []
    labels: list[np.ndarray] = []

    for _, g in df.groupby("ncodpers", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        P = g[prod_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
        for i in range(len(g) - 1):
            curr = P[i]
            nxt = P[i + 1]
            delta = nxt - curr
            # 新規獲得のみ（0→1）
            y = np.clip(delta, 0, 1)
            y = np.where((curr < 0.5) & (nxt > 0.5), 1.0, 0.0)
            feat_rows.append(g.iloc[i])
            labels.append(y.astype(np.float64))

    if not feat_rows:
        raise ValueError("no consecutive months per customer")

    X_df = pd.DataFrame(feat_rows).reset_index(drop=True)
    Y = np.stack(labels, axis=0)
    return X_df, Y, prod_cols


def build_xy_pairs_fast(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Vectorized version of ``build_xy_pairs`` (identical output, 4-5x less memory).

    Replaces the per-customer Python loop with a single sort + boolean
    mask over adjacent rows.  Output rows appear in the same order as
    ``build_xy_pairs`` (customer-by-customer in sort order, then
    fecha-by-fecha within each customer), so caller code can swap the
    two functions without index drift.
    """
    prod_cols = detect_product_columns(df)
    if not prod_cols:
        raise ValueError("No product columns ind_*_fin_*ult1 found")

    df_sorted = df.sort_values(["ncodpers", "fecha_dato"], kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    if n < 2:
        raise ValueError("need at least 2 rows")

    P = np.empty((n, len(prod_cols)), dtype=np.float64)
    for j, c in enumerate(prod_cols):
        P[:, j] = (
            pd.to_numeric(df_sorted[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        )

    ncodpers = df_sorted["ncodpers"].to_numpy()
    same_customer_next = ncodpers[:-1] == ncodpers[1:]
    valid_idx = np.flatnonzero(same_customer_next)
    if valid_idx.size == 0:
        raise ValueError("no consecutive months per customer")

    curr = P[valid_idx]
    nxt = P[valid_idx + 1]
    Y = ((curr < 0.5) & (nxt > 0.5)).astype(np.float64)

    X_df = df_sorted.iloc[valid_idx].reset_index(drop=True)
    return X_df, Y, prod_cols


def featurize(X_df: pd.DataFrame, prod_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """学習用特徴量。日付・顧客IDは学習行列から除外（リーク防止）。"""
    drop = {"fecha_dato", "ncodpers", "indrel_1mes", "indrel", "conyuemp", "indfall"}
    X_df = X_df.drop(columns=[c for c in X_df.columns if c in drop], errors="ignore")
    cat_candidates = [
        "ind_empleado",
        "sexo",
        "ind_nuevo",
        "indresi",
        "indext",
        "canal_entrada",
        "segmento",
        "nomprov",
        "pais_residencia",
        "tiprel_1mes",
        "tipodom",
        "indfall",
    ]
    cat_cols = [c for c in cat_candidates if c in X_df.columns]
    num_cols = [
        c
        for c in ("age", "antiguedad", "ind_actividad_cliente", "renta", "cod_prov")
        if c in X_df.columns
    ]
    use = [c for c in cat_cols + num_cols + prod_cols if c in X_df.columns]
    X = X_df[use].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    for c in prod_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.int32)
    return X, list(X.columns)
