"""EDA gate for the Instacart product-level experiment (arXiv v3): measures what README section 7
marks as [reference value, unmeasured] and decides H1-H3 before the main runs.

Writes results/eda.json (tracked). Steps:
  1. raw files: extract the Kaggle zip if needed (nested *.csv.zip), SHA-256 -> data/raw/SHA256SUMS.txt
  2. counts: users per eval_set, orders per user, basket size, split rows
  3. features + labels for the default split (also written to data/processed/, so the main
     run can start from them) -- prevalence vector, top-N candidates
  4. imbalance width, w_j = n_-/n_+ range per candidate N
  5. dead-label frequency in random calibration splits of the test rows (10 seeds)
  6. timing benchmark: 100 labels spread over the frequency range, weighted and unweighted,
     through the production pool (sec/label, projected hours, saturation, within-label AUC,
     MAP@7 on those labels, lgb.train vs LGBMClassifier parity, initial-score check)
  7. hypotheses H1-H3 with pass/fail and a recommended N

Usage (scheduled; ~12 min):  see scripts/run_instacart_eda.bat
Smoke:  uv run python code/instacart_train/eda.py --max-users 3000 --n-bench-labels 10 --n-jobs 2 --cache-dir data/processed_smoke --out results/eda_smoke.json
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_matched_pair as tmp  # noqa: E402
from build_features import build_all, select_top_n, write_processed  # noqa: E402
from instacart_io import (  # noqa: E402
    KAGGLE_ZIP,
    RAW_FILES,
    atomic_write_json,
    extract_zip,
    load_or_build_cache,
    log,
    make_default_bases,
    redirect_log,
    select_users,
    subset_cache,
    write_sha256sums,
)

K = 7


def map_at_k(y: np.ndarray, s: np.ndarray, k: int = K) -> float:
    """Kaggle MAP@K: rows without positives are excluded; stable argsort (same as the rederive code)."""
    keep = y.sum(1) > 0
    y = y[keep]
    s = s[keep]
    if len(y) == 0:
        return float("nan")
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    ap = (prec * rel).sum(1) / np.minimum(y.sum(1), k)
    return float(ap.mean())


def dist(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "p5": float(np.percentile(x, 5)), "median": float(np.median(x)), "mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def parity_and_init_checks(X_tr, X_va, Y_tr, label_cols: np.ndarray, spw_pool: np.ndarray, p_va_pool: dict, args, n_check: int = 3) -> dict:
    """(a) lgb.train-on-shared-Dataset vs LGBMClassifier on the same label (max |diff| of p_va);
    (b) initial score of a 1-round model with no admissible split = raw_score - leaf_value,
    compared with logit(prevalence) and logit(weighted mean) -- Proposition 3's premise."""
    import warnings

    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    n = X_tr.shape[0]
    out: dict = {"parity": [], "init_score": []}
    Y_csc = Y_tr.tocsc()
    pick = np.linspace(0, len(label_cols) - 1, n_check).round().astype(int)
    for pos in pick:
        col = int(label_cols[pos])
        y = np.asarray(Y_csc[:, col].todense()).ravel().astype(np.float32)
        npos = int(y.sum())
        pi = npos / n
        for weighted in (False, True):
            spw = (n - npos) / max(npos, 1) if weighted else 1.0
            extra = {"scale_pos_weight": spw} if weighted else {}
            clf = LGBMClassifier(n_estimators=args.n_trees, learning_rate=args.learning_rate, num_leaves=args.num_leaves, random_state=args.seed, verbose=-1, force_row_wise=True, n_jobs=1, **extra)
            clf.fit(X_tr, y.astype(np.int64))
            p_sk = clf.predict_proba(X_va)[:, 1].astype(np.float32)
            cname = "weighted" if weighted else "unweighted"
            diff = float(np.abs(p_sk - p_va_pool[cname][:, pos]).max())
            out["parity"].append({"label_col": col, "config": cname, "n_pos": npos, "max_abs_diff_p_va": diff})
            # initial score: one round, no split admissible (min_data_in_leaf > n) -> raw = init + leaf
            params = {"objective": "binary", "learning_rate": args.learning_rate, "num_leaves": 2, "min_data_in_leaf": n + 1, "seed": args.seed, "verbose": -1, "force_row_wise": True, "num_threads": 1, "feature_pre_filter": False, **extra}
            rec: dict = {"label_col": col, "config": cname, "n_pos": npos, "logit_prevalence": logit(pi) if 0 < pi < 1 else None, "logit_weighted_mean": logit(spw * npos / (spw * npos + (n - npos))) if 0 < npos < n else None}
            try:
                ds = lgb.Dataset(X_tr, label=y, params=params, free_raw_data=False)
                bst = lgb.train(params, ds, num_boost_round=1)
                raw = float(bst.predict(X_tr[:1], raw_score=True, num_threads=1)[0])
                tree = bst.dump_model()["tree_info"][0]["tree_structure"]
                leaf = tree.get("leaf_value")
                rec["raw_score_1round"] = raw
                rec["root_leaf_value"] = leaf
                # A tree with a single leaf is turned into a constant tree that carries only the
                # initial score (LightGBM gbdt.cpp: AsConstantTree(init_scores) in iteration 1),
                # so the raw score of a 1-round model with no admissible split IS the initial score.
                rec["init_score"] = raw if leaf is not None else None
                if leaf is None:
                    rec["note"] = "tree split despite min_data_in_leaf > n; init not isolated"
                else:
                    lp, lw = rec["logit_prevalence"], rec["logit_weighted_mean"]
                    rec["matches_logit_prevalence"] = None if lp is None else bool(abs(raw - lp) < 1e-6)
                    rec["matches_logit_weighted_mean"] = None if lw is None else bool(abs(raw - lw) < 1e-6)
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:300]
            out["init_score"].append(rec)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Instacart product-level EDA gate")
    p.add_argument("--raw-dir", type=Path, default=HERE / "data" / "raw")
    p.add_argument("--cache-dir", type=Path, default=HERE / "data" / "processed", help="cache and processed bundle directory")
    p.add_argument("--out", type=Path, default=HERE / "results" / "eda.json")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--candidates", type=str, default="1000,2000,5000")
    p.add_argument("--min-pos", type=int, default=20)
    p.add_argument("--max-labels", type=int, default=5000)
    p.add_argument("--n-bench-labels", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--block-size", type=int, default=10)
    p.add_argument("--n-trees", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--night1-configs", type=int, default=7)
    p.add_argument("--night2-configs", type=int, default=6)
    p.add_argument("--budget-hours", type=float, default=20.0)
    args = p.parse_args()
    redirect_log(args.log)
    t_all = time.time()
    res: dict = {}

    # 1. raw files
    raw_dir = args.raw_dir
    zip_path = raw_dir / KAGGLE_ZIP
    missing = [f for f in RAW_FILES if not (raw_dir / f).is_file()]
    if missing:
        if not zip_path.is_file():
            sys.exit(f"missing {missing} and no {zip_path}; run the Kaggle download first")
        log(f"extracting {zip_path}")
        res["extract"] = extract_zip(zip_path, raw_dir)
    res["raw_files"] = write_sha256sums(raw_dir)
    log(f"raw files: {list(res['raw_files'])}")

    # 2. cache and counts
    cache = load_or_build_cache(raw_dir, args.cache_dir)
    meta = cache["_meta"]
    sel = select_users(cache, args.max_users, args.seed)
    cache = subset_cache(cache, sel)
    n_u = cache["n_u"].astype(np.int64)
    n_users = len(n_u)
    n_orders = len(cache["orders_user_idx"])
    order_row = np.zeros((n_users, int(n_u.max()) + 2), np.int64) - 1
    order_row[cache["orders_user_idx"], cache["orders_order_number"]] = np.arange(n_orders)
    irow = order_row[cache["items_user_idx"], cache["items_order_number"]]
    basket = np.bincount(irow, minlength=n_orders)
    res["counts"] = {
        "n_users_all": meta["n_users_all"],
        "n_users_with_labelled_last_order": meta["n_users_selected"],
        "n_users_used": int(n_users),
        "user_subset": None if sel is None else {"max_users": args.max_users, "seed": args.seed},
        "eval_set_order_counts": meta["eval_set_order_counts"],
        "n_orders_all": meta["n_orders_all"],
        "n_orders_selected_users": int(n_orders),
        "n_products": meta["n_products"],
        "n_aisles": meta["n_aisles"],
        "n_departments": meta["n_departments"],
        "prior_line_items_all": meta["n_prior_items_all"],
        "train_line_items_all": meta["n_train_items_all"],
        "line_items_selected_users": meta["n_items_selected"] if sel is None else int(len(irow)),
        "orders_per_user_incl_labelled": dist(n_u),
        "basket_size_all_orders": dist(basket),
        "basket_size_labelled_order": dist(basket[order_row[np.arange(n_users), n_u]]),
        "split_rows_default": {"train": int(n_users), "val": int(n_users), "test": int(n_users)},
        "split_rows_train_all_prior": int(np.maximum(n_u - 3, 0).sum()),
    }
    log(f"counts: {n_users:,} users, {n_orders:,} orders, basket median {np.median(basket):.0f}")

    # 3. default split, labels, features (written to cache_dir as the processed bundle)
    bases = make_default_bases(cache["n_u"])
    built = build_all(cache, bases, args.min_pos, args.max_labels)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_processed(args.cache_dir, cache, bases, built, {"from": "eda.py", **args_dict})
    tr, va, te = bases.rows(0), bases.rows(1), bases.rows(2)
    n_train = int(len(tr))
    n_pos_tr = built["n_pos"]["train"]
    prevalence = n_pos_tr / n_train
    res["prevalence_train"] = {
        "n_train_rows": n_train,
        "n_pos_per_product": n_pos_tr.tolist(),
        "n_products_with_pos": int((n_pos_tr > 0).sum()),
        "n_products_ge_min_pos": built["top_info"]["n_products_ge_min_pos"],
        "mean_positives_per_row": float(n_pos_tr.sum() / n_train),
        "top_n_used_for_features": built["top_info"],
    }

    # 4. imbalance and weight range per candidate N
    cands = sorted({int(c) for c in args.candidates.split(",") if c.strip()} | {int(args.max_labels), int(built["top_info"]["n_products_ge_min_pos"])})
    imb: dict = {}

    def stats_for(idx: np.ndarray, name: str) -> dict:
        npos = n_pos_tr[idx]
        w = (n_train - npos) / np.maximum(npos, 1)
        return {
            "n_labels": int(len(idx)),
            "prevalence_min": float(npos.min() / n_train),
            "prevalence_max": float(npos.max() / n_train),
            "decades": float(np.log10(npos.max() / max(npos.min(), 1))),
            "w_min": float(w.min()),
            "w_median": float(np.median(w)),
            "w_max": float(w.max()),
            "ln_w_min": float(np.log(w.min())),
            "ln_w_max": float(np.log(w.max())),
            "frac_labels_w_ge_1e3": float((w >= 1e3).mean()),
            "frac_labels_w_ge_1e4": float((w >= 1e4).mean()),
        }

    imb["all_products_with_pos"] = stats_for(np.flatnonzero(n_pos_tr > 0), "all")
    for N in cands:
        idx, info = select_top_n(n_pos_tr, args.min_pos, N)
        if len(idx) == 0:
            continue
        imb[f"top{N}"] = {**stats_for(idx, f"top{N}"), "N_effective": int(len(idx)), "cap_applied": info["cap_applied"]}
    res["imbalance"] = imb
    log(f"imbalance: {[(k, round(v['w_max']), v['n_labels']) for k, v in imb.items()]}")

    # 5. dead labels in random calibration splits of the test rows
    Y_te_all = built["Y_all"][te].tocsc()
    n_te = int(len(te))
    dead: dict = {}
    for N in cands:
        idx, _ = select_top_n(n_pos_tr, args.min_pos, N)
        if len(idx) == 0:
            continue
        Yn = Y_te_all[:, idx].tocsr()
        by_frac = {}
        for fr in (0.3, 0.1, 0.03, 0.01):
            ncal = int(n_te * fr)
            d = []
            for s in range(10):
                rng = np.random.default_rng(1000 + s)
                ci = rng.permutation(n_te)[:ncal]
                d.append(int((np.asarray(Yn[ci].sum(0)).ravel() == 0).sum()))
            by_frac[str(fr)] = {"n_cal": ncal, "dead_mean": float(np.mean(d)), "dead_min": int(min(d)), "dead_max": int(max(d)), "frac_seeds_any_dead": float(np.mean([x > 0 for x in d]))}
        dead[f"top{len(idx)}"] = {"N": int(len(idx)), "test_pos_zero_labels": int((np.asarray(Yn.sum(0)).ravel() == 0).sum()), "by_frac": by_frac}
    res["dead_labels"] = dead

    # 6. timing benchmark through the production pool
    bench: dict = {}
    if not args.skip_benchmark:
        top_idx = built["top_idx"]
        N = len(top_idx)
        freq_order = np.argsort(-n_pos_tr[top_idx], kind="stable")  # positions within top-N, most frequent first
        nb = min(args.n_bench_labels, N)
        label_cols = np.sort(freq_order[np.linspace(0, N - 1, nb).round().astype(int)]).astype(np.int64)
        features_path = args.cache_dir / "features.npz"
        labels_path = args.cache_dir / "labels_topN.npz"
        Y_tr = built["Y_top"][tr]
        Y_va = built["Y_top"][va]
        y_va = np.asarray(Y_va[:, label_cols].todense()).astype(np.int8)
        y_te = np.asarray(built["Y_top"][te][:, label_cols].todense()).astype(np.int8)
        p_va_pool: dict = {}
        per_cfg: dict = {}
        for cname in ("unweighted", "weighted"):
            work = args.cache_dir / "eda_bench" / cname
            r = tmp.fit_config_parallel(cname=cname, cfg=tmp.CONFIGS[cname], features_path=features_path, labels_path=labels_path, label_cols=label_cols, work_dir=work, args=args, row_mask=None)
            p_va, p_te = r["p_va"].astype(np.float64), r["p_te"].astype(np.float64)
            p_va_pool[cname] = r["p_va"]
            aucs = [roc_auc_score(y_va[:, j], p_va[:, j]) for j in range(nb) if 0 < y_va[:, j].sum() < len(y_va)]
            per_cfg[cname] = {
                "wall_sec": round(r["wall_sec"], 1),
                "sec_per_label_wall": round(r["wall_sec"] / nb, 3),
                "fit_sec": {"mean": round(float(r["fit_sec"].mean()), 2), "median": round(float(np.median(r["fit_sec"])), 2), "max": round(float(r["fit_sec"].max()), 2)},
                "n_pos": r["n_pos"].tolist(),
                "scale_pos_weight": r["spw"].tolist(),
                "saturation_val": {"frac_exact_1": float((r["p_va"] >= 1.0).mean()), "frac_ge_1m1e9": float((p_va >= 1 - 1e-9).mean()), "frac_exact_0": float((r["p_va"] <= 0.0).mean()), "frac_exact_1_per_label": (r["p_va"] >= 1.0).mean(0).tolist()},
                "saturation_test": {"frac_exact_1": float((r["p_te"] >= 1.0).mean()), "frac_ge_1m1e9": float((p_te >= 1 - 1e-9).mean())},
                "within_label_auc_val_mean": float(np.mean(aucs)) if aucs else None,
                "map7_val_bench_labels": map_at_k(y_va, p_va),
                "map7_test_bench_labels": map_at_k(y_te, p_te),
            }
            for bp in r["block_dir"].glob("block_*.npz"):
                bp.unlink()
            log(f"benchmark {cname}: {per_cfg[cname]['sec_per_label_wall']} s/label wall, sat1={per_cfg[cname]['saturation_val']['frac_exact_1']:.4f}, auc={per_cfg[cname]['within_label_auc_val_mean']}")
        pop = np.tile(prevalence[top_idx][label_cols], (len(va), 1))
        sec_per_label = max(per_cfg["unweighted"]["sec_per_label_wall"], per_cfg["weighted"]["sec_per_label_wall"])
        projected = {}
        for Nc in cands:
            Ne = min(Nc, built["top_info"]["n_products_ge_min_pos"])
            projected[f"top{Nc}"] = {"N_effective": Ne, "hours_per_config": round(Ne * sec_per_label / 3600, 2), "night1_hours": round(Ne * sec_per_label * args.night1_configs / 3600, 1), "night2_hours": round(Ne * sec_per_label * args.night2_configs / 3600, 1)}
        f = tmp.load_csr_bundle(features_path)
        checks = parity_and_init_checks(f["X_tr"], f["X_va"], Y_tr, label_cols, None, p_va_pool, args)
        bench = {
            "n_bench_labels": int(nb),
            "label_cols": label_cols.tolist(),
            "product_ids": cache["product_id"][top_idx][label_cols].astype(int).tolist(),
            "n_train": n_train,
            "n_features": int(f["X_tr"].shape[1]),
            "n_jobs": args.n_jobs,
            "num_threads": args.num_threads,
            "configs": per_cfg,
            "popularity_map7_val_bench_labels": map_at_k(y_va, pop),
            "projected": projected,
            "lgb_train_vs_sklearn_max_abs_diff": max(c["max_abs_diff_p_va"] for c in checks["parity"]),
            "parity_checks": checks["parity"],
            "init_score_check": checks["init_score"],
        }
    res["timing_benchmark"] = bench

    # 7. hypotheses
    top_key = f"top{built['top_info']['N']}"
    top_stats = imb.get(top_key) or imb[max((k for k in imb if k.startswith("top")), key=lambda k: imb[k]["n_labels"])]
    dead_key = next((k for k in dead if k == f"top{built['top_info']['N']}"), None)
    dead_30 = dead[dead_key]["by_frac"]["0.3"]["dead_mean"] if dead_key else None
    sat = bench["configs"]["weighted"]["saturation_val"]["frac_exact_1"] if bench else None
    hyp = {
        "H1": {"criterion": "max w_j over the top-N used for LightGBM >= 1e3 (all products >= 1e4)", "value_topN": top_stats["w_max"], "value_all": imb["all_products_with_pos"]["w_max"], "pass": bool(top_stats["w_max"] >= 1e3)},
        "H2": {"criterion": "dead labels >= 1 in a 30% calibration split of the test rows (mean of 10 seeds), top-N", "value": dead_30, "pass": None if dead_30 is None else bool(dead_30 >= 1)},
        "H3": {"criterion": "weighted LightGBM: fraction of validation cells at exactly 1.0 >= 0.05 (benchmark labels)", "value": sat, "pass": None if sat is None else bool(sat >= 0.05)},
    }
    rec_N = None
    if bench:
        ok = [Nc for Nc in cands if imb.get(f"top{Nc}", {}).get("w_max", 0) >= 1e3 and bench["projected"][f"top{Nc}"]["night1_hours"] <= args.budget_hours]
        rec_N = max(ok) if ok else None
    hyp["recommended_N"] = {"value": rec_N, "rule": f"largest candidate with H1 and projected night-1 hours <= {args.budget_hours}", "candidates": cands}
    res["hypotheses"] = hyp
    res["_meta"] = {"generated_at": datetime.now(UTC).isoformat(), "args": args_dict, "csv_sha256": meta["csv_sha256"], "cache_build_sec": meta.get("build_sec"), "elapsed_sec": round(time.time() - t_all, 1)}
    atomic_write_json(args.out, res)
    log(f"hypotheses: {hyp}")
    log(f"wrote {args.out} in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
