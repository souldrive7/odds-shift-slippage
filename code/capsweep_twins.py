"""Add the unweighted capped twins to an existing {dataset}_capsweep.json without recomputing it.

Why a separate script: the canonical Santander cap sweep was computed from prediction arrays some of
which (unweighted, weighted, weighted_mds0.7, weighted_mds2) no longer exist on disk, and LightGBM did
not reproduce them bit for bit when retrained on the same machine (see ``retrain_check`` below). A full
``run_experiment.py --part capsweep`` would therefore silently replace canonical values with values from
different arrays. This script leaves every existing entry untouched and only

  * inserts ``models[<cap spec>]["unweighted_twin"]`` for every cap whose ``unweighted_mds{c}`` arrays
    exist: raw and per-label-isotonic MAP@7 of the capped unweighted model, its in-sample ceiling and
    within-label AUC, the ideal odds shift applied to it, and the odds-shift share of the capped pair
    (loss = twin raw - weighted-capped raw, both on the 70% evaluation split; the weighted value is the
    one already in the file);
  * records ``retrain_check``: raw MAP@7 and saturation of any retrained ``weighted`` / ``unweighted``
    arrays found next to the twins, against the values the canonical ladder file carries, so the
    non-reproducibility is a number in the release rather than a footnote;
  * appends the twins' SHA-256 to ``_meta.inputs`` and stamps ``_meta.twins_added_at``.

Usage:  python code/capsweep_twins.py --dataset santander --arrays-dir <dir with unweighted_mds*/> \
            [--out-dir artifacts/results/canonical]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_dataset, sha256  # noqa: E402
from run_experiment import (  # noqa: E402
    K,
    atomic_write_json,
    cal_split,
    logit,
    map_at_k,
    oracle_ceiling,
    per_label_iso_multi,
    sigmoid,
    within_label_auc,
)

EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="santander")
    ap.add_argument("--arrays-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=EXP / "artifacts" / "results" / "canonical")
    args = ap.parse_args()

    ds = get_dataset(args.dataset, args.arrays_dir)
    path = args.out_dir / f"{ds.key}_capsweep.json"
    cs = json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    y = ds.y("te")
    va, te = cal_split(len(y))
    pi_train = ds.pi_train
    added = []
    for spec in ds.role("cap_N"):
        c = spec.cap
        s_name = f"S_weighted_mds{c:g}"
        if s_name not in cs["models"]:
            print(f"skip {spec.config}: no {s_name} in {path.name}")
            continue
        r = cs["models"][s_name]
        assert abs(float(r["cap"]) - float(c)) < 1e-12
        print(f"[twins:{ds.key}] {spec.config}")
        qn = ds.scores(spec.config)
        params = ds.lgbm_params(spec.config)
        assert float(params.get("max_delta_step") or 0.0) == float(c), (spec.config, params)
        ln_w = np.asarray(r["ln_w_per_label"], dtype=np.float64)
        cal_n, _, _ = per_label_iso_multi(qn, y, va, pi_train, ("prior",))
        ceil_n = oracle_ceiling(qn, y)
        whatif = sigmoid(logit(qn) + ln_w)
        n_raw_all, n_raw_te = map_at_k(y, qn), map_at_k(y[te], qn[te])
        whatif_te = map_at_k(y[te], whatif[te])
        s_raw_te = r["map_raw"]["map7_te"]
        loss_te = n_raw_te - s_raw_te
        r["unweighted_twin"] = {
            "config": spec.config,
            "map_raw": {"map7_all": n_raw_all, "map7_te": n_raw_te},
            "map_per_label_iso_prior": {"map7_te": map_at_k(y[te], cal_n["prior"][te])},
            "oracle_in_sample_per_label_iso": {"map7_all": map_at_k(y, ceil_n)},
            "map_whatif_N_plus_lnw": {"map7_te": whatif_te},
            "loss_te": loss_te,
            "odds_shift_share_te": float((n_raw_te - whatif_te) / loss_te) if loss_te > 0 else None,
            "saturation": {"frac_exact_1": float((qn >= 1.0).mean())},
            "within_label_auc_mean_sub200k": within_label_auc(qn, y),
            "note": "same max_delta_step, no scale_pos_weight; the weighted values it is compared with are the canonical entries of this model",
        }
        cs["_meta"].setdefault("inputs", {})[f"{ds.key}/{spec.config}/predictions.npz"] = sha256(ds.npz(spec.config))
        added.append(spec.config)
        del qn, cal_n, ceil_n, whatif

    # retrained core arrays, if present: are they the canonical ones? (they were not, on 2026-09-12)
    lad_path = args.out_dir / f"{ds.key}_ladder.json"
    if lad_path.exists():
        lad = json.loads(lad_path.read_text(encoding="utf-8"))
        chk = {}
        for cfg, key in (("weighted", "S_lgbm_spw"), ("unweighted", "N_lgbm_unweighted")):
            if not ds.available(cfg) or key not in lad:
                continue
            q = ds.scores(cfg)
            h = sha256(ds.npz(cfg))
            canon_h = lad["_meta"]["inputs"].get(f"{ds.key}/{cfg}/predictions.npz")
            chk[cfg] = {
                "sha256": h,
                "canonical_sha256": canon_h,
                "identical_file": h == canon_h,
                "map_raw": {"map7_all": map_at_k(y, q), "map7_te": map_at_k(y[te], q[te])},
                "canonical_map_raw": lad[key]["raw"],
                "saturation": {"frac_exact_1": float((q >= 1.0).mean())},
                "canonical_saturation_frac_exact_1": lad[key]["saturation"]["frac_exact_1"],
                "run_meta_generated_at": ds.meta(cfg).get("generated_at"),
            }
            del q
        if chk:
            cs["retrain_check"] = {
                "note": "arrays retrained on a second machine with the same data, script and configuration (the canonical arrays live on the machine that trained them); "
                "LightGBM 4.6 with force_row_wise and 8 threads did not reproduce them bit for bit",
                "models": chk,
            }
    cs["_meta"]["twins_added_at"] = datetime.now(UTC).isoformat()
    cs["_meta"]["twins_added"] = added
    cs["_meta"]["twins_elapsed_sec"] = round(time.time() - t0, 1)
    atomic_write_json(path, cs)
    print("wrote", path, "twins:", added, "k =", K)


if __name__ == "__main__":
    main()
