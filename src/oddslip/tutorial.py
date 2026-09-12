# %% [markdown]
# # oddslip in 10 minutes
#
# You have a per-row top-K ranker trained one-vs-rest with a label-specific positive-class weight
# (`scale_pos_weight = n-/n+`, `pos_weight`, ...). This notebook shows, on synthetic data with known
# marginals and then optionally on the Santander arrays, how to
#
# 1. **predict** how much top-K the weights cost if the learner realized the textbook odds shift
#    (`predict_rank_loss`), and see how far a real learner slips from that prediction;
# 2. **repair** the weighted scores with per-label isotonic regression whose dead-label policy is an
#    explicit argument (`calibrate_per_label`), and see why the fallback decides the sign;
# 3. **choose** between per-label maps, a shared map and no calibration without touching evaluation
#    labels (`select_calibrator_cv`).
#
# Nothing here retrains a model: the functions act on score matrices `(n_rows, n_labels)`.

# %%
import sys
from pathlib import Path

import numpy as np

# the package directory `oddslip/` is next to this notebook (src/oddslip in the release repository)
for cand in (Path.cwd(), Path.cwd().parent, Path.cwd() / "code"):
    if (cand / "oddslip").is_dir():
        sys.path.insert(0, str(cand))
        break
from oddslip import (  # noqa: E402
    apply_selection,
    calibrate_per_label,
    calibrate_shared,
    elkan_inversion,
    map_at_k,
    odds_shift,
    predict_rank_loss,
    select_calibrator_cv,
)

K = 7
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Synthetic data with known marginals
#
# 3,000 rows, 12 labels whose prevalences span three decades (0.3 down to 0.0005), so the recommended
# weights `w_j = n-/n+` span 2 to 2,000. The "unweighted model" is the true marginal with noise; that
# is the object a proper loss estimates without weights.

# %%
n, L = 3000, 12
prior = np.geomspace(0.3, 0.0005, L)
z_true = rng.normal(size=(n, L)) * 1.2 + np.log(prior / (1 - prior))
p_true = 1 / (1 + np.exp(-z_true))
y = (rng.random((n, L)) < p_true).astype(np.int8)
p_unweighted = 1 / (1 + np.exp(-(z_true + rng.normal(size=(n, L)) * 0.3)))
w = (1 - y.mean(0)) / np.maximum(y.mean(0), 1 / n)  # n-/n+ per label, as scale_pos_weight would set it
print("weights w_j:", np.round(w[[0, 4, 8, 11]], 1), "... (min %.1f, max %.0f)" % (w.min(), w.max()))
print("MAP@7 of the unweighted model:", round(map_at_k(y, p_unweighted, K), 3))

# %% [markdown]
# ## 2. The textbook prediction, and the exact inversion it promises
#
# Elkan's identity says a weight `w_j` moves label `j`'s log-odds by `ln w_j`. `predict_rank_loss`
# applies exactly that shift to the unweighted scores (a *what-if* weighted model) and reports the
# MAP@K it loses. If a learner realized the shift exactly, `elkan_inversion` would undo it exactly.

# %%
ideal = odds_shift(p_unweighted, w)
d = predict_rank_loss(p_unweighted, w, y=y, k=K, p_weighted=ideal)
print("MAP@7  unweighted %.3f -> ideal weighted %.3f  (predicted loss %.3f)" % (d["map_unweighted"], d["map_whatif"], d["predicted_loss"]))
print("inversion restores the unweighted scores exactly:", np.allclose(elkan_inversion(ideal, w), p_unweighted))
print("odds-shift share of the ideal model's loss:", d["odds_shift_share"])

# %% [markdown]
# ## 3. Slippage: a learner that does not realize the shift
#
# A finite booster does not produce `logit(p) + ln w_j`. Two things happen in practice (the paper
# measures both on LightGBM): the realized shift falls short of `ln w_j` when the leaf step is capped,
# and without a cap the rare labels' cells saturate at exactly 1.0 and tie. We emulate both here:
# the realized shift is `0.6 ln w_j` (shortfall) and every cell whose ideal score exceeds 0.999 is
# clipped to 1.0 (saturation ties).

# %%
z_w = np.log(p_unweighted / (1 - p_unweighted)) + 0.6 * np.log(w)
p_weighted = 1 / (1 + np.exp(-z_w))
p_weighted = np.where(odds_shift(p_unweighted, w) > 0.999, 1.0, p_weighted)
d = predict_rank_loss(p_unweighted, w, y=y, k=K, p_weighted=p_weighted)
print("MAP@7  unweighted %.3f | what-if %.3f | trained-like weighted %.3f" % (d["map_unweighted"], d["map_whatif"], d["map_weighted"]))
print("odds-shift share of the actual loss: %.0f%%   (the rest is slippage)" % (100 * d["odds_shift_share"]))
print("cells at exactly 1.0: %.1f%%" % (100 * (p_weighted >= 1.0).mean()))
print("analytic inversion with the known w_j: MAP@7 %.3f  (it over-corrects the shortfall and cannot untie the 1.0 cells)" % map_at_k(y, elkan_inversion(p_weighted, w), K))

# %% [markdown]
# ## 4. Repair: per-label calibration, and the fallback that decides the sign
#
# Split the rows: 30% calibration, 70% evaluation (in deployment, calibrate on a held-out period that
# precedes the test period). Per-label isotonic regression is fitted on the calibration rows only.
# A label with no calibration positives is *dead*: `fallback="prior"` maps it to its training
# prevalence, `"identity"` leaves the raw (inflated) score in place, `"exclude"` removes it.

# %%
idx = rng.permutation(n)
cal, ev = idx[: int(0.3 * n)], idx[int(0.3 * n) :]
y_cal = y[cal].copy()
y_cal[:, -1] = 0  # make the rarest label dead in the calibration split
for fb in ("prior", "identity", "exclude"):
    rep, info = calibrate_per_label(p_weighted[cal], y_cal, p_weighted[ev], prior=prior, fallback=fb)
    print("fallback=%-8s dead labels=%s  MAP@7 on the evaluation rows: %.3f" % (fb, info["dead_labels"], map_at_k(y[ev], rep, K)))
print("for comparison  raw weighted %.3f | unweighted %.3f | shared (pooled) isotonic %.3f" % (map_at_k(y[ev], p_weighted[ev], K), map_at_k(y[ev], p_unweighted[ev], K), map_at_k(y[ev], calibrate_shared(p_weighted[cal], y_cal, p_weighted[ev]), K)))

# %% [markdown]
# With the identity fallback the dead label keeps its saturated raw score of 1.0 while every calibrated
# label shrinks to its prevalence, so the dead label sits at the top of every row: the *dead-label
# takeover*. The prior fallback removes it. The shared map cannot repair a per-label distortion; it
# only re-breaks ties.

# %% [markdown]
# ## 5. Choosing the calibrator without evaluation labels
#
# `select_calibrator_cv` runs K-fold cross-validation *inside the calibration split* over per-label
# isotonic with a positives threshold `tau` (labels with at most `tau` calibration positives keep the
# prior), the shared map, and no calibration, and picks the best out-of-fold MAP@K. Then
# `apply_selection` applies the choice to new scores.

# %%
sel = select_calibrator_cv(p_weighted[cal], y_cal, prior, k=K)
print("selected:", sel["choice"], "tau =", sel["tau"])
print({k: round(v, 3) for k, v in sel["oof_map"].items()})
out = apply_selection(sel, p_weighted[cal], y_cal, p_weighted[ev], prior)
print("MAP@7 on the evaluation rows after the selected repair: %.3f" % map_at_k(y[ev], out, K))

# %% [markdown]
# ## 6. Optional: the same three calls on the Santander matched pair
#
# If the prediction arrays of the matched LightGBM pair are present (regenerated from the public
# Kaggle CSV by `santander_train/train_matched_pair.py`; see `DATA.md`), the cell below runs the same
# analysis on a 200,000-row subsample of the test rows. The paper's numbers use all rows and are
# produced by `run_experiment.py`; this cell is a quick check, not the paper.

# %%
import os

# release layout: data/santander/outputs_matched_pair/{weighted,unweighted}/predictions.npz (see DATA.md);
# elsewhere point ODDSLIP_ARRAYS_DIR at the directory that holds the two config folders
candidates = [
    Path(os.environ["ODDSLIP_ARRAYS_DIR"]) if os.environ.get("ODDSLIP_ARRAYS_DIR") else None,
    Path.cwd() / "data" / "santander" / "outputs_matched_pair",
    Path.cwd().parent / "data" / "santander" / "outputs_matched_pair",
]
arr = next((c for c in candidates if c is not None and (c / "weighted" / "predictions.npz").exists() and (c / "unweighted" / "predictions.npz").exists()), None)
if arr is None:
    print("Santander arrays not found; skipping (see DATA.md).")
else:
    import json

    S = np.load(arr / "weighted" / "predictions.npz")
    N = np.load(arr / "unweighted" / "predictions.npz")
    meta = json.load(open(arr / "weighted" / "run_meta.json", encoding="utf-8"))
    y_te = S["Y_te"].astype(np.int8)
    sub = np.random.default_rng(42).choice(len(y_te), min(200_000, len(y_te)), replace=False)
    y_s, pS, pN = y_te[sub], S["p_te"][sub].astype(np.float64), N["p_te"][sub].astype(np.float64)
    n_pos = np.array([r["n_pos"] for r in meta["per_label"]], dtype=np.float64)
    n_train = meta["n_train"]
    w_true = (n_train - n_pos) / np.maximum(n_pos, 1)
    pi_train = n_pos / n_train
    d = predict_rank_loss(pN, w_true, y=y_s, k=K, p_weighted=pS)
    print("Santander subsample (%d rows): MAP@7 unweighted %.3f | what-if %.3f | weighted %.3f | odds-shift share %.0f%%" % (len(sub), d["map_unweighted"], d["map_whatif"], d["map_weighted"], 100 * d["odds_shift_share"]))
    half = len(sub) // 2
    cal_i, ev_i = np.arange(half), np.arange(half, len(sub))
    rep, info = calibrate_per_label(pS[cal_i], y_s[cal_i], pS[ev_i], prior=pi_train, fallback="prior")
    print("per-label isotonic (prior fallback) on the weighted model: MAP@7 %.3f on the held-out half; dead labels: %s" % (map_at_k(y_s[ev_i], rep, K), info["dead_labels"]))
    print("analytic inversion: MAP@7 %.3f" % map_at_k(y_s[ev_i], elkan_inversion(pS[ev_i], w_true), K))
