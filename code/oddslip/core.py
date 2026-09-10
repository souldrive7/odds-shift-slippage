"""oddslip -- three functions for per-row top-K ranking under per-label class weights.

The name is provisional (docs/arxiv_v3_title_candidates.md). Everything here is extracted from
run_experiment.py and mulan_dose.py, the code that produces every
number of the paper, and kept numerically identical to it. Dependencies: numpy, scikit-learn.

    predict_rank_loss(p_unweighted, w, y=None, k=7, p_weighted=None)
        Elkan's identity as a what-if device: the scores a weighted model would produce if it
        realized the odds shift ln w_j exactly, and (with labels) the MAP@K loss that shift alone
        explains. The gap to the trained weighted model is slippage.
    calibrate_per_label(p_cal, y_cal, p, prior, fallback="prior", tau=0)
        Per-label isotonic regression fitted on a calibration split, with the dead-label policy as an
        explicit argument: a label with at most `tau` calibration positives (or no negatives) is mapped
        to its training prevalence ("prior"), left untouched ("identity") or removed ("exclude").
    select_calibrator_cv(p_cal, y_cal, prior, k=7, taus=(0, 1, 2, 5, 10), n_folds=5, seed=42)
        Choose per-label isotonic (with its positives threshold tau), the shared isotonic map, or no
        calibration by K-fold cross-validation of MAP@K inside the calibration split; test labels are
        never used. `apply_selection` applies the choice.

Scores are (n_rows, n_labels) arrays of probabilities in [0, 1]; labels are 0/1 arrays of the same
shape. MAP@K follows the Kaggle definition (rows without positives excluded, ties broken by the
stable argsort, i.e. by column index).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

EPS = 1e-15
FALLBACKS = ("prior", "identity", "exclude")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def map_at_k(y: np.ndarray, s: np.ndarray, k: int = 7) -> float:
    """Kaggle MAP@K: rows without positives are excluded; stable argsort breaks ties by column."""
    y = np.asarray(y)
    s = np.asarray(s)
    keep = y.sum(1) > 0
    if not keep.any():
        return float("nan")
    y = y[keep]
    s = s[keep]
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(y, idx, 1).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)
    ap = (prec * rel).sum(1) / np.minimum(y.sum(1), k)
    return float(ap.mean())


def odds_shift(p: np.ndarray, w: np.ndarray, a: float = 1.0) -> np.ndarray:
    """The population effect of a positive-class weight w_j on a proper-loss minimizer:
    logit(p) + a * ln w_j (Elkan's identity; a=1 is the full shift)."""
    return sigmoid(logit(p) + a * np.log(np.asarray(w, dtype=np.float64)))


def elkan_inversion(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Undo the odds shift analytically: q / (q + w (1 - q)). Exact only if the learner realized ln w_j."""
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return q / (q + w * (1.0 - q))


def predict_rank_loss(p_unweighted: np.ndarray, w: np.ndarray, y: np.ndarray | None = None, k: int = 7, p_weighted: np.ndarray | None = None) -> dict:
    """What-if device. Returns the odds-shifted scores and, with labels, the MAP@K they lose against
    the unweighted model (the part of a weighted model's loss that the textbook account explains).
    With the trained weighted model's scores as well, returns its actual loss and the odds-shift share."""
    whatif = odds_shift(p_unweighted, w)
    out: dict = {"whatif_scores": whatif}
    if y is not None:
        m_un = map_at_k(y, p_unweighted, k)
        m_wi = map_at_k(y, whatif, k)
        out.update({"map_unweighted": m_un, "map_whatif": m_wi, "predicted_loss": m_un - m_wi})
        if p_weighted is not None:
            m_w = map_at_k(y, p_weighted, k)
            loss = m_un - m_w
            out.update({"map_weighted": m_w, "actual_loss": loss, "odds_shift_share": (m_un - m_wi) / loss if loss > 0 else None, "slippage_loss": m_wi - m_w})
    return out


def dead_labels(y_cal: np.ndarray, tau: int = 0) -> np.ndarray:
    """Labels whose calibration split has at most `tau` positives or no negatives."""
    y_cal = np.asarray(y_cal)
    pos = y_cal.sum(0)
    return np.flatnonzero((pos <= tau) | (pos == len(y_cal)))


def calibrate_per_label(p_cal: np.ndarray, y_cal: np.ndarray, p: np.ndarray, prior: np.ndarray | None = None, fallback: str = "prior", tau: int = 0) -> tuple[np.ndarray, dict]:
    """Per-label isotonic regression fitted on (p_cal, y_cal) and applied to p, with an explicit
    dead-label policy. Returns (calibrated scores, info) where info lists the dead labels."""
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")
    p_cal = np.asarray(p_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal)
    p = np.asarray(p, dtype=np.float64)
    if p_cal.shape[1] != p.shape[1] or p_cal.shape != y_cal.shape:
        raise ValueError("p_cal, y_cal and p must share the label axis; p_cal and y_cal the same shape")
    dead = dead_labels(y_cal, tau)
    if fallback == "prior":
        if prior is None:
            raise ValueError("fallback='prior' needs the training prevalence per label (prior); passing the calibration-split mean would map dead labels to zero")
        prior = np.asarray(prior, dtype=np.float64)
        if prior.shape != (p.shape[1],):
            raise ValueError("prior must have one entry per label")
    out = p.copy()
    dead_set = set(dead.tolist())
    for j in range(p.shape[1]):
        if j in dead_set:
            if fallback == "prior":
                out[:, j] = prior[j]
            elif fallback == "exclude":
                out[:, j] = -np.inf
            continue
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
        m.fit(p_cal[:, j], y_cal[:, j].astype(np.float64))
        out[:, j] = m.predict(p[:, j])
    return out, {"dead_labels": dead.tolist(), "n_dead": int(len(dead)), "fallback": fallback, "tau": tau}


def calibrate_shared(p_cal: np.ndarray, y_cal: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One isotonic map shared by all labels (rank-neutral within a row up to ties)."""
    p_cal = np.asarray(p_cal, dtype=np.float64)
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9)
    m.fit(p_cal.ravel(), np.asarray(y_cal).ravel().astype(np.float64))
    p = np.asarray(p, dtype=np.float64)
    return m.predict(p.ravel()).reshape(p.shape)


def select_calibrator_cv(p_cal: np.ndarray, y_cal: np.ndarray, prior: np.ndarray, k: int = 7, taus: tuple[int, ...] = (0, 1, 2, 5, 10), n_folds: int = 5, seed: int = 42) -> dict:
    """Cross-validated choice among per-label isotonic (prior fallback, positives threshold tau),
    the shared map and no calibration, by out-of-fold MAP@K on the calibration split only.
    Ties go to the smaller tau, then to per-label over shared over raw (deterministic)."""
    p_cal = np.asarray(p_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal)
    n = len(y_cal)
    if n_folds < 2 or n < n_folds:
        raise ValueError("need at least two folds and at least n_folds calibration rows")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_folds)
    cands = [f"per_label_tau{t}" for t in taus] + ["shared", "raw"]
    oof = {c: np.zeros_like(p_cal) for c in cands}
    for f in range(n_folds):
        ho = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        for t in taus:
            oof[f"per_label_tau{t}"][ho], _ = calibrate_per_label(p_cal[tr], y_cal[tr], p_cal[ho], prior, "prior", tau=t)
        oof["shared"][ho] = calibrate_shared(p_cal[tr], y_cal[tr], p_cal[ho])
    oof["raw"] = p_cal
    scores = {c: map_at_k(y_cal, oof[c], k) for c in cands}
    best = max(cands, key=lambda c: (scores[c], -cands.index(c)))
    if best in ("shared", "raw"):
        return {"choice": best, "tau": None, "oof_map": scores}
    return {"choice": "per_label", "tau": int(best.replace("per_label_tau", "")), "oof_map": scores}


def apply_selection(selection: dict, p_cal: np.ndarray, y_cal: np.ndarray, p: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """Apply the result of select_calibrator_cv to new scores."""
    if selection["choice"] == "raw":
        return np.asarray(p, dtype=np.float64)
    if selection["choice"] == "shared":
        return calibrate_shared(p_cal, y_cal, p)
    return calibrate_per_label(p_cal, y_cal, p, prior, "prior", tau=selection["tau"])[0]
