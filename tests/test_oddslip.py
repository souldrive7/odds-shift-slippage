"""Tests for the oddslip helpers (code/oddslip)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# release layout: code/oddslip
_ROOT = Path(__file__).resolve().parents[1]
for _cand in (_ROOT / "code",):
    if (_cand / "oddslip").is_dir():
        sys.path.insert(0, str(_cand))
        break

from oddslip import (  # noqa: E402
    apply_selection,
    calibrate_per_label,
    elkan_inversion,
    map_at_k,
    odds_shift,
    predict_rank_loss,
    select_calibrator_cv,
)


def _synthetic(n=600, L=6, seed=0):
    rng = np.random.default_rng(seed)
    prior = np.array([0.30, 0.15, 0.08, 0.04, 0.02, 0.005])[:L]
    z = rng.normal(size=(n, L)) * 1.5 + np.log(prior / (1 - prior))
    p = 1 / (1 + np.exp(-z))
    y = (rng.random((n, L)) < p).astype(np.int8)
    return p, y, prior


def test_map_at_k_known_values():
    y = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    s = np.array([[0.9, 0.5, 0.1], [0.2, 0.9, 0.8], [0.5, 0.5, 0.5]])
    # row 1: AP = 1; row 2: hits at ranks 1 and 2 -> (1 + 1) / 2 = 1; row 3 has no positive -> excluded
    assert map_at_k(y, s, k=3) == pytest.approx(1.0)
    s2 = np.array([[0.1, 0.9, 0.5], [0.2, 0.9, 0.8], [0.5, 0.5, 0.5]])
    # row 1: the positive is at rank 3 -> AP = 1/3
    assert map_at_k(y, s2, k=3) == pytest.approx((1 / 3 + 1.0) / 2)


def test_odds_shift_inverts_and_common_weight_is_rank_neutral():
    p, y, _ = _synthetic()
    w = np.array([2.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    q = odds_shift(p, w)
    assert np.allclose(elkan_inversion(q, w), p, atol=1e-9)
    common = odds_shift(p, np.full(6, 50.0))
    assert map_at_k(y, common, 3) == pytest.approx(map_at_k(y, p, 3))
    # a label-specific weight changes the ranking, and the what-if device reports that loss
    d = predict_rank_loss(p, w, y=y, k=3, p_weighted=q)
    assert d["predicted_loss"] > 0
    assert d["actual_loss"] == pytest.approx(
        d["predicted_loss"]
    )  # the ideal weighted model IS the what-if
    assert d["odds_shift_share"] == pytest.approx(1.0)


def test_calibrate_per_label_dead_label_policies_and_monotonicity():
    p, y, prior = _synthetic()
    cal, ev = np.arange(300), np.arange(300, 600)
    y_cal = y[cal].copy()
    y_cal[:, 5] = 0  # label 5 has no calibration positives -> dead
    with pytest.raises(ValueError):
        calibrate_per_label(p[cal], y_cal, p[ev], fallback="prior")  # prior must be explicit
    out_prior, info = calibrate_per_label(p[cal], y_cal, p[ev], prior, "prior")
    assert info["dead_labels"] == [5]
    assert np.allclose(out_prior[:, 5], prior[5])
    out_id, _ = calibrate_per_label(p[cal], y_cal, p[ev], prior, "identity")
    assert np.allclose(out_id[:, 5], p[ev, 5])
    out_ex, _ = calibrate_per_label(p[cal], y_cal, p[ev], prior, "exclude")
    assert np.all(np.isneginf(out_ex[:, 5]))
    for j in range(5):  # calibrated columns are monotone in the input
        order = np.argsort(p[ev, j])
        assert np.all(np.diff(out_prior[order, j]) >= -1e-12)
    # tau: a label with few positives is treated as dead when tau is at least that count
    y_cal[:, 4] = 0
    y_cal[:3, 4] = 1
    _, info_tau = calibrate_per_label(p[cal], y_cal, p[ev], prior, "prior", tau=3)
    assert set(info_tau["dead_labels"]) == {4, 5}


def test_select_calibrator_cv_is_deterministic_and_applies():
    p, y, prior = _synthetic(n=400)
    cal, ev = np.arange(200), np.arange(200, 400)
    sel1 = select_calibrator_cv(p[cal], y[cal], prior, k=3)
    sel2 = select_calibrator_cv(p[cal], y[cal], prior, k=3)
    assert sel1 == sel2
    assert set(sel1["oof_map"]) == {
        "per_label_tau0",
        "per_label_tau1",
        "per_label_tau2",
        "per_label_tau5",
        "per_label_tau10",
        "shared",
        "raw",
    }
    assert sel1["choice"] in ("per_label", "shared", "raw")
    out = apply_selection(sel1, p[cal], y[cal], p[ev], prior)
    assert out.shape == p[ev].shape
    assert np.isfinite(map_at_k(y[ev], out, 3))
