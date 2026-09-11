"""oddslip -- what-if loss prediction, per-label calibration with an explicit dead-label fallback, and
cross-validated calibrator selection for per-row top-K ranking under per-label class weights."""

from .core import (
    apply_selection,
    calibrate_per_label,
    calibrate_shared,
    dead_labels,
    elkan_inversion,
    logit,
    map_at_k,
    odds_shift,
    predict_rank_loss,
    select_calibrator_cv,
    sigmoid,
)

__all__ = [
    "apply_selection",
    "calibrate_per_label",
    "calibrate_shared",
    "dead_labels",
    "elkan_inversion",
    "logit",
    "map_at_k",
    "odds_shift",
    "predict_rank_loss",
    "select_calibrator_cv",
    "sigmoid",
]
__version__ = "0.0.1"
