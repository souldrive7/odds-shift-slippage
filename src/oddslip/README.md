# oddslip

`oddslip` is the package name for *odds-shift slippage*: the gap between the log-odds shift a per-label class
weight promises (ln w_j) and the shift a finite learner realizes. Three functions for per-row top-K
ranking under per-label class weights, extracted from the code that produces every number of the
paper (`run_experiment.py`, `mulan_dose.py`) and kept numerically identical to it.
Dependencies: numpy, scikit-learn.

```python
import numpy as np
from oddslip import predict_rank_loss, calibrate_per_label, select_calibrator_cv, apply_selection

# p_unweighted, p_weighted: (n, L) probabilities from the same configuration without / with w_j = n-/n+
# y: (n, L) labels; w: (L,) weights; prior: (L,) training prevalence
d = predict_rank_loss(p_unweighted, w, y=y, k=7, p_weighted=p_weighted)
d["odds_shift_share"]        # the part of the weighted model's MAP@7 loss that the textbook odds shift explains

# calibration split (p_cal, y_cal) -> repaired scores for p, dead labels mapped to the prior
p_rep, info = calibrate_per_label(p_cal, y_cal, p, prior=prior, fallback="prior", tau=0)
info["dead_labels"]

# choose per-label (tau), shared or raw by 5-fold CV of MAP@7 inside the calibration split
sel = select_calibrator_cv(p_cal, y_cal, prior, k=7)
p_sel = apply_selection(sel, p_cal, y_cal, p, prior)
```

Design rules (from the paper): the dead-label policy is an explicit argument, never a silent
pass-through; the prior is the training prevalence, not the calibration-split mean; selection never
touches evaluation labels. Tests: `python -m pytest tests/test_oddslip.py`.
