# code/ — every number in the paper, recomputed from fixed inputs

This directory produces all evidence cited in the paper. Nothing here retrains the Santander
models: the Santander parts read fixed prediction arrays and recompute every calibrator,
ladder entry, bootstrap interval and diagnostic from them. The MULAN, leaf and synthetic parts
train small models from public data with fixed seeds.

## Inputs

| Input | File | Content | How it is pinned |
|---|---|---|---|
| Santander test labels, WGBoost scorer, earlier weighted run | `uq_arrays.npz` | `Y_te` (labels), `p_te_wg` (WGBoost, direct probabilities), `p_te_br` (an earlier run of the weighted LightGBM configuration; the second row of the max_delta_step table) | SHA-256 `843c419c…` (full value in `data/PRIMARY_ARRAYS.sha256` and in every result file's `_meta.inputs`) |
| Santander matched pair | `matched_pair/<config>/predictions.npz` | `p_te`, `p_va`, `Y_te`, `Y_va` for `weighted` (LGBM-w, the paper's weighted model) and `unweighted` (LGBM-0, the same configuration without weights), trained by one script in one session; `weighted_mds0.7`, `weighted_mds2` (`max_delta_step` variants); produced by `santander_train/train_matched_pair.py` | SHA-256 in `data/PRIMARY_ARRAYS.sha256` and every result file's `_meta.inputs` |
| Santander earlier unweighted scorer | `predictions.npz` | `p_te_lgbm` (LightGBM one-vs-rest, no weights, 100 trees, feature/row subsampling; the robustness column LGBM-0_100) | SHA-256 `bec72a93…` |
| Santander training counts | `data/santander_run_meta.json` | `n_train` and per-label `n_pos` of the training period (gives `w_j = n_-/n_+` and the training prevalence) | shipped |
| MULAN benchmarks | fetched at run time from the public MULAN mirror | ARFF/XML for emotions, scene, flags, birds, yeast, genbase, enron, medical | content SHA-256 recorded under `fetch_provenance` in the result file |

The `.npz` arrays are not in the repository. Put them in `data/` or in any
directory and pass `--arrays-dir` (or set `DRC_ARRAYS_DIR`). See `DATA.md` for how to obtain them.

## Run

```bash
python code/run_experiment.py --part santander    # results/santander_ladder.json   (~3 min, 2.5M rows)
python code/run_experiment.py --part whatif       # results/santander_whatif.json   (~1 min)
python code/run_experiment.py --part bootstrap    # results/santander_bootstrap.json (~5 min, B=2000)
python code/run_experiment.py --part mulan        # results/mulan_dose_response.json (~10 min, 8 datasets x 5 learners x 4 weights)
python code/run_experiment.py --part mulan_stats  # results/mulan_cal_stats.json    (~15 s, no training)
python code/run_experiment.py --part leaf         # results/leaf_check.json         (~3 min)
python code/run_experiment.py --part synthetic    # results/synthetic_check.json    (~5 min)
python code/run_experiment.py --part deploy       # results/santander_deploy.json   (~3 min; calibrate on the validation period)
python code/run_experiment.py --part calsize      # results/santander_calsize.json  (~10 min; calibration-split size sweep)
python code/run_experiment.py --part mulan_tau    # results/mulan_tau_select.json   (~5 min; cross-validated tau / shared selection + bootstrap)
python code/run_experiment.py --part dead_freq    # results/dead_label_frequency.json (~30 s; labels only)
```

Every result file is written atomically and carries a `_meta` block with the input hashes, the
calibration-split seed and fraction, the dead-label policies and the MAP definition.

## What each part computes

- **santander**: the repair ladder for the three scorers. Raw; analytic (Elkan) inversion with the
  known weights; per-label isotonic regression fitted on a 30 % calibration split with three explicit
  policies for labels that have no calibration positives (identity, prior, exclude); per-label
  intercept-only shift, Platt and beta calibration with the prior policy; pooled isotonic; the
  label-free prevalence-matching shift; the popularity baseline; and the in-sample per-label isotonic
  ceiling. Also the exact-value saturation shares and the mean within-label AUC.
- **whatif**: the ideal odds shift `logit(p) + ln w_j` applied to the unweighted model, the top-7 overlap
  statistics, and the share of the weighted model's loss that the shift explains.
- **bootstrap**: per-row bootstrap intervals (B = 2000, seed 42) for the ladder entries and their differences.
- **mulan**: dose-response across 8 datasets, 5 learners and 4 weight magnitudes, with the same ladder per cell.
- **mulan_stats**: calibration-split statistics (positives per label) per dataset.
- **leaf**: the leaf-saturation lemma checked exactly on single trees and empirically on LightGBM.
- **synthetic**: the odds shift, its inversion and saturation against known marginals.
- **deploy**: the deployment protocol (calibrators fitted on the validation period, evaluated on all test rows) for every matched-pair config.
- **calsize**: calibration-split fraction {0.3, 0.1, 0.03, 0.01, 0.003} x 5 seeds: per-label (prior) vs shared isotonic, dead labels, positives per label.
- **mulan_tau**: the recipe's selection rule: 5-fold CV inside the calibration split over tau in {0,1,2,5,10} and the shared map, evaluated on the test split, with paired row-bootstrap intervals for the default LightGBM cells.
- **dead_freq**: how often a calibration split of a given size leaves a label without positives (MULAN train files and the Santander test period).

Calibration split, reproduced exactly by

```python
rng = np.random.default_rng(42); idx = np.arange(n); rng.shuffle(idx); n_val = int(0.3 * n)
```

MAP@7 follows the Kaggle definition (rows without positives excluded, stable argsort).

## Modules

| File | Role |
|---|---|
| `run_experiment.py` | entry point; Santander parts and the CLI |
| `mulan_dose.py` | MULAN dose-response and calibration statistics |
| `mulan_io.py` | MULAN fetch (hash-pinned) and ARFF parser |
| `leaf_check.py` | leaf-saturation lemma checks |
| `synthetic_check.py` | synthetic study with known marginals |
| `learners.py` | one-vs-rest learners with a per-label positive weight (LightGBM, XGBoost, logistic regression, MLP) |
| `santander_train/train_matched_pair.py` | trains the matched Santander pair from the Kaggle CSV (with `io_data.py`, `train_models.py`); ~30 min on a laptop, ~17 GB RAM |
