# Data

No source data is redistributed and no prediction array is shipped. This repository ships the
result files (`artifacts/results/canonical/*.json`), the metadata of every training run, and the hashes that pin the
inputs those results were computed from.

Read this before `REPRODUCE.md`. It states which parts of the paper a reader can recompute from what
is here, and which parts need files that are not.

## 0. What is here, and what is not

Shipped:

- `artifacts/results/canonical/*.json` — every *result* number in the paper and the supplement. `REPRODUCE.md` maps each
  table, figure and claim to its result file and to the command that regenerates it. Six macros are
  computed by `make_tables.py` from code rather than read from a result file (`\nnLabels`,
  `\nnSurveyImpl`, `\nshiftSearchBound`, `\nshiftClipLogit`, and the `\nsatMarginF` / `\nsatMarginD`
  of §4), and a short list of design constants — split fractions, the CI level, hyper-parameter
  values, the body rows of Supp. Table S7 and the rounded "factor of 1.2" in Limitations — is typed
  by hand in the `.tex` sources; `README.md` and `REPRODUCE.md` enumerate both.
- `publications/shared/figures/numbers.json` — the resolved value of every macro used in the text.
- `data/santander/outputs_*/<config>/run_meta.json` and `data/instacart/outputs_*/<config>/run_meta.json`
  — 32 files: row counts, hyper-parameters, per-label positives and the per-label weight of every
  training run.
- `data/PRIMARY_ARRAYS.sha256` — 34 lines, `<sha256>  <result stem>:<array path>`, generated from the
  `_meta.inputs` block of the six `*_ladder*.json` files.
- `data/instacart/SHA256SUMS_raw.txt` (hashes of the six Instacart CSV files and of the zip) and
  `data/instacart/results/eda.json` (the Instacart data description and the design checks).

Not here, and not reconstructible from this repository alone:

1. **The raw Kaggle files** of both benchmarks. Kaggle competition terms apply; download them
   yourself (§1, §2).
2. **The prediction arrays.** They are pinned by SHA-256 but not redistributed (§3). The training
   scripts under `code/santander_train/` and `code/instacart_train/` regenerate every array of the
   matched pairs and of the MLPs from the raw data.
3. **Three Santander scorer columns** whose training scripts are *not* in this release: the earlier
   run of the weighted configuration (`S_orig_lgbm_spw`), WGBoost (`W_wgboost`), and the earlier
   100-tree unweighted LightGBM (`Nh_lgbm_unweighted_100`). They come from the author's thesis
   pipeline. All three arrays are hash-pinned, and nothing here retrains them. The earlier weighted
   run and the 100-tree run have their configurations recorded in Supp. Tables S4 and S2; for the
   WGBoost scorer the paper records only the method and its citation (Supp. Table S1), and no
   hyper-parameter of that run is given anywhere in this release. All three are robustness rows;
   every headline number of the paper's §3 (the matched-pair section) comes from the matched pairs.

The MULAN and synthetic parts need no local data at all: MULAN is fetched over the network at run
time (§5) and the synthetic study is generated from fixed seeds (§6).

## 1. Santander Product Recommendation (Kaggle)

- Source: <https://www.kaggle.com/competitions/santander-product-recommendation/data>,
  `train_ver2.csv`, 2,292,759,599 bytes (2.13 GiB), SHA-256
  `51f919f6cc7145b214010895b303c0e76070cc947112cb3d31d060c23cc29c98`. Kaggle competition terms
  apply; the file is not redistributed here.
- **Split.** `code/santander_train/train_matched_pair.py` sorts the public training file by
  `fecha_dato` and cuts it at fixed row-count fractions — 15% validation, 20% test, the remaining
  65% training — giving **8,248,933** training rows, **1,903,599** validation rows and **2,538,132**
  test rows over **24** products. The validation period is held out from model fitting and precedes
  the test period. Because the cut is by row count and not by calendar month, the two boundaries
  fall *inside* a month, so the boundary months are shared across the split. Both arms of every
  matched pair see the identical split, so that sharing cannot produce any difference the paper
  reports. This is not the Kaggle leaderboard split and the scores are not comparable to it: MAP@7
  follows the Kaggle definition on the rows with at least one added product (2.9% of the 1,776,693
  evaluation rows), whereas the public leaderboard counts rows without additions as zero.
- **Matched pair.** `S_lgbm_spw` and `N_lgbm_unweighted` are LightGBM one-vs-rest models trained by
  one script in one session with identical configuration — 60 trees, learning rate 0.05, 31 leaves,
  no feature or row subsampling, seed 42, the same feature pipeline and split — differing only in
  that `S` sets `scale_pos_weight = n_-/n_+` per label (78 to 8,248,932) and `N` sets no weight. The
  leaf-step caps (`max_delta_step` 0.3, 0.7, 1, 2, 5), the two extra doses (`sqrt(r)`, `10r`) and the
  three 90% train-row draws of each arm come from the same script.
- **Calibration protocols.** The *test-split* protocol fits calibrators on a random 30% of the test
  rows (seed 42, 761,439 rows) and evaluates on the remaining 1,776,693; the *deployment* protocol
  fits on the held-out validation period and evaluates on all test rows. Bootstrap intervals use
  B = 2000 per-row resamples of the 51,403 evaluation rows that carry a positive.
- **MLP.** `code/instacart_train/train_mlp_posweight.py --dataset santander` trains the shared-trunk
  MLP pair on the dense feature cache written by `train_matched_pair.py --save-features-cache`.

## 2. Instacart Online Grocery Shopping Dataset 2017

- Source: the Kaggle competition *Instacart Market Basket Analysis* (2017). The original
  `instacart.com/datasets/grocery-shopping-2017` page no longer resolves and the competition page is
  not reachable through the Kaggle API; the same six CSV files are available as the Kaggle dataset
  `psparks/instacart-market-basket-analysis` (207,073,669-byte zip, byte-identical to the
  `yasserh/instacart-online-grocery-basket-analysis-dataset` mirror; the competition download itself
  could no longer be fetched for comparison). SHA-256 of the
  zip and of every CSV are in `data/instacart/SHA256SUMS_raw.txt`. Kaggle terms apply; nothing is
  redistributed.
- **Formulation.** Next-basket top-K recommendation, not the Kaggle reorder-set task and not
  comparable to its leaderboard: one row per user and reference order, features from that user's
  prior orders only, within-user chronological split — fit on order *n*−2, calibrate on order *n*−1,
  test on order *n*. 131,209 users with a labelled last order give 131,209 rows in each of the three
  splits. See `code/instacart_train/README.md`.
- **Label set.** The 4,000 most frequent products among the 9,035 with at least 20 positives in the
  training rows (cut at 4,000 for compute budget); the rule and the counts are in every Instacart
  `outputs_matched_pair/<config>/run_meta.json` under `label_set` (13 of the 32 shipped files; the
  two Instacart MLP and the 17 Santander `run_meta.json` files do not carry the key). The
  all-product MLP arms score all labels; see §3.
- `data/instacart/results/eda.json` records the data description (users, orders, basket sizes), the
  imbalance range, and the two checks that fixed the design (LightGBM `lgb.train` vs
  `LGBMClassifier` parity; common initial score with and without weights).

## 3. The prediction arrays

`code/datasets.py` looks for the arrays under `data/<benchmark>/`. `--arrays-dir` overrides the
directory of the `--dataset` being run; the environment variable `DRC_ARRAYS_DIR` overrides the
Santander root only. Every path below is relative to `data/`.

| Path | Configs | Keys | Consumed by |
|---|---|---|---|
| `santander/outputs_matched_pair/<config>/predictions.npz` | 15: `unweighted`, `weighted`, `weighted_mds{0.3,0.7,1,2,5}`, `weighted_sqrt`, `weighted_10r`, `weighted_sub{0,1,2}`, `unweighted_sub{0,1,2}` | `p_te`, `p_va` (float32), `Y_te`, `Y_va` (int8) | `santander_ladder`, `_whatif`, `_tie`, `_capsweep`, `_calsize`, `_bootstrap`, `_deploy`, `_seeds` |
| `santander/outputs_matched_pair/features_cache/` | — | `X_{tr,va,te}.npy` (float32), `Y_{tr,va,te}.npy` (int8) | input to the Santander MLP only |
| `santander/outputs_mlp/<w1\|wr>/predictions.npz` | 2 | same keys as above | `santander_mlp_ladder`, `santander_mlp_whatif` |
| `santander/outputs_uq_regen/uq_arrays.npz` | 1 | `Y_te`, `p_te_br` (earlier weighted run), `p_te_wg` (WGBoost) | `santander_ladder`, `_bootstrap`, `_calsize`, `_whatif` |
| `santander/outputs_lgbm_baseline/predictions.npz` | 1 | `p_te_lgbm` (earlier unweighted 100-tree run) | `santander_ladder`, `_bootstrap`, `_calsize`, `_whatif` |
| `instacart/data/processed/{features,labels_topN,labels_all,split}.npz` | — | features, top-4,000 labels, all-product labels, split index | written by `code/instacart_train/build_features.py`; read by `train_matched_pair.py` and `train_mlp_posweight.py`; label bundles are re-checked against every `Y_te` |
| `instacart/outputs_matched_pair/<config>/predictions.npz` | 13: `unweighted`, `weighted`, `weighted_mds{0.3,0.7,1,2,5}`, `weighted_sub{0,1,2}`, `unweighted_sub{0,1,2}` (no dose variants) | `p_te`, `p_va` (float32), `Y_te`, `Y_va` (int8) | `instacart_ladder`, `_whatif`, `_tie`, `_capsweep`, `_calsize`, `_bootstrap`, `_deploy`, `_seeds` |
| `instacart/outputs_mlp/<w1\|wr>/predictions.npz` | 2 | same keys, top-4,000 columns | `instacart_mlp_ladder`, `instacart_mlp_whatif` |
| `instacart/outputs_mlp/<w1\|wr>/{p_te_all,p_va_all}.npy` | 4 | float32 label-major `(L, n)` memmaps over all products | `instacart_mlp_all_ladder_w1`, `..._wr` (via `code/chunked.py`) |

`data/PRIMARY_ARRAYS.sha256` lists the 34 array hashes recorded by the six `*_ladder*.json` files. It
is not a complete inventory: four Instacart draw arrays (`unweighted_sub{0,1,2}`, `weighted_sub2`)
are used only by `instacart_seeds`, `_bootstrap`, `_calsize` and `_deploy`, and are pinned in the
`_meta.inputs` block of those files instead. The key strings inside `_meta.inputs` are the paths as
each run saw them and differ in shape between result files; the SHA-256 is what identifies an array,
not the key.

Only the 32 matched-pair and MLP `run_meta.json` files are shipped
(`data/…/outputs_{matched_pair,mlp}/<config>/run_meta.json`, local paths stripped by the export).
The `outputs_uq_regen` and `outputs_lgbm_baseline` rows have no shipped `run_meta.json` — their
arrays are pinned by SHA-256 in the result files; the only descriptions of them here are the
`_meta.models` entries of `results/santander_ladder.json` (which record the 100-tree run's
hyper-parameters verbatim) and the paper's Supp. Tables S1, S2 and S4 (see §0) — and there is no
`run_meta.json` for `features_cache/` or the Instacart processed bundles either. The arrays
themselves are derived from Kaggle data; the author will deposit them, or provide them on request,
where the competition terms allow. Otherwise regenerate them with
`code/santander_train/train_matched_pair.py` (with `io_data.py`, `train_models.py`) and
`code/instacart_train/{build_features.py,train_matched_pair.py,train_mlp_posweight.py}`.

## 4. Precision of the stored predictions

**This is the single most important caveat for anyone reproducing or re-analysing the Santander and
Instacart saturation numbers.** It does not apply to the MULAN, leaf-check or synthetic parts, which
store no array (see the second bullet below).

Every prediction array is stored as single-precision probabilities (`p_te`, `p_va`, `p_te_all`,
`p_va_all` are float32; labels are int8). The largest float32 strictly below 1 is
1 − 2⁻²⁴ ≈ 0.99999994, whose logit is **16.6 nat**; any probability above it rounds to exactly
`1.0f` on storage. In double precision the same boundary (1 − 2⁻⁵³) would sit at **36.7 nat**.

So a cell reported as "at exactly 1.0" is a cell whose raw margin exceeded 16.6 nat — not a cell for
which the learner produced an infinite score. Consequently:

- The Santander and Instacart saturated shares (16.6% of the Santander weighted model's cells at
  exactly 1.0 and 36.5% at exactly 0.0; 19.4% at 1.0 on Instacart), every tie attribution,
  and every shift-identifiability count (5 of 24 Santander labels with no saturated cell against 19
  labels where the shift is not identified; 1 of 24 for the Santander MLP) is a property of the
  **stored array at that precision** — the object a deployed ranker consumes — and not of the
  learner's internal margins.
- The MULAN, leaf-check and synthetic saturation columns are **not** affected by the storage
  precision. Those pipelines never write an array: `mulan_dose.py`, `leaf_check.py` and
  `synthetic_check.py` build their predictions in memory and measure saturation there
  (`sat_ge_1m1e9` and `sat_le_1e9` in `mulan_dose_response.json` and `synthetic_check.json`;
  `sat_ge_1m1e3` and `sat_ge_1m1e9` in `leaf_check.json`, where `sat_ge_1m1e3` is a 0.999 threshold,
  about four orders of magnitude coarser than the float32 boundary — 1e-3 against 2⁻²⁴ ≈ 6e-8).
  The saturated-share column of the full MULAN table comes from `sat_ge_1m1e9`. The arithmetic
  behind these columns is not uniformly double precision, though. The leaf check is float64
  throughout (scikit-learn decision trees and LightGBM). In the MULAN and synthetic sweeps the
  LightGBM and logistic-regression learners return float64 probabilities, but XGBoost and the torch
  MLP return float32 ones that `learners.py` only then holds in a float64 array, so for those two
  learners the 1 − 1e-9 threshold is in practice the same float32 boundary of 16.6 nat rather than
  the 20.7 nat the threshold itself would imply.
- A pipeline that keeps the raw margin instead can still order those cells. A post-hoc calibrator
  applied to the stored probabilities cannot; that is the sense in which the ties lie beyond any
  separable map.
- The prevalence-matching shift `b_j` is solved by bisection on [−40, 40] after clipping
  probabilities to [ε, 1−ε], so a clipped cell sits at ±34.5 nat and pins the solution near that
  bound whatever the learner did. Every `b_j` the paper quotes therefore comes from labels with no
  saturated cell.
- Casting the *shipped* arrays to float64 changes nothing: a stored `1.0f` is exactly 1.0 in double
  as well. What moves the numbers is re-running the Santander and Instacart training scripts in
  float64, or storing the margins rather than the probabilities. That changes those two benchmarks'
  saturation shares and tie attributions, and with them every rung that reads a tie — the weighted
  model's raw and repaired MAP@K included, because the tied cells are currently ordered by label
  index (stable argsort) and re-ordering only them by the unweighted model's order already returns
  5% of the loss on Santander and 54% on Instacart. What it would *not* change is the direction of
  the matched-pair comparison. Neither operation touches the MULAN, leaf-check or synthetic parts,
  which store no array.

## 5. MULAN benchmarks

- Fetched at run time from the public MULAN mirror
  (`raw.githubusercontent.com/tsoumakas/mulan/master/data/multi-label/<name>/`) by `code/mulan_io.py`.
  There is no local cache: the MULAN parts need network access, and every fetched ARFF/XML is pinned
  by content SHA-256 and byte count under `fetch_provenance` in `results/mulan_dose_response.json`
  (33 files), `results/mulan_tau_select.json` (33) and `results/leaf_check.json` (9).
- **Selection rule.** The **11** datasets are every dataset in the MULAN GitHub data folder that
  ships a standard train/test split, less `tmc2007` (61 MB) and the ten `Corel16k` subsets
  (`Corel16k001`–`Corel16k010`, 366 MB together, each with its own split), left out for size and
  total compute. That is the only rule applied, and a reader can check it against the distribution:
  22 datasets there ship a standard split, and the ones that ship none — `bookmarks`, `cal500`,
  `foodtruck` — fall outside the rule. The list is `DATASETS` in `code/mulan_dose.py`.
- **Split.** The standard train/test files are used as published. The calibration split is
  `max(int(0.3 × n_train_file), 50)` rows drawn from the training file at seed 42; the rest is the
  fit split. `n_fit + n_cal` is therefore the size of the upstream training file (for `flags` the
  floor binds: 38 → 50).
- **Metric.** MAP@K with **K = min(7, L)** on the standard test file. On the three datasets with
  L ≤ 7 (`emotions`, `scene`, `flags`) the depth covers every label, so the metric orders but selects
  nothing; those anchor the low-dose end.
- **Weight grid.** `w_j ∈ {1, sqrt(r_j), r_j, 10 r_j}` with `r_j = n_{-,j} / n_{+,j}` measured on the
  fit split, crossed with five learners (LightGBM default, LightGBM strongly regularized, XGBoost,
  ℓ2-regularized logistic regression with sample weights, and a two-layer MLP with per-label
  `pos_weight`). A label with no positive in the fit split is recorded as `r = 1` rather than as a
  ratio, so an exact `min r = 1` means the dataset contains such a label.

| Dataset | L | fit rows | cal. rows | test rows | features | K | min r | max r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| emotions | 6 | 274 | 117 | 202 | 72 | 6 | 1.49 | 3 |
| scene | 6 | 848 | 363 | 1,196 | 294 | 6 | 3.51 | 6 |
| flags | 7 | 79 | 50 | 65 | 19 | 7 | 0.32 | 6 |
| yeast | 14 | 1,050 | 450 | 917 | 103 | 7 | 0.32 | 80 |
| birds | 15 | 226 | 96 | 323 | 264 | 7 | 7.69 | 112 |
| genbase | 27 | 325 | 138 | 199 | 1,186 | 7 | 1.00 | 324 |
| medical | 45 | 234 | 99 | 645 | 1,449 | 7 | 1.00 | 233 |
| enron | 53 | 787 | 336 | 579 | 1,001 | 7 | 0.93 | 786 |
| bibtex | 159 | 3,416 | 1,464 | 2,515 | 1,836 | 7 | 5.85 | 227 |
| Corel5k | 374 | 3,150 | 1,350 | 500 | 499 | 7 | 1.00 | 3,149 |
| delicious | 983 | 9,044 | 3,876 | 3,185 | 500 | 7 | 1.49 | 903 |

The largest imbalance ratio the sweep reaches is `max r` = 3,149 on Corel5k (so a largest weight of
31,490 at the `10r` arm) — three orders of magnitude below Santander's `max r` of 8,248,932 — so the
sweep is a dose-response over imbalance magnitude that also reaches the label count of the Santander
benchmark (24) and comes within roughly a factor of four of the Instacart top-4,000 label set
(delicious, L = 983; the all-product MLP arm scores 49,688). The two datasets that carry most of
the collapse, Corel5k and delicious, have 500 and 3,185 test rows; the paper says so in its Limitations.

MULAN data retain their upstream terms.

## 6. Synthetic study

`code/synthetic_check.py` generates its own data from fixed seeds; no external data and no network.

## 7. Scope limits of individual result files

Three files cover less than their names suggest. A reader who assumes otherwise will mis-read them.

- **`results/leaf_check.json`** covers **3** MULAN datasets — `enron`, `medical`, `scene` — not all
  11. It has two blocks per dataset: single decision trees (`min_samples_leaf` = 20), which check
  Lemma 1's predicted saturation against the measured one, and LightGBM (200 trees) at
  `min_child_samples ∈ {5, 20, 50, 200}`. Both are crossed with `w ∈ {1, r, 10r, 100r}` — a heavier
  weight grid than the dose-response, which stops at `10r`.
- **`results/dead_label_frequency.json`** covers the **8** pre-extension MULAN datasets
  (`emotions`, `scene`, `flags`, `birds`, `yeast`, `genbase`, `enron`, `medical`) plus `santander`,
  `instacart` and `instacart_all`. `bibtex`, `Corel5k` and `delicious` are **not** in it. It reports
  the mean, min and max number of labels with no calibration positive over 10 random splits, at
  calibration fractions {0.05, 0.1, 0.2, 0.3, 0.5} for the MULAN blocks and
  {0.001, 0.003, 0.01, 0.03, 0.1, 0.3} for the `santander`, `instacart` and `instacart_all` blocks;
  it trains nothing. The paper's headline dead-label figure — a mean of 8.2 of the 24 Santander
  labels dead over the 10 seeds, at 25,381 calibration rows — is the `0.01` entry of the
  `santander` block, a fraction the MULAN grid does not contain.
- **`results/mulan_cal_stats.json`** covers all 11 datasets but records only calibration-split
  statistics (median, minimum and counts of calibration positives per label). It carries no
  `fetch_provenance`.

### Which files carry a `_meta` block

- **`_meta` with an `inputs` map of array hashes** (the ones a reader can audit against
  `PRIMARY_ARRAYS.sha256`): `santander_ladder`, `santander_whatif`, `santander_tie`,
  `santander_capsweep`, `santander_calsize`, `santander_bootstrap`, `santander_deploy`,
  `santander_seeds`, `santander_mlp_ladder`, `santander_mlp_whatif`, and the same ten for Instacart,
  plus `instacart_mlp_all_ladder_w1` and `..._wr`. `gate_check.json` is the exception in kind: its
  `_meta.inputs` hashes *result files*, not arrays.
- **`_meta` without `inputs`** (timestamp and run parameters only): `leaf_check.json`
  (`generated_at`), `synthetic_check.json` (`generated_at`), `mulan_cal_stats.json` (`generated_at`,
  `cal_split`), `dead_label_frequency.json` (`generated_at`, `n_seeds`, `last_dataset`,
  `elapsed_sec`).
- **No `_meta` block at all**: `mulan_dose_response.json` and `mulan_tau_select.json`. Their
  provenance is the top-level `fetch_provenance` map, and their configuration is in the top-level
  `learners`, `weight_schemes` / `weights`, `taus`, `cv` and `bootstrap` keys. Do not look for
  `_meta` in these two.
