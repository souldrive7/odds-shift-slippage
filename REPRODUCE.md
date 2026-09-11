# Reproduce

Every result number in the paper is either a macro in `publications/full-paper/figures/numbers.tex` or a cell in a
generated table body `publications/full-paper/figures/tab_*.tex`. Both are written by `publications/full-paper/make_tables.py` from
`artifacts/results/canonical/*.json`. The exceptions are typed by hand in the `.tex` sources: design constants (split
fractions, hyper-parameters, the CI level, the first 500,000 rows the top-7 overlap is read over),
listed under Fixed choices below; the body rows of Supplement Table S7, which are literal LaTeX in
`supplement.tex` (last row of the map below, and its count macro is likewise a literal in
`make_tables.py`); and a rounding quoted in prose, the "factor of 1.2" in Limitations. This file maps each
paper element to the result file behind it, to the command that regenerates that file, to how long
that command actually took, and to where the number surfaces in the paper.

It also replaces the old `REPORT.md`, which described the same mapping in a second place and drifted
away from this one. `REPORT.md` is now a stub pointing here, and `supplement.tex` names this file.
Six macros are set by `make_tables.py` from code rather than from a result file --- `\nnLabels`,
`\nnSurveyImpl`, `\nshiftSearchBound`, `\nshiftClipLogit`, `\nsatMarginF` and `\nsatMarginD` ---
and everything else in `figures/numbers.json` comes from `artifacts/results/canonical/`.

## What can and cannot be reproduced from this repository

This decides how far you can get, so it comes first.

- **Reproducible with no data download.** The MULAN dose-response, the MULAN calibration-split
  statistics, the cross-validated selection rule, the leaf-saturation checks and the synthetic
  study. They fetch or generate their own inputs; the MULAN ARFF/XML files are downloaded at run
  time from the public mirror and pinned by content SHA-256 under `fetch_provenance` in the result
  file — except `mulan_cal_stats.json`, which fetches the same files but records no
  `fetch_provenance` (`DATA.md` section 7).
- **Reproducible once you rebuild the prediction arrays.** Everything on Santander and Instacart.
  The raw Kaggle files are not redistributed and the prediction arrays (`predictions.npz`,
  `p_te_all.npy`) are too large to ship. `code/santander_train/` and `code/instacart_train/`
  regenerate them from the public Kaggle downloads. `data/PRIMARY_ARRAYS.sha256`, and every result
  file's `_meta.inputs`, record the SHA-256 of the arrays we used, so you can tell whether you
  rebuilt the same object. See `DATA.md`. One part straddles the two bullets: `--part dead_freq`
  reads the arrays *and*, only when `dead_label_frequency.json` is absent from `--out-dir`, also
  computes its MULAN block (the 8 pre-extension datasets, computed with no arrays; see `DATA.md`
  section 7). The file is shipped in `artifacts/results/canonical/`, so with the commands below the block is reused and
  nothing is downloaded.
- **Not reproducible from this repository.** Three Santander scorers come from earlier runs whose
  training code is not part of this release: WGBoost (the W column of Supplement Table S1), an
  earlier run of the same weighted configuration (the "no cap, earlier run" row of Table S4) and an
  earlier 100-tree subsampled unweighted LightGBM (Table S2). `run_experiment.py` reads them from
  `uq_arrays.npz` and a second `predictions.npz` and records their hashes. If those files are
  absent, the ladder omits exactly those blocks and every other rung is unaffected — but
  `make_tables.py` then fails: it reads the `W_wgboost` block of `santander_ladder.json` unguarded,
  so the tables, the gate and the PDF build all stop. Keep the shipped `santander_ladder.json` if
  you cannot supply `uq_arrays.npz`.
- **Not shipped, build them yourself.** `main.pdf`, `supplement.pdf` and `main.bbl`.

## Verify the paper's numbers without recomputing (under a minute)

```bash
sha256sum -c SHA256SUMS.txt                             # every shipped file is intact
pip install -r requirements.txt
cd paper && python verify_numbers.py                    # it runs make_tables.py itself
```

Do not run `make_tables.py` yourself first: `verify_numbers.py` reads the shipped
`figures/numbers.tex`, regenerates it and compares, so regenerating it beforehand makes check 1
below vacuous — the shipped file is exactly what that check exists to test.

`verify_numbers.py` performs three checks, and it is worth being precise about them because they are
narrower than "the numbers are right":

1. it re-runs `make_tables.py` and fails if `figures/numbers.tex` was stale (only `numbers.tex` is
   compared; the table bodies are rewritten as a side effect but not diffed);
2. every macro used in `main.tex` and `supplement.tex` is defined (full-line comments are skipped,
   so a commented-out sentence may reserve a macro that does not exist yet);
3. the prose of those two files contains no hand-typed number of the three shapes its regex
   recognises — a decimal with two or more places, a percentage, or an integer written with
   thousands separators — outside an `\input figures/...` line, apart from an explicit allow-list of
   design constants such as 30% and 0.05.

So the gate makes it hard for the manuscript to carry a hand-typed result number *of those three
shapes*. It is not a proof that every number in the paper comes from `results/`: check 3 does not see
plain integers or one-decimal numbers, and a thousands separator written the LaTeX way escapes it as
well — `$500{,}000$` in `main.tex`, the "factor of $1.2$" in its Limitations and the "10 code bases"
of the Table S7 caption are all hand-typed and all pass today. Nor does the gate check that a result file is correct, that a table body is current, or
that a macro is used in the sentence it was computed for. `git diff figures/` after `make_tables.py`
is what catches a stale table body.

## Where files are written (read this before rerunning anything)

`code/run_experiment.py` and `code/chunked.py` default their output to
`artifacts/results/canonical/`, while `paper/make_tables.py` reads the same canonical directory.
Run them from the repository root and pass `--out-dir artifacts/results/canonical` when overriding
the default. `code/gate_check.py` reads the canonical directory automatically.

The commands below use two shell variables:

```bash
RUN="python code/run_experiment.py --out-dir artifacts/results/canonical"
ARR="--arrays-dir /path/to/arrays"   # Santander / Instacart only; one subdirectory per configuration
```

`--arrays-dir` points at the directory that holds one subdirectory per trained configuration
(`weighted/`, `unweighted/`, `weighted_mds0.3/`, ...), each containing `predictions.npz`. For
Santander the environment variable `DRC_ARRAYS_DIR` points one level higher: it overrides the
Santander *root*, the directory that contains `outputs_matched_pair/`, `outputs_mlp/`,
`outputs_uq_regen/` and `outputs_lgbm_baseline/`, and the code appends the per-configuration
subdirectory itself. Setting it to what `--arrays-dir` wants makes no model available: the run logs
`available models []` and then fails with `FileNotFoundError` on `<dir>/unweighted/predictions.npz`
— the labels are read from the unweighted configuration before any rung is attempted — so nothing is
written and no partial ladder survives. The Instacart parts additionally take
`--labels-bundle` for the sparse label file. `data/santander/` and `data/instacart/` in this
repository hold the `run_meta.json` of each matched-pair and MLP run (32 files; the three legacy
Santander scorers have none — see `DATA.md` section 3), plus `data/instacart/results/eda.json` (cited in
Sec. 3 as the check that both boosters start from the same initial score) and
`data/instacart/SHA256SUMS_raw.txt` — not the arrays.

## The map: paper element to result file to command

Runtimes are wall clock, measured on the machine that produced the shipped results (Python 3.12.10,
Windows, CPU only, default `--n-jobs 8`). They are read from each result file's `_meta.elapsed_sec`,
except the three MULAN parts, whose durations are recorded in the chain log that ran them, the
synthetic study, which stores `elapsed_sec` at the top level of its result file rather than inside
`_meta`, and the leaf and gate checks, noted in their rows. Instacart runs roughly 5 to 50 times
slower than Santander part for part: it has 4,000 labels against 24.

| Paper element | Result file | Command | Measured runtime | Where it appears |
|---|---|---|---|---|
| Fig. 1A Santander bars; Table 1 columns S / N; Sec. 3 repair ladder; Supp. Tables S1, S2, S4 | `santander_ladder.json` | `$RUN --dataset santander --part ladder $ARR` | 1,312 s (22 min) | macros `\nS…`, `\nN…`, `\nW…`, `\nNh…`, `\nSorig…`, the `\nSmds…` cap rungs, and the calibration-split statistics `\ncalPosMinAlive`, `\ncalPosMedAlive`; bodies `tab_ladder_v3_body.tex`, `tab_ladder_body.tex`, `tab_ladder_earlier_body.tex`, `tab_mds_body.tex` |
| Fig. 1A "ideal odds shift" bar; the Sec. 3 loss decomposition and top-7 overlap | `santander_whatif.json` | `$RUN --dataset santander --part whatif $ARR` | 7.9 s | `\nwhatif…`, `\noddsShare`, `\noverlapWhatif`, `\nfracRowsChangedWhatif` |
| Bootstrap intervals in Sec. 3; the evaluation-row denominators of Sec. 3. No supplement table carries an interval: the supplement refers to these intervals only indirectly, in the Table S3 caption, via Table 1 of the main text | `santander_bootstrap.json` | `$RUN --dataset santander --part bootstrap $ARR` | 98 s | `\nci…`, `\nboot…`, and `\nfracRowsWithPos`, `\nnEvalRows` |
| Supp. Table S3, deployment protocol (calibrators fitted on the validation period) | `santander_deploy.json` | `$RUN --dataset santander --part deploy $ARR` | 946 s (16 min) | `\ndep…`; body `tab_deploy_body.tex` |
| Supp. Table S5, calibration-split size sweep | `santander_calsize.json` | `$RUN --dataset santander --part calsize $ARR` | 465 s (7.8 min) | `\ncalsize…`; body `tab_calsize_body.tex` |
| Fig. 1B Santander points; Fig. 2 (main text) Santander points; the Sec. 3 cap sweep | `santander_capsweep.json` | `$RUN --dataset santander --part capsweep $ARR` | 343 s (5.7 min) | `\ncap…`, `\nSmds…TEtaC`, `\nSmds…bPriorMax`; also writes `tab_capsweep_santander_body.tex`, which v3 does not typeset |
| Tie attribution in Sec. 3 (re-ordering only the cells at exactly 1.0) | `santander_tie.json` | `$RUN --dataset santander --part tie $ARR` | 16 s | `\ntie…`, `\nStie…` |
| Mean and SD over the three 90% train-row draws, Sec. 3 | `santander_seeds.json` | `$RUN --dataset santander --part seeds $ARR` | 108 s | `\nseed…` |
| Table 1 columns MS / MN; read by `make_fig_regime.py` but **omitted from Fig. 2**: this pair's shift is identifiable on only 1 of the 24 Santander labels, too few for a median, so the script drops the point | `santander_mlp_ladder.json` | `$RUN --dataset santander_mlp --part ladder $ARR` | 185 s (3.1 min) | `\nM…`, `\nMS…`, `\nMN…`, `\nMcalPosMinAlive`, `\nMcalPosMedAlive`; body `tab_ladder_v3_body.tex` |
| The MLP odds-shift share in the contributions and in Sec. 3 | `santander_mlp_whatif.json` | `$RUN --dataset santander_mlp --part whatif $ARR` | 6.9 s | `\nModdsShare`, `\nMwhatif…` |
| Fig. 1A Instacart bars; Table 1 columns IS / IN; Sec. 3 Instacart paragraph | `instacart_ladder.json` | `$RUN --dataset instacart --part ladder $ARR` | 6,156 s (1 h 43 min) | `\nI…`, `\nIS…`, `\nIN…`, the `\nISmds…` cap rungs, and `\nIcalPosMinAlive`, `\nIcalPosMedAlive`; body `tab_ladder_v3_body.tex` |
| Fig. 1A Instacart "ideal odds shift" bar; the Instacart decomposition | `instacart_whatif.json` | `$RUN --dataset instacart --part whatif $ARR` | 382 s (6.4 min) | `\nIwhatif…`, `\nIoddsShare`, `\nIoverlap…` |
| Instacart bootstrap intervals in Sec. 3 | `instacart_bootstrap.json` | `$RUN --dataset instacart --part bootstrap $ARR` | 1,302 s (22 min) | `\nIci…`, `\nIboot…` |
| The Instacart deployment protocol quoted in Sec. 3 | `instacart_deploy.json` | `$RUN --dataset instacart --part deploy $ARR` | 10,316 s (2 h 52 min) | `\ndepI…`, `\nIproto…` |
| Instacart calibration-split size sweep. Computed, but **not cited by v3**: `make_tables.py` reads only the Santander sweep, so this file backs no macro and no table body | `instacart_calsize.json` | `$RUN --dataset instacart --part calsize $ARR` | 7,740 s (2 h 9 min) | nowhere in the paper |
| Fig. 1B Instacart points; Fig. 2 (main text) Instacart points; the Sec. 3 cap sweep | `instacart_capsweep.json` | `$RUN --dataset instacart --part capsweep $ARR` | 2,907 s (48 min) | `\nIcap…`, `\nISmds…TEtaC`, `\nISmds…bPriorMax`; also writes `tab_capsweep_instacart_body.tex`, which v3 does not typeset |
| Instacart tie attribution in the contributions and Sec. 3 | `instacart_tie.json` | `$RUN --dataset instacart --part tie $ARR` | 832 s (14 min) | `\nItie…`, `\nIStie…` |
| Instacart mean and SD over the three 90% train-row draws | `instacart_seeds.json` | `$RUN --dataset instacart --part seeds $ARR` | 1,389 s (23 min) | `\nIseed…` |
| Table 1 columns IMS / IMN; the Instacart MLP pair of Fig. 2 (main text), the one of the four matched pairs where the analytic inversion pays (the all-product MLP two rows below meets the same two conditions and the inversion pays there too) | `instacart_mlp_ladder.json` | `$RUN --dataset instacart_mlp --part ladder $ARR` | 1,553 s (26 min) | `\nIM…`, `\nIMS…`, `\nIMN…`, `\nIMcalPosMinAlive`, `\nIMcalPosMedAlive`; body `tab_ladder_v3_body.tex` |
| The Instacart MLP odds-shift share | `instacart_mlp_whatif.json` | `$RUN --dataset instacart_mlp --part whatif $ARR` | 319 s (5.3 min) | `\nIMwhatif…`, `\nIModdsShare` |
| All-product Instacart MLP (49,688 labels), weighted arm, Sec. 3 | `instacart_mlp_all_ladder_wr.json` | `python code/chunked.py --config wr --part ladder --out-dir results` | 7,605 s (2 h 7 min) | `\nIMA…`, `\nIMAS…` |
| All-product Instacart MLP, unweighted arm, Sec. 3 | `instacart_mlp_all_ladder_w1.json` | `python code/chunked.py --config w1 --part ladder --out-dir results` | 7,920 s (2 h 12 min) | `\nIMAN…` |
| Dead-label frequency against split size, quoted under the dead-label takeover remark (Remark 2 of the main text; its label is `prop:dead`, but what prints is a remark) | `dead_label_frequency.json` | `$RUN --dataset santander --part dead_freq $ARR`, then the same with `--dataset instacart` | 3.8 s for the last (Instacart) pass over an already-populated file. The MULAN block of the shipped file covers the **8** pre-extension datasets (emotions, scene, flags, birds, yeast, genbase, enron, medical) at 5 fractions x 10 seeds, and is reused whenever `dead_label_frequency.json` already exists in `--out-dir`. Deleting the file to force a recompute now scans all 11 datasets and will *not* reproduce the shipped block. The file merges one block per dataset and `_meta.elapsed_sec` records only the run that wrote it | `\ndeadFreq…` |
| Table 2 (MULAN dose-response) and Supp. Tables S11–S16 (every cell) | `mulan_dose_response.json` | `$RUN --part mulan` | 6,840 s (1 h 54 min) | `\nnMulan…`, `\nnDoseCells`, `\nnCollapseCells`, `\ndel…`, `\ncorel…`, `\nenron…`, `\nbibtex…`, `\nbirds…`, `\ngenbase…`, `\nflags…`, `\nmed…`; bodies `tab_mulan_compact_body.tex`, `tab_mulan_full_part1.tex` … `part6.tex` |
| The "cal. pos." column of Table 2 (median calibration positives per label) | `mulan_cal_stats.json` | `$RUN --part mulan_stats` | 19 s | the per-dataset `\ncalPosMed<dataset>…` macros only; the calibration-positives column of `tab_mulan_compact_body.tex`. The Santander/Instacart/MLP `\ncalPosMedAlive` and `\ncalPosMinAlive` are calibration-split statistics of the corresponding `*_ladder.json`, not of this file |
| **Supp. Table S6, the cross-validated selection rule** — the evidence for recipe step 5 and for the Sec. 4 negative result | `mulan_tau_select.json` | `$RUN --part mulan_tau` | 3,612 s (1 h 0 min) | `\ntau…`, `\n…TauSel`, `\n…CiTauZero`, `\n…CiTauSel`, `\nnHurtAfterTau`, `\nnTauSelBelowRaw`, `\nnSharedBelowRaw`; body `tab_mulan_tau_body.tex` |
| Supp. Tables S8 and S9, leaf saturation (single-tree identity check and the LightGBM sweep) | `leaf_check.json` | `$RUN --part leaf` | 147 s (sum of the three per-dataset `elapsed_sec`; no total is recorded) | `\nleaf…`; bodies `tab_leaf_tree_body.tex`, `tab_leaf_lgbm_body.tex` |
| Supp. Table S10, the synthetic study with known marginals | `synthetic_check.json` | `$RUN --part synthetic` | 349 s (5.8 min) | `\nsyn…`; body `tab_synthetic_body.tex` |
| Pre-registered accept-line gates G2 and G3 — provenance only, not cited in the paper | `gate_check.json` | `python code/gate_check.py` (reads `artifacts/results/canonical/`) | not recorded; it only reads result files | nowhere in the paper |
| Supp. Table S7, calibrators on a single-class calibration window | *none* | — | — | rows hand-written in `supplement.tex` from `docs/dead_label_survey.md`; the count macro is a literal in `make_tables.py` |

That is every result file in `results/` and every value of `--part`: `ladder` (alias `santander`),
`whatif`, `bootstrap`, `deploy`, `calsize`, `dead_freq`, `tie`, `capsweep`, `seeds`, `mulan`,
`mulan_stats`, `mulan_tau`, `leaf`, `synthetic`.

Some table bodies that `make_tables.py` writes are not typeset by v3 and are shipped only because
the generator writes them: `tab_capsweep_santander_body.tex` and `tab_capsweep_instacart_body.tex`
(the cap sweep appears as Fig. 1B instead), and `tab_mulan_calstats_body.tex` and
`tab_mulan_full_body.tex` (superseded by the compact Table 2 and by the six-part split S11–S16).

## Build the paper

The figure scripts are named after an older numbering, and the numbering is a trap: the script
called `make_fig2.py` makes the **supplement's** Fig. S1, and `make_fig_regime.py` makes the **main
text's** Fig. 2. Run all three.

```bash
cd paper
python verify_numbers.py     # runs make_tables.py itself: numbers.tex, numbers.json, every tab_*.tex
python make_fig1.py          # -> figures/fig1_decomposition_mechanism.pdf   = MAIN TEXT Fig. 1
                             #    from {santander,instacart}_{ladder,whatif,capsweep}.json
python make_fig_regime.py    # -> figures/fig2_inversion_regime.pdf          = MAIN TEXT Fig. 2
                             #    from {santander,instacart}_{capsweep,mlp_ladder}.json + figures/numbers.json
python make_fig2.py          # -> figures/fig2_shift_vs_lnw.pdf              = SUPPLEMENT Fig. S1
                             #    from santander_ladder.json
pdflatex main       && bibtex main       && pdflatex main       && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

The class is `acmart` (`sigconf,nonacm`) and the bibliography style is `ACM-Reference-Format`, so a
TeX installation carrying `acmart` is required; `main.bbl` is not shipped, which is why `bibtex` is
not optional. `make_bundle.sh` runs the same steps for both documents and packs the arXiv upload:
`main.tex`, `main.bbl`, only the `figures/` files that `main.tex` actually pulls in, and the compiled
supplement as the ancillary file `anc/supplement.pdf`.

The supplement is numbered independently of the main text: sections S1–S6, Tables S1–S16, Fig. S1.

## Fixed choices

Design constants, not results. They are recorded here because a rerun that changes any of them will
not reproduce the shipped numbers.

**Calibration split.** Santander and Instacart use the *test-split* protocol: 30% of the test rows,
selected by `numpy.random.default_rng(42)` — literally

```python
rng = np.random.default_rng(42); idx = np.arange(n); rng.shuffle(idx); n_val = int(0.3 * n)
```

— with the remaining 70% used for evaluation. On Santander that is 761,439 calibration rows out of
2,538,132 test rows. The *deployment* protocol (`--part deploy`) instead fits the calibrators on the
held-out validation period and evaluates on all test rows. MULAN uses the same seed and the same
fraction but carves the split out of the **training** rows, with a floor of 50 rows
(`ncal = max(int(len(x_tr) * 0.3), 50)`); the models are then fitted on what remains, which is the
`n_fit` column of Table 2.

**Metric.** MAP@K with K = 7, Kaggle definition: rows without positives excluded, stable argsort so
that ties break by label index. On MULAN the depth is `min(7, L)`, so on a dataset with L at most 7
the metric orders but selects nothing.

**Dead-label policies.** A label without calibration positives is handled by one of three named
policies — `identity` (keep the raw score), `prior` (the training prevalence) and `exclude` (drop it
from the candidate set). The policy is an explicit argument of every calibrator, never a fallback
buried in an exception handler; `prior` is the paper's recommendation.

**Bootstrap.** Per-row resampling, B = 2000, seed 42, over the evaluation rows that have at least one
positive: 51,403 on Santander and 89,525 on Instacart. Intervals are 95% percentile intervals (the
2.5th and 97.5th percentiles of the resampled means); the level itself is hand-typed, in the Fig. 1
caption of `main.tex` and the Table S6 caption of `supplement.tex`. The MULAN selection-rule
intervals use the same B = 2000, paired.

**Top-7 overlap window.** The top-7 overlap and changed-row shares of Sec. 3 are read over the first
`m = min(500_000, len(y))` test rows — 500,000 on Santander, all 131,209 on Instacart. The `500,000`
in the main text is hand-typed.

**Threshold grid and cross-validation.** The selection rule searches the positives threshold
over {0, 1, 2, 5, 10} — a label with at most that many calibration positives is mapped to its
prior — plus the shared isotonic map plus "do not calibrate", by 5-fold cross-validation *inside the
calibration split*: fit on four folds, score the held-out fold, pool the out-of-fold scores and
evaluate MAP@K once. Ties go to the smaller threshold and to per-label over shared, so the choice is
deterministic. It is evaluated on 110 cells (5 learners, 2 weight settings, 11 datasets).

**Leaf-step cap sweep.** `max_delta_step` c in {0.3, 0.7, 1, 2, 5}, against T = 60 boosting rounds at
learning rate eta = 0.05, so the step budget T·eta·c is the quantity the cap sweep tests.

**Train-row draws.** The `seeds` part is the mean and SD over three 90% train-row draws (seeds 0, 1,
2) of the matched pair; the Santander and Instacart pairs have no other random restarts.

**Split-size and frequency sweeps.** `--part calsize` sweeps the calibration fraction over
{0.3, 0.1, 0.03, 0.01, 0.003} with 5 seeds (42, 1, 2, 3, 4). `--part dead_freq` uses 10 seeds per
split size. `--part tie` re-breaks the tied cells at random over 5 seeds.

**MULAN sweep.** 11 datasets — emotions, scene, flags, yeast, birds, genbase, medical, enron,
bibtex, Corel5k, delicious — chosen by a rule rather than by result: every MULAN dataset that ships a
standard train/test split, except tmc2007 (61 MB) and Corel16k (366 MB), left out for size. Five
learners (LightGBM default, LightGBM regularized, XGBoost, logistic regression, MLP with
`pos_weight`), giving 55 learner-dataset cells, each at four weight magnitudes: w = 1, w = sqrt(r),
w = r and w = 10r, with r = n_-/n_+ per label. The selection-rule run covers the same learners and
datasets at w = 1 and w = r only.

**float32 storage.** This is the most important caveat for anyone rerunning the Santander and
Instacart saturation numbers. The Santander and Instacart prediction arrays — and only those — are
stored as single-precision probabilities. A cell "at exactly 1.0" is therefore one whose raw margin
exceeds 16.6 nat; in double precision the same threshold would be 36.7 nat. Every Santander and
Instacart saturation share, tie attribution and identifiability count is a property of the stored
probability array — the object a deployed ranker consumes — and not a claim that the learner
produced an infinite score. The MULAN, leaf-check and synthetic saturation columns are not
float32 storage artefacts: those parts write no array and measure saturation in process. The
arithmetic is not uniformly double precision, though — XGBoost and the torch MLP hand
`learners.py` float32 probabilities that it only then holds in a float64 array, so for those two
learners the 1 − 1e−9 threshold sits at the same 16.6-nat boundary rather than at the 20.7 nat it
implies (`DATA.md` section 4). The leaf check is float64 throughout. A pipeline that keeps the raw margin can still order the saturated cells; a
post-hoc calibrator applied to the probabilities cannot. Casting the shipped arrays to float64
changes nothing — a stored 1.0f is exactly 1.0 in double as well. What moves the numbers is
regenerating the Santander and Instacart predictions in double precision, i.e. changing the dtype
those two training scripts write: that would change their saturation shares and tie attributions,
and with them the raw and repaired MAP@7 of the affected rungs. What it would not change is the
*direction* of the matched-pair comparison.

**Shift identifiability.** The label-free prevalence-matching shift is solved by bisection on
[-40, +40] after clipping probabilities to [eps, 1-eps] with eps = 1e-15, so a clipped cell sits at
34.5 nat and pins the solution near that bound whatever the learner did. Every realized shift the
paper quotes therefore comes from labels with no saturated cell.

## The library and its tests

`code/oddslip/` is the small library the recipe is packaged as. Its README describes three entry
points (`predict_rank_loss`, `calibrate_per_label`, `select_calibrator_cv`); `__all__` exports 11
names in total — those three plus `apply_selection`, `calibrate_shared`, `odds_shift`,
`elkan_inversion`, `dead_labels`, `map_at_k`, `logit` and `sigmoid` — and the package ships an
executed tutorial notebook. The code is extracted from `run_experiment.py` and `mulan_dose.py` and
kept numerically identical to them, but the paper's numbers are produced by those scripts, not by
this package. Run its tests with

```bash
python -m pytest tests/test_oddslip.py
```

## Checking a rerun

```bash
git diff --stat results/     # only _meta.generated_at / elapsed_sec should change
cd paper && python make_tables.py && git diff --stat figures/
```

An empty second diff means every number in the paper reproduced, macros and table bodies alike. A
non-empty one names the file; each result file's `_meta.inputs` names the array. Compare its hash
against `data/PRIMARY_ARRAYS.sha256` where the array appears there — that ledger is built from the
`*_ladder*.json` files only, so four Instacart draw arrays (`unweighted_sub{0,1,2}`, `weighted_sub2`)
are absent from it — and otherwise against the `_meta.inputs` block of the result file itself.
