# Don't Reweight, Calibrate

Code, results and paper source for

> **Don't Reweight, Calibrate: Per-Label Class Weights Distort Top-K Ranking, the Textbook
> Inversion Does Not Repair It, and Per-Label Calibration Does**
> Akifumi Goto, 2026. arXiv preprint (identifier to be added).

## What the paper shows

Ranking many binary labels within a row (a customer's next products, a document's tags) is
Bayes-optimal by the marginal probabilities. Training one-vs-rest models with a label-specific
positive-class weight (`scale_pos_weight = n_-/n_+`) shifts each label's log-odds by `ln w_j` and
therefore changes the within-row ranking. On the Santander product-recommendation benchmark
that textbook shift explains only a fifth of the top-K loss of the weighted model; the rest is a
per-label monotone distortion produced by saturation of the boosted trees, which the analytic
inversion cannot undo but per-label calibration can, provided that labels without calibration
positives are mapped to their prior rather than left on the raw scale. Dose-response experiments on
eight MULAN benchmarks with five learners and a synthetic study reproduce the odds shift, its exact
inversion under a realizable learner, and the failure of the inversion under capacity limits.

## Layout

```
paper/        main.tex, refs.bib, generated figures/ (numbers.tex macros, table bodies, Fig. 1)
              make_tables.py (results -> macros), make_fig1.py, verify_numbers.py, make_bundle.sh
code/         run_experiment.py and modules; every result file is produced here (see code/README.md)
results/      the JSON result files the paper is generated from (SHA-256 of inputs recorded in _meta)
data/         santander_run_meta.json (training counts) and PRIMARY_ARRAYS.sha256; the arrays are not shipped
scripts/      check_release_safety.py
SHA256SUMS.txt  hashes of every file in this repository
```

## Verify the paper's numbers without recomputing (< 1 min)

```bash
sha256sum -c SHA256SUMS.txt                  # every shipped file is intact
cd paper && python make_tables.py && python verify_numbers.py
```

`make_tables.py` regenerates `figures/numbers.tex` and every table body from `results/*.json`;
`verify_numbers.py` fails if the prose contains a hand-typed result number, if a macro is
undefined, or if `numbers.tex` is stale. The manuscript therefore cannot cite a number that is
not in `results/`.

## Recompute the results

```bash
pip install -r requirements.txt
python code/run_experiment.py --part mulan_stats   # 15 s, no training, fetches MULAN files (hash-pinned)
python code/run_experiment.py --part leaf          # ~3 min
python code/run_experiment.py --part synthetic     # ~5 min
python code/run_experiment.py --part mulan         # ~10 min
# Santander parts need the two prediction arrays (see DATA.md):
python code/run_experiment.py --part santander --arrays-dir /path/to/arrays
python code/run_experiment.py --part whatif    --arrays-dir /path/to/arrays
python code/run_experiment.py --part bootstrap --arrays-dir /path/to/arrays
```

Every run rewrites the corresponding file in `results/` atomically; `git diff results/` then shows
whether anything but the `_meta` timestamp changed.

## Build the paper

```bash
cd paper
python verify_numbers.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_bundle.sh` runs the same steps and packs the arXiv upload (`main.tex`, `main.bbl`,
`figures/`).

## Data

See `DATA.md`. The Santander source data is Kaggle's and is not redistributed; the two fixed
prediction arrays are pinned by SHA-256 in `data/PRIMARY_ARRAYS.sha256`. MULAN benchmarks are
fetched at run time from the public MULAN mirror and pinned by content hash in the result files.

## Citation and license

See `CITATION.cff`. Code and documentation are under the MIT license (`LICENSE`); datasets keep
their own terms.
