# Reproduce

Every number in the paper is a macro in `paper/figures/numbers.tex` or a cell in a generated
table body `paper/figures/tab_*.tex`; both are written by `paper/make_tables.py` from
`results/*.json`. This file maps each paper element to its result file and the command that
regenerates it.

| Paper element | Result file | Command | Time |
|---|---|---|---|
| Fig. 1A, Sec. 4.1 decomposition (odds-shift share, overlap statistics) | `results/santander_whatif.json` | `python code/run_experiment.py --part whatif --arrays-dir <dir>` | ~1 min |
| Fig. 1B, Table 1 repair ladder, saturation shares, AUC, dead-label diagnostics | `results/santander_ladder.json` | `python code/run_experiment.py --part santander --arrays-dir <dir>` | ~3 min |
| Bootstrap intervals in Sec. 4.1 | `results/santander_bootstrap.json` | `python code/run_experiment.py --part bootstrap --arrays-dir <dir>` | ~5 min |
| Table 2 and the appendix MULAN tables | `results/mulan_dose_response.json` | `python code/run_experiment.py --part mulan` | ~10 min |
| Calibration positives per label (Table 2) | `results/mulan_cal_stats.json` | `python code/run_experiment.py --part mulan_stats` | 15 s |
| Leaf-saturation table and the LightGBM sweep | `results/leaf_check.json` | `python code/run_experiment.py --part leaf` | ~3 min |
| Synthetic table | `results/synthetic_check.json` | `python code/run_experiment.py --part synthetic` | ~5 min |

Then

```bash
cd paper
python make_tables.py        # results -> figures/numbers.tex, figures/tab_*.tex
python make_fig1.py          # results -> figures/fig1_decomposition_ladder.pdf
python verify_numbers.py     # gate: no hand-typed numbers, no undefined macro, numbers.tex current
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Fixed choices

- Calibration split: 30 % of the test rows, `numpy.random.default_rng(42)` permutation
  (`rng.shuffle(idx); n_val = int(0.3 * n)`); MULAN uses the same seed and fraction on the
  training rows with a minimum of 50 calibration rows.
- MAP@K: Kaggle definition, rows without positives excluded, stable argsort (ties broken by
  label index).
- Dead-label policies (labels without calibration positives): identity, prior (training
  prevalence), exclude. The policy is a named argument of every calibrator.
- Bootstrap: per-row resampling, B = 2000, seed 42.
- Learners and weight schemes for MULAN and the synthetic study: `code/learners.py`,
  `code/mulan_dose.py` (`WEIGHT_SCHEMES`), `code/synthetic_check.py`.

## Checking a rerun

```bash
git diff --stat results/         # only _meta.generated_at / elapsed_sec should change
cd paper && python make_tables.py && git diff --stat figures/   # empty if every number reproduced
```
