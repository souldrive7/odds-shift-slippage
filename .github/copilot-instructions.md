# Odds-Shift Slippage repository instructions

This repository contains the implementation, experiments, result
artifacts and papers for the Odds-Shift Slippage project.

## Sources of truth

- Scientific claims: `publications/arxiv/` (canonical); `publications/full-paper/` is the
  archived long version, `publications/ecir/` the anonymised conference version
- Canonical results: `artifacts/results/canonical/`
- Public implementation: `src/oddslip/`
- Shared bibliography: `tex/bibliography/references.bib`
- Shared result macros and generated tables/figures: `publications/shared/figures/`
  (written by `publications/shared/make_tables.py` and `make_fig*.py`)
- Canonical repository URL:
  `https://github.com/souldrive7/odds-shift-slippage`

## Scientific rules

1. Do not invent, interpolate, or estimate experimental results.
2. Do not claim that label reweighting is always harmful.
3. Limit the main claim to one-vs-rest per-instance top-K ranking.
4. Keep the test-split and deployment protocols distinct.
5. Distinguish stored float32 probability saturation from information
   that can remain available in raw margins.
6. Labels without calibration positives require prior fallback or
   explicit exclusion.
7. The full paper determines the scope of scientific claims.
8. Describe the Santander 23% result as the share of the observed loss
   accounted for by the what-if odds-shift rung, not as a unique causal
   decomposition.

## Editing rules

1. Do not directly edit generated figures, tables, or result macros.
2. Modify the canonical result file or generation script instead.
3. Do not run expensive model training unless explicitly requested.
4. Preserve Git history when moving files inside this repository.
5. Build the affected publication after editing its source.
6. Do not modify data or results merely to make tests pass.
7. Report missing files or inconsistent results instead of fabricating them.
8. Make small, reviewable changes.
9. Do not commit or push unless explicitly requested.