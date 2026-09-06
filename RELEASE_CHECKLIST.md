# Release checklist

## Before every push

| Check | Command | Expected |
|---|---|---|
| Shipped files intact | `sha256sum -c SHA256SUMS.txt` | all OK |
| Numbers gate | `cd paper && python make_tables.py && python verify_numbers.py` | `OK: no hand-typed result numbers; all macros defined` |
| Release safety | `python scripts/check_release_safety.py` | `OK: 0 matches` |
| No arrays committed | `git ls-files | grep -E '\.npz$'` | nothing |

## Before making the repository public

- [ ] Add the Santander feature pipeline and the two training scripts that produced the arrays
      (currently outside this repository; the paper's Reproducibility section refers to them).
- [ ] Deposit the two prediction arrays (568 MB) or state where they can be requested (`DATA.md`).
- [ ] Put the arXiv identifier in `CITATION.cff` and `README.md`.
- [ ] Tag the release and, optionally, mint a Zenodo DOI from the tag.
