# Release checklist

## Before every push

| Check | Command | Expected |
|---|---|---|
| Shipped files intact | `sha256sum -c SHA256SUMS.txt` | all OK |
| Numbers gate | `cd paper && python make_tables.py && python verify_numbers.py` | `OK: no hand-typed result numbers; all macros defined` |
| Release safety | `python scripts/check_release_safety.py` | `OK: 0 matches` |
| No arrays committed | `git ls-files | grep -E '\.npz$'` | nothing |

## Before making the repository public

- [x] Ship the training script of the matched Santander pair (`code/santander_train/`). The WGBoost
      and the earlier 100-tree scripts stay outside; their configurations are recorded.
- [ ] Deposit the prediction arrays or state where they can be requested (`DATA.md`); the matched pair can be regenerated from the Kaggle CSV.
- [ ] Put the arXiv identifier in `CITATION.cff` and `README.md`.
- [ ] Tag the release and, optionally, mint a Zenodo DOI from the tag.
