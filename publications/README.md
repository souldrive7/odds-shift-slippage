# Paper versions

Three versions of one paper share one set of generated inputs. Every result number in every version
is a macro from `shared/figures/numbers.tex`, and every table body and figure comes from
`shared/figures/`; the gate `shared/verify_numbers.py` scans all versions and fails on any hand-typed
result number or undefined macro.

| Directory | Role | Format | Length | Status |
|---|---|---|---|---|
| `arxiv/` | **Canonical.** The version to read, cite and submit to arXiv (cs.IR, cross-list cs.LG). | acmart `sigconf,nonacm`, two columns | 10 pages incl. references; supplement 9 pages (arXiv ancillary file) | current |
| `ecir/` | Conference version (double-anonymous review; venue named after notification): the arXiv text condensed to the LNCS limit and anonymised. | Springer `llncs` | 12 pages + references; supplement 9 pages (hosted in the anonymous mirror) | current |
| `full-paper/` | Archived long version (the pre-condensation text). Kept for diffing; not edited further. | acmart `sigconf,nonacm` | 14 pages; supplement 7 pages | frozen |

## What lives where

```
shared/
  make_tables.py        artifacts/results/canonical/*.json -> figures/numbers.{tex,json}, figures/tab_*.tex
  make_fig1.py          figures/fig1_decomposition_mechanism.pdf   (loss waterfall + step budget)
  make_fig_regime.py    figures/fig2_inversion_regime.pdf          (when the inversion pays)
  make_fig_mulan.py     figures/fig3_mulan_collapse.pdf            (11 MULAN datasets x 5 learners)
  make_fig2.py          figures/fig2_shift_vs_lnw.pdf              (supplement Fig. S1)
  figures/fig0_schematic.tex   data-free TikZ schematic (arXiv version only)
  verify_numbers.py     the number gate, run over every */main.tex and */supplement.tex
arxiv/    main.tex supplement.tex make_bundle.sh       (bundle: arxiv_upload.tar.gz, flat paths)
ecir/     main.tex supplement.tex derive_from_arxiv.py derive_supplement.py
full-paper/  main.tex supplement.tex
```

Each `main.tex` reads the shared inputs through two relative paths set in its preamble:
`\graphicspath{{../shared/}}` for figures and `\inputbody{figures/...}` for table bodies (a TeX
primitive `\input`, because the LaTeX `\input{...}` form runs file hooks that break inside `tabular`).
The bibliography is `../../tex/bibliography/references.bib` for all versions.

## Build

```bash
python publications/shared/verify_numbers.py          # regenerates numbers/tables, then checks every version
cd publications/shared && python make_fig1.py && python make_fig_regime.py && python make_fig_mulan.py && python make_fig2.py
cd ../arxiv   && pdflatex main && bibtex main && pdflatex main && pdflatex main    # same for supplement
cd ../ecir    && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../arxiv   && bash make_bundle.sh                    # arXiv upload tarball, compiled standalone as a check
```

The conference version was started from the arXiv text by `ecir/derive_from_arxiv.py` (class swap,
anonymisation, float layout) and then hand-trimmed to 12 pages; re-running the script discards the
hand edits, so use it only to restart. `ecir/derive_supplement.py` regenerates the anonymised
supplement from `arxiv/supplement.tex` (it appends a proofs section, since the conference main text keeps
its proofs in the supplement) and can be re-run at any time.

## Diffing versions

The three main files share section order, paragraph order, macro names and figure files, so a plain
text diff is readable:

```bash
git diff --no-index publications/arxiv/main.tex publications/ecir/main.tex
git diff --no-index publications/full-paper/main.tex publications/arxiv/main.tex
```

`latexdiff` (if installed) gives a marked-up PDF of the same comparison.

## Anonymity of the conference version

`ecir/main.tex` and `ecir/supplement.tex` carry no author block, no acknowledgments and no
non-anonymous URL; the placeholder `\anonurl` must be set to the issued anonymous-mirror address
before submission. Check before every upload:

```bash
grep -rniE "goto|shiga|souldrive7|0009-0004|anthropic|claude" publications/ecir/*.tex   # expect no output
```

The conference version does not cite the arXiv version (double-anonymous review).

The arXiv version's Acknowledgments carry a one-sentence disclosure of AI-tool use (coding, language
editing and manuscript review; all content verified by the author). Commit trailers that name an AI tool
record that use; they do not denote authorship. `derive_from_arxiv.py` drops the section for review; the
same sentence is restored verbatim in the camera-ready, which is what the publisher's policy asks for
(describe the use, confirm author accountability; an AI is never an author).
