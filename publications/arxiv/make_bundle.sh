#!/usr/bin/env bash
# Build the arXiv upload bundle for the canonical (arXiv) version of the paper.
# Usage (Git Bash, from publications/arxiv):  bash make_bundle.sh
# Produces arxiv_upload.tar.gz containing main.tex, main.bbl, figures/*.pdf, figures/*.tex and the compiled
# supplement as the arXiv ancillary file anc/supplement.pdf (arXiv does not compile anc/).
# arXiv compiles with pdflatex + acmart from TeX Live; the .bbl is shipped so bibtex is not needed.
# The working-tree sources read their generated inputs from ../shared/figures/; the bundle is flat, so the
# bundled main.tex is rewritten to read figures/ next to it (and the shared bibliography as refs.bib).
set -euo pipefail
cd "$(dirname "$0")"
SHARED=../shared
"${PYTHON:-python}" "$SHARED/verify_numbers.py"
for doc in main supplement; do
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  bibtex "$doc" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  grep "Output written" "$doc.log"
done
rm -rf _bundle && mkdir -p _bundle/figures _bundle/anc
cp main.bbl _bundle/
cp ../../tex/bibliography/references.bib _bundle/refs.bib
# Flatten the shared paths for the self-contained bundle:
#   \input{../shared/figures/numbers.tex} -> \input{figures/numbers.tex}
#   \inputbody{figures/X.tex}             -> \input figures/X.tex   (TeX primitive form, safe inside tabular)
#   \graphicspath{{../shared/}}           -> \graphicspath{{./}}
#   \bibliography{../../tex/bibliography/references} -> \bibliography{refs}
sed -e '/^%.*shared\//d' \
    -e '/\\newcommand{\\inputbody}/d' \
    -e 's#\\input{\.\./shared/figures/numbers\.tex}#\\input{figures/numbers.tex}#' \
    -e 's#\\inputbody{\(figures/[A-Za-z0-9_]*\.tex\)}#\\input \1#g' \
    -e 's|\\inputbody{#3}|\\input #3|' \
    -e 's#\\graphicspath{{\.\./shared/}}#\\graphicspath{{./}}#' \
    -e 's#\\bibliography{\.\./\.\./tex/bibliography/references}#\\bibliography{refs}#' \
    main.tex > _bundle/main.tex
if grep -q 'shared/' _bundle/main.tex; then echo "ERROR: a ../shared/ path survived in _bundle/main.tex"; grep -n 'shared/' _bundle/main.tex; exit 1; fi
# Ship only what main.tex actually pulls in (figures and table bodies). The supplement travels pre-compiled
# as anc/supplement.pdf, so its figure and its table bodies are not needed. The standalone compile below is
# what guarantees this list is complete.
grep -o 'figures/[A-Za-z0-9_]*\.\(pdf\|tex\)' _bundle/main.tex | sort -u | while read -r f; do
  cp "$SHARED/$f" "_bundle/$f"
done
# TikZ schematic (if the version uses it) is a .tex under figures/ and is caught by the grep above.
echo "bundled inputs:"; ls _bundle/figures/
cp supplement.pdf _bundle/anc/supplement.pdf
# standalone compile test of the bundle
( cd _bundle && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && grep "Output written" main.log )
rm -f _bundle/main.aux _bundle/main.log _bundle/main.out _bundle/main.pdf
tar -czf arxiv_upload.tar.gz -C _bundle .
ls -la arxiv_upload.tar.gz
sha256sum arxiv_upload.tar.gz
