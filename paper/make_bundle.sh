#!/usr/bin/env bash
# Build the arXiv upload bundle for manuscript/arxiv_v3.
# Usage (Git Bash, from manuscript/arxiv_v3):  bash make_bundle.sh
# Produces arxiv_v3_upload.tar.gz containing main.tex, main.bbl, figures/*.pdf, figures/*.tex and the
# compiled supplement as the arXiv ancillary file anc/supplement.pdf (arXiv does not compile anc/).
# arXiv compiles with pdflatex + acmart from TeX Live; the .bbl is shipped so bibtex is not needed.
set -euo pipefail
cd "$(dirname "$0")"
python verify_numbers.py
for doc in main supplement; do
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  bibtex "$doc" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  grep "Output written" "$doc.log"
done
rm -rf _bundle && mkdir -p _bundle/figures _bundle/anc
cp main.tex main.bbl _bundle/
# Ship only what main.tex actually pulls in. The supplement travels pre-compiled as anc/supplement.pdf,
# so its figure and its table bodies are not needed; shipping them risks arXiv picking up a stale file.
# The standalone compile below is what guarantees this list is complete.
grep -o 'figures/[A-Za-z0-9_]*\.\(pdf\|tex\)' main.tex | sort -u | while read -r f; do
  cp "$f" "_bundle/$f"
done
echo "bundled inputs:"; ls _bundle/figures/
cp supplement.pdf _bundle/anc/supplement.pdf
# standalone compile test of the bundle
( cd _bundle && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && grep "Output written" main.log )
rm -f _bundle/main.aux _bundle/main.log _bundle/main.out _bundle/main.pdf
tar -czf arxiv_v3_upload.tar.gz -C _bundle .
ls -la arxiv_v3_upload.tar.gz
sha256sum arxiv_v3_upload.tar.gz
