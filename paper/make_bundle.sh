#!/usr/bin/env bash
# Build the arXiv upload bundle for manuscript/arxiv_v2.
# Usage (Git Bash, from manuscript/arxiv_v2):  bash make_bundle.sh
# Produces arxiv_v2_upload.tar.gz containing main.tex, main.bbl, figures/*.pdf, figures/*.tex.
# arXiv compiles with pdflatex + acmart from TeX Live; the .bbl is shipped so bibtex is not needed.
set -euo pipefail
cd "$(dirname "$0")"
python verify_numbers.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
grep "Output written" main.log
rm -rf _bundle && mkdir -p _bundle/figures
cp main.tex main.bbl _bundle/
cp figures/*.pdf figures/numbers.tex figures/tab_*.tex _bundle/figures/
# standalone compile test of the bundle
( cd _bundle && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && grep "Output written" main.log )
rm -f _bundle/main.aux _bundle/main.log _bundle/main.out _bundle/main.pdf
tar -czf arxiv_v2_upload.tar.gz -C _bundle .
ls -la arxiv_v2_upload.tar.gz
sha256sum arxiv_v2_upload.tar.gz
