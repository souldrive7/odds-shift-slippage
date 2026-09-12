r"""Derive the ECIR (LNCS, double-anonymous) skeleton from the canonical arXiv source.

Usage:  python derive_from_arxiv.py            (from publications/ecir)
Writes  main.tex  next to this script, overwriting it.  Run it once to (re)start the ECIR version from the
current arXiv text; the ECIR main.tex is then hand-trimmed to the 12-page LNCS limit, so re-running this
script discards those hand edits (git diff shows them).

What it changes, mechanically:
  * document class acmart -> llncs (runningheads); acmart-only preamble dropped; llncs' own theorem
    environments are used (no \newtheorem);
  * author block -> Anonymous; acknowledgments removed; GitHub URL -> anonymous mirror placeholder;
    the two sentences that reveal earlier work by the same author are neutralised;
  * figure*/table* -> figure/table, \columnwidth -> \textwidth, \Description dropped, wide content wrapped
    in \resizebox; bibliography style -> splncs04; \Supp{...} -> "supplementary material (...)".
Every result number stays a macro from ../shared/figures/numbers.tex.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "arxiv" / "main.tex"
DST = HERE / "main.tex"
ANON_URL = "https://anonymous.4open.science/r/odds-shift-slippage-XXXX"  # TODO: set the issued mirror URL

s = SRC.read_text(encoding="utf-8")

# ---- preamble ---------------------------------------------------------------------------------------
pre_start = s.index("\\documentclass")
body_start = s.index("\\begin{document}")
preamble = r"""\documentclass[runningheads]{llncs}
% ECIR 2027 submission (double-anonymous). Derived from ../arxiv/main.tex by derive_from_arxiv.py, then
% hand-trimmed to the 12-page LNCS limit. Every number is a macro from ../shared/figures/numbers.tex.
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\graphicspath{{../shared/}}
\makeatletter\newcommand{\inputbody}[1]{\@@input ../shared/#1 }\makeatother
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xspace}
\usepackage{enumitem}
\usepackage{placeins}
\usepackage{float}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,patterns}
\usepackage[hidelinks]{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\definecolor{figblue}{HTML}{0072B2}
\definecolor{figorange}{HTML}{D55E00}
\definecolor{figgray}{HTML}{7F7F7F}
\input{../shared/figures/numbers.tex}
\crefname{proposition}{Proposition}{Propositions}
\Crefname{proposition}{Proposition}{Propositions}
\crefname{lemma}{Lemma}{Lemmas}
\Crefname{lemma}{Lemma}{Lemmas}
\crefname{corollary}{Corollary}{Corollaries}
\Crefname{corollary}{Corollary}{Corollaries}
\crefname{remark}{Remark}{Remarks}
\Crefname{remark}{Remark}{Remarks}
\newcommand{\Sm}{\textsf{LGBM-w}\xspace}
\newcommand{\Wm}{\textsf{WGB}\xspace}
\newcommand{\Nm}{\textsf{LGBM-0}\xspace}
\newcommand{\mapk}{\mathrm{MAP}@K\xspace}
\newcommand{\mapseven}{\mathrm{MAP}@7\xspace}
\newcommand{\Supp}[1]{supplementary material, #1}  % PDF in the anonymous repository
\setlength{\textfloatsep}{9pt plus 2pt minus 3pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}

"""
body = s[body_start:]

# ---- title block ------------------------------------------------------------------------------------
body = re.sub(
    r"\\title\{(.*?)\}\n\n\\author\{Akifumi Goto\}.*?\\begin\{abstract\}",
    lambda m: "\\title{" + m.group(1) + "}\n\\titlerunning{Odds-Shift Slippage in One-vs-Rest Rankers}\n"
    "\\author{Anonymous Author(s)}\n\\authorrunning{Anonymous}\n\\institute{Anonymous Institution}\n"
    "\\maketitle\n\n\\begin{abstract}",
    body,
    flags=re.S,
)
body = body.replace(
    "\\end{abstract}\n\n\\maketitle\n",
    "\\keywords{class imbalance \\and class weights \\and multi-label classification \\and top-$K$ recommendation "
    "\\and probability calibration \\and isotonic regression \\and gradient boosting}\n\\end{abstract}\n",
)

# ---- anonymisation ----------------------------------------------------------------------------------
body = re.sub(r"\\section\*\{Acknowledgments\}.*?(?=\\bibliographystyle)", "", body, flags=re.S)
body = body.replace("\\url{https://github.com/souldrive7/odds-shift-slippage}", "\\url{" + ANON_URL + "}")
body = body.replace(
    "The same wrapper produced a published negative result of ours; and since \\cref{sec:dose} shows",
    "The same wrapper has produced a published negative result; and since \\cref{sec:dose} shows",
)
body = body.replace("\\bibliographystyle{ACM-Reference-Format}", "\\bibliographystyle{splncs04}")

# ---- LNCS layout ------------------------------------------------------------------------------------
body = body.replace("\\begin{figure*}[t]", "\\begin{figure}[t]").replace("\\end{figure*}", "\\end{figure}")
body = body.replace("\\begin{table*}[t]", "\\begin{table}[t]").replace("\\end{table*}", "\\end{table}")
body = re.sub(r"\\Description\{.*?\}\n", "", body, flags=re.S)
body = body.replace("width=\\columnwidth", "width=.72\\textwidth")
body = body.replace("\\inputbody{figures/fig0_schematic.tex}",
                    "\\resizebox{\\textwidth}{!}{\\inputbody{figures/fig0_schematic.tex}}")
# wide tables: wrap the tabular in \resizebox
body = re.sub(r"(\\begin\{tabular\}\{@\{\}lcccccccc@\{\}\}.*?\\end\{tabular\})",
              r"\\resizebox{\\textwidth}{!}{%\n\1}", body, flags=re.S)
body = re.sub(r"(\\begin\{tabular\}\{@\{\}lrrrrrrrr\|cc\|ccccc\|cc@\{\}\}.*?\\end\{tabular\})",
              r"\\resizebox{\\textwidth}{!}{%\n\1}", body, flags=re.S)
# llncs: captions above tables are the convention; keep as is (caption first) -- acmart source already has it.
# acmart's \paragraph is a run-in heading; llncs has \paragraph too (run-in), fine.

for tok in ("souldrive7", "Goto", "Shiga", "0009-0004", "Anthropic", "Claude"):
    assert tok not in body, f"identifying token left in body: {tok}"
DST.write_text(preamble + body, encoding="utf-8", newline="\n")
print("wrote", DST)
