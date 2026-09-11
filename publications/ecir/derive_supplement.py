r"""Derive the anonymous ECIR supplement from the arXiv supplement.

Usage:  python derive_supplement.py        (from publications/ecir)
Writes  supplement.tex next to this script (overwriting it). The ECIR supplement is the arXiv supplement
with the author block anonymised, a proofs section (S8) appended -- the ECIR main text keeps its proofs
in the supplement -- and no non-anonymous URL. Re-run after editing ../arxiv/supplement.tex.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "arxiv" / "supplement.tex"
DST = HERE / "supplement.tex"

s = SRC.read_text(encoding="utf-8")
ANON_BLOCK = "\\author{Anonymous Author(s)}\n\\affiliation{\\institution{Anonymous Institution}\\country{}}\n"
s, n = re.subn(r"\\author\{Akifumi Goto\}.*?\\country\{Japan\}\n\}\n", lambda m: ANON_BLOCK, s, flags=re.S)
assert n == 1, "author block not found"
s = s.replace("\\pdfinfo{ /Author (Akifumi Goto)", "\\pdfinfo{ /Author (Anonymous)")
s = re.sub(r"\\url\{https://github\.com/[^}]*\}", "the anonymous repository", s)

PROOFS = r"""
\section{Proofs}
\label{app:proofs}
\paragraph{Shared and distinct separable maps (Remark~1).} Let $g_1,\dots,g_L$ be non-decreasing maps on a common interval $I$. (a) If $g_1=\dots=g_L$, the separable calibrator never creates a strict inversion: $s_{ij}>s_{ik}$ implies $\tilde s_{ij}\ge\tilde s_{ik}$, with equality only when both scores fall in the same plateau of the shared map; this is monotonicity. (b) If the maps are continuous and $g_j(a)<g_k(a)$ for some interior $a\in I$, continuity of $g_j$ gives $b>a$ in $I$ with $g_j(b)<g_k(a)$, and the row with $s_{ij}=b>a=s_{ik}$ has $\tilde s_{ij}<\tilde s_{ik}$. If no such $b$ exists because $a$ is the right endpoint, use continuity of $g_k$ to pick $b<a$ with $g_k(b)>g_j(a)$ and the row $s_{ik}=b<a=s_{ij}$.

\paragraph{Proposition~1 (odds shift).} The weighted loss is the unweighted loss under the tilted distribution whose posterior odds are $w_j$ times the original; the flip condition is the difference of the shifted logits.

\paragraph{Lemma~1 (leaf saturation).} $-wn_+\ln\rho-n_-\ln(1-\rho)$ is minimized at $\rho=wn_+/(wn_++n_-)$, and $1-\rho=n_-/(wn_++n_-)\le n_-/(wn_+)$.

\paragraph{Proposition~2 (step budget).} Each round adds at most $\eta c$ in absolute value, so the first two bounds telescope. The mean of $\sigma(F^{w}_j(x_i)-b)$ over $i$ is decreasing in $b$; at $b=T\eta c$ every argument is at most $\operatorname{logit}\hat\pi_j$, so the mean is at most $\hat\pi_j$, and at $b=-T\eta c$ it is at least $\hat\pi_j$; the solution therefore lies in $[-T\eta c,T\eta c]$. The last claim compares the realized shift with the shift of Proposition~1.

\paragraph{Corollary~2 (when the inversion is exact).} The inverted score of label $j$ at $x$ is $F^{0}_j(x)+\delta_j(x)$. If $\delta_j(x)$ does not depend on $j$ it shifts every label of the row equally and leaves the within-row order of $F^{0}$ unchanged; $\delta\equiv0$ is the special case. Under a cap, write $D_j(x)=F^{w}_j(x)-F^{0}_j(x)$, so $|D_j|\le 2T\eta c$ by Proposition~2: pointwise equality is $D_j(x)=\ln w_j$, giving $\ln w_j\le 2T\eta c$; a row-constant residual gives $\ln w_j-\ln w_k=D_j(x)-D_k(x)$, which bounds the spread $\max_j\ln w_j-\min_j\ln w_j$ by $4T\eta c$ and leaves the magnitude free, which is why a common weight (Corollary~1(a)) escapes the bound entirely.

"""
marker = "\\bibliographystyle{ACM-Reference-Format}"
assert s.count(marker) == 1
s = s.replace(marker, PROOFS + marker)
for tok in ("Goto", "Shiga", "souldrive7", "0009-0004", "Anthropic"):
    assert tok not in s, f"identifying token left: {tok}"
DST.write_text(s, encoding="utf-8", newline="\n")
print("wrote", DST)
