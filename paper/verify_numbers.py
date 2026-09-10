"""Number gate for arxiv_v3/main.tex.

Rule: the prose never contains a hand-typed result number. Every result number is a macro
(\\nXxx from figures/numbers.tex) or lives in a generated table body (figures/tab_*.tex).

Checks:
 1. figures/numbers.tex is regenerated from the JSON results and identical to what make_tables.py emits now.
 2. every \\n<Name> macro used in main.tex, supplement.tex and poster/ibis2026_poster.tex is defined
    (full-line comments are skipped, so "% TODO(v3)" lines may reserve macros that do not exist yet).
 3. main.tex prose (outside \\input'd tables and outside the bibliography) contains no decimal number
    of the form d.ddd, no percentage like dd.d\\%, and no integer with thousands separators.
    Allowed exceptions are listed in ALLOW (theory constants such as 10^{-3}).
Exit code 1 on any failure; prints every offending line with its line number.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
NUMS = HERE / "figures" / "numbers.tex"
ALLOW = {
    "10^{-3}",
    "10^{-9}",
    "0.3",
    "1.0",
    "0.0",
    "0.5",
    "0.01",
    "0.12",
    "1/\\pi",
    "4 leaves",
    "30 trees",
}
# design constants that are not results: split fractions, CI level, hyper-parameter sets
DESIGN = {
    "30\\%",
    "70\\%",
    "95\\%",
    "100\\%",  # Fig. 2's axis definition: fully restoring the unweighted ranking
    "50,200",
    "20,50,200",
    "5,20,50,200",
    "0.03",
    "0.05",
    "0.001",
    "0.005",
    # Instacart design constants (the dataset README and its EDA result file): labelled
    # users = rows per split, top-N products of the matched pair, all products, feature columns
    "131,209",
    "4,000",
    "49,688",
    "8,320",
    "90\\%",  # train-row draws (the "seed" substitute of the matched pairs)
}

fail = 0

# 1. regenerate and compare
before = NUMS.read_text(encoding="utf-8") if NUMS.exists() else ""
subprocess.run([sys.executable, str(HERE / "make_tables.py")], check=True, cwd=HERE)
after = NUMS.read_text(encoding="utf-8")
if before != after:
    print("[FAIL] figures/numbers.tex was stale; regenerated (rebuild the PDF)")
    fail += 1

defined = set(re.findall(r"\\newcommand\{\\(n[A-Za-z0-9]+)\}", after))
SOURCES = [
    p for p in (TEX, HERE / "supplement.tex", HERE / "poster" / "ibis2026_poster.tex") if p.exists()
]  # the supplement and the IBIS poster share numbers.tex
# Comment lines are excluded from the macro check: "% TODO(v3)" lines reserve sentences whose macros
# do not exist until the corresponding result file is generated (they must stay commented until then).
tex = "\n".join(
    line
    for p in SOURCES
    for line in p.read_text(encoding="utf-8").splitlines()
    if not line.lstrip().startswith("%")
)
LATEX_N = {
    "newcommand",
    "noindent",
    "nosep",
    "newtheorem",
    "newpage",
    "nonumber",
    "notag",
    "nabla",
    "ne",
    "neq",
    "not",
    "nu",
    "null",
    "normalsize",
    "newline",
    "name",
    "nocite",
    "number",
    "numberwithin",
    "noalign",
    "nolinkurl",
}
used = {u for u in re.findall(r"\\(n[A-Za-z0-9]+)", tex) if u not in LATEX_N}
missing = sorted(u for u in used if u not in defined)
if missing:
    print("[FAIL] undefined macros:", missing)
    fail += 1

# 3. hand-typed numbers in prose
in_bib = False
pat = re.compile(r"(?<![A-Za-z\\{])(\d{1,3}(?:,\d{3})+|\d+\.\d{2,}|\d+\.\d\\%|\d+\\%)")
for src in SOURCES:
    for ln, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("\\bibliography") or line.startswith("%"):
            continue
        if "\\input{figures/" in line or "\\input figures/" in line:
            continue
        for m in pat.finditer(line):
            tok = m.group(0)
            if tok in DESIGN or any(tok in d for d in DESIGN):
                continue
            if any(a in line for a in ALLOW) and tok in "".join(ALLOW):
                continue
            # allow numbers inside \label/\ref/\cite keys and years
            if re.search(r"\\(cite|label|cref|Cref|ref)\{[^}]*" + re.escape(tok), line):
                continue
            print(f"[MISS] {src.name} line {ln}: hand-typed number '{tok}': {line.strip()[:110]}")
            fail += 1

print("OK: no hand-typed result numbers; all macros defined" if fail == 0 else f"{fail} problem(s)")
sys.exit(1 if fail else 0)
