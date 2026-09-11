"""Build and execute oddslip/tutorial.ipynb from tutorial.py (percent-format cells).

    python build_tutorial.py            # writes tutorial.ipynb with executed outputs
    python build_tutorial.py --no-exec  # convert only

The .py file is the source of truth (diff-friendly); the notebook is a rendered copy for GitHub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent


def parse_percent(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, list[str]]] = []
    kind = None
    for line in text.splitlines():
        if line.startswith("# %%"):
            kind = "markdown" if "[markdown]" in line else "code"
            cells.append((kind, []))
            continue
        if kind is None:
            continue
        cells[-1][1].append(line)
    out = []
    for kind, lines in cells:
        if kind == "markdown":
            body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines).strip("\n")
        else:
            body = "\n".join(lines).strip("\n")
        if body:
            out.append((kind, body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--arrays-dir", default=None, help="directory holding weighted/ and unweighted/ predictions.npz for the optional Santander cell (exported to the kernel as ODDSLIP_ARRAYS_DIR)")
    args = ap.parse_args()
    if args.arrays_dir:
        import os

        os.environ["ODDSLIP_ARRAYS_DIR"] = str(Path(args.arrays_dir).resolve())
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, body in parse_percent((HERE / "tutorial.py").read_text(encoding="utf-8")):
        nb.cells.append(nbformat.v4.new_markdown_cell(body) if kind == "markdown" else nbformat.v4.new_code_cell(body))
    if not args.no_exec:
        from nbclient import NotebookClient

        # run with the package directory's parent as cwd so that `import oddslip` and the array lookup work
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE.parent)}}).execute()
    out = HERE / "tutorial.ipynb"
    nbformat.write(nb, out)
    n_err = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
    print(f"wrote {out} ({len(nb.cells)} cells, {n_err} error outputs)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
