"""Collect the canonical prediction arrays for a Zenodo dataset record.

Every array the paper's numbers were computed from is pinned by SHA-256 in the ``_meta.inputs`` block of
artifacts/results/canonical/*.json. This script walks those (key, hash) pairs, finds each file under the
local arrays directory (the layout that ``receive_arrays.py`` places into), re-hashes it, and copies the
matching ones into an upload directory laid out as ``<dataset>/<config>/<file>`` together with the
run_meta.json that sits beside each array and an ``ARRAYS_SHA256SUMS.txt`` manifest. Nothing is
uploaded; the Zenodo record is created by hand from the upload directory.

Keys come in three shapes: ``<dataset>/<config>/predictions.npz`` (LightGBM and MLP matched pairs, located
by path), ``<config>/p_*_all.npy`` (all-product MLP arrays, dataset-ambiguous, located by hash among the
files of that name), and bare file names such as ``uq_arrays.npz`` (thesis scorers, located by hash under
the whole arrays root). Inputs that are themselves result files (``*.json``) are skipped.

Usage:  python scripts/pack_arrays_for_zenodo.py [--thesis <local arrays root>] [--out <upload dir>] [--dry-run]
Defaults: --thesis from $ODDSLIP_ARRAYS_ROOT; --out from $ODDSLIP_ZENODO_DIR.
Exit code 1 if any pinned array is missing or does not match its hash.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from receive_arrays import CANON, LAYOUT, sha256  # noqa: E402

# canonical key prefix -> local directory (relative to --thesis), from receive_arrays.LAYOUT
PREFIX_DIR = {prefix: dest for prefix, dest in LAYOUT.values()}
PREFIX_DIR["matched_pair"] = "santander_product_proxy/outputs_matched_pair"  # santander_tie/seeds spelling
# bare keys pinned by the Santander ladder (code/run_experiment.py: uq_p, lg_p)
BARE = {
    "uq_arrays.npz": "santander_product_proxy/outputs_uq_regen/uq_arrays.npz",
    "predictions.npz": "santander_product_proxy/outputs_lgbm_baseline/predictions.npz",
}


def canonical_pairs() -> set[tuple[str, str]]:
    """Every (key, sha256) pair pinned by any canonical result file; result-file inputs excluded."""
    out: set[tuple[str, str]] = set()
    for f in sorted(CANON.glob("*.json")):
        try:
            inputs = json.loads(f.read_text(encoding="utf-8")).get("_meta", {}).get("inputs", {})
        except Exception:
            continue
        for k, v in inputs.items():
            if not k.endswith(".json"):
                out.add((k, v))
    return out


class Locator:
    def __init__(self, thesis: Path) -> None:
        self.thesis = thesis
        self._hash_cache: dict[Path, str] = {}

    def _sha(self, p: Path) -> str:
        if p not in self._hash_cache:
            self._hash_cache[p] = sha256(p)
        return self._hash_cache[p]

    def find(self, key: str, want: str) -> tuple[Path | None, str | None]:
        """Return (path, sha) — the path is the file that carries the pinned hash if one exists, else the
        path the key points at (so a MISMATCH can be reported), else None."""
        parts = key.split("/")
        if len(parts) == 3 and parts[0] in PREFIX_DIR:
            p = self.thesis / PREFIX_DIR[parts[0]] / parts[1] / parts[2]
            return (p, self._sha(p)) if p.exists() else (None, None)
        if key in BARE:
            p = self.thesis / BARE[key]
            return (p, self._sha(p)) if p.exists() else (None, None)
        # ambiguous or bare key: every file of that name under the arrays root, matched by hash
        name = Path(key).name
        candidates = [p for p in self.thesis.rglob(name) if p.is_file() and (len(parts) == 1 or p.parent.name == parts[-2])]
        for p in candidates:
            if self._sha(p) == want:
                return p, want
        return (candidates[0], self._sha(candidates[0])) if candidates else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", type=Path, default=os.environ.get("ODDSLIP_ARRAYS_ROOT"), help="local arrays root (default: $ODDSLIP_ARRAYS_ROOT)")
    ap.add_argument("--out", type=Path, default=os.environ.get("ODDSLIP_ZENODO_DIR"), help="upload directory (default: $ODDSLIP_ZENODO_DIR)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.thesis is None or (args.out is None and not args.dry_run):
        raise SystemExit("set --thesis/--out or the environment variables ODDSLIP_ARRAYS_ROOT and ODDSLIP_ZENODO_DIR")
    args.thesis = Path(args.thesis)
    if args.out is not None:
        args.out = Path(args.out)
    loc = Locator(args.thesis)
    rows: list[tuple[str, str]] = []        # (sha256, upload path) — one line per copied file
    aliases: list[tuple[str, str]] = []     # (other key, upload path) — same bytes pinned under another key
    placed: dict[str, str] = {}             # sha256 -> upload path
    missing, mismatch, total_bytes = [], [], 0
    # most specific keys first (dataset/config/file before config/file before bare file names)
    for key, want in sorted(canonical_pairs(), key=lambda kv: (-kv[0].count("/"), kv[0])):
        src, have = loc.find(key, want)
        if src is None:
            missing.append(key)
            print(f"MISSING  {key}")
            continue
        if have != want:
            mismatch.append(key)
            print(f"MISMATCH {key:56s} {have[:16]} != {want[:16]}  ({src})")
            continue
        if want in placed:
            aliases.append((key, placed[want]))
            print(f"ALIAS    {key:56s} = {placed[want]}")
            continue
        placed[want] = key
        total_bytes += src.stat().st_size
        rows.append((want, key))
        print(f"MATCH    {key:56s} {src.stat().st_size / 1e9:6.2f} GB")
        if not args.dry_run:
            dest = args.out / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not (dest.exists() and sha256(dest) == want):
                shutil.copy2(src, dest)
            meta = src.parent / "run_meta.json"
            if meta.exists():
                shutil.copy2(meta, dest.parent / "run_meta.json")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ARRAYS_SHA256SUMS.txt").write_text(
            "".join(f"{h}  {k}\n" for h, k in sorted(rows, key=lambda r: r[1]))
            + "".join(f"# alias: {k} -> {p}\n" for k, p in sorted(aliases)),
            encoding="utf-8", newline="\n",
        )
    print(f"{len(rows)} distinct arrays matched ({len(aliases)} aliases), {total_bytes / 1e9:.2f} GB; {len(missing)} missing, {len(mismatch)} mismatch")
    raise SystemExit(1 if (missing or mismatch) else 0)


if __name__ == "__main__":
    main()
