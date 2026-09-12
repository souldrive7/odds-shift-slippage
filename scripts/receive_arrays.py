"""Verify transferred prediction arrays against the canonical hashes and place them.

Reads every ``_meta.inputs`` block in artifacts/results/canonical/*.json (the SHA-256 of each array the
paper's numbers were computed from), walks the transfer folder, and for every
``<dataset>/<config>/predictions.npz`` (or MLP ``p_*_all.npy``) found there:

  * recomputes the SHA-256 and compares it with the canonical one -> MATCH / MISMATCH / UNKNOWN;
  * on MATCH, copies the file (and run_meta.json) into the local arrays directory unless an identical
    file is already there.

Usage:  python scripts/receive_arrays.py [--transfer <transfer folder>] [--thesis <local arrays root>] [--dry-run]
Defaults come from the environment variables ODDSLIP_TRANSFER_DIR and ODDSLIP_ARRAYS_ROOT.
Exit code 1 if any MISMATCH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "artifacts" / "results" / "canonical"

# transfer subfolder -> (canonical key prefix, local destination)
LAYOUT = {
    "santander_outputs_matched_pair": ("santander", "santander_product_proxy/outputs_matched_pair"),
    "outputs_matched_pair": ("instacart", "instacart_product/outputs_matched_pair"),
    "santander_outputs_mlp": ("santander_mlp", "santander_product_proxy/outputs_mlp"),
    "instacart_outputs_mlp": ("instacart_mlp", "instacart_product/outputs_mlp"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hashes() -> dict[str, set[str]]:
    """key -> every SHA-256 pinned under that key. MLP ``.npy`` keys (``w1/p_te_all.npy``) carry no
    dataset prefix, so Santander and Instacart pin different hashes under the same key."""
    out: dict[str, set[str]] = {}
    for f in sorted(CANON.glob("*.json")):
        try:
            inputs = json.loads(f.read_text(encoding="utf-8")).get("_meta", {}).get("inputs", {})
        except Exception:
            continue
        for k, v in inputs.items():
            out.setdefault(k, set()).add(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", type=Path, default=os.environ.get("ODDSLIP_TRANSFER_DIR"), help="transfer folder (default: $ODDSLIP_TRANSFER_DIR)")
    ap.add_argument("--thesis", type=Path, default=os.environ.get("ODDSLIP_ARRAYS_ROOT"), help="local arrays root (default: $ODDSLIP_ARRAYS_ROOT)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.transfer is None or args.thesis is None:
        raise SystemExit("set --transfer/--thesis or the environment variables ODDSLIP_TRANSFER_DIR and ODDSLIP_ARRAYS_ROOT")
    args.transfer, args.thesis = Path(args.transfer), Path(args.thesis)
    print("transfer folder:", args.transfer)
    canon = canonical_hashes()
    bad = 0
    for sub, (prefix, dest_rel) in LAYOUT.items():
        base = args.transfer / sub
        if not base.exists():
            continue
        for cfg_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            cfg = cfg_dir.name
            for arr in sorted(list(cfg_dir.glob("*.npz")) + list(cfg_dir.glob("*.npy"))):
                key = f"{prefix}/{cfg}/{arr.name}" if arr.suffix == ".npz" else f"{cfg}/{arr.name}"
                have = sha256(arr)
                wants = canon.get(key)
                status = "UNKNOWN" if wants is None else ("MATCH" if have in wants else "MISMATCH")
                bad += status == "MISMATCH"
                shown = "" if wants is None else (have[:16] if have in wants else "/".join(w[:16] for w in sorted(wants)))
                print(f"{status:8s} {key:48s} {have[:16]} {shown}")
                if status == "MATCH" and not args.dry_run:
                    dest = args.thesis / dest_rel / cfg
                    dest.mkdir(parents=True, exist_ok=True)
                    for name in (arr.name, "run_meta.json"):
                        s, d = cfg_dir / name, dest / name
                        if s.exists() and not (d.exists() and sha256(d) == sha256(s)):
                            shutil.copy2(s, d)
                            print(f"         placed {d}")
    print("all transferred arrays match the canonical hashes" if bad == 0 else f"{bad} MISMATCH")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
