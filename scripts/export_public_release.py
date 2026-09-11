"""Validate the public tree and maintain its tracked-file SHA-256 manifest.

This repository is currently the source and release tree, so the exporter does not copy files.
It validates the checked-in tree, runs the release-safety scan, and writes or checks SHA256SUMS.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "SHA256SUMS.txt"


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    names = result.stdout.decode("utf-8").split("\0")
    return [
        ROOT / name
        for name in names
        if name and Path(name).as_posix() != "SHA256SUMS.txt" and (ROOT / name).is_file()
    ]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_text(files: list[Path]) -> str:
    rows = []
    for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    return "\n".join(rows) + "\n"


def run_safety_scan() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release_safety.py"), "--root", str(ROOT)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if SHA256SUMS.txt is stale")
    args = parser.parse_args()

    run_safety_scan()
    expected = manifest_text(release_files())
    current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
    if args.check:
        if current != expected:
            print("[export_public_release] FAIL: SHA256SUMS.txt is stale", file=sys.stderr)
            return 1
        print(f"[export_public_release] OK: {expected.count(chr(10))} manifest entries")
        return 0

    MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
    print(f"[export_public_release] wrote {expected.count(chr(10))} manifest entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())