"""Release-safety scanner for the public repository.

Identity-agnostic: this file contains no personal path or institution as a literal. It detects
the *categories* of information that should not leak into a release:

  - any e-mail address
  - any personal user-profile path (a Windows Users profile dir, or a POSIX home / WSL mount)
  - any academic / institutional domain (*.ac.<cc>, *.edu)
  - any ORCID iD
  - any GitHub / GitLab user or org URL

ALLOWLIST exempts the author's own public identifiers (the paper is not anonymous) and the
third-party hosts the paper cites.

Usage:
    python scripts/check_release_safety.py             # scan repo root
    python scripts/check_release_safety.py --root .    # explicit root
    python scripts/check_release_safety.py --verbose   # print every match

Exit code 0 on clean, 1 on any non-allowlisted match.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("windows_user", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.IGNORECASE)),
    ("posix_home", re.compile(r"/(?:home|mnt/c/Users)/[A-Za-z][\w.-]*")),
    ("academic_domain", re.compile(r"\b[\w.-]+\.(?:ac\.[a-z]{2}|edu)\b", re.IGNORECASE)),
    ("orcid", re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")),
    ("code_host_user", re.compile(r"\b(?:github|gitlab)\.com/[\w.-]+", re.IGNORECASE)),
]

ALLOWLIST: tuple[str, ...] = (
    # author's own public identifiers (intentional; the preprint is not anonymous)
    "souldrive7@gmail.com",
    "github.com/souldrive7",
    # third-party public references cited by the paper or used by the code
    "raw.githubusercontent.com/tsoumakas",  # MULAN benchmark mirror
    "github.com/tsoumakas",  # MULAN upstream
    "github.com/takuomatsubara",  # WGBoost reference implementation
    "github.com/microsoft",  # LightGBM
    "github.com/dmlc",  # XGBoost
    "kaggle.com",  # Santander benchmark host
    "download.pytorch.org",
)

TEXT_EXTS = {
    ".py",
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".bib",
    ".tex",
    ".sh",
    ".ps1",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".ini",
    ".cff",
    ".sha256",
    ".log",
    ".aux",
    ".out",
    ".bbl",
    ".blg",
}
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".mypy_cache"}


def _allowlisted(matched_text: str) -> bool:
    low = matched_text.lower()
    return any(a in low for a in ALLOWLIST)


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[tuple[str, int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for name, pat in PATTERNS:
            for m in pat.finditer(line):
                hit = m.group(0)
                if _allowlisted(hit):
                    continue
                findings.append((name, i, hit if len(hit) <= 200 else hit[:200] + "..."))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Release-safety / PII scanner.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.stderr.write(f"[check_release_safety] root not found: {root}\n")
        return 1
    n_scanned = n_dirty = total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        n_scanned += 1
        findings = scan_file(path)
        if findings:
            n_dirty += 1
            total += len(findings)
            sys.stderr.write(
                f"[check_release_safety] {path.relative_to(root)}: {len(findings)} match(es)\n"
            )
            if args.verbose:
                for name, lineno, snippet in findings:
                    sys.stderr.write(f"  L{lineno}  [{name}]  {snippet}\n")
    if n_dirty:
        sys.stderr.write(
            f"[check_release_safety] FAIL: {total} match(es) in {n_dirty} of {n_scanned} text file(s).\n"
        )
        return 1
    sys.stdout.write(f"[check_release_safety] OK: 0 matches across {n_scanned} text file(s).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
