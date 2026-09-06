"""MULAN benchmark I/O: fetch the public ARFF/XML files by URL and parse them.

Self-contained (no dependency outside this directory). Every fetched file is pinned by its
SHA-256 in `_FETCH_PROV`, which the result JSON records as `fetch_provenance`.
The parser is the frozen one used for every MULAN number in the paper: dense or sparse ARFF,
nominal attributes mapped to integer codes, missing values to 0, features z-scored per column.
"""

from __future__ import annotations

import hashlib
import re
import ssl
import urllib.request

import numpy as np

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

BASE = "https://raw.githubusercontent.com/tsoumakas/mulan/master/data/multi-label/{}"

# SHA-256 + byte count of every fetched file, keyed by relative path. The remote
# ARFF/XML are pinned by content hash -- the analog of a local NPZ SHA-256.
_FETCH_PROV: dict[str, dict] = {}


def _fetch(path):
    raw = urllib.request.urlopen(BASE.format(path), timeout=90, context=_CTX).read()
    _FETCH_PROV[path] = {
        "url": BASE.format(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    return raw.decode("utf-8", "ignore")


def parse_labels_xml(text):
    return set(re.findall(r'<label\s+name="([^"]+)"', text))


def parse_mulan_arff(text, label_names):
    """Parse a MULAN ARFF (dense or sparse). Returns X (features), Y (binary labels)."""
    attrs = []  # (name, kind, nom_map)
    data_lines = []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            m = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.*)", s, re.IGNORECASE)
            name = m.group(1) or m.group(2) or m.group(3)
            spec = m.group(4).strip()
            if spec.startswith("{"):
                vals = [v.strip().strip("'\"") for v in spec.strip("{} ").split(",")]
                attrs.append((name, "nom", {v: i for i, v in enumerate(vals)}))
            else:
                attrs.append((name, "num", None))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            data_lines.append(s)

    n_attr = len(attrs)
    label_idx = [i for i, (nm, _, _) in enumerate(attrs) if nm in label_names]
    feat_idx = [i for i in range(n_attr) if i not in set(label_idx)]
    assert label_idx, "no label attributes matched the XML label set"

    n = len(data_lines)
    full = np.zeros((n, n_attr), dtype=float)
    for r, ln in enumerate(data_lines):
        ln = ln.strip()
        if ln.startswith("{"):  # sparse row: {idx val, idx val, ...}
            for tok in ln.strip("{} ").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                idx, val = tok.split(None, 1)
                idx = int(idx)
                kind, nom = attrs[idx][1], attrs[idx][2]
                full[r, idx] = nom.get(val.strip().strip("'\""), 0) if kind == "nom" else float(val)
        else:
            cells = [c.strip() for c in ln.split(",")]
            if len(cells) != n_attr:
                continue
            for i, c in enumerate(cells):
                kind, nom = attrs[i][1], attrs[i][2]
                if c in ("?", ""):
                    full[r, i] = 0.0
                elif kind == "nom":
                    full[r, i] = nom.get(c.strip("'\""), 0)
                else:
                    full[r, i] = float(c)
    Y = full[:, label_idx].astype(int)
    X = full[:, feat_idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, Y
