"""Dataset registry for the arXiv v3 rederivation: where each prediction array lives, which
configuration plays S (weighted), N (unweighted), the caps, the weight doses and the 90% train-row
draws, and how the per-label weights are recovered from ``run_meta.json``.

Every dataset exposes the same API, so the ``part_*`` functions in run_experiment.py are
dataset-generic and write the same JSON schema for Santander and Instacart, LightGBM and MLP.

Arrays are loaded on demand and never cached here (7 Instacart models x 5 GB would not fit
next to their float64 copies); callers hold one or two at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    """A ladder entry: ``name`` is the key used in the result JSON, ``config`` the directory
    under the arrays dir, ``role`` selects which parts use it."""

    name: str
    config: str
    role: str  # S | N | cap | dose | sub_S | sub_N
    cap: float | None = None
    note: str = ""


def _lgbm_specs(caps: tuple[float, ...] = (0.3, 0.7, 1.0, 2.0, 5.0), dose: bool = False) -> list[ModelSpec]:
    def cap_cfg(c: float) -> str:
        return f"weighted_mds{c:g}"

    specs = [
        ModelSpec("S_lgbm_spw", "weighted", "S", None, "LightGBM 60 trees, lr 0.05, 31 leaves, no subsampling, seed 42, scale_pos_weight=n_-/n_+"),
        ModelSpec("N_lgbm_unweighted", "unweighted", "N", None, "same as S_lgbm_spw without scale_pos_weight"),
    ]
    specs += [ModelSpec(f"S_{cap_cfg(c)}", cap_cfg(c), "cap", c, f"S with max_delta_step={c:g}") for c in caps]
    if dose:
        specs += [
            ModelSpec("S_weighted_sqrt", "weighted_sqrt", "dose", None, "S with scale_pos_weight = sqrt(n_-/n_+)"),
            ModelSpec("S_weighted_10r", "weighted_10r", "dose", None, "S with scale_pos_weight = 10 n_-/n_+"),
        ]
    specs += [ModelSpec(f"S_weighted_sub{s}", f"weighted_sub{s}", "sub_S", None, f"S on a 90% train-row draw, seed {s}") for s in range(3)]
    specs += [ModelSpec(f"N_unweighted_sub{s}", f"unweighted_sub{s}", "sub_N", None, f"N on a 90% train-row draw, seed {s}") for s in range(3)]
    return specs


MLP_SPECS = [
    ModelSpec("S_mlp_posweight", "wr", "S", None, "shared-trunk MLP, BCEWithLogits pos_weight=n_-/n_+ per label"),
    ModelSpec("N_mlp_unweighted", "w1", "N", None, "same MLP with pos_weight=1"),
]


class Dataset:
    def __init__(self, key: str, arrays_dir: Path, specs: list[ModelSpec], learner: str, labels_bundle: Path | None = None, all_labels_layout: bool = False) -> None:
        self.key = key
        self.arrays_dir = Path(arrays_dir)
        self.specs = specs
        self.learner = learner
        self.labels_bundle = labels_bundle
        self.all_labels_layout = all_labels_layout
        self._meta: dict[str, dict] = {}
        self._y: dict[str, np.ndarray] = {}

    # ---- files -------------------------------------------------------------------------
    def config_dir(self, cfg: str) -> Path:
        return self.arrays_dir / cfg

    def npz(self, cfg: str) -> Path:
        return self.config_dir(cfg) / "predictions.npz"

    def available(self, cfg: str) -> bool:
        return self.npz(cfg).exists() and (self.config_dir(cfg) / "run_meta.json").exists()

    def spec(self, name_or_cfg: str) -> ModelSpec:
        for s in self.specs:
            if name_or_cfg in (s.name, s.config):
                return s
        raise KeyError(name_or_cfg)

    def role(self, role: str) -> list[ModelSpec]:
        return [s for s in self.specs if s.role == role and self.available(s.config)]

    def ladder_models(self) -> list[ModelSpec]:
        """S, N, caps and doses that exist (the draws are summarised by part_seeds only)."""
        return [s for s in self.specs if s.role in ("S", "N", "cap", "dose") and self.available(s.config)]

    @property
    def S(self) -> ModelSpec:
        return self.spec_by_role("S")

    @property
    def N(self) -> ModelSpec:
        return self.spec_by_role("N")

    def spec_by_role(self, role: str) -> ModelSpec:
        for s in self.specs:
            if s.role == role:
                return s
        raise KeyError(role)

    # ---- metadata --------------------------------------------------------------------
    def meta(self, cfg: str) -> dict:
        if cfg not in self._meta:
            self._meta[cfg] = json.loads((self.config_dir(cfg) / "run_meta.json").read_text(encoding="utf-8"))
        return self._meta[cfg]

    @property
    def n_train(self) -> int:
        """Full training rows (the N configuration; draws record their own n_train)."""
        m = self.meta(self.N.config)
        return int(m.get("n_train_full", m["n_train"]))

    @property
    def n_pos(self) -> np.ndarray:
        return np.array([d["n_pos"] for d in self.meta(self.N.config)["per_label"]], dtype=np.float64)

    @property
    def pi_train(self) -> np.ndarray:
        return self.n_pos / self.n_train

    @property
    def w_true(self) -> np.ndarray:
        """n_-/n_+ on the full training rows: the recommended weight and the analytic inversion's w."""
        return (self.n_train - self.n_pos) / np.maximum(self.n_pos, 1)

    def spw(self, cfg: str) -> np.ndarray:
        """The weight actually used by configuration ``cfg`` (scale_pos_weight or pos_weight per label)."""
        pl = self.meta(cfg)["per_label"]
        key = "scale_pos_weight" if "scale_pos_weight" in pl[0] else "pos_weight"
        return np.array([d[key] for d in pl], dtype=np.float64)

    @property
    def label_names(self) -> list:
        return list(self.meta(self.N.config)["label_names"])

    @property
    def L(self) -> int:
        return len(self.label_names)

    def lgbm_params(self, cfg: str) -> dict:
        m = self.meta(cfg)
        return {k: m.get(k) for k in ("n_trees", "learning_rate", "num_leaves", "max_delta_step", "random_state", "train_subsample", "subsample_seed")}

    # ---- arrays ----------------------------------------------------------------------
    def y(self, split: str = "te") -> np.ndarray:
        """Labels of the test (te) or validation (va) rows, int8, from the N configuration; every
        other configuration and the label bundle (if present) must agree cell for cell."""
        if split not in self._y:
            with np.load(self.npz(self.N.config)) as z:
                self._y[split] = z[f"Y_{split}"].astype(np.int8)
            if self.labels_bundle is not None and self.labels_bundle.exists():
                self._assert_bundle(split)
        return self._y[split]

    def _assert_bundle(self, split: str) -> None:
        import scipy.sparse as sp

        with np.load(self.labels_bundle) as z:
            name = f"Y_{split}"
            m = sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=tuple(z[f"{name}_shape"]))
        cols = self.meta(self.N.config).get("label_cols")
        yb = np.asarray((m if cols is None else m[:, cols]).todense()).astype(np.int8)
        assert np.array_equal(yb, self._y[split]), f"{self.key}: Y_{split} in predictions.npz differs from {self.labels_bundle}"

    def scores(self, cfg: str, split: str = "te", check_y: bool = True) -> np.ndarray:
        with np.load(self.npz(cfg)) as z:
            q = z[f"p_{split}"].astype(np.float64)
            if check_y:
                assert np.array_equal(z[f"Y_{split}"].astype(np.int8), self.y(split)), f"{self.key}/{cfg}: Y_{split} differs from the N configuration"
        return q

    def check_n_pos(self, cfg: str) -> None:
        """w_j must be reproducible from the labels: run_meta n_pos equals the column sums of the
        training labels when the label bundle is present (LightGBM datasets)."""
        if self.labels_bundle is None or not self.labels_bundle.exists():
            return
        import scipy.sparse as sp

        m = self.meta(cfg)
        if m.get("train_subsample") is not None:
            return  # draws: n_pos is on the drawn rows, checked by the trainer itself
        with np.load(self.labels_bundle) as z:
            Y = sp.csr_matrix((z["Y_tr_data"], z["Y_tr_indices"], z["Y_tr_indptr"]), shape=tuple(z["Y_tr_shape"]))
        cols = m.get("label_cols")
        n_pos = np.asarray((Y if cols is None else Y[:, cols]).sum(0)).ravel().astype(np.int64)
        assert np.array_equal(n_pos, np.array([d["n_pos"] for d in m["per_label"]], dtype=np.int64)), f"{self.key}/{cfg}: run_meta n_pos differs from the label bundle"

    def inputs_meta(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for s in self.specs:
            if self.available(s.config):
                out[f"{self.key}/{s.config}/predictions.npz"] = sha256(self.npz(s.config))
        return out

    def all_labels_paths(self, cfg: str) -> dict[str, Path]:
        return {split: self.config_dir(cfg) / f"p_{split}_all.npy" for split in ("te", "va")}


def _santander_dir() -> Path:
    env = os.environ.get("DRC_ARRAYS_DIR")
    return Path(env) if env else EXP / "data" / "santander"


REGISTRY: dict[str, dict] = {
    "santander": {"arrays": lambda: _santander_dir() / "outputs_matched_pair", "specs": _lgbm_specs(dose=True), "learner": "lightgbm", "labels": None},
    "instacart": {"arrays": lambda: EXP / "data" / "instacart" / "outputs_matched_pair", "specs": _lgbm_specs(), "learner": "lightgbm", "labels": EXP / "data" / "instacart" / "data" / "processed" / "labels_topN.npz"},
    "santander_mlp": {"arrays": lambda: _santander_dir() / "outputs_mlp", "specs": MLP_SPECS, "learner": "mlp", "labels": None},
    "instacart_mlp": {"arrays": lambda: EXP / "data" / "instacart" / "outputs_mlp", "specs": MLP_SPECS, "learner": "mlp", "labels": None},
}


def get_dataset(key: str, arrays_dir: Path | None = None) -> Dataset:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key}; choose from {list(REGISTRY)}")
    r = REGISTRY[key]
    return Dataset(key, arrays_dir or r["arrays"](), r["specs"], r["learner"], r["labels"], all_labels_layout=key.endswith("_mlp"))
