"""多ラベル分類モデル群。"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _sanitize_feature_names(X: Any) -> Any:
    """LightGBM が拒否する JSON 特殊文字を含む列名を安全名に置換．

    実 Kaggle Santander のカテゴリ値（例: '02 - PARTICULARES'）が
    OneHotEncoder で `segmento_02 - PARTICULARES` 等を生成し，LightGBM が
    "Do not support special JSON characters in feature name" を投げる．
    安全策として全列を `f_NNNN` に統一する（順序保存）．
    """
    try:
        import pandas as pd  # local

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = [f"f_{i:04d}" for i in range(X.shape[1])]
            return X
    except ImportError:
        pass
    return X


try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # type: ignore


WeightMode = Literal["none", "balanced", "scale_pos_weight"]


def build_preprocess(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="NA")),
            (
                "oh",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50),
            ),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    ct = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])
    # LGBM が fit 時に feature names を記録するため、pandas を通し predict でも名前付きに揃える
    ct.set_output(transform="pandas")
    # 実データのカテゴリ値由来の特殊文字（カンマ・括弧・改行等）を回避するため
    # ColumnTransformer の出力を Pipeline で安全名にリネーム．
    sanitizer = FunctionTransformer(_sanitize_feature_names, validate=False)
    sanitizer.set_output(transform="pandas")
    return Pipeline([("ct", ct), ("sanitize", sanitizer)])


def make_lr_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = build_preprocess(cat_cols, num_cols)
    mo = MultiOutputClassifier(
        LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")
    )
    return Pipeline([("prep", pre), ("clf", mo)])


def make_lgbm_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    *,
    n_estimators: int = 80,
    learning_rate: float = 0.05,
    weight_mode: WeightMode = "balanced",
) -> Pipeline:
    """OvR multi-label LGBM 構築（``weight_mode`` で不均衡対処を切り替え）．

    - "none"        : 重みなし。確率は歪まないが少数派の AUC が低下しがち．
                      **較正の上限を測る参照系**として使う．
    - "balanced"    : sklearn の class_weight='balanced'（既存挙動）．
                      確率を歪めるため事後校正が必要．
    """
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not installed")
    pre = build_preprocess(cat_cols, num_cols)
    cw: Any = "balanced" if weight_mode == "balanced" else None
    mo = MultiOutputClassifier(
        LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            class_weight=cw,
            verbose=-1,
            force_row_wise=True,
        )
    )
    return Pipeline([("prep", pre), ("clf", mo)])


class PerLabelScalePosLGBM:
    """ラベル毎に ``scale_pos_weight=neg/pos`` で LGBM を訓練するシンプルラッパ．

    MultiOutputClassifier は 1 つの推定器を全ラベルに使い回すので，
    ラベル毎に異なる重みを与えるためには自前ループが必要．
    """

    def __init__(
        self,
        *,
        n_estimators: int = 80,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        weighted: bool = True,
        max_delta_step: float = 0.0,
        weight_power: float = 1.0,
        weight_multiplier: float = 1.0,
    ) -> None:
        """``weighted=False`` trains the identical configuration without ``scale_pos_weight``
        (the matched unweighted control); ``max_delta_step`` caps the leaf step per round
        (LightGBM default 0.0 = unbounded). ``weight_power`` / ``weight_multiplier`` set
        ``scale_pos_weight = multiplier * (n_neg/n_pos) ** power`` (dose-response: 0.5 -> sqrt(r),
        multiplier 10 -> 10r). Defaults reproduce the original behaviour.
        """
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm not installed")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.random_state = int(random_state)
        self.weighted = bool(weighted)
        self.max_delta_step = float(max_delta_step)
        self.weight_power = float(weight_power)
        self.weight_multiplier = float(weight_multiplier)
        self.models_: list[Any] = []
        self.n_labels_: int = 0
        self.scale_pos_: list[float] = []

    def fit(self, X: Any, Y: np.ndarray) -> PerLabelScalePosLGBM:
        assert LGBMClassifier is not None
        Y = np.asarray(Y).astype(int)
        n, L = Y.shape
        self.n_labels_ = L
        self.models_ = []
        self.scale_pos_ = []
        for j in range(L):
            yj = Y[:, j]
            n_pos = int(yj.sum())
            n_neg = int(n - n_pos)
            spw = float(self.weight_multiplier * (n_neg / max(n_pos, 1)) ** self.weight_power) if self.weighted else 1.0
            extra: dict[str, Any] = {}
            if self.weighted:
                extra["scale_pos_weight"] = spw
            if self.max_delta_step > 0.0:
                extra["max_delta_step"] = self.max_delta_step
            clf = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbose=-1,
                force_row_wise=True,
                **extra,
            )
            clf.fit(X, yj)
            self.models_.append(clf)
            self.scale_pos_.append(spw)
        return self

    def predict_proba_matrix(self, X: Any) -> np.ndarray:
        """(n, L) の陽性確率を返す（MultiOutputClassifier と異なり list ではなく行列）．"""
        ps = []
        for clf in self.models_:
            p = clf.predict_proba(X)
            ps.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
        return np.stack(ps, axis=1)


def make_lgbm_per_label_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    *,
    n_estimators: int = 80,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """ラベル毎 ``scale_pos_weight`` を使う OvR-LGBM Pipeline．

    最終 step ``clf`` は ``PerLabelScalePosLGBM`` を持ち，
    ``pipe.named_steps['clf'].predict_proba_matrix(...)`` で (n, L) を取得．
    """
    pre = build_preprocess(cat_cols, num_cols)
    est = PerLabelScalePosLGBM(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("clf", est)])


def make_mlp_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    *,
    hidden: tuple[int, ...] = (128, 64),
) -> Pipeline:
    pre = build_preprocess(cat_cols, num_cols)
    mo = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
    )
    return Pipeline([("prep", pre), ("clf", mo)])


def split_cat_num(feature_names: list[str], prod_cols: list[str]) -> tuple[list[str], list[str]]:
    cat = [
        f
        for f in feature_names
        if f not in prod_cols
        and f not in ("age", "antiguedad", "ind_actividad_cliente", "renta", "cod_prov")
    ]
    num = [
        f
        for f in feature_names
        if f in ("age", "antiguedad", "ind_actividad_cliente", "renta", "cod_prov")
    ]
    num = num + [f for f in prod_cols if f in feature_names]
    cat = [f for f in cat if f in feature_names]
    return cat, num
