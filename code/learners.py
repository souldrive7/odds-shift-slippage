"""One-vs-rest learners with a per-label positive-class weight, shared by mulan_dose and synthetic_check.

Every learner returns test probabilities of shape (n_te, L). Labels with a single class in
training get the constant training mean (no fallback ambiguity: this is the "prior" policy).
"""

from __future__ import annotations

import numpy as np

LGBM_DEFAULT = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20)
LGBM_STRONG = dict(n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=50, reg_lambda=1.0)


def _single_class(yj):
    return yj.sum() == 0 or yj.sum() == len(yj)


def lgbm_ovr(x_tr, y_tr, x_list, w, params=None, seed=42):
    from lightgbm import LGBMClassifier

    params = params or LGBM_DEFAULT
    L = y_tr.shape[1]
    outs = [np.zeros((len(x), L)) for x in x_list]
    for j in range(L):
        yj = y_tr[:, j]
        if _single_class(yj):
            for o in outs:
                o[:, j] = yj.mean()
            continue
        clf = LGBMClassifier(random_state=seed, verbosity=-1, scale_pos_weight=float(w[j]), **params)
        clf.fit(x_tr, yj)
        for o, x in zip(outs, x_list, strict=True):
            o[:, j] = clf.predict_proba(x)[:, 1]
    return outs


def xgb_ovr(x_tr, y_tr, x_list, w, seed=42):
    from xgboost import XGBClassifier

    L = y_tr.shape[1]
    outs = [np.zeros((len(x), L)) for x in x_list]
    for j in range(L):
        yj = y_tr[:, j]
        if _single_class(yj):
            for o in outs:
                o[:, j] = yj.mean()
            continue
        clf = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=1,
            scale_pos_weight=float(w[j]), random_state=seed, verbosity=0, n_jobs=4, tree_method="hist",
        )
        clf.fit(x_tr, yj)
        for o, x in zip(outs, x_list, strict=True):
            o[:, j] = clf.predict_proba(x)[:, 1]
    return outs


def logreg_ovr(x_tr, y_tr, x_list, w, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x_tr)
    xs_tr = sc.transform(x_tr)
    xs_list = [sc.transform(x) for x in x_list]
    L = y_tr.shape[1]
    outs = [np.zeros((len(x), L)) for x in x_list]
    for j in range(L):
        yj = y_tr[:, j]
        if _single_class(yj):
            for o in outs:
                o[:, j] = yj.mean()
            continue
        sw = np.where(yj == 1, w[j], 1.0)
        clf = LogisticRegression(C=C, max_iter=2000).fit(xs_tr, yj, sample_weight=sw)
        for o, x in zip(outs, xs_list, strict=True):
            o[:, j] = clf.predict_proba(x)[:, 1]
    return outs


def mlp_posweight(x_tr, y_tr, x_list, w, hidden=64, epochs=200, lr=1e-3, seed=42):
    """Shared-trunk MLP, BCE-with-logits with per-label pos_weight = w_j (torch)."""
    import torch

    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x_tr)
    X = torch.tensor(sc.transform(x_tr), dtype=torch.float32)
    Y = torch.tensor(y_tr, dtype=torch.float32)
    L = y_tr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, L))
    pw = torch.tensor(np.asarray(w, dtype=np.float32))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    bs = min(256, n)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(net(X[idx]), Y[idx]).backward()
            opt.step()
    net.eval()
    outs = []
    with torch.no_grad():
        for x in x_list:
            outs.append(torch.sigmoid(net(torch.tensor(sc.transform(x), dtype=torch.float32))).numpy().astype(np.float64))
    # constant-training-mean for single-class labels (the network cannot learn them meaningfully)
    for j in range(L):
        if _single_class(y_tr[:, j]):
            for o in outs:
                o[:, j] = y_tr[:, j].mean()
    return outs


LEARNERS = {
    "lgbm_default": lambda a, b, c, w: lgbm_ovr(a, b, c, w, LGBM_DEFAULT),
    "lgbm_strong": lambda a, b, c, w: lgbm_ovr(a, b, c, w, LGBM_STRONG),
    "xgb": xgb_ovr,
    "logreg": logreg_ovr,
    "mlp_posweight": mlp_posweight,
}
