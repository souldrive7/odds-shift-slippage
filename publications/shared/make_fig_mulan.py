"""Fig. 3 -- the MULAN dose-response at a glance, drawn from mulan_dose_response.json only.

Small multiples, one panel per learner, one row per dataset (sorted by label count L). In every row:
  * filled blue circle   MAP@K of the unweighted model (w = 1), raw scores
  * filled orange circle MAP@K of the weighted model (w_j = n_-/n_+), raw scores
  * open orange circle   the weighted model after per-label isotonic regression with the prior fallback
  * grey tick            the popularity baseline of the dataset (rank every row by training prevalence)
A thin line joins w=1 to w=r, so its length is what the recommended weight costs (or buys) before any
repair. A row is shaded where the weighted model loses more than half of its unweighted MAP@K -- the
"collapse" definition of the paper (make_tables.py, `nnCollapseCells`: map_raw(w=r) < 0.5 * map_raw(w=1)),
so the shaded cells are exactly the cells the abstract counts.

K = min(7, L) per dataset (the dataset's `k` field). The datasets with L <= 7 rank everything and
select nothing; they anchor the low-dose end of the sweep and are kept for that reason.

Palette: Okabe-Ito blue/orange for the two arms (same roles as in Fig. 1), grey for the reference
baseline; identity is also carried by fill and by the legend, never by hue alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = next(
    c
    for c in (
        ROOT / "artifacts" / "results" / "canonical",
        ROOT / "results",
    )
    if (c / "mulan_dose_response.json").exists()
)
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#7F7F7F"
INK = "#333333"
COLLAPSE_RATIO = 0.5  # same threshold as make_tables.py (nnCollapseCells)

LEARNERS = [  # key in the result file, panel title (matches the paper's naming)
    ("lgbm_default", "LightGBM (default)"),
    ("lgbm_strong", "LightGBM (regularized)"),
    ("xgb", "XGBoost"),
    ("logreg", "Logistic regression"),
    ("mlp_posweight", "MLP (pos_weight)"),
]

d = json.load(open(RES / "mulan_dose_response.json", encoding="utf-8"))
datasets = sorted(d["datasets"].items(), key=lambda kv: (kv[1]["L"], kv[0]))
learners = [(k, t) for k, t in LEARNERS if k in d["learners"]]
assert learners, "no known learner in mulan_dose_response.json"

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6}
)
n_rows = len(datasets)
fig, axes = plt.subplots(
    1, len(learners), figsize=(7.4, 0.62 + 0.26 * n_rows), sharey=True, sharex=True
)
axes = list(axes) if len(learners) > 1 else [axes]

ylabels = [f"{name}  ($L{{=}}{v['L']}$)" for name, v in datasets]
ys = list(range(n_rows))[::-1]  # smallest L on top
n_collapse = 0
for ax, (lk, title) in zip(axes, learners, strict=True):
    for y, (name, v) in zip(ys, datasets, strict=True):
        a = v["cells"].get(f"{lk}|w=1")
        b = v["cells"].get(f"{lk}|w=r")
        if a is None or b is None:
            ax.text(0.5, y, "n/a", fontsize=6, color=GRAY, ha="center", va="center")
            continue
        raw1, rawr, rep = a["map_raw"], b["map_raw"], b["map_per_label_iso_prior"]
        collapsed = rawr < COLLAPSE_RATIO * raw1
        n_collapse += collapsed
        if collapsed:
            ax.axhspan(y - 0.5, y + 0.5, color=ORANGE, alpha=0.10, lw=0, zorder=0)
        pop = v["popularity_map"]
        ax.plot([pop, pop], [y - 0.32, y + 0.32], color=GRAY, lw=1.0, zorder=1)
        ax.plot([raw1, rawr], [y, y], color=GRAY, lw=0.8, zorder=2)
        ax.scatter([raw1], [y], s=18, color=BLUE, zorder=4, linewidths=0)
        ax.scatter([rawr], [y], s=18, color=ORANGE, zorder=4, linewidths=0)
        ax.scatter(
            [rep], [y], s=22, facecolor="white", edgecolor=ORANGE, linewidths=1.0, zorder=3
        )
    ax.set_title(title, fontsize=7, loc="left", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_yticks(ys, ylabels, fontsize=6.5)
axes[0].set_ylim(-0.7, n_rows - 0.3)
for ax in axes:
    ax.tick_params(axis="y", length=0)
fig.supxlabel("MAP@K on the standard test split,  $K=\\min(7,L)$", fontsize=7.5, y=0.135)

handles = [
    plt.Line2D([], [], ls="", marker="o", mfc=BLUE, mec=BLUE, ms=5, label="unweighted ($w{=}1$), raw"),
    plt.Line2D([], [], ls="", marker="o", mfc=ORANGE, mec=ORANGE, ms=5, label="weighted ($w_j{=}n_-/n_+$), raw"),
    plt.Line2D(
        [], [], ls="", marker="o", mfc="white", mec=ORANGE, ms=5.5,
        label="weighted $+$ per-label isotonic (prior fallback)",
    ),
    plt.Line2D([], [], ls="-", color=GRAY, lw=1.0, marker="|", ms=7, label="popularity baseline"),
    plt.matplotlib.patches.Patch(
        facecolor=ORANGE, alpha=0.10, lw=0,
        label=f"collapse: weighted raw below half of unweighted ({n_collapse} cells)",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=6.5,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.4,
)
fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.24, wspace=0.14)
out = HERE / "figures" / "fig3_mulan_collapse.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", out, f"| collapse cells shaded: {n_collapse} of {n_rows * len(learners)}")
