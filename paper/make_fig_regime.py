"""Fig. 2 -- when the analytic inversion works, drawn from results JSON only.

One point per trained weighted model (both datasets, both learners, every leaf-step cap).
x: how much of the intended shift the learner realized, median of b_j / ln w_j over the labels where
   b_j is identifiable -- no cell of that label is clipped. A clipped cell pins the bisection near
   its bound and reports a shift that belongs to the clip, not to the learner, so a model with too
   few identifiable labels is listed in the caption instead of plotted.
y: what the analytic inversion returns, as a share of the weighted model's top-K loss against its
   unweighted twin -- 100% would restore the unweighted ranking, negative means it makes things worse.
A filled marker means no cell saturates; an open one means some do (the share is written beside it).
The reading: the inversion pays only where the realized shift is close to the intended one AND nothing
saturates, which is exactly the hypothesis of Corollary `cor:exact`.

Inputs: {santander,instacart}_capsweep.json (the capped models and the uncapped booster) and
{santander,instacart}_mlp_ladder.json (the MLP pairs). Models whose weighted score already exceeds
their unweighted twin have no loss to return and are listed in the caption instead of plotted.
Palette: Okabe-Ito subset, marker shape separates the datasets, fill separates the saturation regime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = next(
    c
    for c in (
        HERE.parent / "results",
        HERE.parent / "results",
    )
    if (c / "santander_ladder.json").exists()
)
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#7F7F7F"
SEARCH_BOUND = 39.9  # bisection bound of the prevalence-matching shift (run_experiment.py)
SAT_EPS = 1e-4  # below this share of cells at exactly 1.0 we call the model unsaturated
MIN_IDENT = 3  # a median over fewer identifiable labels than this is not a measurement


def _load(name):
    p = RES / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


N_ROWS = {  # test rows per dataset, from the generated macros (never hand-typed)
    k: int(json.load(open(HERE / "figures" / "numbers.json", encoding="utf-8"))[m].replace(",", ""))
    for k, m in (("santander", "nTest"), ("instacart", "InTest"))
}


def shift_ratio(b_list, lnw_list, identifiable) -> tuple[float | None, int]:
    """Median b_j / ln w_j over the labels where b_j is IDENTIFIABLE.

    A label with a cell at exactly 1.0 has no model-determined b_j: the clipped cells contribute a
    constant whatever b is, so the bisection settles just under its bound and reports a shift of ~38
    nat that belongs to the clip, not to the learner. Filtering on |b| < SEARCH_BOUND alone keeps
    those, which is how this figure previously put the Santander MLP at 6.7x its intended shift.
    `identifiable[j]` must be False for any label with an extreme cell."""
    pairs = [
        (b, w)
        for b, w, ok in zip(b_list, lnw_list, identifiable, strict=True)
        if ok and b is not None and w not in (0, None) and abs(b) < SEARCH_BOUND
    ]
    if not pairs:
        return None, 0
    return float(np.median([b / w for b, w in pairs])), len(pairs)


points, dropped, unidentifiable = [], [], []
for key, label, marker in (("santander", "Santander", "o"), ("instacart", "Instacart", "s")):
    cs = _load(f"{key}_capsweep.json")
    if cs is not None:
        n_raw = cs["N_raw"]
        for r in cs["models"].values():
            # the cap sweep stores saturation only as a dataset-level fraction, but it stores the
            # number of cells strictly inside (1e-6, 1-1e-6) in BOTH models per label; a label whose
            # count is the full test set has no extreme cell, which is the condition we need
            ident = [n == N_ROWS[key] for n in r["b_direct"]["n_inner_cells_per_label"]]
            ratio, n_id = shift_ratio(r["b_prior"]["per_label"], r["ln_w_per_label"], ident)
            raw = r["map_raw"]["map7_te"]
            elkan = r["map_elkan_inversion_known_w"]["map7_te"]
            name = "no cap" if r.get("cap") is None else f"$c{{=}}{r['cap']:g}$"
            if ratio is None or n_id < MIN_IDENT:
                unidentifiable.append(f"{label} {name}")
                continue
            if n_raw <= raw:
                dropped.append(f"{label} {name}")
                continue
            points.append(
                dict(
                    label=label,
                    marker=marker,
                    name=name,
                    learner="LightGBM",
                    ratio=ratio,
                    share=100 * (elkan - raw) / (n_raw - raw),
                    sat=r["saturation"]["frac_exact_1"],
                )
            )
    lad = _load(f"{key}_mlp_ladder.json")
    if lad is not None:
        S, N = lad["S_mlp_posweight"], lad["N_mlp_unweighted"]
        pm = S["prior_match_shift_label_free"]
        ident = [s == 0 for s in S["saturation"]["frac_exact_1_per_label"]]
        ratio, n_id = shift_ratio(pm["shift_b"], pm["ln_w"], ident)
        raw, elkan, n_raw = (
            S["raw"]["map7_te"],
            S["elkan_inversion_known_w"]["map7_te"],
            N["raw"]["map7_te"],
        )
        if ratio is None or n_id < MIN_IDENT:
            unidentifiable.append(f"{label} MLP ({n_id} identifiable labels)")
        elif n_raw > raw:
            points.append(
                dict(
                    label=label,
                    marker=marker,
                    name="MLP",
                    learner="MLP",
                    ratio=ratio,
                    share=100 * (elkan - raw) / (n_raw - raw),
                    sat=S["saturation"]["frac_exact_1"],
                )
            )
if not points:
    print("warning: no cap sweep or MLP ladder available; figure not written", file=sys.stderr)
    raise SystemExit(0)

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6}
)
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# no model's median ratio reaches 1.1, so the axis stops just above 1
XMIN, XMAX = 0.07, 1.75
CLIP = 0.073  # a model that realized no shift at all would sit on the left edge, with an arrow
ax.axhspan(-110, 0, color=GRAY, alpha=0.08, lw=0)
ax.axhline(0, color=GRAY, lw=0.8)
ax.axvline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 3)))
ax.text(0.97, 148, "shift fully realized", fontsize=6.5, ha="right", va="top", color=BLUE)
ax.text(0.075, -66, "inversion makes it worse", fontsize=6.5, ha="left", va="center", color=GRAY)
ax.text(
    0.30,
    38,
    "the inversion pays only where the shift\nwas realized AND nothing saturated",
    fontsize=6,
    ha="center",
    va="bottom",
    color="#333333",
    linespacing=1.2,
)

# only the models that carry the message are labelled; the capped cluster is named once, above
LABELLED = {
    ("Santander", "$c{=}5$"): (-8, -3, "right"),
    ("Instacart", "MLP"): (-8, -3, "right"),
    ("Santander", "no cap"): (7, 4, "left"),
    ("Instacart", "no cap"): (-7, -4, "right"),
    ("Santander", "$c{=}0.3$"): (0, 9, "center"),
}
for p in points:
    saturated = p["sat"] >= SAT_EPS
    x = max(p["ratio"], CLIP)
    ax.scatter(
        x,
        p["share"],
        marker=p["marker"],
        s=44,
        facecolor="none" if saturated else ORANGE,
        edgecolor=ORANGE,
        linewidths=1.2,
        zorder=3,
    )
    if p["ratio"] < CLIP:
        ax.annotate(
            "",
            (XMIN, p["share"]),
            (x, p["share"]),
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8},
        )
    off = LABELLED.get((p["label"], p["name"]))
    if off is None:
        continue
    dx, dy, ha = off
    name = f"{p['label'][0]}: {p['name']}" if p["name"] == "MLP" else p["name"]
    note = name + (f"\n{100 * p['sat']:.0f}% at 1.0" if saturated else "")
    ax.annotate(
        note,
        (x, p["share"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va="bottom" if dy > 0 else "top",
        fontsize=6,
        color="#333333",
    )

ax.set_xscale("log")
ax.set_xlim(XMIN, XMAX)
ax.set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
ax.set_ylim(-110, 150)
ax.set_yticks([-100, -50, 0, 50, 100, 150])
ax.set_xlabel("realized $/$ intended shift  (median $b_j/\\ln w_j$)")
ax.set_ylabel("loss returned by the\nanalytic inversion (%)")
handles = [
    plt.Line2D([], [], ls="", marker="o", mfc=ORANGE, mec=ORANGE, ms=6, label="Santander"),
    plt.Line2D([], [], ls="", marker="s", mfc=ORANGE, mec=ORANGE, ms=6, label="Instacart"),
    plt.Line2D([], [], ls="", marker="o", mfc="none", mec=ORANGE, ms=6, label="cells saturate"),
]
ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.2)
fig.tight_layout()
out = HERE / "figures" / "fig2_inversion_regime.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", out, "| omitted (no loss to return):", ", ".join(dropped) or "none")
print("  omitted (shift not identifiable):", ", ".join(unidentifiable) or "none")
print("  plotted:", ", ".join(f"{p['label']} {p['name']} x={p['ratio']:.2f}" for p in points))
