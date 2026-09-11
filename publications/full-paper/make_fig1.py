"""Fig. 1 (v3) -- the loss decomposition on two datasets and the mechanism, drawn from results JSON only.

Panel A (decomposition, one row group per dataset): MAP@7 of the unweighted model (N), the ideal
odds-shifted model (N logit + ln w, what Elkan's identity predicts for a weighted model that realizes
the shift), the weighted model actually trained with the identical configuration (S), its analytic
inversion with the known w_j, and its per-label isotonic repair with the prior fallback; the in-sample
per-label ceiling is a tick.
Panel B (mechanism, cap sweep): for each leaf-step cap c the largest realized per-label shift
max_j |b_j| (label-free prevalence matching) against the step budget T*eta*c of Proposition
`prop:budget`, plus the uncapped model, whose shift runs into the search bound; the share of test
cells at exactly 1.0 is written next to the points where it is not negligible.

Inputs: {santander,instacart}_{ladder,whatif,capsweep}.json. A dataset whose ladder or what-if file
is missing is skipped with a warning; a missing cap sweep leaves panel B without that dataset.
Palette: Okabe-Ito subset (CVD-safe), validated with the dataviz validator. Marker shape, not hue,
separates the datasets in panel B (the hue there is the weighted model's, as in panel A).
"""

from __future__ import annotations

import json
import sys
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
    if (c / "santander_ladder.json").exists()
)
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#7F7F7F"
SEARCH_BOUND = (
    39.9  # bisection bound of the label-free prevalence-matching shift (run_experiment.py)
)


def _load(name):
    p = RES / name
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


DATASETS = [  # key, panel label, marker for panel B
    ("santander", "Santander", "o"),
    ("instacart", "Instacart", "s"),
]
panels = []
for key, title, marker in DATASETS:
    lad, wi, cs = (
        _load(f"{key}_ladder.json"),
        _load(f"{key}_whatif.json"),
        _load(f"{key}_capsweep.json"),
    )
    if lad is None or wi is None:
        print(f"warning: {key}: ladder or what-if file missing; skipped", file=sys.stderr)
        continue
    if cs is None:
        print(f"warning: {key}: cap sweep missing; panel B drawn without it", file=sys.stderr)
    n_labels = len(lad["_meta"]["label_names"])
    panels.append((key, f"{title} ({n_labels:,} labels)", marker, lad, wi, cs))
assert panels, "santander_ladder.json and santander_whatif.json are required"

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6}
)
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.4, 1.6 + 1.1 * len(panels)), gridspec_kw={"width_ratios": [1.45, 1.0]}
)

# ---- A: decomposition, one row group per dataset
methods = [
    "unweighted (same config, $w{=}1$)",
    "ideal odds shift: $\\mathrm{logit}\\,q+\\ln w_j$",
    "weighted, as trained ($w_j{=}n_-/n_+$)",
    "weighted $+$ analytic inversion",
    "weighted $+$ per-label iso (prior)",
]
GROUP = len(methods) + 1.6
yticks, ylabels = [], []
for gi, (_key, title, _marker, lad, wi, _cs) in enumerate(panels):
    S, N = lad["S_lgbm_spw"], lad["N_lgbm_unweighted"]
    vals = [
        wi["N_raw"],
        wi["whatif_N_plus_a_lnw"]["1.0"],
        wi["S_actual"],
        S["elkan_inversion_known_w"]["map7_all"],
        S["per_label_iso_prior"]["map7_te"],
    ]
    ceil = S["oracle_in_sample_per_label_iso"]["map7_all"]
    n_cal = N["per_label_iso_prior"]["map7_te"]  # the unweighted model under the same calibrator
    base = gi * GROUP
    ys = [base + i for i in range(len(vals))]
    bars = ax1.barh(ys, vals, color=[BLUE, GRAY, ORANGE, ORANGE, ORANGE], height=0.6)
    bars[3].set_alpha(0.55)
    bars[4].set_hatch("////")
    bars[4].set_edgecolor("white")
    ax1.plot([ceil, ceil], [ys[4] - 0.42, ys[4] + 0.42], color=ORANGE, lw=1.2)
    # open marker: where the same per-label map takes the unweighted model (the like-for-like reference)
    ax1.scatter(
        [n_cal], [ys[0]], marker="o", s=22, facecolor="white", edgecolor=BLUE, lw=1.0, zorder=3
    )
    loss = N["raw"]["map7_te"] - S["raw"]["map7_te"]
    odds = wi["loss_share"]["odds_shift_part"]
    repair = (
        (S["per_label_iso_prior"]["map7_te"] - S["raw"]["map7_te"]) / loss
        if loss > 0
        else float("nan")
    )
    if repair <= 1.0:
        repair_note = f"  returns {100 * repair:.0f}%, ceiling {ceil:.3f}"
    else:  # the calibrator also lifts w=1: quote the ceiling, not a share above 100%
        repair_note = f"  ceiling {ceil:.3f}"
    notes = [
        f"  ($\\circ$ {n_cal:.3f})",
        "" if odds is None else f"  odds shift = {100 * odds:.0f}% of the loss",
        "",
        f"  returns {100 * (vals[3] - vals[2]) / loss:.0f}%" if loss > 0 else "",
        repair_note,
    ]
    for i, (y, v, note) in enumerate(zip(ys, vals, notes, strict=True)):
        x_text = max(v, ceil) if i == 4 else (max(v, n_cal) if i == 0 else v)
        ax1.text(x_text + 0.01, y, f"{v:.3f}{note}", va="center", fontsize=6.5)
    ax1.text(0.0, base - 0.95, title, fontsize=7.5, fontweight="bold", ha="left", va="center")
    yticks += ys
    ylabels += methods
ax1.set_yticks(yticks, ylabels, fontsize=6.5)
# the open marker has no legend entry of its own; say what it is inside the panel
ax1.text(
    1.60,
    -1.45,
    "$\\circ$ = the $w{=}1$ model under the same per-label calibrator",
    fontsize=6,
    ha="right",
    va="center",
    color=BLUE,
)
ax1.set_ylim(len(panels) * GROUP - 1.9, -1.6)
ax1.set_xlim(0, 1.62)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_xlabel("MAP@7")
ax1.set_title("A. Textbook prediction, trained model, repair", fontsize=8, loc="left")

# ---- B: mechanism, realized shift against the step budget
CAPS = [0.3, 0.7, 1.0, 2.0, 5.0]
xt = list(range(len(CAPS) + 1))
bound_drawn = False
ymax = 1.0
series = []
for _key, title, marker, _lad, _wi, cs in panels:
    if cs is None:
        continue
    pts = {}
    for r in cs["models"].values():
        cap = r.get("cap")
        pts[None if cap is None else float(cap)] = r
    xs, ys_, sats, at_bound, bounds = [], [], [], [], []
    for i, c in enumerate([*CAPS, None]):
        r = pts.get(c)
        if r is None:
            continue
        xs.append(i)
        ys_.append(r["b_prior"]["max_abs"])
        sats.append(r["saturation"]["frac_exact_1"])
        at_bound.append(r["b_prior"]["n_at_search_bound"] > 0)
        bounds.append(r.get("T_eta_c"))
    series.append((title, marker, xs, ys_, sats, at_bound))
    if not bound_drawn and any(b is not None for b in bounds):
        bx = [x for x, b in zip(xs, bounds, strict=True) if b is not None]
        by = [b for b in bounds if b is not None]
        ax2.plot(bx, by, color=GRAY, lw=1.0, ls="--", label="step budget $T\\eta c$ (Prop. 2)")
        ymax = max(ymax, max(by))
        bound_drawn = True
    ymax = max(
        ymax,
        max(y for y, ab in zip(ys_, at_bound, strict=True) if not ab)
        if any(not ab for ab in at_bound)
        else ymax,
    )
ytop = ymax * 1.18
for si, (title, marker, xs, ys_, sats, at_bound) in enumerate(series):
    draw_y = [min(y, ytop * (0.96 - 0.05 * si)) for y in ys_]
    ax2.plot(xs, draw_y, color=ORANGE, lw=0.8, alpha=0.5)
    ax2.scatter(
        [x for x, ab in zip(xs, at_bound, strict=True) if not ab],
        [y for y, ab in zip(draw_y, at_bound, strict=True) if not ab],
        marker=marker,
        s=30,
        color=ORANGE,
        zorder=3,
        label=title.split(" (")[0],
    )
    ax2.scatter(
        [x for x, ab in zip(xs, at_bound, strict=True) if ab],
        [y for y, ab in zip(draw_y, at_bound, strict=True) if ab],
        marker="^",
        s=34,
        color=ORANGE,
        zorder=3,
    )
    dx, ha = (
        (-0.12, "right") if si == 0 else (0.12, "left")
    )  # labels of the two datasets go to opposite sides
    for x, y, s, ab in zip(xs, draw_y, sats, at_bound, strict=True):
        if ab or s >= 0.005:
            ax2.text(
                x + dx, y, f"{100 * s:.1f}% at $1.0$", fontsize=6, ha=ha, va="center", color=ORANGE
            )
if any(ab for _t, _m, _x, _y, _s, at_bound in series for ab in at_bound):
    ax2.scatter(
        [], [], marker="^", s=34, color=ORANGE, label=f"at the search bound ($\\geq{SEARCH_BOUND}$)"
    )
ax2.set_xticks(xt, [f"{c:g}" for c in CAPS] + ["none"])
ax2.set_xlim(-0.4, len(CAPS) + 0.6)
ax2.set_ylim(0, ytop)
# matplotlib is not LaTeX: an escaped underscore outside mathtext prints the backslash.
ax2.set_xlabel("leaf-step cap $c$ (max_delta_step)")
ax2.set_ylabel("$\\max_j |b_j|$  (nat)")
ax2.set_title("B. Realized shift vs. step budget", fontsize=8, loc="left")
if series:
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(
            -0.02,
            0.86,
        ),  # below the saturation annotations, above the small-cap points
        fontsize=6.5,
        frameon=False,
        handletextpad=0.3,
    )
else:
    ax2.text(
        0.5,
        0.5,
        "cap sweep not available",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        color=GRAY,
    )

fig.tight_layout()
out = HERE / "figures" / "fig1_decomposition_mechanism.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", out)
