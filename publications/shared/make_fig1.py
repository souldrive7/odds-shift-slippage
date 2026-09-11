"""Fig. 1 -- the loss decomposition on two datasets and the mechanism, drawn from results JSON only.

Panel A (decomposition as a waterfall, one row group per dataset). Reading top to bottom:
  unweighted model (w = 1)                       full bar, MAP@7 of the matched unweighted twin
  - ideal odds shift  (Prop. `prop:odds`)        floating bar from the what-if score up to the
                                                  unweighted score: what Elkan's identity predicts a
                                                  weighted model that realizes ln w_j exactly would lose
  - slippage                                     floating bar from the trained weighted score up to
                                                  the what-if score: the rest of the observed loss
  weighted model, as trained (w_j = n_-/n_+)     full bar, the score the weight actually produced
  + analytic inversion (known w_j)               floating bar from the weighted score up to the inverted
  + per-label isotonic, prior fallback           floating bar from the weighted score up to the repaired
                                                  score; tick = in-sample per-label ceiling; open blue
                                                  marker = the unweighted twin under the same calibrator
Every number printed in the panel is read from the JSON (shares are recomputed here from the same
fields make_tables.py uses); nothing is typed by hand. The odds-shift share is the share of the
*observed* loss the what-if rung accounts for, not a causal attribution (copilot-instructions rule 8).

Panel B (mechanism, cap sweep): for each leaf-step cap c the largest realized per-label shift
max_j |b_j| (label-free prevalence matching) against the step budget T*eta*c of Proposition
`prop:budget`, plus the uncapped model, whose shift runs into the search bound; the share of test
cells at exactly 1.0 is written next to the points where it is not negligible.

Inputs: {santander,instacart}_{ladder,whatif,capsweep}.json. A dataset whose ladder or what-if file
is missing is skipped with a warning; a missing cap sweep leaves panel B without that dataset.
Palette: Okabe-Ito blue (unweighted arm) / orange (weighted arm) with a neutral grey for the textbook
what-if reference; roles are also carried by position, hatch and direct labels, never by hue alone.
Marker shape, not hue, separates the datasets in panel B.
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
LIGHT_ORANGE = "#F0B27A"  # the inversion rung: the weighted arm's hue, lightened
INK = "#333333"
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
    1, 2, figsize=(7.4, 1.7 + 1.25 * len(panels)), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42}
)

# ---- A: decomposition as a waterfall, one row group per dataset
ROWS = [
    "unweighted, same config ($w{=}1$)",
    "$-$ ideal odds shift (Prop. 1)",
    "$-$ slippage (the rest of the loss)",
    "weighted, as trained ($w_j{=}n_-/n_+$)",
    "$+$ analytic inversion by $\\ln w_j$",
    "$+$ per-label isotonic, prior fallback",
]
GROUP = len(ROWS) + 1.7
BAR_H = 0.62
yticks, ylabels = [], []
x_right = 1.0
for gi, (_key, title, _marker, lad, wi, _cs) in enumerate(panels):
    S, N = lad["S_lgbm_spw"], lad["N_lgbm_unweighted"]
    n_raw = wi["N_raw"]
    whatif = wi["whatif_N_plus_a_lnw"]["1.0"]
    s_raw = wi["S_actual"]
    inv = S["elkan_inversion_known_w"]["map7_all"]
    rep = S["per_label_iso_prior"]["map7_te"]
    ceil = S["oracle_in_sample_per_label_iso"]["map7_all"]
    n_cal = N["per_label_iso_prior"]["map7_te"]  # the unweighted model under the same calibrator
    loss = n_raw - s_raw
    odds = wi["loss_share"]["odds_shift_part"]
    slip = 1.0 - odds
    base = gi * GROUP
    ys = [base + i for i in range(len(ROWS))]

    def bar(y, lo, hi, color, hatch=None, alpha=1.0, z=2):
        b = ax1.barh(y, hi - lo, left=lo, height=BAR_H, color=color, alpha=alpha, zorder=z)
        if hatch:
            b[0].set_hatch(hatch)
            b[0].set_edgecolor("white")
        return b

    # connectors between the levels of the waterfall (recessive)
    for lvl, y0, y1 in ((n_raw, ys[0], ys[1]), (whatif, ys[1], ys[2]), (s_raw, ys[2], ys[3])):
        ax1.plot([lvl, lvl], [y0, y1], color=GRAY, lw=0.6, ls=":", zorder=1)
    bar(ys[0], 0, n_raw, BLUE)
    bar(ys[1], whatif, n_raw, GRAY)
    bar(ys[2], s_raw, whatif, ORANGE, alpha=0.45)
    bar(ys[3], 0, s_raw, ORANGE)
    bar(ys[4], s_raw, inv, LIGHT_ORANGE)
    bar(ys[5], s_raw, rep, ORANGE, hatch="////")
    ax1.plot([ceil, ceil], [ys[5] - 0.45, ys[5] + 0.45], color=ORANGE, lw=1.3, zorder=4)
    # open marker: where the same per-label map takes the unweighted model (like-for-like reference)
    ax1.scatter(
        [n_cal], [ys[0]], marker="o", s=24, facecolor="white", edgecolor=BLUE, lw=1.1, zorder=5
    )
    # direct labels: value, then what it means as a share of the observed loss
    inv_share = (inv - s_raw) / loss if loss > 0 else float("nan")
    rep_share = (rep - s_raw) / loss if loss > 0 else float("nan")
    rep_note = (
        f"{rep:.3f}  returns {100 * rep_share:.0f}%, ceiling {ceil:.3f}"
        if rep_share <= 1.0
        else f"{rep:.3f}  above $w{{=}}1$ raw, ceiling {ceil:.3f}"
    )
    labels = [
        (max(n_raw, n_cal), f"{n_raw:.3f}   ($\\circ$ {n_cal:.3f} same calibrator)"),
        (n_raw, f"$-${n_raw - whatif:.3f}  = {100 * odds:.0f}% of the loss"),
        (whatif, f"$-${whatif - s_raw:.3f}  = {100 * slip:.0f}% of the loss"),
        (s_raw, f"{s_raw:.3f}"),
        (max(inv, s_raw), f"{inv:.3f}  returns {100 * inv_share:.0f}%"),
        (max(rep, ceil), rep_note),
    ]
    for y, (x, text) in zip(ys, labels, strict=True):
        ax1.text(x + 0.012, y, text, va="center", fontsize=6.3, color=INK)
    ax1.text(0.0, base - 1.0, title, fontsize=7.5, fontweight="bold", ha="left", va="center")
    yticks += ys
    ylabels += ROWS
ax1.set_yticks(yticks, ylabels, fontsize=6.5)
ax1.tick_params(axis="y", length=0)
ax1.set_ylim(len(panels) * GROUP - 2.0, -1.7)
ax1.set_xlim(0, 1.62)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.grid(axis="x", color="#E3E3E3", lw=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.set_xlabel("MAP@7")
ax1.set_title("A. Loss decomposition and repair", fontsize=8, loc="left")

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
                x + dx, y, f"{100 * s:.1f}% at $1.0$", fontsize=6, ha=ha, va="center", color=INK
            )
if any(ab for _t, _m, _x, _y, _s, at_bound in series for ab in at_bound):
    ax2.scatter(
        [], [], marker="^", s=34, color=ORANGE, label=f"at the search bound ($\\geq{SEARCH_BOUND}$)"
    )
ax2.axvspan(len(CAPS) - 0.5, len(CAPS) + 0.6, color=GRAY, alpha=0.08, lw=0, zorder=0)
ax2.text(len(CAPS) + 0.55, ytop * 0.02, "no cap:\nsaturation", fontsize=6, ha="right", va="bottom", color=GRAY)
ax2.set_xticks(xt, [f"{c:g}" for c in CAPS] + ["none"])
ax2.set_xlim(-0.4, len(CAPS) + 0.6)
ax2.set_ylim(0, ytop)
# matplotlib is not LaTeX: an escaped underscore outside mathtext prints the backslash.
ax2.set_xlabel("leaf-step cap $c$ (max_delta_step)")
ax2.set_ylabel(r"largest realized shift $\max_j |b_j|$ (nat)", fontsize=7)
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
