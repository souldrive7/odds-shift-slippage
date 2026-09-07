"""Fig. 1 -- the loss decomposition and the repair ladder, drawn from results JSON only.

Left panel: MAP@7 of the unweighted model (N), the ideal odds-shifted model (N logit + ln w,
what Elkan's identity predicts), and the actually trained weighted model (S).
Right panel: the repair ladder for S, W, N (raw, per-label isotonic with identity / prior
fallback, pooled isotonic, popularity, oracle ceiling).

Palette: Okabe-Ito subset (CVD-safe), validated with the dataviz validator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = next(
    c
    for c in (
        HERE.parent / "results",
        HERE.parents[1] / "experiments" / "arxiv_v2_rederive" / "results",
    )
    if (c / "santander_ladder.json").exists()
)
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#7F7F7F"

lad = json.load(open(RES / "santander_ladder.json", encoding="utf-8"))
wi = json.load(open(RES / "santander_whatif.json", encoding="utf-8"))

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6}
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.0, 1.5]})

# ---- left: decomposition
labels = [
    "unweighted\n(LGBM-0)",
    "ideal odds shift\n$\\mathrm{logit}\\,q + \\ln w$",
    "weighted, as trained\n(LGBM-w)",
    "LGBM-w, analytic\ninversion with $w$",
    "LGBM-w, per-label iso\n(prior fallback)",
]
S0 = lad["S_lgbm_spw"]
vals = [
    wi["N_raw"],
    wi["whatif_N_plus_a_lnw"]["1.0"],
    wi["S_actual"],
    S0["elkan_inversion_known_w"]["map7_all"],
    S0["per_label_iso_prior"]["map7_te"],
]
cols = [BLUE, GRAY, ORANGE, ORANGE, ORANGE]
bars = ax1.barh(range(5), vals, color=cols, height=0.55)
bars[3].set_alpha(0.55)
bars[4].set_hatch("////")
bars[4].set_edgecolor("white")
ax1.set_yticks(range(5), labels)
ax1.set_xlim(0, 0.9)
ax1.set_xlabel("MAP@7 (Santander, 2.5M rows)")
for i, v in enumerate(vals):
    ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
share = wi["loss_share"]
S_ = lad["S_lgbm_spw"]
N_raw = lad["N_lgbm_unweighted"]["raw"]["map7_te"]
loss = N_raw - S_["raw"]["map7_te"]
repair = (S_["per_label_iso_prior"]["map7_te"] - S_["raw"]["map7_te"]) / loss
ties = (N_raw - S_["oracle_in_sample_per_label_iso"]["map7_all"]) / loss
ax1.text(
    0.02,
    1.47,
    f"odds shift: {100 * share['odds_shift_part']:.0f}% of the loss",
    fontsize=7,
    color=GRAY,
    ha="left",
    va="center",
)
ax1.text(
    0.36,
    3.0,
    f"recovers {100 * (vals[3] - vals[2]) / loss:.0f}%",
    fontsize=7,
    color=ORANGE,
    ha="left",
    va="center",
)
ax1.text(
    0.02,
    4.62,
    f"recovers {100 * repair:.0f}%\n({100 * ties:.0f}% of the loss lies above the ceiling)",
    fontsize=7,
    color=ORANGE,
    ha="left",
    va="center",
)
ax1.set_ylim(5.15, -0.5)
ax1.set_title("A. Textbook prediction vs. reality", fontsize=8, loc="left")

# ---- right: repair ladder
models = [
    ("S_lgbm_spw", "LGBM-w\n(weighted)", ORANGE),
    ("W_wgboost", "WGB (Wasserstein GB,\ndirect prob.)", GREEN),
    ("N_lgbm_unweighted", "LGBM-0\n(unweighted, same config)", BLUE),
]
methods = [
    ("raw", lambda r: r["raw"]["map7_te"], "o"),
    ("per-label iso, identity", lambda r: r["per_label_iso_identity"]["map7_te"], "x"),
    ("per-label iso, prior", lambda r: r["per_label_iso_prior"]["map7_te"], "s"),
    ("pooled iso", lambda r: r["pooled_iso"]["map7_te"], "^"),
    ("oracle ceiling", lambda r: r["oracle_in_sample_per_label_iso"]["map7_all"], "|"),
]
pop = lad["popularity_train_prevalence"]["map7_te"]
XLO, XHI = 0.40, 0.85  # zoom on the decisive range; values below XLO are drawn at the edge
ax2.axvline(pop, color=GRAY, lw=0.8, ls=":", label="popularity (train prevalence)")
for yi, (key, _name, col) in enumerate(models):
    r = lad[key]
    xs = [f(r) for _, f, _ in methods]
    xs_draw = [max(x, XLO + 0.005) for x in xs]
    ax2.plot([min(xs_draw), max(xs_draw)], [yi, yi], color=col, lw=1.0, alpha=0.5)
    n_off = 0
    for (mname, _f, mk), x, xd in zip(methods, xs, xs_draw, strict=True):
        ax2.scatter(
            [xd],
            [yi],
            marker=mk,
            s=28 if mk != "|" else 60,
            color=col,
            zorder=3,
            label=mname if yi == 0 else None,
        )
        if x < XLO:  # off-axis value: annotate with an arrow-like marker and the number
            ax2.text(
                xd + 0.008,
                yi + 0.26 + 0.2 * n_off,
                f"◂ {mname} {x:.3f}",
                color=col,
                fontsize=6,
                va="center",
            )
            n_off += 1
ax2.set_yticks(range(3), [m[1] for m in models])
ax2.set_ylim(2.5, -0.8)
ax2.set_xlim(XLO, XHI)
ax2.set_xlabel("MAP@7 on the 70% evaluation split (axis starts at 0.40)")
ax2.set_title("B. Repair ladder (dead-label policy decides the sign)", fontsize=8, loc="left")
ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    fontsize=6.5,
    frameon=False,
    ncol=3,
    handletextpad=0.3,
    columnspacing=1.0,
)
fig.tight_layout()
out = HERE / "figures" / "fig1_decomposition_ladder.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", out)
