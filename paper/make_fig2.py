"""Fig. 2 -- per-label mechanism on Santander: the intended shift ln w_j, the shift the weighted
model actually realized (label-free prevalence-matching shift b_j, capped at the search bound),
and the share of test cells at exactly 1.0 per label. Drawn from results JSON only.

Reads santander_ladder.json: S_lgbm_spw.prior_match_shift_label_free.{shift_b, ln_w} and
S_lgbm_spw.saturation.frac_exact_1_per_label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = next(
    c
    for c in (
        HERE.parent / "results",
        HERE.parents[1] / "experiments" / "arxiv_v2_rederive" / "results",
    )
    if (c / "santander_ladder.json").exists()
)
ORANGE, BLUE, GRAY = "#D55E00", "#0072B2", "#7F7F7F"
CAP = 39.9  # bisection bound of the prior-matching shift (hi=40)

lad = json.load(open(RES / "santander_ladder.json", encoding="utf-8"))
S = lad["S_lgbm_spw"]
ln_w = np.array(S["prior_match_shift_label_free"]["ln_w"])
b = np.array(S["prior_match_shift_label_free"]["shift_b"])
sat = np.array(S["saturation"].get("frac_exact_1_per_label", [np.nan] * len(ln_w)))
capped = np.abs(b) >= CAP

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6}
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.1, 1.0]})

# ---- left: realized shift vs intended shift
lim = float(max(ln_w.max(), 1.0)) + 1.0
ax1.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls="--", label="$b_j=\\ln w_j$ (Prop. 1)")
ok = ~capped
ax1.scatter(ln_w[ok], b[ok], s=22, color=ORANGE, zorder=3, label="realized shift $b_j$")
if capped.any():
    ax1.scatter(
        ln_w[capped],
        np.full(capped.sum(), CAP + 0.1),
        s=30,
        marker="^",
        color=BLUE,
        zorder=3,
        label=f"unreachable (search bound), {int(capped.sum())} labels",
    )
ax1.set_xlabel("intended shift $\\ln w_j$ (nat)")
ax1.set_ylabel("prevalence-matching shift $b_j$ (nat)")
ax1.set_xlim(0, lim)
ax1.set_ylim(0, max(CAP + 2.0, lim))
ax1.legend(fontsize=6.5, frameon=False, loc="upper left")
ax1.set_title("A. Realized vs. intended shift per label", fontsize=8, loc="left")

# ---- right: saturation share vs ln w
order = np.argsort(ln_w)
ax2.scatter(ln_w[ok], 100 * sat[ok], s=22, color=ORANGE, zorder=3)
if capped.any():
    ax2.scatter(ln_w[capped], 100 * sat[capped], s=30, marker="^", color=BLUE, zorder=3)
ax2.set_xlabel("intended shift $\\ln w_j$ (nat)")
ax2.set_ylabel("test cells at exactly 1.0 (%)")
ax2.set_xlim(0, lim)
ax2.set_title("B. Saturation per label", fontsize=8, loc="left")

fig.tight_layout()
out = HERE / "figures" / "fig2_shift_vs_lnw.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", out, "| capped labels:", int(capped.sum()), "| order of ln w:", order.tolist())
