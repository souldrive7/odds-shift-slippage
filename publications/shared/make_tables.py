"""Generate every number the manuscript cites from the released result JSON files.

Outputs (all under figures/):
  numbers.tex   \\newcommand macros (\\nXxx) used in prose -- numbers are never hand-typed
  numbers.json  flat key -> value map (for verify_numbers.py and for the reader)
  tab_*.tex     table bodies included with \\input

Run:  python make_tables.py   (from the paper directory)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


BS = chr(92)
# learner keys as they are printed in the paper; used by several sections
LNAME = {
    "lgbm_default": "LightGBM",
    "lgbm_strong": "LightGBM (regularized)",
    "xgb": "XGBoost",
    "logreg": "logistic regression",
    "mlp_posweight": "MLP",
}


def _results_dir() -> Path:
    """Locate canonical results, with the legacy directory as a read-only fallback."""
    for cand in (
        ROOT / "artifacts" / "results" / "canonical",
        ROOT / "results",
    ):
        if (cand / "santander_ladder.json").exists():
            return cand
    raise FileNotFoundError("results directory with santander_ladder.json not found")


RES = _results_dir()
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

NUM: dict[str, float | int | str] = {}


def load(name):
    return json.load(open(RES / name, encoding="utf-8"))


def _why(e):
    """Name what is missing. repr(FileNotFoundError) hides the filename, which is the whole point."""
    if isinstance(e, FileNotFoundError):
        name = getattr(e, "filename", None) or e
        return f"no result file {name}"
    return f"missing key {e} in a result file"


# A TeX control sequence is letters only, so a macro name may not carry a digit.
# Dataset names do (corel5k), and the failure is a "Missing \begin{document}" at the
# \newcommand line, which names neither the digit nor the dataset.
_DIGIT_WORD = str.maketrans(
    dict(
        zip(
            "0123456789",
            ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"],
            strict=True,
        )
    )
)


def tex_name(key: str) -> str:
    """Macro-safe form of a key: digits spelled out, so corel5k -> Corelfivek."""
    return key.translate(_DIGIT_WORD)


def put(key, val, fmt=None):
    key = tex_name(key)
    NUM[key] = val if fmt is None else fmt
    return NUM[key]


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:+.1f}"


def write(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _try_load(name):
    try:
        return load(name)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- v3 helpers: second dataset, second learner
# Macro families (TeX stops a control sequence at the first digit, so no digits in names):
#   Santander LightGBM  S / N            (v2 names, emitted by the block below, unchanged)
#   Instacart LightGBM  IS / IN          dataset-level scalars  I...
#   Santander MLP       MS / MN          dataset-level scalars  M...
#   Instacart MLP       IMS / IMN        dataset-level scalars  IM...   (top-N columns)
#   Instacart MLP, all  IMAS / IMAN      dataset-level scalars  IMA...  (chunked.py, 49,688 products)
#   cap variants        {tag_s}mds{PointThree,PointSeven,One,Two,Five}...  (Santander keeps Smdsa / Smdsb too)
# The Santander block (S / W / N / Nh) is left as it was so that the v2 macros stay byte-identical.
CAP_TAG = {0.3: "PointThree", 0.7: "PointSeven", 1.0: "One", 2.0: "Two", 5.0: "Five"}
SEARCH_BOUND = (
    39.9  # bisection bound of the label-free prevalence-matching shift (run_experiment.py)
)
MIN_IDENT_LABELS = 3  # below this a median over identifiable labels is not worth quoting
# run_experiment.py clips probabilities to [EPS, 1-EPS] with EPS = 1e-15 before taking a logit, so a
# clipped cell sits at +-ln((1-EPS)/EPS) nat. The paper states both, because the shift estimator's
# failure mode is exactly that these constants, not the learner, set b_j on a saturated label.
SHIFT_EPS_LOGIT = 34.5


def emit_ladder_generic(prefix, tag_s, tag_n, lad, s_key, n_key, whatif=None, bootstrap=None):
    """Ladder macros for a dataset/learner pair with the santander_ladder.json schema."""
    S, N, m = lad[s_key], lad[n_key], lad["_meta"]
    put(f"{prefix}nTest", f"{m['n_test']:,}")
    put(f"{prefix}nTrain", f"{m['n_train']:,}")
    put(f"{prefix}nLabels", f"{len(m['label_names']):,}")
    put(f"{prefix}nVal", f"{m['cal_split']['n_val']:,}")
    put(f"{prefix}wMax", f"{max(m['scale_pos_weight_S']):,.0f}")
    put(f"{prefix}wMin", f"{min(m['scale_pos_weight_S']):,.0f}")
    put(f"{prefix}lnwMin", f"{min(__import__('math').log(w) for w in m['scale_pos_weight_S']):.1f}")
    put(f"{prefix}lnwMax", f"{max(__import__('math').log(w) for w in m['scale_pos_weight_S']):.1f}")
    put(f"{prefix}popMAP", f3(lad["popularity_train_prevalence"]["map7_te"]))
    csp = lad.get("calibration_split_positives")
    if csp:
        put(f"{prefix}calPosMinAlive", csp["cal_pos_min_alive"])
        put(f"{prefix}calPosMedAlive", f"{csp['cal_pos_median_alive']:,.0f}")
        put(f"{prefix}nDeadTest", len(csp["dead_in_test_idx"]))
    for tag, r in ((tag_s, S), (tag_n, N)):
        put(f"{tag}raw", f3(r["raw"]["map7_te"]))
        put(f"{tag}plIsoId", f3(r["per_label_iso_identity"]["map7_te"]))
        put(f"{tag}plIsoPrior", f3(r["per_label_iso_prior"]["map7_te"]))
        put(f"{tag}plIsoExcl", f3(r["per_label_iso_exclude"]["map7_te"]))
        put(f"{tag}pooled", f3(r["pooled_iso"]["map7_te"]))
        put(f"{tag}ceil", f3(r["oracle_in_sample_per_label_iso"]["map7_all"]))
        put(f"{tag}auc", f3(r["within_label_auc_mean_sub200k"]))
        put(f"{tag}satOne", f"{100 * r['saturation']['frac_exact_1']:.1f}")
        put(f"{tag}satZero", f"{100 * r['saturation']['frac_exact_0']:.1f}")
        put(f"{tag}nDead", r["per_label_iso_identity"]["n_dead_in_cal"])
        for kind in ("offset", "platt", "beta"):
            if f"per_label_{kind}_prior" in r:
                put(f"{tag}pl{kind.capitalize()}Prior", f3(r[f"per_label_{kind}_prior"]["map7_te"]))
        put(
            f"{tag}plIsoIdRel",
            pct(r["per_label_iso_identity"]["map7_te"] / r["raw"]["map7_te"] - 1),
        )
        put(
            f"{tag}plIsoPriorRel",
            pct(r["per_label_iso_prior"]["map7_te"] / r["raw"]["map7_te"] - 1),
        )
    put(
        f"{tag_s}elkan", f3(S["elkan_inversion_known_w"]["map7_te"])
    )  # 70% evaluation split, as the mds table
    put(f"{tag_s}elkanAll", f3(S["elkan_inversion_known_w"]["map7_all"]))
    put(f"{tag_s}labelFree", f3(S["prior_match_shift_label_free"]["map7_te"]))
    # labels on which the prevalence-matching shift is NOT identifiable: it either ran to the
    # bisection bound or was pinned by a saturated cell (see emit_shift_ratio)
    put(
        f"{prefix}nCappedShift",
        sum(
            1
            for x, s in zip(
                S["prior_match_shift_label_free"]["shift_b"],
                S["saturation"]["frac_exact_1_per_label"],
                strict=True,
            )
            if x is None or abs(x) >= SEARCH_BOUND or s > 0
        ),
    )
    loss = N["raw"]["map7_te"] - S["raw"]["map7_te"]
    put(f"{tag_s}loss", f3(loss))
    put(
        f"{tag_s}ceilGap", f3(N["raw"]["map7_te"] - S["oracle_in_sample_per_label_iso"]["map7_all"])
    )
    put(
        f"{tag_s}plIsoPriorCeilGap",
        f3(S["oracle_in_sample_per_label_iso"]["map7_all"] - S["per_label_iso_prior"]["map7_te"]),
    )
    share = None
    if loss > 0:

        def share(x):
            return f"{100 * x / loss:.0f}"

        put(
            f"{prefix}repairShare", share(S["per_label_iso_prior"]["map7_te"] - S["raw"]["map7_te"])
        )
        put(
            f"{prefix}ceilShare",
            share(S["oracle_in_sample_per_label_iso"]["map7_all"] - S["raw"]["map7_te"]),
        )
        put(
            f"{prefix}tieShare",
            share(N["raw"]["map7_te"] - S["oracle_in_sample_per_label_iso"]["map7_all"]),
        )
        put(
            f"{prefix}elkanShare",
            share(S["elkan_inversion_known_w"]["map7_all"] - S["raw"]["map7_te"]),
        )
        put(
            f"{prefix}labelFreeShare",
            share(S["prior_match_shift_label_free"]["map7_te"] - S["raw"]["map7_te"]),
        )
        for kind in ("offset", "platt", "beta"):
            if f"per_label_{kind}_prior" in S:
                put(
                    f"{prefix}{kind}Share",
                    share(S[f"per_label_{kind}_prior"]["map7_te"] - S["raw"]["map7_te"]),
                )
    if whatif:
        wi = whatif
        put(f"{prefix}whatifN", f3(wi["N_raw"]))
        put(f"{prefix}whatifIdeal", f3(wi["whatif_N_plus_a_lnw"]["1.0"]))
        put(f"{prefix}whatifHalf", f3(wi["whatif_N_plus_a_lnw"]["0.5"]))
        put(f"{prefix}whatifS", f3(wi["S_actual"]))
        if wi["loss_share"]["odds_shift_part"] is not None:
            put(f"{prefix}oddsShare", f"{100 * wi['loss_share']['odds_shift_part']:.0f}")
            put(f"{prefix}satShare", f"{100 * wi['loss_share']['residual_saturation_part']:.0f}")
            if loss > 0:
                extra = (S["per_label_iso_prior"]["map7_te"] - S["raw"]["map7_te"]) / loss - wi[
                    "loss_share"
                ]["odds_shift_part"]
                put(f"{prefix}repairExtraShare", f"{100 * extra:.0f}")
        put(f"{prefix}overlapWhatif", f"{wi['top7_overlap_N_vs_whatif']['mean']:.2f}")
        put(f"{prefix}overlapS", f"{wi['top7_overlap_N_vs_S_actual']['mean']:.2f}")
        put(
            f"{prefix}fracRowsChangedWhatif",
            f"{100 * wi['top7_overlap_N_vs_whatif']['frac_rows_changed']:.0f}",
        )
    if bootstrap:
        put(f"{prefix}bootB", bootstrap["_meta"]["B"])
        put(f"{prefix}bootN", f"{bootstrap['_meta']['n_rows_with_positives_te']:,}")
        for key, tag in (
            ("S_per_label_iso_prior_minus_popularity", "ciSpriorMinusPop"),
            ("S_per_label_iso_identity_minus_popularity", "ciSidMinusPop"),
            ("N_per_label_iso_prior_minus_N_raw", "ciNprior"),
            ("N_raw_minus_S_per_label_iso_prior", "ciNminusS"),
            ("N_raw_minus_S_raw", "ciNminusSraw"),
            ("S_per_label_iso_prior_minus_S_elkan_inversion_known_w", "ciSpriorMinusElkan"),
        ):
            d = bootstrap["differences"].get(key)
            if d:
                put(
                    f"{prefix}{tag}",
                    f"{d['point']:+.3f}~[{d['ci95'][0]:+.3f},\\,{d['ci95'][1]:+.3f}]",
                )


def emit_deploy_generic(prefix, tag_s, tag_n, dep, s_cfg, n_cfg):
    """Deployment-protocol macros ({dataset}_deploy.json): calibrators fitted on the validation period."""
    S, N = dep[s_cfg], dep[n_cfg]
    for tag, r in ((tag_s, S), (tag_n, N)):
        put(f"dep{tag}raw", f3(r["raw"]))
        put(f"dep{tag}plIsoPrior", f3(r["per_label_iso_prior"]["map7"]))
        put(f"dep{tag}plIsoId", f3(r["per_label_iso_identity"]["map7"]))
        put(f"dep{tag}pooled", f3(r["pooled_iso"]))
        put(f"dep{tag}ceil", f3(r["oracle_in_sample_per_label_iso"]))
        put(f"dep{tag}nDeadVal", r["per_label_iso_prior"]["n_dead_in_val"])
        # distance from each deployment-protocol rung to its own in-sample ceiling, so the
        # text quotes measured gaps instead of a single rounded bound that fits only one arm
        put(
            f"dep{tag}CeilGap",
            f3(r["oracle_in_sample_per_label_iso"] - r["per_label_iso_prior"]["map7"]),
        )
        if "per_label_offset_prior" in r:
            put(f"dep{tag}plOffsetPrior", f3(r["per_label_offset_prior"]))
    put(f"dep{prefix}popMAP", f3(S["popularity"]))
    put(f"dep{prefix}nVal", f"{S['n_val']:,}")
    if "elkan_inversion_known_w" in S:
        put(f"dep{tag_s}elkan", f3(S["elkan_inversion_known_w"]))
    loss = N["raw"] - S["raw"]
    put(f"dep{tag_s}loss", f3(loss))
    if loss > 0:
        put(
            f"dep{prefix}SrepairShare",
            f"{100 * (S['per_label_iso_prior']['map7'] - S['raw']) / loss:.0f}",
        )
        if "per_label_offset_prior" in S:
            put(
                f"dep{prefix}SoffsetShare",
                f"{100 * (S['per_label_offset_prior'] - S['raw']) / loss:.0f}",
            )
    put(f"dep{tag_n}prGain", f"{N['per_label_iso_prior']['map7'] - N['raw']:+.3f}")


def emit_capsweep(prefix, tag_s, cs, stem):
    """Cap-sweep macros ({dataset}_capsweep.json): the realized shift against the step budget T*eta*c,
    read from the JSON per model (never from literals). Also writes figures/tab_capsweep_{stem}_body.tex."""
    n_caps = n_ok = 0
    worst = float("-inf")
    rows = []
    for r in cs["models"].values():
        if r.get("cap") is None:  # the uncapped weighted model
            put(f"{prefix}capNoneSatOne", f"{100 * r['saturation']['frac_exact_1']:.1f}")
            put(f"{prefix}capNoneBPriorMax", f"{r['b_prior']['max_abs']:.2f}")
            put(f"{prefix}capNoneBDirectMax", f"{r['b_direct']['max_abs']:.2f}")
            put(f"{prefix}capNoneNatBound", r["b_prior"]["n_at_search_bound"])
            put(f"{prefix}capNoneRaw", f3(r["map_raw"]["map7_te"]))
            put(f"{prefix}capNoneElkan", f3(r["map_elkan_inversion_known_w"]["map7_te"]))
            put(f"{prefix}capNonePlIsoPrior", f3(r["map_per_label_iso_prior"]["map7_te"]))
            put(f"{prefix}capLnwMax", f"{r['ln_w']['max']:.1f}")
            put(f"{prefix}capLnwMedian", f"{r['ln_w']['median']:.1f}")
            continue
        tec, bc = r["T_eta_c"], r["bound_check"]
        assert abs(tec - r["T"] * r["eta"] * r["cap"]) < 1e-9
        t = f"{tag_s}mds{CAP_TAG[float(r['cap'])]}"
        put(f"{t}raw", f3(r["map_raw"]["map7_te"]))
        put(f"{t}satOne", f"{100 * r['saturation']['frac_exact_1']:.1f}")
        put(f"{t}elkan", f3(r["map_elkan_inversion_known_w"]["map7_te"]))
        put(f"{t}plIsoPrior", f3(r["map_per_label_iso_prior"]["map7_te"]))
        put(f"{t}bPriorMax", f"{r['b_prior']['max_abs']:.2f}")
        put(f"{t}bPriorMedian", f"{r['b_prior']['median']:.2f}")
        put(f"{t}bDirectMax", f"{r['b_direct']['max_abs']:.2f}")
        put(f"{t}TEtaC", f"{tec:.1f}")
        put(f"{t}fracWithinBound", f"{100 * bc['frac_labels_abs_b_prior_le_T_eta_c']:.0f}")
        put(
            f"{t}fracWithinBoundSlack",
            f"{100 * bc['frac_labels_abs_b_prior_le_T_eta_c_plus_0.1']:.0f}",
        )
        put(f"{t}maxExcess", f"{bc['max_excess_b_prior']:+.2f}")
        put(f"{t}fracLnwAbove", f"{100 * bc['frac_labels_ln_w_gt_T_eta_c']:.0f}")
        put(f"{t}nAtBound", r["b_prior"]["n_at_search_bound"])
        put(f"{t}whatifBound", f3(r["map_whatif_N_plus_bound"]["map7_all"]))
        n_caps += 1
        n_ok += bc["frac_labels_abs_b_prior_le_T_eta_c_plus_0.1"] == 1.0
        worst = max(worst, bc["max_excess_b_prior"])
        rows.append((float(r["cap"]), t))
    if not rows:
        raise KeyError("no capped model in the cap sweep")
    put(f"{prefix}capN", n_caps)
    put(f"{prefix}capNwithinBound", n_ok)
    put(f"{prefix}capAllWithinBound", "yes" if n_ok == n_caps else "no")
    put(f"{prefix}capMaxExcess", f"{worst:+.2f}")
    first = next(r for r in cs["models"].values() if r.get("cap") is not None)
    put(f"{prefix}capT", first["T"])
    put(f"{prefix}capEta", first["eta"])
    put(f"{prefix}capNraw", f3(cs["N_raw"]))
    write(
        FIG / f"tab_capsweep_{stem}_body.tex",
        "\n".join(
            f"${c:g}$ & {NUM[t + 'TEtaC']} & {NUM[t + 'bPriorMax']} & {NUM[t + 'fracWithinBound']} & {NUM[t + 'satOne']} & {NUM[t + 'raw']} & {NUM[t + 'elkan']} & {NUM[t + 'plIsoPrior']} \\\\"
            for c, t in sorted(rows)
        )
        + "\n",
    )


def emit_tie(prefix, tag_s, tie):
    """Tie-attribution macros ({dataset}_tie.json) for the uncapped weighted model."""
    r = next(v for v in tie["models"].values() if v.get("cap") is None)
    put(f"{tag_s}tieCellsExactOne", f"{100 * r['frac_cells_exact_1']:.1f}")
    put(f"{tag_s}tieRowsGeOne", f"{100 * r['saturated_cells_per_row']['frac_rows_ge1']:.0f}")
    put(f"{tag_s}tieRowsGeK", f"{100 * r['saturated_cells_per_row']['frac_rows_ge_k']:.0f}")
    put(f"{tag_s}tieCellsPerRowMean", f"{r['saturated_cells_per_row']['mean']:.2f}")
    put(f"{tag_s}tieColumnOrder", f3(r["column_order"]["map7_all"]))
    if "unweighted_order" in r:
        put(f"{tag_s}tieRandom", f3(r["random"]["map7_all_mean"]))
        put(f"{tag_s}tieUnwOrder", f3(r["unweighted_order"]["map7_all"]))
        put(f"{tag_s}tieOracle", f3(r["oracle"]["map7_all"]))
        if r.get("recovered_share_unweighted_order") is not None:
            put(f"{prefix}tieShareUnw", f"{100 * r['recovered_share_unweighted_order']:.0f}")
            put(f"{prefix}tieShareOracle", f"{100 * r['recovered_share_oracle']:.0f}")
    put(f"{prefix}tieNraw", f3(tie["N_raw"]["map7_all"]))


def emit_seeds(prefix, sd):
    """Mean +- SD over the 90% train-row draws ({dataset}_seeds.json)."""
    su = sd["summary"]
    if not su:
        raise KeyError("no draws")
    put(f"{prefix}seedN", sd["_meta"]["n_draws"])
    for key, tag, scale in (
        ("S_raw_te", "seedSraw", 1),
        ("N_raw_te", "seedNraw", 1),
        ("S_per_label_iso_prior_te", "seedSplIsoPrior", 1),
        ("N_per_label_iso_prior_te", "seedNplIsoPrior", 1),
        ("S_pooled_iso_te", "seedSpooled", 1),
        ("S_oracle_te", "seedSceil", 1),
        ("N_oracle_te", "seedNceil", 1),
        ("S_elkan_inversion_known_w_te", "seedSelkan", 1),
        ("S_saturation_frac_exact_1", "seedSsatOne", 100),
        ("loss_share_odds_shift", "seedOddsShare", 100),
    ):
        v = su[key]
        mean, s = scale * v["mean"], (scale * v["sd"] if v["sd"] is not None else None)
        fmt = ".1f" if scale == 100 else ".3f"
        put(f"{prefix}{tag}", f"{mean:{fmt}}" if s is None else f"{mean:{fmt}} \\pm {s:{fmt}}")


def _median(xs):
    xs = sorted(xs)
    m = len(xs)
    return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def emit_shift_ratio(tag_s, lad, s_key):
    """How much of the intended shift the learner actually put in.

    b_j solves mean_i sigmoid(logit q_ij + b) = pi_j by bisection on [-SEARCH_BOUND, SEARCH_BOUND].
    On a label with a cell at exactly 1.0 that equation has no solution driven by the model: the
    clipped cells contribute a constant whatever b is, so the bisection settles wherever the
    remaining cells put it, which is typically just under the bound. Filtering on |b| < SEARCH_BOUND
    alone therefore keeps labels whose b_j is an artefact of the clip (measured: 4 of the 9 labels
    it kept on Santander LightGBM, 7 of 9 on the Santander MLP). b_j is identifiable only where no
    cell saturates, so that is the filter."""
    import math

    m = lad[s_key]
    pm = m["prior_match_shift_label_free"]
    sat = m["saturation"]["frac_exact_1_per_label"]
    keep, dropped_sat = [], 0
    for b, w, s in zip(pm["shift_b"], pm["ln_w"], sat, strict=True):
        if b is None or w == 0 or abs(b) >= SEARCH_BOUND:
            continue
        if s > 0:  # a saturated cell makes b_j unidentifiable, not large
            dropped_sat += 1
            continue
        keep.append(b / w)
    put(f"{tag_s}nSatDroppedShift", dropped_sat)
    put(f"{tag_s}nLabelsShift", len(pm["shift_b"]))
    if len(keep) < MIN_IDENT_LABELS:
        # too few identifiable labels to quote a median; the paper must say so instead
        put(f"{tag_s}nIdentShift", len(keep))
        return
    median = _median(keep)
    put(f"{tag_s}bOverLnwMedian", f"{median:.2f}")
    put(f"{tag_s}nIdentShift", len(keep))
    put(f"{tag_s}bOverLnwMin", f"{min(keep):.2f}")
    put(f"{tag_s}bOverLnwMax", f"{max(keep):.2f}")
    assert math.isfinite(median)


# deployable rungs of the ladder: the oracle ceiling is a reference, not something you can ship
CAP_RUNGS = (
    "raw",
    "elkan_inversion_known_w",
    "pooled_iso",
    "per_label_offset_prior",
    "per_label_platt_prior",
    "per_label_beta_prior",
    "prior_match_shift_label_free",
    "per_label_iso_prior",
)


def emit_cap_rungs(prefix, tag_s, lad):
    """Is per-label isotonic with the prior fallback really the best rung at every cap?

    The cap-sweep JSON carries only raw / inversion / per-label iso, so the full comparison has to
    come from the ladder file, which runs every rung on each capped model. Measured, it is not the
    best rung everywhere: on Santander at c=0.3 it is the worst, below doing nothing, and its own
    in-sample ceiling is below the raw score there, so no per-label isotonic map can help."""
    n_best = n_caps = 0
    for cap, tag in CAP_TAG.items():
        m = lad.get(f"S_weighted_mds{cap:g}")
        if m is None:
            continue
        vals = {
            k: m[k]["map7_te"]
            for k in CAP_RUNGS
            if isinstance(m.get(k), dict) and m[k].get("map7_te") is not None
        }
        if "per_label_iso_prior" not in vals:
            continue
        n_caps += 1
        best = max(vals.values())
        n_best += vals["per_label_iso_prior"] >= best - 1e-9
        t = f"{tag_s}mds{tag}"
        put(f"{t}Ceil", f3(m["oracle_in_sample_per_label_iso"]["map7_all"]))
        put(f"{t}PlBetaPrior", f3(vals["per_label_beta_prior"]))
        put(f"{t}BestRung", f3(best))
    put(f"{prefix}nCapsIsoBest", n_best)
    put(f"{prefix}nCapsRungs", n_caps)


def emit_shift_direct(tag_s, cap, s_key):
    """Second, instrument-independent reading of the realized shift, for the matched pairs where the
    cap sweep ran: the per-label median of logit(S) - logit(N) over the cells strictly inside
    (1e-6, 1-1e-6) in BOTH models. No bisection and no clipped cell enters it, so it cannot inherit
    the artefact above. It needs the unweighted twin, which is why it is the check and not the
    headline: the prevalence-matching shift is label-free and deployable without a matched pair."""
    m = cap["models"][s_key]
    lnw = m["ln_w_per_label"]
    bd = m["b_direct"]["per_label"]
    n_inner = m["b_direct"]["n_inner_cells_per_label"]
    keep = [
        b / w
        for b, w, n in zip(bd, lnw, n_inner, strict=True)
        if b is not None and w not in (0, None) and n > 0
    ]
    if len(keep) < MIN_IDENT_LABELS:
        raise KeyError("too few labels with unclipped cells for the direct shift")
    put(f"{tag_s}bDirectOverLnwMedian", f"{_median(keep):.2f}")
    put(f"{tag_s}nDirectShift", f"{len(keep):,}")


def emit_mlp_all(prefix, tag, res):
    """All-product Instacart MLP ladder (chunked.py: instacart_mlp_all_ladder_{wr,w1}.json)."""
    mp = res["map7"]
    put(f"{prefix}nLabels", f"{res['L']:,}")
    put(f"{prefix}nTest", f"{res['n_test']:,}")
    put(f"{prefix}wMax", f"{res['w_true']['max']:,.0f}")
    put(f"{prefix}wMedian", f"{res['w_true']['median']:,.0f}")
    put(f"{prefix}popMAP", f3(mp["popularity"]["map7_te"]))
    put(f"{tag}raw", f3(mp["raw"]["map7_te"]))
    put(f"{tag}plIsoPrior", f3(mp["per_label_iso_prior"]["map7_te"]))
    put(f"{tag}plIsoId", f3(mp["per_label_iso_identity"]["map7_te"]))
    put(f"{tag}labelFree", f3(mp["prior_match_label_free"]["map7_te"]))
    put(f"{tag}elkan", f3(mp["elkan_inversion_known_w"]["map7_te"]))
    put(f"{tag}whatifPi", f3(mp["whatif_from_pi"]["map7_te"]))
    put(f"{tag}satOne", f"{100 * res['saturation']['frac_exact_1']:.1f}")
    put(f"{tag}satZero", f"{100 * res['saturation']['frac_exact_0']:.1f}")
    put(f"{tag}auc", f3(res["within_label_auc_mean_sub200k"]))
    put(f"{tag}nDead", f"{res['dead_labels']['n_dead_in_cal_split']:,}")
    put(f"{tag}nDeadVal", f"{res['dead_labels']['n_dead_in_val']:,}")
    put(f"{tag}nCappedShift", res["prior_match_shift_label_free"]["n_at_search_bound"])
    put(f"dep{tag}raw", f3(res["deploy"]["raw"]["map7_all"]))
    put(f"dep{tag}plIsoPrior", f3(res["deploy"]["per_label_iso_prior"]["map7_all"]))


# Table 1 (v3): (Santander, Instacart) x (LightGBM, MLP) x (w=r, w=1); "--" where a result is not available yet
COLS_V3 = [
    ("S", ""),
    ("N", ""),
    ("MS", "M"),
    ("MN", "M"),
    ("IS", "I"),
    ("IN", "I"),
    ("IMS", "IM"),
    ("IMN", "IM"),
]


def _c3(tag, key, bold=False):
    v = str(NUM.get(f"{tag}{key}", "--"))
    return f"\\textbf{{{v}}}" if bold and v != "--" else v


def write_ladder_v3():
    rows = [
        ("raw score", lambda t, p: _c3(t, "raw")),
        (
            "analytic inversion with $w_j$",
            lambda t, p: _c3(t, "elkan") if t.endswith("S") else "--",
        ),
        ("per-label iso, dead$\\to$prior", lambda t, p: _c3(t, "plIsoPrior", bold=True)),
        ("pooled (shared) isotonic", lambda t, p: _c3(t, "pooled")),
        ("popularity (train prevalence)", lambda t, p: str(NUM.get(f"{p}popMAP", "--"))),
        None,
        ("oracle per-label ceiling", lambda t, p: _c3(t, "ceil")),
        ("mean within-label AUC", lambda t, p: _c3(t, "auc")),
        ("cells at $1.0$ (\\%)", lambda t, p: _c3(t, "satOne")),
        ("labels without calibration positives", lambda t, p: _c3(t, "nDead")),
    ]
    write(
        FIG / "tab_ladder_v3_body.tex",
        "\n".join(
            "\\midrule"
            if row is None
            else f"{row[0]} & " + " & ".join(row[1](t, p) for t, p in COLS_V3) + " \\\\"
            for row in rows
        )
        + "\n",
    )


# ---------------------------------------------------------------- Santander ladder
lad = load("santander_ladder.json")
S, W, N = lad["S_lgbm_spw"], lad["W_wgboost"], lad["N_lgbm_unweighted"]
pop = lad["popularity_train_prevalence"]["map7_te"]
rows = [("S", S), ("W", W), ("N", N)]
Nh = lad.get("Nh_lgbm_unweighted_100")  # earlier 100-tree, subsampled run (4th column)
if Nh:
    rows.append(("Nh", Nh))
csp = lad.get("calibration_split_positives")
if csp:
    put("calPosMinAlive", csp["cal_pos_min_alive"])
    put("calPosMedAlive", f"{csp['cal_pos_median_alive']:,.0f}")
    put("nDeadTest", len(csp["dead_in_test_idx"]))
    put("nDeadCalAlsoTest", len(set(csp["dead_in_cal_idx"]) & set(csp["dead_in_test_idx"])))
put("nTest", f"{lad['_meta']['n_test']:,}")
put("nTrain", f"{lad['_meta']['n_train']:,}")
put("nLabels", 24)
put("nVal", f"{lad['_meta']['cal_split']['n_val']:,}")
put("wMax", f"{max(lad['_meta']['scale_pos_weight_S']):,.0f}")
put("wMin", f"{min(lad['_meta']['scale_pos_weight_S']):,.0f}")
put("popMAP", f3(pop))
for tag, r in rows:
    put(f"{tag}raw", f3(r["raw"]["map7_te"]))
    put(f"{tag}plIsoId", f3(r["per_label_iso_identity"]["map7_te"]))
    put(f"{tag}plIsoPrior", f3(r["per_label_iso_prior"]["map7_te"]))
    put(f"{tag}plIsoExcl", f3(r["per_label_iso_exclude"]["map7_te"]))
    put(f"{tag}pooled", f3(r["pooled_iso"]["map7_te"]))
    put(f"{tag}ceil", f3(r["oracle_in_sample_per_label_iso"]["map7_all"]))
    put(f"{tag}auc", f3(r["within_label_auc_mean_sub200k"]))
    put(f"{tag}satOne", f"{100 * r['saturation']['frac_exact_1']:.1f}")
    put(f"{tag}satZero", f"{100 * r['saturation']['frac_exact_0']:.1f}")
    put(f"{tag}nDead", r["per_label_iso_identity"]["n_dead_in_cal"])
put("SplIsoIdRel", pct(S["per_label_iso_identity"]["map7_te"] / S["raw"]["map7_te"] - 1))
put("SplIsoPriorRel", pct(S["per_label_iso_prior"]["map7_te"] / S["raw"]["map7_te"] - 1))
put("WplIsoIdRel", pct(W["per_label_iso_identity"]["map7_te"] / W["raw"]["map7_te"] - 1))
put("WplIsoPriorRel", pct(W["per_label_iso_prior"]["map7_te"] / W["raw"]["map7_te"] - 1))
put("NplIsoPriorRel", pct(N["per_label_iso_prior"]["map7_te"] / N["raw"]["map7_te"] - 1))
put("Selkan", f3(S["elkan_inversion_known_w"]["map7_all"]))
# three-way decomposition of the weighted model's loss (all from the ladder)
loss = N["raw"]["map7_te"] - S["raw"]["map7_te"]
put("Sloss", f3(loss))
put(
    "tieShare",
    f"{100 * (N['raw']['map7_te'] - S['oracle_in_sample_per_label_iso']['map7_all']) / loss:.0f}",
)
put(
    "repairShare", f"{100 * (S['per_label_iso_prior']['map7_te'] - S['raw']['map7_te']) / loss:.0f}"
)
put(
    "ceilShare",
    f"{100 * (S['oracle_in_sample_per_label_iso']['map7_all'] - S['raw']['map7_te']) / loss:.0f}",
)
put(
    "elkanShare",
    f"{100 * (S['elkan_inversion_known_w']['map7_all'] - S['raw']['map7_te']) / loss:.0f}",
)
put("SceilGap", f3(N["raw"]["map7_te"] - S["oracle_in_sample_per_label_iso"]["map7_all"]))
put(
    "SplIsoPriorCeilGap",
    f3(S["oracle_in_sample_per_label_iso"]["map7_all"] - S["per_label_iso_prior"]["map7_te"]),
)
b = S["prior_match_shift_label_free"]["shift_b"]
# not identifiable = ran to the bisection bound OR pinned by a saturated cell (see emit_shift_ratio)
put(
    "nCappedShift",
    sum(
        1
        for x, s in zip(b, S["saturation"]["frac_exact_1_per_label"], strict=True)
        if x is None or abs(x) >= SEARCH_BOUND or s > 0
    ),
)
for kind in ("offset", "platt", "beta"):
    if f"per_label_{kind}_prior" in S:
        for tag, r in rows:
            put(f"{tag}pl{kind.capitalize()}Prior", f3(r[f"per_label_{kind}_prior"]["map7_te"]))
        put(
            f"{kind}Share",
            f"{100 * (S[f'per_label_{kind}_prior']['map7_te'] - S['raw']['map7_te']) / loss:.0f}",
        )
put(
    "labelFreeShare",
    f"{100 * (S['prior_match_shift_label_free']['map7_te'] - S['raw']['map7_te']) / loss:.0f}",
)
put("SplIsoPriorFour", f"{S['per_label_iso_prior']['map7_te']:.4f}")
put("SceilFour", f"{S['oracle_in_sample_per_label_iso']['map7_all']:.4f}")
put(
    "WplPlattIsoGap",
    f3(W["per_label_iso_prior"]["map7_te"] - W["per_label_platt_prior"]["map7_te"])
    if "per_label_platt_prior" in W
    else "",
)
put(
    "NplPlattIsoGap",
    f3(N["per_label_iso_prior"]["map7_te"] - N["per_label_platt_prior"]["map7_te"])
    if "per_label_platt_prior" in N
    else "",
)
put(
    "NplBetaIsoGap",
    f3(N["per_label_iso_prior"]["map7_te"] - N["per_label_beta_prior"]["map7_te"])
    if "per_label_beta_prior" in N
    else "",
)
dd = W["per_label_iso_identity"].get("dead_label_diagnostics")
if dd:
    put("WdeadMeanInTop", f"{dd['mean_dead_labels_in_top7']:.2f}")
    put("WdeadAllRowsPct", f"{100 * dd['frac_rows_all_dead_in_top7']:.0f}")
    put("WdeadRawMean", f"{dd['mean_raw_score_dead_labels']:.3f}")
    put("WaliveCalMax", f"{dd['max_mean_calibrated_score_alive_labels']:.4f}")
put("SlabelFree", f3(S["prior_match_shift_label_free"]["map7_te"]))

TAGS = [
    t for t, _ in rows if t != "Nh"
]  # main table: S, W, N; the earlier run goes to the appendix
if csp and "n_rows_with_positives_te" in lad.get("_meta", {}):
    pass
# share of evaluation rows with at least one positive (MAP averages over these rows only)
try:
    _bs = load("santander_bootstrap.json")
    _n_te_rows = lad["_meta"]["n_test"] - lad["_meta"]["cal_split"]["n_val"]
    put("fracRowsWithPos", f"{100 * _bs['_meta']['n_rows_with_positives_te'] / _n_te_rows:.1f}")
    put("nEvalRows", f"{_n_te_rows:,}")
except (FileNotFoundError, KeyError):
    pass


def _cols(key, bold=False, fmt=None):
    cells = []
    for t in TAGS:
        v = NUM.get(f"{t}{key}", "--") if fmt is None else fmt(t)
        cells.append(f"\\textbf{{{v}}}" if bold else str(v))
    return " & ".join(cells)


tab_ladder = "\n".join(
    [
        f"raw score & {_cols('raw')} \\\\",
        f"per-label iso, dead$\\to$identity & {_cols('plIsoId')} \\\\",
        f"per-label iso, dead$\\to$prior & {_cols('plIsoPrior', bold=True)} \\\\",
        f"per-label iso, dead$\\to$excluded & {_cols('plIsoExcl')} \\\\",
        *(
            [
                f"per-label logit shift (intercept only), dead$\\to$prior & {_cols('plOffsetPrior')} \\\\"
            ]
            if "SplOffsetPrior" in NUM
            else []
        ),
        *(
            [
                f"per-label Platt, dead$\\to$prior & {_cols('plPlattPrior')} \\\\",
                f"per-label beta, dead$\\to$prior & {_cols('plBetaPrior')} \\\\",
            ]
            if "SplPlattPrior" in NUM
            else []
        ),
        f"pooled (shared) isotonic & {_cols('pooled')} \\\\",
        f"label-free prevalence shift & {NUM['SlabelFree']} & "
        + " & ".join("--" for _ in TAGS[1:])
        + " \\\\",
        "popularity (train prevalence) & " + " & ".join(str(NUM["popMAP"]) for _ in TAGS) + " \\\\",
        "\\midrule",
        f"oracle per-label ceiling & {_cols('ceil')} \\\\",
        f"mean within-label AUC & {_cols('auc')} \\\\",
        f"cells at 1.0 / 0.0 (\\%) & {_cols('', fmt=lambda t: str(NUM[t + 'satOne']) + ' / ' + str(NUM[t + 'satZero']))} \\\\",
    ]
)
write(FIG / "tab_ladder_body.tex", tab_ladder + "\n")
if Nh:  # appendix: the matched control against the earlier 100-tree, subsampled run
    keys = [
        ("raw score", "raw"),
        ("per-label iso, dead$\\to$identity", "plIsoId"),
        ("per-label iso, dead$\\to$prior", "plIsoPrior"),
        ("per-label iso, dead$\\to$excluded", "plIsoExcl"),
        ("per-label logit shift (intercept only)", "plOffsetPrior"),
        ("per-label Platt", "plPlattPrior"),
        ("per-label beta", "plBetaPrior"),
        ("pooled (shared) isotonic", "pooled"),
        ("oracle per-label ceiling", "ceil"),
        ("mean within-label AUC", "auc"),
    ]
    write(
        FIG / "tab_ladder_earlier_body.tex",
        "\n".join(
            f"{lab} & {NUM['N' + k]} & {NUM['Nh' + k]} \\\\"
            for lab, k in keys
            if "N" + k in NUM and "Nh" + k in NUM
        )
        + "\n",
    )
    put(
        "NhMaxAbsDiff",
        f"{max(abs(float(NUM['N' + k]) - float(NUM['Nh' + k])) for _, k in keys if k != 'auc' and 'N' + k in NUM and 'Nh' + k in NUM):.3f}",
    )

# ---------------------------------------------------------------- max_delta_step variants (B2)
for cfg, tag in (
    ("S_orig_lgbm_spw", "Sorig"),
    ("S_weighted_mds0.7", "Smdsa"),
    ("S_weighted_mds2", "Smdsb"),
):
    r = lad.get(cfg)
    if not r:
        continue
    put(f"{tag}raw", f3(r["raw"]["map7_te"]))
    put(f"{tag}satOne", f"{100 * r['saturation']['frac_exact_1']:.1f}")
    put(f"{tag}satZero", f"{100 * r['saturation']['frac_exact_0']:.1f}")
    put(f"{tag}plIsoPrior", f3(r["per_label_iso_prior"]["map7_te"]))
    put(f"{tag}pooled", f3(r["pooled_iso"]["map7_te"]))
    put(f"{tag}ceil", f3(r["oracle_in_sample_per_label_iso"]["map7_all"]))
    put(f"{tag}auc", f3(r["within_label_auc_mean_sub200k"]))
    if "elkan_inversion_known_w" in r:
        put(f"{tag}elkan", f3(r["elkan_inversion_known_w"]["map7_te"]))
        put(f"{tag}elkanAll", f3(r["elkan_inversion_known_w"]["map7_all"]))
        lo_ = N["raw"]["map7_te"] - r["raw"]["map7_te"]
        if lo_ > 0:
            put(
                f"{tag}elkanShare",
                f"{100 * (r['elkan_inversion_known_w']['map7_te'] - r['raw']['map7_te']) / lo_:.0f}",
            )
            put(
                f"{tag}repairShare",
                f"{100 * (r['per_label_iso_prior']['map7_te'] - r['raw']['map7_te']) / lo_:.0f}",
            )
    put(f"{tag}loss", f3(N["raw"]["map7_te"] - r["raw"]["map7_te"]))
    if (
        "prior_match_shift_label_free" in r
    ):  # realized per-label shift (prevalence matching), excluding unreachable labels
        bb = [abs(x) for x in r["prior_match_shift_label_free"]["shift_b"] if abs(x) < 39.9]
        put(f"{tag}shiftMax", f"{max(bb):.2f}" if bb else "--")
        put(
            f"{tag}nCapped",
            sum(1 for x in r["prior_match_shift_label_free"]["shift_b"] if abs(x) >= 39.9),
        )
# total logit movement a leaf step cap allows in T rounds at learning rate eta: T * eta * cap (design constants of the runs)
put("mdsBoundA", f"{60 * 0.05 * 0.7:.1f}")
put("mdsBoundB", f"{60 * 0.05 * 2.0:.1f}")
put("lnwMin", f"{min(__import__('math').log(w) for w in lad['_meta']['scale_pos_weight_S']):.1f}")
put("lnwMax", f"{max(__import__('math').log(w) for w in lad['_meta']['scale_pos_weight_S']):.1f}")
if "Smdsaraw" in NUM:
    put("SorigMinusS", f"{lad['S_orig_lgbm_spw']['raw']['map7_te'] - S['raw']['map7_te']:+.4f}")
    # Santander has its own dose-response: the same configuration was trained at
    # sqrt(r) and 10r as well. The paper used to say these arms did not exist.
    for _k, _tag in (("S_weighted_sqrt", "SsqrtRaw"), ("S_weighted_10r", "StenrRaw")):
        put(_tag, f3(lad[_k]["raw"]["map7_te"]))
    put("SsqrtPlIsoPrior", f3(lad["S_weighted_sqrt"]["per_label_iso_prior"]["map7_te"]))
    put("StenrPlIsoPrior", f3(lad["S_weighted_10r"]["per_label_iso_prior"]["map7_te"]))
    for _k, _tag in (("S_weighted_sqrt", "Ssqrt"), ("S_weighted_10r", "Stenr")):
        put(f"{_tag}SatOne", f"{100 * lad[_k]['saturation']['frac_exact_1']:.1f}")
    put("Selkan", f3(S["elkan_inversion_known_w"]["map7_te"]))  # same split as the mds rows
    mds_rows = [
        ("no cap (\\Sm)", "S"),
        ("no cap, earlier run", "Sorig"),
        ("cap $0.7$", "Smdsa"),
        ("cap $2$", "Smdsb"),
    ]
    lines = []
    for label, t in mds_rows:
        elk = NUM.get(f"{t}elkan", NUM.get("Selkan", "--"))
        lines.append(
            f"{label} & {NUM[t + 'raw']} & {NUM[t + 'satOne']} & {elk} & {NUM[t + 'plIsoPrior']} & {NUM[t + 'ceil']} & {NUM[t + 'auc']} \\\\"
        )
    write(FIG / "tab_mds_body.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- what-if decomposition
try:
    wi = load("santander_whatif.json")
    put("whatifN", f3(wi["N_raw"]))
    put("whatifIdeal", f3(wi["whatif_N_plus_a_lnw"]["1.0"]))
    put("whatifHalf", f3(wi["whatif_N_plus_a_lnw"]["0.5"]))
    put("whatifS", f3(wi["S_actual"]))
    put("oddsShare", f"{100 * wi['loss_share']['odds_shift_part']:.0f}")
    put("satShare", f"{100 * wi['loss_share']['residual_saturation_part']:.0f}")
    put("overlapWhatif", f"{wi['top7_overlap_N_vs_whatif']['mean']:.2f}")
    put("overlapS", f"{wi['top7_overlap_N_vs_S_actual']['mean']:.2f}")
    put("fracRowsChangedWhatif", f"{100 * wi['top7_overlap_N_vs_whatif']['frac_rows_changed']:.0f}")
    put("idealRel", pct(wi["whatif_N_plus_a_lnw"]["1.0"] / wi["N_raw"] - 1))
    put("SactualRel", pct(wi["S_actual"] / wi["N_raw"] - 1))
    # share of the loss that per-label isotonic recovers beyond the ideal odds-shift share
    odds_share = (wi["N_raw"] - wi["whatif_N_plus_a_lnw"]["1.0"]) / (wi["N_raw"] - wi["S_actual"])
    repair_share = (S["per_label_iso_prior"]["map7_te"] - S["raw"]["map7_te"]) / loss
    put("repairExtraShare", f"{100 * (repair_share - odds_share):.0f}")
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- bootstrap
try:
    bs = load("santander_bootstrap.json")
    put("bootB", bs["_meta"]["B"])
    put("bootN", f"{bs['_meta']['n_rows_with_positives_te']:,}")
    for key, tag in [
        ("S_per_label_iso_prior_minus_popularity", "ciSpriorMinusPop"),
        ("S_per_label_iso_identity_minus_popularity", "ciSidMinusPop"),
        ("W_per_label_iso_prior_minus_W_raw", "ciWprior"),
        ("N_per_label_iso_prior_minus_N_raw", "ciNprior"),
        ("N_raw_minus_S_per_label_iso_prior", "ciNminusS"),
    ]:
        d = bs["differences"][key]
        put(tag, f"{d['point']:+.3f}~[{d['ci95'][0]:+.3f},\\,{d['ci95'][1]:+.3f}]")
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- MULAN dose response
try:
    mu = load("mulan_dose_response.json")
    lines = []
    for name, d in mu["datasets"].items():
        for lname in mu["learners"]:
            for wname in mu["weight_schemes"]:
                c = d["cells"].get(f"{lname}|{wname}")
                if not c or "error" in c:
                    continue
                wi_ = c.get("map_whatif_unweighted_plus_lnw")
                lname_tex = lname.replace("_", "\\_")
                lines.append(
                    f"{name} & {d['L']} & {lname_tex} & {wname} & {c['map_raw']:.3f} & "
                    f"{'--' if wi_ is None else f'{wi_:.3f}'} & {c['map_elkan_inversion_known_w']:.3f} & {c['map_per_label_iso_identity']:.3f} & "
                    f"{c['map_per_label_iso_prior']:.3f} & {c['map_pooled_iso']:.3f} & {c['map_oracle_ceiling']:.3f} & {c['within_label_auc']:.3f} & "
                    f"{c['distinct_scores_per_label_test']:.1f} & {100 * c['sat_ge_1m1e9']:.1f} \\\\"
                )
    write(FIG / "tab_mulan_full_body.tex", "\n".join(lines) + "\n")
    # Split into parts of two datasets each so that each fits one page of the appendix.
    # The part COUNT follows the data: a fixed range(4) silently dropped the three
    # datasets added when the sweep went from 8 to 11, while the supplement went on
    # promising "every MULAN cell". The assert below makes that failure impossible.
    names = list(mu["datasets"])
    per_part = 2
    n_parts = -(-len(names) // per_part)
    covered, calls = [], []
    for part in range(n_parts):
        sel = set(names[per_part * part : per_part * (part + 1)])
        part_lines = [ln for ln in lines if ln.split(" & ")[0] in sel]
        covered += part_lines
        write(FIG / f"tab_mulan_full_part{part + 1}.tex", chr(10).join(part_lines) + chr(10))
        calls.append(
            BS
            + f"mulanfulltable{{All MULAN cells, part {part + 1} of {n_parts}.}}"
            + f"{{tab:mulan_full_{part + 1}}}{{figures/tab_mulan_full_part{part + 1}.tex}}"
        )
    assert covered == lines, (
        f"the per-cell tables print {len(covered)} of {len(lines)} MULAN rows; "
        "every dataset must appear in exactly one part"
    )
    # the supplement inputs this, so the number of printed tables always matches the data
    write(FIG / "tab_mulan_full_parts.tex", chr(10).join(calls) + chr(10))
    put("nMulanFullParts", n_parts)
    # compact headline: per dataset, lgbm_default w=1 vs w=r (raw / whatif / inversion / per-label prior / ceiling)
    comp = []
    try:
        calstats = load("mulan_cal_stats.json")["datasets"]
    except FileNotFoundError:
        calstats = {}
    for name, d in mu["datasets"].items():
        a = d["cells"].get("lgbm_default|w=1")
        b = d["cells"].get("lgbm_default|w=r")
        if not a or not b or "error" in a or "error" in b:
            continue
        calmed = f"{calstats[name]['cal_pos_per_label_median']:.0f}" if name in calstats else "--"
        # K = min(7, L) is the evaluated depth; on a dataset with L <= 7 it selects nothing, so the
        # metric measures ordering only. min r is reported because a label with r < 1 is
        # majority-positive, where w_j = r_j down-weights the positive class instead of up-weighting it.
        comp.append(
            f"{name} & {d['L']} & {min(7, d['L'])} & {d['n_fit']} & {d['n_test']} & {calmed} & "
            f"{d['r_neg_over_pos']['min']:.1f} & {d['r_neg_over_pos']['max']:.0f} & {d['popularity_map']:.3f} & {a['map_raw']:.3f} & {a['map_per_label_iso_prior']:.3f} & {b['map_raw']:.3f} & "
            f"{b['map_whatif_unweighted_plus_lnw']:.3f} & {b['map_elkan_inversion_known_w']:.3f} & {b['map_per_label_iso_prior']:.3f} & {b['map_oracle_ceiling']:.3f} & "
            f"{a['distinct_scores_per_label_test']:.0f} & {b['distinct_scores_per_label_test']:.0f} \\\\"
        )
    write(FIG / "tab_mulan_compact_body.tex", "\n".join(comp) + "\n")
    put("nMulan", len(mu["datasets"]))
    put("nLearners", len(mu["learners"]))
    dss = mu["datasets"].values()
    # datasets where the evaluated depth covers every label, so top-K selects nothing
    put("nMulanKeqL", sum(1 for d in dss if min(7, d["L"]) >= d["L"]))
    # datasets carrying a majority-positive label, where the recommended weight is a down-weight
    put("nMulanMajPos", sum(1 for d in dss if d["r_neg_over_pos"]["min"] < 1.0))
    put("nMulanMaxrMax", f"{max(d['r_neg_over_pos']['max'] for d in dss):.0f}")

    # named cells used in prose (never hand-typed)
    def cell(ds, lname, wname):
        return mu["datasets"][ds]["cells"].get(f"{lname}|{wname}", {})

    en1, enr = cell("enron", "lgbm_strong", "w=1"), cell("enron", "lgbm_strong", "w=r")
    if en1 and enr:
        put("enronStrongRawWone", f3(en1["map_raw"]))
        put("enronStrongRawWr", f3(enr["map_raw"]))
        put("enronStrongInvWr", f3(enr["map_elkan_inversion_known_w"]))
        put("enronStrongWhatifWr", f3(enr["map_whatif_unweighted_plus_lnw"]))
    me1, me10 = cell("medical", "lgbm_strong", "w=1"), cell("medical", "lgbm_strong", "w=10r")
    if me1 and me10:
        put("medStrongDistinctWone", f"{me1['distinct_scores_per_label_test']:.0f}")
        put("medStrongDistinctWtenr", f"{me10['distinct_scores_per_label_test']:.1f}")
        put("medStrongCeilWone", f3(me1["map_oracle_ceiling"]))
        put("medStrongCeilWtenr", f3(me10["map_oracle_ceiling"]))
    md = mu["datasets"].get("medical")
    if md:
        put("medL", md["L"])
        put("medDeadCal", md["n_labels_zero_pos_cal"])
        gaps = [
            c["map_per_label_iso_prior"] - c["map_per_label_iso_identity"]
            for c in md["cells"].values()
            if "error" not in c
        ]
        put("medIdVsPriorMaxGap", f"{max(gaps):.2f}")
    # genbase/XGBoost is the counter-example to "a distinct-score collapse sinks the ceiling":
    # the count collapses further than on medical and the ceiling and the inversion both hold
    _gx1 = mu["datasets"]["genbase"]["cells"]["xgb|w=1"]
    _gxr = mu["datasets"]["genbase"]["cells"]["xgb|w=r"]
    put("genbaseXgbDistinctWone", f"{_gx1['distinct_scores_per_label_test']:.1f}")
    put("genbaseXgbDistinctWr", f"{_gxr['distinct_scores_per_label_test']:.1f}")
    put("genbaseXgbCeilWone", f3(_gx1["map_oracle_ceiling"]))
    put("genbaseXgbCeilWr", f3(_gxr["map_oracle_ceiling"]))
    put("genbaseXgbInvWr", f3(_gxr["map_elkan_inversion_known_w"]))
    put("genbaseXgbRawWone", f3(_gx1["map_raw"]))
    put(
        "medStrongSatWtenr",
        f"{100 * mu['datasets']['medical']['cells']['lgbm_strong|w=10r']['sat_ge_1m1e9']:.2f}",
    )
    put("nMulanModest", sum(1 for d in mu["datasets"].values() if d["r_neg_over_pos"]["max"] < 100))
    # the modest-imbalance claim needs both scopes: the default learner alone is
    # one-signed, and across the five learners the bound is much wider
    modest = [n for n, d in mu["datasets"].items() if d["r_neg_over_pos"]["max"] < 100]
    _dl = [
        mu["datasets"][n]["cells"]["lgbm_default|w=r"]["map_raw"]
        - mu["datasets"][n]["cells"]["lgbm_default|w=1"]["map_raw"]
        for n in modest
    ]
    put("modestDefaultWorst", f"{min(_dl):+.3f}")
    put("modestDefaultBest", f"{max(_dl):+.3f}")
    _al = []
    for n in modest:
        for lname in mu["learners"]:
            c1 = mu["datasets"][n]["cells"].get(f"{lname}|w=1")
            cr = mu["datasets"][n]["cells"].get(f"{lname}|w=r")
            if c1 and cr and "error" not in c1 and "error" not in cr:
                _al.append((cr["map_raw"] - c1["map_raw"], n, lname))
    _lo, _hi = min(_al), max(_al)
    put("modestAnyWorst", f"{_lo[0]:+.3f}")
    put("modestAnyWorstCell", f"{_lo[1]} ({LNAME[_lo[2]]})")
    put("modestAnyBest", f"{_hi[0]:+.3f}")
    put(
        "nMulanLtSeventy",
        sum(1 for _n, d in mu["datasets"].items() if d["r_neg_over_pos"]["max"] < 70),
    )
    for ds, tag in (("birds", "birds"), ("medical", "med")):
        dd_ = mu["datasets"][ds]
        c1 = dd_["cells"]["lgbm_default|w=1"]
        put(f"{tag}Ncal", dd_["n_cal"])
        put(f"{tag}RawWone", f3(c1["map_raw"]))
        put(f"{tag}PlIsoPriorWone", f3(c1["map_per_label_iso_prior"]))
    gb = mu["datasets"]["genbase"]["cells"]
    put("genbaseDistinctWone", f"{gb['lgbm_default|w=1']['distinct_scores_per_label_test']:.0f}")
    put("genbaseDistinctWr", f"{gb['lgbm_default|w=r']['distinct_scores_per_label_test']:.0f}")
    mc = mu["datasets"]["medical"]["cells"]
    put("medLogregRawWone", f3(mc["logreg|w=1"]["map_raw"]))
    put("medLogregRawWr", f3(mc["logreg|w=r"]["map_raw"]))
    put("medXgbRawWone", f3(mc["xgb|w=1"]["map_raw"]))
    put("medXgbRawWr", f3(mc["xgb|w=r"]["map_raw"]))
    # datasets on which per-label iso (prior) hurts the unweighted model, by learner
    hurt = {}
    for name, d in mu["datasets"].items():
        for lname in mu["learners"]:
            c = d["cells"].get(f"{lname}|w=1", {})
            if c and "error" not in c and c["map_per_label_iso_prior"] < c["map_raw"] - 0.005:
                hurt.setdefault(name, []).append(lname)
    put("nHurtAllLearners", sum(1 for v in hurt.values() if len(v) == len(mu["learners"])))
    put("hurtDatasets", ", ".join(sorted(hurt)))
    # default LightGBM only: datasets where per-label iso (prior) lowers MAP@K by more than 0.005 at w=1
    deltas = {
        name: d["cells"]["lgbm_default|w=1"]["map_per_label_iso_prior"]
        - d["cells"]["lgbm_default|w=1"]["map_raw"]
        for name, d in mu["datasets"].items()
    }
    put("hurtDefaultDatasets", ", ".join(sorted(n for n, v in deltas.items() if v < -0.005)))
    put("nHurtDefault", sum(1 for v in deltas.values() if v < -0.005))
    put("maxAbsDeltaElsewhere", f"{max(abs(v) for n, v in deltas.items() if v >= -0.005):.3f}")
    try:
        cs = load("mulan_cal_stats.json")["datasets"]
        for name, s in cs.items():
            put(f"calPosMed{name.capitalize()}", f"{s['cal_pos_per_label_median']:.0f}")
        med_lines = []
        for name in mu["datasets"]:
            s = cs.get(name)
            if s:
                med_lines.append(
                    f"{name} & {s['n_cal']} & {s['cal_pos_per_label_median']:.0f} & {s['n_labels_lt10_cal_pos']} \\\\"
                )
        write(FIG / "tab_mulan_calstats_body.tex", "\n".join(med_lines) + "\n")
    except FileNotFoundError:
        pass
    # how often the what-if prediction is within 0.03 of the trained weighted model (default LightGBM, w=r)
    close = 0
    drops_modest = []
    for _name, d in mu["datasets"].items():
        b = d["cells"].get("lgbm_default|w=r", {})
        a = d["cells"].get("lgbm_default|w=1", {})
        if b and "error" not in b:
            close += abs(b["map_whatif_unweighted_plus_lnw"] - b["map_raw"]) <= 0.03
            if d["r_neg_over_pos"]["max"] < 100:
                drops_modest.append(a["map_raw"] - b["map_raw"])
    put("nWhatifClose", close)
    put("maxDropModest", f"{max(drops_modest):.3f}")
    put(
        "enronDefaultWhatifWr",
        f3(mu["datasets"]["enron"]["cells"]["lgbm_default|w=r"]["map_whatif_unweighted_plus_lnw"]),
    )
    put("enronDefaultRawWr", f3(mu["datasets"]["enron"]["cells"]["lgbm_default|w=r"]["map_raw"]))
    put(
        "genbaseDefaultWhatifWr",
        f3(
            mu["datasets"]["genbase"]["cells"]["lgbm_default|w=r"]["map_whatif_unweighted_plus_lnw"]
        ),
    )
    put(
        "genbaseDefaultRawWr", f3(mu["datasets"]["genbase"]["cells"]["lgbm_default|w=r"]["map_raw"])
    )
    # ------------------------------------------------------------ the collapse on MULAN
    # Corel5k and delicious are the two added datasets whose label count is in the regime
    # of Section 3. Read the collapse the same way as there: how far raw falls at w=r,
    # whether the loss is between labels (within-label AUC) rather than within them, and
    # whether the separable repair returns the pair to the unweighted level.
    HALF = 0.5
    cells_all, collapsed = [], []
    for name, d in mu["datasets"].items():
        for lname in mu["learners"]:
            c1 = d["cells"].get(f"{lname}|w=1")
            cr = d["cells"].get(f"{lname}|w=r")
            if not c1 or not cr or "error" in c1 or "error" in cr:
                continue
            cells_all.append((name, lname, c1, cr))
            if cr["map_raw"] < HALF * c1["map_raw"]:
                collapsed.append((name, lname, c1, cr))
    put("nDoseCells", len(cells_all))
    put("nCollapseCells", len(collapsed))
    put("collapseDatasets", ", ".join(sorted({n for n, _l, _a, _b in collapsed})))
    put("nDelCollapseLearners", sum(1 for n, _l, _a, _b in collapsed if n == "delicious"))
    put("nCorelCollapseLearners", sum(1 for n, _l, _a, _b in collapsed if n == "corel5k"))
    # whatever is left over, so the three counts always add up to nCollapseCells
    put(
        "nOtherCollapseCells",
        sum(1 for n, _l, _a, _b in collapsed if n not in ("delicious", "corel5k")),
    )
    # per-dataset readings for the two, default LightGBM
    for name, tag in (("corel5k", "corel"), ("delicious", "del")):
        d = mu["datasets"][name]
        c1 = d["cells"]["lgbm_default|w=1"]
        cr = d["cells"]["lgbm_default|w=r"]
        put(f"{tag}L", f"{d['L']:,}")
        put(f"{tag}Nfit", f"{d['n_fit']:,}")
        put(f"{tag}NtestDose", f"{d['n_test']:,}")
        put(f"{tag}Maxr", f"{d['r_neg_over_pos']['max']:,.0f}")
        put(f"{tag}Pop", f3(d["popularity_map"]))
        put(f"{tag}DeadFit", d["n_labels_zero_pos_fit"])
        put(f"{tag}DeadCalDose", cr["n_dead_in_cal"])
        put(f"{tag}RawWone", f3(c1["map_raw"]))
        put(f"{tag}RawWr", f3(cr["map_raw"]))
        put(f"{tag}PlIsoPriorWone", f3(c1["map_per_label_iso_prior"]))
        put(f"{tag}PlIsoPriorWr", f3(cr["map_per_label_iso_prior"]))
        put(f"{tag}PlIsoIdWr", f3(cr["map_per_label_iso_identity"]))
        put(f"{tag}PooledWr", f3(cr["map_pooled_iso"]))
        put(f"{tag}InvWr", f3(cr["map_elkan_inversion_known_w"]))
        put(f"{tag}CeilWr", f3(cr["map_oracle_ceiling"]))
        put(f"{tag}AucWone", f"{c1['within_label_auc']:.3f}")
        put(f"{tag}AucWr", f"{cr['within_label_auc']:.3f}")
    # the two controls that keep us from naming a threshold: bibtex has the label count
    # without the weight spread, enron has the weight spread without the label count
    en = mu["datasets"]["enron"]
    put("enronL", f"{en['L']:,}")
    put("enronMaxr", f"{en['r_neg_over_pos']['max']:,.0f}")
    bx = mu["datasets"]["bibtex"]
    put("bibtexL", f"{bx['L']:,}")
    put("bibtexMaxr", f"{bx['r_neg_over_pos']['max']:,.0f}")
    put("bibtexRawWone", f3(bx["cells"]["lgbm_default|w=1"]["map_raw"]))
    put("bibtexRawWr", f3(bx["cells"]["lgbm_default|w=r"]["map_raw"]))
    # does the separable repair put the weighted arm back where the unweighted one sits
    # under the same calibrator? Measured on every learner of the two collapse datasets.
    gaps, pooled_gaps = [], []
    for name in ("corel5k", "delicious"):
        d = mu["datasets"][name]
        for lname in mu["learners"]:
            c1 = d["cells"].get(f"{lname}|w=1")
            cr = d["cells"].get(f"{lname}|w=r")
            if not c1 or not cr or "error" in c1 or "error" in cr:
                continue
            gaps.append(c1["map_per_label_iso_prior"] - cr["map_per_label_iso_prior"])
            pooled_gaps.append(c1["map_per_label_iso_prior"] - cr["map_pooled_iso"])
    put("maxRepairGapCollapse", f"{max(gaps):.3f}")
    put("minPooledGapCollapse", f"{min(pooled_gaps):.3f}")
    put("nRepairCollapseCells", len(gaps))
    # the shape of the sweep away from the collapse cells, so the abstract can state it
    # instead of asserting a monotone dose-response the data does not show
    gain_cells = mono_cells = 0
    for _n, d in mu["datasets"].items():
        for lname in mu["learners"]:
            c1 = d["cells"].get(f"{lname}|w=1")
            cr = d["cells"].get(f"{lname}|w=r")
            if not c1 or not cr or "error" in c1 or "error" in cr:
                continue
            if cr["map_raw"] >= HALF * c1["map_raw"] and cr["map_raw"] > c1["map_raw"]:
                gain_cells += 1
            seq = [
                d["cells"][f"{lname}|{w}"]["map_raw"]
                for w in mu["weight_schemes"]
                if f"{lname}|{w}" in d["cells"] and "error" not in d["cells"][f"{lname}|{w}"]
            ]
            if len(seq) == len(mu["weight_schemes"]) and all(
                seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1)
            ):
                mono_cells += 1
    put("nDoseGain", gain_cells)
    put("nDoseMonotone", mono_cells)
    # the prior-vs-identity fallback gap on Corel5k is a default-LightGBM reading, not the
    # dataset's: on XGBoost the two fallbacks are indistinguishable
    _cg = [
        d_["map_per_label_iso_prior"] - d_["map_per_label_iso_identity"]
        for lname in mu["learners"]
        if (d_ := mu["datasets"]["corel5k"]["cells"].get(f"{lname}|w=r")) and "error" not in d_
    ]
    put("corelIdGapMin", f"{min(_cg):.3f}")
    put("corelIdGapMax", f"{max(_cg):.3f}")
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- synthetic
try:
    sy = load("synthetic_check.json")
    import statistics as st

    groups: dict[tuple, list] = {}
    for c in sy["cells"]:
        groups.setdefault((c["prev_decades"], c["weight_exponent"]), []).append(c)
    lines = []
    for (pd_, e), cs in sorted(groups.items()):

        def m(path, cs=cs):
            vals = []
            for c in cs:
                v = c
                for k in path:
                    v = v[k]
                vals.append(v)
            return st.mean(vals)

        cells = [
            f"{pd_:.0f}",
            f"{e:.1f}",
            f"{m(['ln_w_spread']):.1f}",
            f"{m(['map_true_p']):.3f}",
            f"{m(['logreg', 'map_raw']):.3f}",
            f"{m(['logreg', 'map_elkan_inversion']):.3f}",
            f"{m(['lgbm_small', 'map_raw']):.3f}",
            f"{m(['lgbm_small', 'map_elkan_inversion']):.3f}",
            f"{m(['lgbm_small', 'map_oracle_ceiling']):.3f}",
            f"{100 * m(['lgbm_small', 'sat_ge_1m1e9']):.0f}",
            f"{m(['lgbm_large', 'map_raw']):.3f}",
            f"{m(['lgbm_large', 'map_elkan_inversion']):.3f}",
            f"{m(['lgbm_large', 'map_oracle_ceiling']):.3f}",
            f"{100 * m(['lgbm_large', 'sat_ge_1m1e9']):.0f}",
            f"{m(['xgb', 'map_raw']):.3f}",
            f"{m(['xgb', 'map_elkan_inversion']):.3f}",
            f"{m(['mlp_posweight', 'map_raw']):.3f}",
            f"{m(['mlp_posweight', 'map_elkan_inversion']):.3f}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    write(FIG / "tab_synthetic_body.tex", "\n".join(lines) + "\n")
    cfg = sy["config"]
    put("synL", cfg["L"])
    put("synN", f"{cfg['n_tr']:,}")
    put("synSeeds", len(cfg["seeds"]))
    put("synK", cfg["k"])
    g3 = groups.get((3.0, 1.0), [])
    if g3:
        put("synTrueD", f3(st.mean(c["map_true_p"] for c in g3)))
        put("synLRrawD", f3(st.mean(c["logreg"]["map_raw"] for c in g3)))
        put("synLRinvD", f3(st.mean(c["logreg"]["map_elkan_inversion"] for c in g3)))
        put("synLRrawDeZero", f3(st.mean(c["logreg"]["map_raw"] for c in groups[(3.0, 0.0)])))
        put("synSmallInvD", f3(st.mean(c["lgbm_small"]["map_elkan_inversion"] for c in g3)))
        put("synSmallCeilD", f3(st.mean(c["lgbm_small"]["map_oracle_ceiling"] for c in g3)))
        put("synSmallSatD", f"{100 * st.mean(c['lgbm_small']['sat_ge_1m1e9'] for c in g3):.0f}")
        put("synLargeRawD", f3(st.mean(c["lgbm_large"]["map_raw"] for c in g3)))
        put("synLargeInvD", f3(st.mean(c["lgbm_large"]["map_elkan_inversion"] for c in g3)))
        put("synLargeCeilD", f3(st.mean(c["lgbm_large"]["map_oracle_ceiling"] for c in g3)))
        put("synLargeSatD", f"{100 * st.mean(c['lgbm_large']['sat_ge_1m1e9'] for c in g3):.0f}")
        put("synXgbRawD", f3(st.mean(c["xgb"]["map_raw"] for c in g3)))
        put("synXgbInvD", f3(st.mean(c["xgb"]["map_elkan_inversion"] for c in g3)))
        put("synMlpInvD", f3(st.mean(c["mlp_posweight"]["map_elkan_inversion"] for c in g3)))
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- leaf lemma
try:
    lf = load("leaf_check.json")
    lines = []
    for name, d in lf["datasets"].items():
        for wname, t in d["tree_exact"].items():
            lines.append(
                f"{name} & {wname} & {100 * t['predicted_sat_from_lemma']:.1f} & {100 * t['measured_sat']:.1f} & {t['map_raw']:.3f} & {t['map_oracle_ceiling']:.3f} & {t['distinct_scores_per_label']:.0f} \\\\"
            )
    write(FIG / "tab_leaf_tree_body.tex", "\n".join(lines) + "\n")
    lines = []
    for name, d in lf["datasets"].items():
        for mc in lf["min_child"]:
            cells = [d["lgbm"][f"min_child={mc}|{w}"] for w in lf["weights"]]
            lines.append(
                f"{name} & {mc} & "
                + " & ".join(f"{100 * c['sat_ge_1m1e3']:.1f}" for c in cells)
                + " & "
                + " & ".join(f"{c['map_oracle_ceiling']:.3f}" for c in cells)
                + " \\\\"
            )
    write(FIG / "tab_leaf_lgbm_body.tex", "\n".join(lines) + "\n")
    all_sat = [c["sat_ge_1m1e3"] for d in lf["datasets"].values() for c in d["lgbm"].values()]
    put("leafLgbmMaxSat", f"{100 * max(all_sat):.0f}")
    # the single-tree implementation check saturates further than the LightGBM sweep, so the
    # two bounds are emitted separately and the prose can name which table each one belongs to
    tree_sat = [
        t["measured_sat"] for d in lf["datasets"].values() for t in d["tree_exact"].values()
    ]
    put("leafTreeMaxSat", f"{100 * max(tree_sat):.1f}")
    med = lf["datasets"]["medical"]["lgbm"]
    put("leafMedCeilFiftyWone", f3(med["min_child=50|w=1"]["map_oracle_ceiling"]))
    put("leafMedCeilFiftyWhundred", f3(med["min_child=50|w=100r"]["map_oracle_ceiling"]))
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- MULAN tau selection (recipe check)
try:
    tu = load("mulan_tau_select.json")
    put("nTauCands", len(tu["taus"]))
    put("tauList", ", ".join(str(t) for t in tu["taus"]))
    put("tauFolds", tu["cv"]["n_folds"])
    put("tauBootB", tu["bootstrap"]["B"])

    def ci_str(c):
        return f"{c['point']:+.3f}~[{c['ci95'][0]:+.3f},\\,{c['ci95'][1]:+.3f}]"

    lines = []
    hurt0 = hurt_sel = 0  # default LightGBM, w=1: CI of (calibrated - raw) entirely below zero
    n_shared = 0
    for name, d in tu["datasets"].items():
        c = d["cells"].get("lgbm_default|w=1")
        if not c or "error" in c:
            continue
        sel = c["choice"] if c["choice"] in ("shared", "raw") else f"$\\tau={c['tau_selected']}$"
        n_shared += c["choice"] == "shared"
        hurt0 += c["ci_tau0_minus_raw"]["ci95"][1] < 0
        hurt_sel += c["ci_selected_minus_raw"]["ci95"][1] < 0
        d_sel = "--" if c["choice"] == "raw" else ci_str(c["ci_selected_minus_raw"])
        lines.append(
            f"{name} & {d['n_cal']} & {d['n_test']} & {c['map_raw']:.3f} & {c['map_per_label_iso_prior_tau0']:.3f} & {ci_str(c['ci_tau0_minus_raw'])} & {sel} & {c['map_selected']:.3f} & {d_sel} & {c['map_shared']:.3f} \\\\"
        )
        tag = {
            "birds": "birds",
            "medical": "med",
            "flags": "flags",
            "genbase": "genbase",
            "enron": "enron",
            "corel5k": "corel",
            "delicious": "del",
        }.get(name)
        if tag:
            put(f"{tag}Ntest", d["n_test"])
            put(
                f"{tag}TauSel",
                c["choice"] if c["choice"] in ("shared", "raw") else str(c["tau_selected"]),
            )
            put(f"{tag}PlIsoTauSel", f3(c["map_selected"]))
            put(f"{tag}CiTauZero", ci_str(c["ci_tau0_minus_raw"]))
            put(f"{tag}CiTauSel", ci_str(c["ci_selected_minus_raw"]))
            put(f"{tag}Shared", f3(c["map_shared"]))
    write(FIG / "tab_mulan_tau_body.tex", "\n".join(lines) + "\n")
    put("nHurtTauZeroCI", hurt0)
    put("nHurtAfterTau", hurt_sel)
    put("nTauChoseShared", n_shared)
    # what-if CI at w=r (default LightGBM): how many datasets have |what-if - raw| CI covering 0
    wc = 0
    for name, d in tu["datasets"].items():
        c = d["cells"].get("lgbm_default|w=r", {})
        if c.get("ci_whatif_minus_raw"):
            lo_, hi_ = c["ci_whatif_minus_raw"]["ci95"]
            wc += lo_ <= 0 <= hi_
            tag = {"flags": "flags", "genbase": "genbase", "enron": "enron"}.get(name)
            if tag:
                put(f"{tag}CiWhatif", ci_str(c["ci_whatif_minus_raw"]))
    put("nWhatifCiCoversZero", wc)
    # across all learners and both weights: the worst outcome of the selection rule against raw, and where
    worst_cell = min(
        (
            (c["map_selected"] - c["map_raw"], name, key)
            for name, d in tu["datasets"].items()
            for key, c in d["cells"].items()
            if "error" not in c
        ),
        key=lambda t: t[0],
    )
    put("tauWorstSelMinusRaw", f"{worst_cell[0]:+.3f}")
    put(
        "tauWorstSelCell",
        f"{worst_cell[1]}, {LNAME[worst_cell[2].split('|')[0]]}, ${worst_cell[2].split('|')[1].replace('w=', 'w{=}')}$",
    )
    wc_ = tu["datasets"][worst_cell[1]]["cells"][worst_cell[2]]
    oof_ = wc_["oof_map_by_candidate"]
    put("tauWorstOofRaw", f3(oof_["raw"]))
    put(
        "tauWorstOofSel",
        f3(
            oof_["shared"]
            if wc_["choice"] == "shared"
            else oof_[f"per_label_tau{wc_['tau_selected']}"]
        ),
    )
    put("tauWorstOofMin", f3(min(oof_.values())))
    put("tauWorstTestRaw", f3(wc_["map_raw"]))
    put("tauWorstTestSel", f3(wc_["map_selected"]))
    put("tauWorstTestMin", f3(min(wc_["map_per_label_iso_prior_by_tau"].values())))
    # every cell where the selected candidate ends more than 0.01 below raw, named
    fails = [
        (name, key)
        for name, d in tu["datasets"].items()
        for key, c in d["cells"].items()
        if "error" not in c and c["map_selected"] < c["map_raw"] - 0.01
    ]
    put(
        "tauFailCells",
        "; ".join(
            f"{n} ({LNAME[k.split('|')[0]]}, ${k.split('|')[1].replace('w=', 'w{=}')}$)"
            for n, k in fails
        ),
    )
    # Table 6: non-raw selections whose paired interval lies entirely above zero
    put(
        "nTauSigGain",
        sum(
            1
            for d in tu["datasets"].values()
            for k, c in d["cells"].items()
            if k == "lgbm_default|w=1"
            and "error" not in c
            and c["choice"] != "raw"
            and c["ci_selected_minus_raw"]["ci95"][0] > 0
        ),
    )
    n_cells = sum(
        1 for d in tu["datasets"].values() for c in d["cells"].values() if "error" not in c
    )
    put("nTauCells", n_cells)
    # the shared map is not a safe fallback: it is what the rule picked in its own worst cell
    shared_bad = [
        (name, k, c)
        for name, d in tu["datasets"].items()
        for k, c in d["cells"].items()
        if "error" not in c and c["map_shared"] < c["map_raw"] - 0.01
    ]
    put("nSharedBelowRaw", len(shared_bad))
    put("nSharedBelowRawWone", sum(1 for _n, k, _c in shared_bad if k.endswith("|w=1")))
    _sw = min(shared_bad, key=lambda t: t[2]["map_shared"] - t[2]["map_raw"])
    _swd = _sw[2]["map_shared"] - _sw[2]["map_raw"]
    put("sharedWorstDelta", f"{_swd:+.3f}")
    _swl, _sww = _sw[1].split("|")
    put("sharedWorstCell", f"{_sw[0]} ({LNAME[_swl]}, $w{{=}}{_sww[2:]}$)")
    put(
        "nTauSelBelowRaw",
        sum(
            1
            for d in tu["datasets"].values()
            for c in d["cells"].values()
            if "error" not in c and c["map_selected"] < c["map_raw"] - 0.01
        ),
    )
    put(
        "nTauZeroBelowRaw",
        sum(
            1
            for d in tu["datasets"].values()
            for c in d["cells"].values()
            if "error" not in c and c["map_per_label_iso_prior_tau0"] < c["map_raw"] - 0.01
        ),
    )
    put(
        "nTauChoseRaw",
        sum(
            1
            for d in tu["datasets"].values()
            for c in d["cells"].values()
            if "error" not in c and c["choice"] == "raw"
        ),
    )
    worst0 = min(
        c["map_per_label_iso_prior_tau0"] - c["map_raw"]
        for d in tu["datasets"].values()
        for c in d["cells"].values()
        if "error" not in c
    )
    put("tauWorstZeroMinusRaw", f"{worst0:+.3f}")
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- deployment protocol (calibrate on val period)
try:
    dp = load("santander_deploy.json")
    tagmap = {
        "unweighted": "depN",
        "weighted": "depS",
        "weighted_mds0.7": "depSmdsa",
        "weighted_mds2": "depSmdsb",
    }
    for cfg, tag in tagmap.items():
        r = dp.get(cfg)
        if not r:
            continue
        put(f"{tag}raw", f3(r["raw"]))
        put(f"{tag}plIsoId", f3(r["per_label_iso_identity"]["map7"]))
        put(f"{tag}plIsoPrior", f3(r["per_label_iso_prior"]["map7"]))
        put(f"{tag}plIsoExcl", f3(r["per_label_iso_exclude"]["map7"]))
        put(f"{tag}pooled", f3(r["pooled_iso"]))
        put(f"{tag}plOffsetPrior", f3(r["per_label_offset_prior"]))
        put(f"{tag}plPlattPrior", f3(r["per_label_platt_prior"]))
        put(f"{tag}ceil", f3(r["oracle_in_sample_per_label_iso"]))
        put(f"{tag}nDeadVal", r["per_label_iso_prior"]["n_dead_in_val"])
    if "depNraw" in NUM and "depSraw" in NUM:
        put("depPop", f3(dp["unweighted"]["popularity"]))
        put("nValRows", f"{dp['unweighted']['n_val']:,}")
        lo_ = dp["unweighted"]["raw"] - dp["weighted"]["raw"]
        put("depSloss", f3(lo_))
        put(
            "depSrepairShare",
            f"{100 * (dp['weighted']['per_label_iso_prior']['map7'] - dp['weighted']['raw']) / lo_:.0f}",
        )
        put(
            "depSoffsetShare",
            f"{100 * (dp['weighted']['per_label_offset_prior'] - dp['weighted']['raw']) / lo_:.0f}",
        )
        put(
            "depNprGain",
            f"{dp['unweighted']['per_label_iso_prior']['map7'] - dp['unweighted']['raw']:+.3f}",
        )
        put(
            "depSminusN",
            f"{dp['weighted']['per_label_iso_prior']['map7'] - dp['unweighted']['raw']:+.3f}",
        )
        # largest difference between the two protocols on the per-label isotonic (prior) rung
        put(
            "protoMaxDiff",
            f3(
                max(
                    abs(
                        dp["weighted"]["per_label_iso_prior"]["map7"]
                        - S["per_label_iso_prior"]["map7_te"]
                    ),
                    abs(
                        dp["unweighted"]["per_label_iso_prior"]["map7"]
                        - N["per_label_iso_prior"]["map7_te"]
                    ),
                )
            ),
        )
        rows_dep = [
            ("raw score", "raw"),
            ("per-label iso, dead$\\to$prior", "plIsoPrior"),
            ("per-label iso, dead$\\to$identity", "plIsoId"),
            ("per-label logit shift (intercept only)", "plOffsetPrior"),
            ("pooled (shared) isotonic", "pooled"),
            ("oracle per-label ceiling", "ceil"),
        ]
        write(
            FIG / "tab_deploy_body.tex",
            "\n".join(f"{lab} & {NUM['depS' + k]} & {NUM['depN' + k]} \\\\" for lab, k in rows_dep)
            + "\n",
        )
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- calibration-split size sweep (B3)
try:
    cz = load("santander_calsize.json")
    import statistics as st_

    lines = []
    band = None  # on the unweighted model N: smallest split at which per-label (prior) still beats the shared map
    lose = None  # on N: largest split at which per-label loses to the shared map
    s_pl_min, s_po_max = 1.0, 0.0
    max_sd = 0.0
    for fr in cz["fracs"]:
        cells = []
        for tag in ("S", "N"):
            rs = [r for r in cz["models"][tag] if r["frac"] == fr]
            pl = st_.mean(r["per_label_iso_prior"] for r in rs)
            po = st_.mean(r["pooled_iso"] for r in rs)
            raw_ = st_.mean(r["raw"] for r in rs)
            dead = st_.mean(r["n_dead"] for r in rs)
            med = st_.mean(r["cal_pos_median"] for r in rs)
            for k_ in ("per_label_iso_prior", "pooled_iso", "raw"):
                max_sd = max(max_sd, st_.pstdev([r[k_] for r in rs]))
            if tag == "S":  # split statistics are identical for both models (same rows): print once
                cells += [f"{rs[0]['n_cal']:,}", f"{med:.0f}", f"{dead:.1f}"]
            cells += [
                f"{raw_:.3f}",
                f"{pl:.3f}",
                f"{po:.3f}",
            ]
            # macro names must not contain digits (TeX stops the control sequence at the first digit)
            key = f"calsize{tag}" + {
                0.3: "ThirtyPct",
                0.1: "TenPct",
                0.03: "ThreePct",
                0.01: "OnePct",
                0.003: "PointThreePct",
            }.get(
                fr,
                "Frac"
                + str(fr)
                .replace(".", "p")
                .replace("0", "Zero")
                .replace("1", "One")
                .replace("3", "Three"),
            )
            put(f"{key}Pl", f3(pl))
            put(f"{key}Pooled", f3(po))
            put(f"{key}Raw", f3(raw_))
            put(f"{key}Dead", f"{dead:.1f}")
            put(f"{key}Ncal", f"{rs[0]['n_cal']:,}")
            put(f"{key}Med", f"{med:.0f}")
            if tag == "N":
                if pl > po:
                    band = (fr, rs[0]["n_cal"], med)
                elif lose is None:
                    lose = (fr, rs[0]["n_cal"], med, pl, raw_)
            else:
                s_pl_min = min(s_pl_min, pl)
                s_po_max = max(s_po_max, po)
        lines.append(f"{fr} & " + " & ".join(cells) + " \\\\")
    write(FIG / "tab_calsize_body.tex", "\n".join(lines) + "\n")
    if band:
        put("calsizeSmallestFracPlWins", str(band[0]))
        put("calsizeSmallestNcalPlWins", f"{band[1]:,}")
        put("calsizeSmallestMedPlWins", f"{band[2]:.0f}")
    if lose:
        put("calsizeLoseFrac", str(lose[0]))
        put("calsizeLoseNcal", f"{lose[1]:,}")
        put("calsizeLoseMed", f"{lose[2]:.0f}")
        put("calsizeLosePl", f3(lose[3]))
        put("calsizeLoseRaw", f3(lose[4]))
    put("calsizeSplMin", f3(s_pl_min))
    put("calsizeSpooledMax", f3(s_po_max))
    put("calsizeMaxSd", f"{max_sd:.4f}")
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- dead-label frequency (gate G)
try:
    df_ = load("dead_label_frequency.json")
    sa = df_["santander"]["by_frac"]
    put("deadFreqSantOnePct", f"{sa['0.01']['dead_mean']:.1f}")
    put("deadFreqSantTenPct", f"{sa['0.1']['dead_mean']:.1f}")
    put("deadFreqSantThirtyPct", f"{sa['0.3']['dead_mean']:.1f}")
    put("deadFreqSantOnePctNcal", f"{sa['0.01']['n_cal']:,}")
    n_any = sum(1 for r in df_["mulan"].values() if r["by_frac"]["0.3"]["dead_mean"] > 0)
    put("nMulanDeadAtThirty", n_any)
    n_any10 = sum(1 for r in df_["mulan"].values() if r["by_frac"]["0.1"]["dead_mean"] > 0)
    put("nMulanDeadAtTen", n_any10)
    put("deadFreqMedicalThirty", f"{df_['mulan']['medical']['by_frac']['0.3']['dead_mean']:.0f}")
    put("deadFreqEnronThirty", f"{df_['mulan']['enron']['by_frac']['0.3']['dead_mean']:.1f}")
    put("deadFreqBirdsTen", f"{df_['mulan']['birds']['by_frac']['0.1']['dead_mean']:.1f}")
    # implementation survey of single-class calibration windows (docs/dead_label_survey_20260907.md
    # in the thesis repository): 17 implementation rows across 11 code bases, one identity pass-through
    put("nSurveyImpl", 17)
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", _why(e))

# ---------------------------------------------------------------- v3: Instacart, MLP, cap sweep, ties, draws, all products
# Everything below is appended after the v2 macros, so the first 408 lines of numbers.tex do not move.
put(
    "InLabels", "4,000"
)  # top-N products of the Instacart matched pair (label_meta.json; overwritten from the ladder JSON)
LADDERS = [  # prefix, tag_s, tag_n, S key, N key, S config, N config, results stem
    ("I", "IS", "IN", "S_lgbm_spw", "N_lgbm_unweighted", "weighted", "unweighted", "instacart"),
    ("M", "MS", "MN", "S_mlp_posweight", "N_mlp_unweighted", "wr", "w1", "santander_mlp"),
    ("IM", "IMS", "IMN", "S_mlp_posweight", "N_mlp_unweighted", "wr", "w1", "instacart_mlp"),
]
for prefix, tag_s, tag_n, s_key, n_key, s_cfg, n_cfg, stem in LADDERS:
    try:
        emit_ladder_generic(
            prefix,
            tag_s,
            tag_n,
            load(f"{stem}_ladder.json"),
            s_key,
            n_key,
            whatif=_try_load(f"{stem}_whatif.json"),
            bootstrap=_try_load(f"{stem}_bootstrap.json"),
        )
    except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
        print("skipped section:", _why(e))
    try:
        emit_deploy_generic(prefix, tag_s, tag_n, load(f"{stem}_deploy.json"), s_cfg, n_cfg)
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
    try:
        # how far the deployment protocol moves the per-label isotonic rung away from the
        # test-split one, so the text quotes a measured bound rather than "to the third decimal"
        _dp = load(f"{stem}_deploy.json")
        _ld = load(f"{stem}_ladder.json")
        _diffs = [
            abs(
                _dp[cfg]["per_label_iso_prior"]["map7"] - _ld[key]["per_label_iso_prior"]["map7_te"]
            )
            for cfg, key in ((s_cfg, s_key), (n_cfg, n_key))
        ]
        put(f"{prefix}protoMaxDiffPair", f3(max(_diffs)))
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
for prefix, tag_s, stem in (("", "S", "santander"), ("I", "IS", "instacart")):
    try:
        emit_capsweep(prefix, tag_s, load(f"{stem}_capsweep.json"), stem)
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
    try:
        emit_tie(prefix, tag_s, load(f"{stem}_tie.json"))
    except (FileNotFoundError, KeyError, StopIteration) as e:
        print("skipped section:", _why(e))
    try:
        emit_seeds(prefix, load(f"{stem}_seeds.json"))
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
for cfg, tag in (("wr", "IMAS"), ("w1", "IMAN")):
    try:
        emit_mlp_all("IMA", tag, load(f"instacart_mlp_all_ladder_{cfg}.json"))
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
for tag_s, s_key, stem in (
    ("S", "S_lgbm_spw", "santander"),
    ("MS", "S_mlp_posweight", "santander_mlp"),
    ("IS", "S_lgbm_spw", "instacart"),
    ("IMS", "S_mlp_posweight", "instacart_mlp"),
):
    try:
        emit_shift_ratio(tag_s, load(f"{stem}_ladder.json"), s_key)
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
    try:  # the clipping-free cross-check, where the cap sweep ran (LightGBM pairs)
        emit_shift_direct(tag_s, load(f"{stem}_capsweep.json"), s_key)
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
    try:  # which rung actually wins at each cap
        emit_cap_rungs("" if tag_s == "S" else tag_s[:-1], tag_s, load(f"{stem}_ladder.json"))
    except (FileNotFoundError, KeyError) as e:
        print("skipped section:", _why(e))
# the two instrument constants the shift estimator depends on, so the paper can state them
put("shiftClipLogit", f"{SHIFT_EPS_LOGIT:.1f}")
put("shiftSearchBound", "40")
# The prediction arrays are stored as single-precision probabilities, so "exactly 1.0"
# is a statement about that representation: it means a raw margin above this many nat.
# Emitted as a macro so the paper states the instrument rather than implying an
# infinite score. See docs/audit_20260910_gates.md, Gate 2.
_F32_BELOW_ONE = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
put("satMarginF", f"{math.log(_F32_BELOW_ONE / (1.0 - _F32_BELOW_ONE)):.1f}")
put("satMarginD", f"{math.log((1.0 - 2.0**-53) / 2.0**-53):.1f}")
write_ladder_v3()

# ---------------------------------------------------------------- emit macros
macros = ["% AUTO-GENERATED by make_tables.py -- do not edit"]
assert all(k.isalpha() for k in NUM), (
    f"non-letter macro names: {sorted(k for k in NUM if not k.isalpha())}"
)
for k, v in NUM.items():
    macros.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
write(FIG / "numbers.tex", "\n".join(macros) + "\n")
write(FIG / "numbers.json", json.dumps(NUM, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {len(NUM)} macros and tables to {FIG}")
