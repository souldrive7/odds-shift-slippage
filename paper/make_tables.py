"""Generate every number the manuscript cites from experiments/arxiv_v2_rederive/results/*.json.

Outputs (all under figures/):
  numbers.tex   \\newcommand macros (\\nXxx) used in prose -- numbers are never hand-typed
  numbers.json  flat key -> value map (for verify_numbers.py and for the reader)
  tab_*.tex     table bodies included with \\input

Run:  python make_tables.py   (from manuscript/arxiv_v2)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _results_dir() -> Path:
    """Release layout (../results) first, then the thesis-repository layout."""
    for cand in (
        HERE.parent / "results",
        HERE.parents[1] / "experiments" / "arxiv_v2_rederive" / "results",
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


def put(key, val, fmt=None):
    NUM[key] = val if fmt is None else fmt
    return NUM[key]


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:+.1f}"


def write(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


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
put("nCappedShift", sum(1 for x in b if abs(x) >= 39.9))
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

TAGS = [t for t, _ in rows]  # S, W, N (+ Nh)


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
if "Smdsaraw" in NUM:
    put("SorigMinusS", f"{lad['S_orig_lgbm_spw']['raw']['map7_te'] - S['raw']['map7_te']:+.4f}")
    put("Selkan", f3(S["elkan_inversion_known_w"]["map7_te"]))  # same split as the mds rows
    mds_rows = [
        ("\\Sm, as trained (\\texttt{max\\_delta\\_step} $=0$)", "S"),
        ("\\Sm, earlier run of the same configuration", "Sorig"),
        ("\\Sm, \\texttt{max\\_delta\\_step} $=0.7$", "Smdsa"),
        ("\\Sm, \\texttt{max\\_delta\\_step} $=2$", "Smdsb"),
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
    print("skipped section:", repr(e))

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
    print("skipped section:", repr(e))

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
    # split into four parts (two datasets each) so that each fits one page of the appendix
    names = list(mu["datasets"])
    for part in range(4):
        sel = set(names[2 * part : 2 * part + 2])
        part_lines = [ln for ln in lines if ln.split(" & ")[0] in sel]
        write(FIG / f"tab_mulan_full_part{part + 1}.tex", "\n".join(part_lines) + "\n")
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
        comp.append(
            f"{name} & {d['L']} & {d['n_fit']} & {calmed} & {d['r_neg_over_pos']['max']:.0f} & {d['popularity_map']:.3f} & {a['map_raw']:.3f} & {a['map_per_label_iso_prior']:.3f} & {b['map_raw']:.3f} & "
            f"{b['map_whatif_unweighted_plus_lnw']:.3f} & {b['map_elkan_inversion_known_w']:.3f} & {b['map_per_label_iso_prior']:.3f} & {b['map_oracle_ceiling']:.3f} & "
            f"{a['distinct_scores_per_label_test']:.0f} & {b['distinct_scores_per_label_test']:.0f} \\\\"
        )
    write(FIG / "tab_mulan_compact_body.tex", "\n".join(comp) + "\n")
    put("nMulan", len(mu["datasets"]))
    put("nLearners", len(mu["learners"]))

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
    put("nMulanModest", sum(1 for d in mu["datasets"].values() if d["r_neg_over_pos"]["max"] < 100))
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
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", repr(e))

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
    print("skipped section:", repr(e))

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
    med = lf["datasets"]["medical"]["lgbm"]
    put("leafMedCeilFiftyWone", f3(med["min_child=50|w=1"]["map_oracle_ceiling"]))
    put("leafMedCeilFiftyWhundred", f3(med["min_child=50|w=100r"]["map_oracle_ceiling"]))
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", repr(e))

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
    LNAME = {
        "lgbm_default": "LightGBM",
        "lgbm_strong": "LightGBM (regularized)",
        "xgb": "XGBoost",
        "logreg": "logistic regression",
        "mlp_posweight": "MLP",
    }
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
    n_cells = sum(
        1 for d in tu["datasets"].values() for c in d["cells"].values() if "error" not in c
    )
    put("nTauCells", n_cells)
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
    print("skipped section:", repr(e))

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
    print("skipped section:", repr(e))

# ---------------------------------------------------------------- calibration-split size sweep (B3)
try:
    cz = load("santander_calsize.json")
    import statistics as st_

    lines = []
    band = None  # on the unweighted model N: smallest split at which per-label (prior) still beats the shared map
    lose = None  # on N: largest split at which per-label loses to the shared map
    s_pl_min, s_po_max = 1.0, 0.0
    for fr in cz["fracs"]:
        cells = []
        for tag in ("S", "N"):
            rs = [r for r in cz["models"][tag] if r["frac"] == fr]
            pl = st_.mean(r["per_label_iso_prior"] for r in rs)
            po = st_.mean(r["pooled_iso"] for r in rs)
            raw_ = st_.mean(r["raw"] for r in rs)
            dead = st_.mean(r["n_dead"] for r in rs)
            med = st_.mean(r["cal_pos_median"] for r in rs)
            cells += [
                f"{rs[0]['n_cal']:,}",
                f"{med:.0f}",
                f"{dead:.1f}",
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
except (FileNotFoundError, KeyError) as e:  # result not (yet) available in the expected schema
    print("skipped section:", repr(e))

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
    print("skipped section:", repr(e))

# ---------------------------------------------------------------- emit macros
macros = ["% AUTO-GENERATED by make_tables.py -- do not edit"]
for k, v in NUM.items():
    macros.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
write(FIG / "numbers.tex", "\n".join(macros) + "\n")
write(FIG / "numbers.json", json.dumps(NUM, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {len(NUM)} macros and tables to {FIG}")
