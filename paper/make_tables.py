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

tab_ladder = "\n".join(
    [
        f"raw score & {NUM['Sraw']} & {NUM['Wraw']} & {NUM['Nraw']} \\\\",
        f"per-label iso, dead$\\to$identity & {NUM['SplIsoId']} & {NUM['WplIsoId']} & {NUM['NplIsoId']} \\\\",
        f"per-label iso, dead$\\to$prior & \\textbf{{{NUM['SplIsoPrior']}}} & \\textbf{{{NUM['WplIsoPrior']}}} & \\textbf{{{NUM['NplIsoPrior']}}} \\\\",
        f"per-label iso, dead$\\to$excluded & {NUM['SplIsoExcl']} & {NUM['WplIsoExcl']} & {NUM['NplIsoExcl']} \\\\",
        *(
            [
                f"per-label logit shift (intercept only), dead$\\to$prior & {NUM['SplOffsetPrior']} & {NUM['WplOffsetPrior']} & {NUM['NplOffsetPrior']} \\\\"
            ]
            if "SplOffsetPrior" in NUM
            else []
        ),
        *(
            [
                f"per-label Platt, dead$\\to$prior & {NUM['SplPlattPrior']} & {NUM['WplPlattPrior']} & {NUM['NplPlattPrior']} \\\\",
                f"per-label beta, dead$\\to$prior & {NUM['SplBetaPrior']} & {NUM['WplBetaPrior']} & {NUM['NplBetaPrior']} \\\\",
            ]
            if "SplPlattPrior" in NUM
            else []
        ),
        f"pooled (shared) isotonic & {NUM['Spooled']} & {NUM['Wpooled']} & {NUM['Npooled']} \\\\",
        f"label-free prevalence shift & {NUM['SlabelFree']} & -- & -- \\\\",
        f"popularity (train prevalence) & {NUM['popMAP']} & {NUM['popMAP']} & {NUM['popMAP']} \\\\",
        "\\midrule",
        f"oracle per-label ceiling & {NUM['Sceil']} & {NUM['Wceil']} & {NUM['Nceil']} \\\\",
        f"mean within-label AUC & {NUM['Sauc']} & {NUM['Wauc']} & {NUM['Nauc']} \\\\",
        f"cells at 1.0 / 0.0 (\\%) & {NUM['SsatOne']} / {NUM['SsatZero']} & {NUM['WsatOne']} / {NUM['WsatZero']} & {NUM['NsatOne']} / {NUM['NsatZero']} \\\\",
    ]
)
write(FIG / "tab_ladder_body.tex", tab_ladder + "\n")

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
            f"{name} & {d['L']} & {d['n_fit']} & {calmed} & {d['r_neg_over_pos']['max']:.0f} & {a['map_raw']:.3f} & {a['map_per_label_iso_prior']:.3f} & {b['map_raw']:.3f} & "
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
    put("maxAbsDeltaElsewhere", f"{max(abs(v) for n, v in deltas.items() if v >= -0.005):.2f}")
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

# ---------------------------------------------------------------- emit macros
macros = ["% AUTO-GENERATED by make_tables.py -- do not edit"]
for k, v in NUM.items():
    macros.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
write(FIG / "numbers.tex", "\n".join(macros) + "\n")
write(FIG / "numbers.json", json.dumps(NUM, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {len(NUM)} macros and tables to {FIG}")
