"""Accept-line gates G2 and G3, computed from result JSON only (no training).

G2 (after the Instacart ladder + what-if): the decomposition has the Santander shape.
G3 (after the MLP ladders + the cap sweeps): the mechanism is learner-agnostic and the step-budget
proposition (main.tex label prop:budget, |b_j| <= T*eta*c) is not falsified.

The criteria are the ones fixed in the approved plan of 2026-09-07 (section 10) BEFORE the results
existed; this script only reads them off. Usage:

    python gate_check.py            # prints the table and the verdict, writes results/gate_check.json
    python gate_check.py --quiet    # verdict lines only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # the verdict lines are Japanese; a CP932 console must not garble them
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_CANONICAL_RES = ROOT / "artifacts" / "results" / "canonical"
RES = _CANONICAL_RES if _CANONICAL_RES.exists() else HERE / "results"

S_LGBM, N_LGBM = "S_lgbm_spw", "N_lgbm_unweighted"
S_MLP, N_MLP = "S_mlp_posweight", "N_mlp_unweighted"


def load(name: str):
    p = RES / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(name: str) -> str | None:
    p = RES / name
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, gate: str, cid: str, what: str, criterion: str, value, passed: bool | None) -> None:
        status = "NA" if passed is None else ("PASS" if passed else "FAIL")
        self.rows.append({"gate": gate, "id": cid, "what": what, "criterion": criterion, "value": value, "status": status})

    def status(self, gate: str, ids: tuple[str, ...]) -> str:
        sel = [r for r in self.rows if r["gate"] == gate and r["id"] in ids]
        if not sel or any(r["status"] == "NA" for r in sel):
            return "NA"
        return "PASS" if all(r["status"] == "PASS" for r in sel) else "FAIL"


def repair_share(S: dict, N: dict) -> float | None:
    loss = N["raw"]["map7_te"] - S["raw"]["map7_te"]
    if loss <= 0:
        return None
    return (S["per_label_iso_prior"]["map7_te"] - S["raw"]["map7_te"]) / loss


def elkan_share(S: dict, N: dict) -> float | None:
    loss = N["raw"]["map7_te"] - S["raw"]["map7_te"]
    if loss <= 0:
        return None
    return (S["elkan_inversion_known_w"]["map7_te"] - S["raw"]["map7_te"]) / loss


def gate_g2(c: Checks) -> None:
    lad, wi, dep = load("instacart_ladder.json"), load("instacart_whatif.json"), load("instacart_deploy.json")
    if lad is None or S_LGBM not in lad or N_LGBM not in lad:
        for cid, what, crit in (
            ("H4a", "odds-shift share of the loss", "10% <= share <= 40%"),
            ("H4b", "per-label iso (prior) repair share", ">= 80% of the loss"),
            ("H8", "popularity below the unweighted raw score", "pop < N raw"),
            ("H7", "per-label iso (prior) on the unweighted model", "|N plIsoPrior - N raw| <= 0.01"),
            ("H6", "deployment vs test-split protocol (S, per-label iso prior)", "|diff| <= 0.01"),
        ):
            c.add("G2", cid, what, crit, None, None)
        return
    S, N = lad[S_LGBM], lad[N_LGBM]
    odds = wi["loss_share"]["odds_shift_part"] if wi else None
    c.add("G2", "H4a", "odds-shift share of the loss", "10% <= share <= 40%", None if odds is None else round(100 * odds, 1), None if odds is None else 10 <= 100 * odds <= 40)
    rs = repair_share(S, N)
    c.add("G2", "H4b", "per-label iso (prior) repair share", ">= 80% of the loss", None if rs is None else round(100 * rs, 1), None if rs is None else rs >= 0.8)
    pop = lad["popularity_train_prevalence"]["map7_te"]
    c.add("G2", "H8", "popularity below the unweighted raw score", "pop < N raw", {"pop": round(pop, 4), "N_raw": round(N["raw"]["map7_te"], 4)}, pop < N["raw"]["map7_te"])
    d7 = N["per_label_iso_prior"]["map7_te"] - N["raw"]["map7_te"]
    c.add("G2", "H7", "per-label iso (prior) on the unweighted model", "|N plIsoPrior - N raw| <= 0.01", round(d7, 4), abs(d7) <= 0.01)
    if dep and "weighted" in dep:
        d6 = dep["weighted"]["per_label_iso_prior"]["map7"] - S["per_label_iso_prior"]["map7_te"]
        c.add("G2", "H6", "deployment vs test-split protocol (S, per-label iso prior)", "|diff| <= 0.01", round(d6, 4), abs(d6) <= 0.01)
    else:
        c.add("G2", "H6", "deployment vs test-split protocol (S, per-label iso prior)", "|diff| <= 0.01", None, None)


def gate_g3(c: Checks) -> None:
    for key, label in (("santander_mlp", "Santander MLP"), ("instacart_mlp", "Instacart MLP")):
        lad = load(f"{key}_ladder.json")
        tag = "H5s" if key.startswith("santander") else "H5i"
        if lad is None or S_MLP not in lad or N_MLP not in lad:
            for suf, what, crit in (("a", f"{label}: weights lower the raw score", "S raw < N raw"), ("b", f"{label}: saturation at exactly 1.0", "frac > 0"), ("c", f"{label}: inversion returns less than the repair", "elkan share < repair share"), ("d", f"{label}: repair share", ">= 50%")):
                c.add("G3", tag + suf, what, crit, None, None)
            continue
        S, N = lad[S_MLP], lad[N_MLP]
        c.add("G3", tag + "a", f"{label}: weights lower the raw score", "S raw < N raw", {"S_raw": round(S["raw"]["map7_te"], 4), "N_raw": round(N["raw"]["map7_te"], 4)}, S["raw"]["map7_te"] < N["raw"]["map7_te"])
        sat = S["saturation"]["frac_exact_1"]
        c.add("G3", tag + "b", f"{label}: saturation at exactly 1.0", "frac > 0", round(100 * sat, 2), sat > 0)
        es, rs = elkan_share(S, N), repair_share(S, N)
        c.add("G3", tag + "c", f"{label}: inversion returns less than the repair", "elkan share < repair share", None if es is None or rs is None else {"elkan": round(100 * es, 1), "repair": round(100 * rs, 1)}, None if es is None or rs is None else es < rs)
        c.add("G3", tag + "d", f"{label}: repair share", ">= 50%", None if rs is None else round(100 * rs, 1), None if rs is None else rs >= 0.5)
    for key, label in (("santander", "Santander"), ("instacart", "Instacart")):
        cs = load(f"{key}_capsweep.json")
        tag = "Ps" if key == "santander" else "Pi"
        caps = [] if cs is None else [r for r in cs["models"].values() if r.get("cap") is not None and "bound_check" in r]
        if not caps:
            c.add("G3", tag, f"{label}: |b_j| <= T*eta*c at every cap (prop:budget)", "all caps: frac(|b_prior| <= Tec + 0.1) == 1", None, None)
            continue
        frac = {f"{r['cap']:g}": r["bound_check"]["frac_labels_abs_b_prior_le_T_eta_c_plus_0.1"] for r in caps}
        excess = {f"{r['cap']:g}": round(r["bound_check"]["max_excess_b_prior"], 3) for r in caps}
        ok = all(v == 1.0 for v in frac.values())
        c.add("G3", tag, f"{label}: |b_j| <= T*eta*c at every cap (prop:budget)", "all caps: frac(|b_prior| <= Tec + 0.1) == 1", {"n_caps": len(caps), "frac_within": frac, "max_excess": excess}, ok)


# The verdicts are part of the released artifact, so they state what each outcome means
# for the paper's claims and nothing about the project's internal planning.
VERDICT = {
    ("G2", "PASS"): "G2 holds: on Instacart too the ideal odds shift is the minority of the loss, per-label isotonic regression with the prior fallback recovers most of it, and the weighted model ranks below popularity. The mechanism is carried by two datasets, not one.",
    ("G2", "FAIL"): "G2 fails: the shape of the decomposition is dataset-dependent. The claim must be narrowed to that, and a third large-scale dataset is needed before the mechanism can be stated in general.",
    ("G2", "NA"): "G2 not evaluated: instacart_ladder.json or instacart_whatif.json is missing.",
    ("G3", "PASS"): "G3 holds: the MLP reproduces the odds shift, the saturation and the failure of the analytic inversion, and prop:budget is not falsified at any cap. The mechanism is not specific to trees.",
    ("G3", "FAIL"): "G3 fails: if H5 falls, the mechanism must be restricted to boosted trees and the generality claim withdrawn; if the step-budget check falls, the proposition must be withdrawn and reported as an observation. Either outcome weakens the mechanism and belongs in the paper.",
    ("G3", "NA"): "G3 not evaluated: the MLP ladder or cap-sweep JSON is missing.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    c = Checks()
    gate_g2(c)
    gate_g3(c)
    g2 = c.status("G2", ("H4a", "H4b", "H8"))
    g3 = c.status("G3", ("H5sa", "H5sb", "H5sc", "H5ia", "H5ib", "H5ic", "Ps", "Pi"))
    if not args.quiet:
        print(f"{'gate':4} {'id':5} {'status':6} {'what':64} {'criterion':40} value")
        for r in c.rows:
            print(f"{r['gate']:4} {r['id']:5} {r['status']:6} {r['what'][:64]:64} {r['criterion'][:40]:40} {r['value']}")
        print()
    print(VERDICT[("G2", g2)])
    print(VERDICT[("G3", g3)])
    inputs = [f"{k}.json" for k in ("instacart_ladder", "instacart_whatif", "instacart_deploy", "santander_mlp_ladder", "instacart_mlp_ladder", "santander_capsweep", "instacart_capsweep")]
    out = {
        "G2": g2,
        "G3": g3,
        "checks": c.rows,
        "verdict": {"G2": VERDICT[("G2", g2)], "G3": VERDICT[("G3", g3)]},
        "_meta": {"generated_at": datetime.now(UTC).isoformat(), "inputs": {n: sha256(n) for n in inputs}, "core_ids": {"G2": ["H4a", "H4b", "H8"], "G3": ["H5sa", "H5sb", "H5sc", "H5ia", "H5ib", "H5ic", "Ps", "Pi"]}},
    }
    tmp = RES / "gate_check.json.partial"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    tmp.replace(RES / "gate_check.json")


if __name__ == "__main__":
    main()
