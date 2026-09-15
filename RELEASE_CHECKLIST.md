# Release checklist

`scripts/export_public_release.py` validates this public tree, runs the release-safety scan and
maintains `SHA256SUMS.txt`. A future source repository can add a copy step around this validator;
until then, this repository is both the source and the public tree.

That is also the failure this list exists to prevent. A previous push shipped a tree that had not
been re-exported, so the published repository was a snapshot of an older paper while the source had
moved on. Nothing in the old checklist would have caught it, because no step ran the exporter.
**Step 4 is the one step that cannot be skipped.**

Commands are written for this repository (plain `python`, dependencies from `requirements.txt`).
**Every fenced block below starts at the repository root**; a `cd` inside a block applies to that
block only. The paper versions live under `publications/` (`arxiv/` canonical, `full-paper/` archived;
generated numbers and figures in `shared/`), the public package under
`src/oddslip/`, and the release validator under `scripts/export_public_release.py`.

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | Numbers gate | `python publications/shared/verify_numbers.py` | `OK: no hand-typed result numbers; all macros defined` |
| 2 | PDFs and arXiv bundle rebuild | `cd publications/arxiv && bash make_bundle.sh` | arXiv main 10 pages (twice), supplement 9 pages |
| 3 | Library tests | `python -m pytest tests/test_oddslip.py -q` | `4 passed` |
| 4 | **Release export is current** | `python scripts/export_public_release.py --check` | `[export_public_release] OK: ... manifest entries` |
| 5 | Release safety scan | `python scripts/check_release_safety.py --root . --verbose` | `[check_release_safety] OK: 0 matches ...` |
| 6 | No arrays committed | `git ls-files \| grep -E '\.(npy\|npz)$'` | no output |
| 7 | Manifest matches the tree | `sha256sum -c SHA256SUMS.txt \| grep -v ': OK$'` | no output |
| 8 | Fresh clone is the current export | clone, `sha256sum -c SHA256SUMS.txt`, then `diff` its manifest against the exported tree's | every line `OK`; `diff` silent |

Steps 1-3 run in the source repository before the export; steps 5-7 run here after it; step 8 runs
after the push. The counts below were measured on 2026-09-10 and move as result files are added —
the go/no-go is the exact line each step prints, not the count.

---

## 1. Numbers gate

```bash
python publications/shared/verify_numbers.py
```

Expected, as the last line, exit code 0:

```
OK: no hand-typed result numbers; all macros defined
```

**Do not run `make_tables.py` first.** `verify_numbers.py` runs it itself and compares
`figures/numbers.tex` before and after; running it by hand beforehand makes the staleness check pass
unconditionally and defeats the only step that notices a result file changed after the last build.
If it prints

```
[FAIL] figures/numbers.tex was stale; regenerated (rebuild the PDF)
```

the macros on disk are now correct but the PDFs are not — go to step 2 and then re-run step 4.

What this gate does and does not do. It checks three things: that `figures/numbers.tex` is what
`make_tables.py` emits from `artifacts/results/canonical/*.json` right now; that every `\nXxx` macro used in `main.tex`
and `supplement.tex` is defined; and that the prose of those two files, outside a generated table
body, contains no decimal with two or more places, no percentage and no thousands-separated integer,
apart from a short whitelist of design constants (split fractions, the CI level, hyper-parameter
values, the Instacart row and column counts) listed at the top of the script.

Two limits are worth knowing. **The pattern leaks on three shapes.** It matches `d.dd`, `d.d\%`,
`d\%` and `n,nnn`, so single-place decimals and plain integers are invisible to it — `main.tex`
still carries a hand-typed factor of `$1.2$` in the Limitations paragraph and the gate passes — and
a thousands separator written the LaTeX way escapes it as well, because the braces break the digit
run: `$500{,}000$` in Sec. 3 passes, as does the "10 code bases" of the Table S7 caption. All three
are hand-typed today and `REPRODUCE.md` lists the same three survivors. And in the **source**
repository — where steps 1-3 are run — both the macro check and the hand-typed-number check also cover
or undefined-macro line there names a file this tree does not contain.

It does not check that a macro was placed in the right sentence, that a table body says what its
caption says, or that the results themselves are right. A number moved from one claim to another
passes this gate.

## 2. Rebuild both PDFs and the arXiv bundle

```bash
cd publications/arxiv
bash make_bundle.sh
```

The name understates what it does. `make_bundle.sh` re-runs the numbers gate, builds each document
with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` and prints its page count — and then assembles the
arXiv upload: it writes `_bundle/` with `main.tex`, `main.bbl` and exactly the `figures/` files
`main.tex` pulls in — the two figure PDFs, the macro file `numbers.tex` and the two generated table
bodies — copies `supplement.pdf` to `_bundle/anc/`, compiles the bundle standalone as a
completeness test, and writes and hashes `arxiv_upload.tar.gz`. So `main.pdf` is compiled twice,
once in the working tree and once in the bundle, and three `Output written` lines appear:

```
Output written on main.pdf (10 pages, ...).
Output written on supplement.pdf (9 pages, ...).
bundled inputs:
<the five `figures/` files copied into _bundle/figures/>
Output written on main.pdf (10 pages, ...).
```

followed by a listing of `arxiv_upload.tar.gz` and its SHA-256. The byte sizes vary; the page
counts must not change without a reason you can name, and the two `main.pdf` counts must agree — if
the bundle compiles to a different length, `_bundle/` is missing an input. Then confirm there is
nothing left to resolve:

```bash
grep -nE "There were undefined|Rerun to get|Citation .* undefined|Reference .* undefined" publications/arxiv/main.log publications/arxiv/supplement.log
```

Expected: no output. Both logs also carry the usual under- and overfull-box warnings — around twenty
in `main.log`, half a dozen in `supplement.log` — and `main.log` alone reports

```
LaTeX Font Warning: Font shape `T1/zi4/m/it' undefined
LaTeX Font Warning: Some font shapes were not available, defaults substituted.
```

— a missing italic of the monospace face, and the end-of-run summary of that same substitution.
`supplement.log` carries neither. None of this is a go/no-go; the `grep` above is.

Build in the source repository, not here. `check_release_safety.py` scans `.log` and `.aux` files,
and `main.log` records absolute local paths, so a build left in this tree will fail step 5. The next
export deletes it anyway.

## 3. Library tests

```bash
python -m pytest tests/test_oddslip.py -q
```

Expected:

```
4 passed
```

The four tests cover the identities the paper leans on: MAP@K against hand-computed values, the
odds shift and its exact inversion (plus rank-neutrality of a common weight), the three dead-label
policies of the per-label calibrator with its monotonicity, and determinism of the cross-validated
calibrator selection.

## 4. Run the exporter

From the source repository:

```bash
python scripts/export_public_release.py
```

Expected, ending in exit code 0:

```
[check_release_safety] OK: 0 matches across <n> text file(s).
[export_public_release] wrote <n> manifest entries
```

`[OK] export clean` is the whole check. Without it, do not push. Three failures are worth naming
because each means something different:

- `missing sources:` — a file in the map no longer exists in the source. The map, not the tree, is
  what needs fixing.
- `[REWRITE] <file>: '<text>' found n times, expected m` — a shipped source file drifted and the
  thesis-layout path rewrite no longer applies where it used to. Update `REWRITES`; do not delete
  the rule.
- `[FORBIDDEN] <file>:<line>:` — an internal path or project token reached an exported text file.
  Fix the source file.

Push only from a tree the exporter has just written. If you edited anything here by hand, the edit
is lost at the next export and the two repositories now disagree.

## 5. Release safety scan

The exporter already ran this. Run it again standalone whenever anything in this tree was touched
after the export — including a local LaTeX build.

```bash
python scripts/check_release_safety.py --root . --verbose
```

Expected:

```
[check_release_safety] OK: 0 matches across <n> text file(s).
```

It looks for categories rather than for one person's details: e-mail addresses, Windows user-profile
and POSIX home paths, academic domains, ORCID iDs, and code-host user URLs. The author's own public
identifiers and the third-party hosts the paper cites are allowlisted, so a new match is genuinely
new.

## 6. No arrays committed

```bash
git ls-files | grep -E '\.(npy|npz)$'
git status --porcelain | grep -E '\.(npy|npz)$'
```

Expected: no output from either (`grep` exits 1).

**The check must cover `.npy` as well as `.npz`.** Most prediction arrays are `.npz`, but the
all-product Instacart MLP predictions are `.npy` memmaps — `p_te_all.npy` and `p_va_all.npy` for
each of the two arms, four of the 34 lines in `data/PRIMARY_ARRAYS.sha256` (`DATA.md` §3 lists the
same four). A check written for `.npz` alone would have passed them straight into the repository. `.gitignore` covers both extensions; this step confirms it held.

The arrays are deliberately not shipped. What ships is their SHA-256 in `data/PRIMARY_ARRAYS.sha256`
and in the `_meta.inputs` block of every result file that consumed them, the 32 `run_meta.json` files
of the matched-pair and MLP training runs (the two earlier-run array sources ship no `run_meta.json`
— see `DATA.md` §3), and the training scripts that regenerate them from the public Kaggle data. See
`DATA.md`.

## 7. Manifest matches the working tree

```bash
sha256sum -c SHA256SUMS.txt | grep -v ': OK$'
wc -l < SHA256SUMS.txt
```

Expected: no output from the first command. The second prints one line per tracked file except
`SHA256SUMS.txt` itself, so `git ls-files | wc -l` should be exactly one greater.

## 8. Fresh-clone verification

Run this after pushing and before telling anyone the repository exists. Steps 1-7 all run against
files the exporter wrote; only a clone tests what a reader actually receives.

```bash
git clone <remote-url> <clone-dir>
cd <clone-dir>
sha256sum -c SHA256SUMS.txt | grep -v ': OK$'
git ls-files | wc -l
```

Expected: no output from `sha256sum -c`, and a file count one greater than the manifest's line
count.

**Those two numbers are internal to the clone and cannot see a stale push.** The exporter wrote the
manifest over the same bytes that were committed, so an old export that was never re-exported still
verifies perfectly and still counts one more tracked file than manifest lines. To test that the
clone is the *current* export, compare it against the tree the exporter just wrote and confirm that
tree is fully pushed:

```bash
diff <clone-dir>/SHA256SUMS.txt <export-tree>/SHA256SUMS.txt
git -C <export-tree> status --porcelain
git -C <export-tree> rev-parse HEAD origin/main
```

Expected: no output from `diff`, no output from `git status --porcelain`, and the two revisions
identical. That comparison — not `sha256sum -c` — is what catches a stale push.

The clone-internal check is still worth running, because of line endings. The manifest is computed over LF bytes, and
`.gitattributes` (`* text=auto eol=lf`, with `.pdf`, `.png` and `.npz` marked binary) is the only
thing that makes a checkout reproduce them. This project has already had a SHA-256 manifest that
verified perfectly in the working tree and failed in a fresh clone because the checkout rewrote line
endings. Step 7 cannot see that; only step 8 can.

---

## Before the arXiv submission

The repository went public on 2026-09-12 (release v0.4.0, Zenodo DOI 10.5281/zenodo.22719149) ahead of the
arXiv submission, so that the URL in the paper resolves when the preprint appears. All of the following must
stay true.

- [ ] **The arXiv identifier is filled in.** — DONE 2026-09-15: arXiv:2609.13810 (v1 2026-09-12) in
      `CITATION.cff` and `README.md`. It appeared as a placeholder in three places:
      `CITATION.cff` (`message:` and `preferred-citation.notes:`) and `README.md` (the byline,
      "arXiv preprint (identifier to be added)"). Check with

      ```bash
      grep -rn "to be added" README.md CITATION.cff
      ```

      Expected at that moment: no output. Fill the identifier in the source repository and re-export
      (step 4); do not edit these files here.

- [ ] **The URL the paper prints resolves.** `main.tex` states that everything is produced by the
      `code/` and `publications/` directories of
      <https://github.com/souldrive7/odds-shift-slippage>. Open that URL in a logged-out browser
      and confirm it loads. A preprint that cites a private repository is worse than one that cites
      none.

- [ ] **The repository the URL reaches is the exported one.** Re-run step 8 against the public
      remote, logged out — including the `diff` of the clone's `SHA256SUMS.txt` against the freshly
      exported tree. That `diff` is the check that would have caught the stale push; `sha256sum -c`
      inside the clone would not have, because a stale export is internally consistent.

- [ ] **`LICENSE` is correct and scoped.** MIT, covering the code and documentation in this
      repository only, explicitly not the datasets.

- [ ] **Dataset terms are satisfied.** No source data is redistributed anywhere in the tree: the
      Santander raw file is referenced by URL and the Instacart files by the Kaggle dataset that
      mirrors them (`psparks/instacart-market-basket-analysis`; the original page no longer resolves
      — `DATA.md` §2), both pinned by SHA-256 — the Instacart raw hashes ship as
      `data/instacart/SHA256SUMS_raw.txt` — and the MULAN ARFF/XML files are fetched from upstream
      at run time with their content hashes recorded. Kaggle competition terms
      and MULAN's upstream terms apply to the data and are stated in `DATA.md`. Confirm step 6 is
      still clean and that `data/` contains only metadata.

- [ ] **What cannot be reproduced is stated, not hidden.** `DATA.md` §0 names the three Santander
      scorer columns whose training scripts are not shipped — the earlier run of the weighted
      configuration (`S_orig_lgbm_spw`), WGBoost (`W_wgboost`) and the earlier 100-tree unweighted
      LightGBM (`Nh_lgbm_unweighted_100`) — and points to what the paper records about each: Supp.
      Tables S4 and S2 give the configurations of the earlier weighted run and of the 100-tree run;
      for WGBoost, Supp. Table S1 records only the method and its citation, and no hyper-parameter of
      that run appears anywhere in this release. Re-read that section before publishing rather than
      after a reader asks.

- [ ] Tag the release, and optionally mint a Zenodo DOI from the tag. Tag only a commit that passed
      steps 1-8 in order. The order that keeps every identifier consistent is:
      1. `CITATION.cff` `version:` and `pyproject.toml` `version` agree (currently `0.4.1`); the tag is `v` + that version.
      2. `git tag -a v0.4.1 -m "arXiv version" && git push origin v0.4.1` (after the final push of the branch).
      3. Make the repository public; confirm the CI workflow (`.github/workflows/ci.yml`) is green on `main`.
      4. Enable the repository on Zenodo and create a GitHub release from the tag; Zenodo reads `.zenodo.json`
         and mints the DOI. Paste the DOI into `CITATION.cff` (`identifiers:`) and `README.md`.
      5. Submit to arXiv (`publications/arxiv/arxiv_upload.tar.gz`) — DONE 2026-09-12 as submit/8069900 from
         release v0.4.1 (bundle sha256 28584d04...); identifier assigned as arXiv:2609.13810 (cs.IR, cross-list
         cs.LG; DOI 10.48550/arXiv.2609.13810), filled in `CITATION.cff`, `README.md` and `.zenodo.json` on 2026-09-15.