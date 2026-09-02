# LEADERBOARD_PRD — the rivals leaderboard (t58)

> **Superseded in part by `plan/BENCHMARK_DESIGN.md` (t77), and applied to the board by t82.**
> Where the two disagree, BENCHMARK_DESIGN wins. What it replaced, and where:
>
> | this PRD said | BENCHMARK_DESIGN says | applied by |
> |---|---|---|
> | three boards `main` / `dev` / `entrants` (§3) | two regimes, `eic.19→20,21,22` and `eic.pilot→20,21,22`, plus the anchor block (§3, §6, §9) | t82 |
> | protocols P1 / P2 / P3 (§1, §3) | retired. Eval scope is one thing for every regime, so the regime id plus the `scope` field carry it (§9) | t82 |
> | Dataset 2 / Dataset 3 (§3) | retired. `truth: store (ENCODE4 2020)` and `truth: challenge (2019)`, a toggle on the page rather than a board (§6, §9) | t82 |
> | the headline board is P2 genome-wide (§1) | the ranked number is `held-out` — chr20+21+22. `genome-wide` is reported for comparability and never ranked (§4) | t80, t82 |
> | one score slot per method per board | three numbers per trainable method per regime: `V_` breadth, `V_` matched (never ranked), `B_` (§5.2) | t80, t82 |
> | ranks always computed (§5.4) | nothing ranks until the noise floor on the new panels is measured; rows go up **unranked** before then (§15) | t82, t86 |
> | the rows stamped at M4 are the board | every one of those rows is **void** — the old board scored chr19, the chromosome CANDI trained on (§3.3) | t82 |
>
> §5.1 (the metric registry), §5.4 (the composite arithmetic), §6 (stdlib only), §7 (provenance),
> §8 (rendering) and §9 (publishing) are unchanged and still govern.

Product requirements for the CANDI rivals leaderboard. The decisions in §1 were settled with
the PI in a quiz session on 2026-08-27. Execute them; do not re-open them here. Everything
else in this document is design that implements those decisions and is open to review on the
t58 PR.

The board is the public face of the rivals program (`plan/RIVALS_PLAN.md`). It is also the
instrument the experiment-lane merge gate is waiting for (`CLAUDE.md` → "Experiment lanes —
TODO"). This PRD builds the instrument. **Defining the gate itself stays a separate, PI-gated
ruling** — nothing here creates one.

## 1. Locked decisions

| decision | ruling |
|---|---|
| composite score | mean of category sub-scores; a category sub-score is the mean rank of the method inside that category |
| headline board | ~~P2 genome-wide~~ — superseded: the ranked number is `held-out`, chr20+21+22, per regime (BENCHMARK_DESIGN §4). Method-class badges and both leakage caveats still print |
| hosting | GitHub Pages, rebuilt by a GitHub Action when rows land on `main` |
| how rows enter | `tools/leaderboard.py add` stamps one score json into a row; full provenance is mandatory; scoring itself stays a manual Fir run |

## 2. Purpose

One page that answers three questions:

1. **Where does CANDI stand** against retrained rivals (Avocado, ChromImpute, eDICE,
   Lavawizard), naive baselines, and the EIC entrants — on one composite number and on
   category boards.
2. **Is each gap real** — every number carries its noise floor, and gaps under the floor never
   decide a rank.
3. **Is CANDI climbing** — each CANDI version is a dated row; the versions draw a line over
   time above the table.

## 3. Boards — **superseded by BENCHMARK_DESIGN §3, §6 and §9**

Two regimes, two tabs, plus one block underneath that is not a tab. The regimes differ only in
the loci each method's transferable parameters were fit on, and both score the same chromosomes.

| container | what it is | rows |
|---|---|---|
| **`eic.19`** (primary) | `eic.19→20,21,22` — fit on chr19, scored on chr20+21+22 | CANDI, the four retrained rivals, the five naive baselines |
| **`eic.pilot`** (ablation) | `eic.pilot→20,21,22` — fit on the 44 hg38 ENCODE Pilot Regions, minus their overlap with the eval chromosomes | the same ten methods |
| **anchor block** | the 2019 field. It carries no regime, because we never trained it, so it sits under the ranked table rather than in it | the 23 challenge submissions plus `Average` and `Avocado_p0` |

Deferred placeholders, named so the axis exists: `eic.gw→20,21,22`, `merged.*`.

The retired ids `main` / `dev` / `entrants` appear nowhere in `leaderboard/`, and neither do
P1 / P2 / P3 or Dataset 2 / Dataset 3. A row is addressed by method, regime, truth, panel,
scope and metric; if any field is unknown, the row does not go in the ranked table.

Sub-boards (views within a board, not tabs): one per metric category (§5), plus a
**CANDI-lineage board** for the covariate diagnostics (§5.3).

## 4. The fairness contract

What the PI's train-test-leakage question settled, made mechanical.

**Equal and checkable.** Every ranked method trained on the same corpus store, with the
declared eval tracks excluded from training. Same 25 bp grid, same scorer, same store version.
A row records the store manifest hash and the regime file; the build refuses a row that does
not match the board's frozen eval-set hash.

**Not equalizable — disclosed instead.** Position-parameterized rivals (Avocado, ChromImpute)
train on every genomic position by construction; CANDI trained on chr19 positions only. No
choice of eval chromosome removes this asymmetry. The board handles it three ways:

1. **Badges** on every row: `position-transductive` / `position-generalizing`, and
   `zero-shot cell types` / `retrained per setting`.
2. **Caveats printed on the Main board**, both directions: CANDI is position-in-sample on
   chr19; the transductive rivals are position-in-sample everywhere.
3. ~~**No P2-minus-chr19 view.**~~ **Superseded by BENCHMARK_DESIGN §3.3 and §4.** The
   asymmetry is now answered by the design rather than disclosed: chr19 is a training
   chromosome and is not in the ranked scope at all. The ranked number is chr20+21+22, where
   no method's transferable parameters were fit. `genome-wide` stays as a second, never-ranked
   aggregation, and it is blank for any method fit at every position — a blanked cell was
   never computed.

## 5. Metrics, categories, and the composite

### 5.1 The metric registry

One file, `leaderboard/registry.json`, is the single source of truth. Per metric key
(`EVAL.md` names): its category, its direction (higher-better or lower-better), its arm
(`pval` or `count`), its noise floor if one is measured, and its display format. The build
reads the registry; nothing about a metric is hard-coded in the front end.

Categories at launch:

| category | metrics (pval arm unless noted) | in composite? |
|---|---|---|
| **pointwise** | `mse`, `gwcorr`, `gwspear`, `mse1obs` | yes |
| **distributional** | `crps` with its `pit_ks` + `coverage_95` companions | yes |
| **peaks** | `auprc` | only if every ranked row has it |
| **count arm** | `crps` + `crps_oracle_scaled` + `scale_error`, `nb_nll` | no — sub-board (§5.2) |
| **covariate diagnostics** | `covuse`, `covshare`, `depthdir`, `depthcounterfact`, `covspec`, `depthblind`+`biokeep` | no — CANDI lineage only (§5.3) |

Quoting rules ride along: count-arm `crps` never renders without its
`oracle_scaled`/`scale_error` split (paired columns under one header); `depthblind` never
renders without `biokeep`; pval Gaussian CRPS always renders with `pit_ks` and `coverage_95`.

### 5.2 Missing-arm policy

B1b says pval-only rivals have no count arm, and no counts are invented. Rule: **a category
enters the composite only when every ranked row on that board has all of its metrics.** The
pval arm is the shared core — every method has it. The count arm becomes a sub-board ranking
only the methods that emit counts. Loss-tier NLLs never enter the composite: different
distribution families make them incomparable across methods; they show per-family only.

This rule is deterministic and self-updating: if a future rival emits counts, the count-arm
sub-board grows a row; the composite does not move.

### 5.3 Covariate diagnostics

The C block interrogates a decoder that is conditioned on covariates. Rivals have no such
decoder, so these metrics cannot rank methods. They rank **CANDI versions against each other**
— the covariate-sensitivity sub-board is the CANDI-lineage board, and it feeds the climb
story, not the cross-method composite.

### 5.4 The composite, exactly

Per board:

1. For each in-composite metric: rank the rows by the registry direction. Two rows whose gap
   is under the metric's noise floor are **tied** and share a rank range. A metric with no
   measured floor uses plain ranks.
2. Category sub-score = mean rank across the category's metrics (ties propagate as
   rank intervals).
3. Composite = unweighted mean of category sub-scores.
4. Displayed rank is a **spread**: best and worst achievable rank given the tie intervals
   ("1–3", not "1"), the LMArena pattern.

Floors at launch: count-arm macro CRPS ~0.09 target-clustered (`AGENTS.md` §7.2); the
pval-arm floor comes from t57 (`floors.json`) when it lands — until then pval CRPS ranks
plain and its deltas are not greyed, stated in the legend. CANDI's own version-over-version
delta arrows use the stricter seed-alone bar, pooled CRPS 0.1195.

## 6. Data model and CLI

```
leaderboard/
  registry.json          # §5.1 — metrics, categories, directions, floors
  boards.json            # the regimes, the anchor block, and §1's truth/panel/scope/marker
                         #   vocabularies; per container: eval-set id, frozen hashes, caveats
  rows/<regime>/<truth>.<panel>.<scope>/<method>@<version>.json   # one stamped row each
  anchor/<truth>.<panel>.<scope>/<method>@<version>.json          # the 2019 field, no regime
  void/<former board>/<method>@<version>.json                     # §3.3, kept, never numbered
  releases/v<N>.json     # frozen snapshots of compiled boards
site → built to _site/ by the Action:  index.html + app.js + style.css + leaderboard.json
```

The path **is** the address (BENCHMARK_DESIGN §1). `add` has no default for truth, panel or
scope, and it checks each against `boards.json`'s own vocabulary, so a row whose address does
not resolve has nowhere on disk to live. That is what makes "if any field is unknown, the row
does not go in the ranked table" a property of the tree rather than a line in a document.

A row json carries: method, version tag, date, method-class badges, per-metric values, and a
**mandatory provenance block** — git SHA of the scoring code, score-json path in
`cruxvault/results/`, `FIR_PATH`, store manifest hash, regime file, σ-table id (B1a rows),
scorer version, and flags of record (`--crps-approx k` etc.). A row missing any provenance
field does not build.

`tools/leaderboard.py`, three verbs:

- `add <score.json> --board <b> --truth <t> --panel <p> --scope <s> --method <m>
  --version <v> …` — validate against the schema, the address vocabulary and the container's
  frozen hashes, extract registry metrics, write the row. Refuses NaN, missing tracks not
  declared with `--allow-missing`, provenance gaps, an unresolvable address, a count or peak
  metric under challenge truth, and an unknown row marker.
- `build` — compile every row into `leaderboard.json` + the static site. Pure function of the
  repo tree: same input, bit-identical output (sorted keys, no timestamps).
- `check` — re-run every gate on the existing rows, rebuild, and diff bit-exact. CI runs this.

`freeze` (post-launch): copy the compiled boards to `releases/v<N>.json`; the site's version
picker serves frozen releases; `latest` is an alias, the HELM pattern.

## 7. Reliability checks

1. **Schema gate** — a malformed or NaN-bearing score json fails `add` loudly; nothing renders
   as a silent blank.
2. **Provenance required** — no SHA, no FIR_PATH, no σ-table id (where applicable) → no row.
3. **Frozen eval-set hash** — a row scored on a different store or regime than the board
   declares is refused; nobody quietly scores on easier data.
4. **Deterministic rebuild** — `leaderboard.py check` in CI diffs the compiled output
   bit-exact, the `tools/golden.py` idea applied to the board.
5. **Noise-floor stamping** — floors live in the registry and render in the cells
   ("0.412 ± 0.09" style); sub-floor deltas grey out with a "~" prefix; ranks are spreads.
6. **Tests** — `tests/test_leaderboard.py`: schema round-trip, composite arithmetic on a
   synthetic fixture (including tie propagation and the missing-arm rule), determinism of
   `build`, and a render smoke test that the site loads `leaderboard.json`.

The leaderboard touches `tools/`, `leaderboard/`, and `.github/workflows/` only — never
`src/candi/` — so `tools/golden.py check` stays trivially bit-exact and t58 merges on the
normal non-experiment gate.

## 8. Front end

Static HTML + vanilla JS + inline SVG. No framework, no chart library, no external requests.
Elements borrowed from the leaderboard survey (2026-08-27):

- **Rank spreads with floor-driven ties** (LMArena) — §5.4.
- **Score ± floor printed in the cell** (Terminal-Bench).
- **Paired lax/strict columns** (EvalPlus) — the CRPS split.
- **Climb chart** (Papers with Code) — composite vs date; CANDI versions as labeled points on
  a line, baselines as flat dotted lines, a ± floor band shaded around the leader.
- **Version chips + delta arrows** — `candi @ 2026-08-14 · a3f1c2`; arrows grey under the
  0.1195 self-comparison bar.
- **Per-row provenance links + verified badge** (SWE-bench) — every row links its score json,
  crux task, and FIR path; the badge renders only when `check` resolved the artifacts.
- **Method-class badges** (Papers with Code "extra training data" tag, generalized) — §4.
- **Reproducibility footer** (Open LLM Leaderboard) — the exact `candi.bench.external`
  command, scoring-code SHA, and data-freeze date.
- **Category column groups behind toggle chips**; medal tints only when the gap to the next
  row clears the floor.
- **Frozen releases + latest alias** (HELM).

Not borrowed: Elo machinery, parameter sliders, cost axes — wrong tools at ten rows.

## 9. Publishing

GitHub Actions workflow: on push to `main` touching `leaderboard/` or `tools/leaderboard.py`,
run `leaderboard.py check`, then `build`, then deploy `_site/` to GitHub Pages. The board is
therefore always the latest `main`, by construction. **Note of record:** Pages makes the page
public even while the repo is private; the PI accepted this in the quiz session.

## 10. Out of scope for t58

- The experiment-lane merge gate definition — PI ruling, raised separately once the board is
  live.
- Automated scoring in CI — scoring stays on Fir; only compilation and rendering are
  automated.
- ~~B pairs untouched~~ — superseded: `B_` is a first-class panel beside `V_`, predicted exactly once from the checkpoint selected on `V_` (BENCHMARK_DESIGN §5).
- Elo / pairwise ranking, cost columns, non-static hosting.

## 11. Milestones

| # | ships | done when |
|---|---|---|
| M1 | registry, row schema, `leaderboard.py add/build/check`, tests | `pytest tests/ -q` green; synthetic fixture board compiles deterministically |
| M2 | static site rendering `leaderboard.json` | all §8 elements render from the fixture; no external requests |
| M3 | Pages workflow | the fixture board is live on the Pages URL |
| ~~M4~~ | ~~real rows stamped~~ | **void — BENCHMARK_DESIGN §3.3.** Every M4 row scored chr19, the chromosome CANDI trained on. The files are kept in `leaderboard/void/`, named and dated, and the compiler drops their numbers on the way into the payload. |

| M5 | the regime board | the §9 naming applied across `boards.json`, `help.json`, `app.js`, `index.html` and this PRD; the anchor block, the truth toggle and the two row markers render; both regimes are unfrozen and refuse every `add` until the retrains land (t82) |

Rows come only from score jsons already recorded under `cruxvault/results/` — stamping invents
no numbers. Nothing may be stamped under either regime until its `frozen` hashes are set
against the rebuilt store manifest and the new regime json, which is part of stamping the
first retrained row.
