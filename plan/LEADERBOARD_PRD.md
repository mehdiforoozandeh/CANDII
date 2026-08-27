# LEADERBOARD_PRD — the rivals leaderboard (t58)

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
| headline board | P2 genome-wide, with method-class badges, both leakage caveats printed, and a strict P2-minus-chr19 toggle |
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

## 3. Boards

Three boards, three tabs. They score different data and are **never mixed in one table**.

| board | eval set | rows |
|---|---|---|
| **Main** (headline) | P2 — genome-wide, 23 chromosomes, declared eval pairs of `regime.eic_val` | CANDI versions, retrained rivals, naive baselines |
| **Dev** | P1 — chr21, same declared pairs | same rows; the fast board new CANDI versions hit first |
| **Entrants** | Dataset-3, Max's vendored 001 scorer | the 23 EIC entrant submissions + CANDI + baselines |

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
3. **Strict toggle**: one click rescores the Main board on P2 minus chr19. CANDI is then fully
   position-out-of-sample while rivals keep their advantage — CANDI's strict number is
   conservative by construction.

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

Per board, per view (default / strict):

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
  boards.json            # per board: eval-set id, frozen store/regime hashes, caveat text
  rows/<board>/<method>@<version>.json   # one stamped row each (source of truth)
  releases/v<N>.json     # frozen snapshots of compiled boards
site → built to _site/ by the Action:  index.html + app.js + style.css + leaderboard.json
```

A row json carries: method, version tag, date, method-class badges, per-metric values, and a
**mandatory provenance block** — git SHA of the scoring code, score-json path in
`cruxvault/results/`, `FIR_PATH`, store manifest hash, regime file, σ-table id (B1a rows),
scorer version, and flags of record (`--crps-approx k` etc.). A row missing any provenance
field does not build.

`tools/leaderboard.py`, three verbs:

- `add <score.json> --board <b> --method <m> --version <v> …` — validate against the schema
  and the board's frozen hashes, extract registry metrics, write the row. Refuses NaN,
  missing tracks not declared with `--allow-missing`, and provenance gaps.
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
- **Paired lax/strict columns** (EvalPlus) — the CRPS split, and default vs strict view.
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
- B pairs (`regime.eic_test.json`) — untouched, as ruled in the rivals plan.
- Elo / pairwise ranking, cost columns, non-static hosting.

## 11. Milestones

| # | ships | done when |
|---|---|---|
| M1 | registry, row schema, `leaderboard.py add/build/check`, tests | `pytest tests/ -q` green; synthetic fixture board compiles deterministically |
| M2 | static site rendering `leaderboard.json` | all §8 elements render from the fixture; no external requests |
| M3 | Pages workflow | the fixture board is live on the Pages URL |
| M4 | real rows stamped | P1 board full; P2 board with every method whose genome-wide scores have landed; entrants board from t54 |

M4 rows come only from score jsons already recorded under `cruxvault/results/` — stamping
invents no numbers. Methods whose P2 runs are still in flight (Lavawizard, and any rescore in
progress) join the board when their jsons land, through the same `add` gate.
