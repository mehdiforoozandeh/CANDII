---
id: t118
type: task
title: learn the generator g(C, C') that outputs the transformation f (X' = f(X), f = g(C, C')) on the base↔arm pairs of the counterfactual corpus, with the QuantileMatching and pseudoreplicate-oracle rungs, and score the pre-registered checks
category: implementation
parent: 
blocked_by: None
refs: h15, q1
hypothesis_refs: 
status: open
created: 2026-09-23T14:40:48
updated: 2026-10-02T12:29:01
---

# t118 — learn the generator g(C, C') that outputs the transformation f (X' = f(X), f = g(C, C')) on the base↔arm pairs of the counterfactual corpus, with the QuantileMatching and pseudoreplicate-oracle rungs, and score the pre-registered checks

Refs:: [[h15_one_transformation_conditioned_on_the_so\|h15]], [[q1_do_the_recorded_experimental_covariates_\|q1]]

## Why

tests whether recorded covariates alone map one processing of a track onto another; its oracle rung needs the pseudoreplicates of t117 (on branch data-acquisition/t117-pseudoreplicates)

## What we want

The design was finalised 2026-09-25 and is recorded in `plan/T118_COUNTERFACTUAL_F.md`: g(C, C') outputs f, f(X) predicts X'; the competitor is the same model trained with scrambled covariates; never-trained arm → arm pairs test whether g learned how C and C' relate. g is trained in two versions, one per track (7) and one across all 7 tracks; no per-arm version. Done so far: the noSolution / QuantileMatching reference rungs (v2, Nibi `/project/def-maxwl/mforooz/t118/rungs_v2/`). Both versions of g decide pass/fail. Settled 2026-09-25 (plan §6): the pseudoreplicate oracle is dropped, the bars are set, and the architecture is ladder A–D (design E parked).

## Progress

- 2026-09-25: row 1 delivered. All 576 pre-registered runs (designs A–D, g reads C and C′) are trained, scored, law-tested and aggregated. Each design's report is linked from its hypothesis, and the main claim's report from [[h15_one_transformation_conditioned_on_the_so|the main claim]].
- 2026-10-01: row 2 (g also reads the bin's own value) and its shuffled-bin twin delivered under [[t120_build_and_run_row_2_of_the_g_x_f_gr|the row-2 build-and-run task]]. Exploratory; nothing is ticked and no verdict is recorded. Evidence: `results/row2/GRID_SUMMARY.md`. Run record: `plan/T118_STATUS.md`, from "Row 2 execution" to "Delivered 2026-10-01". Drafted interpretation: [[q4_how_much_capacity_does_a_covariate_condi|the capacity question]].
- Open:
  - a principled fix for the p-space explosion. In row 1, one or two pairs per affected run (each a sparse source mapped to the full-depth base) score CRPS from about 25 to about 10⁷ in −log10 p; row 2 has more such records. The PI rejected capping σ, and rejected keeping the runs with report-only. No task exists yet.
  - a hand check of the depth-law computation, before the unmet depth law is read as a model failure. No task exists yet.
  - the diagnosis of design D's explosion: D has no knots, so the cause found in B and C (a drifting σ at the lowest knot) does not explain it. No task exists yet.
  - [[t119_rebuild_the_two_dnase_mapq_arms_c12|the rebuild of the two DNase MAPQ arms]], so the MAPQ cut applies.
- Not marked done: the PI decides.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
