---
id: h12
type: idea
schema: 2
title: A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space
parent: q4
status: running
rule: all
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the shuffle and swap checks are reported for this rung and gate only the main claim"
measurement: "design A of the architecture ladder. f: log μᵢ = a + b·xᵢ, x = log(1 + counts) or log(−log10 p), plus one dispersion value (NB n or log-normal σ). g is a small MLP on [C, C'] that outputs (a, b, dispersion). g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-17T21:47:47"
updated: "2026-09-25T02:03:08"
null_approved: "2026-09-25T01:30:29"
null_hash: 15406e39fd11a69f
lock: bbcce5f35c2388e9
locked: "2026-09-25T02:03:08"
lock_at: running
---

# h12 — A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

A stretch and a shift of the signal may turn one run into the other. In counts this was not enough until the model also saw each bin's own value.

## TL;DR

Design A, the lowest rung of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]]: one affine map in log space per pair. Exploratory readings, no verdict. It beats its scrambled-covariate twin in −log10 p (12 of 12 cells) but not in counts (3 of 12). Design A2, where g also reads the bin value, fixes counts and makes −log10 p explode. On never-trained arm → arm pairs, A gains nothing over the twin in counts.

## Expect

Design A (per-bin affine on the log scale) beats its scrambled-covariate twin by more than 2 × the seed wobble, per mark class, in counts and −log10 p, for both g versions.

_Pre-run: the PI accepted the bars on 2026-09-25 ("i accept the bars") and they were locked that morning, before the row-1 training array was submitted. Row 2 ran under the PI's ruling "don't worry about the claims for now" (2026-09-28)._

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

The simplest rung: it can express depth (a moves by the log depth ratio) and dynamic range (b), and nothing about shape. If it beats its twin, the covariates act partly as a rescale.

## Idea / Hypothesis

A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars set by the PI 2026-09-25 (defaults accepted: pass/fail is not the priority now). -->
- [ ] beatstwin: (twin = the no-covariates twin, (C, C') re-scrambled at every step) D_twin − D > 2 x the seed wobble of D (max pairwise |Δ| over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g; no further bar on the size of the gain (PI 2026-09-25)
      fails-if:: the covariates add nothing a scrambled-covariate twin of the same design cannot already do
      discriminates:: true
- [ ] lawtest: (against both twins: no-covariates and labels-as-ids) on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, this design beats each twin by more than 2 x seed wobble, for both versions of g. For depth → depth pairs the predicted count scale must follow the depth ratio (within 10% of the depth ratio; PI 2026-09-25)
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw

## Planned Intervention

Run under [[t118_learn_the_covariate_conditioned_tra|the g → f ladder task]]; design in `plan/T118_COUNTERFACTUAL_F.md`. Seven tracks, 242 base ↔ arm pairs; chr22 validates, chr19 + chr21 score. Two versions of g (per track, across tracks), 3 seeds, both spaces, with the no-covariates and labels-as-ids twins. Row 2 ([[t120_build_and_run_row_2_of_the_g_x_f_gr|the row-2 task]], `plan/T118_ROW2_SPEC.md`) reruns it as A2.

## Run Links

- Row 1 aggregate: `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/A`; runs `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/A_*` (kit K3). Train array 22657297 (+ retry 22667089); law arrays 22667088, 22667090, 22667091, 22669098, 22676228; aggregation 22682909 (2026-09-25). Code 34e5705 (python), 82cf8db (slurm).
- Row 2: `nibi:/project/def-maxwl/mforooz/t118/row2/runs` (real, ids: kit K5; no-covariates: K6), shuffled-bin twin `nibi:/project/def-maxwl/mforooz/t118/row2_xshuf/runs` (K6), aggregate `nibi:/project/def-maxwl/mforooz/t118/row2/agg`. Arrays 22855414 (main), 22862752 (no-covariates), 22862753 (twin); job record in `plan/T118_STATUS.md`, "Row 2 execution".

## Artifacts

- [Report: design A, all 144 runs, drafted check readings](results/h12/report.md)
- [Checks: value against bar for every check, both versions of g](results/h12/checks_A.json)
- [Nibi paths](results/h12/FIR_PATH.txt)
- results/h12/figures/fig1_ladder.png CRPS by model, class and space
- results/h12/figures/fig3_law_grid.png law test: gain over each twin on never-trained pairs
- results/h12/figures/fig4_depth_law.png predicted count scale against the depth ratio
- results/h12/figures/fig7_snippets.png 10 kb track snippets
- [Row 2, design A2: report](results/row2/A2/report.md)
- [Row 2, design A2: drafted checks, including against row 1 and the shuffled-bin twin](results/row2/checks_A2.json)
- results/row2/A2/figures/fig1_ladder.png row 2, A2: CRPS by model, class and space
- [Grid summary: rows 1 and 2, every cell with its seed wobble](results/row2/GRID_SUMMARY.md)
- [Trained versus never-trained pairs: depth law and every knob](results/h19/report.md)
- [Knob generalization report](results/t118/knob_generalization/report.pdf)
- [Lab deck: the g → f grid](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C ([deck note](results/row2/DECK.md))

## Findings

CRPS on all bins, one g per track, mean of 3 seeds ± seed wobble (largest difference between two seeds); split = oracle-scaled + scale error.

- Row 1, counts: DNase 3.06 ± 0.11 (1.36 + 1.70), twin 2.51 ± 0.23 (1.41 + 1.10); the loss is in scale. Broad 0.443 ± 0.003 against 0.507. Beats-twin met in 3 of 12 cells.
- Row 1, −log10 p: DNase 0.752 ± 0.018 against 0.876; met in 12 of 12. No exploding record (CRPS > 20).
- Never-trained arm → arm pairs: beats the no-covariates twin in 6 of 12 count and 4 of 12 p cells; relative count gain over the twin −0.03, as on trained pairs (−0.04) ([[h19_g_fits_trained_base_arm_depth_pairs_but_|trained versus never-trained]]). Depth law unmet, 6 of 6.
- Row 2, counts: A2 DNase 0.849 ± 0.009 (0.837 + 0.012), against 1.73 for its no-covariates twin and 3.02 for the twin that reads x from a random bin.
- Row 2, −log10 p: A2 explodes (narrow 436 ± 1 310, broad 3.7 × 10⁴), against the spec's prediction. A principled fix is open ([[t124_find_a_principled_fix_for_the_log10|the p-space fix task]]).
