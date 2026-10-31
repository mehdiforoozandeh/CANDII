---
id: h13
type: idea
schema: 2
title: A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design
parent: q4
status: running
rule: all
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the shuffle and swap checks are reported for this rung and gate only the main claim"
measurement: "design C of the architecture ladder. f: convolve x with a kernel of 33 bins (825 bp), initialised to the identity and unconstrained, that g outputs, then apply the monotone curve of design B. The same kernel is used at every position. g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-17T21:47:47"
updated: "2026-09-25T02:03:08"
null_approved: "2026-09-25T01:30:29"
null_hash: 15406e39fd11a69f
lock: 9b12859990615e9d
locked: "2026-09-25T02:03:08"
lock_at: running
---

# h13 — A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

Letting each bin borrow from its neighbours lets the model widen or sharpen peaks, not only raise or lower them.

## TL;DR

Design C of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]]: one 33-bin kernel, then the 12-knot curve, per pair. Exploratory readings, no verdict. C beats its scrambled-covariate twin in 11 of 12 cells in each space. It beats design B in 5 of 12 count cells and 2 of 12 p cells. On never-trained arm → arm pairs it keeps more count gain than A or B. Design C2, where g also reads the bin value, is the one clear row-2 gain in counts (DNase), and it explodes in p.

## Expect

Design C (kernel, then monotone curve) beats its scrambled twin and beats design B, each by more than 2 × the larger seed wobble, per mark class, in both spaces, for both g versions.

_Pre-run: the PI accepted the bars on 2026-09-25 ("i accept the bars") and they were locked that morning, before the row-1 training array was submitted. Row 2 ran under the PI's ruling "don't worry about the claims for now" (2026-09-28)._

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

Adds global shape: broadening and smoothing (fragment extension, single- vs paired-end, read length) and sharpening. Beating [[h17_a_generator_g_c_c_that_outputs_a_per_bin|design B]] means some knob acts on shape, not only on level.

## Idea / Hypothesis

A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars set by the PI 2026-09-25 (defaults accepted: pass/fail is not the priority now). -->
- [ ] beatstwin: (twin = the no-covariates twin, (C, C') re-scrambled at every step) D_twin − D > 2 x the seed wobble of D (max pairwise |Δ| over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g; no further bar on the size of the gain (PI 2026-09-25)
      fails-if:: the covariates add nothing a scrambled-covariate twin of the same design cannot already do
      discriminates:: true
- [ ] lawtest: (against both twins: no-covariates and labels-as-ids) on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, this design beats each twin by more than 2 x seed wobble, for both versions of g. For depth → depth pairs the predicted count scale must follow the depth ratio (within 10% of the depth ratio; PI 2026-09-25)
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw
- [ ] beatsbelow: D of design B ([[h17_a_generator_g_c_c_that_outputs_a_per_bin|the per-bin monotone-curve design]]) − D of this design > 2 x the larger of the two seed wobbles, per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g
      fails-if:: the extra form adds nothing: design B already captures what the covariates do

## Planned Intervention

Run under [[t118_learn_the_covariate_conditioned_tra|the g → f ladder task]]; design in `plan/T118_COUNTERFACTUAL_F.md`. Seven tracks, 242 base ↔ arm pairs; chr22 validates, chr19 + chr21 score. Two versions of g (per track, across tracks), 3 seeds, both spaces, with the no-covariates and labels-as-ids twins. Row 2 ([[t120_build_and_run_row_2_of_the_g_x_f_gr|the row-2 task]], `plan/T118_ROW2_SPEC.md`) reruns it as C2.

## Run Links

- Row 1 aggregate: `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/C`; runs `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/C_*` (kit K3). Train array 22657297 (+ retry 22667089); law arrays 22667088, 22667090, 22667091, 22669098, 22676228; aggregation 22682907 (2026-09-25). Code 34e5705 (python), 82cf8db (slurm).
- Row 2: `nibi:/project/def-maxwl/mforooz/t118/row2/runs` (real, ids: kit K5; no-covariates: K6), shuffled-bin twin `nibi:/project/def-maxwl/mforooz/t118/row2_xshuf/runs` (K6), aggregate `nibi:/project/def-maxwl/mforooz/t118/row2/agg`. Arrays 22855414 (main), 22862752 (no-covariates), 22862753 (twin); job record in `plan/T118_STATUS.md`, "Row 2 execution".

## Artifacts

- [Report: design C, all 144 runs, drafted check readings](results/h13/report.md)
- [Checks: value against bar for every check, both versions of g](results/h13/checks_C.json)
- [Nibi paths](results/h13/FIR_PATH.txt)
- results/h13/figures/fig1_ladder.png CRPS by model, class and space
- results/h13/figures/fig3_law_grid.png law test: gain over each twin on never-trained pairs
- results/h13/figures/fig4_depth_law.png predicted count scale against the depth ratio
- results/h13/figures/fig5_learned_f.png the learned kernel and curve
- [Row 2, design C2: report](results/row2/C2/report.md)
- [Row 2, design C2: drafted checks, including against row 1 and the shuffled-bin twin](results/row2/checks_C2.json)
- results/row2/C2/figures/fig1_ladder.png row 2, C2: CRPS by model, class and space
- [Grid summary: rows 1 and 2, every cell with its seed wobble](results/row2/GRID_SUMMARY.md)
- [Trained versus never-trained pairs: depth law and every knob](results/h19/report.md)
- [Knob generalization report](results/t118/knob_generalization/report.pdf)
- [Lab deck: the g → f grid](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C ([deck note](results/row2/DECK.md))

## Findings

CRPS on all bins, one g per track, mean of 3 seeds ± seed wobble (largest difference between two seeds); split = oracle-scaled + scale error.

- Row 1, counts: DNase 0.818 ± 0.027 (0.802 + 0.016), no-covariates twin 1.83 ± 0.42 (1.30 + 0.52); narrow 0.379 ± 0.004 against B's 0.392 ± 0.005.
- Row 1, −log10 p: DNase 0.526 ± 0.035 against 0.767. Narrow is 0.502 ± 0.858 because one pair explodes (CRPS > 20); a principled fix is open ([[t124_find_a_principled_fix_for_the_log10|the p-space fix task]]).
- Never-trained arm → arm pairs: beats the no-covariates twin in 12 of 12 count and 4 of 12 p cells. The relative count gain over the twin is 0.35 on trained pairs and 0.19 on never-trained same-knob pairs ([[h19_g_fits_trained_base_arm_depth_pairs_but_|trained versus never-trained]]). Depth law unmet, 6 of 6.
- Row 2: C2 counts DNase 0.730 ± 0.003, against 0.821 for the twin that reads x from a random bin. C2 beats C in 2 of 12 count cells, both DNase with one g per track. In p, C2 explodes (DNase 1.6 × 10⁴; across-track records narrow / broad 7 / 6).
