---
id: h14
type: idea
schema: 2
title: A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design
parent: q4
status: running
rule: all
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the shuffle and swap checks are reported for this rung and gate only the main claim"
measurement: "design D of the architecture ladder. f: a few dilated convolution layers (4 layers × 32 channels, kernel 5, dilations 1, 2, 4, 8: 61 bins ≈ 1.5 kb); g outputs a per-channel scale and shift for each layer (feature-wise modulation). g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-17T21:47:47"
updated: "2026-09-25T02:03:08"
null_approved: "2026-09-25T01:30:29"
null_hash: 15406e39fd11a69f
lock: afafe6eed1a8bd29
locked: "2026-09-25T02:03:08"
lock_at: running
---

# h14 — A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

A small network can reshape peaks differently from background, where a single kernel treats every place the same.

## TL;DR

Design D, the top rung of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]]: a small dilated CNN whose layers g scales and shifts. Exploratory readings, no verdict. In counts, with one g per track D scores best of the four; it beats its scrambled-covariate twin in 11 of 12 cells, but beats design C in only 1 of 12. In −log10 p it explodes on some pairs, cause unknown, and beats its twin in 3 of 12 cells. Design D2 (g also reads the bin value) adds nothing in counts.

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

Shape that depends on local context. Beating [[h13_a_small_covariate_conditioned_convolutio|design C]] means a knob's effect depends on what the signal around a bin looks like.

## Idea / Hypothesis

A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars set by the PI 2026-09-25 (defaults accepted: pass/fail is not the priority now). -->
- [ ] beatstwin: (twin = the no-covariates twin, (C, C') re-scrambled at every step) D_twin − D > 2 x the seed wobble of D (max pairwise |Δ| over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g; no further bar on the size of the gain (PI 2026-09-25)
      fails-if:: the covariates add nothing a scrambled-covariate twin of the same design cannot already do
      discriminates:: true
- [ ] lawtest: (against both twins: no-covariates and labels-as-ids) on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, this design beats each twin by more than 2 x seed wobble, for both versions of g. For depth → depth pairs the predicted count scale must follow the depth ratio (within 10% of the depth ratio; PI 2026-09-25)
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw
- [ ] beatsbelow: D of design C ([[h13_a_small_covariate_conditioned_convolutio|the kernel-and-curve design]]) − D of this design > 2 x the larger of the two seed wobbles, per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g
      fails-if:: the extra form adds nothing: design C already captures what the covariates do

## Planned Intervention

Run under [[t118_learn_the_covariate_conditioned_tra|the g → f ladder task]]; design in `plan/T118_COUNTERFACTUAL_F.md`. Seven tracks, 242 base ↔ arm pairs; chr22 validates, chr19 + chr21 score. Two versions of g (per track, across tracks), 3 seeds, both spaces, with the no-covariates and labels-as-ids twins. Row 2 ([[t120_build_and_run_row_2_of_the_g_x_f_gr|the row-2 task]], `plan/T118_ROW2_SPEC.md`) reruns it as D2.

## Run Links

- Row 1 aggregate: `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/D`; runs `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/D_*` (kit K3). Train array 22657297 (+ retry 22667089); law arrays 22667088, 22667090, 22667091, 22669098, 22676228; aggregation 22682908 (2026-09-25). Code 34e5705 (python), 82cf8db (slurm).
- Row 2: `nibi:/project/def-maxwl/mforooz/t118/row2/runs` (real, ids: kit K5; no-covariates: K6), shuffled-bin twin `nibi:/project/def-maxwl/mforooz/t118/row2_xshuf/runs` (K6), aggregate `nibi:/project/def-maxwl/mforooz/t118/row2/agg`. Arrays 22855414 (main), 22862752 (no-covariates), 22862753 (twin); job record in `plan/T118_STATUS.md`, "Row 2 execution".

## Artifacts

- [Report: design D, all 144 runs, drafted check readings](results/h14/report.md)
- [Checks: value against bar for every check, both versions of g](results/h14/checks_D.json)
- [Nibi paths](results/h14/FIR_PATH.txt)
- results/h14/figures/fig1_ladder.png CRPS by model, class and space
- results/h14/figures/fig3_law_grid.png law test: gain over each twin on never-trained pairs
- results/h14/figures/fig4_depth_law.png predicted count scale against the depth ratio
- results/h14/figures/fig5_learned_f.png the learned f made visible
- [Row 2, design D2: report](results/row2/D2/report.md)
- [Row 2, design D2: drafted checks, including against row 1 and the shuffled-bin twin](results/row2/checks_D2.json)
- results/row2/D2/figures/fig1_ladder.png row 2, D2: CRPS by model, class and space
- [Grid summary: rows 1 and 2, every cell with its seed wobble](results/row2/GRID_SUMMARY.md)
- [Trained versus never-trained pairs: depth law and every knob](results/h19/report.md)
- [Knob generalization report](results/t118/knob_generalization/report.pdf)
- [Lab deck: the g → f grid](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C ([deck note](results/row2/DECK.md))

## Findings

CRPS on all bins, one g per track, mean of 3 seeds ± seed wobble (largest difference between two seeds); split = oracle-scaled + scale error.

- Row 1, counts: DNase 0.772 ± 0.036 (0.760 + 0.012), no-covariates twin 1.59 ± 0.02 (1.27 + 0.32); narrow 0.375 ± 0.004 against C's 0.379 ± 0.004.
- Row 1, −log10 p: DNase 1.9 × 10⁴ ± 4.9 × 10⁴ (mostly scale error); its twin explodes too. D has no knots, so the lowest-knot σ cause found in B and C does not apply. The diagnosis ([[t123_diagnose_design_d_s_log10_p_explosi|the D explosion task]]) and a principled fix ([[t124_find_a_principled_fix_for_the_log10|the p-space fix task]]) have no recorded result.
- Never-trained arm → arm pairs: beats the no-covariates twin in 9 of 12 count and 1 of 12 p cells. The relative count gain over the twin is 0.33 on trained and 0.23 on never-trained same-knob pairs, the largest never-trained gain of the four ([[h19_g_fits_trained_base_arm_depth_pairs_but_|trained versus never-trained]]). Depth law unmet, 6 of 6.
- Row 2: D2 counts DNase 0.749 ± 0.038; it beats D in no cell. In p, D2 narrow is 2.8 × 10⁶ ± 7.8 × 10⁶.
- Follow-up: [[h22_a_composable_g_one_shared_map_from_each_|the composable g]] rebuilds this rung as an encoder-decoder.
