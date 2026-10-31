---
id: h17
type: idea
schema: 2
title: A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design
parent: q4
status: running
rule: all
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the shuffle and swap checks are reported for this rung and gate only the main claim"
measurement: "design B of the architecture ladder. f: a monotone spline on the log scale (12 knots at fixed training quantiles of x; g outputs positive steps) applied to every bin, plus dispersion as a function of level. g is a small MLP on [C, C'] that outputs the knots and the dispersion curve. Same function class as QuantileMatching, but chosen from C, C' instead of fitted on X'. g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-25T01:26:25"
updated: "2026-09-25T02:03:08"
null_approved: "2026-09-25T01:30:29"
null_hash: 15406e39fd11a69f
lock: e1d6dc947b754ebe
locked: "2026-09-25T02:03:08"
lock_at: running
---

# h17 — A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

A bendable curve instead of a straight stretch lets weak and strong signal change by different amounts.

## TL;DR

Design B of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]]: one monotone 12-knot curve in log space per pair. Exploratory readings, no verdict. In counts B beats its scrambled-covariate twin (11 of 12 cells) and design A (9 of 12). In −log10 p a few pairs explode. On never-trained arm → arm pairs most of the count gain is lost. Design B2, where g also reads the bin value, changes counts little and explodes more in p.

## Expect

Design B (per-bin monotone curve) beats its scrambled twin and beats design A, each by more than 2 × the larger seed wobble, per mark class, in both spaces, for both g versions.

_Pre-run: the PI accepted the bars on 2026-09-25 ("i accept the bars") and they were locked that morning, before the row-1 training array was submitted. Row 2 ran under the PI's ruling "don't worry about the claims for now" (2026-09-28)._

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

Still magnitude only: beating [[h12_a_covariate_conditioned_affine_map_on_th|design A]] means the knobs change weak and strong bins differently. B has QuantileMatching's function class, chosen from C, C' instead of fitted on X'.

Why B exists (PI, 2026-09-25): one-warp "is not using C and C' at all", so B uses the same function class but chooses the curve from C and C′. It measures how much of one-warp's gain the covariates alone reproduce.

## Idea / Hypothesis

A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars set by the PI 2026-09-25 (defaults accepted: pass/fail is not the priority now). -->
- [ ] beatstwin: (twin = the no-covariates twin, (C, C') re-scrambled at every step) D_twin − D > 2 x the seed wobble of D (max pairwise |Δ| over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g; no further bar on the size of the gain (PI 2026-09-25)
      fails-if:: the covariates add nothing a scrambled-covariate twin of the same design cannot already do
      discriminates:: true
- [ ] lawtest: (against both twins: no-covariates and labels-as-ids) on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, this design beats each twin by more than 2 x seed wobble, for both versions of g. For depth → depth pairs the predicted count scale must follow the depth ratio (within 10% of the depth ratio; PI 2026-09-25)
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw
- [ ] beatsbelow: D of design A ([[h12_a_covariate_conditioned_affine_map_on_th|the per-bin affine design]]) − D of this design > 2 x the larger of the two seed wobbles, per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g
      fails-if:: the extra form adds nothing: design A already captures what the covariates do

## Planned Intervention

Run under [[t118_learn_the_covariate_conditioned_tra|the g → f ladder task]]; design in `plan/T118_COUNTERFACTUAL_F.md`. Seven tracks, 242 base ↔ arm pairs; chr22 validates, chr19 + chr21 score. Two versions of g (per track, across tracks), 3 seeds, both spaces, with the no-covariates and labels-as-ids twins. Row 2 ([[t120_build_and_run_row_2_of_the_g_x_f_gr|the row-2 task]], `plan/T118_ROW2_SPEC.md`) reruns it as B2.

## Run Links

- Row 1 aggregate: `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/B`; runs `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/B_*` (kit K3). Train array 22657297 (+ retry 22667089); law arrays 22667088, 22667090, 22667091, 22669098, 22676228; aggregation 22682910 (2026-09-25). Code 34e5705 (python), 82cf8db (slurm).
- Row 2: `nibi:/project/def-maxwl/mforooz/t118/row2/runs` (real, ids: kit K5; no-covariates: K6), shuffled-bin twin `nibi:/project/def-maxwl/mforooz/t118/row2_xshuf/runs` (K6), aggregate `nibi:/project/def-maxwl/mforooz/t118/row2/agg`. Arrays 22855414 (main), 22862752 (no-covariates), 22862753 (twin); job record in `plan/T118_STATUS.md`, "Row 2 execution".

## Artifacts

- [Report: design B, all 144 runs, drafted check readings](results/h17/report.md)
- [Checks: value against bar for every check, both versions of g](results/h17/checks_B.json)
- [Nibi paths](results/h17/FIR_PATH.txt)
- results/h17/figures/fig1_ladder.png CRPS by model, class and space
- results/h17/figures/fig3_law_grid.png law test: gain over each twin on never-trained pairs
- results/h17/figures/fig4_depth_law.png predicted count scale against the depth ratio
- results/h17/figures/fig5_learned_f.png g's curve per arm over QuantileMatching's per-pair curve
- [Row 2, design B2: report](results/row2/B2/report.md)
- [Row 2, design B2: drafted checks, including against row 1 and the shuffled-bin twin](results/row2/checks_B2.json)
- results/row2/B2/figures/fig1_ladder.png row 2, B2: CRPS by model, class and space
- [Grid summary: rows 1 and 2, every cell with its seed wobble](results/row2/GRID_SUMMARY.md)
- [Trained versus never-trained pairs: depth law and every knob](results/h19/report.md)
- [Knob generalization report](results/t118/knob_generalization/report.pdf)
- [Lab deck: the g → f grid](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C ([deck note](results/row2/DECK.md))

## Findings

CRPS on all bins, one g per track, mean of 3 seeds ± seed wobble (largest difference between two seeds); split = oracle-scaled + scale error.

- Row 1, counts: DNase 0.824 ± 0.004 (0.816 + 0.007), no-covariates twin 1.54 ± 0.02 (1.17 + 0.37); narrow 0.392 against 0.497. QuantileMatching scores 0.886 on DNase (its table has no split).
- Row 1, −log10 p: DNase 0.541 ± 0.111 against 0.744; met in 7 of 12. With one g across tracks, DNase is 24.9 ± 62.7: on a sparse source mapped to the full-depth base, the lowest knot's σ drifts (`plan/T118_STATUS.md`). A principled fix is open ([[t124_find_a_principled_fix_for_the_log10|the p-space fix task]]).
- Never-trained arm → arm pairs: beats the no-covariates twin in 8 of 12 count and 1 of 12 p cells. The relative count gain over the twin falls from 0.29 (trained) to 0.06 (never-trained, same knob) ([[h19_g_fits_trained_base_arm_depth_pairs_but_|trained versus never-trained]]). Depth law unmet, 6 of 6.
- Row 2: B2 counts DNase 0.817 ± 0.008; it beats B in 0 of 12. Only B2 meets the depth law anywhere (1 of 6 cells). In p, exploding records rise (narrow / broad 3 / 4, against 1 / 0 in row 1).
