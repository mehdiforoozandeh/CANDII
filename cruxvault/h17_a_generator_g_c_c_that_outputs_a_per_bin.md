---
id: h17
type: idea
schema: 2
title: A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design
parent: q4
status: idea
rule: all
measurement: "design B of the architecture ladder. f: a monotone spline on the log scale (12 knots at fixed training quantiles of x; g outputs positive steps) applied to every bin, plus dispersion as a function of level. g is a small MLP on [C, C'] that outputs the knots and the dispersion curve. Same function class as QuantileMatching, but chosen from C, C' instead of fitted on X'. g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-25T01:26:25"
updated: "2026-09-25T01:30:29"
null_approved: "2026-09-25T01:30:29"
null_hash: 15406e39fd11a69f
---

# h17 — A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

A bendable curve instead of a straight stretch lets weak and strong signal change by different amounts.

## TL;DR

Design B of the architecture ladder (rewritten 2026-09-25 for the g(C, C') → f design; the earlier version compared against a one-warp map and an oracle, both gone). f: a monotone spline on the log scale (12 knots at fixed training quantiles of x; g outputs positive steps) applied to every bin, plus dispersion as a function of level. g is a small MLP on [C, C'] that outputs the knots and the dispersion curve. Same function class as QuantileMatching, but chosen from C, C' instead of fitted on X'. Still magnitude only. Beating the affine design means the knobs change weak and strong bins differently. Its gap to QuantileMatching (reported, never gating) shows how much of a per-pair fit the covariates alone recover. Settled when it beats its own scrambled-covariate twin beyond seed wobble on held-out chromosomes and on never-trained arm → arm pairs, for both versions of g. It must also beat design A, the per-bin affine design, the rung below it.

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

Still magnitude only. Beating the affine design means the knobs change weak and strong bins differently. Its gap to QuantileMatching (reported, never gating) shows how much of a per-pair fit the covariates alone recover.

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

Design record: `plan/T118_COUNTERFACTUAL_F.md` (Architecture ladder). Shuffle and swap checks, Spearman, CRPS on non-zero bins, per-arm results and the gap to QuantileMatching are reported for this design; the pass/fail shuffle and swap checks live in the main claim.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h17/ and link at least the report:
     - [Report](results/h17/report.md)   - results/h17/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
