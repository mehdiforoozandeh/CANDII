---
id: h12
type: idea
schema: 2
title: A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space
parent: q4
status: idea
rule: all
measurement: "design A of the architecture ladder. f: log μᵢ = a + b·xᵢ, x = log(1 + counts) or log(−log10 p), plus one dispersion value (NB n or log-normal σ). g is a small MLP on [C, C'] that outputs (a, b, dispersion). g(C, C') outputs f, f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; both versions of g (one per track, 7; one across all 7 tracks) decide pass/fail; pairs, split, covariates and scoring as in plan/T118_COUNTERFACTUAL_F.md (base↔arm both ways, chr22 validates, chr19 + chr21 score, never-trained arm→arm law test); competitor = this design trained with C and C' scrambled across pairs"
replicates: 7 tracks x base↔arm pairs in both directions (246 pairs; DNase MAPQ arms excluded) x 3 seeds of each g (7 per-track g's and 1 across-track g), and of each twin
verdict: 
metric: 
created: "2026-09-17T21:47:47"
updated: "2026-09-25T12:00:00"
---

# h12 — A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

Once both runs' settings are known, a stretch and a shift of the signal may be enough to turn one run into the other.

## TL;DR

Design A of the architecture ladder (rewritten 2026-09-25 for the g(C, C') → f design; the earlier version compared against a one-warp map and an oracle, both gone). f: log μᵢ = a + b·xᵢ, x = log(1 + counts) or log(−log10 p), plus one dispersion value (NB n or log-normal σ). g is a small MLP on [C, C'] that outputs (a, b, dispersion). The simplest rung. It can express depth (a moves by the log depth ratio) and dynamic range (b), and nothing about shape. If it already beats its twin, the covariates act at least partly as a rescale. Settled when it beats its own scrambled-covariate twin beyond seed wobble on held-out chromosomes and on never-trained arm → arm pairs, for both versions of g. Compared with QuantileMatching and noSolution as references only.

## Null

Normalization: the covariates are ignored — this design does no better than the same design trained with C and C' scrambled across pairs.

## Problem Statement

The simplest rung. It can express depth (a moves by the log depth ratio) and dynamic range (b), and nothing about shape. If it already beats its twin, the covariates act at least partly as a rescale.

## Idea / Hypothesis

A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars marked TODO(PI) are not set; CLAUDE.md forbids inventing a gate. -->
- [ ] beatstwin: D_twin − D > 2 x the seed wobble of D (max pairwise |Δ| over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21, for both versions of g. Further bar on the size of the gain: TODO(PI)
      fails-if:: the covariates add nothing a scrambled-covariate twin of the same design cannot already do
      discriminates:: true
- [ ] lawtest: on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, this design beats its scrambled twin by more than 2 x seed wobble, for both versions of g; bar: TODO(PI). For depth → depth pairs the predicted count scale must follow the depth ratio (tolerance TODO(PI))
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw

## Planned Intervention

Design record: `plan/T118_COUNTERFACTUAL_F.md` (Architecture ladder). Shuffle and swap checks, Spearman, CRPS on non-zero bins, per-arm results and the gap to QuantileMatching are reported for this design; the pass/fail shuffle and swap checks live in the main claim.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h12/ and link at least the report:
     - [Report](results/h12/report.md)   - results/h12/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
