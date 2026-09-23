---
id: h15
type: idea
schema: 2
title: One transformation conditioned on the source and target covariates maps a track's base onto each of its counterfactual arms and back, in both count and −log10 p space, beyond per-arm quantile matching
parent: q1
status: idea
rule: all
measurement: CRPS gap-closed (D_QM − D_f)/(D_QM − D_oracle) on held-out chromosomes, pooled over arms, per mark class, counts (NB CRPS) and −log10 p (Gaussian CRPS) separately; QM = QuantileMatching fitted once per arm on training chromosomes; oracle = one pseudoreplicate of the target predicting the other; f is a covariate-conditioned model trained on base↔arm pairs of the t112 counterfactual corpus
replicates: 7 tracks (1 DNase, 3 narrow, 3 broad marks) x base↔arm pairs in both directions (19 arms per histone track, 7 per DNase track) x 3 seeds of f
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the swap check (C' = C returns X) gates the claim instead of acting as a control"
verdict: 
metric: 
created: "2026-09-23T14:40:11"
updated: "2026-09-23T14:40:27"
null_approved: "2026-09-23T14:40:27"
null_hash: 97c5dcf070e375f7
---

# h15 — One transformation conditioned on the source and target covariates maps a track's base onto each of its counterfactual arms and back, in both count and −log10 p space, beyond per-arm quantile matching

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If you know exactly how two versions of the same experiment were processed, one learned function can turn either version into the other — better than just re-scaling the numbers.

## TL;DR

We learn one f with X' = f(X | C, C'): X is a track, C the covariates it was made with, C' those of the wanted track. Pairs come from the counterfactual corpus of [[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]] — 7 tracks, each with a base and arms that change one processing knob — taken base → arm and arm → base. f is trained on all arms pooled and scored on held-out chromosomes against QuantileMatching (a monotone value-axis map fitted once per arm) and an oracle (one pseudoreplicate of the target predicting the other). Settled by CRPS gap-closed ≥ 0.5 in every mark class, in both output spaces, with the gain beyond 2× seed wobble, and by f collapsing when told a wrong C'.

## Null
Normalization: the covariates are ignored — f does no better than a covariate-free monotone rescale fitted per arm.

## Problem Statement

CANDI's zero-shot claims rest on its covariate conditioning doing real work, and inside CANDI that cannot be isolated. Here the truth is known: each pair differs in one recorded knob. Arm → arm pairs (two knobs at once) are left out on purpose (PI ruling 2026-09-23) and are the natural next test of whether f combines effects it saw one at a time.

## Idea / Hypothesis

One transformation conditioned on the source and target covariates maps a track's base onto each of its counterfactual arms and back, in both count and −log10 p space, beyond per-arm quantile matching

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] gapclosed_all: CRPS gap-closed on all bins >= 0.5 in every mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately; counts as failed wherever D_QM <= D_oracle
      fails-if:: per-arm QuantileMatching already closes the source-to-target gap, so conditioning on (C, C') adds nothing
      discriminates:: true
- [ ] seedgain_all: D_QM − D_f > 2 x f's seed wobble (max pairwise |Δ| of D_f over 3 seeds) in every mark class, counts and p separately
      fails-if:: f's gain over QuantileMatching is inside the wobble a seed change alone produces
- [ ] gapclosed_top1: on the top 1% of bins ranked by the real X', CRPS gap-closed >= 0.5 AND D_QM − D_f > 2 x seed wobble, in every mark class, counts and p separately
      fails-if:: f matches the bulk of the genome but not the peaks, where the knobs change the shape of the signal
- [ ] shufflecollapse: at scoring, C' replaced by the C' of another arm of the same track whose target differs (drawn separately for counts and p); CRPS gap-closed against the true X' <= 0.1 in every mark class
      fails-if:: f does not use C': its output does not change when told the wrong target covariates
- [ ] swapreturn: at scoring, C' set equal to C; CRPS gap-closed toward X (the source) >= 0.5 in every mark class, counts and p separately
      fails-if:: told the target is the source itself, f still transforms X, so its output is not steered by C' as the claim requires

## Planned Intervention

Distances D, all on held-out chromosomes, mean per track then macro:

| space | CRPS | Spearman |
|---|---|---|
| counts | NB CRPS: all bins, non-zero bins, top 1% by real X' | all, non-zero, top 1% |
| −log10 p | plain Gaussian CRPS: same three subsets | same three |

Reported, never gating: Spearman of f minus that of QM (QM keeps ranks, so a gain means f re-orders bins) against 2× seed wobble; CRPS on non-zero bins; gap-closed per arm; a point-output f (MAE/MSE — a point forecast's CRPS is its absolute error) against the distribution-output f.

Caveat: each pseudoreplicate holds half the reads, so the oracle is noisier than a full-depth repeat and gap-closed reads optimistic.

Five claim-directed checks under `all` at 80% power each give 33% joint power; the thresholds may not be loosened to compensate.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h15/ and link at least the report:
     - [Report](results/h15/report.md)   - results/h15/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
