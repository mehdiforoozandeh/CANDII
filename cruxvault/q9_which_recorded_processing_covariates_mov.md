---
id: q9
type: question
schema: 2
title: Which recorded processing covariates move a track the most when changed one at a time, in peak height and in the shape of the signal?
parent: q1
status: open
stale: false
created: "2026-10-26T12:57:52"
updated: "2026-10-26T12:57:52"
---

# q9 — Which recorded processing covariates move a track the most when changed one at a time, in peak height and in the shape of the signal?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

We re-processed the same experiments while changing one lab or software setting at a time, and asked which settings change the data the most: how tall the peaks are, and where the signal falls.

## TL;DR

Ranks the ten counterfactual-arm knobs by how far each moves a track from its base, in counts and −log10 p. Depth leads on peak height, the antibody-efficiency proxy on shape; the control and fragment-extension knobs move least. The middle group (duplicate removal, run type, read length, mapping-quality filter) re-draws its 30M-read subsample, so it is read against the resampling floor.

## Motivation

The PI asked on 2026-10-26: "among all covariates in the counterfactual augmented data, which ones have more effect on the signal. what's the ranking?" The PI kept it small: "this is a simple exploratory experiment. we don't want an over-engineered experiment here", and set the order: settle the metrics first, then rank. Peak height is |log2 fold change| over the top 1% of bins; shape is 1 − Spearman on 1 kb bins (summed to damp counting noise); FRiP was the PI's addition, on each track's own top-1% bins. After the first pass, the PI chose to re-score on peaks the base and the arm share, to remove the bias of base-chosen peaks.

## Question

Which recorded processing covariates move a track the most when changed one at a time, in peak height and in the shape of the signal?

## Protocol

Exploratory (PI 2026-10-26): readings only, no bars, no verdicts. Metrics chosen before the run: peak height = |log2 fold-change| of the mean signal over the base's top-1% 25 bp bins; shape = 1 − Spearman between base and arm on 1 kb bins; FRiP (counts only) = the fraction of a product's reads in its own top-1% bins. chr19 + chr21, blacklist removed. Per mark class (DNase, narrow, broad), the median over tracks; each knob ranked by its strongest level. The p-only knobs get p scores only; the two DNase mapping-quality arms (identical to the base) are dropped.

## Answer so far

First pass, 192 base → arm comparisons ([[t133_rank_the_t112_processing_knobs_by_h|the knob-ranking task]]). Histone marks, counts, strongest level, median over tracks:

- Peak height: depth at 3.75M reads 3.0 (narrow) and 3.0 (broad) log2; antibody proxy at 90% control reads 1.9 / 1.6; read length 36 1.3 / 2.0; mapping-quality, duplicate removal and run type 0.3–1.0.
- Shape: antibody proxy 0.53 / 0.58; depth 3.75M 0.49 / 0.49; the four FASTQ re-run knobs 0.35–0.45. Depth 15M, whose reads are a subset of the base's, reads 0.17.
- −log10 p: the control-scaling, control-identity, control-depth and fragment-extension knobs read 0.04–0.17 on shape, against 0.40–0.74 for the read-side knobs; control depth is the weakest knob.
- DNase: duplicate removal off raises the peaks 2.6 log2 and FRiP by 0.53; paired-end raises them 1.9 log2 and FRiP by 0.34.
- Doubts: (1) the FASTQ re-run arms re-draw the 30M subsample, so their shape scores share a resampling floor near that of depth 7.5M (0.32–0.34); (2) regression to the mean on base-chosen peaks (resolved below); (3) the same selection likely inflates FRiP at low depth; (4) the DNase FRiP gain of 0.53 is unchecked.
- Shared-peak re-score, 2026-10-30, not judged: a shared peak set removes doubt (2); peak-height rank 1 holds, values fall a median 13 %. Against the floor at the arm's depth, peak height is above it in 27 of 27 counts rows (33–763×), shape in 3 of 27 (DNase depth 3.75M, DNase paired-end, broad antibody proxy f0.9; floor seed spread ≤ 0.002). Caveat: a knob arm re-uses the base's reads, the floor does not, so "below the floor" is not "no effect". The join used the first-pass floor; the DNase depth 3.75M row may drop.

## Run Links

- Nibi: `/project/def-maxwl/mforooz/t118/knob_effects/out/` (job 25063638, code 5f95c43 on `exp/t118-counterfactual-f`).
- Shared-peak re-score: Nibi `/project/def-maxwl/mforooz/t139/out/` (job 25389617, kit `/project/def-maxwl/mforooz/t137/code/K1`, 1f634e1).

## Artifacts

- [Knob-ranking digest](results/t133/DIGEST.md)
- [Shared-peak re-score digest](results/t139/DIGEST.md), [ranking](results/t139/ranking.tsv), [knob vs floor](results/t139/knob_vs_floor.tsv)
- [Ranking](results/t133/ranking.tsv), [per level](results/t133/levels.tsv), [per pair](results/t133/pairs.tsv), [run path](results/t133/FIR_PATH.txt)
- [Deck](presentations/q9-knob-ranking/index.html) — published https://claude.ai/artifact/1T1NKMYo5taJjDowse9r2f; vault copy [DECK.md](presentations/q9-knob-ranking/DECK.md)

<!-- crux:ledger:start -->
**1 children** · open 0, leaning 0, settled 1, parked 0 · 1 finding(s) on this question

- `q11` _(Q)_ [[q11_where_along_the_genome_does_each_covaria|Where along the genome does each covariate act: does a track react uniformly to a change in depth, read length or another processing knob, or more in some kinds of regions?]] — *settled · strong*
- `f24` finding (strong) [[f24_peak_height_clears_the_floor_by_a_wide_m|Peak height clears the floor by a wide margin; the shape ranking does not,…]]
<!-- crux:ledger:end -->
