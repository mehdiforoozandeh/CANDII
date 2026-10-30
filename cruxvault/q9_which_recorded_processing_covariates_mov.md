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

Ranks the ten counterfactual-arm knobs by how far each one moves a track from its base, separately for raw counts and −log10 p. First reading: depth leads on peak height, the antibody-efficiency proxy leads on shape, and the four control and fragment-extension knobs move the least. The middle group (duplicate removal, run type, read length, mapping-quality filter) is not yet resolved, because each of those arms also re-draws its 30M-read subsample. A noise floor for that re-draw would settle the middle of the ranking.

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
- Doubts: (1) the FASTQ re-run arms re-draw the 30M subsample, so their shape scores share a resampling floor near that of depth 7.5M (0.32–0.34); (2) peaks are chosen on the base, so any re-drawn arm reads lower by regression to the mean (duplicate removal on broad marks −0.9 log2, where about 0 is expected); (3) the same selection likely inflates FRiP at low depth; (4) the DNase FRiP gain of 0.53 is unchecked.

## Run Links

- Nibi: `/project/def-maxwl/mforooz/t118/knob_effects/out/` (job 25063638, code 5f95c43 on `exp/t118-counterfactual-f`).

## Artifacts

- [Knob-ranking digest](results/t133/DIGEST.md)
- [Ranking](results/t133/ranking.tsv), [per level](results/t133/levels.tsv), [per pair](results/t133/pairs.tsv), [run path](results/t133/FIR_PATH.txt)
- [Deck](presentations/q9-knob-ranking/index.html) — published https://claude.ai/artifact/1T1NKMYo5taJjDowse9r2f; vault copy [DECK.md](presentations/q9-knob-ranking/DECK.md)

<!-- crux:ledger:start -->
**1 children** · ideas 0/0 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0) · sub-questions 0/1 resolved

- `q11` _(Q)_ [[q11_where_along_the_genome_does_each_covaria|Where along the genome does each covariate act: does a track react uniformly to a change in depth, read length or another processing knob, or more in some kinds of regions?]] — *open*
<!-- crux:ledger:end -->
