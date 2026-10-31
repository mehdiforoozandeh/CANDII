---
id: h24
type: idea
schema: 2
title: The depth entries in g's covariate input cause the conditioned generator to miss the depth law on never-trained depth pairs
parent: q7
status: idea
rule: 
measurement: worst-pair |predicted/true depth ratio − 1| on never-trained depth → depth pairs, depth-offset form of design D, with the depth entries of g's input removed vs kept
replicates: 7 tracks × 3 seeds, both g versions
verdict: 
metric: 
created: "2026-10-25T00:22:33"
updated: "2026-10-25T00:58:40"
null_approved: "2026-10-25T00:58:40"
null_hash: 55c193fc0a71623a
---

# h24 — The depth entries in g's covariate input cause the conditioned generator to miss the depth law on never-trained depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The model gets the read depth from a fixed rule and also from a learned part, and the learned part may be what spoils the rule.

## TL;DR

In the depth-offset grid, f removes the source depth before the transform and adds the target depth after it, so a model that ignores the covariates obeys the depth law by construction. On never-trained depth pairs, the covariate-conditioned design D gives a worst-pair |ratio − 1| of 0.87–1.45, and its no-covariates twin gives 0.05–0.14 with the same f and training pairs; so f and the pairs are not the cause. The claim: the depth entries of g's covariate input cause the miss. The test removes those entries from g's input, keeps the offset and every other covariate, and retrains D. It is supported if the worst-pair |ratio − 1| is ≤ 0.10 on all seeds and refuted if it stays above 0.5 (both bars the PI's to confirm). A pass is not guaranteed: without depth in its input, g sees base ↔ depth-arm training pairs as pairs with equal covariates and different targets, so it can still learn a non-identity transform for equal covariates that fights the offset.

## Expect

With the depth entries removed from g's input (offset and other covariates kept), retrained D reaches a worst-pair |ratio − 1| of at most 0.10 on all seeds; it is refuted if that stays above 0.5.

_Written on 2026-10-25, after the depth-offset run that motivates it but before its own test, which has not been built. The agent drafted the bars and marked them as the PI's to confirm; the PI approved the batch without confirming them individually._

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Chance: removing the depth entries moves the never-trained miss by less than the seed wobble (0.05–0.16) over 7 tracks × 3 seeds.

## Problem Statement

In the depth-offset grid the no-covariates twin of design D comes near the depth law (worst pair 0.05–0.14) while the covariate-conditioned D misses it (0.87–1.45), with the same f and the same training pairs.

## Idea / Hypothesis

The depth entries in g's covariate input cause the conditioned generator to miss the depth law on never-trained depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

Not built. It reuses the depth-offset count model of [[t125_build_and_run_the_depth_offset_coun|the depth-offset task]] (design D, offset kept) with the depth entries dropped from g's input. Design: `plan/ETA_OFFSET_AND_H_DESIGN.md`.

## Run Links

_(none yet)_

## Artifacts

- [Depth-offset grid, part 1 depth law](results/t125/GRID_SUMMARY_part1.md), [depth-law table](results/t125/depth_law.tsv)

## Findings

Not run. The motivating readings are in the depth-offset grid: real D 0.869–1.449, D and D2 twins 0.054–0.135, against the 0.10 allowance.
