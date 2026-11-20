---
id: h43
type: idea
schema: 3
title: "On all 256 regions, the leave-one-out average input keeps its place in the U-Net mix-and-match pool: better beyond the seed range on one goal term and worse beyond on none"
parent: q50
rule: 
measurement: select_rule.py arm-summary on the p256_sv scores of abl_average_input_average_input_* against their references, then mix_arms.py qualify, applied by tools/onion_eval/apply_rescoring_rules.py (rule 2.5 of the settled miniCANDI design)
replicates: 3 seeds per model
created: "2026-11-18T16:21:51"
updated: "2026-11-18T16:21:51"
---

# h43 — On all 256 regions, the leave-one-out average input keeps its place in the U-Net mix-and-match pool: better beyond the seed range on one goal term and worse beyond on none

Parent:: [[q50_does_the_leave_one_out_average_of_the_tr]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

U-Net, 3 seeds, all 256 regions: imputation above the reference beyond the larger seed range (16-region basis: 0.432 to 0.466), and none of imputation, crop-pair correction and calibration lower beyond the range, so the rung4 average_input line of mix_arms.py qualify reads qualifies.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, average input stays a U-Net switch for the mix-and-match round (row 13, pre-written rule 2.5); if it fails, it leaves the pool and stays off on all three models.

## Idea / Hypothesis

On all 256 regions, the leave-one-out average input keeps its place in the U-Net mix-and-match pool: better beyond the seed range on one goal term and worse beyond on none

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h43/ and link at least the report:
     - [Report](results/h43/report.md)   - results/h43/curve.png -->
_(none yet)_
