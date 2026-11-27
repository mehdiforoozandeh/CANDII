---
id: h47
type: idea
schema: 3
title: Training correction and same-view pairs on read-disjoint halves (source pr1, target pr2) removes the U-Net's input copy and lifts its correction to conv level without losing imputation
parent: q75
rule: 
measurement: 
replicates: 3 seeds per model; 16 pair regions to screen, 256 to confirm; eic_v1_2
created: "2026-11-21T05:03:19"
updated: "2026-11-21T05:03:19"
---

# h47 — Training correction and same-view pairs on read-disjoint halves (source pr1, target pr2) removes the U-Net's input copy and lifts its correction to conv level without losing imputation

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

U-Net: correction on crop pairs up by more than 2x its seed range (base 0.504, range 0.066), imputation within range of 0.432, copy gain falls from about 0.65 toward the conv models' 0.03-0.09; conv models: correction within range or up

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If training pairs that share reads are the cause of the U-Net trade-off (f140, the q45 and q46 readings), matching training to the disjoint-half scoring breaks the trade-off on the U-Net and may help every model.

## Idea / Hypothesis

Training correction and same-view pairs on read-disjoint halves (source pr1, target pr2) removes the U-Net's input copy and lifts its correction to conv level without losing imputation

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h47/ and link at least the report:
     - [Report](results/h47/report.md)   - results/h47/curve.png -->
_(none yet)_
