---
id: h38
type: idea
schema: 3
title: The gap between depth-only scaling and the floor is largest for the depth and run-type knobs and near 0 for pure subset pairs
parent: q19
rule: 
measurement: "Per covariate knob on the eic_v1 evaluation set (1,480 correction pairs, all 256 pair regions): CRPS of depth-only scaling, of the half-against-half floor, and their gap with intervals over cells."
replicates: 
created: "2026-11-14T00:35:30"
updated: "2026-11-14T00:35:30"
---

# h38 — The gap between depth-only scaling and the floor is largest for the depth and run-type knobs and near 0 for pure subset pairs

Parent:: [[q19_what_do_the_training_free_baselines_reac]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

The gap between depth-only scaling and the floor is largest for the depth and run-type knobs and near 0 for pure subset pairs.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

Names the covariate knobs on which a model has room for correction, and the ones where depth-only scaling already reaches the floor.

## Idea / Hypothesis

The gap between depth-only scaling and the floor is largest for the depth and run-type knobs and near 0 for pure subset pairs

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] The depth and run-type knobs have the two largest gaps.
- [ ] The gap of pure subset pairs is within its interval of 0.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h38/ and link at least the report:
     - [Report](results/h38/report.md)   - results/h38/curve.png -->
_(none yet)_
