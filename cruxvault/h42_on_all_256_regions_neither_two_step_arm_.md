---
id: h42
type: idea
schema: 3
title: On all 256 regions, neither two-step arm (native or fixed base depth) raises the composite of any of the three models by twice the seed range while keeping every goal term within twice the range
parent: q51
rule: 
measurement: select_rule.py arm-summary on the p256_sv scores of abl_two_step_two_step_native_* and abl_two_step_two_step_fixed_* against their references, applied by tools/onion_eval/apply_rescoring_rules.py (rule 2.3 of the settled miniCANDI design)
replicates: 3 seeds per arm and model
created: "2026-11-18T16:21:37"
updated: "2026-11-18T16:21:37"
---

# h42 — On all 256 regions, neither two-step arm (native or fixed base depth) raises the composite of any of the three models by twice the seed range while keeping every goal term within twice the range

Parent:: [[q51_does_two_step_correction_through_a_base_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

On the conv autoencoder, conv + attention and U-Net, 3 seeds each, all 256 regions: no two-step arm has composite (version 3) up by more than twice the larger seed range with imputation, crop-pair correction and calibration all within twice the range. Basis: on 16 regions no two-step arm raised the composite, and native routing lowered crop-pair correction beyond the range on both conv models.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, two-step routing stays off in the settled design (row 24, pre-written rule 2.3); if an arm passes, two-step enters the mix-and-match pool for that model and the PI decides.

## Idea / Hypothesis

On all 256 regions, neither two-step arm (native or fixed base depth) raises the composite of any of the three models by twice the seed range while keeping every goal term within twice the range

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h42/ and link at least the report:
     - [Report](results/h42/report.md)   - results/h42/curve.png -->
_(none yet)_
