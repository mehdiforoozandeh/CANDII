---
id: h45
type: idea
schema: 3
title: Pre and post CAC layers in the testbed g/f form (g reads only covariates and outputs the parameters of a count map f; f before the network removes the C artifact, f after the NB re-applies the C' artifact) raise correction without lowering imputation or calibration
parent: q75
rule: 
measurement: 
replicates: 3 seeds per arm; 16 pair regions to screen correction, 256 regions to confirm finalists; eic_v1_2; f = monotone curve (default) or conditioned CNN; core FiLM on (default) or off; 3 models
created: "2026-11-21T04:04:16"
updated: "2026-11-21T04:04:16"
---

# h45 — Pre and post CAC layers in the testbed g/f form (g reads only covariates and outputs the parameters of a count map f; f before the network removes the C artifact, f after the NB re-applies the C' artifact) raise correction without lowering imputation or calibration

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

On at least 2 of 3 models, correction on crop pairs up beyond the seed range with imputation and calibration within range; monotone-curve f at least as good as conditioned-CNN f; core-FiLM-off arm keeps more than half of the core-on correction gain

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

A pass shows the covariate artifact can be factored out of the core network, a Pareto candidate on any model.

## Idea / Hypothesis

Pre and post CAC layers in the testbed g/f form (g reads only covariates and outputs the parameters of a count map f; f before the network removes the C artifact, f after the NB re-applies the C' artifact) raise correction without lowering imputation or calibration

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h45/ and link at least the report:
     - [Report](results/h45/report.md)   - results/h45/curve.png -->
_(none yet)_
