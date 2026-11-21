---
id: h46
type: idea
schema: 3
title: Where FiLM injects the covariates (encoder only, decoder only, bottleneck only, every block including attention) moves correction beyond the seed range while imputation stays within range
parent: q75
rule: 
measurement: 
replicates: 3 seeds per arm; 16 pair regions to screen correction, 256 regions to confirm finalists; eic_v1_2; 4 placements x 3 models
created: "2026-11-21T04:04:16"
updated: "2026-11-21T04:04:16"
---

# h46 — Where FiLM injects the covariates (encoder only, decoder only, bottleneck only, every block including attention) moves correction beyond the seed range while imputation stays within range

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Decoder-only and all-block injection give correction at least the base value; encoder-only gives correction lower beyond the seed range on at least 2 of 3 models; imputation within range for every placement

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

Locates where covariate information must enter; a decoder-side or all-block win is a Pareto candidate.

## Idea / Hypothesis

Where FiLM injects the covariates (encoder only, decoder only, bottleneck only, every block including attention) moves correction beyond the seed range while imputation stays within range

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h46/ and link at least the report:
     - [Report](results/h46/report.md)   - results/h46/curve.png -->
_(none yet)_
