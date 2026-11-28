---
id: h44
type: idea
schema: 3
title: A U-Net whose count skips are FiLM-modulated by (C, C') keeps the base U-Net imputation and reaches conv-level correction
parent: q75
rule: 
measurement: 
replicates: 3 seeds per arm; 16 pair regions to screen correction, 256 regions to confirm finalists; eic_v1_2
created: "2026-11-21T04:04:16"
updated: "2026-11-21T04:04:16"
---

# h44 — A U-Net whose count skips are FiLM-modulated by (C, C') keeps the base U-Net imputation and reaches conv-level correction

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Imputation within the base U-Net seed range (0.432, range 0.023); correction on crop pairs at least 0.68, i.e. within range of the conv autoencoder's 0.714 and above the base U-Net's 0.504 by more than 2x its range

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

Tests the skip-leak cause of the U-Net trade-off (q42); a pass makes the U-Net the Pareto candidate.

## Idea / Hypothesis

A U-Net whose count skips are FiLM-modulated by (C, C') keeps the base U-Net imputation and reaches conv-level correction

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h44/ and link at least the report:
     - [Report](results/h44/report.md)   - results/h44/curve.png -->
_(none yet)_
