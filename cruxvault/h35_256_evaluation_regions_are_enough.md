---
id: h35
type: idea
schema: 3
title: 256 evaluation regions are enough
parent: q15
rule: 
measurement: 
replicates: rung 1 base and scrambled_covariates, 3 seeds each, and the four baselines, scored on a 1,024-region superset of the eic_v1 evaluation set; 2,000 cluster-bootstrap draws per size 64, 128, 256, 512, 1,024
created: "2026-11-13T12:42:24"
updated: "2026-11-13T12:42:24"
---

# h35 — 256 evaluation regions are enough

Parent:: [[q15_are_256_evaluation_regions_enough]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

yes: the region half-width of a paired composite difference at 256 is at most 1 seed spread, and resolved orderings hold in at least 95% of region draws; per-assay intervals are limited by the 3 validation cells, not by regions

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, the evaluation set keeps 256 regions; if it fails at 256 and holds at 512, the count rises to 512 (PI ruling 41).

## Idea / Hypothesis

256 evaluation regions are enough

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Deciding metric: region-bootstrap half-width; ordering stability; split-half seed spread.
- [ ] Settling rule: section D of Amendment 7 (keep 256; raise to 512; raise to 1024 needs the PI; keep 256, add seeds).

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h35/ and link at least the report:
     - [Report](results/h35/report.md)   - results/h35/curve.png -->
_(none yet)_
