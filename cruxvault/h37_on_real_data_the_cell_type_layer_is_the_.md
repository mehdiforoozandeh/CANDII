---
id: h37
type: idea
schema: 3
title: On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks
parent: q18
rule: 
measurement: "The onion layers per assay from the eic_v1 layers file (job 27164943): differences of consecutive pair values along the replicate ladder, with intervals."
replicates: 
created: "2026-11-14T00:35:30"
updated: "2026-11-14T00:35:30"
---

# h37 — On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, imputation of narrow marks and accessibility is mainly a cell-type problem, and broad marks at 25 bp are bounded by sampling noise.

## Idea / Hypothesis

On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Per narrow-mark and accessibility assay, the cell-type layer exceeds every other layer beyond its interval.
- [ ] Per broad-mark assay at 25 bp, the sampling layer exceeds every other layer beyond its interval.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h37/ and link at least the report:
     - [Report](results/h37/report.md)   - results/h37/curve.png -->
_(none yet)_
