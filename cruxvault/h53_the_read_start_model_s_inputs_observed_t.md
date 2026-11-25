---
id: h53
type: idea
schema: 3
title: The read-start model's inputs (observed tracks, average-signal input, input transform, covariates, masking) were built differently from the reference by mistake
parent: q81
rule: 
measurement: 
replicates: 
created: "2026-11-25T00:25:43"
updated: "2026-11-25T00:25:43"
---

# h53 — The read-start model's inputs (observed tracks, average-signal input, input transform, covariates, masking) were built differently from the reference by mistake

Parent:: [[q81_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

A code audit finds an unintended difference between the read-start and read-count input paths beyond the intended change from overlaps to starts.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, the comparison is unfair and the negative result is a pipeline bug.

## Idea / Hypothesis

The read-start model's inputs (observed tracks, average-signal input, input transform, covariates, masking) were built differently from the reference by mistake

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h53/ and link at least the report:
     - [Report](results/h53/report.md)   - results/h53/curve.png -->
_(none yet)_
