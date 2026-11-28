---
id: h50
type: idea
schema: 3
title: The NB moment match, which treats neighbouring start counts as independent, understates the converted variance and causes the read-start model's overconfidence
parent: q81
rule: 
measurement: 
replicates: 
created: "2026-11-25T00:25:43"
updated: "2026-11-25T00:25:43"
---

# h50 — The NB moment match, which treats neighbouring start counts as independent, understates the converted variance and causes the read-start model's overconfidence

Parent:: [[q81_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Correcting the converted dispersion for spatially correlated errors moves the deviance calibration ratio from 1.31 most of the way to the reference's 1.06, and leaves the imputation skill unchanged.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, the calibration loss is a conversion artefact that a covariance-aware conversion removes, not a property of read-start training.

## Idea / Hypothesis

The NB moment match, which treats neighbouring start counts as independent, understates the converted variance and causes the read-start model's overconfidence

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h50/ and link at least the report:
     - [Report](results/h50/report.md)   - results/h50/curve.png -->
_(none yet)_
