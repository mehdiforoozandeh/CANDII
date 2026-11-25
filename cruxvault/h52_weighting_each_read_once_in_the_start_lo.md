---
id: h52
type: idea
schema: 3
title: Weighting each read once in the start loss, instead of about 2.4 times through overlapping bins, gives the read-start model less effective training signal per window
parent: q81
rule: 
measurement: 
replicates: 
created: "2026-11-25T00:25:43"
updated: "2026-11-25T00:25:43"
---

# h52 — Weighting each read once in the start loss, instead of about 2.4 times through overlapping bins, gives the read-start model less effective training signal per window

Parent:: [[q81_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

The read-start loss or optimiser differs from the reference in a way the shared learning rate, batch and window budget do not compensate.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, reweighting the loss or raising the learning rate would recover the loss at no data cost.

## Idea / Hypothesis

Weighting each read once in the start loss, instead of about 2.4 times through overlapping bins, gives the read-start model less effective training signal per window

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h52/ and link at least the report:
     - [Report](results/h52/report.md)   - results/h52/curve.png -->
_(none yet)_
