---
id: h55
type: idea
schema: 3
title: "The read-start model's H3K36me3 loss is a broad level error: it misplaces the total starts over gene-body domains, not per-bin noise"
parent: q78
rule: 
measurement: 
replicates: 
created: "2026-11-25T02:18:02"
updated: "2026-11-25T02:18:02"
---

# h55 — The read-start model's H3K36me3 loss is a broad level error: it misplaces the total starts over gene-body domains, not per-bin noise

Parent:: [[q78_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Smoothing the start means does not raise H3K36me3 skill, residuals stay correlated over more than 100 bins, and predicted total starts differ from observed by more than 5% on the regions

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

names where the largest share (0.093 of 0.168) of the skill gap comes from, and so which fix to try

## Idea / Hypothesis

The read-start model's H3K36me3 loss is a broad level error: it misplaces the total starts over gene-body domains, not per-bin noise

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h55/ and link at least the report:
     - [Report](results/h55/report.md)   - results/h55/curve.png -->
_(none yet)_
