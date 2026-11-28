---
id: h58
type: idea
schema: 3
title: "Deduplication is a per-position cap whose expected effect is closed form in read-start space: kept starts per strand per 25 bp bin = 25(1-(1+lambda/(25a))^-a), with one fitted shape a per assay."
parent: q95
rule: 
measurement: 
replicates: 
created: "2026-11-27T16:24:50"
updated: "2026-11-27T16:24:50"
---

# h58 — Deduplication is a per-position cap whose expected effect is closed form in read-start space: kept starts per strand per 25 bp bin = 25(1-(1+lambda/(25a))^-a), with one fitted shape a per assay.

Parent:: [[q95_why_does_minicandi_fail_to_follow_the_de]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

On the dedup-off/dedup-on view pairs, the start-space cap predicts the dedup-on track closer to the half-vs-half floor than the count-space saturation law, in the top 1% of bins.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, a closed-form dedup layer like the depth offset is possible, and it needs the start representation.

## Idea / Hypothesis

Deduplication is a per-position cap whose expected effect is closed form in read-start space: kept starts per strand per 25 bp bin = 25(1-(1+lambda/(25a))^-a), with one fitted shape a per assay.

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h58/ and link at least the report:
     - [Report](results/h58/report.md)   - results/h58/curve.png -->
_(none yet)_
