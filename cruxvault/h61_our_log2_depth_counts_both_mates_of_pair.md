---
id: h61
type: idea
schema: 3
title: Our log2_depth counts both mates of paired-end reads, so PE depth is about 2x the depth MACS2 and our counts see.
parent: q95
rule: 
measurement: 
replicates: 
created: "2026-11-27T16:24:51"
updated: "2026-11-27T16:24:51"
---

# h61 — Our log2_depth counts both mates of paired-end reads, so PE depth is about 2x the depth MACS2 and our counts see.

Parent:: [[q95_why_does_minicandi_fail_to_follow_the_de]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

The PE rows' recorded depth is about 2x the number of fragments in their tagAligns.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

A wrong depth for PE would bias every SE/PE correction and the depth offset.

## Idea / Hypothesis

Our log2_depth counts both mates of paired-end reads, so PE depth is about 2x the depth MACS2 and our counts see.

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h61/ and link at least the report:
     - [Report](results/h61/report.md)   - results/h61/curve.png -->
_(none yet)_
