---
id: h51
type: idea
schema: 3
title: Sparse per-bin start targets and inputs (more zeros, lower replicate agreement) cause the read-start imputation loss
parent: q78
rule: 
measurement: 
replicates: 
created: "2026-11-25T00:25:43"
updated: "2026-11-25T00:25:43"
---

# h51 — Sparse per-bin start targets and inputs (more zeros, lower replicate agreement) cause the read-start imputation loss

Parent:: [[q78_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Replicate agreement of starts is well below that of overlap counts, and the assays with the largest drop lose the most imputation skill.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, read starts lose information per bin that the model cannot recover, which argues against read-start training at 25 bp.

## Idea / Hypothesis

Sparse per-bin start targets and inputs (more zeros, lower replicate agreement) cause the read-start imputation loss

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h51/ and link at least the report:
     - [Report](results/h51/report.md)   - results/h51/curve.png -->
_(none yet)_
