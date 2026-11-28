---
id: h54
type: idea
schema: 3
title: The fixed training budget (80,000 windows, about 1M parameters) is too small for the noisier start target, so the read-start model is under-trained
parent: q81
rule: 
measurement: 
replicates: 
created: "2026-11-25T00:25:43"
updated: "2026-11-25T00:25:43"
---

# h54 — The fixed training budget (80,000 windows, about 1M parameters) is too small for the noisier start target, so the read-start model is under-trained

Parent:: [[q81_why_does_the_read_start_minicandi_lose_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

The read-start model's held-out loss is still falling at the end of training, or more training raises its imputation skill toward the reference.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If true, more windows or capacity would close the skill gap and read-start training stays viable.

## Idea / Hypothesis

The fixed training budget (80,000 windows, about 1M parameters) is too small for the noisier start target, so the read-start model is under-trained

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

One read-start seed trained to 160,000 windows (about 260 CPU-h). Deferred: the PI answered "no, not now" on 2026-11-25, relayed by the foreman. The hypothesis stays without an answer until that run is approved.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h54/ and link at least the report:
     - [Report](results/h54/report.md)   - results/h54/curve.png -->
_(none yet)_
