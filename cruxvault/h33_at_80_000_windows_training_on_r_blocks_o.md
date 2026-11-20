---
id: h33
type: idea
schema: 3
title: At 80,000 windows, training on R blocks of U is as good as training on all of U
parent: q13
rule: 
measurement: 
replicates: 3 seeds per arm; rung 4 base at 80,000 windows on eic_v1; arms terr_r64, terr_r256, terr_r1024, terr_r4096 against blocks_minus_eval (all of U)
created: "2026-11-13T12:42:23"
updated: "2026-11-13T12:42:23"
---

# h33 — At 80,000 windows, training on R blocks of U is as good as training on all of U

Parent:: [[q13_how_many_training_windows_and_how_much_d]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

within 1 seed spread for R = 4,096 and 1,024 (each block visited 1.2 and 4.9 times); lower by more than 2 seed spreads for R = 64 (26 Mb, 78 visits); R = 256 between them

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

Gives the smallest distinct territory a rung 4 run needs at the full budget; a knee far below U says territory is not the limit.

## Idea / Hypothesis

At 80,000 windows, training on R blocks of U is as good as training on all of U

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Deciding metric: the composite; per-assay headline 1.
- [ ] Settling rule: the knee is the smallest R within 1 seed spread of the control and whose paired region interval includes 0; reported, no default changes.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h33/ and link at least the report:
     - [Report](results/h33/report.md)   - results/h33/curve.png -->
_(none yet)_
