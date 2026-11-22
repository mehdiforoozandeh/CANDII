---
id: h48
type: idea
schema: 3
title: On the same predictions, each deviance term ranks miniCANDI arms like the current term of its goal and separates arms from seed noise at least as well
parent: q77
rule: 
measurement: 
replicates: 
created: "2026-11-22T14:12:26"
updated: "2026-11-22T14:12:26"
---

# h48 — On the same predictions, each deviance term ranks miniCANDI arms like the current term of its goal and separates arms from seed noise at least as well

Parent:: [[q77_does_a_poisson_deviance_eval_set_imputat]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Spearman of arm means >= 0.7 between the deviance term and the current term for imputation and correction; for each goal the deviance term's between-arm variance share >= the current term's; the simulation checks C1-C5 pass at 5 x 200k bins

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

if it holds, the deviance set can stand beside the current set as a depth-invariant count view; if ranks disagree, the doc must say which goal's ranking changes and why

## Idea / Hypothesis

On the same predictions, each deviance term ranks miniCANDI arms like the current term of its goal and separates arms from seed noise at least as well

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h48/ and link at least the report:
     - [Report](results/h48/report.md)   - results/h48/curve.png -->
_(none yet)_
