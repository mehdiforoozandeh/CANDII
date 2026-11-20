---
id: h36
type: idea
schema: 3
title: Seeing the evaluation loci in training does not raise the score
parent: q16
rule: 
measurement: 
replicates: 3 seeds per arm; rung 4 base at 80,000 windows on eic_v1; arms today, no_eval, eval_heavy
created: "2026-11-13T12:42:24"
updated: "2026-11-13T12:42:24"
---

# h36 — Seeing the evaluation loci in training does not raise the score

Parent:: [[q16_does_the_validation_score_depend_on_havi]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

equivalence at today's rate: `today` minus `no_eval` is inside the margin of plus or minus 1 seed spread, with a paired region interval inside the same margin; `eval_heavy` (half of the windows from the evaluation regions) gains a positive amount below 2 seed spreads

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, today's protocol, which lets training windows fall inside the evaluation regions, does not bias the validation score; else the PI rules on the protocol (ruling 39).

## Idea / Hypothesis

Seeing the evaluation loci in training does not raise the score

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Deciding metric: the composite; headline 1 in counts.
- [ ] Settling rule: reported as a finding (ruling 39); "no detectable gap" is stated only if both the seed rule and the region interval are inside the margin; else the gap is reported with its interval.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h36/ and link at least the report:
     - [Report](results/h36/report.md)   - results/h36/curve.png -->
_(none yet)_
