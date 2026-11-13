---
id: h32
type: idea
schema: 3
title: A smaller budget of windows ranks the arms of a rung as 80,000 windows does
parent: q13
rule: 
measurement: "Ranking test: arms = the equal-budget arms of the rung (7 for rung 1, 8 for rung 4; matched_compute left out); statistic = the composite, mean of 3 seeds, per arm. A pair is resolved when its difference at 80,000 windows exceeds 2 seed spreads. A budget passes when (1) every resolved pair keeps its order, (2) the top arm is the same or is not resolved from it at 80,000, and (3) the Spearman rank correlation with the 80,000-window composites is at least 0.8; (3) counts only when seed 0 against the mean of seeds 1 and 2 at 80,000 has Spearman at least 0.8, else it prints not informative; with no resolved pair the test prints not informative. The same test is printed for each goal term and for the logged validation loss."
replicates: 3 seeds (0, 1, 2) per arm and budget; budgets 20,000 and 40,000 windows against the ladder's 80,000; rung 1 (7 arms) and rung 4 (8 arms)
created: "2026-11-13T12:42:23"
updated: "2026-11-13T12:42:23"
---

# h32 — A smaller budget of windows ranks the arms of a rung as 80,000 windows does

Parent:: [[q13_how_many_training_windows_and_how_much_d]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

pass at 40,000 for both rung classes; pass at 20,000 for rung 1 only (its curve is flat by about 24,000)

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, prototype screening of a rung class may use the smallest passing budget, which shortens the feedback loop.

## Idea / Hypothesis

A smaller budget of windows ranks the arms of a rung as 80,000 windows does

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Deciding metric: the ranking test (composite of 3 seeds per arm; resolved pairs keep order, same top arm, Spearman at least 0.8 when the 80,000-window ranking is stable).
- [ ] Settling rule: the smallest budget that passes is the prototype budget of that rung class.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h32/ and link at least the report:
     - [Report](results/h32/report.md)   - results/h32/curve.png -->
_(none yet)_
