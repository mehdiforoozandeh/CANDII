---
id: q5
type: question
schema: 2
title: Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?
parent: q1
status: open
stale: false
created: "2026-10-08T16:58:00"
updated: "2026-10-08T16:58:00"
---

# q5 — Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

We predict how a track's read counts would look under different lab conditions. Here we first remove the original sequencing depth and add the new depth back at the end. Does that change how good the prediction is?

## TL;DR

In [[q4_how_much_capacity_does_a_covariate_condi|the earlier g → f grid]], no design kept predicted counts proportional to depth when only depth changed. Here f reads eta_in = log(1 + X) − log d, with d the source depth in millions of reads, and outputs eta_out; the target counts are negative binomial with log μ′ = log d′ + eta_out. Everything else is the earlier grid's, except one rule: each design's spread sees what its mean sees. That moves row-1 A's spread from one value per pair to one set by level, so a row-1 A arm with the old spread runs beside it. We expect the offset to add no capacity when g reads both depths, and only to move g's do-nothing output to "scale by d′/d". It is read design for design against the earlier grid and against the no-covariates twin (a g whose covariates are scrambled at every step, so it learns one average map): by CRPS with its oracle-scaled and scale-error parts, and by MSE on log(1 + counts) of the predicted mean, on all bins and on the top 1% of bins by true target value, each with its seed spread; and by the depth law on depth-only pairs.

## Question

Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory (PI 2026-10-08): readings only — no pass/fail bars, no ticks, no verdicts. The design, fork by fork, is `plan/ETA_OFFSET_AND_H_DESIGN.md` on the branch `exp/t118-counterfactual-f`.

## Answer so far

Part 1 (`results/t125/GRID_SUMMARY_part1.md`): ± is the seed wobble; a gain clears above twice the larger one.

- No depth-law cell of 72 is within 0.10; no gain over the earlier grid clears.
- Only design A clears the earlier grid: its per-track DNase CRPS (oracle-scaled + scale error) is 1.247 ± 0.016 = 1.109 + 0.138, against 3.06 ± 0.113 = 1.36 + 1.70. The offset alone gives 0.927 of this 1.813 gain.
- Real beats the twin on MSE in all 54 cells (per-track B, DNase: 0.194 ± 0.003 against 0.365 ± 0.013). This twin keeps the offset; the earlier one did not.
- All 24 CRPS > 20 records sit in the across-track A2 twin.
- Training devices are mixed. The earlier grid has no MSE.

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
