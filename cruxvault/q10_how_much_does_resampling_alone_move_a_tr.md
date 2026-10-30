---
id: q10
type: question
schema: 2
title: How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?
parent: q1
status: open
stale: false
created: "2026-10-29T20:07:43"
updated: "2026-10-29T20:07:43"
---

# q10 — How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If you split one experiment's reads into two random halves, the two halves already disagree a little, and no correction can do better than that disagreement.

## TL;DR

Every covariate correction is read against a floor: the change that pure read resampling causes with no covariate changed. This asks three things on the t117 pseudoreplicates (pr1/pr2 halves of all 130 counterfactual products), thinned along a depth ladder: how much resampling moves peak height, 1 kb shape and per-bin divergence at each depth; where in the genome the split-half disagreement concentrates; and how it shrinks with depth, read off the curve at the 30M the arms are built at. Signal level is fitted first as the null, since counting noise alone makes low-count bins irreproducible; mappability and repeats, GC, and peak versus domain context are read on the residual. It is settled when the floor at 30M is stated per mark class with its depth curve, and the residual regional pattern is either named or shown to be absent. The floor becomes the oracle for every reading under q1, beside the monotone scale map and the shuffled-covariate twin (PI ruling 2026-10-29).

## Question

How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
Locked before any run: draws are the t117 halves thinned to a depth ladder (no new alignment); signal level is the null and is fitted before any regional feature is read; every q1 reading is scored after this floor lands, against all three baselines.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
