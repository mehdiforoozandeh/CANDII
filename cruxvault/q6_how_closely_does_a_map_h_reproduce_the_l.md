---
id: q6
type: question
schema: 2
title: How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?
parent: q1
status: open
stale: false
created: "2026-10-08T16:58:00"
updated: "2026-10-08T16:58:00"
---

# q6 — How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

A peak caller turns read counts into a significance score with a fixed recipe. Can we copy that step, with the textbook formula or with something learned, and how much information does the copy need?

## TL;DR

h maps a negative-binomial prediction of the target counts, plus the matched control, to P = −log10 p, and is scored against the real P track. The readings span two axes. Complexity runs from the peak caller's own formula (the MACS2 Poisson upper tail at the local control rate), with nothing learned, to learned affine and MLP maps and the formula plus a learned correction. Inputs run from the bin's count and the control in the same bin to the full set: control windows, genome background, both depths, the control ratio, fragment length and a count window one fragment wide. Two readings sit inside: how uncertainty passes through h (the count distribution pushed through h exactly, against the same plus h's own learned noise), and whether h trained on a count model's predictions repairs that model's errors (every learned h is scored both on the true counts and on the predictions). Readings are CRPS on P and MSE on log(1 + P) of the predicted mean, on all bins and on the top 1% of bins by true P, each with its seed spread.

## Question

How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory (PI 2026-10-08): readings only — no pass/fail bars, no ticks, no verdicts. The design, fork by fork, is `plan/ETA_OFFSET_AND_H_DESIGN.md` on the branch `exp/t118-counterfactual-f`.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
