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

5 of 8 designs; final after the PI stop of 2026-10-19 ([digest](results/t125/part2_digest.md)). ± is seed wobble; clear is > 2 ×.

- True X′, narrow, log CRPS (oracle-scaled + scale error): formula 0.1177 bin-only, 0.0865 fragment-wide; rungs 1/5: affine 0.1791 ± 0.0010 (0.1367 + 0.0423) / 0.124 ± 0.0008, MLP 0.1603 ± 0.0034 / 0.0836 ± 0.0004; formula+MLP 0.0798 ± 0.0013.
- True-X′ push h beats h fit to f's NB in all 30 rows, clearly (D2, MLP rung 5: 0.1172 ± 0.0004 (0.1145 + 0.0027) against 0.1352 (0.1302 + 0.0050)).
- On D, D2, push chain h trail earlier p models: CRPS on P 0.178–0.276 against 0.143 ± 0.0016, gap oracle-scaled. On A2, C2 those exploded; they did not.
- Own noise lowers log CRPS in all 78 rows (76 clear), MSE in 5 (none clear).
- Caveats: one track, one fnb seed, D2 mixture error 9.7e-4 (bar 1e-3), affine rung-5 own-noise explosions.

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
