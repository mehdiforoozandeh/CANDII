---
id: q19
type: question
schema: 3
title: "What do the training-free baselines reach on the three goals, i.e. where does a model have room: imputation (the position average against the marginal NB and the noise floor), covariate correction (the gap between depth-only scaling and the half-against-half floor, per covariate knob: the size of the artifact that a model could remove), calibration (PIT and the variance ratio of the baselines)?"
parent: root
created: "2026-11-14T00:35:07"
updated: "2026-11-14T00:35:07"
---

# q19 — What do the training-free baselines reach on the three goals, i.e. where does a model have room: imputation (the position average against the marginal NB and the noise floor), covariate correction (the gap between depth-only scaling and the half-against-half floor, per covariate knob: the size of the artifact that a model could remove), calibration (PIT and the variance ratio of the baselines)?

Parent:: [[candii]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Iteration 1 of Amendment 9 of the onion plan (PI directive of 2026-11-14: iterative deployment); results go to INSTRUCTIONS/peeling_the_onion/onion paper.md.

## Bears on

Fixes the reference every model result is compared with, and shows on which goal and covariate knob a model can gain the most.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**2 children** · open 1, leaning 1, settled 0, parked 0 · 3 finding(s) on this question

- `h38` [[h38_the_gap_between_depth_only_scaling_and_t|The gap between depth-only scaling and the floor is largest for the depth and run-type knobs and near 0 for pure subset pairs]] — *leaning · moderate*
- `q32` _(Q)_ [[q32_which_reference_should_define_the_1_of_t|Which reference should define the 1 of the correction gap-closed scale (headline metric 4), given that the position average goes below the half-against-half floor?]] — *open*
- `f57` finding (moderate) [[f57_covariate_correction_headline_metric_4_o|Covariate correction (headline metric 4 of the evaluation protocol, gap…]]
- `f58` finding (moderate) [[f58_imputation_reference_the_position_averag|Imputation reference: the position average recovered 0.119 [-0.156, 0.194]…]]
- `f59` finding (moderate) [[f59_calibration_reference_both_baselines_pas|Calibration reference: both baselines passed the pseudoreplicate lower…]]
<!-- crux:ledger:end -->
