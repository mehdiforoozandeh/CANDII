---
id: q37
type: question
schema: 3
title: How much of rung 1's correction gain over depth-only scaling comes from depth and how much from the other covariates, read by decoding the model at the source covariates and the target depth?
parent: q34
created: "2026-11-14T16:38:14"
updated: "2026-11-14T16:38:14"
---

# q37 — How much of rung 1's correction gain over depth-only scaling comes from depth and how much from the other covariates, read by decoding the model at the source covariates and the target depth?

Parent:: [[q34_what_does_the_eta_formulation_change_in_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Proposed experiment 3 of results/eta_space/MODEL_DESIGN.md section 6 and the split of results/gap_closed_floor/GAP_FLOOR.md section 5: one extra prediction pass on the pair regions with c_out set to the source covariates, then one scoring; no training.

## Bears on

Splits headline metric 4 into a denoising part and a covariate part for the model itself, so it says whether eta-space decoding gives a model readout that the count-space score alone does not.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
