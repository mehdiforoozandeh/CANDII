---
id: q36
type: question
schema: 3
title: Does hiding depth from the encoders (a hard offset) change rung 1's correction gap closed, its depth law on subset pairs and its count scale error, against the base (offset plus soft covariate) and the tf_raw arm (soft covariate, no offset)?
parent: q34
created: "2026-11-14T16:38:14"
updated: "2026-11-14T16:38:14"
---

# q36 — Does hiding depth from the encoders (a hard offset) change rung 1's correction gap closed, its depth law on subset pairs and its count scale error, against the base (offset plus soft covariate) and the tf_raw arm (soft covariate, no offset)?

Parent:: [[q34_what_does_the_eta_formulation_change_in_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Proposed experiment 2 of results/eta_space/MODEL_DESIGN.md section 6: a new arm with column 0 of the encoder copies of C and C' set to a constant; 2 arms x 3 seeds, about 15 GPU-slice-hours and 72 CPU-hours.

## Bears on

Decides the open model choice of the parent: if the hard offset matches or beats the base, eta can be made depth-free in the model with no loss; if it loses, depth carries information beyond the offset and stays a covariate.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
