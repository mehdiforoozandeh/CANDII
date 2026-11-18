---
id: q73
type: question
schema: 3
title: How does the final miniCANDI score on the 51 EIC blind tracks, against the published EIC board and the baselines, once the blind rows are built?
parent: root
created: "2026-11-18T16:24:41"
updated: "2026-11-18T16:24:41"
---

# q73 — How does the final miniCANDI score on the 51 EIC blind tracks, against the published EIC board and the baselines, once the blind rows are built?

Parent:: [[candii]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

PI ruling 63: build the blind rows only (51 blind base rows + 11 controls + 10 reruns, about 22,700 allocated CPU-h); ruling 68 moved the build to Nibi; the PI runs the submit command (slurm/onion_data/closeout_d4_submit.sh). Rulings 21, 24, 25: the final model trains on training plus validation tracks with the design frozen on validation; the blind tracks are scored once even if the done rule fails; the board entry is seed 0, declared in the freeze file before any blind read. Not built.

## Bears on

The one held-out comparison with published imputation methods; the blind split differs from training in platform, lab and run type, so validation scores cannot predict it.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
