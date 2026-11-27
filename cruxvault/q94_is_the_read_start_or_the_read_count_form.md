---
id: q94
type: question
schema: 3
title: Is the read-start or the read-count formulation better for the scaled miniCANDI when both get equal data, windows, parameters and tuning, judged in count space, start space and 1 kb sums, and what explains the winner?
parent: q72
created: "2026-11-27T16:24:08"
updated: "2026-11-27T16:24:08"
---

# q94 — Is the read-start or the read-count formulation better for the scaled miniCANDI when both get equal data, windows, parameters and tuning, judged in count space, start space and 1 kb sums, and what explains the winner?

Parent:: [[q72_which_architecture_and_which_data_versio]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Is the read-start or the read-count formulation better for the scaled miniCANDI when both get equal data, windows, parameters and tuning, judged in count space, start space and 1 kb sums, and what explains the winner?

## Bears on

Decides the input and target representation of the scaled model, and whether a per-position dedup layer is possible.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**2 children** · open 2, leaning 0, settled 0, parked 0

- `h56` [[h56_the_read_count_model_s_advantage_comes_f|The read-count model's advantage comes from target smoothing (statistical efficiency), not from more information: a read-start-input model trained on count targets through a fixed fragment-span kernel layer matches the read-count model.]] — *open*
- `h57` [[h57_the_read_count_model_still_beats_the_rea|The read-count model still beats the read-start model when both are scored in strand-summed read-start space (Max's slam dunk).]] — *open*
<!-- crux:ledger:end -->
