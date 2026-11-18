---
id: q71
type: question
schema: 3
title: Does the architecture order at equal training windows hold when every architecture gets the same training compute (2.4e15 FLOPs)?
parent: q39
created: "2026-11-18T16:24:40"
updated: "2026-11-18T16:24:40"
---

# q71 — Does the architecture order at equal training windows hold when every architecture gets the same training compute (2.4e15 FLOPs)?

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Close-out item A9: 12 matched-compute trainings queued on Fir (jobs 4345024 to 4345035, pending through the Fir outage), then their scoring; 1,507 to 2,393 CPU-h. Rule: closeout A9 section 6.

## Bears on

If an architecture moves up beyond the range at matched compute, the equal-window order favored fast learners and scaling must use the matched order (row 28).

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
