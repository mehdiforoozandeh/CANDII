---
id: q67
type: question
schema: 3
title: Does a bin size other than 25 bp (1 to 125 bp) change miniCANDI's goal terms, beyond what the change of context length explains?
parent: q48
created: "2026-11-18T16:24:27"
updated: "2026-11-18T16:24:27"
---

# q67 — Does a bin size other than 25 bp (1 to 125 bp) change miniCANDI's goal terms, beyond what the change of context length explains?

Parent:: [[q48_which_minicandi_design_choices_move_impu]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

PI decision 6 of 2026-11-17: the full design, next session (75 runs, about 2,340 CPU-h, ready queue entry 9, DESIGN=full). Prerequisite: the 1 bp tables are built and not verified, and the resolution evaluation set is not built (task t354). Rule: closeout A7 section 9.5.

## Bears on

Decides the resolution row (26) for scaling; the variance read says almost none of the variance below 25 bp is reproducible, so a finer bin is expected to add noise, not signal.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
