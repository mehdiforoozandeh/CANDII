---
id: q64
type: question
schema: 3
title: Which combination of the switches that qualified one at a time gives the best miniCANDI on the conv autoencoder, conv + attention and the U-Net, on all 256 regions (the mix-and-match round)?
parent: q48
created: "2026-11-18T16:24:27"
updated: "2026-11-18T16:24:27"
---

# q64 — Which combination of the switches that qualified one at a time gives the best miniCANDI on the conv autoencoder, conv + attention and the U-Net, on all 256 regions (the mix-and-match round)?

Parent:: [[q48_which_minicandi_design_choices_move_impu]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

PI decision 8 of 2026-11-17: gate first (256-region scores of the 5 single runs that the rescoring lacks, about 474 CPU-h), then the round for the switches that pass (24 arms plus 3 references, 3 seeds, tables eic_v1_2, about 5,390 CPU-h). Pool today: conv autoencoder: same-view only; conv + attention: concatenation, fixed input depth, imputation weight 4, NB CRPS, same-view only; U-Net: average input, Pearson transform, fixed input depth. Rule written before any run: closeout A13 section 4.6. Fixed-depth arms are also scored at native depth. Next session.

## Bears on

Turns the single-switch ablations into the design: it decides the provisional rows 6, 10, 13, 19, 20, 22 and 25 of the settled miniCANDI design.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
