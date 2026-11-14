---
id: q30
type: question
schema: 3
title: Does a per-bin loss weight from annotation or peak density, applied inside the window at 25 bp, improve the model as a stage-3 loss arm?
parent: q14
created: "2026-11-14T02:17:34"
updated: "2026-11-14T02:17:34"
---

# q30 — Does a per-bin loss weight from annotation or peak density, applied inside the window at 25 bp, improve the model as a stage-3 loss arm?

Parent:: [[q14_how_should_training_regions_be_selected_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Recorded 2026-11-14 when the selection experiment was redesigned (Amendment 7, selection section; interim finding 1 of 'onion region budget and selection.md'): at the model's window size, window-level strata are near-uniform, so annotation-based selection can act only inside the window, as a per-bin loss weight. That changes the objective, so it is a candidate loss arm for stage 3, not part of the selection experiment.

## Bears on

It is the only route left for annotation-based emphasis below the window scale; a gain would move the selection answer from where windows are drawn to how bins are weighted.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
