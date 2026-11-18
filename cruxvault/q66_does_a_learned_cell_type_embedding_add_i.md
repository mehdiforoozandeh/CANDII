---
id: q66
type: question
schema: 3
title: Does a learned cell-type embedding add information beyond the input tracks of a cell, as shown by a gain that disappears when the cell id is forced to unknown at scoring?
parent: q48
created: "2026-11-18T16:24:27"
updated: "2026-11-18T16:24:27"
---

# q66 — Does a learned cell-type embedding add information beyond the input tracks of a cell, as shown by a gain that disappears when the cell id is forced to unknown at scoring?

Parent:: [[q48_which_minicandi_design_choices_move_impu]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Built and not run (close-out item A6): configs under tools/minicandi/configs/ablations/cell_embedding/, zero-shot control required before any 'helps' conclusion; about 738 CPU-h at the measured 256-region unit; needs the 256-region references of eic_v1_2. Next session.

## Bears on

Decides row 12 (cell-type embedding off or on); EIC holds out tracks, not cells, so a gain says nothing about unseen cell types.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
