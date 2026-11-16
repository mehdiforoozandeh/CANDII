---
id: q46
type: question
schema: 3
title: Why does the held-out NB loss rank the U-Net first and the conv models third and fourth while the composite ranks the conv models first, and why does more training lower held-out loss without raising the composite?
parent: q39
created: "2026-11-15T21:45:46"
updated: "2026-11-15T21:45:46"
---

# q46 — Why does the held-out NB loss rank the U-Net first and the conv models third and fourth while the composite ranks the conv models first, and why does more training lower held-out loss without raising the composite?

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Why does the held-out NB loss rank the U-Net first and the conv models third and fourth while the composite ranks the conv models first, and why does more training lower held-out loss without raising the composite?

## Bears on

If the composite is at fault (the gap-closed term), the ladder order must be re-read with a corrected metric; if the loss is at fault, the tuning protocol selects on the wrong signal

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**0 children** · open 0, leaning 0, settled 0, parked 0 · 2 finding(s) on this question

- `f85` finding (moderate) [[f85_recomputed_from_stored_per_pair_records_|recomputed from stored per-pair records for 78 seeds, matching the scorer to…]]
- `f88` finding (moderate) [[f88_3_seeds_per_recipe_at_equal_windows_winn|3 seeds per recipe at equal windows; winners were chosen on seed 0 held-out…]]
<!-- crux:ledger:end -->
