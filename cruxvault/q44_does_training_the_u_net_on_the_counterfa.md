---
id: q44
type: question
schema: 3
title: Does training the U-Net on the counterfactual processing views of EIC (depth thinning, single-end, read crop, no duplicate removal, no MAPQ filter) raise the three goal terms over training on the base view alone, and for which view type and which evaluation pairs?
parent: q23
created: "2026-11-15T17:12:07"
updated: "2026-11-15T17:12:07"
---

# q44 — Does training the U-Net on the counterfactual processing views of EIC (depth thinning, single-end, read crop, no duplicate removal, no MAPQ filter) raise the three goal terms over training on the base view alone, and for which view type and which evaluation pairs?

Parent:: [[q23_is_the_model_wired_as_designed_does_scra]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Does training the U-Net on the counterfactual processing views of EIC (depth thinning, single-end, read crop, no duplicate removal, no MAPQ filter) raise the three goal terms over training on the base view alone, and for which view type and which evaluation pairs?

## Bears on

If the views do not raise the scores, the wiring of cross-view training or the correction metric is at fault, or the views are not needed at this scale

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**1 children** · open 1, leaning 0, settled 0, parked 0 · 3 finding(s) on this question

- `q45` _(Q)_ [[q45_why_does_adding_the_no_duplicate_removal|Why does adding the no-duplicate-removal and no-MAPQ-filter views to training lower gap closed on the never-trained pair that combines both (0.08 against 1.77 for base only, half-read floor scale)?]] — *open*
- `f82` finding (weak) [[f82_3_seeds_per_arm_but_the_all_views_refere|3 seeds per arm, but the all-views reference ran on Nibi from an earlier…]]
- `f99` finding (moderate) [[f99_beyond_the_seed_range_on_all_three_model|Beyond the seed range on all three models.]]
- `f106` finding (strong) [[f106_twice_the_seed_range_on_all_three_models|Twice the seed range on all three models with 3 seeds each on all 256…]]
<!-- crux:ledger:end -->
