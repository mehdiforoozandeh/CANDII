---
id: q36
type: question
schema: 3
title: Does hiding depth from the network (a hard offset) change rung 4's goal terms (specific variance, correction gap closed, calibration) against rung-4 base (offset plus soft depth covariate)?
parent: q34
created: "2026-11-14T16:38:14"
updated: "2026-11-14T16:38:14"
---

# q36 — Does hiding depth from the network (a hard offset) change rung 4's goal terms (specific variance, correction gap closed, calibration) against rung-4 base (offset plus soft depth covariate)?

Parent:: [[q34_what_does_the_eta_formulation_change_in_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Proposed experiment 2 of results/eta_space/MODEL_DESIGN.md section 6. PI ruling 48 (2026-11-15, 'Rung 4 only') set the design: one arm, rung 4 with the depth covariate hidden from the network (config key `hide_depth_covariate`; the mean stays depth x 2^eta), 3 seeds, on the existing CPU pipeline; no rung-1 arm and no tf_raw arm. Screening score (pairs on 16 regions).

## Bears on

Decides the open model choice of the parent: if the hard offset matches or beats the base, eta can be made depth-free in the model with no loss; if it loses, depth carries information beyond the offset and stays a covariate.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**1 children** · open 1, leaning 0, settled 0, parked 0 · 4 finding(s) on this question

- `q41` _(Q)_ [[q41_why_does_hiding_depth_from_rung_4_drop_i|Why does hiding depth from rung 4 drop its uncertainty term below the marginal NB while its mean-based terms move much less: does the predicted NB size lose a needed dependence on depth?]] — *open*
- `f81` finding (moderate) [[f81_read_from_the_refreshed_rung_4_arm_table|Read from the refreshed rung-4 arm table and the select_rule terms, 3 paired…]]
- `f83` finding (moderate) [[f83_3_seeds_per_arm_and_the_input_side_drop_|3 seeds per arm and the input-side drop exceeds both seed ranges; 16 regions…]]
- `f100` finding (moderate) [[f100_beyond_the_range_on_the_u_net_on_three_t|Beyond the range on the U-Net on three terms; the mechanism is not checked.]]
- `f113` finding (moderate) [[f113_three_seeds_per_arm_on_all_256_regions_w|Three seeds per arm on all 256 regions with a rule written before the…]]
<!-- crux:ledger:end -->
