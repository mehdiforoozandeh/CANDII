---
id: q75
type: question
schema: 3
title: Can one miniCANDI recipe match or beat the best base model on imputation, correction and calibration at the same time (Pareto dominance over the conv autoencoder, conv + attention and U-Net bases on eic_v1_2)?
parent: q72
created: "2026-11-21T04:04:04"
updated: "2026-11-21T04:04:04"
---

# q75 — Can one miniCANDI recipe match or beat the best base model on imputation, correction and calibration at the same time (Pareto dominance over the conv autoencoder, conv + attention and U-Net bases on eic_v1_2)?

Parent:: [[q72_which_architecture_and_which_data_versio]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Can one miniCANDI recipe match or beat the best base model on imputation, correction and calibration at the same time (Pareto dominance over the conv autoencoder, conv + attention and U-Net bases on eic_v1_2)?

## Bears on

A recipe that dominates on all three terms removes the architecture trade-off, so the scaling choice becomes that recipe instead of a pick among three tied models.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**4 children** · open 4, leaning 0, settled 0, parked 0 · 1 finding(s) on this question

- `h44` [[h44_a_u_net_whose_count_skips_are_film_modul|A U-Net whose count skips are FiLM-modulated by (C, C') keeps the base U-Net imputation and reaches conv-level correction]] — *open*
- `h45` [[h45_pre_and_post_cac_layers_in_the_testbed_g|Pre and post CAC layers in the testbed g/f form (g reads only covariates and outputs the parameters of a count map f; f before the network removes the C artifact, f after the NB re-applies the C' artifact) raise correction without lowering imputation or calibration]] — *open*
- `h46` [[h46_where_film_injects_the_covariates_encode|Where FiLM injects the covariates (encoder only, decoder only, bottleneck only, every block including attention) moves correction beyond the seed range while imputation stays within range]] — *open*
- `h47` [[h47_training_correction_and_same_view_pairs_|Training correction and same-view pairs on read-disjoint halves (source pr1, target pr2) removes the U-Net's input copy and lifts its correction to conv level without losing imputation]] — *open*
- `f143` finding (strong) [[f143_imputation_and_correction_clear_the_bars|Imputation and correction clear the bars by several seed ranges; calibration…]]
<!-- crux:ledger:end -->
