---
id: q78
type: question
schema: 3
title: Why does the read-start miniCANDI lose to the read-count version on imputation and calibration?
parent: q75
created: "2026-11-25T00:25:26"
updated: "2026-11-25T00:25:26"
---

# q78 — Why does the read-start miniCANDI lose to the read-count version on imputation and calibration?

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Why does the read-start miniCANDI lose to the read-count version on imputation and calibration?

## Bears on

A fixable cause (the conversion, the training budget) would reopen read-start training as a recipe ingredient; an intrinsic cause closes it.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**6 children** · open 1, leaning 5, settled 0, parked 0

- `h50` [[h50_the_nb_moment_match_which_treats_neighbo|The NB moment match, which treats neighbouring start counts as independent, understates the converted variance and causes the read-start model's overconfidence]] — *leaning · weak*
- `h51` [[h51_sparse_per_bin_start_targets_and_inputs_|Sparse per-bin start targets and inputs (more zeros, lower replicate agreement) cause the read-start imputation loss]] — *leaning · weak*
- `h52` [[h52_weighting_each_read_once_in_the_start_lo|Weighting each read once in the start loss, instead of about 2.4 times through overlapping bins, gives the read-start model less effective training signal per window]] — *leaning · weak*
- `h53` [[h53_the_read_start_model_s_inputs_observed_t|The read-start model's inputs (observed tracks, average-signal input, input transform, covariates, masking) were built differently from the reference by mistake]] — *leaning · moderate*
- `h54` [[h54_the_fixed_training_budget_80_000_windows|The fixed training budget (80,000 windows, about 1M parameters) is too small for the noisier start target, so the read-start model is under-trained]] — *open*
- `h55` [[h55_the_read_start_model_s_h3k36me3_loss_is_|The read-start model's H3K36me3 loss is a broad level error: it misplaces the total starts over gene-body domains, not per-bin noise]] — *leaning · weak*
<!-- crux:ledger:end -->
