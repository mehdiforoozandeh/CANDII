---
id: q34
type: question
schema: 3
title: "What does the eta formulation change in the miniCANDI model: is the depth offset already in it, and would a hard offset, a dispersion tied across depth, or an eta-space readout of the model help?"
parent: q33
created: "2026-11-14T16:36:51"
updated: "2026-11-14T16:36:51"
---

# q34 — What does the eta formulation change in the miniCANDI model: is the depth offset already in it, and would a hard offset, a dispersion tied across depth, or an eta-space readout of the model help?

Parent:: [[q33_can_the_negative_binomial_be_written_wit]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

The model-design part of the PI's eta question (results/eta_space/MODEL_DESIGN.md). Earlier work on the same idea: q5 (depth offset on f, settled by PI ruling of 2026-10-31: depth stays a learned covariate) and q7.

## Bears on

Says whether 'use eta throughout' asks for a change to the model or only to its evaluation and reporting; if the model already predicts mean = depth x 2^eta, the open choice narrows to hiding depth from the encoders and tying the dispersion.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**4 children** · open 4, leaning 0, settled 0, parked 0 · 1 finding(s) on this question

- `q35` _(Q)_ [[q35_do_rung_1_base_and_the_baselines_meet_th|Do rung 1 base and the baselines meet the depth law on the subset (pure thinning) correction pairs, and what is the count scale error against the true target total on all pairs?]] — *open*
- `q36` _(Q)_ [[q36_does_hiding_depth_from_the_encoders_a_ha|Does hiding depth from the encoders (a hard offset) change rung 1's correction gap closed, its depth law on subset pairs and its count scale error, against the base (offset plus soft covariate) and the tf_raw arm (soft covariate, no offset)?]] — *open*
- `q37` _(Q)_ [[q37_how_much_of_rung_1_s_correction_gain_ove|How much of rung 1's correction gain over depth-only scaling comes from depth and how much from the other covariates, read by decoding the model at the source covariates and the target depth?]] — *open*
- `q38` _(Q)_ [[q38_does_rung_1_s_predicted_nb_size_trend_wi|Does rung 1's predicted NB size trend with target log2 depth on the depth-ladder targets of one base, and does the PIT deviation differ across log2-depth terciles?]] — *open*
- `f67` finding (moderate) [[f67_read_from_the_code_and_from_recorded_fin|Read from the code and from recorded findings, with line citations checked;…]]
<!-- crux:ledger:end -->
