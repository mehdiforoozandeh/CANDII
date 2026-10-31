---
id: q4
type: question
schema: 2
title: How much capacity does a covariate-conditioned transformation need before it closes the gap that a covariate-free monotone map leaves?
parent: q1
status: open
stale: false
created: "2026-09-17T21:46:17"
updated: "2026-09-17T21:46:17"
---

# q4 — How much capacity does a covariate-conditioned transformation need before it closes the gap that a covariate-free monotone map leaves?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5
Once the settings are known, how simple can the translator be — a stretch and a shift, a look at the neighbouring bins, or a full network?

## TL;DR
The parent asks whether the recorded covariates carry the information; this one asks how much capacity is needed to use it. Its children are rungs of increasing capacity, each conditioned on the source and target covariates and each measured on the counterfactual pairs of [[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]] against two comparators: one-warp, the covariate-free monotone map on each output axis fitted on training pairs and scored on held-out pairs, and the rung's own twin fitted with the covariates shuffled across pairs — so capacity and covariate information are separated at every rung. Gap-closed is (D_onewarp − D_model)/(D_onewarp − D_oracle) with the oracle two NB draws at the target depth; the noise floor is 2× the model's own seed |Δ| over the declared seeds. It is settled by the lowest rung whose gap-closed clears the bar in [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]] and whose next rung adds less than that floor: if the lowest rung, the covariates act as a rescaling; if only the top, the information is there but needs long-range or latent structure.

## Question

How much capacity does a covariate-conditioned transformation need before it closes the gap that a covariate-free monotone map leaves?

## Protocol
Locked before any run: every rung is compared to the same one-warp fit and to its own shuffled-covariate twin; rungs are read in order of capacity.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**3 children** · ideas 0/3 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h12` [[h12_a_covariate_conditioned_affine_map_on_th|A covariate-conditioned affine map on the log-count axis closes at least half of the gap that the covariate-free monotone map leaves, and beats its shuffled-covariate twin by more than the seed floor]] — *idea*
- `h13` [[h13_a_small_covariate_conditioned_convolutio|A small covariate-conditioned convolutional map over neighbouring bins closes gap that the conditioned affine map leaves, and beats its shuffled-covariate twin by more than the seed floor]] — *idea*
- `h14` [[h14_an_encoder_decoder_with_the_encoder_cond|An encoder–decoder with the encoder conditioned on the source covariates and the decoder on the target covariates closes gap that the conditioned convolutional map leaves, and beats its shuffled-covariate twin by more than the seed floor]] — *idea*
<!-- crux:ledger:end -->
