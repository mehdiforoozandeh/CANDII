---
id: q2
type: question
schema: 2
title: Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?
parent: q1
status: open
stale: false
created: "2026-09-17T20:35:48"
updated: "2026-09-17T20:35:48"
---

# q2 — Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5
If the only thing that changed between two runs of the same sample is which reads made it into the pile, can knowing that setting translate one run into the other?

## TL;DR
Four processing choices decide which reads enter the treatment pileup, and each moves both the raw counts and the −log10 p signal: read length, duplicate handling, the mapping-quality filter, and — as a proxy for antibody efficiency — the fraction of treatment reads that are really background. For each we hold a counterfactual pair built by the ENCODE pipeline with exactly that one choice changed and its value recorded ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The question is whether a transformation conditioned on the source and target values of that covariate predicts the other member of the pair better than one-warp, the single covariate-free monotone map on each output axis fitted on training pairs and scored on held-out pairs, by the margin locked in [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]]: gap-closed ≥ 0.5 of the one-warp → oracle interval per mark class and a gain above 2× the model's seed |Δ|. It answers yes only if every child holds; a refuted child names a knob whose recorded value does not carry the information.

## Question

Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?

## Protocol
Locked before any run: the children share q1's protocol; each is scored on both output axes; the shuffled-covariate twin of every model is a required control.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**4 children** · ideas 0/4 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h3` [[h3_conditioning_on_the_recorded_read_length|Conditioning on the recorded read length predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h4` [[h4_conditioning_on_the_recorded_duplicate_h|Conditioning on the recorded duplicate handling predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h5` [[h5_conditioning_on_the_recorded_mapping_qua|Conditioning on the recorded mapping-quality filter predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h6` [[h6_conditioning_on_the_recorded_antibody_ef|Conditioning on the recorded antibody-efficiency proxy predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
<!-- crux:ledger:end -->
