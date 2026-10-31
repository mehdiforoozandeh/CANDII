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
If the only thing that changed between two runs of the same sample is which reads made it into the pile, can knowing that setting translate one run into the other? Nobody has tested it yet; early readings say the settings help a little on pairs the model has seen, and seldom on new ones.

## TL;DR
CAC: the read-selection knobs. Four processing choices decide which reads enter the treatment pileup, and each moves both the counts and the −log10 p signal: [[h3_conditioning_on_the_recorded_read_length|read length]], [[h4_conditioning_on_the_recorded_duplicate_h|duplicate removal]], [[h5_conditioning_on_the_recorded_mapping_qua|the MAPQ filter]] and [[h6_conditioning_on_the_recorded_antibody_ef|the antibody proxy]]. Each has a counterfactual pair built by the ENCODE pipeline with that one choice changed and recorded ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The question is whether a transformation conditioned on the source and target values beats its no-covariates twin (C and C′ re-scrambled at every step) by more than 2 × seed wobble, the per-child checks set by the PI on 2026-10-31 in place of [[q1_do_the_recorded_experimental_covariates_|the parent question's]] one-warp → oracle bar. It answers yes only if every child holds. No child has been tested; exploratory readings from runs that trained on all ten knobs at once are summarised below.

## Motivation

The PI asked which processing knobs, beyond depth and run type, move the signal: "what are the main technical covariates and data processing knobs that affect signal" (2026-09-17). A literature survey named ten; the PI wanted "counterfactuals for each of the 10 important knobs" and then the whole tree rebuilt to hold them (2026-09-18). The read-selection and control/caller split was drafted by the agent and reviewed by critics; the PI did not confirm the split in words.

## Question

Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?

## Protocol
Locked before any run: each child is scored on both output axes. PI ruling 2026-10-31: the children are scored against the no-covariates twin (C and C′ re-scrambled at every step), not by q1's one-warp → oracle gap-closed; a C′ swapped at scoring is the outcome-neutral control.

## Answer so far

Readings only, from [[t118_learn_the_covariate_conditioned_tra|the g × f grid]] and [[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder rungs]]; the [knob report](results/q2/KNOB_READINGS.md) has the tables and the CRPS split. Two patterns hold. First, a per-pair monotone map closes about half of the gap toward the pseudoreplicate oracle for duplicate removal (both axes) and read-length counts, but almost none for MAPQ or for read length in p. The one-warp null is therefore plausible for duplicates and unlikely for MAPQ. Second, a generator g that reads the covariates beats its no-covariates twin on trained pairs in counts for every knob (median gain 0.08–0.18 against a seed wobble of 0.003–0.06), and in p by less (0.03–0.14). On never-trained pairs within a knob the p gain vanishes or turns negative, and the learned-decoder encoder-decoder falls below its twin on nearly every never-trained pair. None of this scores the locked bar: the competitor is the twin, not one-warp, and the oracle is a half-depth pseudoreplicate.

<!-- crux:ledger:start -->
**4 children** · open 4, leaning 0, settled 0, parked 0 · 1 finding(s) on this question

- `h3` [[h3_conditioning_on_the_recorded_read_length|Conditioning on the recorded read length predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h4` [[h4_conditioning_on_the_recorded_duplicate_h|Conditioning on the recorded duplicate handling predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h5` [[h5_conditioning_on_the_recorded_mapping_qua|Conditioning on the recorded mapping-quality filter predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h6` [[h6_conditioning_on_the_recorded_antibody_ef|Conditioning on the recorded antibody-efficiency proxy predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `f10` finding (weak) [[f10_readings_only_from_runs_that_trained_on_|Readings only, from runs that trained on all ten knobs at once; the locked…]]
<!-- crux:ledger:end -->
