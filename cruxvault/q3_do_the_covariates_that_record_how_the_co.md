---
id: q3
type: question
schema: 2
title: Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?
parent: q1
status: open
stale: false
created: "2026-09-17T21:46:17"
updated: "2026-09-17T21:46:17"
---

# q3 — Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5
If two p-value tracks were called from the very same reads but with a different background or a different smoothing, can knowing that setting translate one into the other? Nobody has tested it yet; early readings say the settings help a lot.

## TL;DR
Four choices act at or after peak calling and leave the treatment counts bit-identical to base: [[h7_conditioning_on_the_recorded_control_to_|the control ratio]], [[h8_conditioning_on_the_recorded_control_ide|control identity]], [[h9_conditioning_on_the_recorded_control_dep|control depth]] and [[h10_conditioning_on_the_recorded_fragment_ex|fragment extension]]. Each has a counterfactual pair built by the ENCODE pipeline with that one choice changed and recorded ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The question is whether a transformation conditioned on the source and target values predicts the other member's −log10 p signal better than one-warp by the margin of [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]]; a transformation whose count prediction moves has produced an invalid run. It answers yes only if every child holds. No child has been tested; exploratory readings are summarised below.

## Question

Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?

## Protocol
Locked before any run: gap-closed is read on the −log10 p axis only; the count-unchanged control and the shuffled-covariate twin are required on every child.

## Answer so far

Readings only, from [[t118_learn_the_covariate_conditioned_tra|the g × f grid]] and [[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder rungs]]; the [knob report](results/q3/KNOB_READINGS.md) has the tables and the CRPS split. In p these knobs give the strongest covariate signal of all ten: g beats its no-covariates twin by a median 0.36–0.62 on trained pairs, and by 0.31–0.48 on never-trained pairs for the ratio, control identity and control depth, against seed wobbles of 0.003–0.024. Fragment extension is the exception: half of its never-trained p cells explode. The locked bar cannot yet be read, because for three knobs the unchanged source track already scores better than the half-depth pseudoreplicate oracle, so the one-warp → oracle interval is negative. The count-unchanged control has no direct reading; on count-identical never-trained pairs the old design A does worse than its twin, so count predictions can move.

<!-- crux:ledger:start -->
**4 children** · open 4, leaning 0, settled 0, parked 0

- `h7` [[h7_conditioning_on_the_recorded_control_to_|Conditioning on the recorded control-to-treatment scaling predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h8` [[h8_conditioning_on_the_recorded_control_ide|Conditioning on the recorded control identity predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h9` [[h9_conditioning_on_the_recorded_control_dep|Conditioning on the recorded control depth predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
- `h10` [[h10_conditioning_on_the_recorded_fragment_ex|Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does]] — *open*
<!-- crux:ledger:end -->
