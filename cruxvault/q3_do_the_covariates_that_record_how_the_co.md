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
If two p-value tracks were called from the very same reads but with a different background or a different smoothing, can knowing that setting translate one into the other?

## TL;DR
Four choices act at or after peak calling and leave the treatment counts bit-identical to the base: the MACS2 control-to-treatment ratio, which control library is used (matched, another cell's, or none), how deeply that control was sequenced, and how far each read is extended into a fragment. For each we hold a counterfactual pair built by the ENCODE pipeline with exactly that one choice changed and its value recorded ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The question is whether a transformation conditioned on the source and target values predicts the other member's −log10 p signal better than one-warp, the covariate-free monotone map on the p axis fitted on training pairs and scored on held-out pairs, by the margin locked in [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]]. Because the counts do not move, every child carries the same control: a transformation whose count prediction moves has produced an invalid run, not a refutation. It answers yes only if every child holds; a refuted child names a knob whose recorded value does not carry the information.

## Question

Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?

## Protocol
Locked before any run: gap-closed is read on the −log10 p axis only; the count-unchanged control and the shuffled-covariate twin are required on every child.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**4 children** · ideas 0/4 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h7` [[h7_conditioning_on_the_recorded_control_to_|Conditioning on the recorded control-to-treatment scaling predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h8` [[h8_conditioning_on_the_recorded_control_ide|Conditioning on the recorded control identity predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h9` [[h9_conditioning_on_the_recorded_control_dep|Conditioning on the recorded control depth predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h10` [[h10_conditioning_on_the_recorded_fragment_ex|Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
<!-- crux:ledger:end -->
