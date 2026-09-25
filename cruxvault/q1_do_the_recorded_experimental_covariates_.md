---
id: q1
type: question
schema: 2
title: Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?
parent: root
status: open
stale: false
created: "2026-09-01T02:16:11"
updated: "2026-09-01T02:16:11"
---

# q1 — Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?

Parent:: [[candii]]

## ELI5

If two labs measure the same thing with different machine settings, can the settings alone
tell you how to turn one lab's numbers into the other's?

## TL;DR
CANDI is conditioned on experimental covariates on both the input and the output side, and its zero-shot claims rest on that conditioning doing real work. Inside CANDI the question cannot be answered: the covariate pathway and the imputation trunk are entangled, and the recorded depth response is a closed-form thinning identity (`AGENTS.md` §7.2 rule 3). This question asks it in isolation, on material where the truth is known: ten processing knobs, each with a counterfactual pair built by the ENCODE pipeline with exactly that one choice changed and its value recorded ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The thing under test is any transformation conditioned on the source and target covariates — the encoder–decoder CANDI implies is one instance; an affine map or a small conditioned network are others — asked for the target's raw counts and −log10 p signal. Every headline is measured against one-warp, a covariate-free monotone map on each output axis fitted first on training pairs and scored on held-out pairs, as gap-closed on the one-warp → oracle interval; the bar (PI ruling 2026-09-16) is ≥ 0.5 per mark class with a gain above 2× the model's own seed |Δ|. Depth and run type are direct children; the reads-in-the-pileup knobs, the control-and-caller knobs, and the capacity ladder are sub-questions. It is settled by which knobs' covariates close the gap and how much capacity that takes.

## Question

Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Locked before any run under this question:

- The competitor is **not** the uninformed baseline. A covariate-free monotone map on the value
  axis is fit first, and every headline is measured against it.
- No child of this question may quote a genome-wide **correlation** as a headline metric. Depth and
  run type push the level in opposite directions and partly cancel, so a correlation over the whole
  genome yields a false null.
- The conditional entropy among the covariates is measured on the **panel actually used**, before
  any steering result from that panel is read. `AGENTS.md` §7.2 rule 5 records
  `H(run_type | assay_id, read_length) = 0.000 bits` on the shipped 8-assay panel (n = 26 `T_`
  records) against 0.551 bits on the full EIC panel, and `DATA.md` (§"The exposure covariates are
  collinear") makes the same requirement.
- Every number quotes its noise floor and the panel and recipe that floor was measured on
  (`AGENTS.md` §7.2 rule 2).

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**6 children** · ideas 0/3 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0) · sub-questions 0/3 resolved

- `h1` [[h1_conditioning_on_the_recorded_sequencing_|Conditioning on the recorded sequencing depth predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h2` [[h2_conditioning_on_the_recorded_run_type_pr|Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h15` [[h15_one_transformation_conditioned_on_the_so|A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates]] — *idea*
- `q2` _(Q)_ [[q2_do_the_covariates_that_record_which_read|Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?]] — *open*
- `q3` _(Q)_ [[q3_do_the_covariates_that_record_how_the_co|Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?]] — *open*
- `q4` _(Q)_ [[q4_how_much_capacity_does_a_covariate_condi|How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?]] — *open*
<!-- crux:ledger:end -->
