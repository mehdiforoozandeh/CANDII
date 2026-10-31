---
id: q1
type: question
schema: 2
title: Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?
parent: root
status: open
stale: false
created: "2026-09-01T02:16:11"
updated: "2026-10-02T12:29:01"
---

# q1 — Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?

Parent:: [[candii]]

## ELI5

If two labs measure the same thing with different machine settings, can the settings alone
tell you how to turn one lab's numbers into the other's?

## TL;DR
CANDI's zero-shot claims rest on its covariate conditioning, which cannot be tested in isolation inside CANDI. This question asks it where the truth is known: 7 tracks, each a base plus arms that change one of ten processing knobs ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The first claims, on depth and run type, used one-warp, a covariate-free monotone map, as the competitor. Since 2026-09-25 [[h15_one_transformation_conditioned_on_the_so|the main claim]] and the capacity, depth-offset and depth-law sub-questions use the same model trained with scrambled covariates. The sub-questions ask which knobs carry the information, how expressive the transformation must be, whether a depth offset helps, how −log10 p follows from counts, and why the depth law is missed. It is settled by which knobs' covariates beat that twin on unseen pairs, and at what capacity.

## Motivation

The PI's reading of the ENCODE Imputation Challenge: "experimental covariates have a huge impact on the shape and magnitude of the signal and that can be far beyond biology" (2026-09-01). CANDI's zero-shot claim rests on conditioning that corrects exactly this, so the PI asked whether a model "accurately predicts A' from A given both of their covariates", and chose to answer it in a testbed where the truth is known, outside CANDI.

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

Exploratory readings (PI ruling), not judged. Bars are 2 × the 3-seed wobble.

- Trained pairs: the main claim's chosen designs beat the scrambled-covariate twin in 5–6 of 6 mark-class cells ([report](results/h15/report.md)).
- Never-trained arm → arm pairs: the median relative gain over the twin falls from 0.29 to 0.11–0.16 in counts and from 0.15 to 0.004–0.06 in −log10 p ([report](results/h19/report.md)).
- The swap check and the depth law are unmet in every chosen design, though the real data obey the depth law within 0.71%.
- [[q4_how_much_capacity_does_a_covariate_condi|Capacity]]: when g also reads the bin's value, design A improves in counts; B–D change little; −log10 p explodes more often.
- [[q5_does_a_depth_offset_on_f_the_source_dept|Depth offset]]: it helps only design A, mostly in scale error.
- [[q6_how_closely_does_a_map_h_reproduce_the_l|Counts to p]]: the peak caller's formula plus a learned MLP fits best (narrow marks, one track, log CRPS 0.0798 ± 0.0013).
- [[q7_why_do_the_covariate_conditioned_g_f_des|Depth-law miss]]: neither the +1 pseudocount nor the NB loss causes it; a composable encoder–decoder g ([[h22_a_composable_g_one_shared_map_from_each_|its claim]], [deck](presentations/t126-encdec/DECK.md)) did not close it.
- [[q2_do_the_covariates_that_record_which_read|The pileup knobs]] and [[q3_do_the_covariates_that_record_how_the_co|the control knobs]]: g beats its no-covariates twin on trained pairs in counts for every knob, and the p gain vanishes on never-trained pairs; neither question has scored its locked bar.
- The depth and run-type claims were never run in their own design.

<!-- crux:ledger:start -->
**12 children** · open 6, leaning 6, settled 0, parked 0 · 1 finding(s) on this question

- `h1` [[h1_conditioning_on_the_recorded_sequencing_|Conditioning on the recorded sequencing depth predicts a target track from a source track, beyond what a single value-axis map already does]] — *leaning · moderate*
- `h2` [[h2_conditioning_on_the_recorded_run_type_pr|Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does]] — *leaning · weak*
- `h15` [[h15_one_transformation_conditioned_on_the_so|A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates]] — *leaning · weak*
- `q2` _(Q)_ [[q2_do_the_covariates_that_record_which_read|Do the covariates that record which reads entered the treatment pileup carry enough information to map a track across a change in one of them?]] — *leaning · weak*
- `q3` _(Q)_ [[q3_do_the_covariates_that_record_how_the_co|Do the covariates that record how the control and the peak caller were configured carry enough information to map the p-value track across a change in one of them?]] — *leaning · weak*
- `q4` _(Q)_ [[q4_how_much_capacity_does_a_covariate_condi|How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?]] — *open*
- `q5` _(Q)_ [[q5_does_a_depth_offset_on_f_the_source_dept|Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?]] — *open*
- `q6` _(Q)_ [[q6_how_closely_does_a_map_h_reproduce_the_l|How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?]] — *open*
- `q7` _(Q)_ [[q7_why_do_the_covariate_conditioned_g_f_des|Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?]] — *leaning · moderate*
- `q8` _(Q)_ [[q8_does_candi_s_training_which_never_pairs_|Does CANDI's training, which never pairs an experiment with a counterfactual of itself except in depth, leave the trained model unable to correct for a change in the other covariates?]] — *open*
- `q9` _(Q)_ [[q9_which_recorded_processing_covariates_mov|Which recorded processing covariates move a track the most when changed one at a time, in peak height and in the shape of the signal?]] — *open*
- `q10` _(Q)_ [[q10_how_much_does_resampling_alone_move_a_tr|How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?]] — *open*
- `f9` finding (weak) [[f9_exploratory_readings_the_pi_has_not_judg|Exploratory readings the PI has not judged, and the depth and run-type tests…]]
<!-- crux:ledger:end -->
