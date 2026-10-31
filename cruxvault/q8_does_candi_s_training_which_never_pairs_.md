---
id: q8
type: question
schema: 2
title: Does CANDI's training, which never pairs an experiment with a counterfactual of itself except in depth, leave the trained model unable to correct for a change in the other covariates?
parent: q1
status: open
stale: false
created: "2026-10-25T13:48:35"
updated: "2026-10-25T13:48:35"
---

# q8 — Does CANDI's training, which never pairs an experiment with a counterfactual of itself except in depth, leave the trained model unable to correct for a change in the other covariates?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If a model only ever learns from each measurement next to a copy of itself, can it learn how a different lab setting would change that measurement?

## TL;DR

The round-trip-only arm of [[t128_build_and_run_phase_1_of_the_symmet|the symmetric encoder-decoder runs]] collapsed to near identity, because its source and target covariates are equal. CANDI's training has the same gap for every covariate but depth: the target of an unmasked assay is the same experiment, so only depth differs, by independent binomial thinning (`dataset.py:43-65`, `store/dataset.py:142`). A covariate-swap probe on a trained checkpoint, against the counterfactual arms' real tracks, settles it ([[h26_a_trained_candi_checkpoint_told_a_counte|the probe]]).

## Motivation

The PI named this as the next line of work on 2026-10-25, the CANDI covariate-collapse investigation; it is the direct bridge from the testbed back to CANDI. The question text is a code fact the agent read from the data loader. The PI delegated the design: "based on your findings, you recommend the null and verifiables".

## Question

Does CANDI's training, which never pairs an experiment with a counterfactual of itself except in depth, leave the trained model unable to correct for a change in the other covariates?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory until the PI sets bars on the child hypothesis.

## Answer so far

Prior evidence, not judged (2026-10-25). All pre-CANDII numbers are single seed, 12 held-out targets.

- Depth is not ignored, and is not learned: the offset head sets `log2 mu = (d - centre) + eta` (`decoder.py:580-585`), told-depth slope 1.0000. With the offset off, the model learns a slope of 0.89-1.03 from the thinning pairs (`AGENTS.md` §7.1).
- Offset on: the output side ignores the other covariates (assay real→real max |Δη| 0.0023; run_type CI [-0.00066, +0.000087]; §7.1). Weight decay had erased those weights; at weight decay 0 they live, but the run_type direction score is 0.559, about chance (archive h47).
- Offset off: run_type moves imputation the right way (direction 0.69, CI above 0, sign test p = 0.039), but run type, read length and depth are collinear on the panel, so this may carry depth (archive h42, §7.2 item 5).
- Across 90 EIC covariates, none clears its detection floor on counts (archive h60).
- A synthetic testbed whose training held every input-by-output transform pair learned steering (M2 ~0.5 versus 0.02 pooled; archive q16, h30).

The covariate-swap probe ([[h26_a_trained_candi_checkpoint_told_a_counte|the probe]]: t81 checkpoints, chr21, three runs, 2026-10-25/26) shows that the predicted level moves when told read length or run type, so 'ignores' is wrong for the level. Per-region tracking stays unreadable: the depth control failed in every run, and no split-half floor statistic tried separates a real effect from pure thinning. Parked as the baseline (PI ruling 2026-10-29); the test of what CANDI needs moved to [[h28_training_g_on_arm_to_arm_pairs_lets_a_di|the arm-to-arm training claim]].

PI reading 2026-10-31: CANDI uses read length and run type in the wrong direction; told either one, it moves its predicted level, but not the way the real tracks move. This is the current stage of CANDI, unrelated to the covariate artifact correction (CAC) runs on the testbed. The missing counterfactual pairs fit this, but low conditional information and collinearity are joint causes. Dedup, MAPQ, fragment and control knobs are not CANDI inputs at all.

<!-- crux:ledger:start -->
**1 children** · open 1, leaning 0, settled 0, parked 0

- `h26` [[h26_a_trained_candi_checkpoint_told_a_counte|A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates]] — *open*
<!-- crux:ledger:end -->
