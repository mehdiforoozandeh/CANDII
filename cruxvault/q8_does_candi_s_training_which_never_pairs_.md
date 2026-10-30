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

The round-trip-only arm of [[t128_build_and_run_phase_1_of_the_symmet|the symmetric encoder-decoder runs]] collapsed to near identity, because its source and target covariates are equal. This asks whether CANDI's own training has the same gap. In code it does for every covariate but depth: the target of an unmasked assay is the same experiment, so read length and run type are equal on both sides, and only depth differs, through independent binomial thinning (`dataset.py:43-65`, `store/dataset.py:142`). Training never pairs two experiments of one cell type and assay. It is settled by a covariate-swap probe on a trained checkpoint against the counterfactual arms' real tracks (the child hypothesis).

## Question

Does CANDI's training, which never pairs an experiment with a counterfactual of itself except in depth, leave the trained model unable to correct for a change in the other covariates?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory until the PI sets bars on the child hypothesis.

## Answer so far

Prior evidence, not judged (2026-10-25). All pre-CANDII numbers are single seed, 12 held-out targets.

- Depth is not ignored, and is not learned: the offset head sets `log2 mu = (d - centre) + eta` (`decoder.py:580-585`), told-depth slope 1.0000. With the offset off, the model learns a slope of 0.89-1.03 from the thinning pairs (`AGENTS.md` §7.1).
- With the offset on, the output side ignores the other covariates: assay real→real max |Δη| 0.0023, run_type CI [-0.00066, +0.000087] (§7.1). Weight decay had erased those weights; at weight decay 0 they live but the run_type direction score is 0.559, about chance (archive h47).
- With the offset off, run_type moves imputation the right way (direction 0.69, CI above 0, sign test p = 0.039). Run type, read length and depth are collinear on the panel, so this use is observational and may carry depth (archive h42, §7.2 item 5).
- Conditional information is small: across 90 EIC covariates, none clears its detection floor on counts (archive h60).
- In a synthetic testbed whose training held every input-by-output transform pair, steering was learned (M2 ~0.5 versus 0.02 pooled) (archive q16, h30).

Covariate-swap probe on the t81 checkpoints, chr21 (2026-10-25; null not yet approved). The output does move when told read length or run type: told 36 bp, the predicted level falls 0.22-0.89 log2 against a real 1.05-1.08; told PE, it moves 0.19-0.98 for ChIP and +0.52-0.70 for DNase, where the real tracks barely move (-0.08 to +0.01). Whether it tracks the per-region arm effect was not testable: no knob pair cleared the split-half reproducibility floor, the depth positive control failed (told depth also moves eta), and the floor statistic itself is biased down. So 'ignores' is wrong for the level; the per-region question stays open.

Rerun with the PI's repairs, chr21 (2026-10-26; prompts from non-blind arrays only, all 7 tracks kept). It is still not readable. Without deciles, the split-half floor calls 15 of 21 pure-thinning depth pairs reproducible, so it cannot tell a real read-length effect (36 bp r_split 0.28-0.36) from the pseudocount artefact. No run-type pair clears it. The depth control fails again on NLL direction (32 of 42 rows). The per-region question needs a floor statistic that thinning does not pass.

Second rerun with a curve-corrected floor, chr21 (2026-10-26). The pseudocount signal-level curve explains why thinning looked reproducible, but subtracting it pushes the thinning pairs below zero (-0.11 to -0.37), so the corrected floor is biased against any effect. No knob pair clears it; 36 bp read length shows a small residual signal (0.015-0.086). The per-region question is still not readable.

The probe is parked (PI ruling 2026-10-29); this reading stands as the baseline, and the test of what CANDI needs moved to [[h28_training_g_on_arm_to_arm_pairs_lets_a_di|the arm-to-arm training claim]].

Reading: CANDI is in the 'ignores' state for read length and run type, and the missing counterfactual pairs fit that, but low conditional information and collinearity are joint causes. Dedup, MAPQ, fragment and control knobs are not CANDI inputs at all.

<!-- crux:ledger:start -->
**1 children** · ideas 0/1 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h26` [[h26_a_trained_candi_checkpoint_told_a_counte|A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates]] — *idea*
<!-- crux:ledger:end -->
