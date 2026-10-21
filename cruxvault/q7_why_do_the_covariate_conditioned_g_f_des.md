---
id: q7
type: question
schema: 2
title: Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?
parent: q1
status: open
stale: true
created: "2026-10-19T23:40:57"
updated: "2026-10-19T23:40:57"
---

# q7 — Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

When the model turns a track sequenced at one depth into the same track at another depth, its predicted total should scale with the depth ratio; it does not, and we want to know why.

## TL;DR

The depth law: on a pair of runs that differ only in sequencing depth, the predicted count scale (log2 of the sum of predicted means over the sum of source counts) should equal the log2 depth ratio. Every g → f design in [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]] and in [[q5_does_a_depth_offset_on_f_the_source_dept|the depth-offset grid]] misses it. This question collects candidate causes (the +1 pseudocount in log(1 + counts), a failure to carry trained pairs over to never-trained pairs, a g too small, the loss) and candidate fixes. Exploratory (PI 2026-10-19): no bars. It is answered when one cause is identified: removing it brings never-trained depth pairs to the error trained depth pairs already show, and keeping it does not.

## Question

Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory (PI 2026-10-19): readings only. No bars, no ticks, no verdicts.

## Answer so far

Observed 2026-10-19, not judged.

- The real data obey the law: on chr19 + chr21 outside the blacklist, each depth arm's count total matches the manifest depth ratio within 0.71% over all 84 ordered pairs of the 7 tracks ([audit](results/t122/AUDIT.md), [[t122_hand_check_the_depth_law_computatio|the depth-law hand check]]).
- The audit also reported that it recomputed all 24 depth-law check values exactly; that recomputation left no saved output.
- Trained depth pairs fit the law; never-trained arm → arm depth pairs miss it, in every design and both g versions ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]).
- A fix was discussed and not decided: train on arm → arm pairs too, so that the law-test pairs are no longer unseen.
- The loss is not the cause: a Poisson NLL for the mean removes rung A's total-scale bias but leaves the never-trained depth miss (A 0.29, A0 0.38, D 0.28 log2) ([report](results/t127/report.md); PI reading 2026-10-20).
- The +1 is not the cause either: rung A0, which can express the law exactly, misses never-trained pairs as much or more (0.38–0.46 log2).

<!-- crux:ledger:start -->
**5 children** · ideas 2/5 done (supported 0, partial 0, refuted 2, inconclusive 0, invalid-run 0)

- `h18` [[h18_the_1_pseudocount_in_x_log_1_counts_make|The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law]] — *done* — verdict **refuted**, metric `never-trained depth miss, rung A0 (no +1): 0.46 (NB) / 0.38 (Poisson-mean) median |log2 err| vs trained 0.12`
- `h19` [[h19_g_fits_trained_base_arm_depth_pairs_but_|g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs]] — *idea*
- `h20` [[h20_a_larger_g_with_the_same_training_pairs_|A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs]] — *idea*
- `h21` [[h21_fitting_the_mean_by_poisson_nll_and_the_|Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not]] — *done* — verdict **refuted**, metric `never-trained depth miss with Poisson-mean loss: A0 0.38, D 0.28 median |log2 err| vs trained 0.07–0.12`
- `h22` [[h22_a_composable_g_one_shared_map_from_each_|A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training]] — *idea*
<!-- crux:ledger:end -->
