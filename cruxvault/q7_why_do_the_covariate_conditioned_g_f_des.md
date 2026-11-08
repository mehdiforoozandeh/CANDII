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

When the model changes a track's sequencing depth, its predicted total should scale with the depth ratio. It does for the depth changes it was trained on, not for new ones, and we want to know why.

## TL;DR

CAC: why corrections miss the depth law. The depth law: on a pair of runs that differ only in sequencing depth, the predicted count scale (log2 of the sum of predicted means over the sum of source counts) should equal the log2 depth ratio. Every g → f design in [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]] and in [[q5_does_a_depth_offset_on_f_the_source_dept|the depth-offset grid]] meets it on trained base ↔ arm pairs and misses it on never-trained arm → arm pairs. The +1 pseudocount, the loss, the encoder-decoder designs and the sizes of g and f tried do not close the miss; chaining two trained maps through the base does. Exploratory (PI 2026-10-19): no bars. It is answered when removing one cause brings never-trained depth pairs to the trained-pair error, and keeping it does not.

## Motivation

The PI first asked on 2026-10-02, reviewing the capacity-grid deck: "aren't we predicting negative binomial distribution parameterized by eta and log depth. so by definition the model must follow the depth law. why isn't that happening?" A hand-check of the computation came first, before any design was blamed. The question opened on 2026-10-19 with the PI's goal: "depth law is the simplest kind of technical artifact that we want our models g (and it's output f) to be learning near perfectly."

Design: the PI named four suspects (the +1 pseudocount, extrapolation in g, the loss, the check itself) and had them tested in parallel, one hypothesis each; later sessions added g's input wiring, the size of g and f, and arm-to-arm training. The PI framed the target as "make g generalize with the least required number of training examples", and ruled the work exploratory: "we don't want to set a bar for now".

## Question

Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory (PI 2026-10-19): readings only. No bars, no ticks, no verdicts.

## Answer so far

Readings to 2026-10-31, not judged: median |log2 error| of the predicted scale, chr19 + chr21; the seed spread is the noise floor. The children's findings hold their own numbers.

- The data obey the law within 0.71% over 84 ordered pairs ([audit](results/t122/AUDIT.md)).
- Trained fit against never-trained miss: [[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]. Chaining old D's two trained maps through the base gives 0.07 (across-track g) and 0.12 (per-track g, seed spread 0.05) against direct 0.34 and 0.29: the trained maps hold the law, and g's direct answer on an unseen pair does not use it.
- Not the cause: [[h18_the_1_pseudocount_in_x_log_1_counts_make|the pseudocount]], [[h21_fitting_the_mean_by_poisson_nll_and_the_|the loss]] (seed wobble 0.05–0.16), [[h22_a_composable_g_one_shared_map_from_each_|the composable g]] and [[h23_on_never_trained_arm_arm_depth_pairs_the|the symmetric encoder-decoder]], and size at the widths tried ([[h20_a_larger_g_with_the_same_training_pairs_|g]], [[h27_f_at_64_channels_2x_wider_with_the_same_|f]]).
- Base ↔ 7.5M pairs, held out of training, are easy (0.04–0.16) except for encoder-decoder A.
- Through the base, the wider g and f stay near the law (0.09–0.21). With 6 seeds, the wider f's CRPS gain does not replicate on the new seeds. The PI closed the scale question at these sizes (2026-10-31); the across-track g is dropped from the gain test.
- Arm → arm training: [[h28_training_g_on_arm_to_arm_pairs_lets_a_di|the arm-to-arm hypothesis]].

<!-- crux:ledger:start -->
**10 children** · open 2, leaning 2, settled 6, parked 0 · 1 finding(s) on this question

- `h18` [[h18_the_1_pseudocount_in_x_log_1_counts_make|The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law]] — *settled · strong*
- `h19` [[h19_g_fits_trained_base_arm_depth_pairs_but_|g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs]] — *leaning · moderate*
- `h20` [[h20_a_larger_g_with_the_same_training_pairs_|A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs]] — *settled · strong*
- `h21` [[h21_fitting_the_mean_by_poisson_nll_and_the_|Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not]] — *settled · strong*
- `h22` [[h22_a_composable_g_one_shared_map_from_each_|A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training]] — *settled · strong*
- `h23` [[h23_on_never_trained_arm_arm_depth_pairs_the|On never-trained arm → arm depth pairs, the shared-weight encoder-decoder FiLM CNN meets the depth law at the trained-pair level]] — *settled · strong*
- `h24` [[h24_the_depth_entries_in_g_s_covariate_input|The depth entries in g's covariate input cause the conditioned generator to miss the depth law on never-trained depth pairs]] — *open*
- `h27` [[h27_f_at_64_channels_2x_wider_with_the_same_|f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs]] — *settled · strong*
- `h28` [[h28_training_g_on_arm_to_arm_pairs_lets_a_di|Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base]] — *leaning · moderate*
- `h30` [[h30_chaining_one_shared_count_space_correcti|Chaining one shared count-space correction model through the base transfers to never-trained arm-to-arm pairs across every count-moving knob, including cross-knob pairs, better than the direct route, and a pivot-consistency loss lets the direct route match the chained route]] — *open*
- `f23` finding (moderate) [[f23_consistent_across_every_design_loss_and_|Consistent across every design, loss and size tried, with the PI closing the…]]
<!-- crux:ledger:end -->
