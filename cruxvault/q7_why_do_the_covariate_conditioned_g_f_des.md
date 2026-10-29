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

The depth law: on a pair of runs that differ only in sequencing depth, the predicted count scale (log2 of the sum of predicted means over the sum of source counts) should equal the log2 depth ratio. Every g → f design in [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]] and in [[q5_does_a_depth_offset_on_f_the_source_dept|the depth-offset grid]] meets it on trained base ↔ arm pairs and misses it on never-trained arm → arm pairs. The +1 pseudocount, the loss and the encoder-decoder designs tried so far do not close the miss; chaining two trained maps through the base does. Exploratory (PI 2026-10-19): no bars. It is answered when removing one cause brings never-trained depth pairs to the trained-pair error, and keeping it does not.

## Question

Why do the covariate-conditioned g → f designs miss the depth law on depth-only pairs?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Exploratory (PI 2026-10-19): readings only. No bars, no ticks, no verdicts.

## Answer so far

Readings to 2026-10-29, not judged: median |log2 error| of the predicted scale, chr19 + chr21; the seed spread is the noise floor.

- The data obey the law within 0.71% over 84 ordered pairs ([audit](results/t122/AUDIT.md)).
- Trained pairs fit (0.04–0.12); never-trained arm → arm pairs miss (0.19–0.55), in every design and g version ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]).
- The +1 pseudocount is not the cause: rung A0 can express the law and still misses (0.38–0.46, trained 0.12) ([[h18_the_1_pseudocount_in_x_log_1_counts_make|the pseudocount hypothesis]]).
- The loss is not the cause: a Poisson-NLL mean leaves the miss (A 0.29, A0 0.38, D 0.28; seed wobble 0.05–0.16) ([[h21_fitting_the_mean_by_poisson_nll_and_the_|the loss hypothesis]]).
- Encoder-decoder designs do not close it yet ([[h22_a_composable_g_one_shared_map_from_each_|the composable-g hypothesis]]). Encoder-decoder D gives 0.26–0.52, old D 0.29–0.34; encoder-decoder A also loses trained pairs (1.0–1.4). A symmetric design through a learned clean setting gives 0.29–0.42 in its two complete cross-loss settings (first pass).
- Chaining old D's two trained maps through the base meets the law: 0.07 (across-track g) and 0.12 (per-track g, seed spread 0.05) against direct 0.34 and 0.29. The trained maps hold the law; g's direct answer on an unseen pair does not use it.
- Base ↔ 7.5M pairs, held out of training, are easy (0.04–0.16) except for encoder-decoder A.
- Size is not the cause at the widths tried ([[h20_a_larger_g_with_the_same_training_pairs_|the g-size hypothesis]]). A 4× wider g misses on never-trained depth pairs (0.63 per-track, 0.46 across), and so does a 2× wider f (0.41, 0.34; [[h27_f_at_64_channels_2x_wider_with_the_same_|the f-size hypothesis]]), within noise of old D. More seeds for the wider f are next (PI, 2026-10-29); the across-track g is dropped from the gain test. Through the base, both stay near the law (0.09–0.21). Arm → arm training is not decided.

<!-- crux:ledger:start -->
**8 children** · ideas 2/8 done (supported 0, partial 0, refuted 2, inconclusive 0, invalid-run 0)

- `h18` [[h18_the_1_pseudocount_in_x_log_1_counts_make|The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law]] — *done* — verdict **refuted**, metric `never-trained depth miss, rung A0 (no +1): 0.46 (NB) / 0.38 (Poisson-mean) median |log2 err| vs trained 0.12`
- `h19` [[h19_g_fits_trained_base_arm_depth_pairs_but_|g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs]] — *idea*
- `h20` [[h20_a_larger_g_with_the_same_training_pairs_|A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs]] — *idea*
- `h21` [[h21_fitting_the_mean_by_poisson_nll_and_the_|Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not]] — *done* — verdict **refuted**, metric `never-trained depth miss with Poisson-mean loss: A0 0.38, D 0.28 median |log2 err| vs trained 0.07–0.12`
- `h22` [[h22_a_composable_g_one_shared_map_from_each_|A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training]] — *idea*
- `h23` [[h23_on_never_trained_arm_arm_depth_pairs_the|On never-trained arm → arm depth pairs, the shared-weight encoder-decoder FiLM CNN meets the depth law at the trained-pair level]] — *idea*
- `h24` [[h24_the_depth_entries_in_g_s_covariate_input|The depth entries in g's covariate input cause the conditioned generator to miss the depth law on never-trained depth pairs]] — *idea*
- `h27` [[h27_f_at_64_channels_2x_wider_with_the_same_|f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs]] — *idea*
<!-- crux:ledger:end -->
