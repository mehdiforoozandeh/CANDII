---
title: "E5: scorer extensions (foreign covariates at scoring, artifact variance, per-stratum values, robust variance ratio)"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# E5: scorer extensions

## 1. What the PI asked

- Foreman log, `.orchestrate/plan.md`:3103 (PI rulings, 2026-11-16T23:40Z): "(i) add a scorer option for foreign covariates at scoring time: yes; (ii) scorer extensions (artifact variance by platform/antigen/scale, per-stratum values, robust variance-ratio aggregate): yes, all three."
- `peeling the onion.md`:13-14: "depending on the data-generation process and the type of experiment, this category can be broken down to more granular sources of artifact variation" / "e.g. lab, sequencing technology, antigen, etc."
- `peeling the onion.md`:60-61: "e.g. we can measure how much variance we are predicting in different subsets of the genome: e.g.. promoters, enhancers, gene bodies, foreground, background, heterochromatin, etc." / "same for the amount of variance caused by different artifact sources: depth, sequencing platform, antigen efficiency, etc."
- `miniCANDI architecture ablations.md`:78: "Scrambled covariates, as a control for covariate use."

Success criterion served: 3 (well defined and tested evals). Related items: A2 (scrambled covariates), E3 (computed metrics report, sections 10 to 12), D1 (lab, antibody and donor covariates).

## 2. Question

Four extensions of the scorer, with no change to any existing value:
1. Does a trained base model use its covariates at test time? Give each target the covariates of another experiment of the same assay at scoring and compare the scores with the true-covariate scores, with no new training.
2. How much model error and data variance goes with each artifact source (platform, antibody, lab) and with each scale?
3. What is the calibration in every region stratum, not only in the worst one?
4. What is the predicted-over-replicate variance ratio when single targets do not dominate the mean?

## 3. State today (verified 2026-11-16 in the code and on Fir)

- Foreign covariates. The scrambled-covariates control permutes covariate rows among the windows of a training batch (`covariates.scramble_covariates`). At scoring every batch tiles one job (`tools/minicandi/predict.py`:299-305, `score.py` `_job_arrays`, `predict_store.request_arrays`), so the permutation is an identity: the A2 arm is "trained with scrambled covariates, scored with true covariates" (`run_state/ablation_specs/11_scrambled_covariates.md` section 6). No scorer option existed.
- Covariates in `tables/eic_v1/measurements.tsv` on Fir (2,740 rows): `platform` filled for 2,456 rows (22 values, OBI ids); `lab` filled for 78 rows only; `antibody` and `donor` empty. Item D1 (branch `chunk/onion-CO-D1`) fills lab, antibody and donor in a new table version.
- Artifact variance (E3 report section 10, lines 327-328): no platform or antigen split; no by-scale split of model error in `scores.json` (the error budget is stored summed over the four Haar bands). The data-side decomposition `mixed_model.py` fits cell, experiment, replicate and library; lab was never fitted.
- Per stratum (E3 report line 332): `scores.json` keeps only `pit_deviation_worst_stratum`. `suffstats.npz` already holds the PIT curve of every target in every stratum (`pit/<regime>/<target>/<stratum>`).
- Variance ratio (E3 report line 349): `predicted_over_replicate_variance` is a mean of per-target ratios; one H3K27ac target of `rung2_base_heldout/seed0` has a ratio of 47.07. The per-target ratios are stored (`per_target.<t>.per_rung.pseudoreplicate`), their numerators and denominators are not.
- `selection.composite` in every `scores.json` is composite version 1, with the superseded half-read correction term (E3 report line 351). The current composite (version 3) comes from `select_rule.py arm-summary --runs <runs root> --sub <sub>`.

## 4. Prior leads

- Hiding depth on the input side lowered the U-Net composite (version 2) from 0.444 (range 0.030) to 0.382 (range 0.009) (`results/views_ablation/VIEWS_ABLATION.md` line 20). Depth alone matters, so every foreign mode exists with depth kept and with depth swapped.
- Pointwise MLP, scrambled-all (depth offset included): composite -0.086, beyond the range; scrambled covariates without depth: specific variance -0.019 (`ablation_specs/11_scrambled_covariates.md` prior leads 2 and 3). Prediction: most covariate use is depth use, so `*_keep_depth` modes move the scores little and `both_swap_depth` moves them much.
- U-Net trained with scrambled covariates scored better on all 102 correction pairs (+0.101, beyond; `results/ladder_audit/COVERAGE.md` row 11). Prediction: on the U-Net, foreign covariates with depth kept move the correction term within the seed range.

## 5. Work (done on branch `chunk/onion-CO-E5`)

1. Foreign covariates (`tools/onion_eval/score.py`, `predict_store.py`, `slurm/minicandi/submit_arms.sh`). Option `--foreign-covariates MODE [--foreign-seed N]` (env `ONION_FOREIGN_COVARIATES`, `ONION_FOREIGN_SEED`; submit variable `FOREIGN_COVARIATES`). Off by default.
   - Donor: for each measurement that a model window reads, the base view of a measurement of the same assay in another cell (`in_eic` train or val), drawn once per measurement with a seeded generator from the candidates sorted by id (`foreign_donors`). Fixed for a seed, independent of job order, logged per swapped side and measurement in `scores.json` `foreign_covariates.assignment.<in|out>` (donor, donor cell, the covariate columns that changed), with counts per column in `n_changed_by_column.<side>`.
   - Modes (`FOREIGN_MODES`): side `in` (the inputs' C), `out` (the target's C'), `both`; depth `keep_depth` (column 0, log2 depth, stays true, so the count transform and the NB depth offset are unchanged and only the other covariates are foreign) or `swap_depth` (column 0 is the donor's at the same half-read scale, so the depth offset moves with it). Six modes.
   - Assays with no donor (no base view of the assay in another cell): on `eic_v1` five assays, H2AK9ac, H3K9me2, H3F3A, H3T11ph and H4K12ac, one measurement each among the 521 that the screening jobs read (read-only check on Fir, 2026-11-16). Their rows stay true; they are listed per side in `foreign_covariates.assays_without_donor` and stay in every table and macro, so a foreign macro covers the same assays as the normal one in every mode.
   - Job names: a foreign scoring runs as `mc_<run>_s<seed>_fc<mode>_s<donor seed>`, so a queued normal scoring or another mode never blocks it.
   - `select_rule.py table` and `arm-summary` refuse a table that mixes foreign and normal scores (or two modes or donor seeds) and add a `foreign_covariates` column only when a score carries the key; a normal table is unchanged.
   - Only the model's forward reads the foreign rows. The reference predictors, the truth, the pair depth ratio (headline 5), the copy-leak guardrail and the two-step base leg keep the true covariates.
   - Separation from normal scores: the score directory name must hold `foreign` (score.py refuses any other `--out`, a directory holding a score of another mode or donor seed, and a normal score over a foreign one); the prediction store path is `.../<run>[/same_view]/foreign_<mode>_s<seed>/seed<s>/<sub>`; `STORE.json` records the mode and seed and the metric phase refuses any other store; `select_rule.scorer_keys` gains the key `foreign_covariates` (written only when present), so `check_scorer_keys`, `arm-summary` and `arm_report.py` refuse to mix a foreign score with a normal one.
2. Artifact variance.
   - Data side: `mixed_model.py --extra-factors lab,platform,antibody` adds those columns as factors crossed with cell (a bootstrap draw keeps their labels). A column that the table lacks, or that is empty for some observation of the assay, gets a row with estimable 0 and the note "not available in this table version". Off by default: the rows are unchanged.
   - Model side, at report time (`tools/onion_eval/report_extras.py`, new): `by-covariate` groups per-target CRPS skill, PIT deviation, variance recovered and variance ratio by the target's lab, platform, antibody and donor; `by-scale` gives, per assay, part (shared, specific) and Haar band, the truth variance, the error, the explained part and the added part from the stored Gram matrices (`suffstats.npz gram/...`; the sampling noise is not separated there).
3. Per stratum: new key `pit_deviation_per_stratum` (one metric block per stratum, both imputation tables). Report time: `report_extras.py strata` gives the same values from `suffstats.npz` for every score already stored.
4. Robust variance ratio: new key `predicted_over_replicate_variance_robust` with `median_of_targets` and `ratio_of_means` (per assay; macro = mean over scorable assays); `metrics.variance_ratio` also returns its pooled numerator and denominator. The existing key is unchanged. Report time: `report_extras.py variance-ratio` gives the median for every stored score; the ratio of means needs a scoring with the E5 scorer.
5. `selection.composite_status` (new, additive) labels `selection.composite` as composite version 1, superseded, and names `select_rule.py arm-summary`. The value of `selection.composite` is unchanged and old files are not rewritten.

## 6. How the result answers the question

Acceptance checks:
- With every switch off, `scores.json` equals the pre-E5 one except for the keys `pit_deviation_per_stratum`, `predicted_over_replicate_variance_robust` (top level, `accessibility_only`, `predicted_control`) and `selection.composite_status` (`score.drop_e5_keys` removes exactly these; the four tests that pin pre-E5 outputs use it); `suffstats.npz`, `pairs_per_region.npz` and the prediction-store arrays are byte-identical (legacy model path, store path, baselines; synthetic tables).
- Every new code path has a unit test on synthetic tables (`tests/test_onion_eval_scorer_extensions.py`).
- With `out_keep_depth`, a model that reads only C' column 0 scores exactly as with true covariates; with `out_swap_depth` its scores change.
- The 9 base seeds are scored in at least the `both_keep_depth` and `both_swap_depth` modes (script below).

"Beyond" = difference of seed means larger than the larger seed range (max minus min over 3 seeds), per architecture, against `p16_sv` of the same runs.

| outcome (foreign minus true covariates) | conclusion |
|---|---|
| `*_keep_depth`: correction term lower, beyond, on an architecture | that model uses non-depth covariates to correct; the correction metric tests covariate use |
| `*_keep_depth`: all terms within on all 3 | the base models do not use the covariates beyond depth at test time (agrees with A2 if its arm is within too) |
| `both_swap_depth` lower, beyond, while `both_keep_depth` within | covariate use is depth use |
| `in_keep_depth` and `out_keep_depth` differ in sign or size | the input and output covariate paths carry different information; report both |
| imputation term lower, beyond, under `keep_depth` | covariates carry imputation information (batch or processing) |

Results so far (report time, from the stored `p16_sv` files of the 9 base seeds; Fir `/project/def-maxwl/mforooz/ONION/closeout_E5/tables/`, summaries copied to `cruxvault/results/closeout_E5/`; mean (seed range)):
- Variance ratio at the pseudoreplicate rung, macro over assays. Mean of per-target ratios (stored key): conv autoencoder 2.455 (1.536), conv + attention 2.356 (0.570), U-Net 2.495 (0.521). Median of per-target ratios: 1.569 (0.062), 1.521 (0.200), 2.107 (0.523). The median removes most of the seed range of the two conv models; on H3K27ac the conv autoencoder goes from 7.903 (13.322) to 1.474 (0.271). The ratio of means needs a scoring with the E5 scorer.
- PIT deviation per stratum (14 strata), macro over assays, for example promoters: 0.049 (0.008), 0.038 (0.004), 0.067 (0.006); gene bodies: 0.025 (0.007), 0.020 (0.005), 0.027 (0.007). The per-assay maximum over the strata equals the stored worst-stratum value to 3e-17 (checked on `rung2_base_heldout/seed0`, H3K4me3; 198 assay-seed cells agree to the 6 digits of the table).
- By covariate: lab, antibody and donor give "not available in this table version" (`eic_v1`); platform has 8 values and some empty cells among the scored targets.

## 7. Dependencies, cost, risks

- Depends on nothing for items 1, 3, 4. Lab and antibody (item 2) need the D1 table version: until then their rows say "not available in this table version" (lab is filled for 78 of 2,740 rows today; platform is filled for 2,456).
- Cost of the foreign-covariate pass (Fir, measured on the `p16_sv` scoring of the same 9 seeds, `sacct` jobs 4389803 to 4389848 and the metrics jobs): 36 predict jobs used 88.4 CPU-h and 9 metrics jobs 81.5 CPU-h, so one mode costs 169.9 CPU-h and 45 jobs (9 seeds x (4 + 1)). Four modes: 180 jobs, 4 x 169.9 = 679.6 CPU-h. The optional normal rescore (`TRUE_RESCORE=1`, for the ratio of means) adds 45 jobs and 169.9 CPU-h.
- Risk 1: in EIC, platform, lab and antibody are mostly constant within an experiment and one experiment per cell, so the data-side fit marks them confounded until D2 brings second experiments. The model-side grouping is descriptive (no biology held fixed).
- Risk 2: a donor whose covariates equal the target's except depth makes `keep_depth` an identity for that target; `foreign_covariates.n_with_no_column_changed` counts them.
- Risk 3: `swap_depth` on side C' rescales the NB mean by the depth ratio; its scores mostly measure the depth law, not covariate use.

## 8. Done when

The code is merged, the 9 base seeds are scored in the foreign modes the foreman submits, `select_rule.py arm-summary --sub p16_foreign_<mode>` is read against `p16_sv` with the table above, and the per-stratum, variance-ratio, by-scale and by-covariate tables of the base seeds are in the close-out report (R2).
