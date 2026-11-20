# Ablation 11: scrambled covariates (`scrambled_covariates`)

Phase 1 specification, 2026-11-16. Close-out item A2, branch `chunk/onion-CO-A2`, base d65a980.

## 1. Question

The PI wrote (`miniCANDI architecture ablations.md`, "Ablations at every rung"): "Scrambled covariates, as a control for covariate use." D14 condition 1 (`long-horizon decisions.md` line 104) needs it at every rung with 3 seeds.

Question: do the conv autoencoder, conv + attention and U-Net use the covariates beyond the depth offset? A model that uses them must score worse when its covariates do not belong to its counts, above all on correction.

## 2. Arms

| arm | change from `final/rung{2,3,4}/reference_equal.json` | `simpler_than_base` |
|---|---|---|
| reference | none (existing runs `runs_fir/rung{2,3,4}_base_heldout`, 3 seeds) | |
| `scrambled_covariates` | `scramble_covariates: true` | `false` (a control: within the bars it settles as incumbent) |

What the switch does (`covariates.model_covariates`, `scramble_covariates`): once per forward, each track's covariate rows (C and C', depth included) are permuted among the windows of the batch where that track is present. Only the copies that the covariate encoders read are permuted. The count transform and the NB head keep each track's true depth, so this is the "covariates beyond the depth offset" form. The `scramble_all` form (depth offset also scrambled) is not run here; it ran on the pointwise MLP only (prior lead 2).

## 3. Prior leads

1. U-Net, old base recipe, Nibi, composite version 2 (`results/ladder_audit/COVERAGE.md` row 11): specific variance -0.029, beyond; gap closed on all 102 pairs +0.101, beyond, i.e. the control scored better. On the 18 pairs with no duplicate-keeping side the control was lower: 0.420 against 0.519 (`results/correction_metric_review/REVIEW.md` line 144), within 2 x 0.081.
2. Pointwise MLP, `scramble_all` (depth offset scrambled too): composite -0.086, beyond; uncertainty term -0.280, beyond (`COVERAGE.md` row 12). Prediction: most covariate use is depth use.
3. Pointwise MLP, `scramble_covariates`: specific variance -0.019, beyond; gap closed -0.021, within (`COVERAGE.md` row 11).

## 4. Expectation (before any run)

Reference values (Fir, `select_rule.py arm-summary --runs runs_fir --sub p16_sv`, kit COA2, saved in `cruxvault/results/closeout_A2/base_arm_summary_p16_sv.tsv`), mean (seed range):

| architecture | variance recovered | gap closed, crop pairs, oracle reference | calibration term | composite v3 |
|---|---|---|---|---|
| conv autoencoder | 0.382 (0.028) | 0.718 (0.012) | 0.631 (0.178) | 0.528 (0.045) |
| conv + attention | 0.387 (0.012) | 0.673 (0.031) | 0.652 (0.157) | 0.525 (0.048) |
| U-Net | 0.432 (0.023) | 0.507 (0.071) | 0.610 (0.108) | 0.495 (0.010) |

Expectation: variance recovered -0.01 to -0.03 (beyond on the two conv models, whose ranges are small); gap closed lower by 0.03 to 0.10 (beyond on the two conv models, perhaps within on the U-Net, whose range is 0.071); calibration within (ranges 0.11 to 0.18).

## 5. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds). Judge each architecture separately.

| outcome | conclusion |
|---|---|
| gap closed lower, beyond, on all 3 architectures | the covariates drive correction; the control passes and the correction metric works as a test |
| gap closed within on 1+ architecture | that model corrects by depth or by copying, not by covariates; inspect its crop pairs before any correction verdict on it |
| gap closed higher, beyond | the control fails; the correction term rewards something other than covariate use; stop and review the pairs (as in prior lead 1) |
| variance recovered lower, beyond | the covariates carry imputation information (batch or processing) |
| everything within on all 3 | the covariates add nothing beyond depth at 80,000 windows; the covariate path is not a design lever at this budget |

## 6. Controls and traps

- **The scramble is an identity at scoring (read from the code).** `predict.py` `predict_chrom` (lines 299 to 305) builds each inference batch from consecutive tiles of ONE job: same inputs, same target. So every window of the batch has the same covariate row per track, and a permutation among them changes nothing. The arm is therefore "trained with scrambled covariates, scored with the true covariates". It measures whether the model learns to use covariates when they carry information in training. It does not measure the effect of wrong covariates at test time. Prior lead 1 (U-Net) was scored the same way. A scored-with-wrong-covariates form needs a scorer option (covariates from another experiment of the same assay); I did not build it.
- Parameter count unchanged (the switch builds no module): 1,042,514 / 1,038,018 / 993,599, equal to the reference.
- Off switch: no code changed; the reference config has `scramble_covariates: false`.
- In training, the permutation draws from the global torch RNG, so it follows the seed.

## 7. Readiness

Ready. Configs `tools/minicandi/configs/ablations/scrambled_covariates/rung{2,3,4}/scrambled_covariates.json` and `arms.tsv`, written by `stage2_arms.write` with the arm added at run time; a json diff against `reference_equal.json` shows `scramble_covariates` and `simpler_than_base` only (`tests/test_ablation_configs.py`). Laptop step time (forward + backward, batch 8, 1,024 bins, 4 threads; `cruxvault/results/closeout_A2/step_timing_laptop.txt`) equals the reference within 7%. Training limit 03:00:00 (submit default).

Cost, from the per-run Fir `sacct` figures of `7_cov_embedding.md` section 7 (28.99 / 23.56 / 34.85 CPU-h): 3 seeds x 87.4 = 262.2 CPU-h, 9 runs, 54 jobs.
