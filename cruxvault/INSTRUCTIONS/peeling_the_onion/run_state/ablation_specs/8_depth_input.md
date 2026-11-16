# Ablation 8: depth-controlled count input (`depth_input`)

Status: Phase 1 specification, 2026-11-16. No code changed. Worktree base fb57fc1.

## 1. Question

The PI wrote (`peeling the onion.md`, # miniCANDI, important notes): "as an ablation, the count input data to the model can be depth controlled". The ablation asks: does the model impute, correct and calibrate better or worse when every input track is first brought to one fixed sequencing depth, so that input depth no longer varies across tracks and cells? The targets and C' (the output covariates) stay at native depth.

## 2. Arms

Readings of "depth controlled":
- (a) Subsample the input reads to a fixed reference depth (binomial thinning). The code has this: `input_depth: fixed` (arm `input_depth_fixed` in `stage2_arms.py`).
- (b) Divide the input by library size, i.e. scale to a fixed depth. The base recipe already does this: `count_transform: offset` computes log2((y + 1) / s), with s = depth / 2^24 (`components.py` `CountTransform.forward`, `REF_LOG2_DEPTH = 24.0`). Reading (b) is the reference, so it is no arm.
- (c) Hide the input depth covariate. This ran on the U-Net only (PI rulings 48, 49) and is ablation-independent prior evidence (section 3). It removes information without changing the counts, so it does not control depth.
- (d) Equalize every input exactly, with a reference at or below the shallowest input (1,510,281 reads, 2^20.5). This thins the median base input by about 17x (section 3 depths) and needs a new constant in shared `dataset.py`. Rejected: it measures information loss more than depth control.

I take reading (a). One arm, plus the existing reference:

| arm | keys | what changes |
|---|---|---|
| reference | base recipe, `input_depth: native` | nothing; existing runs `rung{2,3,4}_base_heldout`, 3 seeds each, scored in `p16_sv` on Fir |
| `input_depth_fixed` | `input_depth: fixed` | data path only (below) |

What `input_depth: fixed` does (`tools/minicandi/dataset.py`, `window`, lines 1003 to 1026 at fb57fc1): for each input track whose log2 depth (c_in column 0) is above `FIXED_BASE_LOG2_DEPTH` = 23.0 (8,388,608 reads, the same value for all five assay classes), the counts are binomially thinned by the ratio 2^(23 - log2 depth), and c_in column 0 is set to 23. An input is never made deeper. The draw is seeded by (measurement, file, chromosome, start bin), so a window gives the same draw each time. The thinning acts on raw counts, before `count_transform: offset`, so the transform's size factor reads the thinned depth (s = 2^(23-24) = 0.5). Targets, C' and target kind do not change. A same-view target (kind 3) keeps its native counts (`native_x`). It applies in every mode: training, the held-out loss, prediction and scoring.

The key is read by the dataset, which feeds all architectures. `stage2_arms.py write --arms input_depth_fixed` on each `final/rung{2,3,4}/reference_equal.json` wrote one config with the single change `input_depth: fixed`, status `write`. `registry.build_model` gave the same parameter count as the reference (conv autoencoder 1,042,514; conv + attention 1,038,018; U-Net 993,599), and the seed-0 `state_dict` was identical tensor for tensor. `pytest tests/test_minicandi_dataset_arms.py -k input_depth`: 2 passed.

Depth of the inputs (Fir, `/project/def-maxwl/mforooz/ONION/tables/eic_v1/measurements.tsv`, column `depth_reads`): of the 678 training base views, 653 are deeper than 2^23 (96%) and 505 deeper than 2^24. Median 26,104,201 reads (2^24.6), so the median base input keeps 2^(23 - 24.6) = 32% of its reads. Of the 936 training depth-thinned views (`arm_knob = depth`), 486 are above 2^23 and are thinned; 450 stay native. Every validation crop view is above 2^23 (6 of 6 broad, 9 of 9 narrow), so the crop-pair inputs are thinned at scoring. The arm therefore caps input depth at 2^23. It does not make input depth constant.

## 3. Prior leads

1. U-Net split depth arms (`cruxvault/results/views_ablation/VIEWS_ABLATION.md`, p16 subdirectory, not p16_sv): depth covariate hidden on the input side: composite 0.382 (0.009) against base 0.444 (0.030), specific variance 0.401 against 0.440. Hidden on the output side: composite 0.455 (0.013). Both sides hidden: 0.260 (0.019). Finding f83 records this. COVERAGE.md rows 27 to 29 give the same values. Prediction: the network uses input depth. Our arm keeps the depth covariate truthful (it states the thinned depth), so this lead does not predict a loss by itself; it says input depth carries information.
2. No p16_sv score of `rung4_hide_depth_output` exists on Fir: `runs_fir/rung4_hide_depth_output/seed{0,1}` hold `p16` only, and seed 2 is not there (checked with `ls`, 2026-11-16).
3. Training views (same file): base + depth thinning views composite 0.479 (0.041) against base view only 0.483 (0.050); gap closed, clean pairs 0.505 (0.106) against 0.553 (0.041). Thinned inputs in training did not help beyond the range.
4. Finding f66 (`cruxvault/f66_answers_the_depth_half_of_the_question_o.md`): on log1p counts, thinning to 1/2 and 1/4 lowered the H3K27ac cell-type share from 0.199 to 0.170 and 0.139. Prediction: thinning to 32% lowers the signal the model can recover, so imputation drops.
5. Finding f56 (`cruxvault/f56_the_subset_pair_gap_excludes_0_and_the_d.md`): on subset pairs, the gap sign follows the depth direction: -0.011 on depth-decreasing pairs, 0.081 on depth-increasing pairs. Prediction: a thinned input makes every crop pair a larger depth increase, so correction gets harder.
6. Finding f33 (CAC test bed, `cruxvault/f33_margins_are_11_66x_the_seed_sd_on_every_.md`): part of the correction gain is "denoising shallow sources"; depth scaling fixes scale error but not shape. Prediction: shallower inputs leave more to denoise, against a fixed depth-only reference that reads the native input.
7. Literature digest (archive vault, `CANDI/cruxvault/wiki/digest-depth-as-covariate-vs-divisor.md`): depth belongs in the mean as an offset; dividing it out of the data is lossy. The base recipe already uses the offset.

## 4. Expectation (before any run)

| metric | direction | size |
|---|---|---|
| imputation (variance recovered) | down | 0.01 to 0.03; may stay within the seed range on the conv autoencoder (0.028) |
| correction (gap closed, crop pairs) | down | 0.05 to 0.15; beyond the range on all three |
| calibration | no clear direction | within the range (ranges 0.108 to 0.178) |
| composite | down | 0.02 to 0.05; beyond only on the U-Net (range 0.010) |
| correction-only term | down | read with gap closed |

## 5. How the result answers the question

Judged per architecture against the larger of the two seed ranges.

| outcome | conclusion |
|---|---|
| composite and correction down beyond the range, imputation down or within | Depth control of the input throws away information. Keep native input depth with the offset transform. |
| all four within the range | The model is indifferent to input depth above 2^23. Keep native input; it is simpler (no thinning). |
| composite up beyond the range on 2 or more architectures | Varying input depth hurts generalization. Propose fixed input depth for the design, then test a second reference depth before adopting it. |
| correction up beyond, imputation down beyond | Trade-off. Report both terms; no design change without a PI ruling. |
| the architectures disagree in sign beyond the range | The effect depends on architecture. Report per architecture; no general claim. |

## 6. Controls and traps

- **The scorer's input is thinned too.** Prediction and scoring build windows through `OnionDataset` with the run's config (`predict_store.py` via `score.py` line 943, `self.ds = OnionDataset(..., cfg, ...)`). This keeps train and test inputs consistent. It is part of the arm: the model is judged on thinned inputs. The truth is unchanged: targets of kind 1 and 2 are read natively; a kind-3 target keeps `native_x`.
- **The references stay fixed.** Gap closed reads the depth-only and floor terms from `REFS_FROM = /project/def-maxwl/mforooz/ONION/eval/eic_v1/baselines_p16` (`score_final.sh` on Fir), computed once on native inputs. The oracle-floor gap is computed at report time from `baselines_p16/oracle_floor`. So the arm cannot change its own reference. Control: confirm `REFS_FROM` in the Phase-2 submit script.
- **Parameter count and the off switch.** Unchanged (section 2). The arm is data-only.
- **Kit drift.** The reference runs used kit CM. The arm will run on a new kit. Control: `python tools/golden.py check` on the base config with the new kit before submission.
- **Held-out loss.** It is computed on thinned inputs (`train.py` line 173), so it is not comparable with the reference. Do not use it to compare arms.
- **Depth views below the cap.** 450 training depth views stay native. The arm is a cap, not a constant input depth. State this with every result.
- **Single draw per window.** The seed fixes one thinned sample per (window, measurement). The arm does not add thinning noise across epochs.

## 7. Readiness

| architecture | state |
|---|---|
| conv autoencoder | ready (config from `stage2_arms.py write`; no code) |
| conv + attention | ready |
| U-Net | ready |

Shared files to edit in Phase 2: none.

Cost (Fir `sacct` of the seed-0 reference jobs, allocated CPU-seconds = CPUs x elapsed): conv autoencoder: train 30,252 + predict 10,392 + 7,728 + 8,208 + 6,680 + metrics 29,616 = 92,876 s. Conv + attention: 21,264 + 9,408 + 6,384 + 6,528 + 6,344 + 30,896 = 80,824 s. U-Net: 31,920 + 9,408 + 9,952 + 10,432 + 9,576 + 28,992 = 100,280 s. Per seed over three architectures: 273,980 s = 76.1 CPU-h. Total: 9 runs, 3 x 76.1 = 228 CPU-h. The reference needs no new runs. The thinning adds dataset work; I expect it to be small against the model cost, but I did not measure it.
