# Ablation 3: the training-set average as an input (`average_input`)

Written 2026-11-16 on branch `chunk/onion-AB-3` (base fb57fc1). Phase 1: specification only.

## 1. Question

Does miniCANDI impute, correct and calibrate better when it also reads the per-assay, per-position average of the training cell types as an input? The PI's line (`peeling the onion.md`, miniCANDI inputs): "optional/ablation: average baseline: for each assay type, we can get its average activity along the genome (training set only) - average baseline per-assay type", with size (B, W/resolution, num_all_assays).

## 2. Arms

| arm | keys | change |
|---|---|---|
| reference | `average_input: false` (the base recipe, `configs/final/rung{2,3,4}/reference_equal.json`) | none; the existing base runs `rung{2,3,4}_base_heldout`, seeds 0 to 2, are reused |
| `average_input` | `average_input: true`, `average_h5: /scratch/mforooz/onion/aux/eic_v1_average.h5` | the batch gets `avg` [B, L, A], i.e. the leave-one-cell-out mean of counts x 1e6 / depth_reads over the other training cells' base views; each model adds a learned lift of log1p(avg) to every track's input state |

Per architecture, the lift is: conv autoencoder `avg_w` [36, 16] (+576 parameters); conv + attention `avg_lift` Linear(1, 16) (+32); U-Net `avg_in` Linear(1, 8) (+16). The lift acts at the per-track input, so a masked track also gets its average. In the U-Net, the lifted state therefore also enters the skip connections.

Readings. (a) A per-position average per assay (the PI's stated size is per position); (b) one genome-wide scalar per assay. I take (a): reading (b) holds no positional information, and the covariates already carry assay identity. Fusion has four options in `miniCANDI architecture ablations.md` (input channel; GLM offset, i.e. the network predicts a log-deviation from the average; retrieval attention; post-hoc blend). I take the input channel, because it is the only one built and D10 forbids the blend. One arm: 9 runs.

## 3. Prior leads

1. Rung-1 pointwise model, `average_input`, Nibi `/project/def-maxwl/mforooz/ONION/runs/rung1_average_input`: seeds 0 and 1 DONE, seed 2 empty, none scored. Training loss, mean of the last 40 logged steps (I computed it from `log.jsonl`): arm 0.871 and 0.879; base 0.901, 0.913, 0.928 (range 0.027). The difference of means is 0.039, beyond the base range. This is training loss on training windows only. Predicts: the average carries usable information.
2. Position average as a predictor, `FINAL_ROUND_EQUAL.md` (foreman worktree): variance recovered 0.119, gap closed 0.383, calibration 0.168, composite 0.197. All three base recipes are far above it (composite 0.495 to 0.528). Predicts: the average alone is weak on these metrics, so any gain must come from combination.
3. `cruxvault/wiki/average-activity-baseline.md`: in the ENCODE Imputation Challenge (EIC), the average beat all but two of 23 entries; 8 of 23 entries used it as an explicit input, and entries that did not use it did not consistently correlate less with it. Predicts: models learn the average implicitly, so the gain may be small.
4. `miniCANDI architecture ablations.md` (Max s06): against the cross-cell-type average, the best model's gain was all specific (-0.043), with no change in shared error. The imputation term here is specific variance above noise (`tools/onion_eval/metrics.py`, `specific_variance_recovered`), with the shared part estimated across cell types. Predicts: the average mainly lowers the shared error, which the composite does not score.
5. `miniCANDI design notes.md`: the blended CANDI (a·CANDI + b·Average + c) that ranked first in the EIC was almost the average for H3K4me3 and ATAC-seq (CANDI weights 0.103 and -0.067; I did not check this number against its source). D10 (`long-horizon decisions.md`): the PI expects this arm to be at least as strong as the blended CANDI. That comparison is not available on the 16-region set.

## 4. Expectation (stated before any run)

| metric | expected change, arm minus reference |
|---|---|
| variance recovered (imputation) | +0.01 to +0.04; beyond the range on conv + attention (range 0.012), within it on the other two |
| gap closed (correction) | 0 +/- 0.03; the average holds nothing about the source's artifact |
| calibration | 0 +/- 0.05; within the ranges (0.108 to 0.178) |
| composite | +0.005 to +0.02; within the ranges (0.010 to 0.048) |
| diagnostics | shared error (`error_budget/.../shared`) falls; `count_crps_skill_vs_position_average` rises |

## 5. How the result answers the question

"Beyond" = the difference of means is larger than the larger of the two seed ranges.

| outcome | conclusion |
|---|---|
| imputation beyond and up on 2 or 3 architectures; correction and calibration not down beyond | the average is useful information that the models do not learn alone; it is a candidate for the design (`joins_design: yes`); the next step is the GLM-offset fusion |
| shared error falls, imputation within the range | the models take the shared signal from the average, but the specific term does not reward it; keep the simpler base |
| no change in any metric or diagnostic | the models already learn the average implicitly (lead 3); drop the input |
| imputation up beyond, correction down beyond | the average pulls predictions toward the cross-cell consensus and costs correction; a trade-off for the PI |
| a gain on one architecture only | the effect depends on the architecture; report it, do not generalize |
| any metric down beyond | check the input path (scale of `avg`, staging, scorer config) before any conclusion |

## 6. Controls and traps

- **Leakage of the target's own signal.** Checked. `OnionDataset.average` subtracts the window's own cell; `build_average_h5` sums `train_base` only (training rows). On Fir I read the h5 against `tables/eic_v1/measurements.tsv`: 305 of 305 summed measurements are `in_eic = train`, 36 tracks, no cell twice in a track. Validation and blind tracks are never summed. Cell C02 has a validation control (ENCSR815BDB) and a training control (ENCSR828HXJ) in the sum, but the own-cell subtraction removes it. The one empty-cell control in the sum (ENCSR000AMI, a pseudo-cell of 124 rows) is the control of no validation or blind target (0 of the 36 targets with an empty-cell control). Eleven targeted unit tests passed locally (leave-one-cell-out, h5 guard, missing `avg` raises).
- **The on-the-fly path.** Without `average_h5`, the dataset reads every other training cell directly per window: the same leave-one-cell-out mean, but 305 reads per window. Use the file.
- **Eval-region information.** The average holds other cells' training tracks at the 16 evaluation regions. This is legal training data (it is the EIC average baseline), but the result compares "model plus average" with "model". `count_crps_skill_vs_position_average` reports the arm against the average itself.
- **Scorer input.** The scorer builds its windows with the run config, so `avg` reaches the model at evaluation. A missing `avg` raises (tests above), so a silent no-op cannot happen.
- **Scorer drift.** The reference was scored with kit CM. The arm must be scored by the same `onion_eval` code, or one base seed must be rescored with the new kit.
- **Parameter count.** +576, +32, +16 (at most 0.06%); counts in Readiness.
- **Off switch.** No code change is needed; `average_input: false` builds no extra module, so the base is unchanged.
- **Content of the average.** 18 of 305 summed views are single-end views (tracks with no base view). The control channel has 38 sources. Five assays have one training cell (for example H2AK9ac); their average is 0 for that cell. At evaluation, a validation track's cell adds no own track, so its average uses n cells, against n - 1 in training: a small shift.
- **Speed.** Rung 1 on a GPU slice read 0.192 s per window from an unstaged file, against 0.0423 s for the base (`arms_d10_d15/SUBMITTED.md`). Staged (seed 1), the total wall was 5,787 s, inside the base's 5,030 to 5,911 s. `ONION_STAGE=1` copies the 7.45 GB file to the node; `train.py` checks free space first.

## 7. Readiness

| architecture | state | parameters (base -> arm, local `registry.build_model`) |
|---|---|---|
| conv autoencoder | ready | 1,042,514 -> 1,043,090 |
| conv + attention | ready | 1,038,018 -> 1,038,050 |
| U-Net | ready | 993,599 -> 993,615 |

- Code: `average_input` is read in `rungs/conv_ae.py`, `rungs/conv_attn_ae.py` and `rungs/unet.py`, in this worktree and in kit CM on Fir. Configs come from `stage2_arms.py write --arms average_input --average-h5 <path>`. No shared file needs an edit.
- Data: `/scratch/mforooz/onion/aux/eic_v1_average.h5` exists on Fir and on Nibi (7,451,594,429 bytes, the same md5 on both, 06ffec28...). It was built 2026-11-13 after the eic_v1 tables (07:46).
- Cost, read from Fir `sacct` for conv autoencoder seed 0: training 42:01 x 12 CPUs = 8.40 CPU-h; prediction 4 shards, 68:25 x 8 CPUs = 9.12 CPU-h; metrics 35:08 x 16 CPUs = 9.37 CPU-h. Training from the base walls (2,389 / 1,643 / 2,541 s x 12 CPUs) = 7.96 + 5.48 + 8.47 = 21.9 CPU-h per seed, x 3 = 65.7 CPU-h. Scoring 18.5 CPU-h x 9 = 166.4 CPU-h. Total about 232 CPU-h for 9 runs. The extra read of the average on CPU is not measured; a 1.5x training slowdown adds about 33 CPU-h.
