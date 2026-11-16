# Ablation 2: predict the ChIP-seq control as an output (on against off)

Phase 1 specification, 2026-11-16. Worktree branch `chunk/onion-AB-2`, base fb57fc1.

## 1. Question

Does it help or hurt the three goals (imputation, covariate artifact correction, calibration) when miniCANDI also predicts the ChIP-seq control track? The PI's lines (`peeling the onion.md`, miniCANDI outputs): "ablation: also predict the chip control", and "if we are also predicting chip control, we'd be relying on our own NB predictions of both signal and control".

## 2. Arms

The base of all three architectures already predicts the control (`predict_control: true`, `control_input: true`; read from `tools/minicandi/configs/final/rung{2,3,4}/reference_equal.json`). So the ablation removes the prediction.

| arm | keys | what changes |
|---|---|---|
| reference: base | `predict_control: true`, `control_input: true` | none. The existing runs `runs_fir/rung{2,3,4}_base_heldout/seed{0,1,2}` (Fir) are the reference. |
| `predict_control_off` | `predict_control: false`, `control_input: true` | The dataset drops the control (track 35) from the targets (`dataset.py:992-994`): `y_avail` and `c_out` of the control stay 0. The control input is unchanged. The network is unchanged. |

Readings of the PI's lines. (1) The first line asks whether the control output itself is useful. The trained arm above answers it. I take this reading. (2) The second line asks how good a p-value track is when its lambda comes from the model's own predicted control. A trained OFF arm cannot answer it, because an OFF model has no control prediction. It is a score-only question on the ON base runs (`score.py --lambda-source predicted_control`). It is out of scope here: the `p16_sv` scores hold no signal-space keys (checked: the key list of `rung2_base_heldout/seed0/p16_sv/scores.json`), and 0 of the 43 validation leave-one-assay-out targets is a ChIP-seq target without an observed control (checked on Fir against `tables/eic_v1/measurements.tsv`). The audit row 19 (`ladder_audit/COVERAGE.md`) reads the PI's line as the output-only lambda head. That arm (`control_lambda_head`) also sets `control_input: false`, so it belongs to ablation 1.

Difference from ablation 1 (control as input). Ablation 1 switches the control INPUT off and keeps the control as a target (it then becomes an imputation target, kind 1, scored on every bin). This ablation keeps the input exactly as in the base and removes only the control TARGET and its loss terms.

## 3. Prior leads

1. No trained, scored run has `predict_control: false` (audit `COVERAGE.md` rows 18 and 19: "UNTESTED"; checked `runs_fir` on Fir: no such directory).
2. Smoke runs, not settlements (`results/t209/arm_smoke.tsv`, foreman worktree). `control_none` (both control keys off) ended at loss 0.769 (conv autoencoder) and 0.778 (axial transformer); the matching `control_output_only` arms ended at 0.905 and 0.932. These losses average over different bin sets (`results/t209/ASF.md`, line 34), so they predict nothing about the metrics.
3. The base models predict the control poorly. On the one validation control target, the count CRPS skill against the position average is, seeds 0/1/2: conv autoencoder -0.088 / -0.045 / -0.078; conv + attention -0.003 / -0.023 / -0.032; U-Net +0.015 / +0.011 / +0.006 (`runs_fir/rung{2,3,4}_base_heldout/seed*/p16_sv/scores.json`, key `predicted_control.count_crps_skill_vs_position_average`). Prediction: the control task gives the shared layers little useful signal, so removing it changes little.
4. PI ruling 14 (`long-horizon notepad.md`, line 45): "The EIC model keeps a control output in all cases (D9b)." D9b (`long-horizon decisions.md`, line 75): for the blind Caco-2 and SJSA1 ChIP-seq tracks, lambda "comes from a predicted control". Implication: this arm cannot become the EIC model whatever it scores. `predict.py:291-293` refuses a ChIP target with no observed control when `predict_control` is false. The ablation therefore only measures the cost or benefit of the control output; the decision it can drive is how the control is predicted (shared channel, separate head, loss weight), never whether.
5. No archive-vault result bears on it. The CAC sandbox kept the control as an always-available input that was never a target (archive `q19_can_we_make_dual_conditioning_work_on_re.md`, line 26).

## 4. Expectation (before any run)

Null on all four metrics: each difference of means inside the seed range. Sizes: imputation |delta| < 0.02 (ranges 0.012 to 0.028); correction |delta| < 0.03 (ranges 0.012 to 0.071); calibration inside its range (0.108 to 0.178); composite |delta| < 0.02 (ranges 0.010 to 0.048). The reason: lead 3, and the control is one of up to 36 target channels.

## 5. How the result answers the question

Per architecture; "beyond" = difference of means larger than the larger of the two seed ranges.

| outcome | conclusion |
|---|---|
| all four metrics within range | The control output is free. Keep it (required by ruling 14 anyway). |
| OFF better beyond, on composite or imputation | The shared control channel costs capacity or gradient. Next step: a separate control head or a lower control loss weight, not removal. |
| OFF worse beyond, on composite or imputation | Predicting the control is a useful auxiliary task. Keep it; record it as evidence for multi-task training. |
| correction changes beyond, imputation within | The control target supplies correction signal (its depth pairs train correction). Report per knob. |
| different sign across architectures | Architecture-dependent. Report per architecture; make no general claim. |

## 6. Controls and traps

- **The scorer's input.** The composite does not read the control. Checked: the scorable sets of `evalset_p16.json` exclude `chipseq-control`; the 18 crop pairs are all histone pairs; the 30 control correction pairs are all `never_trained = 0`, so never headline pairs; the selection terms are in `counts_log1p`. With 0 validation ChIP targets lacking an observed control, the scorer never asks an OFF model for a predicted control. So the four metrics read the same targets and pairs in both arms.
- **Side tables.** Under OFF, `Scorer.jobs` drops the control targets (`score.py:2419-2420`). The `predicted_control` table then holds `n_targets: 0` with the reason "predict_control is off ...". (The Phase-1 text said the scorer still scores the control target. That was wrong.) The 30 control pairs are still scored, but `ds.window` drops their control target. So their truth `y_counts` is all zeros (`score.py:1505`), and their rows in the non-headline pair tables are meaningless. Do not quote them. Phase 2 checked this on synthetic tables (`tests/test_ablation_predict_control.py`, 2 passed). The scorer completes under OFF, on the in-process path and on the store path (predict + metrics, same-view on). The same weights scored with ON and OFF configs give equal `selection`, correction-only term and copy-leak blocks. Every other key is equal, except the `predicted_control` table and the non-scorable control row of `subset_pairs_gap_closed`.
- **The loss sum.** Under OFF the main loss is the mean over the non-control masked bins only. Checked on one synthetic batch (conv autoencoder, `synth.py`): OFF `main` = 4.568594 = ON `main_no_control`. So compare training curves by `main_no_control`, never by `main` or `heldout_loss`. The renormalization gives each non-control bin a slightly larger weight; this is part of the arm, not a confound to remove.
- **Inputs bit-identical.** Checked on synthetic tables, 20 batches of 4: `x_counts`, `x_avail`, `c_in`, `x_bin_mask` identical; non-control `y_counts` and `c_out` identical; control targets 80 (ON) against 0 (OFF). The sampler draws the same plan in both arms.
- **Parameter count.** Unchanged, and `state_dict` identical at seed 0 for all three architectures (checked with `registry.build_model`).
- **Kit drift against the reused reference.** The base runs were trained with kit RTf (`train_4246987.out`) and scored with kit CM. The training files (`dataset.py`, `train.py`, `losses.py`, `contract.py`, `registry.py`, `components.py`, `covariates.py`, three rung files) are byte-identical in RTf, CM and this worktree (checked with `cmp` on Fir). The OFF runs must use a kit with these files unchanged, and the same scorer settings as `score_final.sh`; else the base must be rerun (9 more runs).

## 7. Readiness

- Conv autoencoder, conv + attention, U-Net: ready. `predict_control: false` with `control_input: true` builds on all three (param counts 1,042,514 / 1,038,018 / 993,599, equal to the base) and needs no code.
- Configs (Phase 2, written by hand because the arm is not in `stage2_arms.py` `ARMS`): `tools/minicandi/configs/ablations/predict_control/rung{2,3,4}/predict_control_off.json`. Each is `final/rung<r>/reference_equal.json` with `predict_control: false` and `simpler_than_base: true` (the value `stage2_arms.py` gives `control_none`). A json diff showed only these two keys differ. Built with `registry.build_model`: conv autoencoder 1,042,514; conv + attention 1,038,018; U-Net 993,599 parameters, equal to the base.
- Submit options: none beyond the defaults. The training cost equals the base's.
- Data: none missing. Tables, eval set and base runs exist on Fir (`ONION/tables/eic_v1`, `ONION/eval/eic_v1/evalset_p16.json`, `runs_fir/rung{2,3,4}_base_heldout`).

Cost, from `sacct` CPUTimeRAW on Fir of the seed-0 base jobs (train with 12 CPUs; 4 predict shards with 8 CPUs; metrics with 16 CPUs; the `p16_sv` pass of 2026-11-16T16:50Z):

| architecture | train s | predict s | metrics s | total CPU-h |
|---|---|---|---|---|
| conv autoencoder | 30,252 | 9,536+8,432+6,272+8,600 = 32,840 | 33,728 | 96,820 s = 26.9 |
| conv + attention | 21,264 | 6,768+8,400+9,480+8,128 = 32,776 | 31,808 | 85,848 s = 23.8 |
| U-Net | 31,920 | 9,168+9,192+9,096+8,824 = 36,280 | 33,232 | 101,432 s = 28.2 |

One arm x 3 seeds: 3 x (26.9 + 23.8 + 28.2) = 3 x 78.9 = 236.7 CPU-h for 9 runs. The reference costs 0 (reused).
