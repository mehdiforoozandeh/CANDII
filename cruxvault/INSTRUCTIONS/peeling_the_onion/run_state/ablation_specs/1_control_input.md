# Ablation 1: the ChIP-seq control as an input assay (control_input)

Status: Phase-1 specification, 2026-11-16. No run submitted.

## 1. Question

Does the model impute, correct and calibrate better when the ChIP-seq control track (input DNA, i.e. sequenced chromatin with no antibody pull-down) is one of its input assays? The PI's line, `peeling the onion.md`, miniCANDI inputs: "ablation: experiment with adding chip controls (upon availability for those celltypes) as one assay".

## 2. Arms

| arm | keys | role |
|---|---|---|
| control on (base) | `control_input: true`, `predict_control: true` | reference; the existing base runs `rung{2,3,4}_base_heldout`, already scored in `p16_sv` |
| control off (`control_output_only`) | `control_input: false`; everything else as the base | the one new arm |

Each architecture uses its base recipe `tools/minicandi/configs/final/rung{2,3,4}/reference_equal.json`: conv autoencoder, conv + attention, U-Net (`skip_source: both`). The arm key comes from `tools/minicandi/stage2_arms.py` (`ARMS["control_output_only"]`).

What the switch changes (read in the code today):
- Data path only. `dataset.py` `window()` removes every `chipseq-control` measurement from the inputs when `control_input` is false. Track 35 then has `x_avail = 0` and `c_in = 0`, so the model sees it as an absent track, i.e. as if it were always masked.
- The control stays a training target (`predict_control: true`). Without the input it becomes kind 1 (imputed), and kind 1 is scored on every bin (`losses.py` docstring; `contract.loss_mask`). So the arm also moves loss mass onto control imputation. `losses.py` logs `parts["main_no_control"]` (tracks 0 to 34 only) to compare training loss across these arms.
- The model does not read the key. `rungs/conv_ae.py`, `conv_attn_ae.py` and `unet.py` accept it and pass it to the dataset. I built each base with the key true and false at seed 0 (`registry.build_model`): the parameter counts are equal (1,042,514 / 1,038,018 / 993,599) and every `state_dict` tensor is identical.
- The output panel does not change (36 tracks; the control is still predicted).

Readings of the PI's line: (a) control as an input, on against off -- taken; (b) "upon availability" -- the control enters only where the cell has one; this already holds, because the dataset reads the controls that `measurements.tsv` lists. Not taken: `control_none` (no input, no output). PI ruling 14 and D9b keep a control output in the EIC model, and `stage2_arms.py` marks `control_none` `joins_design = no`. The output-only lambda head is ablation 19 in the audit, not this one.

## 3. Prior leads

1. **No trained result exists.** Audit `cruxvault/results/ladder_audit/COVERAGE.md` row 18: "UNTESTED", 0 runs, on any architecture.
2. **Smoke runs, 250 steps, seed 0** (`cruxvault/results/t209/ASF.md`, table rows `control_output_only`): late loss ratio arm/base 1.033 (conv autoencoder) and 1.017 (axial transformer), noise band 0.992-1.004 and 0.980-0.983. Per kind: impute 1.179 / 1.172, correct 0.983 / 0.966, same-view 0.951 / 0.923. ASF explains the rise: control targets become kind 1 on all bins, so `main` averages over a different bin set. Prediction: training loss is not comparable across the arms; only `main_no_control` and the scorer are.
3. **CAC test bed, mappability channel** (`cruxvault/f32_the_mappability_channel_lowers_lowest_ma.md`, confidence weak): a per-bin mappability input lowered CRPS in the lowest-mappability decile on all 6 histone tracks for crop 36 and MAPQ 0. Genome-wide CRPS did not change (`miniCANDI design notes.md`, mappability-and-input-dna-channels). The same note says: "Input-DNA (ChIP control) coverage is a related candidate channel and was not tested." Prediction: a control input helps locally (low-mappability, copy-number regions); the genome-wide headline moves little.
4. **Old CANDI** (`AGENTS.md` invariant 1, archive vault `q19_...md`): the control was always an input and never masked. No on/off result exists there.
5. **PI ruling 14** (`long-horizon notepad.md`, 2026-11-10): the control ablation is allowed, and "The EIC model keeps a control output in all cases (D9b)." Agent call of 2026-11-10: control is an input and an output by default.
6. **Scale of a detectable effect.** Base seed ranges (`cruxvault/results/final_round/FINAL_ROUND_EQUAL.md`): imputation 0.028 / 0.012 / 0.023; correction 0.012 / 0.031 / 0.071; calibration 0.178 / 0.157 / 0.108; composite 0.045 / 0.048 / 0.010 (conv autoencoder / conv + attention / U-Net).

## 4. Expectation (before any run)

- Imputation (variance recovered): control off is lower by 0 to 0.02. All 43 leave-one-assay-out targets of the screening set have a control among their inputs (checked in `evalset_p16.json` on Fir), so the effect can show. I expect it to stay inside the seed range on at least two of three architectures.
- Correction (18 crop pairs): no change in direction. The pair windows hold the source track only (`score.py` `_pair_window`), so the scorer's input is the same in both arms; only the weights differ.
- Calibration: no detectable change; the seed range (0.108 to 0.178) is larger than any plausible effect.
- Composite: inside the seed range.
- Correction-only term: no change.

## 5. How the result answers the question

"Beyond" = the difference of means is larger than the larger of the two seed ranges.

| outcome | conclusion |
|---|---|
| control on is beyond control off on imputation or composite, on 2+ architectures | the control input carries information the other assays lack; keep it as an input |
| control off is beyond control on on any metric, on 2+ architectures | the control input hurts (for example, it lets the model copy a background shape); drop it, and look for the mechanism |
| all differences inside the range, on all three | no evidence that the input matters at this scale; by D13 (`long-horizon decisions.md`) the simpler arm, control off, stays. Report that the test has low power at 3 seeds |
| beyond on one architecture only | an architecture-specific effect; do not change the design; name the architecture |

D13 is the binding rule for the design: an arm beats the incumbent only if its composite is higher by more than 2x the seed spread and no goal drops by more than 2x its spread.

## 6. Controls and traps

1. **Loss level.** The training loss rises by construction (lead 2). Compare `main_no_control` from `log.jsonl`, never `main`.
2. **Scorer input changes, as intended.** In the off arm the 43 imputation targets lose their control input. This is the effect under test, not leakage.
3. **Control correction pairs.** The eval set holds 30 control-track correction pairs (all with `never_trained = 0`). In the off arm their only input is removed, so the model predicts them from DNA and covariates alone, and depth-only scaling is undefined on them (kind 1). They are not among the 102 headline pairs or the 18 crop pairs, so the four metrics are clean. Any all-pairs secondary metric is not comparable; do not quote it.
4. **Predicted-control table.** `scores.json` `predicted_control` holds 1 control target (cell C02). Report it as a diagnostic only (n = 1).
5. **Parameter count.** Unchanged (checked, section 2).
6. **Code drift between the base runs and the new kit.** The base `run.json` files on Fir have `git_sha: unknown`. Before training, the lead diffs the training-path files (`dataset.py`, `losses.py`, `contract.py`, `train.py`, `registry.py`, `rungs/`, `components.py`, `covariates.py`) of the new kit against kit CM. If they differ in a way that moves a number, the base must be retrained in the same batch (9 more runs).
7. **Copy-leak guardrail.** The arm removes an input only, so it cannot add a leak path; the scorer checks the gate anyway.

## 7. Readiness

| architecture | state |
|---|---|
| conv autoencoder | ready: config only. `stage2_arms.py write --arms control_output_only` on `final/rung2/reference_equal.json` wrote it; it differs from the incumbent in `control_input` and the label `simpler_than_base` only |
| conv + attention | ready (same check, `final/rung3/`) |
| U-Net | ready (same check, `final/rung4/`) |

Written configs: `tools/minicandi/configs/ablations/control_input/rung{2,3,4}/control_output_only.json`, each with its `arms.tsv`. A json diff against `reference_equal.json` showed only `control_input` (true -> false) and `simpler_than_base` (null -> true). Parameter counts from `registry.build_model`: conv autoencoder 1,042,514; conv + attention 1,038,018; U-Net 993,599. These equal the base counts.

Submit options: the defaults only. The arm reads one input fewer and adds no compute. The margin under the default training limit is small: the limit is 5,856 s, and the conv-autoencoder base seed 2 trained for 5,691 s (lead's note, Fir job 4246989).

No code change and no shared-file edit. No new data: the controls are in `tables/eic_v1` (the eval set lists them as inputs on Fir).

Cost. Source: `sacct` on Fir, CPUTimeRAW (allocated CPUs x elapsed) of the base jobs `mc_rung{2,3,4}_base_heldout_s{0,1,2}`, one training per seed, and two scoring passes per seed (6 metrics jobs, 24 predict jobs). Per seed: training + 4 predict shards + 1 metrics job.
- conv autoencoder: 44.51/3 + 53.70/6 + 52.28/6 = 14.84 + 8.95 + 8.71 = 32.5 CPU-h
- conv + attention: 17.35/3 + 49.52/6 + 50.13/6 = 5.78 + 8.25 + 8.36 = 22.4 CPU-h
- U-Net: 33.38/3 + 73.24/6 + 54.02/6 = 11.13 + 12.21 + 9.00 = 32.3 CPU-h
- Total: 3 seeds x (32.5 + 22.4 + 32.3) = 3 x 87.2 = 262 CPU-h for 9 runs. The reference is reused (0 runs), unless trap 6 forces a base retrain (another 262 CPU-h).
