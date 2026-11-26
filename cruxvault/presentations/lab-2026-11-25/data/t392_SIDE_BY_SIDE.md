# Read-start miniCANDI against the read-count reference, 16 regions

Task t392. Data eic_v1_2, the 16-region evaluation set (`evalset_p16.json`), 3 seeds per model. Written 2026-11-25 (UTC). This folder is hand-written prose (this file) plus the outputs of `runall.sh` (listed in `MANIFEST.tsv`).

## Models

- **Read-count reference**: `rcp_halves_mix_avg_pearson_unet`, the full-combination miniCANDI (U-Net, read-disjoint pairs, average input, Pearson input) at 1x windows, trained on 25 bp overlap counts. Its scores are reused from task t370 (`ONION/runs_rcp`), never rerun.
- **Read-start model**: `rs_halves_mix_avg_pearson_unet`, the same recipe trained on 25 bp read-start counts: 2 strand channels for ChIP, 1 channel for DNase and ATAC. Inputs, the average input and the targets are read starts (D16). For scoring, each prediction is converted to overlap counts: the span kernel (the fraction of reads that cover each of 17 bins, per strand) is applied, and the result is moment-matched to an NB (`read_start.convert_nb`). Then the unchanged headline and deviance scorers run.

## Result

`table.tsv`; mean over seeds 0-2, with the range (max - min over seeds) in brackets.

| model | imputation | correction | calibration | deviance imputation skill | deviance imputation R2 | deviance correction gap closed | deviance calibration ratio (1 = calibrated) |
|---|---|---|---|---|---|---|---|
| read-count reference | 0.4996 (0.0349) | 0.8156 (0.0105) | 0.6852 (0.0924) | 0.5426 (0.0653) | 0.3423 (0.0175) | 0.9508 (0.0023) | 1.0584 (0.0398) |
| read-start model | 0.4568 (0.0070) | 0.7826 (0.0362) | 0.6064 (0.0679) | 0.3746 (0.0870) | 0.2886 (0.0324) | 0.9452 (0.0044) | 1.3081 (0.0965) |

The headline columns are `imputation_raw`, `correction_v3_raw` and `uncertainty_raw` of `select_rule.py arm-summary`. The deviance columns are the `macro` values of `scores.json["deviance"]` (PR 65 scorer, `tools/onion_eval/deviance.py`). Per-seed values: `per_seed.tsv`. Figure: `side_by_side.png`.

![side by side](side_by_side.png)

The read-start model is lower than the reference on every term. The imputation gap is larger than the seed range of both models: 0.043 on the headline term and 0.168 on the deviance skill. The read-start seeds all fall below the lowest reference seed on both. The deviance correction gap is small (0.0056) but larger than both seed ranges, and every read-start seed (0.9432-0.9476) is below every reference seed (0.9493-0.9516). The headline correction gap (0.033) and the headline calibration gap (0.079) are each smaller than at least one model's seed range. On the deviance calibration ratio, every read-start seed (1.25-1.34) is further from 1 than every reference seed (1.03-1.07), so the read-start model is overconfident. Part of that overconfidence is expected from the conversion. The moment match treats the start counts of neighboring bins as independent, so it understates the variance of the overlap count where the rate is spatially correlated. This biases the calibration term and `dev_calibration_ratio` of the read-start model toward overconfidence. The size of this bias is not measured. The noise floor of these numbers is the seed range in each cell (AGENTS.md 7.2); 16 regions are a screen, not a confirmation (D3).

## Separate checks (not scorer metrics)

**Sanity against read-start truth** (`sanity.tsv`, `tools/t392/sanity.py`; jobs 5808635, 5851878, 5851884): the inner model's prediction against the read-start counts on the 63 imputation targets of the 16 regions. Mean NB deviance per bin is 0.5271-0.5429, and the Pearson r of log1p mean against log1p truth is 0.4449-0.4572 over the 3 seeds.

**Conversion floor** (`run_state/notes/build/BUILD.md`, from the store's `provenance.json`): the nMSE between the observed read starts convolved with the span kernel and the observed overlap counts. This is the error that the conversion alone adds. The `starts.h5` median is 0.0307 for ChIP histone (p95 0.0918), 0.0722 for ChIP control, 0.0137 for DNase and 0.0237 for ATAC. The read-start store gate passed on 8,220 of 8,220 files with 0 differing bins.

**Open edges** (`open_edges.tsv`): each run of consecutive windows is converted as one array, and its 2 ends are filled from the edge bin. On shards 1-3 (shard 0 also holds the copy-leak guardrail windows), every seed has 0.25000 open edges per window. That is 0.25 x 8 / 2,048 = 0.195% of bins within 8 bins of an open edge, below the 0.25% limit.

**Copy-leak check**: the scorer's copy-leak wiring check changes the batch's `x_counts`. The read-start adapter does not read `x_counts`, so for this model the check does not reach the inner model. Input masking is held by the adapter's tests (`tests/test_t392_adapter.py`).

## Compute

Total 836.2 CPU-h of the 900 CPU-h limit (CPUTimeRAW from sacct; plan Log of `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/plan.md`):

- About 574 CPU-h before the final scoring (plan Log total). The itemized parts sum to 567.4; the rest is not itemized here. The parts are the store build and tests (124.9), training (365.8; jobs 5691700/01/02), the seed-0 scoring that timed out on kit RS5 (42.7; jobs 5808608/10/11/12), cancelled seed 1-2 scoring (21.8), sanity jobs (about 1.7) and the staging-fix probes (10.5; jobs 5855901, 5856026, 5864419, 5865079, 5866796).
- 262.2 CPU-h of final scoring on kit RS7 (git fe138144). Predict: 5874564/65/67/68, 5874996-99, 5875002-05. Headline metrics: 5874569, 5875000, 5875006. Deviance: 5875186/87/88. Repeat sanity: 5874570.

## Deviations from the instructions

1. Stop limit raised from 600 to 750 CPU-h (D24, PI, 2026-11-24). Reason: a Fir stall of about 12:00-17:55 UTC on 2026-11-24 held all 3 training seeds for about 21,000 s (about 213 CPU-h).
2. Stop limit raised from 750 to 900 CPU-h (D25, PI, 2026-11-24). Reason: read-start scoring ran about 2.7x slower than the reference per shard, even after the local-copy fix.
3. Scoring stages the read-start files, the read-start average and the overlap average to node-local disk, for read-start runs only (kit RS6/RS7; `ReadStartAdapter.stage_read_starts`, `predict_store._stage`). The values read are bit-identical (8 of 8 probe requests); the read-count path is unchanged.
4. Scoring settings that differ from the reference: no `--nice=100`; shorter time limits (predict 02:00:00 for seed 0, 03:30:00 for seeds 1-2; metrics 00:55:00; deviance 01:30:00); one open-edge log per shard.
5. The deviance pass reads the kept `p16_sv` prediction stores (metrics phase only, `ONION_SCORE_DEVIANCE=1`) instead of a new predict pass. The reference deviance scores are reused from t370.
6. The deviance columns are in this folder, so the separate deviance addendum folder was not made (D23).
7. The runall contract needs `runall.sh` and `MANIFEST.tsv` at the task folder root (`tests/test_results_folders.py`). This folder is therefore `results/t392/`, not the `results/t392/side_by_side/` the plan named.
