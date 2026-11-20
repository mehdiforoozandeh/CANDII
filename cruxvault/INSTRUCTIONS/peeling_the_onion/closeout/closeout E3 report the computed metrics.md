---
title: "E3: report the metrics that are computed and never reported"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# E3: report the metrics that are computed and never reported

## 1. What the PI asked

- `onion eval protocol.md`:63: "Every headline metric is quoted per assay as a macro average, with its seed band."
- `long-horizon decisions.md`:88 (D12): "Max's error budget (level, shared and specific, explained / unexplained / noise / added) and his explained and unexplained variance evaluation are computed for every miniCANDI arm."
- `long-horizon notepad.md`:422 (ruling 32): the explained fraction per layer "is a reported metric beside the headline set. It does not enter the selection rule."
- `long-horizon decisions.md`:66 (D11b): "**Accessibility-only (second table).**"
- `peeling the onion.md`:58: "then we want to show how much of the variation  is being captured by CANDI (and other methods) and how much of the remaining variation is it putting into its predicted aleatoric uncertainty and how calibrated it is."
- `peeling the onion.md`:60: "e.g. we can measure how much variance we are predicting in different subsets of the genome: e.g.. promoters, enhancers, gene bodies, foreground, background, heterochromatin, etc."

Success criterion served: 3 (well defined and tested evals). Audit rows: B5, B9, B11, B14, and the model side of B3.

## 2. Question

Publish, for the trained models, the metric tables that `scores.json` already holds but no report shows: headline metrics 2, 3, 5 and 7 per assay, the explained fraction per variance layer, the error budget, the accessibility-only regime, and per-stratum values.

## 3. State today

- Keys present in `scores.json` (audit rows B5, B9, B11, B14; read on Nibi for `rung4_base/seed0/p16`): `r_spec`, `count_crps_skill_vs_position_average`, `depth_law_abs_log2_error_never_trained`, `predicted_over_replicate_variance`, `explained_fraction_per_layer`, `error_budget`, `accessibility_only`.
- Model reports (`onion paper.md` Results 4-7, `results/final_round/FINAL_ROUND_EQUAL.md`) show only the composite and its three lead terms.
- The table tool exists: `tools/onion_eval/arm_report.py` with subcommands `headline`, `dev-evals`, `accessibility`, `explained`, `per-knob`, `shared-specific`. It refuses to mix evaluation sets or scorer keys.

## 4. Prior leads

- `results/t305/LAYERS_REAL.md`: the sampling layer is 76% to 97% of the three layers per assay at 25 bp; the cell-type layer is 10% to 23%. Prediction: the explained fraction is near 0 on the sampling layer for every model and is informative only on the cell-type layer.
- `results/t307/BASELINES.md` and `onion paper.md`:111: the baselines' metric-7 ratio at the pseudoreplicate rung is 1.21 to 16.97 for the position average. Prediction: models are also above 1 (over-dispersed) at that rung; the upper bound waits for D2.
- `FINAL_ROUND_EQUAL.md`: the U-Net leads on variance recovered (0.432 to 0.445) and trails on gap closed (0.507 to 0.534). Prediction: per-assay tables show where that trade sits.

## 5. Work

1. Choose the runs: the references and final-round arms of the five CPU architectures (Fir `runs_fir`, sub `p16_sv`), the pointwise MLP when A10 scores it, and the baselines (`eval/eic_v1/baselines_p16`).
2. Run on Fir, read-only on the scores: `arm_report.py headline --composite-version 3`, `dev-evals`, `accessibility`, `explained`, `per-knob`, and `shared-specific` with each reference first.
3. Write `results/<task>/METRICS_REPORT.md`: one table per metric, rows = models and baselines, columns = scorable assays and the macro, each cell mean (seed range). Add per-stratum rows (promoters, enhancers, gene bodies, heterochromatin, mappability deciles) for metric 1 and PIT deviation.
4. Add `runall.sh` and `MANIFEST.tsv` so the tables rebuild from the stored scores.
5. Repeat step 2 on the settled scores of E1 and on the new dataset version of D3, when they exist.

## 6. How the result answers the question

Acceptance checks:
- `METRICS_REPORT.md` holds tables for metrics 2, 3, 5 and 7, explained fraction per layer, error budget, accessibility-only and per-stratum values, each per assay with a seed range.
- `bash runall.sh` on Fir rebuilds every table byte-identical (MANIFEST hashes match).
- Each table names its evaluation-set hash, tier (screening or settled) and composite version.
- Every number in the report traces to a `scores.json` path.

## 7. Dependencies, cost, risks

- Depends on nothing; reruns follow E1 and D3.
- Cost: the tool reads stored files; under 2 CPU-h. About 1 agent-day for the write-up.
- Risk 1: metric 7 at the upper bound is undefined until D2; label it "lower bound only".
- Risk 2: H3K9ac is scored in count space from a replicate view (notepad L475); flag it in every per-assay table.
- Risk 3: the accessibility-only regime holds 20 items on `eic_v1` (audit row B14); state its width.

## 8. Done when

`METRICS_REPORT.md`, with `runall.sh` and `MANIFEST.tsv`, reports every listed metric per assay with seed ranges for all scored architectures and the baselines, and its numbers enter the close-out report (R2).
