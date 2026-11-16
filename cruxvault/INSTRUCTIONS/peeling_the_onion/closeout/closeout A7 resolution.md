---
title: "A7: resolution, 1 to 125 bp"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A7: resolution, 1 to 125 bp

## 1. What the PI asked

- `peeling the onion.md`:69: "do different types of variation captured from different sources depend on resolution? for example, do we see most of the data-processing artifacts at very granular resolutions? or do we see cell-type specific variations at let's say 500bp resolution mostly?"
- `long-horizon notepad.md`:394 (PI note 5): "The resolution of the input and output count signal matters. How much variance is lost because 25 bp is too coarse? Ablate from 1 bp to 128 bp."
- `long-horizon notepad.md`:400-401 (rulings 28, 29): "The 1 bp to 128 bp ablation and the analysis of variance lost below 25 bp run on a subset of experiments. The final model stays at 25 bp in this run." and "Resolution grid. 1, 5, 25, 50 and 125 bp."
- `long-horizon decisions.md`:32: "The resolution question in [[peeling the onion]] is an analysis, i.e. a study of which variance sources live at which scales."

Success criteria served: 3 (evals) and 4 (design tested; a reported result, not a selection). Audit rows: D25, B12.

## 2. Question

(1) How much count variance, and how much replicate-reproducible variance, lies below 25 bp, per assay and per onion layer? (2) At an equal token count, does a model at 1, 5, 50 or 125 bp score better than at 25 bp on imputation and calibration, when all are scored on common coarser tables?

## 3. State today

- Design written: plan Amendment 3 section C (`.orchestrate/plan.md` L104-106): the variance analysis on 1 bp arrays (nested block chains 1-5-25-125 and 1-5-25-50); 8 model arms (res_bp 1, 5, 25, 50, 125; matched-context `res5_ctx`; decoupled `in5_out25`, `in25_out5`) on the conv autoencoder and the U-Net, 3 seeds = 48 runs; four score tables at 25, 50, 125 and 250 bp (foreman decision h, L72).
- Model and scorer code merged (chunk RM: `tools/minicandi/res_dataset.py`, `res_adapter.py`, `tools/onion_eval/res_score.py`, `resolution.py`).
- No 1 bp recount and no arm (`COVERAGE.md` row 33). Read sets: on 2026-11-12, 117 of 376 needed experiments had a complete base read set (`results/t240/READSETS.md`, snapshot during the build); the oldest needed read set purges 2027-01-10.
- Storage: the region store takes 124 GB; 1 bp genome-wide would take 1,131 GB (notepad L410).

## 4. Prior leads

- `results/t305/LAYERS_REAL.md`: at 25 bp the sampling layer dominates every assay (76% to 97%); the cell-type layer leads at the broad and domain bands (share up to 0.90). Prediction: most variance below 25 bp is sampling; the reproducible share below 25 bp is small for histone marks and larger for DNase-seq and ATAC-seq.
- `onion eval protocol.md`, sections "resolution-question-prior-evidence" and "resolution-study-open-points" (background). Read them before the run.
- Repetition confound (plan L105): 80,000 windows are about 49 passes at 125 bp and about 0.4 passes at 1 bp. Prediction: fine-resolution arms are under-trained at equal windows.

## 5. Work

1. Recount: re-check `READSETS.md` on Nibi (the build has finished since the snapshot); recount the needed experiments at 1 bp on the evaluation and training regions into the region store (chunk RCa).
2. Variance analysis (chunk RVa): per assay and populated layer, the raw share below 25 bp and the replicate-reproducible share at each chain step (cross-covariance of pseudoreplicate halves).
3. Arms (chunks RMa, RMb): the 8 arms on the conv autoencoder and U-Net at their reference recipes, 3 seeds; record FLOPs per window in `run.json`.
4. Score each arm at every table its output bin divides; report imputation and calibration terms with seed ranges (no correction term: the tables carry no designed pairs).
5. Write `RESOLUTION.md`; report only (ruling 28).

## 6. How the result answers the question

Acceptance checks: `RESOLUTION.md` holds the variance-below-25-bp table per assay and layer, and four score tables; each arm row gives FLOPs per window and passes per region.

| outcome | conclusion |
|---|---|
| reproducible share below 25 bp under 5% for histone marks | 25 bp loses little for histone marks; keep it for scaling |
| a finer arm beats 25 bp beyond the range on the common table | resolution matters for the model, not only for the data; weigh it for scaling |
| `res5_ctx` beats `res5` but not 25 bp | the gain is context, not resolution |
| decoupled arms differ (`in5_out25` against `in25_out5`) | the input and the output sides need different resolutions |

## 7. Dependencies, cost, risks

- Depends on the 1 bp region store (recount).
- Cost: recount about 130 CPU-h; arms about 475 CPU-h (`COVERAGE.md` row 33). Region store 124 GB on Nibi `/scratch`.
- Risk 1: read sets on `/scratch` age out (60-day rule); recount before 2027-01-10.
- Risk 2: unequal FLOPs and passes across arms (plan L105); state both beside every score.

## 8. Done when

`RESOLUTION.md` reports the variance below 25 bp per assay and layer and the 48 arm runs on the four common tables, with the finding recorded in the crux tree.
