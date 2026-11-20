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

## 9. Specification as worked (CO-A7, 2026-11-17)

### 9.1 State today, checked against the artifacts

- No store or recount finer than 25 bp exists. Searched on Nibi and Fir (read-only `find`, depth 4, names `point1bp*`, `*res_v1*`, `*1bp*`, `*bin5*`) under `/project/def-maxwl/mforooz/ONION` and `/scratch/mforooz/onion`: no hit. Neither cluster has `tables/eic_res_v1` or `/scratch/mforooz/onion/res`. The only 1 bp files that ever existed were the synthetic store of the Fir smoke (`results/t248/ARM_SMOKE_3.md`).
- The read sets are now complete (section 3 quoted 117 of 376 from the snapshot of 2026-11-12). On 2026-11-17 every view that the recount needs has its half tagAligns under Nibi `/scratch/mforooz/onion/work`: 412 views, 0 missing, 116.9 GiB of tagAlign (`closeout_a7_submit.sh plan`, dry run; the views are listed in 9.3).
- The resolution code on the umbrella branch (21a637a) was broken: 15 of the 27 tests of `tests/test_minicandi_resolution.py` failed. Cause 1: `OnionDataset` later gained an attribute `regions` (its own table of training regions), and `ResDataset` overwrote it with the store's region list. Cause 2: `train.py` refuses config keys that no code names, and `res_subs_of` read `res_in_sub` and `res_out_sub` through a loop variable. Both are fixed on this branch (27 of 27 pass).
- Nibi `/project` on 2026-11-17: 4,255 of 4,657 GiB used (402 GiB free), 456K of 500K files (44K free) (`diskusage_report`); the foreman read 464K of 500K later the same day. Nibi `/scratch` refuses writes.

### 9.2 Question and design

Question: (1) how much count variance, and how much replicate-reproducible variance, lies below 25 bp, per assay and per onion layer; (2) at an equal parameter count and an equal token count, does a model at 1, 5, 50 or 125 bp score better than the same model at 25 bp, on common coarser score tables.

Part 1 (variance analysis, no model): `scale_variance.py` on the validation views of the store, chains 1-5-25-125 and 1-5-25-50, as plan section C. It is the cheap core of the answer.

Part 2 (model arms). Arms: `res1`, `res5`, `res25`, `res50`, `res125` (1,024 tokens each), `res5_ctx` (5,120 tokens of 5 bp = 25.6 kb), `res125_ctx` (208 tokens of 125 bp = 26.0 kb; 205 tokens would give 25.6 kb, but the trunks downsample by 16, so the context must be a multiple of 16), `in5_out25`, `in25_out5` (25 bp tokens; 5 bp sub-bins on the input or the output side). Configs: `tools/minicandi/configs/resolution/rung{2,3,4}/<arm>.json` = the base recipe `base_heldout.json` plus the arm key and `res_store`. Rung 3 (conv + attention) takes `res_bp` only; the two decoupled arms are rungs 2 and 4 only.

### 9.3 What makes the comparison fair

1. Same regions and the same counting rule. Every arm, the 25 bp reference included, trains on the same 512 training regions of 409,600 bp and is scored on the same eval set (64 eval and 16 fit regions drawn, seed 0, inside the 256 eval regions of `eic_v1`). The reference is `res25`, retrained from the store. The existing 25 bp base runs are no reference: they train genome-wide on counts made with the overlap rule, which does not add across bin sizes.
2. Same number of tokens, not the same context in bp. Reason: at 1,024 tokens every arm has the same parameter count (the trunk width is fitted to the 1.0e6 budget; measured in the Fir smoke, `results/t248/arm_smoke_3.tsv`: 1,042,514 for every rung-2 arm, 993,599 for every rung-4 arm) and nearly the same compute per window (FLOPs per window, rung 2: 1.92e9 at 1 bp, 1.95e9 at 5, 2.11e9 at 25, 2.31e9 at 50, 2.90e9 at 125). That is the trade a model of fixed size faces. The context then runs from 1.0 kb (1 bp) to 128 kb (125 bp), so context and resolution are confounded. `res5_ctx` removes the confound at 5 bp: the same 25.6 kb as `res25`, 5 x the tokens and about 4.6 x the FLOPs (9.76e9 / 2.11e9). `res125_ctx` removes it at 125 bp: 26.0 kb, 208 tokens, about 0.28 x the FLOPs (0.203 x 2.90e9 / 2.11e9). A context-matched arm has fewer or more tokens, so it is matched in context, not in compute. A context-matched 1 bp arm (25,600 tokens) is cut for cost.
3. Evaluation at a common resolution. A prediction finer than the table is aggregated by exact sums: for the k output bins inside a table bin, `mu = sum mu_i` and the negative binomial size by moment matching under conditional independence, `n = (sum mu_i)^2 / sum(mu_i^2 / n_i)`; the comonotone bound `n = (sum mu_i)^2 / (sum mu_i / sqrt(n_i))^2` is reported beside it, and the PIT deviation of the pooled prediction says which holds. The truth is the block sum of the 1 bp point-rule counts. Nothing is interpolated or split: a 125 bp prediction is scored only at 125 and 250 bp, a 50 bp prediction only at 50 and 250 bp. Four tables: 25 bp (arms 1, 5, 25 and both decoupled arms), 50 bp (1, 5, 25, 50), 125 bp (1, 5, 25, 125), 250 bp (all). Span: the whole region at 25 and 50 bp (16,384 and 8,192 bins); the first 256,000 bp at 125 and 250 bp (2,048 and 1,024 bins). Terms per table: imputation gap closed, uncertainty gap closed (calibration), PIT deviation, count CRPS with its oracle-scaled and scale-error split; no correction term (the store tables carry no designed pair).
4. Parameter count and compute are recorded per run (`run.json`: `param_count`, `flops_per_window`) and printed beside each score, with the passes over the training regions.
5. Cost growth with finer bins. (a) Training per window is nearly flat at 1,024 tokens (FLOPs = a + b x res_bp, a = 1.91e9 and b = 7.9e6 per bp on rung 2; a = 1.47e9 and b = 7.75e6 on rung 4, fitted to the smoke values). (b) Coverage falls: 80,000 windows of 1,024 tokens cover 0.39 passes of the 210 Mb training regions at 1 bp (80,000 x 1,024 bp / 209.7 Mb) and 49 passes at 125 bp (80,000 x 128,000 bp / 209.7 Mb). (c) Prediction grows as 25 / res_bp below 25 bp: windows per eval region are 400 at 1 bp, 80 at 5, 16 at 25, 8 at 50, 3 at 125. (d) Matched context grows as 25 / res_bp in tokens (attention quadratically on rung 3). (e) Storage: the region store is about 25 GB gzip (estimate, 9.4), against 1,131 GB raw for a genome-wide 1 bp copy.

### 9.4 Expectation, stated before any run

- Variance: from `results/t305/LAYERS_REAL.md`, the sampling layer is 76% (DNase-seq) to 97% (ChIP-seq control) of the layers at 25 bp, and the cell-type layer is 0.60 to 0.91 at the broad band. So most variance below 25 bp is sampling; the reproducible share below 25 bp is under 5% for the histone marks and larger for DNase-seq and ATAC-seq (cut-site structure).
- Arms: no finer arm beats `res25` beyond the seed range on a common table; `res125` ties `res25` at 125 and 250 bp. The fine arms are under-trained (0.39 passes at 1 bp).

### 9.5 Decision table

| outcome | conclusion |
|---|---|
| reproducible share below 25 bp under 5% for histone marks | 25 bp loses little for histone marks; keep it for scaling |
| reproducible share below 25 bp of 5% or more for DNase-seq or ATAC-seq | accessibility holds signal below 25 bp; name it for scaling |
| a finer arm beats `res25` by more than the seed range on a common table | resolution matters for the model, not only the data; weigh it for scaling |
| `res5_ctx` beats `res5`, and `res5_ctx` does not beat `res25` | the gain of the coarse arms is context, not resolution |
| `res125` beats `res25` at 125 or 250 bp beyond the seed range, and `res125_ctx` does not | the gain is the 128 kb context, not the coarse bin |
| `res125_ctx` beats `res25` at 125 or 250 bp beyond the seed range | the coarse bin itself helps at equal context (or the repetition, 49 passes at 1,024 tokens; 10 at 208) |
| `in5_out25` and `in25_out5` differ beyond the seed range | the input and the output sides need different resolutions |
| the independent and comonotone bounds give different calibration rankings | the pooled calibration of the fine arms is not decided; report both |

Which design decides which row: rows 1 and 2 (variance) are decided by both designs. Rows 3 (5 bp only), 4, 5, 6 and 8 are decided by the reduced design on the conv autoencoder only; the full design adds 1 bp and 50 bp to row 3 and decides them on three architectures. Row 7 (decoupled arms) is decided by the full design only.

Limits of the result (stated with every number):
- A null result at fine bins is not a null for resolution. At 80,000 windows the passes over the training regions run from 0.39 at 1 bp to 49 at 125 bp, so the fine arms are under-trained by construction.
- The calibration comparison at 125 and 250 bp is uneven. A coarse arm predicts the table bin directly, while a finer arm's calibration depends on an assumed pooled negative binomial size (independent or comonotone). Both bounds are reported, and a calibration ranking that changes between them is not decided (last row).
- No noise floor is measured at 64 regions. Every score is quoted with its region bootstrap interval and the seed range (max minus min over 3 seeds); a difference smaller than the seed range is reported as no difference.
- The 64 eval regions and the 16 fit regions are disjoint: `evalset.draw_regions` takes them from disjoint parts of one seeded permutation, `evalset.py check` refuses an overlap, and the `tables` part asserts it again.

Controls and traps: report passes and FLOPs beside every score (repetition confound); the 25 bp reference is `res25` from the store, never the genome-wide base; the store's eval regions are the 256 regions of `eic_v1`, so the store-only verify checks every view against the stored counts; seeds 3; seed range = max minus min.

### 9.6 What reads the work folder, and when it must run

Only the recount (and the plan and probe parts, which list, time and check it) reads `/scratch/mforooz/onion/work`. Training, scoring, the tables and the variance analysis read the store, plus the genome. The read-set check runs in the same scan as the count (`recount.py run --verify-counts`). `verify --no-rescan` later reads the store and the eic_v1 `counts.h5` files under `/scratch/mforooz/onion/measurements` (to match the sha256 recorded by the in-run check), not the tagAligns; that folder is not in the archive list.

Limits of the read-set check, and the independent check of the point rule:
- `--verify-counts` compares the reads' spans, counted with the overlap rule at 25 bp, with the stored eic_v1 counts. It proves that the read set is the stored one. It does not exercise the point rule: a wrong strand sign, an off-by-one point or a wrong fragment length still passes it.
- The `--no-rescan` sum check is circular: it compares the point files with the read count that the same run recorded. It catches a damaged file, not a wrong rule.
- The point rule is checked by two other means. (1) `tests/test_closeout_a7_resolution.py::test_point_rule_hand_fixture`: five reads written by hand on both strands, with positions computed by hand for a shift of 75 bp (`+` [1000, 1036) gives 1075; `-` [2000, 2036) gives 1960; `-` [1100, 1120) gives 1044), compared base by base with `scan_tagaligns`. (2) On real data, the probe job counts the points of its validation view on eval region 0 with a second implementation (gzip and string split, no recount code) and compares its 25 bp sums with `pool(point1bp, 25)` of the store, bin by bin (`[pointcheck] ... PASS|FAIL` in the probe log). The recount waits for that PASS.
- Fragment length. Of the 412 views, 56 ChIP-seq views take the median over ChIP-seq rows and 47 controls take the median of the experiments that name them; neither is measured on the view itself. The value sets the shift only. It is written to the `frag_len` column of the store tables (`recount.py tables`, shift > 0), and `frag_len` is a model input only under `qc_on_input`, which every resolution config leaves off (a test checks this). So a guessed length reaches no model covariate in this item. Passes that MUST finish before the folder leaves scratch:

| pass | views | regions | halves | tagAlign read | reads (tagAlign lines) |
|---|---|---|---|---|---|
| A: training experiments, pooled base views (controls included) | 312 | all 768 | no | 93.6 GiB | 13.77e9 |
| B: validation experiments, pooled base views | 43 | all 768 | yes (pr1, pr2) | 13.1 GiB | 1.98e9 |
| C: validation experiments, per-replicate `br<i>` views (isogenic rung) | 57 | 256 eval | no | 10.1 GiB | 1.55e9 |

Line counts are `depth_reads` of `tables/eic_v1_2/measurements.tsv`. Fragment length per view (no xcor read): the row's `frag_len` (103 views), the mean of its replicates (185), for a control the median over the experiments that name it (47), else the median over ChIP-seq rows (56); 0 for accessibility (21).

### 9.7 Cost (CPU-hours, allocated)

Unit costs. Recount scan: 1.5e6 to 1.75e6 tagAlign lines per second on one laptop core (synthetic 3e6-line tagAlign at 7.6 bytes per line, the real files are 7.3; `recount.scan_tagaligns`); taken as 1.0e6 per second on Nibi. Training (Fir, `base_heldout`, 80,000 windows, sacct, `closeout A6` file): conv autoencoder 5.98, conv + attention 14.34, U-Net 7.06, scaled by the FLOPs ratio of each arm. Prediction: 0.09 CPU-s per forward window (one third of the 0.27 CPU-s of a conv-autoencoder training window), x 1.18 on the U-Net, x 2.4 on conv + attention; windows = 43 targets x 64 regions x the windows per region of 9.3. Score tables: 7.5 CPU-h at 25 bp (the measured 30.0 CPU-h of one 256-region scoring of a conv-autoencoder run x 64 / 256), scaled by bins (50 bp 0.5, 125 bp 0.125, 250 bp 0.0625) and x 1.2 for the second bound. None of these is measured on the store; the probe (9.8) measures the recount and the forward times.

| step | arithmetic | CPU-h |
|---|---|---|
| recount, passes A to C | 17.3e9 lines / 1.0e6 per s = 4.8 h, + 412 views x 20 s writing = 2.3 h, + 21 x 0.1 h setup = 9.2 h; x 2 CPUs | about 18 (cap 21 x 8 h x 2 = 336) |
| tables, store verify, eval set | 1 job, 8 CPUs, about 1 h | about 8 |
| variance analysis | plan estimate | 50 |
| reduced arms: conv autoencoder, `res5`, `res25`, `res125`, `res5_ctx`, `res125_ctx`, 3 seeds = 15 runs | train 3 x (5.5 + 6.0 + 8.2 + 27.7 + 1.7) = 147.3; predict 3 x (5.1 + 1.1 + 0.3 + 5.1 + 0.3) = 35.7; tables 3 x (3 x 1.6875 + 2 x 0.1875) x 7.5 x 1.2 = 146.8 | 330 |
| full arms: 9 arms on conv autoencoder and U-Net, 7 on conv + attention, 3 seeds = 75 runs | as before (907 + 525 + 790 = 2,222) plus `res125_ctx` on 3 rungs, 3 x (3.7 + 4.2 + 6.4) = 42.8 | 2,265 |

`res125_ctx` costs: training 5.98 x 0.28 = 1.7 CPU-h (conv autoencoder); prediction 15 windows per region (409,600 / 26,000) x 43 x 64 = 41,280 windows x 0.09 x 0.28 = 0.3 CPU-h; tables at 125 and 250 bp, 0.1875 x 9 = 1.7. Reduced design total: 18 + 8 + 50 + 330 = about 405 CPU-h, 78 arm jobs (15 trainings, 15 predictions, 48 tables). Full design total: 18 + 8 + 50 + 2,265 = about 2,340 CPU-h, 396 arm jobs. The plan's 475 CPU-h for 48 runs (section 7) used the pilot scoring cost of 0.83 CPU-h, before the measured 256-region cost. The reduced design keeps `res5` (the resolution below 25 bp that the budget affords at full prediction), `res125` (the coarse side) and their two context controls, drops `res1` (25 CPU-h of prediction per run, 0.39 passes), and keeps the variance analysis, which answers part 1 for every assay without a model.

### 9.8 Storage proposal and order

- Store: Nibi `/project/def-maxwl/mforooz/ONION/CO-A7/res_store`. Size about 25 GB (estimate: 9.1% of the reads fall in the 768 regions, about 13 bytes per nonzero 1 bp position after gzip on a synthetic file, plus about 12 MB per 25 bp pooled file); 412 x 3 + 43 x 4 = 1,408 files, plus 42 logs. Fits the 402 GiB and the 36K free files. Nibi `/scratch` refuses writes; `/nearline` is tape and serves no training. Training on Fir needs a Globus copy of the store (foreman).
- Arm outputs (estimate): per run about 52 files (checkpoint, `run.json`, `DONE`, logs, `res_pred/INFO.json` and 43 target caches, 2 to 4 score tables) plus 2 log files per job. Reduced design: 15 x 52 + 78 x 2 = about 940 files. Full design: 75 x 52 + 396 x 2 = about 4,700 files. The group's `/project` was at 464K of 500K files on 2026-11-17 (foreman); the store and the full design together take about 6,200 of the 36K free.
- Order: `plan` and `probe` (DRY_RUN=0; the probe is 1 job, 2 CPUs, at most 2 h = 4 CPU-h, about 20 files; its `[pointcheck]` line must say PASS), then `recount` before the work folder leaves (a view whose check failed is redone on resubmit, never skipped), then `tables` (refused unless every plan view has a passed check), `variance` (after `tables`), `arms`. Script: `slurm/onion_data/closeout_a7_submit.sh` (dry run by default).
