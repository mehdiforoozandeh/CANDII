---
title: "A4: read-derived QC metrics on the input side"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A4: read-derived QC metrics on the input side

## 1. What the PI asked

- `long-horizon decisions.md`:52 (D6): "**Arm "QC on input only".** As above, plus read-derived QC metrics (e.g. NSC, RSC, NRF, PBC, FRiP, fragment length) in C only."
- `long-horizon decisions.md`:53-54: "Read-derived QC never enters C'. A QC value computed from the target's own reads leaks the target." and "The other ablations assume the default arm until this comparison runs."
- `long-horizon decisions.md`:118: "The two arm pairs (D6 and D15) are each tested once against the default, with the other forks held fixed. They add 2 comparisons × at least 3 seeds to the ladder."

Success criterion served: 4 (design settled). Audit rows: D21, A13.

## 2. Question

Does adding read-derived QC metrics to the input covariates C (never to C') improve the three goals over the metadata-only default?

## 3. State today

- QC side table incomplete. `results/t211/SUBMITTED.md`: of 2,662 training and validation rows, NSC, RSC and fragment length are filled for 447 (0.168), NRF, PBC1 and PBC2 for 528 (0.198), FRiP for 0. The QC array covered the 2,215 rows that lacked a value, in 6 batches. About 1,908 QC files exist on Nibi `/scratch/mforooz/onion/qc` (audit row A13). The collect step never ran (`t211` holds only `SUBMITTED.md`).
- Known defect before collect (`t211/SUBMITTED.md` L67): `qc_fill.py collect` writes a row for every measurement, and `dataset.py::_apply_qc_table` overrides listed rows, empty cells included, so the 447 build-filled rows would lose their values. Two fixes are named there.
- FRiP stays empty (no peaks called); NRF and PBC exist only for base and no-dedup rows; arm rows get NSC, RSC and fragment length only (`t211/SUBMITTED.md` L71).
- The arm key exists: `stage2_arms.py` `ARMS["qc_on_input"]` (needs `qc_table`); it is listed in the done-rule set `D14` of that file.

## 4. Prior leads

- `onion data plan.md`, section "qc-metrics-as-covariates" (background): read it for the expected value of QC covariates.
- `results/t211/SUBMITTED.md` L37: the QC route reproduced an earlier smoke on ENCSR000AKT (NSC 1.592533, RSC 1.347368, fragment length 165). Prediction: the values are correct where filled.
- No model result exists (`COVERAGE.md` row 23: UNTESTED). Prediction: small effect on imputation (QC describes the input quality, which the counts already show); possible gain on correction pairs where the source and target differ in duplicate rate or enrichment.
- Leak control: depth hidden on the input side hurt the U-Net (composite 0.382 against 0.444, `VIEWS_ABLATION.md`), so input-side measurement descriptors carry information the model uses.

## 5. Work

1. Count the QC records on Nibi against the 2,215 submitted rows; resubmit failed rows once.
2. Fix the collect so a row with no record keeps its `measurements.tsv` values (fix 1 of `t211` L67); add a test.
3. Collect the side table; report fill rates per column and per view kind.
4. Write the `qc_on_input` arm configs on `final/rung{2,3,4}/reference_equal.json`; check that `c_out` (C') is byte-identical with and without the arm on a batch.
5. Submit on Fir CPU, 3 architectures x 3 seeds = 9 runs, scored into `p16_sv`.
6. Read with `select_rule.py arm-summary`; log the D13 outcome; record a finding.

## 6. How the result answers the question

Build checks: the side table has one row per measurement; the 447 build-filled rows keep their values (diff = 0); `c_out` of a batch is identical with the arm on and off.

| outcome | conclusion |
|---|---|
| QC arm beyond the default on composite or correction, 2+ architectures | QC on input enters the design (D6 arm 2) |
| within the range on all three | by D13 the simpler arm (metadata only) stays |
| worse beyond the range | QC values add noise or leak; inspect the pairs where they differ |

## 7. Dependencies, cost, risks

- Depends on the QC table fill (steps 1 to 3). Coordinate with D1 and D3: the arm should read the same tables version as its reference.
- Cost: QC fill 0.22 CPU-h per row (`t211/SUBMITTED.md` L39), about 490 CPU-h for 2,215 rows, mostly spent; collect under 5 CPU-h; arm 262 CPU-h (`ablation_specs/1_control_input.md` section 7).
- Risk: FRiP, the QC metric named by the PI that most describes enrichment, stays empty. State this as a limit of the arm.

## 8. Done when

The QC side table is collected with fill rates reported, the 9 runs are scored, and the D13 outcome of "QC on input only" against "metadata only" is logged in the notepad.

## Specification as worked (2026-11-17, agent CO-A4)

Branch `chunk/onion-CO-A4`. Numbers were read on Nibi on 2026-11-17 from `ONION/tables/eic_v1_2/measurements.tsv`, `ONION/CO-D2/tables/eic_v1_2x/measurements.tsv`, the records in `/scratch/mforooz/onion/qc/`, `ONION/eval/eic_v1/evalset.json` and `sacct`.

### Corrections to section 3 ("State today")

- The QC records number 953, not about 1,908 (855 ok, 94 no read set, 4 refused). Only batches 1 and 2 (Nibi jobs 27164529, 27235942, 400 tasks each) and a 153-task job 27332401 ran. Batches 3 to 6 were never submitted. So 1,262 of the 2,215 rows have no record (1,131 training, 131 validation).
- The collect defect of `t211` L67 is already fixed (commit 7bfad4a; test `test_collect_keeps_the_build_values`).
- `frip` is empty in every row of eic_v1_2 (2,740 rows) and of the add-on eic_v1_2x (2,811 rows). Blind rows (78) have no QC value except `frag_len` (72).
- Second experiments (eic_v1_2x, role `evalonly`, 71 rows): NSC filled for 28, NRF for 30. These rows are targets only (the dataset refuses them as inputs), so their QC never enters C.
- The 953 records were copied to one tar on /project: `ONION/CO-A4/qc_records_2026-11-13.tar` (md5 024964da982ce098ae6160426da50309).

### Question and arms

Question: does adding read-derived QC (NSC, RSC, NRF, PBC1, PBC2, fragment length; FRiP stays empty) to the input covariates C, never to C', improve the composite or the correction term over the metadata-only default? Arm: `qc_on_input` on `final/rung{2,3,4}/reference_equal.json` (configs `tools/minicandi/configs/ablations/qc_on_input/rung{2,3,4}/qc_on_input.json`), 3 seeds, 80,000 windows, tables eic_v1_2, scored on all 256 regions (`p256_sv`). Reference: `eic_v1_2_rung{2,3,4}_base_heldout`. The trunk width is the base width on all three models (60, 104, 25); the arm adds 896 parameters (embedding input).

### What the model may know at inference (the leak rule)

An input's QC must be computable from the reads of that input alone, as a user can compute it from the track they give. Three cases break this in the data, and the code handles each:
1. The NRF, PBC1 and PBC2 of any row other than a nodedup view come from reads before duplicate removal, which the input does not hold. A pooled nodedup target holds the pre-dedup reads of every replicate of its experiment: 396 replicate-level base rows (342 training, 54 validation) carry NRF/PBC from their own pre-dedup reads, a subset of the pooled nodedup's (independent review, 2026-11-17). Rule (dataset, always on with `qc_on_input`): when any nodedup target of the window has the same `experiment`, those three cells of a non-nodedup input are written absent. A nodedup input keeps them (its own reads). In eic_v1 scoring this concerns at least 43 correction pairs (base source, nodedup target); training draws the same pair kinds.
1b. 24 crop rows (17 training, 7 validation) carry NSC, RSC and fragment length copied from the parent base row (equal in 24 of 24, no own record); 25 non-base rows carry the parent's fragment length. In a crop-to-base pair the input would carry the target's cross-correlation values. Rule: `collect` never keeps a build value of a non-base view (it prints the number of blanked cells), and the "whole" fill pass measures every non-base training/validation row that has no own record.
1c. The `qc_table` values reach C only. The dataset rows keep the frozen values, so a target's `fragment length` (read by the signal-space scoring) is the same with the arm on and off (test).
2. A scored correction source is read from its `pr1` half, but its QC comes from the whole read set, whose `pr2` half may share reads with the target's `pr2` half. Rule: config `qc_table_pr1`, a side table measured on the `pr1` half (`qc_fill.py run --half pr1`); a `pr1` input reads only that table, and a row it does not list is all absent.
3. Accepted limit: in same-view reconstruction (and same-view training pairs) every QC value is computed from all reads of the track, hidden bins included. The depth covariate has the same property.
C' never holds QC (D6; tests check that `c_out` is identical with the arm on and off).

### QC values of validation and blind rows

- As targets (imputation, correction, blind), their QC never enters the model (C' is metadata only).
- As inputs: all 275 scored correction sources are validation rows. Their QC enters C under rule 2. The final model (`train_on_validation`) also reads validation rows as training inputs, so the fill covers the 131 validation rows without a record.
- Blind rows are never inputs (blind cells are served by their training tracks).
- Standardization uses fixed constants (no fit), so no statistic of validation or blind rows reaches the model. Two constants were changed from training rows only: NSC (1.1, 0.1) -> (1.1, 0.3) and PBC2 (5, 5) -> (14, 15), because the 90th percentiles sat at z = 7.8 and 8.0 and the maxima at z = 40 and 47 (training rows: NSC median 1.09, p90 1.88; PBC2 median 13.6, p90 45). Every standardized QC value is clipped to +-5.

### Missing values

An empty QC cell is written as `QC_ABSENT` (-9999). The embedding reads each QC column as a pair (standardized value, presence flag); an absent value enters as value 0 with flag 0. Before this change an absent cell was the column centre, which the model could not tell from a typical value. This matters because absence is structured: NRF and PBC exist only for base and nodedup rows, and no route exists for the 284 rows aligned by the native redacted-BAM pipeline. A fragment length of 0 or less (xcor's peak at shift 0 in 15 ATAC-seq/DNase-seq records) is written absent by `collect`. A dropped QC field (A12 switch) reads as absent.

### Fill passes that must run BEFORE `/scratch/mforooz/onion/work` leaves scratch

Both read the tagAligns under `onion/work` (checked by dry run: 0 of 4,043 files missing) and the `counts.h5` under `/scratch/mforooz/onion/measurements`. Neither writes to /scratch: each array task keeps its records in `$SLURM_TMPDIR` and writes one tar to `/project/def-maxwl/mforooz/ONION/CO-A4/fill/`.
1. Pass "whole": 1,287 rows (1,149 training, 138 validation; 190 have no read-set route and end in seconds), 3,618 files, 308 GB read, 65 tasks of 20 rows. (Before the review fix: 1,262 rows.)
2. Pass "pr1": the 275 scored correction sources (36 without a route), 475 files, 39 GB read, 14 tasks.
The collect step reads only the tars on /project and can run after `onion/work` leaves.

### Cost (allocated CPU-hours)

- Fill rate measured on the 953 records: 222.3 CPU-h (sacct of the 3 jobs) / 859 rows with a read set = 0.259 CPU-h per row (4 CPUs).
- Pass whole: 1,097 x 0.259 = 284 CPU-h. Pass pr1: 239 x 0.259 = 62 CPU-h (upper bound; a half has half the reads). Collect: 0.5 CPU-h. Fill total about 347 CPU-h, 80 jobs, about 164 files on /project.
- Training: the eic_v1_2 base runs used 82.2 CPU-h for 9 runs (conv autoencoder 17.9, conv + attention 43.0, U-Net 21.2). Scoring on 256 regions: 3 x 30.0 + 3 x 25.4 + 3 x 52 = 322.2 CPU-h. Arm total 404 CPU-h, 90 jobs.
- Item total about 751 CPU-h. Prerequisites not counted here: eic_v1_2 eval set and 256-region baselines on the training cluster, and the 9 reference scorings into `p256_sv`.

### Expectation (before any run)

Fill after the 953 old records alone: NSC on 1,302 of 2,662 training and validation rows (0.49), NRF on 843 (0.32); the scoring inputs: imputation inputs 188 of 245, correction sources 125 of 275. After pass whole, NSC is expected on at most about 2,374 rows (all but the 288 rows with no route or a refused read set), NRF only on base and nodedup rows. Prediction: no effect beyond the seed range on imputation; a possible gain on correction, mostly on pairs that differ in duplicate rate.

### Decision table

| outcome (composite or correction-only, 256 regions) | conclusion |
|---|---|
| beyond the seed range, better, on 2 or 3 architectures | QC on input enters the design (D6 arm 2) |
| within the seed range on all three | by D13 the metadata-only default stays |
| worse beyond the range | QC adds noise or leaks; read the 43 base-to-nodedup pairs and the pairs by view first |

### Run order

`MODE=fill` (Nibi, before `onion/work` leaves), then `MODE=collect` (Nibi), then the side tables to the training cluster, then `MODE=train`, all in `tools/minicandi/configs/ablations/qc_on_input/submit_qc_on_input.sh` (dry run by default).

### Limits

FRiP is empty (no peaks are called). 284 rows have no read-set route and stay absent. Accepted limit: the foreign-covariate scoring path (`score.py`) builds C from the dataset rows directly, so it applies neither the leak rules nor the QC side table; do not run it on this arm without that change. A fill pass is never resubmitted: the script refuses when `fill/rows_<pass>.txt` or a chunk tar exists.
