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
