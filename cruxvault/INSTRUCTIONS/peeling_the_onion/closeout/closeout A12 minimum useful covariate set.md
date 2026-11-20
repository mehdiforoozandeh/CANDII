---
title: "A12: minimum useful covariate set"
parent: "[[closeout todo]]"
created: 2026-11-17
---

# A12: minimum useful covariate set

## 1. What the PI asked

PI ruling of 2026-11-17 (verbatim): "let's have all covariate fields that we can included. one last step of this session and one of its deliverables is to find the minimum set of useful covariates. a ranked list, a curve of how much performance do we lose by excluding each covariate and which ones are basically useless and we can safely remove"

## 2. Question

Which covariate fields does miniCANDI need? The answer is (1) a ranked list of the fields, (2) a curve of the score lost as fields are excluded, and (3) the fields that can be removed with no loss beyond the seed range.

## 3. State today

- The corrected tables `eic_v1_2` hold every field that we can fill (lab, antibody, donor, platform, and the processing fields). They are the default from now on.
- Leads: scrambled covariates lower the correction-only term on all three models and leave imputation unchanged. Foreign covariates on the output side at scoring lower gap closed by 0.13 to 0.24; on the input side they change little. Numeric-only covariates are within the seed range on the conv models. Evidence: [[ABLATIONS_READ]] sections 2.7, 2.11, 3.1.

## 4. Work steps

1. List every covariate field of `eic_v1_2`, with its type and its side (input, output).
2. Screening without training: at scoring, replace one field at a time with a foreign value (the scorer option for foreign covariates), per side, on the three base recipes trained on `eic_v1_2`. Rank the fields by the score lost.
3. Confirmation with training: leave-one-field-out runs for the top fields and one run with all low-ranked fields removed, 3 seeds, 3 models.
4. Build the curve: fields removed in rank order (least useful first) against imputation, gap closed, calibration, composite and the correction-only term.
5. State the minimum set: the smallest set whose scores are within the seed range of the full set.

## 5. Done when

The ranked list, the curve and the minimum set are in one results file under `cruxvault/results/`, each number with its seed range, and the finding is recorded.

## 6. Order

This is the last ablation step of the session. It runs after the mix-and-match combinations and before the settled-design document (M1).

## 7. Specification as worked (2026-11-17, item agent CO-A12)

Written before any A12 run. No A12 job ran. Evidence: `cruxvault/results/closeout_A12/` (git-ignored).

### 7.1 Question

Which covariate fields of the tables `eic_v1_2` does miniCANDI need on the three models (conv autoencoder, conv + attention, U-Net)? Deliverables: (1) a ranked list of the fields, (2) the exclusion curve, i.e. the four rule terms and the composite against the number of fields removed, (3) the fields that are safe to remove.

### 7.2 Fields

The covariate vector is fixed by code. C' (output side) holds the 11 META columns. C (input side) holds the same 11, plus the 7 QC columns only when `qc_on_input` is set (`tools/minicandi/contract.py`:87-95; encoder `tools/minicandi/covariates.py`:104-130). The base recipes have `qc_on_input: false` (`configs/rung{2,3,4}/base_heldout.json`). So 11 fields are in scope; the QC fields enter only if item A4 adopts QC on the input side.

Statistics are from `/project/def-maxwl/mforooz/ONION/tables/eic_v1_2/{measurements.tsv,covariates.json}` on Nibi (read on 2026-11-17; script and output in the evidence folder). Rows: train 2,328, validation 334, blind 78. "Cardinality" counts distinct encoded values in training rows (log2 depth in 0.1 bins). "Coverage" is the share of rows with a non-empty cell. "Unseen" counts rows whose value no training row holds (a categorical value outside the vocabulary encodes to id 0). "Assays varying" counts the training assays in which the field takes more than one value; a foreign donor of the same assay can change a field only in those assays.

| field | type | cardinality (train) | coverage train / val / blind | val rows unseen | blind rows unseen | assays varying (of 36) |
|---|---|---|---|---|---|---|
| log2_depth | numeric | 85 | 1.000 / 1.000 / 1.000 | 0 | 0 | 36 |
| run_type_pe | flag | 2 | 1.000 / 1.000 / 1.000 | 0 | 0 | 14 |
| read_length | numeric | 12 | 1.000 / 1.000 / 1.000 | 0 | 60 | 22 |
| dedup_on | flag | 2 | 1.000 / 1.000 / 1.000 | 0 | 0 | 36 |
| mapq_threshold | numeric | 3 | 1.000 / 1.000 / 0.923 | 0 | 0 | 35 |
| pipeline_id | categorical | 4 | 1.000 / 1.000 / 1.000 | 0 | 0 | 8 |
| pipeline_version_id | categorical | 4 | 1.000 / 1.000 / 1.000 | 0 | 0 | 8 |
| lab_id | categorical | 6 | 1.000 / 1.000 / 1.000 | 0 | 0 | 13 |
| platform_id | categorical | 19 | 1.000 / 1.000 / 1.000 | 9 | 0 | 28 |
| antibody_id | categorical | 113 | 0.747 / 0.940 / 0.923 | 18 | 48 | 21 (of 33) |
| donor_id | categorical | 49 | 1.000 / 1.000 / 1.000 | 6 | 0 | 30 |
| nsc, rsc (QC) | numeric | 360 | 0.162 / 0.210 / 0.000 | 70 | 0 | 32 (of 33) |
| nrf, pbc1, pbc2 (QC) | numeric | 456-457 | 0.196 / 0.213 / 0.000 | 71 | 0 | 34 (of 35) |
| frip (QC) | numeric | 0 | 0.000 / 0.000 / 0.000 | 0 | 0 | 0 |
| frag_len (QC) | numeric | 47 | 0.162 / 0.210 / 0.923 | 5 | 0 | 31 (of 33) |

Antibody coverage counts every row; on ChIP-seq target rows it is 0.986 (train) and 0.981 (validation) (`tables/eic_v1_2/CHANGES.md`). `frip` is empty in every row, so it is useless by construction and needs no run.

Overlap between fields, as H(a | b) / H(a) on training rows (0: b fixes a; 1: b says nothing about a). Pairs at or below 0.40: pipeline_id and pipeline_version_id 0.00 both ways (identical information); lab given donor 0.15; lab given cell 0.16; donor given cell 0.05 (donor is a proxy of cell identity); assay given antibody 0.14 (antibody is nested in assay, and the assay is already the track index); read_length given donor 0.30; pipeline given lab 0.34 and given donor 0.22; run_type_pe given donor 0.37; lab given antibody 0.36; given cell: pipeline 0.25, read_length 0.31, run_type_pe 0.37. lab and platform: 0.57 (lab given platform) and 0.69 (platform given lab).

### 7.3 Steps and arms

0. **References (prerequisite).** The full-set runs are the recipe that mix-and-match selects, trained on `eic_v1_2`, 3 seeds, per model. Their 256-region scores (`p256_sv`) are the reference of every comparison. The 256-region references of `eic_v1_2` (4 baselines and the oracle floor under `eval/eic_v1_2/baselines`) do not exist yet on Fir or Nibi (checked 2026-11-17); only `baselines_p16` exists on Fir. Nibi has no `eval/eic_v1_2` at all.
1. **Screening (scoring only, no training).** For each of the 10 fields other than log2 depth, and each side (C' = `out`, C = `in`), score the full-set runs with that ONE field replaced: (a) foreign, i.e. the value of a donor experiment of the same assay in another cell (the E5 donor draw, seed 0), mode `<side>_keep_depth.<field>`; (b) optionally neutral, i.e. the column's fixed centre or id 0, mode `<side>_neutral.<field>`. This extends the whole-row foreign-covariate scoring of E5 (ABLATIONS_READ section 3.1) to one column; the code is `score.py` FOREIGN_COLUMNS_DEFINITION. log2 depth is not screened at scoring, because the batch depth also feeds the count transform and the NB offset.
2. **Confirmation, leave-one-field-out (LOFO).** One training per field (11 fields), 3 models, 3 seeds, with `covariate_drop_fields: [<field>]`: the field reads its neutral value in the two covariate encoders, in training and at scoring; the count transform and the NB depth offset keep the true depth.
3. **Control, grouped removal.** Six group arms (`tools/minicandi/covariate_field_arms.py` GROUPS): pipeline pair; lab + platform; donor + lab; the four view knobs (run_type_pe, read_length, dedup_on, mapq_threshold); all six categorical fields; all fields except log2 depth. The first four are the correlated groups that the rule reads (`CORRELATED_GROUPS`). The last two are reference arms: a useful "all but depth" arm says only that not every field can go at once, so the rule does not read them.
4. **Exclusion curve (rank-guided backward elimination).** Order the 11 fields by the LOFO score S (least useful first). Step k trains with the first k fields removed, k = 2 to 11 (k = 1 is the first LOFO arm): 10 arms. A full greedy search re-ranks at each step and needs 11 + 10 + ... + 2 = 65 arms (585 trainings and 256-region scorings); I chose the fixed order for cost. The curve then shows the hidden value of a correlated pair as a drop at the step that removes the second member.
5. **Read.** `tools/onion_eval/covariate_ranking.py` writes `ranked.tsv`, `effects.tsv`, `curve.tsv`, `curve.svg`, `order.txt` and `RANKING.md`.

### 7.4 Decision rule (stated before any run)

Rule terms, all on the 256-region scoring: imputation (variance recovered), gap closed on crop pairs with the oracle reference (composite version 3 term), calibration, and the correction-only term. The composite is reported beside them and is not a rule term. Per arm, model and term: delta = arm mean minus full-set mean over 3 seeds; R = the larger of the two seed ranges; z = -delta / R.

| outcome | conclusion |
|---|---|
| a cell with z > 1 on at least 2 of 3 models, or z > 2 on one model | the field is **useful**: keep |
| no cell with z > 1 on any model and term (S = max z <= 1), and every scored correlated group arm that holds the field is also S <= 1 | **safe to remove** |
| LOFO safe, but its correlated group arm useful | the members share the value: keep the member with the largest LOFO S (`shared: keep`); the others are `shared: removable if <keeper> kept`. The curve order puts the keeper last |
| LOFO safe, its group arm unclear | **unclear** (group): keep |
| LOFO safe, no group arm scored | `safe alone (group not scored)`: not yet safe to remove |
| fewer than 2 seeds, or a zero seed range, in a cell | the cell is undefined; the field is `undefined` and stays in the kept set |
| one model only (the cheaper plan, 7.7) | useful = z > 1 on that model; otherwise as above |
| z > 1 on exactly one model, below 2x | **unclear**: keep (the PI's default is every field); list it for the PI |
| screening beyond, LOFO within | the retrained model recovers the information from other fields; LOFO governs removal |
| imputation lower beyond for a single field | unexpected (E5: covariates carried no imputation information at test time); check the wiring before any conclusion |
| curve: steps 1 to k all within the seed range, step k + 1 beyond | the minimum set is the 11 fields minus the first k of the order |
| curve: step 1 already beyond | no field is safe to remove; the minimum set is all 11 |

Rank: by S, most useful first. Screening uses the same rule against the same full-set scores; it ranks, it never removes a field.

### 7.5 Prior leads and expectation

Leads (ABLATIONS_READ, foreman worktree `cruxvault/results/ablations_closeout/ABLATIONS_READ.md`; all on `eic_v1`, 16 pair regions):
- Whole-row foreign C' with true depth (section 3.1) lowered gap closed by 0.165 (conv autoencoder), 0.133 (conv + attention), 0.242 (U-Net), calibration by 0.186 to 0.213, and the correction-only term by 0.177 and 0.154 on the conv models; all beyond. Foreign C changed little (U-Net calibration -0.142 only). Imputation was within in every true-depth mode.
- In `rung2_base_heldout/seed0`, that swap changed `read_length` on 124 of 305 targets and `platform_id` on 231 (section 3.1). So the whole-row test could not separate processing knobs from batch fields.
- Numeric-only embedding (all six categorical fields off; section 2.7): within on the conv models; U-Net gap closed -0.133, beyond.
- Scrambled covariates in training (section 2.11): correction-only term lower beyond on all three models.
- Filling lab, antibody and donor (`eic_v1` to `eic_v1_2`, section 3.2): every term within the seed range on all three models.

Expectation: (1) imputation within for every field and group. (2) The view knobs (read_length, run_type_pe, dedup_on, mapq_threshold) rank highest, through gap closed and the correction-only term, because the counterfactual views vary exactly these knobs (`arm_knob` values se, crop, nodedup, mapq in `measurements.tsv`). (3) pipeline_version_id is safe to remove alone (it duplicates pipeline_id); the pipeline pair is safe or unclear. (4) lab, platform, antibody and donor are safe or unclear, singly and as the categorical group on the conv models; the U-Net may lose gap closed when the categorical group goes (numeric-only lead). (5) log2 depth in the encoders: unclear; depth still enters through the offset.

### 7.6 Controls and traps

- Correlated fields hide each other's value in LOFO (pipeline pair; lab with donor and platform; antibody with assay). Control: the group arms and the cumulative curve. There is no antibody + assay group arm: assay identity is not a covariate column. The assay is the track index (`contract.TRACK_INDEX`), i.e. which output channel a row belongs to, so no switch can remove it, and the tables hold no other assay-level column (`assay_class` is not encoded; `contract.META_COLUMNS`). The LOFO antibody arm therefore measures exactly antibody's information beyond the assay, which is the only part that can be removed.
- A same-assay foreign donor cannot change a field that is constant within the assay (pipeline varies in 8 of 36 assays, run_type_pe in 14). The scorer records, per column, how many measurements the swap changed (`foreign_covariates.n_changed_by_column`); a screen with few changes is blind, not null.
- Neutral id 0 is an embedding row that training never updates for a field filled in every training row (lab, donor, platform, pipeline). A neutral screen is therefore out of distribution; read it with that caveat. The LOFO arms do not have this problem: the neutral value is present from the first step of training.
- donor_id nearly fixes the cell (0.05). A donor field can let a model memorize cell identity; 6 validation rows and 0 blind rows hold an unseen donor.
- Antibody: 18 validation and 48 blind rows hold an antibody outside the vocabulary (id 0 at test time).
- Power: 3 seeds; calibration seed ranges are 0.108 to 0.178, so a calibration loss below about 0.11 to 0.18 cannot show. "Safe to remove" means no loss detectable at this power. 11 fields x 12 cells give many chances for one false "beyond"; the 2-model requirement limits this.
- Two-step recipes are supported. `registry.build_model` builds the inner rung without the wrapper keys, and both legs of `two_step.TwoStep` run that inner model, so the field reads its neutral value in leg 1 (C' = base covariates) and in leg 2 (test `test_two_step_drops_the_field_in_both_legs`).

### 7.7 Cost (CPU-hours; measured with sacct on Fir, read-only, 2026-11-17)

- Training per run, `eic_v1_2_rung{2,3,4}_base_heldout`, jobs 4472547 to 4472600 (12 CPUs each): conv autoencoder (19,740 + 17,424 + 27,468) s = 64,632 s; 64,632 / 3 / 3,600 = 5.98 h; conv + attention 154,920 / 3 / 3,600 = 14.34 h; U-Net 76,212 / 3 / 3,600 = 7.06 h. One arm (3 models x 3 seeds) = 3 x (5.98 + 14.34 + 7.06) = 82.2.
- 256-region scoring per run (8 predict jobs of 8 CPUs + 1 metrics job of 16 CPUs), the E1 scorings of `rung{2,3,4}_base_heldout` (jobs 4492698 and after): conv autoencoder 22.8, 28.5, 38.8 (mean 30.03); conv + attention 25.8, 23.1, 27.4 (mean 25.43); U-Net 42.2, 47.6, 70.5 (mean 53.43). These are the per-model rates per run-seed: conv autoencoder 30.0, conv + attention 25.4, U-Net 53.4 (the foreman's figure, about 52, is from another set of jobs; I use my measurement). One arm = 3 x (30.03 + 25.43 + 53.43) = 326.7, i.e. 36.3 per run on average. The brief's figure of about 23 per run is below this measurement.
- One training arm with its scoring, 3 models x 3 seeds = 82.2 + 326.7 = 408.9. Per model (3 seeds): conv autoencoder 3 x (5.98 + 30.03) = 108.1; conv + attention 3 x (14.34 + 25.43) = 119.3; U-Net 3 x (7.06 + 53.43) = 181.5.

| step | arms | trainings | 256-region scorings | CPU-hours |
|---|---|---|---|---|
| 0. reference scorings (full-set runs) | 1 | 0 | 9 | 326.7 |
| 1. screening, 10 fields x 2 sides, foreign | 20 | 0 | 180 | 20 x 326.7 = 6,534 |
| 1'. screening, C' side only | 10 | 0 | 90 | 3,267 |
| 1''. screening, C' side, seed 0 only | 10 | 0 | 30 | 10 x 108.9 = 1,089 |
| 2. LOFO, 11 fields | 11 | 99 | 99 | 11 x 408.9 = 4,498 |
| 3. groups | 6 | 54 | 54 | 6 x 408.9 = 2,453 |
| 4. curve, k = 2 to 11 | 10 | 90 | 90 | 10 x 408.9 = 4,089 |

Full plan (0 + 1 + 2 + 3 + 4) = 326.7 + 6,534 + 4,498 + 2,453 + 4,089 = 17,901. Without screening: 11,366. Not counted: the full-set trainings (mix-and-match item) and the 256-region references of `eic_v1_2` (the 4 baselines took 28.1 CPU-hours in total on 16 pair regions, Fir jobs 4463699 to 4463702). At 256 regions, a per-field screening on both sides (653 CPU-hours per field) costs more than a LOFO training with its scoring (409). My recommendation: run step 1'' (1,089) or no screening, then steps 2 to 4; the PI decides.

Cheaper plan, for the PI to choose: LOFO (11 arms) and groups (6 arms) on the final model only (the one mix-and-match selects; 1 model x 3 seeds), plus its 3 reference scorings, no screening:

| final model | reference (3 scorings) | LOFO + groups (17 arms) | total | with the curve (10 more arms) |
|---|---|---|---|---|
| conv autoencoder | 3 x 30.03 = 90.1 | 17 x 108.1 = 1,837 | 1,927 | + 10 x 108.1 = 3,008 |
| conv + attention | 3 x 25.43 = 76.3 | 17 x 119.3 = 2,029 | 2,105 | + 10 x 119.3 = 3,298 |
| U-Net | 3 x 53.43 = 160.3 | 17 x 181.5 = 3,085 | 3,245 | + 10 x 181.5 = 5,060 |

The submit script runs it with `RUNGS=<2|3|4>`. The ranking tool then applies the one-model rule (7.4). The cost of this plan: the result holds for one architecture only.

Storage: each scoring writes a prediction store of about 55 GB and 1,700 files on /scratch, removed by its metrics job. 180 scorings at once would hold about 306,000 files; Fir scratch was at 834K of 1,000K files. Submit in batches (`ARMS=`, `FIELDS=`, `SIDES=`, `SEEDS=` of the submit script).

### 7.8 Code and script

- Switch at training: `covariate_drop_fields` (`tools/minicandi/covariates.py` `drop_field_indices`, `drop_fields`, `model_covariates(..., drop=)`; read by all six rungs). Off (absent or empty): the rungs call `model_covariates` exactly as before.
- Switch at scoring: `score.py --foreign-covariates <side>_keep_depth.<col>` or `<side>_neutral.<col>` (env `ONION_FOREIGN_COVARIATES`; `submit_arms.sh FOREIGN_COVARIATES`).
- Configs: `tools/minicandi/covariate_field_arms.py {lofo,groups,curve} --base <rung2> --base <rung3> --base <rung4> --out <dir>`, for any base config. The kit folder `configs/ablations/covariate_fields` holds the LOFO and group arms of today's base recipes; regenerate it from the mix-and-match recipe.
- Read: `tools/onion_eval/covariate_ranking.py {runs,screen,tsv}`.
- Submit: `tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh` (`STEP=ref|screen|lofo|groups|curve`, dry run by default; it prints the job count and the CPU-hours).
