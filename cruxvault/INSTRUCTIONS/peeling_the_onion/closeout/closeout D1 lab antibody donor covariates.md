---
title: "D1: fill the lab, antibody and donor covariates"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# D1: fill the lab, antibody and donor covariates

## 1. What the PI asked

- `long-horizon decisions.md`:51: "**Arm "metadata only" (default).** C and C' hold recorded metadata and processing settings only, e.g. depth, run type, read length, lab, platform, pipeline and version, duplicate removal, MAPQ filter, antibody, donor."
- `long-horizon decisions.md`:28: "The agent records donor identity as a covariate wherever a source gives it."
- `peeling the onion.md`:14-15: "e.g. lab, sequencing technology, antigen, etc." and "these are also the type of artifact/variations that we want to be able to model away from the data and make it robust to it (and measure and show that robustness)".
- `peeling the onion.md`:61: "same for the amount of variance caused by different artifact sources: depth, sequencing platform, antigen efficiency, etc."

Success criterion served: 1 ("have the new augmented EIC data + counterfactuals etc."). It also feeds criterion 4, because every model so far trained without these covariates.

Audit rows: A1, A2, B6.

## 2. Question

Fill lab, antibody and donor for every row of the EIC tables, and use one platform vocabulary for all splits. Then measure, on three architectures, whether the filled covariates change the three goals.

## 3. State today

- Nibi `tables/eic_v1/measurements.tsv` (2,740 rows): `lab` is filled on 78 rows (all 78 are blind testbed rows); `antibody` on 0 rows; `donor` on 0 rows. `covariates.json`: `antibody: []`, `donor: []`, `lab` has 2 values (audit row A1, read on Nibi 2026-11-16).
- `results/t183/imbalance.tsv`: lab is empty for all 249 training and 42 validation tracks.
- Platform: training and validation rows hold OBI codes; the blind rows hold the string "Illumina HiSeq 4000". So every blind platform id is unseen in training (audit row A1).
- Sources exist on Nibi (read 2026-11-16): `ONION/catalog/encode/experiments.tsv` (8,494 rows, column `lab`), `libraries.tsv` (columns `donor`, `antibody`, per library), `donors.tsv` (988 donors), `files.tsv` (column `platform`).

## 4. Prior leads

- `onion paper.md`:36: the blind set is "100% paired-end, 100% HiSeq 4000 and 94% from one lab", and the designed views "cannot imitate platform, lab or the reverse directions" (source `results/t170/withheld_design.md`). Prediction: lab and platform are the measurement axes on which the blind set differs most from training, so a model that cannot read them cannot correct for them.
- `t264/PAIR_SMOKE.md`: the two other-lab smoke pairs had total ratios 1.678 and 1.383, outside 1 +- 0.05; the same-lab pair was 0.985. Prediction: a lab effect exists in the counts, at least in level.
- No model result with lab, antibody or donor exists (searched `results/`, `FINAL_ROUND_EQUAL.md`, `COVERAGE.md`).

## 5. Work

1. Join, per measurement row: `lab` from `experiments.tsv` on the experiment accession; `donor` and `antibody` (ENCODE antibody accession) from `libraries.tsv` on the libraries of the row's unit (pooled rows take the set of their libraries; one value if all agree, else a "mixed" token).
2. Map `platform` for all 2,740 rows from `files.tsv` to one vocabulary (OBI code), so the blind rows use the training codes.
3. Write the columns into a side table and a new tables version. Coordinate with D3: one version build carries both the D1 columns and the D3 tail rows, so the models retrain once. Run `check_dataset.py` (every leak rule at 0).
4. Encode each new covariate as a categorical id with an "unknown" token. Count, per split, how many validation and blind ids are unseen in training.
5. Ablation on conv autoencoder, conv + attention and U-Net, 3 seeds each, on Fir CPU: arm "metadata with lab, antibody, donor" against the reference recipe on the new version. Score into `p16_sv`, composite version 3.
6. Record a finding in the crux tree; log the D13 outcome in the notepad.

## 6. How the result answers the question

Build checks:
- `check_dataset.py --tables <new version>` exits 0 with every leak count 0.
- A count over the new `measurements.tsv` shows `lab`, `antibody` (ChIP-seq rows) and `donor` non-empty on at least 95% of training and validation rows; every missing value is listed with its cause.
- The platform vocabulary of `covariates.json` holds no id that occurs only in the blind rows.

Ablation ("beyond" = difference of seed means larger than the larger seed range, max minus min over 3 seeds; D13 bar = 2 x that range):

| outcome | conclusion |
|---|---|
| new covariates beyond the reference on composite or correction, on 2+ architectures | the covariates carry measurement information; the final design keeps them |
| within the range on all three | no evidence at this scale; by D13 the simpler arm stays, but D6 makes these columns part of the default vector, so the PI decides |
| beyond, worse, on 2+ architectures | the ids overfit (for example, lab identifies the cell); inspect the confounding before any design change |

## 7. Dependencies, cost, risks

- Depends on nothing. D2, D3 and D4 depend on it.
- Cost: table build about 7.5 CPU-h allocated (the `eic_v1` build ran 14:32 to 14:46 UTC on 32 CPUs, `t183/DATASET_EIC_V1.md` L13: 14 min x 32 = 7.5 CPU-h; the audit quotes 4.32 CPU-h). Ablation: 262 CPU-h for 9 runs (per-seed cost 32.5 / 22.4 / 32.3 CPU-h, `run_state/ablation_specs/1_control_input.md` section 7). Join and checks: about 1 agent-day.
- Risk 1: in EIC, lab and donor are nearly fixed per cell, so these ids can stand in for cell identity. That can raise imputation without any correction gain. Report the per-goal terms, not the composite only.
- Risk 2: a new tables version moves the evaluation-set hash. By the rule after ruling 38 (notepad L465), comparisons stay within one version; the reference must be retrained on the new version.
- Risk 3: the update-version check of Amendment 6 (`check_dataset.py --superset-of`, plan L45) requires "rows of the older tables changed or missing: 0". D1 changes covariate cells of existing rows, so that check fails by design. The version needs a check that allows changes in the named covariate columns only.
- Risk 4: the scrambled-covariates control (A2) must be rerun on the new vector, or its meaning changes.

## 8. Done when

The new tables version passes `check_dataset.py` with lab, antibody, donor and a unified platform filled as in section 6, and the 9-run ablation is scored and logged in the notepad with its D13 outcome.
