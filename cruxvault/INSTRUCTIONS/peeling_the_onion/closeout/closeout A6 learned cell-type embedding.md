---
title: "A6: learned cell-type embedding"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A6: learned cell-type embedding

## 1. What the PI asked

- `long-horizon notepad.md`:393 (PI note 4, 2026-11-12): "(c) A learned cell-type representation."
- `long-horizon notepad.md`:407 (agent call under the PI notes): "RNA-seq arms and the cell-embedding arm never join it (ruling 27; an embedding cannot serve an unseen cell type)."

Success criterion served: 4 (design tested). Audit row: D24.

## 2. Question

Does a learned per-cell embedding, added as an input, raise imputation over the reference? The answer measures how much cell identity the input assays leave unexplained. It does not change the design, because an embedding cannot serve a new cell type.

## 3. State today

- Arm key exists: `tools/minicandi/stage3_arms.py` arm `cell_embedding` (needs `n_cells`, the number of training cells, from `extra_data.cell_index`); `joins_design = no`.
- No arm trained (`COVERAGE.md` row 32; audit row D24).
- In the EIC split every validation target's cell also has training tracks (`dataset.py` docstring: validation inputs are "the same cell's training tracks"), so each validation cell has a trained embedding.

## 4. Prior leads

- None found for a cell embedding (searched `COVERAGE.md`, `results/`, `FINAL_ROUND_EQUAL.md`, vault questions q1 to q47).
- Related: U-Net scrambled covariates lowered variance recovered by 0.029, beyond the range (`COVERAGE.md` row 11). Prediction: the model already uses per-track identity from C; a cell embedding adds identity per cell, so the gain, if any, is on imputation.
- Risk of a shortcut: with one experiment per cell and assay in EIC, a cell id plus an assay id names one track. Prediction: the arm can memorize training tracks; that helps validation only through shared cell-level structure.

## 5. Work

Sections 5 and 7 are the plan of 2026-11-16. Section 9 replaces them (base recipes, tables eic_v1_2, 256 regions).

1. Write `cell_embedding` configs on `final/rung{2,3,4}/reference_equal.json` with `stage3_arms.py write --tables <tables>` (fills `n_cells`).
2. Check the parameter count; the embedding table adds n_cells x width parameters. If the total moves beyond 10% of 1.0e6, note it.
3. Train 3 architectures x 3 seeds = 9 runs on Fir CPU; score into `p16_sv`.
4. Read against the reference; report next to the RNA-seq arms of A5 in one side table.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds).

| outcome | conclusion |
|---|---|
| embedding beyond the reference on variance recovered, 2+ architectures | cell identity carries information that the input assays do not; motivates cell-state inputs (A5) for the scaling phase |
| within the range | the input assays already describe the cell at this scale |
| embedding worse beyond the range | the embedding overfits training tracks; report it as a warning for any identity input |

## 7. Dependencies, cost, risks

- Depends on nothing.
- Cost: 262 CPU-h (`ablation_specs/1_control_input.md` section 7).
- Risk: the embedding never enters the design (notepad L407); report it as an analysis, not as a selection.

## 8. Done when

The 9 runs are scored and reported beside the RNA-seq arms, with the finding recorded in the crux tree.

## 9. Specification as worked (2026-11-17, overnight)

### 9.1 Question

Does a learned per-cell embedding, added to a model that already reads the cell's other assays and its covariates, raise imputation (variance recovered) above the base recipe? The arm is an analysis only. It never joins the design, because a new cell type has no trained row (`stage3_arms.py`:75-79, `NEVER_IN_DESIGN`).

### 9.2 What cell-type information the model sees today (checked in code)

1. The other assays of the same cell. A training window holds one cell and only that cell's training tracks (`dataset.py`:23). A validation item reads "the same cell's training tracks" as inputs (`dataset.py`:35). Measured on `eval/eic_v1/evalset.json` (Nibi): 609 inputs of the 63 imputation targets are all `in_eic = train` rows.
2. The covariates C and C'. The categorical columns include `donor_id` (`covariates.py`:50-52). In eic_v1 no row has a donor (0 of 2,740 filled). In eic_v1_2 every row has one (2,740 of 2,740), with 49 values over 51 training cells. A donor names its cell alone for 33 of 51 training cells, and for 12 of the 25 cells with an imputation target (computed from `tables/eic_v1_2/measurements.tsv`, Nibi). So on eic_v1_2 the base recipe already holds a near cell identity for about half of the scored cells.
3. Nothing else. No rung reads `meta` or a cell name (grep of `rungs/conv_ae.py`, `rungs/conv_attn_ae.py`, `rungs/unet.py`). The key `cell_id` exists only under `cell_embedding` (`dataset.py`:579-584 and 598-599).

### 9.3 Arm

One arm, `cell_embedding`, on the three base recipes (`configs/rung{2,3,4}/base_heldout.json`, identical key for key to `final/rung{r}/reference_equal.json`). Changed keys: `cell_embedding = true`, `cell_state_dropout = 0.1`, `n_cells = 51`. Configs: `tools/minicandi/configs/ablations/cell_embedding/rung{2,3,4}/cell_embedding.json`, written by `stage3_arms.plan`. Mechanism (`extra_inputs.py`:78-136, `contract.py`:194-262): a 64-wide code from an `nn.Embedding(52, 64)` goes through `G` to a FiLM (feature-wise scale and shift) of the covariate embedding on the input and the output side. It starts as the identity (gamma 1, beta 0).

Parameters (`registry.build_model`, laptop): conv autoencoder 1,042,514 -> 1,058,322 (+5.8% from 1.0e6); conv + attention 1,038,018 -> 1,053,826 (+5.4%); U-Net 993,599 -> 1,009,407 (+0.9%). The arm adds 15,808 parameters on each (embedding 52 x 64 = 3,328; `G` 12,480). All are within 10% of 1.0e6, so the widths stay as in the base.

Runs: 3 architectures x 3 seeds = 9 runs, 80,000 windows, tables eic_v1_2, scored on all 256 regions (`eval/eic_v1_2/evalset.json`) into `p256_sv` with the same-view pass. Reference: `eic_v1_2_rung{2,3,4}_base_heldout` (Fir `runs_fir`, 3 seeds each, trained 2026-11-16), which must also be scored into `p256_sv`.

### 9.4 The zero-shot trap and its controls

- Row numbering: `extra_data.cell_index` numbers the cells of the `in_eic = train` rows from 1 (`extra_data.py`:179-187). Any other cell reads row 0 (unknown). The vocabulary is built from training rows only, so the eic_v1 defect (category lists that held validation and blind values, and so had embedding rows that no training row reached) cannot recur through the index.
- Row 0 is trained in two ways: the per-sample dropout 0.1 sends a known cell to row 0 (`extra_inputs.py`:130), and the sampler draws the shared-control pseudo-cell "" (124 control rows, 1 of 52 cell groups; `dataset.py`:616-624 and 867), whose id is 0.
- Held-out (validation) cells: all 25 cells with a scored item have a training track and so a trained row (ids 2 to 48). A validation track never trains its own cell's row, because mode `train` never reads a validation row (`dataset.py`, role table at line 13).
- Blind cells: the 4 cells with blind rows all have training rows (eic_v1_2, Nibi). This item scores no blind track.
- A cell type with no training row: it reads row 0. The arm then gives the model as trained with 10% "unknown" samples. That is why the arm never joins the design.
- Control 1 (before training; done 2026-11-17 on Nibi): `cell_embedding_audit.py tables` on eic_v1_2 with the eval-set items (identical in eic_v1 and eic_v1_2; only `tables`, `tables_dir`, `tables_sha256` differ, checked on Fir). Result: PASS served (25 of 25 scored cells have an id > 0), PASS no leak (no scored imputation, control or correction measurement is a training row). Evidence: `cruxvault/results/closeout_A6/tables_audit_eic_v1_2.json`.
- Control 2 (after training, one CPU job): `cell_embedding_audit.py runs` on the 9 runs. It checks `n_cells` = 51 and the table hash, and that each row read by a scored cell, and row 0, left the direction of its initial value (rebuilt with the run's seed). AdamW weight decay changes an unused row only by a positive factor, so "changed" is not enough; the share of the row orthogonal to its initial value must be at least 1e-3. A unit test trains a model in which one row gets gradient: that row passes, and every other row has a share below 1e-6 although weight decay changed it.
- Built-in refusals: the dataset refuses `n_cells` that differs from the index (`dataset.py`:579-584), and the scorer refuses a run whose tables hash differs from the eval set (`predict.py`:193-197).

### 9.5 Prior leads (numbers read from files)

- Filling donor, lab, antibody and platform (eic_v1 -> eic_v1_2) left variance recovered within the seed range on all three: conv autoencoder 0.382 (range 0.028) -> 0.370 (0.021); conv + attention 0.387 (0.012) -> 0.386 (0.030); U-Net 0.432 (0.023) -> 0.445 (0.038). Source: foreman worktree `cruxvault/results/ablations_closeout/ABLATIONS_READ.md` section 3.2 (p16_sv). The donor column is a near cell identity on eic_v1_2 (9.2), so a near cell identity already gave no gain beyond the range.
- Foreign covariates at scoring (E5, same file): imputation within the range in every `keep_depth` mode on all three; "the covariates carry no imputation information at test time".
- Scrambled covariates, U-Net, older composite: variance recovered -0.029, beyond (`results/ladder_audit/COVERAGE.md` row 11). Covariates carried some imputation information then.

### 9.6 Expectation (stated before any run)

Variance recovered within the seed range on all three architectures. The model sees the cell's other assays, and on eic_v1_2 a donor that names half of the scored cells. If a gain appears, it is largest on the U-Net, the architecture whose covariate use showed in the older control. Correction terms within the range: a per-cell code carries no information about the requested view.

### 9.7 Decision table

"Beyond" = difference of seed means larger than the larger seed range (max minus min over 3 seeds). Every reading waits for Control 2 to pass on all 9 runs.

| outcome | conclusion |
|---|---|
| Control 2 fails on any run | the arm is void for that run; find the cause (untrained row, wrong `n_cells`, wrong tables); no reading |
| variance recovered beyond the reference on 2+ architectures | cell identity carries information that the input assays and the donor do not; motivates cell-state inputs (A5) for the scaling phase; then score the trained arm with every `cell_id` set to 0 to measure what a new cell type loses (needs a scorer switch; not built) |
| within the range on all three, or beyond on 1 only | the input assays and covariates already describe the cell at this scale |
| variance recovered lower, beyond, on 2+ | the embedding overfits training tracks; report it as a warning for any identity input |
| correction terms move beyond while imputation is within | a per-cell code changes the processing correction; check for a wiring fault first |

Report the four metrics, the composite and the correction-only term beside the RNA-seq arms of A5 in one side table, as analysis.

### 9.8 Cost (CPU-hours)

- Training: the eic_v1_2 base runs on Fir used (sacct `CPUTimeRAW`, 12 CPUs) conv autoencoder 19,740 + 17,424 + 27,468 s; conv + attention 57,792 + 62,256 + 34,872 s; U-Net 28,392 + 23,508 + 24,312 s. Sum 295,764 s = 82.2 CPU-h. The arm adds 1.5% parameters, so training is about 82 CPU-h.
- Scoring on 256 regions: 9 x 23 CPU-h = 207 CPU-h (overnight addendum figure).
- Audit: 1 job x 2 CPUs x at most 0.5 h = 1 CPU-h.
- Total for the arm: 82 + 207 + 1 = 290 CPU-h; 90 jobs (9 training + 72 predict + 9 metrics) and 1 audit job.
- Prerequisites, not in this total: the reference scoring of the 9 eic_v1_2 base runs into `p256_sv` (9 x 23 = 207 CPU-h) and the 256-region baselines of eic_v1_2 (`eval/eic_v1_2/baselines`, 4 baselines and `oracle_floor`), which do not exist on either cluster today.

### 9.9 Submission

`tools/minicandi/configs/ablations/cell_embedding/submit_cell_embedding.sh` (dry run by default; `MODE=train` or `MODE=audit`; `CLUSTER=fir|nibi`). Run it on Fir, where the reference runs are. The dry run on Nibi (kit COA6) found n_cells = 51 in all 3 configs and printed the 9 training lines; it listed the missing eic_v1_2 eval set and baselines on Nibi and the 9 missing reference scorings.
