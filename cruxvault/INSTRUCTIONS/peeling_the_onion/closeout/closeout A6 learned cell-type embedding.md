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
