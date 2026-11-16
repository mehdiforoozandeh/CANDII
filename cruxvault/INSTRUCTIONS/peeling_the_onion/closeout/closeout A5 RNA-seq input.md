---
title: "A5: RNA-seq input"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A5: RNA-seq input

## 1. What the PI asked

- `long-horizon notepad.md`:393 (PI note 4a, 2026-11-12): "(a) Bulk RNA-seq as an extra input, in two forms: RNA-seq aligned to the genome, i.e. one more count track; and the expression vector (log TPM) of the cell as a new input modality, which defines the cell state and the trans-regulatory effects (as in corgi)."
- `long-horizon notepad.md`:399 (ruling 27): "Both bulk RNA-seq forms (aligned track, log-TPM vector) run as ablation arms and are reported in a side table. The final EIC-comparison model does not use RNA-seq, so D8 holds."

Success criterion served: 4 (design tested). Audit row: D23. The arm never joins the final design (ruling 27).

## 2. Question

Does bulk RNA-seq of the target cell, as a stranded track or as a 2,000-gene log-TPM vector, raise imputation (variance recovered) over the reference on the cells that have RNA-seq?

## 3. State today

- Inputs built and checked (`results/t245/RNA_INPUTS.md`): 39 cells (31 same biosample, 8 other donor); expression table 39 cells x 19,909 protein-coding genes, 2,000 chosen by variance, mitochondrial genes excluded; per-cell tracks `/scratch/mforooz/onion/aux/rna/tracks/<cell>.h5` (Nibi), plus and minus strands at 25 bp; final check job 26973982 exited 0. 12 census cells have no RNA-seq input. The index lists "RNA fetch" as a dependency; that fetch is done.
- Arm keys exist: `tools/minicandi/stage3_arms.py` arms `rna_track`, `rna_vector`; `evalset-subset` keeps the targets and pairs of the RNA cells.
- No arm trained (`COVERAGE.md` row 32). The inputs are on Nibi only; the reference runs and the ablation kit are on Fir.

## 4. Prior leads

- `long-horizon notepad.md`:399: 40 of 51 EIC cell types have ENCODE bulk RNA-seq; 5 of the 12 blind cell types have it.
- `results/t305/LAYERS_REAL.md`: the cell-type layer is 10% to 23% of the three layers per assay. Prediction: an input that describes cell state can only act on that share; the sampling layer is out of reach.
- In EIC every validation cell has other assays as inputs, so the cell state is already partly visible. Prediction: small gain, larger for cells with few input assays.
- No other result found (searched `COVERAGE.md`, `results/`, the vault questions).

## 5. Work

1. Build the RNA-cell evaluation subset: `stage3_arms.py evalset-subset --evalset <p16 set> --cells rna_cells.tsv`. Count validation targets and cells in it.
2. Move the 39 track files and the expression table to Fir (rsync on the PI's OK, or Globus), or run the arm and its reference on Nibi. Both arms and the reference run on one cluster and one device type.
3. Write `rna_track` and `rna_vector` configs on `final/rung{2,3,4}/reference_equal.json` with `stage3_arms.py write`.
4. Train 2 arms x 3 architectures x 3 seeds = 18 runs. Score them and the 9 reference runs on the RNA subset.
5. Report in a side table (ruling 27); record a finding.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds), on the RNA subset.

| outcome | conclusion |
|---|---|
| an RNA arm beyond the reference on variance recovered, 2+ architectures | cell-state input carries information the other assays lack; a candidate for the scaling phase (not for the EIC model, ruling 27) |
| within the range | at this scale the other assays already carry the cell state |
| the track helps and the vector does not (or the reverse) | name the form that carries the signal |

## 7. Dependencies, cost, risks

- Depends on the data move (step 2).
- Cost: 2 x 262 = 524 CPU-h for training and scoring (`ablation_specs/1_control_input.md` section 7), plus the reference rescored on the subset (9 scorings).
- Risk 1: 8 cells use another donor's RNA-seq; report them apart.
- Risk 2: the vector is fixed per cell, so the model can use it as a cell identifier; compare with A6 (learned cell embedding) to separate cell state from cell identity.

## 8. Done when

The 18 arm runs and the reference are scored on the RNA subset and reported in a side table, with the finding recorded in the crux tree.
