---
title: "E4: complete the replicate ladder and the crossed mixed model"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# E4: complete the replicate ladder and the crossed mixed model

## 1. What the PI asked

- `peeling the onion.md`:57: "this means that we want our evaluation protocol to be basically an accurate and fair decomposition of how much variance there is and peeling the onion i.e. dividing the total variance by its different components and finally what can't be modelled."
- `long-horizon decisions.md`:85 (D12): "**Primary.** Nested differences along the replicate ladder, in counts: pseudoreplicate, sequencing, technical, isogenic, anisogenic, other experiment (other lab), other cell type."
- `long-horizon decisions.md`:86: "**Supplementary.** A crossed linear mixed model (e.g. cell type, lab, donor, depth) gives order-free variance fractions as a cross-check."
- `long-horizon decisions.md`:87: "The agent reports the pair count and the interval width per rung."
- `long-horizon notepad.md`:418 (ruling 30): "The donor layer is anisogenic minus isogenic. The experiment (lab) layer is other-experiment minus isogenic. The cell-type layer is taken over a pair that shares only the cell type."

Success criterion served: 3 (evals according to the onion goals). Audit rows: B1, B2, B4, and B6 (platform and antigen parts).

## 2. Question

Fill the empty rungs of the replicate ladder (sequencing, technical, anisogenic, other experiment) as far as EIC data allows, report each layer with pair count and interval width, and fit the crossed mixed model on real data as the order-free cross-check.

## 3. State today

- `results/t305/LAYERS_REAL.md`: three layers measured on `eic_v1` (3,908 pairs: 778 pseudoreplicate, 219 isogenic, 2,911 other cell type). "Sequencing, technical, anisogenic (dropped) and other_experiment hold no pair." The replicate layer pools sequencing, library and biological replicate; the cell-type layer includes donor and lab.
- Anisogenic: 4 pairs, all in cell C09, were dropped, because one pair from one cell gives no increment (same file).
- Sequencing and technical rungs need the per-library (54 rows) and per-lane (459 rows) units of build priority 4 (`results/t170/build_plan.md` L112-113); not built (audit row A6).
- Other-experiment rung: D2.
- Mixed model: code merged (`tools/onion_eval/mixed_model.py`; Log 2026-11-11 MM, fixed by TVf); simulation audit only (`results/t233`, `t234`); no real-data fit (audit row B4).

## 4. Prior leads

- `LAYERS_REAL.md` result 2: the replicate layer above sampling is 0.1% to 6% of the three layers. Prediction: sequencing and technical rungs split a small layer; their intervals may not exclude 0.
- `long-horizon notepad.md`:418: in simulation the nested order gave a negative experiment layer (-0.036 against a planted 0.081) because ENCODE crosses donor and experiment. Prediction: on real data the mixed model and the nested layers disagree where donor and lab are crossed.
- `results/t264/PAIR_SMOKE.md`: other-lab totals differ by 38% to 68% after depth matching. Prediction: the experiment layer is large in level; its 25 bp shape share is unknown.

## 5. Work

1. Rung data: take the other-experiment pairs from `eic_v1x` (D2). If the PI approves the full priority-4 build (decision in D4), add the per-library and per-lane units to a tables version `eic_v2` and build technical and sequencing pairs.
2. Anisogenic rung: count anisogenic pairs per assay in the new tables, using the donor column of D1. Report the rung pooled by assay class with the "few pairs" flag where under 5 pairs from 3 cells (scorable-set rule, notepad L51).
3. Rerun `layers.py run` on the new pair list (32,000 MB, per the t305 memory note), per assay, per stratum and per Haar band. Use the sibling-layer rule of ruling 30.
4. Fit `mixed_model.py` on training and validation tracks with factors cell type, lab, donor and depth (depth thinned to one level per assay, notepad L421). Report variance fractions with intervals.
5. Write `LADDER_COMPLETE.md`: one table per assay class with each rung's increment, pairs, cells and interval width, and the mixed-model fractions beside the nested layers.

## 6. How the result answers the question

Acceptance checks:
- `LADDER_COMPLETE.md` lists all seven rungs of D12; an empty rung states the reason (no pair in EIC, or build not approved).
- `telescoping_check` of `layers.py` passes on the new table.

| outcome | conclusion |
|---|---|
| nested layers and mixed-model fractions agree within intervals | the onion decomposition is robust to the order of the ladder |
| they disagree on the donor or lab layer | the crossed design biases the nested order; quote the mixed model for those layers |
| sequencing and technical increments include 0 | at 25 bp these rungs are indistinguishable from sampling on EIC |

## 7. Dependencies, cost, risks

- Depends on D2 (other-experiment pairs) and D1 (donor, lab columns). The sequencing and technical rungs depend on the PI decision on the priority-4 build (D4).
- Cost: the layers rerun took 1 h 19 min on 4 CPUs at 23.4 GiB (`LAYERS_REAL.md`), about 5 CPU-h; mixed model not estimated. Per-library and per-lane builds: a part of the 14,920 CPU-h priority-4 projection (`.orchestrate/plan.md` L1935). About 2 agent-days.
- Risk: EIC has one experiment per cell and assay; some rungs stay empty whatever is built. State this as a limit of EIC, not of the method.

## 8. Done when

`LADDER_COMPLETE.md` reports every D12 rung (value with interval, or the reason it is empty) and the real-data mixed-model fractions beside the nested layers.
