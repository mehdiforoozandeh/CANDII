---
title: "D2: second experiments: table, evaluation set, cross-lab pairs"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# D2: second experiments: table, evaluation set, cross-lab pairs

## 1. What the PI asked

- `long-horizon notepad.md`:423 (ruling 33): "One second experiment per track is reprocessed with the uniform pipeline, preferring another lab. These views enter the evaluation set only and are never used in training, so D8 holds."
- `long-horizon decisions.md`:56 (D7): "Real cross-lab and replicate pairs are also scored."
- `long-horizon decisions.md`:85 (D12): "Nested differences along the replicate ladder, in counts: pseudoreplicate, sequencing, technical, isogenic, anisogenic, other experiment (other lab), other cell type."
- `long-horizon notepad.md`:419 (ruling 31): "A model passes when the ratio is at least 1 at the pseudoreplicate rung and at most 1 at the rung that shares only what the model knows (another experiment of the same cell type, for imputation)."

Success criteria served: 1 (augmented EIC data) and 3 (tested evals). Audit rows: A7, B10, B22, and the other-experiment rung of B1.

## 2. Question

Finish the second-experiment build, put the views into the add-on tables and the add-on evaluation set, and score real cross-lab pairs for correction and the upper bound of the variance-ratio metric (predicted NB variance over replicate variance), for the baselines and the final candidate models.

## 3. State today

- Selection: 88 second experiments (37 other-lab, 51 same-lab; validation 13, of which 7 other-lab; training 75, of which 30 other-lab) (plan Log 2026-11-12, OXs retry 1).
- Build: 75 of 88 rows submitted on Nibi (`results/t264/SUBMITTED.md` L31); 13 DNase-seq rows wait for their EIC partner in the DNase-seq tail (table in the same file). The audit read 63 of 64 tasks of one array completed (Log 2026-11-14 02:36).
- Not done: the collect (chunk OXc), gate GX, the add-on tables `eic_v1x` (OXx), the add-on evaluation set and the experiment layer (OXy), and any model score on the add-on set. Nibi `tables/` holds only `eic_v1` (read 2026-11-16).
- Code is merged: dataset guard (training refused on add-on tables), add-on tables, add-on evaluation set and scoring, layers by pair class, metric 7 per rung with two bounds (Log 2026-11-12, OXd, OXt, OXe, OXl, OXr).

## 4. Prior leads

- `results/t264/PAIR_SMOKE.md`: the two other-lab pairs differ by 68% and 38% in regional totals after depth matching (ratios 1.6779, 1.3826); the same-lab ATAC-seq pair differs by 1.5%. A pair outside 1 +- 0.05 gives an undefined metric-7 ratio. Prediction: many other-lab pairs will be undefined for metric 7, and gate GX clause 3 (more than half undefined) may trip.
- `results/t305/LAYERS_REAL.md`: the cell-type layer today "holds cell type mixed with donor and experiment (lab)" because the other-experiment rung is empty. Prediction: the experiment layer will take a share out of that 10% to 23% cell-type layer.
- `onion paper.md`:111: the metric-7 upper bound "needs a second experiment of the same cell, which eic_v1 lacks".

## 5. Work

Follow the plan chunks of Amendment 4 (`.orchestrate/plan.md` L460-480); do not redesign them.
1. OXc collect: `build_list.py pending --priority 5` (kit KX4) on Nibi; resubmit each failed row once; replace twice-failed rows by their priority-6 fallback. Submit the 13 held rows when D3 verifies their partners, or list them as not built.
2. Gate GX: count built pairs per pair class and assay class; check `MIN_PAIRS = 5` from `MIN_CELLS = 3` (`layers.py:77`); count undefined pairs under the 5% rule.
3. OXx: `build_tables.py addon` to `eic_v1x` on top of the tables version of D1 and D3; `check_dataset.py --addon-of` with every leak count 0.
4. OXy: add-on evaluation set; experiment layer per pair class; baselines (marginal NB, position average, depth-only scaling, pseudoreplicate floor, retrieval) scored with `--addon`.
5. Models: score the reference and the final candidate of each of conv autoencoder, conv + attention and U-Net, 3 seeds, with `score.sh ... --addon`, on Fir CPU. Report the cross-lab correction gap closed and metric 7 at both bounds.

## 6. How the result answers the question

Build checks:
- `check_dataset.py --tables eic_v1x --addon-of <base version>` prints `evaluation-only rows readable in a training mode: 0` and `second-experiment rows: N` with N > 0.
- `OTHER_EXPERIMENT.md` gives the experiment layer per assay class with pair count, cell count and interval.

Results:

| outcome | conclusion |
|---|---|
| other-lab experiment layer with an interval that excludes 0 in 2+ assay classes | lab is a real variance source on EIC; the cell-type layer of t305 was inflated by it |
| model cross-lab gap closed above depth-only scaling beyond the seed range | the model corrects some lab effect, at least in level |
| model gap closed within the range of depth-only scaling | no lab correction; D1 covariates did not help on real pairs |
| more than half the validation pairs undefined for metric 7 | the upper bound is not quoted (gate GX clause 3); report the count |

## 7. Dependencies, cost, risks

- Depends on D1 (lab covariate in C and C'), and on D3 for the 13 held DNase-seq partners.
- Cost: Amendment 4 totals about 2,650 CPU-h, most of it the build already run (Log 2026-11-12). Add-on score: at most 6.6 CPU-h per run (plan L383); 18 model runs = at most 119 CPU-h; layers about 50 CPU-h.
- Risk 1: EIC cells with a second experiment from another lab are few (37 other-lab rows in all, 7 in validation). The lab layer may be estimable for one assay class only.
- Risk 2: an other-lab pair changes lab, donor and antibody together in some rows (pair class `experiment;lab;donor`). Report per pair class.

## 8. Done when

`eic_v1x` and its add-on evaluation set exist and pass their checks, `OTHER_EXPERIMENT.md` reports the experiment layer, and the cross-lab correction and metric-7 bounds are reported for the baselines and 3 architectures x 3 seeds.
