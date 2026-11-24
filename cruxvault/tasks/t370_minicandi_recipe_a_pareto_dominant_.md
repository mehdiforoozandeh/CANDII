---
id: t370
type: task
title: miniCANDI recipe: a Pareto-dominant recipe on imputation, correction and calibration (conditioned skips, pre/post CAC layers, FiLM placement, control prediction, EIC comparison)
category: implementation
parent: 
blocked_by: None
refs: q75, h44, h45, h46, q76, q73
hypothesis_refs: 
status: open
created: 2026-11-21T04:04:24
updated: 2026-11-21T04:04:24
---

# t370 — miniCANDI recipe: a Pareto-dominant recipe on imputation, correction and calibration (conditioned skips, pre/post CAC layers, FiLM placement, control prediction, EIC comparison)

Refs:: [[q75_can_one_minicandi_recipe_match_or_beat_t\|q75]], [[h44_a_u_net_whose_count_skips_are_film_modul\|h44]], [[h45_pre_and_post_cac_layers_in_the_testbed_g\|h45]], [[h46_where_film_injects_the_covariates_encode\|h46]], [[q76_can_minicandi_predict_the_chip_seq_contr\|q76]], [[q73_how_does_the_final_minicandi_score_on_th\|q73]]

## Why

settles the architecture for scaling (q72) and unblocks t369

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

Final architecture, picked by the PI on 2026-11-23 (relayed by the foreman session): the U-Net with
read-disjoint training pairs, average-signal input and Pearson-residual input, the head skip kept,
160,000 training windows (twice the screen budget), ChIP-control loss weight 1, and per-assay
dispersion recalibration after training.

256-region confirm, 3 seeds, eic_v1_2 (`presentations/t370-minicandi-recipe/data/pareto_p256_sv.tsv`,
from `tools/onion_eval/pareto_select.py`; mean, range over seeds):

| arm | imputation | correction | calibration | class |
|---|---|---|---|---|
| final (2x windows) | 0.503 (0.010) | 0.818 (0.016) | 0.717 (0.078) | PASS |
| same at 1x windows | 0.500 (0.035) | 0.818 (0.014) | 0.685 (0.092) | PASS |
| pairs + Pearson only | 0.445 (0.064) | 0.774 (0.031) | 0.625 (0.144) | PASS |
| best base model per term | 0.445 | 0.675 | 0.646 | bar |

Imputation and correction clear the bars by more than the seed range; calibration clears the bar
in the mean (0.717 vs 0.646) but by less than its seed range (0.078). Per-assay recalibration raised
calibration from 0.625 to 0.749 (one seed) on pairs + Pearson at 256 regions (decision D52), so the
shipped model's calibration margin is larger than the table shows. Deviance metrics (deviance
session, PR 65): imputation skill 0.562 (2x) vs 0.543 (1x), correction gap 0.950 vs 0.951,
calibration ratio 1.067 vs 1.059, both within seed range. Training cost: 129.6 CPU-h for the 3
final seeds (sacct, Fir jobs 5356071, 5356078, 5356095). Decisions D50-D54 in `.longrun/DECISIONS.md`.
