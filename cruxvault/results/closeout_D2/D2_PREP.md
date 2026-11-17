# Close-out D2: add-on tables of eic_v1_2 with the second experiments (prep, 2026-11-17)

Nibi jobs 27601259 (13 min 27 s, 4 CPU, about 0.9 CPU-h) and 27602716 (layer pair file), kit COD2 / COD2b. Item root on Nibi: `/project/def-maxwl/mforooz/ONION/CO-D2`. Log: `co_d2_prep_27601259.log` (this folder).

## What exists now

- `CO-D2/eval/eic_v1_2/evalset.json`: rebuilt on Nibi with the D1 arguments; sha256 `26f9e32f...` equals the frozen Fir file, so it is the same eval set.
- `CO-D2/tables/eic_v1_2x`: add-on of `tables/eic_v1_2` with `--catalog`. 71 second-experiment rows appended (all `evalonly`), 71 `other_experiment` pairs (split val), 48 `missing.tsv` rows (17 unbuilt priority-5 rows and 31 priority-6 fallbacks never submitted: "not built").
- `check_dataset.py --addon-of tables/eic_v1_2`: PASS; every leak count 0; `second-experiment rows: 71 (other lab 35, same lab 36)`.
- Covariates of the 71 rows: lab, donor and (histone) antibody filled for every row; 18 rows carry a cause note (mixed donor or antibody, platform from the catalog). 2 rows have a lab outside the 6 training labs (Gregory Crawford, Duke; training-split DNase-seq): they encode to id 0. QC columns (nsc, frag_len) were filled for 28 rows from the work folders.
- `CO-D2/eval/eic_v1_2x/evalset.json`: 13 validation targets with a second experiment, 13 evaluation-only targets, 13 `other_experiment` pair ids; `evalset.py check`: 0 problems.

## Pair smoke (regional totals after depth matching, 256 regions; outside 1 +- 0.05 = metric-7 ratio Undefined)

| split | pair class | outside | within |
|---|---|---|---|
| val | experiment | 0 | 3 |
| val | experiment;donor | 0 | 3 |
| val | experiment;lab | 1 | 2 |
| val | experiment;lab;donor | 2 | 2 |
| train | experiment | 4 | 9 |
| train | experiment;donor | 0 | 17 |
| train | experiment;lab | 7 | 6 |
| train | experiment;lab;donor | 6 | 9 |

All pairs: 20 of 71 outside. Other-lab pairs: 16 of 35 outside; same-lab pairs: 4 of 36. Validation: 3 of 13 outside, so gate GX clause 3 (more than half) does not trip and the metric-7 upper bound can be quoted on 10 pairs. The prediction written before the run (more than half outside) was wrong; the two smoke pairs of t264 were not typical.
