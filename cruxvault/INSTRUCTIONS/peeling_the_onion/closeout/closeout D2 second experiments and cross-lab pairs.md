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

## 9. Specification as worked (2026-11-17, overnight, item agent CO-D2)

### 9.1 State today, read from the artifacts (this replaces section 3 where they differ)

- Tables on Nibi: `eic_v1`, `eic_v1_1`, `eic_v1_2` (section 3 said only `eic_v1`). `eic_v1_2` is byte-identical on Nibi and Fir (sha256 of its six hashed files compared). It holds 0 rows with `in_eic = evalonly`, 0 `other_experiment` pairs, and 0 pairs whose `knobs_differing` names `lab` (counted from `measurements.tsv` and `pairs.tsv`). So no second experiment and no real cross-lab pair is in any table today.
- Lab vocabulary of `eic_v1_2` (`CHANGES.md`): 6 training labs. Every validation second experiment has a lab in that list; 4 training-split DNase-seq second experiments (Gregory Crawford, Duke) do not, so they would encode to id 0.
- Build (sacct of the build arrays, and `counts.h5` + `meta.json` in `/scratch/mforooz/onion/measurements`): 71 of 88 second experiments are built; all 13 validation second experiments are built. Not built: 13 DNase-seq rows held for their tail partner (2015, 2023, 2039, 2040, 2042, 2052, 2067, 2072, 2076, 2077, 2078, 2081, 2084) and 4 DNase-seq rows that failed (2028 FAILED after 6:41 h; 2049, 2075, 2079 NODE_FAIL after about 42 h). All 17 are training-split pairs: they feed the layers only, never a model score.
- A second experiment's `meta.json` has `lab`, `antibody` and `donor` empty (read: `ENCSR487BEW.pooled.uniform.base/meta.json`). The add-on builder copied `meta.json`, so on top of `eic_v1_2` the second experiment would carry no lab in C' while its EIC partner does. Fixed in code (9.4).
- The eval set of `eic_v1_2` (`26f9e32f...`) and the 9 base-recipe runs trained on `eic_v1_2` (`runs_fir/eic_v1_2_rung{2,3,4}_base_heldout`) exist on Fir only. Fir is in an outage until about 2026-11-20.
- Cross-lab correction is already computed by the scorer: an `other_experiment` pair of the add-on eval set is a pair job (EIC track pr1 -> second experiment pr2, the model given the second experiment's own covariates), reported as `replicate_pair_gap_closed_vs_depth_only.per_rung.other_experiment`; depth-only scaling and the floor (the second experiment's pr1 -> pr2) are computed in the same job. Section 5 step 5's "cross-lab correction gap closed" is that value. The scorer had no split by pair class (added, 9.4).
- Pipeline work folders (`/scratch/mforooz/onion/work`, leaving scratch): `build_tables.py addon` reads them only for the QC columns `nsc, rsc, nrf, pbc1, pbc2, frag_len` of the appended rows (`qc_values`). The prep job of 9.5 reads them before the removal. Counts, halves and every metric of this item read `measurements/`, which stays. The build of the 17 missing rows writes new work folders and must run only after the removal.

### 9.2 Gate GX, counted from the ledger and the built directories (before the add-on check)

Built pairs per pair class, training and validation together (cells in brackets):

| assay class | experiment (same lab, same donor) | experiment;lab | experiment;donor | experiment;lab;donor |
|---|---|---|---|---|
| broad histone | 4 (3) | 8 (2) | 5 (4) | 13 (4) |
| narrow histone | 8 (3) | 6 (2) | 11 (8) | 6 (4) |
| DNase-seq | 4 (4) | 2 (2) | 2 (2) | 0 |
| ATAC-seq | 0 | 0 | 2 (2) | 0 |

- Clause 1 trips: `experiment;lab` (other lab, same donor) has fewer than 3 cells in every assay class (`MIN_PAIRS = 5` from `MIN_CELLS = 3`, `layers.py:94`). The pure lab layer is not estimable. Building the 17 missing DNase-seq rows adds 2 `experiment;lab` pairs (rows 2052, 2077), so DNase-seq would reach 4 pairs: still under the minimum.
- Estimable with lab and donor changed together: `experiment;lab;donor` in broad histone (13 pairs, 4 cells) and narrow histone (6 pairs, 4 cells). Pooled other-lab (both lab classes): broad 21 pairs from 6 cells, narrow 12 from 6.
- Validation pairs: 13 (7 other-lab: 3 `experiment;lab`, 4 `experiment;lab;donor`; 6 same-lab). Clause 3 is counted by the pair smoke of the prep job.

### 9.3 Question, steps, expectation (written before any result)

Question: does the model correct the level and shape of a real second experiment of the same cell when it is given that experiment's lab, antibody, donor and platform (cross-lab correction), and where does the experiment layer sit beside the cell-type layer?

Steps: (1) prep (add-on tables `eic_v1_2x` with filled covariates, check, add-on eval set, layer pair file, pair smoke); (2) layers on `eic_v1_2x`; (3) the 5 baselines with `--addon`; (4) the 3 base recipes x 3 seeds trained on `eic_v1_2`, then the 3 final candidates x 3 seeds when M1 names them; (5) report `OTHER_EXPERIMENT.md`.

Expectation (before any run): (a) more than half of the 13 validation pairs are outside 1 +- 0.05 in regional totals, so the metric-7 upper bound is not quoted (clause 3); the two smoke pairs were 1.68 and 1.38. (b) The cross-lab gap closed of the 3 base recipes is between -0.2 and 0.2 and below the same-lab `experiment` class: a 38% to 68% level difference is not predictable from 6 lab ids learned without any second experiment in training. (c) The `experiment;lab;donor` layer of broad histone has an interval that excludes 0.

### 9.4 Code (switches off = old bytes)

- `tools/onion_data/build_tables.py addon --catalog <dir>`: lab, antibody, donor and platform of each appended row by the D1 rule (`fill_covariates.join_rows`); `build_info.json` records the catalog and the cause notes. Off: add-on output on the synthetic fixture has the same digest on the base commit 01248f4 and on this branch (`0041e148...`, every file but `build_info.json`). Test: `tests/test_onion_data_tables_addon_fill.py`.
- `tools/onion_eval/score.py`: with `--addon` only, the key `other_experiment_gap_closed_by_pair_class` (pooled, per pair class, per pair, each with the model, depth-only and floor CRPS). Off: the frozen-form score digest test of `tests/test_onion_eval_addon.py` stays green. Test: `tests/test_onion_eval_addon_pair_class.py`.
- `slurm/onion_data/closeout_d2_submit.sh` (stages prep, layers, baselines, models, build; dry run by default) and `slurm/onion_data/closeout_d2_prep.sh`. Kit `COD2` on Nibi. Item root on Nibi: `/project/def-maxwl/mforooz/ONION/CO-D2`.

### 9.5 Decision table

| outcome | conclusion |
|---|---|
| pair smoke: more than 6 of 13 validation pairs outside 1 +- 0.05 | metric-7 upper bound not quoted (clause 3); report the count per pair class |
| `experiment;lab;donor` layer interval excludes 0 in broad or narrow histone | lab plus donor is a real variance source; the t305 cell-type layer held part of it; the pure lab part stays unseparated (clause 1) |
| `experiment;lab;donor` and `experiment` layers overlap | no evidence that changing lab adds variance beyond a repeat experiment |
| model gap closed on other-lab pairs above depth-only (0) by more than the seed range | the model corrects some lab effect, at least in level |
| model gap closed within the seed range of 0 | no lab correction; the D1 covariates did not help on real pairs |
| model gap closed on other-lab pairs below the same-lab `experiment` class by more than the seed range | lab differences are harder than repeat differences for the model |

### 9.6 Controls and traps

- Lab is confounded with read length, platform and run type (`tech_diff` of the ledger); every validation second experiment is single-end. Report `tech_diff` beside every class.
- The pair smoke depth-matches on the declared `depth_reads`; check the run type of both sides of each pair that is outside the tolerance.
- 7 other-lab validation pairs give a wide interval: quote the pair count and the seed range with every class value; with fewer than 5 pairs a class value is descriptive only.
- The add-on scores need runs trained on `eic_v1_2`. The `eic_v1` runs on Nibi (`runs_fir/rung*_base_heldout`) are refused by the scorer on `eic_v1_2x`; do not build an add-on of `eic_v1` to reuse them (it would carry empty lab covariates).

### 9.7 Cost (CPU-h, allocated)

- prep: 1 job x 4 CPU x 3 h = 12 at most (submitted 2026-11-17, Nibi job 27601259).
- layers: 8 CPU x 12 h = 96 at most; E7a's estimate is 50.
- one add-on score: the full eval set took 11.7 h on 2 CPUs (E2 timing job 4448330: 43 targets, with pairs) = 23.4 CPU-h. The add-on set has at most 26 targets (13 EIC + 13 second) and at most 13 pair jobs, so about 26 / 43 x 9.0 h + 2.6 h = 8.0 h x 2 CPU = 16 CPU-h; the job asks 2 CPU x 14 h = 28.
- baselines 5 x 16 = 80 expected (140 allocated at most); base recipes 9 x 16 = 144 (252); final candidates 9 x 16 = 144 (252).
- Total without the final candidates: 12 + 50 + 80 + 144 = 286 CPU-h expected, 12 + 96 + 140 + 252 = 500 at most.
- The 17 missing DNase-seq rows (not recommended, 9.2): up to 17 x 72 h x 16 CPU = 19,584 allocated; the measured single-end smoke row 2030 used 2.93 h x 16 = 47 CPU-h for 47 M reads, about 1 CPU-h per million reads, so 7,312 M catalog reads give about 7,300 CPU-h.
