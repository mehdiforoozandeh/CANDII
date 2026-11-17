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

## 9. Specification as worked (2026-11-17, overnight, item agent CO-E4)

### 9.1 State today, read from the artifacts (replaces section 3 where they differ)

- Source: the D2 add-on tables Nibi `/project/def-maxwl/mforooz/ONION/CO-D2/tables/eic_v1_2x` (on top of `eic_v1_2`) and the D2 layers eval set `CO-D2/eval/eic_v1_2x/evalset_layers.json`. Counted from their `pairs.tsv`, `measurements.tsv` and `replicate_pairs` (script output, this agent): `pairs.tsv` holds 785 pseudoreplicate (7 blind), 219 isogenic, 4 anisogenic, 71 other_experiment and 2,911 other_cell_type pairs; no sequencing and no technical pair.
- TRAP 1 (found): the D2 layers eval set holds 3,205 pairs and NO pseudoreplicate pair (`evalset.py` keeps a replicate pair only when a != b; the same defect t305 worked around). The D2 `layers` stage would therefore run without the sampling rung. `ladder.py pairlist` rebuilds the pair list as t305 did: + 778 train/val pseudoreplicate pairs, - the 4 anisogenic pairs (all in cell C09; C09 has no isogenic pair) -> 3,979 pairs. The D2 `layers` stage must not be submitted; this item's `layers` stage replaces it.
- Gate counts of D2 9.2 verified on the eval set (pairs, cells): broad histone `experiment;lab` 8 (2: C17, C24), `experiment;lab;donor` 13 (4: C02, C20, C27, C32); narrow histone `experiment;lab` 6 (2), `experiment;lab;donor` 6 (4); DNase-seq `experiment;lab` 2 (2: C17, C23). They agree with D2.
- TRAP 2 (found): the pooled-by-class rows of `layers.py` subtract cell means taken over all assays of a class, so a cell's increment mixes the assays that carry its other_experiment pairs with other assays that carry its isogenic pairs. Pair values differ between assays by more than the layers (t305: H3K27me3 sampling 0.0470, H3K36me3 0.0409; replicate layer about 0.0005). The ladder therefore uses assay-matched increments (9.3). Matched (cell, assay) counts: broad `experiment;lab;donor` keeps 4 cells (C02, C20, C27, C32 have broad isogenic pairs); narrow keeps 3 (C20 has no narrow isogenic pair).
- Sequencing and technical rungs: no pair in any table. The per-library and per-lane units are build priority 4; the PI approved all 722 rows on 2026-11-16 (D4), and the build is blocked (Nibi /scratch over quota, Fir outage). Not part of tonight.
- Mixed model: `mixed_model.py` took only `in_eic` train and val rows, so the second experiments (`evalonly`) never entered it. Switch added (9.4).

### 9.2 Question

Per assay class, report every D12 rung with its increment, interval, pair count and cell count, or the reason it is empty or not estimable; and give the crossed mixed-model variance fractions beside them, with what each factor can be identified from.

### 9.3 The ladder (rows of `ladder.tsv`, one block per assay class, never pooled across assay classes)

| D12 rung | pairs that feed it | increment over | layer (what the increment holds) | status with these counts |
|---|---|---|---|---|
| pseudoreplicate | pr1 vs pr2 of one library | (innermost) | sampling | estimable, all classes |
| sequencing | two runs of one library | pseudoreplicate | sequencing | empty: build priority 4 not built |
| technical | two libraries of one sample | nearest below | library | empty: build priority 4 not built |
| isogenic | br1 vs br2 of one experiment | nearest below (pseudoreplicate) | biological replicate, with sequencing and library | estimable (not ATAC-seq: no pair) |
| anisogenic | two donors in one experiment | isogenic | donor | not estimable: 4 pairs, 1 cell (C09), no isogenic pair in C09; dropped from the pair list |
| other experiment, class `experiment` | same lab, same donor | isogenic | experiment repeat | broad 4 (3 cells), narrow 8 (3), DNase-seq 4 (4): decided by the matched count |
| other experiment, class `experiment;lab` | other lab, same donor | isogenic | lab (pure, ruling 30) | NOT ESTIMABLE in every class (2 cells) |
| other experiment, class `experiment;lab;donor` | other lab, other donor | isogenic | lab with donor | estimable: broad 13 (4), narrow 6 (4 all, 3 matched) |
| other experiment, class `experiment;donor` | same lab, other donor | isogenic | experiment repeat with donor | broad 5 (4), narrow 11 (8), DNase-seq 2 (2), ATAC-seq 2 (2) |
| other cell type | two cell types | isogenic | cell type with donor and experiment (t305 form) | estimable |

- Assay-matched increment: for each (cell, assay) with pairs of the rung and of the rung below, mean pair value of the rung minus that of the rung below; mean over the cell's assays; mean over cells. Interval: 95% cluster bootstrap over cells, t form with sqrt(n/(n-1)) (`layers._interval`). Estimable: at least 5 matched pairs from 3 cells (`layers.MIN_PAIRS`, `layers.MIN_CELLS`, layers.py:100). A row under the rule has an empty value and the flag "not estimable: N pairs from C cells ...". No pooled-across-class row exists (PI decision open; D2 9.9).
- `share` = increment / the class's other_cell_type pair value (the variance between two tracks that share only the assay), the scale of a mixed-model fraction.
- The cell-type layer of ruling 30 (over the same-donor experiment pairs) stays in `layers.tsv` (its own row, flag `donor not separated` where the same-donor classes are too few); the ladder quotes the t305 form so it is comparable with t305.

### 9.4 The crossed mixed model and its identifiability with these counts

- Model (`mixed_model.py`, per assay and Haar band): y = mu + u_cell + u_experiment + u_replicate + u_library + u_lab + e, every observation thinned to one common depth (so depth is one level, notepad L421), REML. replicate = `cell|donor` (crossed with experiment); library = the finest unit of an experiment (in EIC the br units, i.e. biological replicates); lab is crossed with cell (labels shared over cells); e = between pseudoreplicate halves (sampling).
- Mapping to the ladder: e -> sampling; library -> biological replicate (isogenic over pseudoreplicate); replicate -> donor; experiment -> the experiment-specific part of an other-experiment pair; lab -> the part shared by every experiment of one lab; cell -> cell type. The ladder's `lab with donor` increment corresponds to about s_lab + s_experiment + s_replicate (+ library terms); only the sum is comparable.
- Identifiability, counted by `ladder.py design` on `eic_v1_2x` (30 assays with at least 3 observations from 2 cells; file `results/closeout_E4/design.tsv`):
  - Without second experiments: experiment and donor are confounded with cell in all 30 assays; library is estimable in 29. Lab is "estimable" by the partition rule in 11 assays but varies inside no cell: it is then identified only from labs shared across cells, i.e. confounded with which cell types each lab measured. Not a lab effect; never quote it.
  - With second experiments (`--second-experiments`): experiment estimable in 20 assays, with a contrast inside at least 3 cells in 7 (DNase-seq, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3); donor in 7 (at least 3 cells: H3K27ac, H3K36me3, H3K4me1, H3K4me3, H3K9me3); lab in 12 (at least 3 cells: H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3).
  - Quoting rule: a factor's fraction is quoted only where it varies inside at least 3 cells (the ladder's cell rule); elsewhere it is listed as "not identified inside cells". The bootstrap resamples cells, so with 3 to 5 contrasting cells the interval is wide; quote it.

### 9.5 Expectation (before any run)

- Sampling stays the largest layer (t305: 76% to 97%); isogenic over pseudoreplicate stays small (t305: 0.1% to 6%).
- Lab with donor (broad, narrow): above the replicate layer and below the cell-type layer; share of the cell-type pair value between 0.02 and 0.10. t264 other-lab totals differed by 38% to 68% after depth matching, but a level difference is mostly removed by the centering of d, so the 25 bp shape share is small.
- Mixed model: experiment + lab fractions in the 5 identified assays are below the cell fraction; the nested `lab with donor` share and the mixed-model sum (experiment + lab + replicate) agree within intervals.

### 9.6 Decision table

| outcome | conclusion |
|---|---|
| lab-with-donor interval excludes 0, share below cell type | a second experiment from another lab and donor adds a measurable layer; quote it with its 4 (3) cells |
| lab-with-donor interval includes 0 | at 25 bp, lab and donor together are not distinguishable from biological replicates on EIC |
| nested share and mixed-model sum agree within intervals | the decomposition is robust to the order of the ladder for these layers |
| they disagree | the crossed design biases the nested order; quote the mixed model for those layers (ruling 30) |
| pure lab | not estimable on EIC (2 cells per class); a limit of EIC, not of the method |

### 9.7 Controls and traps

- Common depth: an assay's common depth is the shallowest side of any of its pairs, so second experiments can lower it; every `layers.tsv` row carries `common depth set with second experiments`. Values are not comparable with t305.
- One pair (`p9023fbe311cd`, C09 H3K27me3) has donor `mixed` on both sides (D2 9.9): in the mixed model both sides get the same donor label. Listed, not fixed.
- Two training DNase-seq second experiments carry a lab outside the training vocabulary (Duke); irrelevant here (no model).
- Telescoping: `total_variance.telescoping_check` on the new `layers.tsv` (in the report); groups with rungs from different cells are not expected to telescope.
- Switches off: `layers.py` without `--pairs-out` writes a byte-identical `layers.tsv` (test); `mixed_model.py` without `--second-experiments` and `--assays` selects the same observations and assays (tests).

### 9.8 Cost (allocated CPU-h)

- smoke (H3K36me3; done tonight, 9.9): at most 1x0.5 + 4x1 + 2x2 + 1x0.5 = 9.
- layers (pair list, design, layers with per-pair values): 1x0.5 + 1x0.5 + 4x3 = 13 at most; t305 measured 1 h 19 min on 4 CPUs for 3,908 pairs -> about 5.3 + 1 = 6 expected.
- mixed (4 bands, every assay, 200 bootstrap draws): the smoke took 1 min 57 s for 1 assay, 1 band, 20 draws on 2 CPUs (venv setup included). At 10 times the draws, about 15 min per assay and band at most, x 30 assays = 7.5 h per band -> submit with MIXED_CPUS=2 MIXED_TIME=10:00:00: 4 x 2 x 10 = 80 CPU-h allocated at most, about 60 expected.
- report: 1x0.5.
- Submit: `bash slurm/onion_eval/closeout_e4_submit.sh layers|mixed|report nibi` (dry run by default; `DRY_RUN=0` submits).
- Files under `CO-E4` for the full run: layers stage 5 outputs + 3 logs; mixed 4 + 4 logs; report 4 + 1 log; 3 folders: about 24 (not 10).
- Kit: COE4 (git a8d6e13) holds the layers, pair-list and design code, so the `layers` stage can run on it. It lacks the review fixes of 9.10 (`--min-contrast-cells`, the ladder's own telescoping check, the matched share). The `mixed` and `report` stages need a kit with them, or the report runs on the laptop from the small pulled tables (pair values about 4 MB).

### 9.9 Smoke run (2026-11-17; evidence `results/closeout_E4/E4_SMOKE.md`)

- Nibi jobs 27607315 (pair list), 27607317 (layers), 27607318 (mixed model), 27607319 (report), kit COE4; 0.62 CPU-h allocated; 13 files under `/project/def-maxwl/mforooz/ONION/CO-E4`.
- H3K36me3 sampling 0.04092 [0.0366, 0.0452] and biological replicate 0.000369 [0.0000069, 0.00073] agree with t305 (0.0409; 0.0004).
- Telescoping check: 0 of 75 groups pass, because every rung above sampling holds other cells than the rung below. The acceptance check "telescoping passes" cannot hold as written on this pair set; the PI or the foreman decides whether to replace it by a check on common cells only.
- Full layers stage not submitted tonight (foreman order: no more than 50 more files under /project; the stage writes 8 files with its logs and is about 6 CPU-h).

### 9.10 Review fixes (2026-11-17)

- Quote rule in code. `mixed_model.py --min-contrast-cells 3` (off by default; the submit script passes it) blanks the fraction of a fitted factor other than cell that varies inside fewer than 3 cells, with the note "not identified inside cells". It marks a fraction below 1e-6 as a boundary fit. `ladder.py report --design` applies the same rule to tables that were written without the switch.
- The ladder's own telescoping check (`chain_check.tsv`): pseudoreplicate -> isogenic -> other_cell_type on the (cell, assay) units that hold all three. Smoke: 65 of 65 groups are exact; broad histone, all bands, 16 units: 0.03821 + 0.000369 + 0.004489 = 0.04307. The old check on layers.tsv stays, labeled as known to fail (its rungs hold different cells).
- `share` uses the same (cell, assay) units as the increment (column `share_units`). A negative increment is kept and flagged.
- Ladder vs mixed model at the fine band: diagnosis in `results/closeout_E4/E4_MIXED_DIAG.md`. The response is identical and the REML fit is sound. The mixed model's homoscedastic residual and the ladder's one-sided subtraction both fail under track-specific sampling variance (0.021 to 0.049). Do not submit the mixed stage before a per-library residual is added. Quote two-sided ladder rungs as upper values until two-sided subtraction is decided.
