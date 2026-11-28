# Dedup cap test at 25 bp: start-space cap against the count-space saturation law (t396 C10c)

Serves R43, R28, R29, R30. Data only. No model and no seeds. The source of every number is `table.tsv` unless a line names another file.

## Results

Terms used here:
- Gap closed: (D_baseline - D_law) / (D_baseline - D_floor). D is the Poisson deviance of the dedup-on track, i.e. the target. The baseline is depth scaling unless a line says otherwise.
- Attainable floor (D_floor): the expected deviance when the law's own noise model generates the target. For the start-space law, the C4 simulation supplies it. For the count law, the start-space simulation pushed through the dedup-on kernel supplies it, on the 43 `val` pairs only.
- Shared-weight law: `dm_distinct_shared`, the start-space law that is primary under D15. Its weights are the 25 positional weights, shared by all dedup units.
- Interval: the 95% bootstrap interval over experiments, with 1000 draws.

Gate (d) result, on the 43 `val` pairs, pooled over every layer, run type and assay class:

| law | space | scope | gap closed vs depth scaling | interval | D_law | D_depth_scaling | D_floor | explains (>= 0.8) |
|---|---|---|---|---|---|---|---|---|
| shared-weight cap | start | top 1% | -0.56 | -2.68 .. 0.33 | 7.74e7 | 5.76e7 | 2.20e7 | no |
| count saturation | count | top 1% | -1.91 | -4.97 .. -0.38 | 1.77e8 | 6.71e7 | 9.48e6 | no |
| shared-weight cap | start | genome | -0.68 | -2.97 .. 0.47 | 2.19e8 | 1.65e8 | 8.63e7 | no |
| count saturation | count | genome | -2.75 | -7.17 .. -0.22 | 7.15e8 | 2.66e8 | 1.03e8 | no |

1. No law explains the `val` pairs. In the top 1% and genome-wide, both laws had a larger pooled deviance than depth scaling. Both pooled gaps were below 0, and the gate needs 0.8.
2. The shared-weight cap beat the count law. The gap difference (cap minus count) was 1.35 in the top 1%, with the interval 0.29 .. 3.41 over the 43 experiments. Genome-wide, the difference was 2.07, with the interval 0.59 .. 4.47. Both intervals exclude 0. I computed the difference intervals from `fir/pairs_*.tsv` with the same resampling. The column `beats_other_law` holds the same verdict.
3. The cap had room to improve. In the top 1%, the depth-scaling deviance was 2.6 times the attainable floor of the cap (5.76e7 / 2.20e7). The cap's deviance was 3.5 times that floor.
4. A few deep pairs made the pooled `val` gap negative. Per pair in the top 1%, the cap had a median gap of 0.69 (interquartile range 0.21 .. 2.00), and the count law had 0.17 (-1.46 .. 0.89). The cap reached 0.8 on 19 of 43 pairs, and the count law on 15 of 43. The worst pair was ENCSR701KJD H3K9ac (2 units). Its cap deviance was 2.11e7, against 3.27e6 for depth scaling and 1.27e6 for the floor. Per-pair gaps above 1 happen when the floor is above the depth-scaling deviance, so the per-pair numbers are descriptive only.
5. The cap fit its own training pairs better than it predicted the `val` pairs, but it did not reach the gate there either. On the 312 `train` pairs, the cap's top-1% gap was 0.47 (interval 0.25 .. 0.64), and its genome-wide gap was 0.46 (0.26 .. 0.61). These are in-sample numbers. The count law has no floor on `train`, so its `train` gaps are NaN.

### How often the cap beat depth scaling (per pair, D_law < D_depth_scaling)

| set | shared-weight cap, top 1% | count law, top 1% | shared-weight cap, genome | count law, genome |
|---|---|---|---|---|
| `val` (43) | 24 | 21 | 25 | 20 |
| `train` (312, in-sample) | 215 | 196 | 230 | 176 |
| `val`, multi-unit pairs only (27) | 12 | -- | 12 | -- |

Source: `fir/pairs_*.tsv`. On `val`, the cap beat depth scaling on about half of the pairs (24 of 43).

### By assay class (top 1%, gap vs depth scaling, interval)

| class | `val` pairs | cap, `val` | count law, `val` | cap, `train` (in-sample) |
|---|---|---|---|---|
| narrow | 22 | -0.70 (-3.53 .. 0.40) | -2.04 (-6.88 .. -0.18) | 0.20 (-0.18 .. 0.48) |
| broad | 19 | -0.31 (-5.70 .. 0.35) | -2.15 (-8.69 .. 0.02) | 0.27 (-0.16 .. 0.49) |
| DNase | 1 | 0.97 (one pair, no interval) | 0.87 (one pair) | 0.96 (0.88 .. 0.98) |
| control | 1 | -1.04 (one pair) | 1.15 (one pair) | 0.76 (0.31 .. 0.96) |
| ATAC | 0 | -- | -- | -0.11 (-0.82 .. 0.38), no shape (see below) |

- On narrow marks, the cap beat the count law, with a difference interval of 0.20 .. 3.98. On broad marks, the difference interval of -2.13 .. 8.73 includes 0.
- DNase is the only class where the cap explained the pairs. One `val` pair and 16 `train` pairs exceeded 0.8.

### Layers and run types (`val`, top 1%)

- Native layer (6 pairs, all single-ended): cap 0.39 (-4.20 .. 1.88), count law 0.51 (0.04 .. 1.15). Neither law passed the gate, and neither beat the other.
- Uniform layer (37 pairs): cap -0.58 (-2.62 .. 0.30), count law -1.94 (-5.41 .. -0.40). The cap beat the count law.
- Mixed (1 `val` pair): both laws were far worse than depth scaling. The gaps of 17 and 49 come from a floor above the depth-scaling deviance.
- PE: no `val` pair. The 14 PE `train` pairs are only in the in-sample rows.

### Secondary start-space laws (`val`, all pairs, top 1%)

- Independent-slot `dm_distinct` (K = 25 U): -0.68 (-3.12 .. 0.31).
- `occupancy` (uniform positions): -1.60 (-3.77 .. -0.26).
- `start_cap` (the unconditional h58 form): -1.22 (-4.27 .. -0.11).
- The shared-weight law was the best start-space law. `occupancy` was worse than the Dirichlet laws, so a positional shape is necessary. This agrees with C12b.

### Fitted shapes (`per_assay.tsv`) and fits pinned at a bound

- Most histone assays had a shape a between 0.01 and 0.3. The values were 0.88 for H3K4me3, 0.48 for DNase and 0.17 for control. A small a means that the duplicates pile up on one position.
- 7 of 88 fit rows sat on a bound of the search interval 1e-3 .. 1e3.
- At the upper bound 1e3, ATAC-seq (4 PE pairs), DNase-seq PE (2 pairs) and DNase-seq mixed (1 pair) have **no shape**. Their fit fell back to uniform positions (`occupancy`). Their cap rows test occupancy, not a fitted shape.
- At the lower bound 1e-3, a_shared of H3K23me2 (2 pairs) and of H3K27me3 mixed (1 pair) means that every bin keeps about one read per unit. H2BK5ac (a_shared 0.00119) and H2AK9ac (0.00124) sit just above the lower bound.
- The bound-pinned groups hold 10 of the 355 pairs, and none of them is a `val` pair. They do not change the `val` gate.

## What ran

- Pairs: 355 = 312 `train` + 43 `val` dedup-off / dedup-on pairs, all in the store (`rows.tsv`, C10b). The 7 `blind` rows were not read. Native layer: 46 pairs. Uniform layer: 309 pairs.
- Unit rule: `dedup_unit = replicate` (`../dedup_1bp/UNIT_RULE.json`), so K = 25 U. U is the number of replicates of the pooled view, from 1 to 4.
- Shares of reads kept (`totals.tsv`, tables): dedup-on / dedup-off reads had a median of 0.93 and ranged from 0.15 to 0.99.
- Fits used the `train` pairs: one a, one a_shared and one kappa per (assay, run type), with fallback to (assay, all). The bootstrap had 50 draws over `train` experiments. The all/all row is a point fit, and no lookup used it.
- Code: `fit_score.py` (this folder) drives `tools/t396/dedup_cap.py`. `fir_job.sh` runs it on Fir (kit SC10c). The count floor ran with `slurm/t396/dedup_cap.sh floor-count`. Job ids are in `jobs_c10c.tsv`.
- Cost (`sacct`, batch steps, TotalCPU): fit array 6749920 took 1.85 CPU-h, floor array 6756694 took 0.94 CPU-h and score array 6756695 took 0.39 CPU-h. The three probes took 0.78 CPU-h. The total was 3.96 CPU-h.
- `bash run.sh` pulls `fir/` and rebuilds `per_assay.tsv`, `totals.tsv` and `table.tsv`.

## What was not checked

1. The count-space floor exists only for the 43 `val` pairs, so the count law has no `train` gap.
2. The count floor drew from the independent-slot a, not the shared-weight a_shared. The two shapes differ by less than 10% on most assays (`per_assay.tsv`).
3. The `val` set has 1 DNase pair, 1 control pair, 1 mixed pair, 0 ATAC pairs and 0 PE pairs. Those classes have no `val` interval.
4. One shape per assay may be the wrong unit. The `val` failures sit on deep multi-unit pairs, which suggests that the shape varies by library. I did not fit a per-experiment shape.
5. I did not test whether the depth-scaling advantage comes from a few repeat or blacklist bins with very large dedup-off counts.

## Verdict on h58 (proposed; not recorded in the vault)

The proposed verdict is mixed. The expectation reads: "On the dedup-off/dedup-on view pairs, the start-space cap predicts the dedup-on track closer to the half-vs-half floor than the count-space saturation law, in the top 1% of bins". This test replaced the half-vs-half floor with the attainable floor (plan C4). The comparative part held on `val`: the cap beat the count law, with a top-1% gap difference interval of 0.29 .. 3.41. The cap did not explain the pairs. Its pooled `val` gap was -0.56 (interval -2.68 .. 0.33), and the gate needs 0.8. Thus one closed-form shape per assay does not predict deduplication across libraries, except on DNase.
