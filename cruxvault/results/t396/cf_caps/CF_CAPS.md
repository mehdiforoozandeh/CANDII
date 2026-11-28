# Dedup and SE/PE caps on the controlled pairs at 25 bp (t396 iteration 2, chunk B5)

Serves R44, R43, R29, R30, R24, R28, R31, R6. Data only. No model and no seeds. Every number comes from `table.tsv`, `q_sensitivity.tsv` or `per_track.tsv` of this folder, unless a line names another file.

## Terms

- Controlled pair: two testbed rows of one track that differ in one processing setting. Covariate `dedup`: the `.nodedup` track (source) predicts the `.base` track (target). Covariate `run_type`: the `.pe` track (source) predicts the SE `.base` track (target). There are 7 tracks per covariate: 6 histone ChIP-seq and 1 DNase-seq (ENCSR903SKE).
- Gap closed: (D_depth - D_law) / (D_depth - D_floor), pooled over tracks. D is the Poisson deviance of the target. D_depth is the deviance under depth scaling. D_floor is the attainable floor, i.e. the expected deviance when the law's own noise model generates the target.
- Halves floor: the deviance of the target `pr2` half under its `pr1` half, scaled by the genome-wide ratio. It is measured at half depth, so it is larger than a full-depth floor. It is reported, and no gap uses it.
- Start law: `subsampled_dm_distinct`, the start-space cap with the two subsampling rates. Count law: `subsampled_count_saturation`, the count-space saturation law with the same rates.
- Rate q: stored lines / lines before the 30M (DNase 50M) subsample of one arm. rho: the pre-subsample total of the source arm divided by that of the target arm.
- Shape `loto`: `a` and `kappa` fitted leave-one-track-out on the testbed pairs of the covariate. This is the primary shape. Shape `iteration1`: the per-assay `a` and `kappa` of `../dedup_cap/per_assay.tsv`, fitted on EIC `train` pairs.
- Interval: the 95% bootstrap interval over the 7 tracks, with 1000 draws.
- Gate (d): a law explains the pairs if its top-1% gap is >= 0.8. A law beats the other if the interval of the gap difference excludes 0 (column `beats_other_law`).

## Results with the fitted rates (decision D25), top 1% of bins

The target rates q_target are measured (`totals.tsv`). The 14 source rates are fitted, because their totals are NA. `fit` fits each source rate on the pair's own start-space table at the iteration-1 shape.

| covariate | space | law | gap closed | interval | D_law | D_depth | D_floor | halves floor | explains | beats other law |
|---|---|---|---|---|---|---|---|---|---|---|
| dedup | start | start law | 0.892 | 0.780 .. 0.969 | 2.37e7 | 8.06e7 | 1.69e7 | 5.52e7 | yes | yes |
| dedup | count | count law | 0.458 | -0.089 .. 0.614 | 9.61e7 | 1.70e8 | 8.66e6 | 3.31e7 | no | no |
| run type | start | start law | 0.849 | 0.785 .. 0.905 | 2.32e7 | 6.20e7 | 1.63e7 | 5.73e7 | yes | yes |
| run type | count | count law | 0.617 | 0.512 .. 0.743 | 4.79e7 | 1.11e8 | 8.97e6 | 3.32e7 | no | no |

Genome-wide, with the `loto` shape: dedup start 0.976 (0.896 .. 1.063), dedup count 0.904 (0.874 .. 0.935), run type start 1.006 (0.944 .. 1.078), run type count 0.913 (0.878 .. 0.939). Every law explains the pairs genome-wide. The start law beats the count law for both covariates. A gap above 1 means that the law's deviance was below the attainable floor.

1. **The fitted-q numbers are in-sample and favor the start law.** The source rate is fitted on the same start-space table that is then scored. The count law reuses that rate, and no fit on the count table corrects it. The fitted rates imply rho = 1.31 to 1.70 for dedup on the histone tracks and 3.27 on DNase (`per_assay.tsv`, rho = q_target / q_source). For run type they imply rho = 2.28 to 2.92 on the histone tracks and 3.67 on DNase. At these rho values the count law fails under the `loto` shape (see the q sensitivity below). Thus the fitted-q contrast between the two laws is optimistic for the start law. The q-sensitivity grid below is the fairer test.
2. **Without the subsampling rates, the start law fails.** At q_source = q_target = 1 (`dm_distinct_unsubsampled`, `loto`), the top-1% gap was 0.557 (0.178 .. 0.786) for dedup and 0.740 (0.694 .. 0.771) for run type. The rates are a necessary part of the law on these pairs, because each arm is its own subsample.
3. **DNase apart from the 6 histone tracks** (top 1%, `loto`, fitted q). On DNase (1 track, no interval), the start law closed 0.911 (dedup) and 0.853 (run type), and the count law closed 0.601 and 0.658. On the 6 histone tracks, the start law closed 0.874 (0.746 .. 0.981) for dedup and 0.846 (0.768 .. 0.916) for run type. The count law closed 0.069 (-0.122 .. 0.682) and 0.560 (0.502 .. 0.790).
4. **The shape choice changes the count-space verdict.** With the `iteration1` shape and the fitted q, the count law closed 0.913 (0.288 .. 0.960) for dedup and 0.866 (0.227 .. 0.941) for run type in the top 1%. Both pass the gate, but the lower bounds are far below 0.8. The start law closed 0.933 (0.865 .. 0.964) and 0.852 (0.766 .. 0.901). Under this shape neither law beats the other.

## Per track (top 1%, `loto`, fitted q; `per_track.tsv`)

| track | assay | dedup start | dedup count | run type start | run type count |
|---|---|---|---|---|---|
| ENCSR061UOM | H3K4me1 | 0.952 | 0.444 | 0.895 | 0.629 |
| ENCSR131DVD | H3K27ac | 0.967 | 0.539 | 0.866 | 0.669 |
| ENCSR401WUV | H3K9me3 | 1.012 | 0.894 | 0.943 | 0.899 |
| ENCSR564OSQ | H3K27me3 | 0.944 | 0.388 | 0.895 | 0.582 |
| ENCSR643QXU | H3K36me3 | 0.908 | 0.274 | 0.887 | 0.699 |
| ENCSR882CQE | H3K4me3 | 0.626 | -0.194 | 0.697 | 0.484 |
| ENCSR903SKE | DNase-seq | 0.911 | 0.601 | 0.853 | 0.658 |

- The start law reached 0.8 on 6 of 7 tracks for each covariate. H3K4me3 (ENCSR882CQE) failed for both covariates.
- The count law reached 0.8 on 1 of 7 tracks for each covariate (H3K9me3).
- On every track, the attainable floor was below the depth-scaling deviance, so every per-track gap is valid. The only law worse than depth scaling was the dedup count law on H3K4me3.

## q sensitivity (`q_sensitivity.tsv`)

The grid fixes q_source = lines_source / (rho x total_target) and refits the `loto` shape, the count floor and the score at each rho. The attainable floor moves with rho, so each gap has its own denominator.

How plausible each rho is, from the totals that are known:
- The 7 base (target) totals are measured: 134.8M to 239.7M lines (`totals.tsv`). The variant totals are unknown (D25).
- Dedup: the dedup-off library holds every dedup-on read, so rho >= 1 is a hard bound. On the 355 iteration-1 EIC pairs, reads on / off had a median of 0.929, a 10th percentile of 0.640 and a minimum of 0.149 (`../dedup_cap/totals.tsv`). Thus rho = 1.0 to 1.25 is the most plausible band, rho = 1.5 is the 90th percentile, and rho = 2 and 6.69 are the tail.
- Run type: PE lines over SE reads. On the 12 iteration-1 PE parents, the ratio was 1.89 to 2.00 on the 5 histone parents and 2.07 to 2.82 on DNase and ATAC (`../sepe_cap/rows.tsv`, reads_parent / reads_se). Thus rho = 1.89 to 2.0 is the most plausible band for the 6 histone tracks. rho = 2.4 to 2.82 is plausible for DNase. rho = 1.5 and 4.0 are outside every iteration-1 value.
- I did not read the ENCODE portal read counts of the variant arms. They would replace this bracket with a measurement.

Top-1% gap closed, all 7 tracks, `loto` shape (in brackets: the 95% interval):

| rho, dedup | start law | count law | start beats count |
|---|---|---|---|
| fitted | 0.892 (0.780 .. 0.969) | 0.458 (-0.089 .. 0.614) | yes |
| 1.0 | 0.975 (0.943 .. 0.993) | 0.940 (0.735 .. 0.959) | yes |
| 1.075 | 0.974 (0.945 .. 0.994) | 0.953 (0.756 .. 0.972) | yes |
| 1.25 | 0.943 (0.910 .. 0.994) | 0.904 (0.755 .. 0.923) | yes |
| 1.5 | 0.877 (0.822 .. 0.981) | 0.502 (0.249 .. 0.718) | yes |
| 2.0 | 0.761 (0.658 .. 0.960) | -0.047 (-0.980 .. 0.219) | yes |
| 6.69 | 0.388 (0.158 .. 0.809) | -0.805 (-2.905 .. -0.300) | yes |

| rho, run type | start law | count law | start beats count |
|---|---|---|---|
| fitted | 0.849 (0.785 .. 0.905) | 0.617 (0.512 .. 0.743) | yes |
| 1.5 | 0.878 (0.840 .. 0.905) | 0.752 (0.442 .. 0.824) | yes |
| 1.89 | 0.928 (0.897 .. 0.949) | 0.888 (0.665 .. 0.916) | yes |
| 2.0 | 0.937 (0.909 .. 0.956) | 0.909 (0.701 .. 0.936) | yes |
| 2.4 | 0.926 (0.909 .. 0.951) | 0.889 (0.743 .. 0.917) | yes |
| 2.82 | 0.814 (0.749 .. 0.931) | 0.400 (0.359 .. 0.726) | yes |
| 4.0 | 0.539 (0.342 .. 0.862) | -0.696 (-1.100 .. -0.110) | yes |

Verdict on q, `loto` shape:
1. The start law clears 0.8 in the top 1% at every rho of the plausible bands. For dedup it clears 0.8 at rho = 1.0 to 1.5 and fails at rho = 2.0 (0.761) and 6.69 (0.388). For run type it clears 0.8 at rho = 1.5 to 2.82 and fails at rho = 4.0 (0.539). On the 6 histone tracks alone, the dedup start law also clears 0.8 at rho = 2.0 (0.820) and fails only at 6.69 (0.509).
2. The count law also clears 0.8 in the most plausible bands: dedup rho = 1.0 to 1.25 (0.904 to 0.953) and run type rho = 1.89 to 2.4 (0.888 to 0.909). Its lower interval bounds there are 0.665 to 0.756, i.e. below the gate. It fails at the fitted q and at every rho outside these bands.
3. The start law beats the count law at every rho, including the most plausible bands.
4. Thus "the start law explains the pairs" does not depend on q inside the plausible bands. "The count law fails" depends on q: it holds at the fitted q and in the tails, and it does not hold in the most plausible bands. "The start law beats the count law" does not depend on q under the `loto` shape.
5. With the `iteration1` shape, the start law clears 0.8 at every dedup rho (0.902 to 0.941). For run type it fails at rho = 1.5 to 2.0 (0.733 to 0.794) and clears 0.8 at rho >= 2.4. Under this shape the start law beats the count law only at dedup rho = 6.69. The `iteration1` shape was fitted on unsubsampled EIC pairs, so I take the `loto` rows as primary.

## Comparison with iteration 1 (EIC pairs, no controlled subsampling)

- Dedup, `../dedup_cap/DEDUP_CAP.md`, 43 `val` pairs, top 1%: the shared-weight cap closed -0.56 (-2.68 .. 0.33), with D_floor 2.20e7. The count law closed -1.91 (-4.97 .. -0.38). Genome-wide: -0.68 and -2.75. No law explained the pairs, and the cap beat the count law. On the controlled pairs, the start law closed 0.892 (fitted q) and 0.943 to 0.975 (rho = 1.0 to 1.25).
- SE/PE, `../sepe_cap/SEPE_CAP.md`, 12 PE parents, top 1%: the start law `thinned_dm_distinct` (q = 0.5) closed 0.365 (-0.073 .. 0.548), and the count law closed -0.117 (-0.443 .. 0.482). Genome-wide: -0.003 and -0.051. Neither law explained the pairs or beat the other. On the controlled pairs, the start law closed 0.849 (fitted q) and 0.928 to 0.937 (rho = 1.89 to 2.0).
- The two settings differ in three ways. (1) Iteration 1 used unsubsampled views, and here each arm is its own subsample, so the rates enter the law. (2) Iteration 1 scored across many libraries with one shape per assay, and here the shape is fitted leave-one-track-out on 7 tracks. (3) Here depth scaling is a weak baseline, because two independent 30M subsamples of different libraries disagree more than a view and its parent do. In the top 1%, D_depth / D_floor of the start law was 4.8 (dedup) and 3.8 (run type) here, against 2.6 for the iteration-1 dedup cap. A weak baseline makes a high gap easier to reach.

## What ran

- Statistics: B3 arrays 6930445 (dedup) and 6930446 (run type), 14 files under `/scratch/mforooz/sc396/it2/cf_caps/{dedup,sepe}`.
- Fit, count floor and score: `fir_job.sh` (kit SC2B5) per tool and scenario. Fitted q: jobs 6930983 and 6930984. Fixed rho: arrays 6934803 and 6934804 (tasks 1 to 6). See `jobs.tsv`. Outputs: `/project/def-maxwl/mforooz/t396/cf_caps/<tool>/<scenario>/`.
- Cost of the probes (sacct TotalCPU): 31:47 + 35:23 = 1.12 CPU-h. Peak memory: 2.57 GB. The count floor was 4 to 5 minutes per pair, i.e. most of the cost.
- `bash run.sh` pulls `fir/` and runs `assemble.py`, which writes `table.tsv` (fitted q), `per_assay.tsv`, `per_track.tsv` and `q_sensitivity.tsv`.

## What was not checked

1. The ENCODE read counts of the 14 variant arms. They would turn the rho bracket into a measurement.
2. The count floor used one draw per pair. Its draw-to-draw spread is not measured.
3. The genome-wide count floors are close across tracks (1.39e8 to 1.59e8 per pair, job logs). I did not check why.
4. The halves floor is at half depth, so it is not a floor for a full-depth target. A full-depth halves floor needs the halves of two independent full libraries, which do not exist.
5. With 7 tracks and 1 DNase track, the bootstrap is coarse, and DNase has no interval.

## h58 / h60 reading (proposed; not recorded in the vault)

h58 (dedup) and the SE views of h60 (run type): supported on the controlled pairs, with moderate confidence. With the rates in the law and the `loto` shape, the start-space cap explains both covariates in the top 1% at every rho of the plausible bands (dedup 0.943 to 0.975, run type 0.928 to 0.937), and it beats the count-space law at every rho. The count law also explains the pairs in the most plausible bands, but with lower bounds of 0.665 to 0.756, so "the count law fails" is not established. This reverses the iteration-1 EIC result (dedup -0.56, SE views 0.365). The reversal rests on 7 tracks, on unmeasured variant totals and on a depth-scaling baseline that is weaker than in iteration 1.
