# Follow test of the existing models on the controlled pairs (t396 iteration 2, chunk B6)

Serves R21, R23, R39, R20, R24, R22, R6. Every number comes from `table.tsv` or `per_track.tsv` of this folder, or from `../cf_caps/table.tsv` where a line says so. The column "relative deviance change" is computed from `per_track.tsv` (formula in Terms). It is not a column of `table.tsv`.

## What ran

- Models. (1) Read-count: `rcp_combo_w2x_unet` seeds 0-2, 160,000 windows, the final recipe. (2) "read-start, 80,000 windows (unequal)": `rs_halves_mix_avg_pearson_unet` seeds 0-2. **The two models are not an equal comparison.** The read-start model saw half the training windows of the read-count model. The equal-terms test is chunk A11a on the factorial runs.
- Pairs: 7 tracks (6 histone ChIP-seq, 1 DNase-seq, ENCSR903SKE) x 2 covariates (dedup: `.base` vs `.nodedup`; run type: `.base` vs `.pe`) x 2 directions = 28 directed pairs (`pairs.tsv`). Regions: the 256 regions of `evalset_p16.json`.
- Prediction arrays 6931724-6931729 (`jobs.tsv`, 12 slice tasks): all 12 COMPLETED, exit 0 (sacct, 2026-11-28). Every seed folder holds 28 npz files and a `meta.json` that lists 28 pairs. Peak MaxRSS was 9,701,448 KiB of 10,000 MB (95%); no task ran out of memory. Wall time: 17-22 min per read-count task, 67-78 min per read-start task.
- Table job 6939785 (kit SC2B4, CPU, 2 cores, 8000M): COMPLETED in 5 min 12 s, 3 min 39 s CPU. Its MaxRSS was 8,181,400 KiB, i.e. 99.9% of the request. It ran `cf_follow.py table --caps-table` with `../cf_caps/table.tsv` (sha256 checked by `run.sh`).

## Terms

- Follow score of one track: (D_source_cov - D_target_cov) / (D_source_cov - D_halves). D is the Poisson deviance of the target's `pr2` half. D_source_cov uses the model's prediction with the source's output covariates. D_target_cov uses the target's output covariates. D_halves is the half-vs-half floor, i.e. the deviance of `pr2` under the scaled `pr1` half. A pooled score sums numerators and denominators over the included tracks of one seed.
- Denominator rule: a track whose denominator is <= 0 is excluded from its row. The column "tracks excluded" is the largest count over the 3 seeds.
- Halves floor (table column): sum D_halves / sum D_source_cov over the included tracks, mean over seeds.
- Relative deviance change: (sum D_source_cov - sum D_target_cov) / sum D_source_cov over all 7 tracks of one seed. It is positive when the target covariates lower the deviance. It needs no floor, so it is defined on every row.
- Total ratio: predicted total with the target covariates / with the source covariates, mean over seeds.
- Cap-law gap: the gap closed vs depth scaling of `../cf_caps/table.tsv`, shape `loto`, all 7 tracks, fitted rates. Start space uses the start law (`subsampled_dm_distinct`); count space uses the count law (`count_saturation`). B5 scored the direction variant -> base only, so the other direction has no cap value. The cap-law gap has another baseline (depth scaling) and another floor (the attainable floor). It is context for the same pairs, not the same metric.
- Gate (h): a model follows a covariate if the pooled top-1% follow score is >= 0.8.
- Seed range: max - min over the 3 seeds. An effect smaller than its seed range is reported as no effect.

## Main result: the follow score is undefined on most tracks, because the halves floor is not a floor

The model's prediction with the source covariates already had a lower deviance than the half-vs-half floor on most tracks. Over all 7 tracks, sum D_halves / sum D_source_cov was 2.3 to 3.8 genome-wide and 0.13 to 3.2 in the top 1% (`per_track.tsv`, every row and seed). The model predicts a smooth mean; the `pr1` half carries its own Poisson noise, so it is a worse predictor of `pr2` than the model. Thus the denominator was <= 0, and the track was excluded:

- Genome-wide: 7 of 7 tracks were excluded in every seed on 10 of 12 rows. On the other 2 rows only ENCSR903SKE (DNase-seq) stayed, in some seeds.
- Top 1%: 5 to 7 of 7 tracks were excluded on every row. The tracks that stayed were ENCSR903SKE (DNase-seq) and ENCSR882CQE (H3K4me3). In start space, the direction variant -> base excluded all 7 tracks.

Thus every finite follow score below rests on 1 or 2 tracks, and the 5 other histone tracks are absent. The relative deviance change is defined on all 7 tracks and is the reading that carries the result.

## Gate (h)

No model followed either covariate. Every top-1% row had a follow score far below 0.8, or no score. The count of tracks at >= 0.8 was 0 on every row except one: read-start, 80,000 windows (unequal), start space, dedup, base -> variant had 1 track (ENCSR882CQE, H3K4me3). That track scored 2.217, 1.676 and 2.669 over the seeds. A score above 1 means that the deviance fell by more than the source-to-floor distance. The denominator of that track was small, so the score overshoots and is not a clean follow.

## Top 1% of source bins

| model | space | covariate | direction | follow score | seed range | per seed | halves floor | tracks >= 0.8 | tracks excluded | total ratio | cap-law gap | relative deviance change per seed | its seed range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| read-count | count | dedup | base -> variant | 0.010 | 0.035 | 0.012; 0.027; -0.009 | 0.03 | 0 | 5 of 7 | 1.019 | n/a | +0.012; +0.026; -0.006 | 0.032 |
| read-count | count | dedup | variant -> base | 0.001 | 0.384 | 0.041; -0.211; 0.173 | 0.39 | 0 | 6 of 7 | 0.981 | 0.458 | +0.019; -0.051; -0.001 | 0.070 |
| read-count | count | run type | base -> variant | 0.062 | 0.083 | 0.113; 0.045; 0.030 | 0.08 | 0 | 5 of 7 | 1.030 | n/a | +0.091; +0.046; +0.029 | 0.062 |
| read-count | count | run type | variant -> base | -0.216 | 0.960 | -0.750; -0.108; 0.209 | 0.46 | 0 | 6 of 7 | 0.968 | 0.617 | -0.157; -0.079; -0.018 | 0.139 |
| read-start, 80,000 windows (unequal) | count | dedup | base -> variant | 0.098 | 0.038 | 0.081; 0.093; 0.119 | 0.03 | 0 | 5 of 7 | 1.198 | n/a | +0.072; +0.085; +0.114 | 0.041 |
| read-start, 80,000 windows (unequal) | count | dedup | variant -> base | 0.436 | 0.394 | 0.250; 0.415; 0.644 | 0.37 | 0 | 5 of 7 | 0.826 | 0.458 | +0.110; +0.176; +0.305 | 0.196 |
| read-start, 80,000 windows (unequal) | count | run type | base -> variant | -0.058 | 0.111 | -0.117; -0.006; -0.051 | 0.08 | 0 | 5 of 7 | 0.950 | n/a | -0.093; -0.006; -0.046 | 0.087 |
| read-start, 80,000 windows (unequal) | count | run type | variant -> base | -0.101 | 0.272 | -0.246; -0.083; 0.026 | 0.56 | 0 | 5 of 7 | 1.052 | 0.617 | -0.071; -0.022; +0.010 | 0.081 |
| read-start, 80,000 windows (unequal) | start | dedup | base -> variant | 0.094 | 0.021 | 0.086; 0.090; 0.106 | 0.17 | 1 | 5 of 7 | 1.186 | n/a | +0.060; +0.068; +0.080 | 0.020 |
| read-start, 80,000 windows (unequal) | start | dedup | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.832 | 0.892 | +0.060; +0.087; +0.181 | 0.121 |
| read-start, 80,000 windows (unequal) | start | run type | base -> variant | -0.071 | 0.147 | -0.149; -0.001; -0.065 | 0.35 | 0 | 5 of 7 | 0.953 | n/a | -0.068; -0.001; -0.036 | 0.068 |
| read-start, 80,000 windows (unequal) | start | run type | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.050 | 0.849 | -0.044; -0.012; +0.000 | 0.044 |

## Genome-wide

| model | space | covariate | direction | follow score | seed range | per seed | halves floor | tracks >= 0.8 | tracks excluded | total ratio | cap-law gap | relative deviance change per seed | its seed range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| read-count | count | dedup | base -> variant | 0.114 | 0.285 | -0.029; 0.256; n/a | 0.85 | 0 | 7 of 7 | 0.988 | n/a | -0.003; +0.004; +0.008 | 0.011 |
| read-count | count | dedup | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.012 | 0.904 | +0.017; +0.005; -0.004 | 0.020 |
| read-count | count | run type | base -> variant | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.024 | n/a | +0.032; +0.021; +0.005 | 0.027 |
| read-count | count | run type | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.977 | 0.913 | -0.021; -0.019; +0.010 | 0.030 |
| read-start, 80,000 windows (unequal) | count | dedup | base -> variant | 0.266 | 0.224 | 0.310; 0.356; 0.132 | 0.87 | 0 | 6 of 7 | 1.001 | n/a | +0.027; +0.010; +0.009 | 0.018 |
| read-start, 80,000 windows (unequal) | count | dedup | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.994 | 0.904 | -0.021; +0.026; +0.032 | 0.053 |
| read-start, 80,000 windows (unequal) | count | run type | base -> variant | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.014 | n/a | -0.004; +0.005; +0.000 | 0.009 |
| read-start, 80,000 windows (unequal) | count | run type | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.987 | 0.913 | -0.009; -0.009; -0.006 | 0.003 |
| read-start, 80,000 windows (unequal) | start | dedup | base -> variant | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.001 | n/a | +0.013; +0.008; +0.006 | 0.007 |
| read-start, 80,000 windows (unequal) | start | dedup | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.994 | 0.976 | -0.008; +0.011; +0.014 | 0.022 |
| read-start, 80,000 windows (unequal) | start | run type | base -> variant | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 1.012 | n/a | -0.001; +0.003; -0.000 | 0.004 |
| read-start, 80,000 windows (unequal) | start | run type | variant -> base | n/a | n/a | n/a; n/a; n/a | n/a | 0 | 7 of 7 | 0.989 | 1.006 | -0.004; -0.004; -0.002 | 0.002 |

In the genome-wide row "read-count, count, dedup, base -> variant", seeds 0 and 1 kept ENCSR903SKE and seed 2 excluded all 7 tracks. The score 0.114 is the mean of 2 seeds on 1 track.

## Readings, each compared with the seed range

1. **Read-count model, both covariates: no follow.** Every follow score and every relative deviance change is within or near its seed range. One row moved by a small amount on every seed: run type, base -> variant, top 1%: +0.091, +0.046, +0.029 (seed range 0.062). The largest change is run type, variant -> base, top 1%: -0.157, -0.079, -0.018. All 3 seeds are negative, so the target covariates made the prediction slightly worse. The total ratio stayed within 0.968 to 1.030 on every row. Thus the read-count model leaves its output nearly unchanged when the dedup or run-type covariate is swapped. This agrees with iteration 1, where the dedup prompt left the total unchanged on 15 of 16 targets.
2. **Read-start, 80,000 windows (unequal), dedup, top 1%: a small, consistent move in the right direction.** In both spaces and both directions, the relative deviance change is positive on every seed: count +0.072 to +0.114 (base -> variant) and +0.110 to +0.305 (variant -> base); start +0.060 to +0.080 and +0.060 to +0.181. Each minimum is above 0 and, for base -> variant, above the seed range. The total ratio moved in the right direction: 1.186 to 1.198 when the target is the dedup-off track (more reads), and 0.826 to 0.832 when the target is the dedup track. The pooled follow score where defined was 0.094 to 0.436, far below gate (h). Genome-wide the absolute change was at most 0.032 on every seed.
3. **Read-start, 80,000 windows (unequal), run type: no follow.** Every change is within its seed range or negative on all seeds (base -> variant, top 1%: count -0.093 to -0.006, start -0.068 to -0.001).
4. **Cap law beside the models.** On the variant -> base rows, the cap law closed 0.892 (dedup) and 0.849 (run type) of the gap in start space in the top 1%, and 0.458 and 0.617 in count space. Genome-wide it closed 0.904 to 1.006. No model came near these values on the same pairs. The two metrics have different baselines and floors, so this comparison shows the size of the gap only. A closed-form law with two fitted subsampling rates corrects these pairs. Neither existing model does.

## Stated limits

- Source and target share reads (one library, two processing settings). So the follow score is an upper bound on the correction of an unseen library. A model that fails here fails on an unseen library too.
- The two models are not an equal comparison: the read-start model had 80,000 windows and the read-count model 160,000. The read-start result in reading 2 can therefore not be attributed to the read-start formulation. Chunk A11a tests the 4 factorial arms on equal terms.
- The halves floor is measured at half depth and is above the model's own deviance on most tracks. The follow score as defined is therefore undefined on most tracks. This applies to chunk A11b too, because it uses the same tool.
- 7 tracks, each arm its own 30M-line subsample (50M for DNase). The finite follow scores rest on ENCSR903SKE and ENCSR882CQE only.
- The cap-law gap uses fitted source rates (decision D25), in-sample; `../cf_caps/CF_CAPS.md` reports the sensitivity.

## Reproduce

`bash run.sh` pulls `fir/table.tsv` and `fir/per_track.tsv` from `/project/def-maxwl/mforooz/t396/cf_follow/table/` and writes `table.tsv` and `per_track.tsv` without the leading "# stated limit" line. The Fir side was job 6939785, recorded here because `jobs.tsv` belongs to chunk B4.
