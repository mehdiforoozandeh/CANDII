# Where the validation scale error comes from

Diagnosis of the 30 scored validation tracks (miniCANDI rcp_combo_w2x_unet, seed 0, raw, scored on the mean). No new predictions. Data: chr1, chr9 and chr21 (17.4 M bins, 25 bp, blacklist deleted). Per track, `tools/eic_entry/diag_scale.py` splits MSE / Var(truth) into offset + scale + nonlinear + residual (`tools/eic_decomp/errdecomp.calibrate`). The scored track was recomputed exactly (maximum difference 0 on all 30). Source: `DIAG_TABLE.tsv`, `DIAG_BY_ASSAY.tsv`. Compute: 30 jobs of about 35 s each.

## Result

1. The error is a scale error. nMSE (MSE / Var(truth)) is above 1 on 19 of 30 tracks on these chromosomes. On those 19, the scale term is a median 87 % of the error. With an oracle affine map, nMSE falls below 1 on all 30 (median 2.06 to 0.67). The "oracle" here is the best affine map fitted on the truth; it is a bound only, not a fix. On 20 of 30, the slope ratio SD(pred) / (r SD(truth)) is above 1.3, i.e. predicted peaks are too high for how well they correlate. It is below 0.8 on 6.
2. Depth is not the main cause. The predicted total count is 0.77 to 1.33 times the observed total (`k_depth`). Scaling the prediction to the observed total lowers nMSE on 19 of 30, but leaves most bad tracks above 1 (e.g. H3K4me3 cell 2: 8.72 to 6.95).
3. Lambda is not the cause. For ChIP-seq, lambda comes from the observed control and the depth covariate, so it is the same for the prediction and the observed reads. For the one DNase track, the predicted pileup scored against the observed lambda is slightly worse (0.66 to 0.68).
4. The cause is in the counts, and the -log10 p transform enlarges it. In count space, the predicted mean is over-dispersed on 16 of 30 tracks: the slope ratio of mu against the observed counts is above 1.3. The p-space slope ratio follows the count-space one (log-log r = 0.79, power 1.66). The p-value transform is convex, so a count overshoot at peaks becomes a larger p-value overshoot. 15 of the 19 bad tracks have a count slope above 1.3.
5. On 9 tracks the conversion's own floor is bad. The experiment's own observed reads, through the same conversion, give nMSE above 1 against the 2019 truth, with slope ratio 2.2 to 6.8: C17M29, C18M21, C20M22, C25M21 (nMSE 23.4), C27M24, C31M25, C32M08, C32M20, C34M32. There the store's reads or control differ from what the organizers processed. For 4 of the 19 bad tracks (C17M29, C18M21, C25M21, C31M25) this is the whole cause, because their count slope is below 1.

## Fixes, ranked by expected gain

| fix | tracks it can help | fair source of the fit | cost | bound |
|---|---|---|---|---|
| Per-assay calibration of the predicted mean (shrink toward the mean, in counts or in p-space) | 15 of 19 bad | model predictions on training cells, target masked (in-sample, so it under-corrects), or a cross-fitted fold | predictions on training tracks (GPU, recipe session) + fit and rescore of 30 tracks, about 15 CPU-h | oracle affine: every track below nMSE 1 |
| Match the conversion to the truth's reads: test the native store layer (or the 2019 read depth) on the 9 bad-floor tracks | 9, of which 4 only have this cause | public metadata of the organizers' files | about 1 CPU-h to test on observed counts | the floor's own nMSE |
| Rescale predicted totals to the depth covariate | small | depth covariate (public metadata) | under 1 CPU-h | oracle depth helps only modestly |

A factor tuned on the validation truth is not used and does not count.
