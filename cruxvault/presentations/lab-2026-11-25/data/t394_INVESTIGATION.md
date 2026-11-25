# Why the read-start miniCANDI loses to the read-count version

Task t394, written 2026-11-25 (UTC). This follows the negative result of t392 (`results/t392/SIDE_BY_SIDE.md`). On 16 regions over 3 seeds, the read-start model was lower than the read-count reference on imputation (deviance skill 0.375 vs 0.543), and its deviance calibration ratio was 1.31 vs 1.06 (overconfident). The PI asked for the causes with minimal compute. Compute used: 0.37 CPU-h (Fir jobs 6026174, 6026439, 6026962, 6027889, 6027890, 6027896, 6027897, 6027898). The other checks read code, logs and score files only.

This file is hand-written. The outputs of `runall.sh` are listed in `MANIFEST.tsv`. Three agents did the checks. One measured data on Fir. Two only read code, training logs and store attributes, and wrote no files. Their numbers below are quoted from their reports, with the source they gave.

## The two losses have different causes

The calibration gap and the imputation-skill gap are separate effects. The variance correction that closes most of the calibration gap leaves the skill unchanged, because the skill uses only the predicted mean.

## Ranked candidate causes

| rank | candidate | what it explains | evidence | status |
|---|---|---|---|---|
| 1 | (a) the moment match treats neighbouring start counts as independent, so it understates the converted variance | most of the calibration gap; none of the skill gap | `conv_sparsity/summary_all/table_link.tsv` | supports, weak (the model's own error-correlation length is unmeasured) |
| 2 | (e) training budget or capacity for a noisier per-bin target | possibly much of the skill gap; not tested | learning curves (agent report) | open |
| 3 | (b) sparse start targets and inputs | some of the skill gap at most | `conv_sparsity/summary_pair/table_b.tsv`, `summary_all/per_assay_scores_vs_measures.tsv` | mixed, weak |
| 4 | (c) the loss weights each read once instead of about 2.4 times | little | code reading (agent report) | against, weak |
| 5 | (d) inputs built differently from the reference by mistake | ruled out | code audit (agent report) | against, moderate |

### (a) Conversion independence: explains most of the overconfidence

The `convert_nb` function weights the Poisson part of the start variance by the kernel (binomial thinning), and the overdispersion part by the kernel squared. It adds no covariance between neighbouring bins.
- **Observed half-differences.** pr1 − pr2 of the observed starts measures shot noise only. Its neighbouring-bin covariance was 0.00 ± 0.02 of the predicted variance for single-end data, and 0.22 for ATAC, where mate pairs cause it. Over all 49 measurements, the code's formula predicted the overlap half-difference variance within a ratio of 1.033. So the conversion is correct for pure read noise.
- **Spatially correlated errors.** A model's errors are correlated along the genome. A stand-in predictor made of box-smoothed pr1 starts (width W bins) inflates the overdispersion term by a mean factor c of 2.68 (W = 9), 3.49 (W = 41) and 4.91 (W = 401), on 256 regions.
- **Applied to the model.** Dividing the model's converted dispersion n by c per assay (recomputed on the stored predictions; at c = 1 this matched the scorer exactly) moves the calibration ratio from 1.3081 to 1.1672 (W = 9), 1.1164 (W = 41) and 1.0294 (W = 401). The gap to the reference is 1.3081 − 1.0584 = 0.2497. W = 41 closes (1.3081 − 1.1164) / 0.2497 = 77% of it. W = 9 closes 56%, and W = 401 closes 116%.
- **Limit.** The model's own error-correlation length was not measured. The prediction store holds only the converted mean and dispersion, not the start-space predictions.

### (b) Sparsity: halves replicate agreement, but does not track the per-assay loss

Mean over 49 measurements on the 16 pair regions (`table_b.tsv`, row "ALL"):

| series | zero fraction | mean per bin | r, pr1 vs pr2 |
|---|---|---|---|
| starts, one strand channel | 0.886 | 0.146 | 0.248 |
| overlap counts | 0.621 | 0.756 | 0.499 |

The kernel-convolved starts correlate r = 0.974 with the observed overlap counts. So the starts hold the same information in total; each per-bin target is only noisier. Over the 9 scorable assays, the drop in replicate r does not predict the per-assay skill gap (Spearman ρ = 0.20, p = 0.61).

### Where the skill gap sits

Per assay, mean of 3 seeds (`split/per_assay_deviance_split.tsv`):

| assay | skill, read-start | skill, reference | model deviance, read-start | floor, read-start | scale error, read-start | monotone error, read-start |
|---|---|---|---|---|---|---|
| H3K36me3 | −0.428 | 0.409 | 0.738 | 0.481 | 0.003 | 0.046 |
| H3K79me2 | 0.500 | 0.754 | 0.701 | 0.399 | 0.039 | 0.134 |
| H3K4me3 | 0.434 | 0.635 | 0.669 | 0.515 | 0.005 | 0.083 |
| H3K27ac | 0.516 | 0.317 | 0.712 | 0.582 | 0.002 | 0.050 |

- **H3K36me3 dominates.** Its gap of 0.837 is 0.837 / 9 = 0.093 of the 0.168 macro gap.
- **The read-start model is worse than the position average on H3K36me3.** Its deviance is 0.738, against 0.661 for the per-position training average. H3K36me3 is a broad mark over gene bodies.
- **The error is in the shape.** The scale error (0.003) and the monotone error (0.046) are small, so a recalibration would not fix H3K36me3.
- **Not uniform.** The read-start model is better on H3K27ac (0.516 vs 0.317).

### (c) Loss weighting: little

Notes from the code reading:
- Both models minimise a per-entry NB negative log-likelihood mean, on the same code path, with the same learning-rate schedule, batch size and window budget (`tools/minicandi/read_start.py:147-192`, `train.py:57-58`).
- Both strands are equal. A read covers about 2.4 bins: the kernel row sums were 2.44 in 6 of 9 sampled files, and up to 4.04 in 2.
- The read-start gradients are about 3 times smaller. AdamW, the weight-update rule, is invariant to that scale.

### (e) Training budget or capacity: open

Notes from the learning curves:
- The parameter counts are equal: 993,695 vs 993,615.
- The held-out loss fell by 5.8% (read-start) and 7.1% (reference) from step 1,000 to the end.
- Both curves are nearly flat over the last 20% (−0.27% and −0.22%). But the learning rate has decayed by then, so this does not exclude under-training.
- The train vs held-out gap is somewhat larger for read-start (+35% vs +27%), with batch noise.

### (d) Inputs: no unintended difference

Results of the code audit:
- Inputs, the average input and the targets are all read starts (D16). The strands are a last axis, passed through `Linear(2, D)`. Training and scoring read them the same way.
- The read-start average is built with the same rule as the reference average (starts × 1e6 / depth, leave-one-cell-out), from the same measurement list.
- The Pearson input transform fits its statistics on the start inputs, not on overlap tables.
- The depth covariate is in reads.
- Masking uses the parent's mask.

One intended difference favours the reference on correction: masked bins of overlap counts leak through neighbouring bins, and masked starts do not.

## The cheapest test that separates the top two causes of the skill loss

The top two candidates for the skill gap are (e), too little training or capacity for a noisier target, and (b), an information loss from sparse starts. Candidate (b) is already weakened, because the summed starts carry the same information (r = 0.974). The cheapest separating test therefore asks whether the read-start model improves with more training.

**Proposed test:** one read-start seed trained to 160,000 windows. That is twice the budget, and the same doubling that the final read-count recipe uses. Estimated cost: about 260 CPU-h. If its H3K36me3 and macro skill approach the reference, budget is the cause. If they stay flat, the cause is in the target or the architecture.

**A cheaper first step, about 1 CPU-h, no training.** The seed-0 training state is kept. Score the same checkpoint with its start-space predictions box-smoothed before conversion (W = 9 and 41). If smoothing alone recovers much of the H3K36me3 skill, the model has the signal but has not learned to smooth it, which points to (e). If it does not, (e) stays open. This step also gives the model's own error-correlation length, which pins down the calibration share of (a).

Neither test is started. Both need the PI's word.
