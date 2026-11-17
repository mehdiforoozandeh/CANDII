# E2: how close can a formula on 25 bp counts get to the pipeline p-value track?

PI question (2026-11-17): "can we make define the formula such that it reproduces the real pipeline exactly or at least get close to it? why aren't we doing that if it's possible".

**PI ruling 2026-11-17: "Adopt the box spread".** `box_d` is now the pileup of every signal route on branch chunk/onion-CO-E2 (`--signal-pileup inbin` keeps the old rule; the form is recorded as `signal_pileup`). `box_d` was chosen among 4 variants on 42 validation targets (41 ChIP-seq, 1 DNase-seq). It is a fitted choice, not a derivation. The adopted code zero-pads the box at region ends and masks the `box_reach(d)` bins at each end from the signal-space metrics; the comparison below zero-padded too and did not mask them (2 to 8 bins of 16,384 per region end).

## 1. Differences between our formula and the pipeline

The pipeline and its sources: `tools/onion_eval/signal_from_counts.py` docstring, `cruxvault/results/t179/ROUTES.md`, `src/candi/store/pval_from_counts.py` docstring (MACS2 v2.2.4 source lines). The count rule: `tools/t112/bin25.py::counts_from_tagalign`. The truth binning: `tools/t112/bin25.py::pval_from_bigwig` (the 25 bp MEAN of the bp-level bigwig; `align_views.py` records it as "-log10 p, MACS2 signal step, 25 bp mean").

| # | pipeline (MACS2 v2.2.4 via ENCODE) | our formula today (`inbin`) | recoverable from 25 bp counts? |
|---|---|---|---|
| 1 | ChIP-seq: each read's 5' end is extended by d bp in its strand direction (`--extsize d --shift 0`); the treatment pileup at a bp is the number of fragments that cover it | a bin's count is the number of READ SPANS (rl bp) that overlap the bin, times d / (rl + 24); there is no extension into neighbouring bins | partly: the fragment's extra d - rl bp can be spread to neighbours with a kernel, but the side is unknown (difference 2) |
| 2 | the extension goes to one side, set by the read's strand | the strand is not in the counts | no: lost at binning |
| 3 | positions at 1 bp | positions at 25 bp; a read is counted in every bin it touches, so adjacent bins share reads | no: the within-bin position is lost |
| 4 | the track is the 25 bp MEAN of the bp-level -log10 p | the score of the bin's mean pileup | approximately: interpolate the pileup inside the bin and average the scores (`box_ext_submean`) |
| 5 | ChIP-seq control lambda: max of lambda_bg and the control at d, 1 kb and 10 kb windows around each bp, scaled by N_treat / N_ctl | the same rule on 25 bp control counts; the d term is the in-bin control count | the 1 kb and 10 kb terms: yes (window means); the d term: partly, as difference 1 |
| 6 | ATAC-seq / DNase-seq: 5' cut site shifted by -75 and extended to 150 bp; llocal 10 kb from the treatment; no control | the same rule with the in-bin pileup count * 150 / (rl + 24) | partly, as difference 1 (the read span is not the cut site) |
| 7 | the p-value is computed on the integer pileup (`int(p)` in `ScoreTrack.pyx`) | floor of the estimated pileup | yes (identical rule; the input differs) |
| 8 | duplicates kept (`--keep-dup all`), same tagAlign | counts from the same filtered tagAlign | yes (same reads), not checked line by line |
| 9 | fragment length d: the pipeline's xcor value or the replicate mean | the table value, else an estimate (32 of 41 ChIP-seq targets) | yes where the table holds the pipeline's value; an estimate can differ |
| 10 | the control file the pipeline used (pooled, possibly subsampled by the pipeline's control rules) | the control row of the tables and its `depth_reads` | not checked |
| 11 | the pipeline's region edges are the chromosome ends | each 2^14-bin eval region is computed alone, so the 1 kb and 10 kb windows are cut at the region edges (2.4% of bins) | yes, with flanks |

Information lost at binning: the strand (2) and the bp positions (3). Both set a floor on the error that no formula on 25 bp counts can remove. Every other difference can be reduced.

## 2. Variants tested (`nb_signal_checks.py formula --pileup`)

Nibi jobs 27595109 to 27595112 (kit COE2d, git 71c947e), about 5 min each on 1 CPU (0.3 CPU-h in total). There were 42 validation targets with a pipeline track: 41 ChIP-seq and 1 DNase-seq. The truth is the pipeline's `signal.h5` (25 bp mean). No bin was non-finite. Files: `formula_ext/<variant>/formula_check.tsv`, `formula_ext/variants_summary.tsv`.

Each value is the median over targets. Peak tiles are 500 bp tiles called at -log10 p >= 2 (the AG2 scorer's pipeline-derived peaks); precision and recall are those of the formula's calls against the pipeline's calls.

| variant | kernel | ChIP Pearson (all 41 / table d 9 / estimated d 32) | ChIP slope | ChIP tile recall / precision / F1 | DNase Pearson | DNase tile F1 |
|---|---|---|---|---|---|---|
| inbin (today) | none | 0.831 / 0.887 / 0.830 | 0.749 | 0.957 / 0.421 / 0.573 | 0.943 | 0.379 |
| box_d | centred box of d bp | **0.921** / 0.944 / 0.918 | **0.962** | 0.758 / **0.853** / **0.782** | **0.994** | **0.857** |
| box_ext | centred box of d - rl bp | 0.908 / 0.929 / 0.902 | 0.917 | 0.821 / 0.726 / 0.752 | 0.990 | 0.845 |
| box_ext_submean | box_ext, mean of the score over 5 positions per bin | 0.909 / 0.935 / 0.907 | 0.938 | 0.800 / 0.761 / 0.763 | 0.991 | 0.849 |

Slope is the least-squares slope of the pipeline on the formula; 1 means no scale bias. The `inbin` row reproduces the earlier check (0.831; `formula/formula_check.json`).

## 3. Reading

- The best variant is `box_d`, i.e. the read-span count spread over a centred box of the fragment length. On ChIP-seq it raised the median Pearson from 0.831 to 0.921, the slope from 0.749 to 0.962 and the peak-tile F1 from 0.573 to 0.782. On the one DNase-seq target it raised Pearson from 0.943 to 0.994.
- Today's in-bin pileup over-calls peaks: tile precision 0.42 at recall 0.96. The cause is that a count is not spread to the bins that its fragment covers, so the pileup is too spiky and too many single bins cross -log10 p = 2. `box_d` brings precision to 0.85, and recall falls to 0.76: the box flattens narrow peaks a little too much.
- `box_d` is wider than the derived `box_ext` kernel (d - rl). One read-span count already spans (rl + 24) / 25 bins. Spreading it by another d makes the total kernel wider than one fragment. That `box_d` still won suggests extra smoothing also absorbs the lost strand and position (differences 2 and 3). This is a fitted choice among 4 variants on 42 targets, not a derivation.
- The bp-mean emulation (`box_ext_submean` against `box_ext`) changed Pearson by +0.001 and F1 by +0.011. The aggregation (difference 4) is a small part of the gap.
- Remaining gap with `box_d`: median Pearson 0.921 (0.944 where d is the pipeline's own value), so about 0.06 to 0.08 below an exact match. The fragment length explains part of it (table d 0.944 against estimated d 0.918). The rest is the strand and position lost at binning (differences 2, 3), which no formula on 25 bp counts can recover. An exact reproduction needs the tagAlign (bp positions and strands), i.e. the pipeline itself. That is possible for observed data, but not for a model that predicts 25 bp counts.

## 4. Model-side version (built)

The model predicts a NB per 25 bp bin, X_k ~ NB(n_k, mu_k), i.e. Poisson(mu_k G_k) with G_k ~ Gamma(n_k, mean 1), marginally. The `box_d` pileup of bin i is a weighted sum Y_i = sum_k w_k X_(i+k) with the box weights w (sum 1), scaled by s = d / (rl + 24).
- Plug-in: the score of floor(s * sum_k w_k mu_(i+k)) against the lambda of the same rule. It needs no assumption about the joint law of the bins.
- Expectation (the headline mode): Y_i gets a NB with mean m = sum_k w_k mu_k and a size n_s that matches the over-dispersion part of its variance. The model gives no covariance between bins, so two bounds are implemented (`--signal-box-dependence`, recorded as `signal_box_dependence`):
  - `comonotone` (default): the G_k move together. The over-dispersion variance is at most (sum_k w_k mu_k / sqrt(n_k))^2 by Cauchy-Schwarz, with equality when every n_k is equal. So n_s = m^2 / (sum_k w_k mu_k / sqrt(n_k))^2. This is the UPPER variance bound. With equal sizes, n_s = n.
  - `independent`: the G_k are independent. The over-dispersion variance is sum_k w_k^2 mu_k^2 / n_k, so n_s = m^2 / sum_k w_k^2 mu_k^2 / n_k. This is the LOWER bound for positively correlated bins.
  - In both forms the Poisson part of the NB is m (one bin's), not sum_k w_k^2 mu_k, because adjacent read-span counts share reads.
  - A unit test checks both forms by simulation (comonotone and independent Gamma-Poisson bins). The reviewer's case (w = 1/2, mu = 10, n = 0.1 and 100) gives an over-dispersion variance of 266.1 (comonotone bound), 250.25 (independent) and 2.0 (the box mean of the sizes, the first version, which was wrong).
- Direction: the -log10 p score is convex in the count, so E[score] RISES with the variance. With mu = 10 and lambda = 5, a NB size of 1000 gives 2.03 and a size of 0.3 gives 5.20. The comonotone default therefore gives the LARGER expected scores, i.e. more confident signal and peaks; the independent form gives the smaller.
- Why comonotone is the default: over-dispersion here is local enrichment, which is smooth at the fragment scale (d is 4 to 16 bins), and adjacent read-span bins share reads. Independence across bins less than one fragment apart is the less plausible end. The truth lies between the two forms, and both will be reported on conv autoencoder seed 0 when it is scored. Calibrating the correlation on observed pseudoreplicates is a separate task.
- Kernel width (residual, not fixed): a read-span count already covers (rl + 24) / 25 bins, and `box_d` spreads it by another d, so the effective kernel is about d + rl wide, wider than one fragment. The narrower "box of d - rl" variant, which corrects for this, tested worse on the 42 targets (ChIP-seq Pearson 0.908 against 0.921; tile F1 0.752 against 0.782).
- Predicted-control route (ChIP-seq, PI comment of 2026-11-17, a headline row beside the observed-control route): the same `box_d` rule spreads the predicted control mean (lambda's d term = box_d of the predicted control pileup, with r = 1 because the control is predicted at the treatment's depth; the 1 kb and 10 kb terms are window means of it). Treatment and control both come from the model, so the p-value and the peaks need no observed control. The scorer applies `box_d` on this route (`signal_pc.npy`, `ctl_pred.npy`).
