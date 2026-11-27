# Smoothing test of the read-start miniCANDI (seed 0)

Task t395, written 2026-11-25 (UTC). It follows the investigation of t394 (`../t394/INVESTIGATION.md`), which proposed this test as the cheapest step before a longer training run. Compute: 3.53 CPU-h (Fir jobs 6033442, 6034117-20, 6040094; sacct CPUTimeRAW 12,704 s). This file is hand-written; the outputs of `runall.sh` are listed in `MANIFEST.tsv`.

## Design

- **Model and targets.** The seed-0 read-start model (scoring kit RS7) on 9 leave-one-assay-out targets: 3 each of H3K36me3, H3K4me3 and H3K27ac, over the 16 pair regions (262,144 bins per target).
- **Smoothing.** A copy of `ReadStartAdapter.forward` smooths the start-space mean within each stitched run, before conversion. The smoothing is a centered moving mean of W bins per strand, which keeps the total mass. The dispersion is unchanged. Then `read_start.convert_nb` converts as in scoring.
- **Scoring.** `Scorer._imputation_group` scored W = 1 live, and W = 9 and 41 by replay. So the position-average, floor and baseline terms are the scorer's own.
- **Error-correlation length.** The autocorrelation of the start-space residual (observed starts minus predicted start mean), normalized by the model's own overdispersion (sum of mu² / n). Lags are in 25 bp bins.
- **c, the variance inflation.** c = 1 + (summed off-diagonal residual covariance through the kernel) / (the model's own overdispersion summed with kernel²). The t394 formula was unstable on model residuals, because its denominator was −3.4 to 27.4 (`c_linkformula`).

## Reproduction check

At W = 1, the live converted mean was bit-identical (float32) to the stored prediction on all 9 targets. On all 256 regions, the stored predictions reproduced the scorer's seed-0 per-assay skill to every printed digit: H3K36me3 −0.3618, H3K4me3 0.4464, H3K27ac 0.5647 (`summary/table_skill.tsv`, columns `repro_store_256` and `scorer_rs_seed0_256`).

## Result

`summary/table_skill.tsv`. Skill is the deviance imputation skill; the 3 values per cell in the last four columns are the 3 targets.

| assay | skill W = 1 / 9 / 41 (16 regions) | skill, 256 regions, read-start / reference (seed 0) | lag where correlation < 0.5 | lag where correlation < 0.1 | c | predicted / observed total starts |
|---|---|---|---|---|---|---|
| H3K36me3 | −0.070 / −0.104 / −0.347 | −0.362 / 0.391 | 274, 123, >400 | >400, >400, >400 | 11.1, 5.2, 11.1 | 1.15, 1.28, 1.06 |
| H3K4me3 | 0.475 / 0.469 / 0.410 | 0.446 / 0.707 | 15, 69, 69 | 175, 174, 114 | 3.9, 17.6, 24.3 | 0.93, 1.02, 1.08 |
| H3K27ac | 0.648 / 0.645 / 0.612 | 0.565 / 0.202 | 4, 56, 2 | 170, >400, 31 | 4.8, 4.4, 3.4 | 0.99, 0.91, 0.91 |

The 16-region skills are noisy: H3K36me3 at W = 1 is −0.070 on 16 regions and −0.362 on 256. Compare values only within the 16-region columns.

**Smoothing does not recover skill.** H3K36me3 falls from −0.070 (W = 1) to −0.104 (W = 9) and −0.347 (W = 41). H3K4me3 and H3K27ac also fall slightly. So the loss is not per-bin noise in the start-space mean.

**The H3K36me3 error is a broad level error.** Its residuals stay correlated past 400 bins (10 kb); the correlation at lag 400 was 0.47, 0.27 and 0.54. The model overpredicts the total starts on these regions by 6% to 28%. H3K36me3 marks gene bodies over tens of kb, so this is a scale error over broad domains, local to the windows. The scorer's per-assay scale error stays small (0.003, `../t394/INVESTIGATION.md`), because it fits one scale over the whole track.

## Calibration ratio after n → n / c

`out/calib_seed0_table.json`, the stored seed-0 predictions of all 43 targets over 256 regions. Each measured target used its own c. The other 6 scorable assays used the mean over the 9 measured targets, 9.53.

| | macro | H3K36me3 | H3K4me3 | H3K27ac |
|---|---|---|---|---|
| read-start, as scored | 1.2485 | 1.400 | 1.284 | 1.141 |
| read-start, n / c | 0.8751 | 1.071 | 0.910 | 0.959 |
| reference, seed 0 | 1.0679 | | | |

The measured c over-corrects: 0.8751 is below 1. Part of the neighbour covariance is a long-range error in the mean, not variance. Several values of c exceed the bound of about 5 to 7 for fully correlated noise. So the independence assumption in the conversion is large enough to explain the whole calibration gap, and the measured c is an upper bound on its share.

## What this means for the open causes

- **Training budget (e).** This test does not point to noise from under-training. It does not exclude a model that has not learned the broad H3K36me3 domains; only a longer run tests that.
- **Conversion independence (a).** The model's errors are correlated over 2 to more than 400 bins, so the conversion understates the converted variance. The size bound now covers the whole calibration gap.
- **H3K36me3.** The failure is a broad level error in the predicted starts, not a per-bin shape error that smoothing can remove.
