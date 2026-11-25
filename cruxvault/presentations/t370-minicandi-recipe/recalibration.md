---
title: Dispersion recalibration of the final miniCANDI, and how well its intervals cover large peaks
kind: paper-section draft (Results)
task: t370
sources:
  - results/t370/recal_study/recal_study.md
  - results/t370/recal_study/peak_coverage_ref.md
  - results/t370/recal_study/recal_study.pair_factors.json
---

# Dispersion recalibration, and interval coverage of large peaks

==Model: the final miniCANDI (U-Net + read-disjoint pairs + average and Pearson inputs + 160k windows), 3 seeds, data eic_v1_2. Recalibration divides the negative binomial (NB) size n by one factor c per assay (n' = n / c; variance mu + mu²/n'). It never changes the mean mu, so imputation and correction scores cannot change. We fitted c on 32 held-out regions and scored on 32 other held-out regions, which share no region with the fit set. The fit picks the c that minimizes the PIT deviation, i.e. the distance of the probability-integral-transform histogram from flat. Lower is better, and 0 is perfectly calibrated.==

## Why recalibrate, and what a user can fit

==A user who imputes an assay in a new cell has no truth for that assay in that cell, so a factor for an imputed track can only come from other cells. A user who corrects an observed track has that track, so a factor can come from the same cell. We therefore measured two fits. (1) The same-cell fit uses the same cells and tracks as the score set, in other regions. (2) The deployment fit is a 2-fold cross-fit over cells: each cell is scored with a factor fitted on the other half of the cells. For imputation, the deployment fit is the honest number. For correction, the same-cell fit is reachable in practice.==

## Calibration before and after

PIT deviation, mean over 22 assays, then mean over 3 seeds (seed values in brackets).

| kind | none | same-cell fit | deployment fit (other cells) | correction factor applied to imputation |
|---|---|---|---|---|
| imputation | 0.0200 (0.0246, 0.0170, 0.0183) | 0.0150 (0.0198, 0.0110, 0.0142) | 0.0185 (0.0235, 0.0144, 0.0176) | 0.0191 |
| correction | 0.0089 (0.0078, 0.0084, 0.0105) | 0.0066 (0.0059, 0.0059, 0.0079) | 0.0078 (0.0071, 0.0074, 0.0089) | n/a |

Source: `results/t370/recal_study/recal_study.md`.

==Every seed improves under every fit. For imputation, the deployment fit lowers the PIT deviation by 7.5% (0.0200 to 0.0185). This gain is smaller than the range over seeds (0.0076 with no fit). The same-cell fit lowers it by 25%, but that number is an upper bound that no user can reach for an imputed track. For correction, the same-cell fit lowers it by 26% (0.0089 to 0.0066), and a user can reach this gain.==

==Three limits apply. First, the factors are unstable across seeds: for H3K9me3 they are 0.394, 1.83 and 0.0178. Second, 6 assays reach the Poisson floor in at least one seed (c at the grid cap, log2 −16.5). Their predicted variance cannot fall further, because an NB distribution cannot be narrower than a Poisson distribution. Third, under the deployment fit, 11 imputation entries keep c = 1, because their assay has a held-out track in one cell only.==

==On the EIC validation tracks, the wide-grid recalibration of seed 0 raised the expected-p-value MSE on 7 of 8 tracks, for example C02M22 from 9.5 to 16.1 (pval session, `results/t390/val_recal_pair_s0/` on PR 64). Recalibration therefore improves calibration, but it can hurt a mean-based score that is computed from the predicted distribution.==

## Do the intervals capture large peaks?

==The PI asked whether our intervals capture large peaks even where the mean misses them. We measured the coverage of central NB intervals, i.e. the fraction of bins whose observed count lies inside the interval. We split bins by the size of the observed count and by the size of the predicted mean. Choosing bins by their observed value lowers coverage even for a perfectly calibrated model, because the bins with the highest observed values are mostly upper-tail draws. We therefore added a calibrated reference: each bin's count is replaced by a draw from its own predicted NB, and the same strata are applied. The gap between our coverage and the reference is the real miss.==

Imputation, 90% central interval, mean over assays and seeds; coverage (fraction above the upper bound / below the lower bound).

| bins | ours, no recalibration | ours, deployment fit | calibrated reference |
|---|---|---|---|
| all | 0.963 (0.034 / 0.003) | 0.963 (0.033 / 0.003) | 0.973 (0.025 / 0.002) |
| observed top 10% | 0.840 (0.159 / 0.001) | 0.840 (0.157 / 0.003) | 0.857 (0.143 / 0.000) |
| observed top 1% | 0.669 (0.329 / 0.002) | 0.677 (0.317 / 0.007) | 0.710 (0.290 / 0.000) |
| observed top 0.1% | 0.576 (0.422 / 0.001) | 0.605 (0.390 / 0.005) | 0.554 (0.446 / 0.000) |
| predicted top 1% | 0.891 (0.047 / 0.062) | 0.880 (0.039 / 0.081) | 0.926 (0.045 / 0.029) |

Correction pairs, 90% interval: observed top 1%, ours 0.579, reference 0.599; predicted top 1%, ours 0.927, reference 0.951. Source: `results/t370/recal_study/peak_coverage_ref.md`.

==Our intervals capture the largest observed peaks almost as well as a calibrated model can. At the top 1% of observed bins, ours cover 0.669 and the reference covers 0.710. At the top 0.1%, ours cover 0.576 and the reference covers 0.554. Most of the apparent miss at large peaks is therefore the selection effect, not a defect of the model. The real defect is at the bins where we predict the largest values. There, the observed count falls below the lower bound in 6.2% of bins, against 2.9% for the calibrated reference. This is the overshoot that the EIC board shows, and in interval terms it is mild. Recalibration does not change either result.==

## What this means for the paper

==(1) Report calibration of imputed tracks with the deployment fit, and of corrected tracks with the same-cell fit. (2) Report the recalibration gain for imputation as small and inside seed noise. (3) Report that the intervals capture large observed peaks at close to the calibrated rate. (4) Report that the mean overshoots where it predicts the largest values. The PI rejected a fix that shrinks the predicted peaks. A fix belongs in training, e.g. a better model of large peaks, and is not yet tested.==
