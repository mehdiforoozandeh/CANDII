---
title: What each part of the final miniCANDI recipe adds
kind: paper-section draft (Results)
task: t370
deck: https://claude.ai/artifact/Jhi5EjXAF5d1gRoTktaU97
sources:
  - presentations/t370-minicandi-recipe/data/deck_base_p256_sv.tsv
  - presentations/t370-minicandi-recipe/data/deck_arms_p256_sv.tsv
  - presentations/t370-minicandi-recipe/data/pareto_p256_sv.tsv
  - presentations/t370-minicandi-recipe/data/ens_p256_sv.tsv
---

# What each part of the final miniCANDI recipe adds

==The final miniCANDI is a U-Net with five additions: (1) read-disjoint training pairs, (2) an average-signal input, (3) a Pearson-residual input, (4) twice the training windows (160k), and (5) per-assay dispersion recalibration after training. We scored each step on the eic_v1_2 tables with three objectives: imputation, correction (on 256 regions) and calibration. Every number below is a mean over 3 seeds, with the seed range (max minus min) in brackets.==

==The bar for each objective is the best of the three base models (conv autoencoder, conv + attention, U-Net): imputation 0.445 (U-Net), correction 0.675 (conv + attention), calibration 0.646 (U-Net).==

| model | imputation | correction | calibration | Pareto class |
|---|---|---|---|---|
| U-Net (base) | 0.445 (0.038) | 0.535 (0.062) | 0.646 (0.158) | base |
| + read-disjoint pairs | 0.426 (0.021) | 0.730 (0.071) | 0.687 (0.029) | NEAR |
| pairs + Pearson-residual input | 0.445 (0.064) | 0.774 (0.031) | 0.625 (0.144) | PASS |
| pairs + average-signal input | 0.483 (0.023) | 0.779 (0.010) | 0.608 (0.190) | PASS |
| pairs + average + Pearson | 0.500 (0.035) | 0.818 (0.014) | 0.685 (0.092) | PASS |
| + 2× windows (final) | 0.503 (0.010) | 0.818 (0.016) | 0.717 (0.078) | PASS |

Source: `data/deck_base_p256_sv.tsv`, `data/deck_arms_p256_sv.tsv`; class from `data/pareto_p256_sv.tsv` (pareto_select.py).

## Read-disjoint pairs

==The pairs train the model to predict one half of a track's reads from the other half, so the input and the target share no read. They moved correction from 0.535 to 0.730 (+0.195), past the best base model (0.675). Imputation fell from 0.445 to 0.426 (−0.019), inside the base U-Net's seed range (0.038). Calibration rose from 0.646 to 0.687 and its seed range fell from 0.158 to 0.029. The pairs alone class as NEAR: imputation stays below the bar.==

## Average-signal input

==Added to the pairs model, the average-signal input raised imputation from 0.426 to 0.483 (+0.057) and correction from 0.730 to 0.779 (+0.049). Calibration fell from 0.687 to 0.608 (−0.080), and its seed range grew to 0.190.==

## Pearson-residual input

==Added alone to the pairs model, the Pearson-residual input raised imputation from 0.426 to 0.445 (+0.019) and correction from 0.730 to 0.774 (+0.045). Calibration fell to 0.625 (−0.062), with a seed range of 0.144. Added on top of the average input, it raised all three: imputation 0.483 to 0.500 (+0.017), correction 0.779 to 0.818 (+0.039), calibration 0.608 to 0.685 (+0.078). The two inputs together are the first model that clears all three bars by more than the bar model's seed range on correction.==

## Twice the training windows

==Doubling the windows from 80k to 160k changed the means by less than their seed ranges: imputation +0.004, correction 0.000, calibration +0.032 (range 0.078). The imputation seed range fell from 0.035 to 0.010. The Pareto score rose from 0.249 to 0.451. We picked this version for its tighter imputation and higher score; the two are equal within seed noise. Training cost was 129.6 CPU-h for 3 seeds (Fir jobs 5356071, 5356078, 5356095).==

## Recalibration

==Recalibration multiplies the negative binomial dispersion by one factor per assay, fitted on calibration regions after training. It does not change the mean prediction, so imputation and correction cannot change. On a three-seed ensemble of the pairs + Pearson model, calibration rose from 0.637 to 0.767 (+0.130); imputation stayed at 0.458 and correction at 0.790 (`data/ens_p256_sv.tsv`). Two limits apply. First, the calibration regions share cells and tracks with the scored set, so the gain is an upper estimate. Second, the factors for the final architecture are not fitted yet, so the final model's recalibrated calibration is not measured.==

## Summary

==Each addition has one main job. The pairs carry correction, the average input carries imputation, the Pearson input adds gains only on top of the average input, the extra windows reduce seed variance, and recalibration lifts calibration alone. The final model beats the best base model by +0.058 on imputation, +0.143 on correction and +0.071 on calibration, before recalibration.==
