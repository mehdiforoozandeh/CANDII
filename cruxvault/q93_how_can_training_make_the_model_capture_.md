---
id: q93
type: question
schema: 3
title: How can training make the model capture large peaks accurately, without a post-hoc shrink of the predicted peaks?
parent: q79
created: "2026-11-26T04:25:13"
updated: "2026-11-26T04:25:13"
---

# q93 — How can training make the model capture large peaks accurately, without a post-hoc shrink of the predicted peaks?

Parent:: [[q79_do_the_final_minicandi_s_negative_binomi]]

## ELI5

The model's predicted mean is too high at the regions where it predicts the largest signal; the PI wants this fixed in training, not by shrinking the predictions afterwards.

## TL;DR

The PI said (2026-11-25): "i don't like the peak shrink factor idea. we must find a way to more accurately capture large peaks. or at least we should show that our confidence intervals capture the large peaks even our mean doesnt". Evidence so far: at the bins with the top 1% of predicted means, the observed count falls below the 90% interval in 6.2% of bins, against 2.9% for a calibrated reference; at the top 1% of observed bins, coverage is 0.669 against 0.710 for the reference (results/t370/recal_study/peak_coverage_ref.md). The count overshoot on the EIC validation tracks is in results/t390/val_scale_diag on implementation/t390-eic-pvalue-entry. Settled when a training change is tested that lowers the top-predicted overshoot without loss of imputation, correction or calibration, or when the interval-coverage defence is accepted as the reported result.

## Question

How can training make the model capture large peaks accurately, without a post-hoc shrink of the predicted peaks?

## Bears on

The parent found that intervals cover large observed peaks near the calibrated rate, but the mean overshoots where it predicts the largest values; the PI rejected a peak shrink factor, so an answer here decides whether the overshoot is fixed in training or reported as a limit with interval coverage as the defence.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
