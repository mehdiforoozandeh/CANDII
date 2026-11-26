---
id: q92
type: question
schema: 3
title: Which output covariates make sense as prompts for an imputed experiment that was never done?
parent: q80
created: "2026-11-26T04:25:13"
updated: "2026-11-26T04:25:13"
---

# q92 — Which output covariates make sense as prompts for an imputed experiment that was never done?

Parent:: [[q80_does_the_final_minicandi_follow_a_change]]

## ELI5

To impute an experiment that was never done, we must tell the model how that experiment would have been run; some settings (read depth) can be chosen freely, while others (data quality, lab) have no true value.

## TL;DR

The PI asked (2026-11-25): "some of these covariates, especially in the target side, must make sense for an imputed assay before imputing something. Do we know its frip score? Do we know its quality? ... Does prompting make sense for all of the covariates that we have right now in mini candy, or does it work just for a subset". In the final miniCANDI the output side holds the 11 metadata columns only (tools/minicandi/contract.py META_COLUMNS; run.json qc_on_input false), so FRiP and other QC scores are not prompts today. The covariate-prompt test (results/t370/recal_study/covariate_prompt_acetyl.md, covariate_prompt_methyl.md) found that the model follows depth (16/16 targets) and read length (12/16), and that an unknown lab, platform or antibody does not widen the predicted intervals (rule met in 9 of 46 cases). Settled when each output covariate is classed as user-choosable, inferable from the inputs, or unknowable for a never-done experiment, with the rule for each class tested.

## Question

Which output covariates make sense as prompts for an imputed experiment that was never done?

## Bears on

If only a subset of the output covariates (e.g. depth, read length) has a meaningful value for a never-done experiment, the covariate-prompt result in the parent holds only for that subset, and quality-type covariates (FRiP, lab, antibody) need a default or must be dropped from the output side.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
