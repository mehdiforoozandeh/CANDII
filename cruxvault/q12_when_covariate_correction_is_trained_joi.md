---
id: q12
type: question
schema: 3
title: When covariate correction is trained jointly with imputation of masked assays and reconstruction, does the combined objective change how well the correction transfers to covariate combinations never seen in training, compared with training it alone?
parent: q1
created: "2026-10-31T16:09:16"
updated: "2026-10-31T16:09:16"
---

# q12 — When covariate correction is trained jointly with imputation of masked assays and reconstruction, does the combined objective change how well the correction transfers to covariate combinations never seen in training, compared with training it alone?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

When covariate correction is trained jointly with imputation of masked assays and reconstruction, does the combined objective change how well the correction transfers to covariate combinations never seen in training, compared with training it alone?

## Bears on

The testbed ruled out model form only for correction trained alone; if joint training changes transfer, model form and objective design stay open for CANDI, not only data

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**1 children** · open 1, leaning 0, settled 0, parked 0

- `h29` [[h29_adding_a_cross_assay_imputation_or_maske|Adding a cross-assay imputation or masked-bin denoising objective to one shared count-space correction model changes how well the correction transfers to never-trained arm-to-arm pairs, compared with the same model trained on correction alone]] — *open*
<!-- crux:ledger:end -->
