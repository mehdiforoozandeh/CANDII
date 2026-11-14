---
id: h40
type: idea
schema: 3
title: scrambled_covariates lowers the correction term by more than 2 seed spreads and leaves imputation within 2; dna_off lowers imputation
parent: q23
rule: 
measurement: Screening scores (pairs on 16 regions) of the rung-1 arms scrambled_covariates and dna_off against base, 3 seeds each; each arm minus base on each goal term against 2 seed spreads.
replicates: 3 seeds (0, 1, 2) per arm
created: "2026-11-14T00:35:31"
updated: "2026-11-14T00:35:31"
---

# h40 — scrambled_covariates lowers the correction term by more than 2 seed spreads and leaves imputation within 2; dna_off lowers imputation

Parent:: [[q23_is_the_model_wired_as_designed_does_scra]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

scrambled_covariates lowers the correction term by more than 2 seed spreads and leaves imputation within 2; dna_off lowers imputation.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

If it holds, the model uses the covariates for correction and the DNA sequence for imputation, as designed.

## Idea / Hypothesis

scrambled_covariates lowers the correction term by more than 2 seed spreads and leaves imputation within 2; dna_off lowers imputation

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] scrambled_covariates minus base on the correction term is worse by more than 2 seed spreads.
- [ ] scrambled_covariates minus base on the imputation term is within 2 seed spreads.
- [ ] dna_off minus base on the imputation term is worse.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h40/ and link at least the report:
     - [Report](results/h40/report.md)   - results/h40/curve.png -->
_(none yet)_
