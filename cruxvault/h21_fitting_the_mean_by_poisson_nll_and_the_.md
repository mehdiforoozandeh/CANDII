---
id: h21
type: idea
schema: 2
title: Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not
parent: q7
status: idea
rule:
measurement:
replicates:
verdict:
metric:
created: 2026-10-19T23:41:09
updated: 2026-10-19T23:41:09
---

# h21 — Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The score we train on lets predicted totals drift; a different score forces the totals to match.

## TL;DR

With a log link and a free intercept, the Poisson NLL's optimum sets the predicted total equal to the observed total on the training pairs. The negative binomial NLL balances them only with per-bin weights n/(n + μ), so totals can drift. If this drift causes the miss, a model whose mean is trained by Poisson NLL and whose dispersion is trained by negative binomial NLL meets the depth law on never-trained depth pairs where the full negative binomial model does not. If the miss is a failure to carry over to never-trained pairs instead, fixing totals on trained pairs leaves never-trained pairs as they are.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

[[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]]: designs A, A0 (no pseudocount) and D, per-track g, 3 seeds, counts only, no bar.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h21/ and link at least the report:
     - [Report](results/h21/report.md)   - results/h21/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
