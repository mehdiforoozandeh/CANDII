---
id: h22
type: idea
schema: 2
title: A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training
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

# h22 — A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

If the model maps every run to a shared middle ground and back, it can connect any two runs, even a pair it never saw.

## TL;DR

One network φ maps one side's covariates c to (α(c), β(c)). The source is encoded as z = (log(1 + X) − α(c)) / β(c) and decoded as log μ = α(c′) + β(c′)·z, so a never-trained arm → arm pair is the composition of two trained legs by construction. If composition is what is missing, this design brings never-trained depth pairs, including the 7.5M arm held out of training, to the error of trained pairs, while the retrained old design A does not.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

[[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder design A task]]: β learned and β fixed at 1 (a pure shift), against the retrained old design A; counts only, both g versions, 3 seeds, each with a no-covariates twin; no bar. Design: `plan/T126_ENCDEC_A.md`.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h22/ and link at least the report:
     - [Report](results/h22/report.md)   - results/h22/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
