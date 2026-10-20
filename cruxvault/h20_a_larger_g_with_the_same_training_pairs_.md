---
id: h20
type: idea
schema: 2
title: A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs
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

# h20 — A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The part of the model that reads the lab settings may be too small to learn how sequencing depth works.

## TL;DR

g is a small MLP (2 × 64) on the source and target covariates. If its size limits it, a wider or deeper g, with the same training pairs and loss, brings never-trained depth pairs onto the law. Raised by the PI on 2026-10-19.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

No run planned yet.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h20/ and link at least the report:
     - [Report](results/h20/report.md)   - results/h20/curve.png -->
_(none yet)_

## Findings

Observed 2026-10-19; the PI has not ruled. No run with a larger g exists. The current g already fits trained depth pairs to a median |log2 error| of 0.04–0.10, and design D, which modulates a convolutional network, also misses never-trained pairs ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]). That pattern leans against size as the cause. A larger g could still learn a map that carries over better.