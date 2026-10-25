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

g is a small MLP (2 × 64) on the source and target covariates. If its size limits it, a wider or deeper g, with the same training pairs and loss, brings never-trained depth pairs onto the law. Raised by the PI on 2026-10-19. No larger-g run exists; the readings so far lean against size as the cause.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

Every design misses the depth law on never-trained arm → arm pairs ([[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law question]]). Capacity is the cheapest suspect to rule out before a structural fix, because a 2 × 64 g may be too small to represent the depth ratio as a rule.

## Idea / Hypothesis

A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

No run is planned yet.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h20/ and link at least the report:
     - [Report](results/h20/report.md)   - results/h20/curve.png -->
No larger-g run exists. The readings below come from:
- [Trained versus never-trained report](results/h19/report.md)
- [Encoder-decoder depth table](results/t126/depth_table.tsv)
- [Via-base depth table (arm Dviabase)](results/t128/depth_table.tsv)

## Findings

Exploratory readings; the PI has not ruled. No run with a larger g exists. Three readings lean against size as the cause, without testing it:

- The current g fits trained depth pairs to a median |log2 error| of 0.04–0.10, in every design ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]).
- Design D, which modulates a convolutional network, and its encoder-decoder variants also miss never-trained pairs (0.26–0.52).
- The same 2 × 64 g meets the law when its two trained maps run in series through the base (0.07–0.12), so it already holds the depth ratio for each trained leg.

A larger g could still learn a map that carries over better.
