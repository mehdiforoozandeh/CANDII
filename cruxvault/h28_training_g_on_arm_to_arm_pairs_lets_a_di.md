---
id: h28
type: idea
schema: 2
title: Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base
parent: q7
status: idea
rule:
measurement:
replicates:
verdict:
metric:
created: 2026-10-29T20:07:43
updated: 2026-10-29T20:07:43
---

# h28 — Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

If the translator also practises on pairs where neither side is the original track, can it translate between two altered tracks directly, without a detour through the original?

## TL;DR

The direct g → f designs fit trained base ↔ arm depth pairs and miss the depth law on never-trained arm → arm pairs, while chaining two trained maps through the base meets it (0.07–0.12 against 0.29–0.34 direct). That fits two readings: g needs the base as a pivot, or g simply never saw an arm → arm pair. This claims the second: trained on some arm → arm pairs as well, a direct map meets the law on held-out arm → arm pairs. It decides what CANDI needs, since CANDI has a natural base only for depth (the full-depth track): if this holds, augmenting CANDI's corpus with counterfactual pairs is the route; if it fails, the pivot is. Settled by the median |log2 error| of the predicted scale on held-out arm → arm depth pairs against the via-base chain and the seed floor.

## Null

<!-- spec 09: the null hypothesis — the most parsimonious alternative explanation under
     which this result appears although the claim is false. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the null hypothesis, the most parsimonious alternative explanation — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

Arm of the PI's 2026-10-29 ruling on the unseen-pair gap: run beside the t135 pivot seeds, with the pivot as the default reading. Augmenting CANDI (by BAM or FASTQ re-derivation) is deferred until this and t135 say whether the architecture generalizes without it.

## Idea / Hypothesis

Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h28/ and link at least the report:
     - [Report](results/h28/report.md)   - results/h28/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
