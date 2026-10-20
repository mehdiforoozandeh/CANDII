---
id: h18
type: idea
schema: 2
title: The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law
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

# h18 — The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The model adds 1 to every count before it reads it, and that could bend how predicted totals scale with depth.

## TL;DR

Design A predicts log μ = a + b·log(1 + X). With the parameters a pure rescale needs (a = ln r, b = 1, r the depth ratio), μ = r(1 + X), so the predicted total overshoots r by log2(1 + 1/m), m the source's mean count per bin. If the pseudocount is the cause, removing it brings never-trained depth pairs onto the law. Design A0 in [[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]] removes it: μ = e^a X^b + e^c, with e^c a learned floor.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

Low-depth sources have a small m, so the pseudocount is large relative to the signal exactly where the never-trained depth pairs start.

## Idea / Hypothesis

The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

[[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]]: design A0 against design A, per-track g, 3 seeds, counts only, no bar.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h18/ and link at least the report:
     - [Report](results/h18/report.md)   - results/h18/curve.png -->
- [Report: the pseudocount overshoot and design A's fitted (a, b) on 504 depth-pair readings](results/h18/report.md)
- results/h18/rows.json every reading with a, b, ln r, m and the overshoot
- results/h18/analyze.py recomputes each predicted scale from the run and the source histogram

## Findings

Observed 2026-10-19, not concluded. Over design A's 504 depth-pair readings (both g versions, 3 seeds), the pure-rescale parameters would overshoot by 0.40–2.96 log2 (1.04–2.96 on never-trained pairs). The observed never-trained miss is about 0.3 log2. The fitted g does not take the pure-rescale parameters: median b is 1.55 on trained pairs and 1.72 on never-trained pairs, and median a − ln r is −1.58 and −1.77.