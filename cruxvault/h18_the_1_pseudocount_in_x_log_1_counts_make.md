---
id: h18
type: idea
schema: 2
title: The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law
parent: q7
status: done
rule: all
measurement: 
replicates: 
verdict: refuted
metric: "never-trained depth miss, rung A0 (no +1): 0.46 (NB) / 0.38 (Poisson-mean) median |log2 err| vs trained 0.12"
created: "2026-10-19T23:41:09"
updated: "2026-10-20T18:52:04"
lock: 99cd2a93a9c09a4e
locked: "2026-10-20T18:52:04"
lock_at: close
---

# h18 — The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The model adds 1 to every count before it reads it, and that could bend how predicted totals scale with depth.

## TL;DR

Design A predicts log μ = a + b·log(1 + X). With the parameters a pure rescale needs (a = ln r, b = 1, r the depth ratio), μ = r(1 + X), so the predicted total overshoots r by log2(1 + 1/m), m the source's mean count per bin. If the pseudocount is the cause, removing it brings never-trained depth pairs onto the law. Design A0 in [[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]] removes it: μ = e^a X^b + e^c, with e^c a learned floor. Rung A0 still misses never-trained pairs, so the +1 is not the cause.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Chance: a single seed or track happens to land near the law; the median over 3 seeds and 7 tracks guards against it.

## Problem Statement

Low-depth sources have a small m, so the pseudocount is large relative to the signal exactly where the never-trained depth pairs start.

## Idea / Hypothesis

The +1 pseudocount in x = log(1 + counts) makes the g → f designs miss the depth law

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
Checks written after the run, at the PI's request on 2026-10-20; the work was exploratory with no bar set beforehand.

- [ ] Dropping the +1 (rung A0, μ = e^a·X^b + e^c) brings the never-trained depth → depth median |log2 error| down to the trained-pair level (about 0.1) under either loss
      fails-if:: Rung A0's never-trained median |log2 error| stays well above its trained-pair level
      (found: never-trained 0.46 with the NB loss, wobble 0.16, and 0.38 with the Poisson-mean loss, wobble 0.07; trained 0.12 — [report](results/t127/report.md))
- [x] [outcome-neutral] The target tracks themselves obey the depth law on the scored bins
      fails-if:: Σy/ΣX differs from the manifest depth ratio by more than 0.01 log2
      (found: within 0.004 log2)

## Planned Intervention

[[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]]: design A0 against design A, per-track g, 3 seeds, counts only, no bar.

## Run Links

- Loss head-to-head runs (Nibi, not Fir): `/project/def-maxwl/mforooz/t118/ppml_vs_nb/runs/<run_name>/`, 126 runs ([task table](results/t127/tasks.tsv)).
- Training and scoring: array 24645902, retries 24646347 and 24647920; kit `/project/def-maxwl/mforooz/t118/ppml_vs_nb/code/K2` (git 2d6b6b0).
- Analysis: `/project/def-maxwl/mforooz/t118/ppml_vs_nb/analysis/` (extra metrics job 24652327) ([run locations](results/t127/FIR_PATH.txt)).

## Artifacts

<!-- what the run produced. Keep files under results/h18/ and link at least the report:
     - [Report](results/h18/report.md)   - results/h18/curve.png -->
- [Report: the pseudocount overshoot and design A's fitted (a, b) on 504 depth-pair readings](results/h18/report.md)
- results/h18/rows.json every reading with a, b, ln r, m and the overshoot
- results/h18/analyze.py recomputes each predicted scale from the run and the source histogram
- [Loss head-to-head report](results/t127/report.md)
- [Lab deck: Poisson-mean vs full NB loss](presentations/t127-ppml-vs-nb/index.html) — built 2026-10-20; no published artifact URL was found

## Findings

Observed 2026-10-19. Over design A's 504 depth-pair readings, the pure-rescale parameters would overshoot by 0.40–2.96 log2, far more than the observed never-trained miss (about 0.3). The fitted g avoids them: median b is 1.55 (trained) and 1.72 (never-trained), and median a − ln r is −1.58 and −1.77.

Observed 2026-10-20 in the loss head-to-head ([report](results/t127/report.md)): rung A0, which drops the +1 and can express the law exactly (μ = e^a·X^b + e^c), does not meet it either. Never-trained depth → depth median |log2 err| is 0.46 (NB loss, wobble 0.16) and 0.38 (Poisson-mean loss, wobble 0.07), against rung A's 0.33 and 0.29; trained pairs fit (0.12). So removing the +1 does not fix the miss on never-trained pairs. PI understood and agreed 2026-10-20 that neither A nor A0's g learns the depth ratio as a rule.

Why not supported (PI, 2026-10-20): the +1 distorts rung A's fitted parameters, but removing it does not close the depth-law miss. Rung A0 can express the law exactly and still misses never-trained depth pairs by 0.38–0.46 log2, as much as or more than rung A. The +1 is therefore not the cause; the miss comes from g failing to carry the trained fit over to never-trained pairs.
