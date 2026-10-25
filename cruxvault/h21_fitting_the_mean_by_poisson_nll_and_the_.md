---
id: h21
type: idea
schema: 2
title: Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not
parent: q7
status: done
rule: all
measurement: 
replicates: 
verdict: refuted
metric: "never-trained depth miss with Poisson-mean loss: A0 0.38, D 0.28 median |log2 err| vs trained 0.07–0.12"
created: "2026-10-19T23:41:09"
updated: "2026-10-25T00:58:40"
lock: 1cdab54a770689f7
locked: "2026-10-20T18:52:04"
lock_at: close
null_approved: "2026-10-25T00:58:40"
null_hash: 24641e914971592d
---

# h21 — Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The score we train on lets predicted totals drift; a different score forces the totals to match.

## TL;DR

With a log link and a free intercept, the Poisson NLL's optimum sets the predicted total equal to the observed total on the training pairs. The negative binomial NLL balances them only with per-bin weights n/(n + μ), so totals can drift. If this drift causes the miss, a model whose mean is trained by Poisson NLL and whose dispersion is trained by negative binomial NLL meets the depth law on never-trained depth pairs where the full negative binomial model does not. If the miss is a failure to carry over to never-trained pairs instead, fixing totals on trained pairs leaves never-trained pairs as they are. The run fixed rung A's totals and left the never-trained miss.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Chance: a single seed or track happens to land near the law; the median over 3 seeds and 7 tracks guards against it.

## Problem Statement

Rung A over-predicts totals on some pairs (DNase Σμ/Σy ≈ 13–15 on dedup → base), and the NB loss does not force predicted totals to match. The PI (2026-10-19) asked whether that drift is what makes every design miss the depth law ([[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law question]]).

## Idea / Hypothesis

Fitting the mean by Poisson NLL and the dispersion by negative binomial NLL meets the depth law where the full negative binomial NLL does not

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
Checks written after the run, at the PI's request on 2026-10-20; the work was exploratory with no bar set beforehand.

- [ ] With the Poisson-NLL mean, the never-trained depth → depth median |log2 error| falls to the trained-pair level (about 0.1) in at least one rung that can express the law (A0 or D)
      fails-if:: Both A0 and D stay well above their trained-pair level with the Poisson-mean loss
      (found: A0 0.38, wobble 0.07; D 0.28, wobble 0.09; trained 0.07–0.12 — [report](results/t127/report.md))
- [x] [outcome-neutral] The target tracks themselves obey the depth law on the scored bins
      fails-if:: Σy/ΣX differs from the manifest depth ratio by more than 0.01 log2
      (found: within 0.004 log2)

## Planned Intervention

[[t127_head_to_head_poisson_nll_mean_nb_nl|the loss head-to-head task]]: designs A, A0 (no pseudocount) and D, per-track g, 3 seeds, counts only, no bar.

## Run Links

- Loss head-to-head runs (Nibi, not Fir): `/project/def-maxwl/mforooz/t118/ppml_vs_nb/runs/<run_name>/`, 126 runs ([task table](results/t127/tasks.tsv)).
- Training and scoring: array 24645902, retries 24646347 and 24647920; kit `/project/def-maxwl/mforooz/t118/ppml_vs_nb/code/K2` (git 2d6b6b0).
- Analysis: `/project/def-maxwl/mforooz/t118/ppml_vs_nb/analysis/` (extra metrics job 24652327) ([run locations](results/t127/FIR_PATH.txt)).

## Artifacts

<!-- what the run produced. Keep files under results/h21/ and link at least the report:
     - [Report](results/h21/report.md)   - results/h21/curve.png -->
- [Poisson-mean vs NB head-to-head report](results/t127/report.md)
- results/t127/report.pdf figures for every comparison aspect
- [Lab deck: Poisson-mean vs full NB loss](presentations/t127-ppml-vs-nb/DECK.md) — published https://claude.ai/artifact/3QpVJzhgeZBwfSXu19y5Yj

## Findings

The head-to-head run (126 runs: rungs A, A0, D × nb/ppml × 7 tracks × 3 seeds) did not fix the depth
law. Never-trained depth → depth median |log2 err| moved A 0.33 → 0.29 (wobble 0.05), A0 0.46 → 0.38
(0.16), D 0.27 → 0.28 (0.13), and the worst-pair check stayed at 0.46–0.68 in every arm. The loss did
remove rung A's total-scale bias (scale_error 0.253 → 0.005, wobble 0.025), and it left NB likelihood
and PIT calibration unchanged. PI reading (2026-10-20): the Poisson-mean loss fixes rung A's scale
bias but not the depth law, so it is not the cure. The box stays unticked; closing is the PI's call.

Why not supported (PI, 2026-10-20): the Poisson-mean loss balances totals on the trained pairs, which is what it guarantees, but the depth-law miss is on never-trained pairs. There g still has to produce a ratio it never saw, and changing the loss does not teach it the rule. The loss is therefore not the cause.
