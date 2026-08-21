---
id: t21
type: task
title: "the C-block: six covariate-sensitivity instruments (use, share, direction, specificity, invariance, bio-conservation guard)"
category: implementation
parent: t17
blocked_by: t19
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T23:52:14"
updated: "2026-08-20T23:54:27"
---

# t21 — the C-block: six covariate-sensitivity instruments (use, share, direction, specificity, invariance, bio-conservation guard)

Refs:: _(none)_

## Why

CANDI's conditioning claim currently rests on M2/M3, which have no valid null and a collapse loophole

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The C-block: six covariate-sensitivity instruments — commit d5b1804](results/t21/DELIVERABLE.md) — covariate.py with c1_use, c2_share, c3_direction, c4_specificity, c5_invariance, c6_conservation (D8, all six ship), plus layer 3: two stub models whose conditioning we wrote ourselves, so every instrument has a KNOWN answer. Depth-only stub gives C1 significant for depth under both nulls and uniform elsewhere, C2 share (1,0,0,0), C3 monotone with the argmin at the true depth, C4 mass only in the depth row; the ignore-everything stub gives share (0.25,0.25,0.25,0.25) — the conditional-model-that-ignored-its-condition failure made into a test. The one that proves the guard guards: a deliberately COLLAPSED encoder fails C6 while passing C5. TWO ESTIMATOR BUGS WERE CAUGHT AND FIXED, neither visible on real data because only the stub knows the truth. c2_share needed an inner-sample bias correction and common random numbers: a mean over K inner draws is itself noisy so the naive variance estimates Var(E[f|X_S]) + E[Var(f|X_S)]/K, and that second term is positive and largest exactly for the subsets carrying no information, leaking share onto irrelevant covariates — depth read 0.902 on a model that provably reads nothing else, and moved 0.99 to 0.87 between seeds; with the correction and paired draws it reads 1.00, stable to 0.01. ilisi must cap k at one below the smallest batch or its floor is unreachable. Shapley not Sobol (D10) because AGENTS.md section 329 records H(run_type | assay_id, read_length) = 0.000 bits on 26 T_ records, so Sobol's independence assumption fails and indices stop summing to one; 16 subsets is exact, so the exponential objection does not apply. C1 runs both nulls side by side (D9) and reports conditional_null_degenerate so an empty conditional null cannot be misread as the model ignoring the covariate. C5 cannot be returned without C6 (D13) by signature rather than convention. 23 stub-model tests; re-verified at 6470a6a with 688 passing and golden 0 ULP.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
