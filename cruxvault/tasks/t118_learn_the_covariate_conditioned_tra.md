---
id: t118
type: task
title: learn the covariate-conditioned transformation f (X' = f(X | C, C')) on the base↔arm pairs of the counterfactual corpus, with the QuantileMatching and pseudoreplicate-oracle rungs, and score the pre-registered checks
category: implementation
parent: 
blocked_by: None
refs: h15, q1
hypothesis_refs: 
status: open
created: 2026-09-23T14:40:48
updated: 2026-09-23T14:40:48
---

# t118 — learn the covariate-conditioned transformation f (X' = f(X | C, C')) on the base↔arm pairs of the counterfactual corpus, with the QuantileMatching and pseudoreplicate-oracle rungs, and score the pre-registered checks

Refs:: [[h15_one_transformation_conditioned_on_the_so\|h15]], [[q1_do_the_recorded_experimental_covariates_\|q1]]

## Why

tests whether recorded covariates alone map one processing of a track onto another; its oracle rung needs the pseudoreplicates of t117 (on branch data-acquisition/t117-pseudoreplicates)

## What we want

The design was finalised 2026-09-25 and is recorded in `plan/T118_COUNTERFACTUAL_F.md`: g(C, C') outputs f, f(X) predicts X'; the competitor is the same model trained with scrambled covariates; never-trained arm → arm pairs test whether f learned how C and C' relate. Done so far: the noSolution / QuantileMatching / pseudoreplicate-oracle rungs (v2, Nibi `/project/def-maxwl/mforooz/t118/rungs_v2/`). Open: the oracle's redefinition, the bars against the twin and for the arm → arm test, and f's architecture.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
