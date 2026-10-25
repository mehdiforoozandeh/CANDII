---
id: t127
type: task
title: "head-to-head: Poisson-NLL mean + NB-NLL dispersion vs full NB NLL, counts, per-track g, rungs A, A-no-pseudocount (mu = e^a X^b + e^c), D, 3 seeds; exploratory, no bar"
category: implementation
parent: t118
blocked_by: None
refs: q1, h21, h18
hypothesis_refs: 
status: done
created: "2026-10-19T23:31:53"
updated: "2026-10-20T17:12:36"
---

# t127 — head-to-head: Poisson-NLL mean + NB-NLL dispersion vs full NB NLL, counts, per-track g, rungs A, A-no-pseudocount (mu = e^a X^b + e^c), D, 3 seeds; exploratory, no bar

Refs:: [[q1_do_the_recorded_experimental_covariates_\|q1]], [[h21_fitting_the_mean_by_poisson_nll_and_the_\|h21]], [[h18_the_1_pseudocount_in_x_log_1_counts_make\|h18]]

## Why

t118 depth law unmet; NB NLL balances totals only weighted, Poisson NLL forces sum mu = sum y; PI 2026-10-19

## Informs

- [[h21_fitting_the_mean_by_poisson_nll_and_the_|whether a Poisson-NLL mean meets the depth law]]
- [[h18_the_1_pseudocount_in_x_log_1_counts_make|whether the +1 pseudocount causes the miss]] (design A0)

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Poisson-mean vs NB head-to-head report](results/t127/report.md)
- [Lab deck](presentations/t127-ppml-vs-nb/DECK.md) — published https://claude.ai/artifact/3QpVJzhgeZBwfSXu19y5Yj

## Evidence

Exploratory readings (PI-accepted 2026-10-20). The Poisson-NLL mean removes rung A's total-scale bias (scale_error 0.253 → 0.005, seed wobble 0.025) and costs no NB likelihood. It does not fix the depth law: never-trained depth → depth median |log2 error| is A 0.33 → 0.29, A0 0.46 → 0.38, D 0.27 → 0.28 (wobble 0.05–0.16). Rung A0, with no +1, misses as much as rung A. See [[h21_fitting_the_mean_by_poisson_nll_and_the_|the loss hypothesis]] and [[h18_the_1_pseudocount_in_x_log_1_counts_make|the pseudocount hypothesis]].
