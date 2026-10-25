---
id: t129
type: task
title: merge the loss head-to-head code (Poisson-mean loss, rung A0, score --extras) from exp/t118-ppml-vs-nb into exp/t118-counterfactual-f, resolving the score.py and train.py conflicts against the t126/t128 changes; tests on Nibi
category: implementation
parent: t118
blocked_by: None
refs: h21, h18
hypothesis_refs: 
status: open
created: 2026-10-25T00:18:27
updated: 2026-10-25T00:18:27
---

# t129 — merge the loss head-to-head code (Poisson-mean loss, rung A0, score --extras) from exp/t118-ppml-vs-nb into exp/t118-counterfactual-f, resolving the score.py and train.py conflicts against the t126/t128 changes; tests on Nibi

Refs:: [[h21_fitting_the_mean_by_poisson_nll_and_the_\|h21]], [[h18_the_1_pseudocount_in_x_log_1_counts_make\|h18]]

## Why

the two branches share tools/t118/ladder; later runs need both the ppml loss and the encoder-decoder rungs in one tree

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
