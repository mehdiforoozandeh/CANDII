---
id: t129
type: task
title: merge the loss head-to-head code (Poisson-mean loss, rung A0, score --extras) from exp/t118-ppml-vs-nb into exp/t118-counterfactual-f, resolving the score.py and train.py conflicts against the t126/t128 changes; tests on Nibi
category: implementation
parent: t118
blocked_by: None
refs: h21, h18
hypothesis_refs: 
status: done
created: "2026-10-25T00:18:27"
updated: "2026-10-25T03:26:42"
---

# t129 — merge the loss head-to-head code (Poisson-mean loss, rung A0, score --extras) from exp/t118-ppml-vs-nb into exp/t118-counterfactual-f, resolving the score.py and train.py conflicts against the t126/t128 changes; tests on Nibi

Refs:: [[h21_fitting_the_mean_by_poisson_nll_and_the_\|h21]], [[h18_the_1_pseudocount_in_x_log_1_counts_make\|h18]]

## Why

the two branches share tools/t118/ladder; later runs need both the ppml loss and the encoder-decoder rungs in one tree

## Progress

- 2026-10-25: merged; both sides' defaults unchanged (bit-identical rung-A smoke, Nibi 24967176). One merge bug fixed (load_run read t126/t128 loss formulas as the ladder loss). Nibi 24969201: 1740 passed, 2 failed, both failing on the unmerged code too.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merge and test record](results/t129/MERGE.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
