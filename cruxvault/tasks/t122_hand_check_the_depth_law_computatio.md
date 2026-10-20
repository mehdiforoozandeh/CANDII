---
id: t122
type: task
title: hand-check the depth-law computation (predicted count scale against the true depth ratio) before reading the unmet depth law as a model failure
category: implementation
parent: t118
blocked_by: None
refs: h15, q4
hypothesis_refs: 
status: open
created: 2026-10-02T12:36:03
updated: 2026-10-02T12:36:03
---

# t122 — hand-check the depth-law computation (predicted count scale against the true depth ratio) before reading the unmet depth law as a model failure

Refs:: [[h15_one_transformation_conditioned_on_the_so\|h15]], [[q4_how_much_capacity_does_a_covariate_condi\|q4]]

## Why

the depth law is unmet in every row-1 design and in all but one row-2 cell; a wrong computation would look the same

## Progress

- 2026-10-19: audited. On chr19 + chr21 outside the blacklist, the real depth arms' count totals match the manifest depth ratios within 0.71% over all 84 ordered pairs of the 7 tracks ([audit](results/t122/AUDIT.md)). The audit also reported recomputing all 24 check values exactly; that left no saved output. The miss is followed under [[q7_why_do_the_covariate_conditioned_g_f_des|why the designs miss the depth law]]. Not marked done: the PI decides.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
