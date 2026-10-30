---
id: t135
type: task
title: Run phase 3b of the pivot study: 3 more seeds for the bigger f and the small reference, per-track g, 42 runs (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: q7, h27
hypothesis_refs: 
status: open
created: 2026-10-29T14:31:18
updated: 2026-10-29T14:31:18
---

# t135 — Run phase 3b of the pivot study: 3 more seeds for the bigger f and the small reference, per-track g, 42 runs (exploratory)

Refs:: [[q7_why_do_the_covariate_conditioned_g_f_des\|q7]], [[h27_f_at_64_channels_2x_wider_with_the_same_\|h27]]

## Why

PI option B 2026-10-29: give the bigger f's 0.0146 CRPS gain through the base the statistical power to read

## Decisions and runs

- PI 2026-10-29: the train arrays 1211830 and 1211840 were cut from a 12 h to a 7 h time limit, and the law arrays 1211839 and 1211842 to 6 h, after the 2026-10-29 Fir queue diagnosis found 12 h over-requested against the measured run times (the PI ran `scontrol update`; verified by `squeue`). Never resubmit.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
