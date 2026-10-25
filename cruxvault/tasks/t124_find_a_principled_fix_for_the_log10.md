---
id: t124
type: task
title: find a principled fix for the −log10 p CRPS explosion in the per-bin designs (rows 1 and 2); capping σ and report-only were rejected by the PI
category: implementation
parent: t118
blocked_by: None
refs: h15, q4
hypothesis_refs: 
status: open
created: 2026-10-02T12:36:47
updated: 2026-10-02T12:36:47
---

# t124 — find a principled fix for the −log10 p CRPS explosion in the per-bin designs (rows 1 and 2); capping σ and report-only were rejected by the PI

Refs:: [[h15_one_transformation_conditioned_on_the_so\|h15]], [[q4_how_much_capacity_does_a_covariate_condi\|q4]]

## Why

until it is fixed, many row-2 −log10 p cells cannot be read: some means reach 10³–10⁶, with wobbles as large

## Progress

- 2026-10-24: no fix is recorded in the vault, in `plan/T118_STATUS.md` or in any commit on any branch. Capping σ and report-only stay rejected; a censored log-normal was withdrawn because it rested on a wrong diagnosis (`plan/T118_STATUS.md`, "The p-space explosion").

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
