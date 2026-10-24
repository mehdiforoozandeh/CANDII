---
id: t120
type: task
title: build and run row 2 of the g x f grid: g reads the source bin value as well as the covariates, for designs A-D
category: implementation
parent: t118
blocked_by: None
refs: q1
hypothesis_refs: 
status: open
created: 2026-09-26T22:57:56
updated: 2026-10-02T12:29:01
---

# t120 — build and run row 2 of the g x f grid: g reads the source bin value as well as the covariates, for designs A-D

Refs:: [[q1_do_the_recorded_experimental_covariates_\|q1]]

## Why

fills the second row of the 2x4 grid so each f design can be compared with g reading (C,C') vs (x,C,C')

## Progress

- 2026-10-01: built and run on Nibi. All 576 row-2 runs (designs A2–D2) and all 192 shuffled-bin twin runs are trained, scored on chr19 + chr21 and law-tested. They are aggregated with the 576 row-1 runs: 1 344 runs in total. The PI ruled the phase exploratory: run everything, conclude later, file no hypotheses before the runs. No hypothesis is filed, nothing is ticked, and no verdict is recorded.
- Evidence (gitignored, in the vault): `results/row2/GRID_SUMMARY.md` holds the drafted readings, the check tallies and every cell with its seed wobble. `results/row2/grid.md` holds the full grid; `results/row2/<design>/report.md` and `results/row2/checks_*.json` hold the per-design reports and checks. `results/row2/FIR_PATH.txt` names the Nibi directories.
- Run record, job ids and kits: `plan/T118_STATUS.md`, from "Row 2 execution" to "Delivered 2026-10-01".
- The drafted interpretation is under [[q4_how_much_capacity_does_a_covariate_condi|the capacity question]].
- Not marked done: the PI decides.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
