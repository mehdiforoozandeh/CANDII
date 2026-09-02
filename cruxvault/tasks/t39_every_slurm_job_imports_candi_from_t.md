---
id: t39
type: task
title: every SLURM job imports candi from the shared kit, not from KIT -- the venv's editable install pins /project/.../CANDII/src
category: hpc-setup
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-21T18:44:39
updated: 2026-08-21T18:44:39
---

# t39 — every SLURM job imports candi from the shared kit, not from KIT -- the venv's editable install pins /project/.../CANDII/src

Refs:: _(none)_

## Why

measured on Fir 2026-08-21: KIT selects which scripts run, the library comes from the shared clone at whatever branch it is parked on. t31 failed this way; a change rather than an addition would have produced numbers instead of an error. Affects train.sh, t22_equiv.sh and every other job script

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
