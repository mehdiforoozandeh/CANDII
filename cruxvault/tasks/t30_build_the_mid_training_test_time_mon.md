---
id: t30
type: task
title: "build the mid-training/test-time monitor module: two metric tiers, V_/T_ dials, fixed window sets, wandb"
category: implementation
parent: 
blocked_by: t28
refs: 
hypothesis_refs: 
status: done
created: "2026-08-21T15:02:53"
updated: "2026-08-22T17:18:11"
---

# t30 — build the mid-training/test-time monitor module: two metric tiers, V_/T_ dials, fixed window sets, wandb

Refs:: _(none)_

## Why

replaces quick_eval; one scorer for CANDI, baselines and test-time inference (2026-08-21 design pingpong)

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Measured wall-clocks, job 56232885](results/t30/TIMING.md) — commits 2f56cb1+e5687cc on implementation/t30-monitor, PR #17; candi.monitor: impute dial per --eval-every, denoise+gap once at end on the selected checkpoint

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
