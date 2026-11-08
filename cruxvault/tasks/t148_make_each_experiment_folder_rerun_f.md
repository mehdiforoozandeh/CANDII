---
id: t148
type: task
title: make each experiment folder rerun from one bash runall.sh (no args, skip-if-exists, frozen once cited); commit small results folders under a 1 MB cap; pilot on t138
category: implementation
parent: 
blocked_by: None
refs: q10
hypothesis_refs: 
status: done
created: "2026-11-08T03:12:17"
updated: "2026-11-08T04:53:05"
---

# t148 — make each experiment folder rerun from one bash runall.sh (no args, skip-if-exists, frozen once cited); commit small results folders under a 1 MB cap; pilot on t138

Refs:: [[q10_how_much_does_resampling_alone_move_a_tr\|q10]]

## Why

finding evidence links break on a fresh clone; reproduction today is per-task scripts plus job ids

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [t138 runall pilot, merged dcc97ef](results/t138/runall.sh)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
