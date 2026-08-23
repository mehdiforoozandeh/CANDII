---
id: t33
type: task
title: diagnose C2 reporting output_is_constant=true while C1, C3 and C4 all show the output varying -- [Cause](results/t33/FINDING.md) — clamp below estimator resolution + ddof mismatch. Fix landed in 2f56cb1: unclamped total with naive/bias/se split, ddof=1 both, output_is_constant retired
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-21T15:03:12"
updated: "2026-08-22T17:18:11"
---

# t33 — diagnose C2 reporting output_is_constant=true while C1, C3 and C4 all show the output varying -- cause only, no fix

Refs:: _(none)_

## Why

PI ruling 2026-08-21: likely C2 conflates 'below resolution' with 'zero variance', but that must be shown not assumed

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Cause](results/t33/FINDING.md) — clamp below estimator resolution + ddof mismatch. Fix landed in 2f56cb1: unclamped total with naive/bias/se split, ddof=1 both, output_is_constant retired

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
