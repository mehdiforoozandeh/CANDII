---
id: t15
type: task
title: fill an absent control with MISSING rather than 0 in StoreDataset
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-19T12:49:05
updated: 2026-08-19T12:49:05
---

# t15 — fill an absent control with MISSING rather than 0 in StoreDataset

Refs:: _(none)_

## Why

control_avail = 0 already makes it harmless, but 0 in control_meta[0] reads as log2(depth) = 0 rather than unknown, and every other absent channel in the same batch dict uses MISSING. One-line change; kept separate because it moves a tensor value and therefore needs its own before/after.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
