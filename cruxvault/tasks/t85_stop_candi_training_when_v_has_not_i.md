---
id: t85
type: task
title: stop CANDI training when V_ has not improved for more than 3 epochs
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-31T13:16:24"
updated: "2026-08-31T15:28:19"
---

# t85 — stop CANDI training when V_ has not improved for more than 3 epochs

Refs:: _(none)_

## Why

the eic.19 retrain burned 9 of 11 GPU-hours after its best epoch; nothing stops a run whose validation curve has turned

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Early stopping on a stalled V_](results/t85/DELIVERABLE.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
