---
id: t43
type: task
title: fix covshare's variance attribution: the harness predictor maps inner-block rows to different units, leaking across-unit variance into the bias term
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-22T17:18:52
updated: 2026-08-22T17:18:52
---

# t43 — fix covshare's variance attribution: the harness predictor maps inner-block rows to different units, leaking across-unit variance into the bias term

Refs:: _(none)_

## Why

the leaked bias is subtracted straight off the total variance and is the likely real trigger of the retired false constant-output verdicts on live runs; shares survive but the total is wrong

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
