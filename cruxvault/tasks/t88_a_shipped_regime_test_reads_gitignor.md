---
id: t88
type: task
title: a shipped-regime test reads gitignored cruxvault/results, so it fails in every fresh clone
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-31T16:31:35"
updated: "2026-09-01T19:52:55"
---

# t88 — a shipped-regime test reads gitignored cruxvault/results, so it fails in every fresh clone

Refs:: _(none)_

## Why

found while gating t83: a fresh worktree baseline is 1193+1F not 1195, and every agent given a worktree will hit it

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Fresh-worktree pytest](results/t88/FRESH_WORKTREE_PYTEST.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
