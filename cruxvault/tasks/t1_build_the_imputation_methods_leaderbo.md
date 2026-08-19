---
id: t1
type: task
title: build the imputation-methods leaderboard that defines the exp/ merge gate
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-18T17:12:05
updated: 2026-08-18T17:12:05
---

# t1 — build the imputation-methods leaderboard that defines the exp/ merge gate

Refs:: _(none)_

## Why

unblocks the exp/ merge gate, which CLAUDE.md leaves explicitly TODO — until it lands no exp/ branch can merge.

The gate cannot be "the number went up", and CLAUDE.md says so: the target-clustered noise floor on
macro CRPS is ~0.09, and a seed change alone moves pooled CRPS by 0.1195 (`AGENTS.md` §7.2 rule 2).
Several recorded between-arm gaps are smaller than both. So the leaderboard has to place CANDI against
published imputation methods on a shared benchmark, with a resolution stated next to every entry.
Do not invent a weaker gate to unblock a branch — raise it with the PI instead.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
