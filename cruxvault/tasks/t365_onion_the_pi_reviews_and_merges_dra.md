---
id: t365
type: task
title: "Onion: the PI reviews and merges draft PR 61 (umbrella branch implementation/t152-peeling-the-onion)"
category: admin
parent: t152
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-11-18T16:21:33"
updated: "2026-11-20T13:47:11"
---

# t365 — Onion: the PI reviews and merges draft PR 61 (umbrella branch implementation/t152-peeling-the-onion)

Refs:: _(none)_

## Why

lands the run on main; the worktree removal waits for it

## Current scope (close-out reconciliation, 2026-11-18)

- Draft PR mehdiforoozandeh/CANDII#61. Merging to main is the PI's action.
- State at the end of the run (WHERE_WE_ARE): full test suite 6,950 passed, 9 skipped, 0 failed at code state 6b2a2273; `tools/golden.py` check against origin/main 0d7936a6: bit-identical.
- Blocks t356 (removal of the worktrees and chunk branches).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged f1864282](INSTRUCTIONS/peeling_the_onion/run_state/HANDOFF_NEXT_SESSION.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
