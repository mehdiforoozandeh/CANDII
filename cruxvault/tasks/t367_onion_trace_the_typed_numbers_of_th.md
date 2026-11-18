---
id: t367
type: task
title: Onion: trace the typed numbers of the supervisor deck of 2026-11-18 to vault files
category: visualization
parent: t152
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:33
updated: "2026-11-18T23:23:12"
---

# t367 — Onion: trace the typed numbers of the supervisor deck of 2026-11-18 to vault files

Refs:: _(none)_

## Why

the deck numbers are literals in build_prezit.py and crux deck --verify does not check them

## Current scope (close-out reconciliation, 2026-11-18)

- Deck: `presentations/onion-supervisor-2026-11-18/` (published https://claude.ai/artifact/BjH384bFDCzZFRgtCvdg51). Its numbers are literals typed by the foreman in `build_prezit.py`, so `crux deck --verify` does not check them (run Log 2026-11-18T21:05Z).
- Do: trace each number to a vault file or a finding, fix any that differs, rebuild and republish to the same URL.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
