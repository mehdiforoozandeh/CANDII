---
id: t336
type: task
title: "Onion: choose the pair-region count, derive the evaluation set, rescore (PCa)"
category: implementation
parent: t327
blocked_by: t335, t332
refs: q15
hypothesis_refs: 
status: done
created: "2026-11-14T02:21:34"
updated: "2026-11-18T23:23:12"
---

# t336 — Onion: choose the pair-region count, derive the evaluation set, rescore (PCa)

Refs:: [[q15_are_256_evaluation_regions_enough\|q15]]

## Why

PI ruling 42: settled scores need the chosen count

## Close-out note (2026-11-18)

The 16-versus-256 check failed its pre-written rule (Fir job 4534658), so pair jobs stay on all 256 regions; the rescoring is t350.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Run Log, 16-versus-256 check of 2026-11-17 (Fir job 4534658)](INSTRUCTIONS/peeling_the_onion/run_state/plan.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
