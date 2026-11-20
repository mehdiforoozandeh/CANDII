---
id: t330
type: task
title: "Onion: prediction store and forward speed per rung (SP1)"
category: implementation
parent: t327
blocked_by: None
refs: q27
hypothesis_refs: 
status: done
created: "2026-11-14T02:21:31"
updated: "2026-11-14T02:22:05"
---

# t330 — Onion: prediction store and forward speed per rung (SP1)

Refs:: [[q27_are_the_onion_run_s_estimators_scorers_a\|q27]]

## Why

forward passes dominate model scoring

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged predict_store.py](results/predict_store_speed/FORWARD_SPEED.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
