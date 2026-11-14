---
id: t327
type: task
title: Onion: fast model scoring for a fast feedback loop (PI directive 2026-11-14; Amendment 8)
category: implementation
parent: t152
blocked_by: None
refs: q27
hypothesis_refs: 
status: open
created: 2026-11-14T02:21:21
updated: 2026-11-14T02:21:21
---

# t327 — Onion: fast model scoring for a fast feedback loop (PI directive 2026-11-14; Amendment 8)

Refs:: [[q27_are_the_onion_run_s_estimators_scorers_a\|q27]]

## Why

a model scoring took about 26 h at 256 regions; target about 15 min

## Decisions and runs

- PI directive 2026-11-14: scoring a model is too slow; optimize it for a fast feedback loop of architecture optimization. Foreman decision: the scorer may be re-baselined once to a faster, numerically equivalent version (scorer_version in scores.json). Done: memory (t328), prediction store (t330), derived evaluation set (t331), check tool (t332). Open: SF0, SP2, SP3, SRa, PCa. The baselines submission is t306 (chunk SBa became I1b).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
