---
id: t116
type: task
title: retire the four tests that pin nb_crps's pre-fix NaN at large n
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-27T11:25:14"
updated: "2026-09-19T00:25:00"
---

# t116 — retire the four tests that pin nb_crps's pre-fix NaN at large n

Refs:: _(none)_

## Why

PR #26 (t56) fixed the nb_crps large-dispersion NaN, but four tests merged from crossing branches still assert the defect; main is red until they pin the post-fix behavior instead

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged PR #30](results/t116/MERGED.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
