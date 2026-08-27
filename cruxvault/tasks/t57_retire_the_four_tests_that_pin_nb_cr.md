---
id: t57
type: task
title: retire the four tests that pin nb_crps's pre-fix NaN at large n
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-27T11:25:14
updated: 2026-08-27T11:25:14
---

# t57 — retire the four tests that pin nb_crps's pre-fix NaN at large n

Refs:: _(none)_

## Why

PR #26 (t56) fixed the nb_crps large-dispersion NaN, but four tests merged from crossing branches still assert the defect; main is red until they pin the post-fix behavior instead

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
