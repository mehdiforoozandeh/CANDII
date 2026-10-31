---
id: t56
type: task
title: fix nb_crps NaN overflow at large dispersion n and NaN-as-loss in beats_marginal
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-25T23:43:38"
updated: "2026-09-19T00:24:58"
---

# t56 — fix nb_crps NaN overflow at large dispersion n and NaN-as-loss in beats_marginal

Refs:: _(none)_

## Why

the pre-registered Poisson-limit floor n=1e6 in EVAL 5.1 is unscoreable: nb_crps returns NaN from n~3e4 up, and bench's beats_marginal counts a NaN track as a loss instead of absent (t49 evidence: cruxvault/results/t49/p1_spec)

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged PR #26](results/t56/MERGED.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
