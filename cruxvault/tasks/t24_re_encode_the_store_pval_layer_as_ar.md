---
id: t24
type: task
title: re-encode the store pval layer as arcsinh fixed point so peak summits stop truncating
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-20T23:47:13
updated: 2026-08-20T23:47:13
---

# t24 — re-encode the store pval layer as arcsinh fixed point so peak summits stop truncating

Refs:: _(none)_

## Why

62 of 363 EIC pval tracks clip at 655.35 today; nothing downstream can trust the pval track until the codec changes

Plan: [PVAL_CODEC_PLAN.md](../PVAL_CODEC_PLAN.md) — decisions D25-D29, the file-by-file code
change, the doc updates, and the Fir jobs.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
