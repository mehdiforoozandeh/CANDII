---
id: t44
type: task
title: close the covariate block's denoise-arm input leak: no leave-one-out mask under kind=denoise, so the target column sits verbatim in the encoder input at DSF 1
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-22T17:18:52
updated: 2026-08-22T17:18:52
---

# t44 — close the covariate block's denoise-arm input leak: no leave-one-out mask under kind=denoise, so the target column sits verbatim in the encoder input at DSF 1

Refs:: _(none)_

## Why

measured 4/4 windows leaked on the denoise arm vs 0/4 on impute; the block's own docstring names this as exactly what must not happen

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
