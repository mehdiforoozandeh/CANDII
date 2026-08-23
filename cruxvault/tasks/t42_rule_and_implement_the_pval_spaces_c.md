---
id: t42
type: task
title: "rule and implement the pval spaces contract: eval metrics in -log10 p, predictions inverted"
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-22T17:18:36"
updated: "2026-08-22T17:18:36"
---

# t42 — rule and implement the pval spaces contract: eval metrics in -log10 p, predictions inverted

Refs:: _(none)_

## Why

store-trained gaussian predictions were scored in arcsinh space against raw -log10 p truth - every store pval number compared two spaces

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Deliverable](results/t42/DELIVERABLE.md) — delta-method inversion to -log10 p before every pval-arm benchmark metric, space recorded per row; gaussian_nll stays in training space by design; 2f56cb1

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
