---
id: t79
type: task
title: rewrite the eic regimes to chr19 and pilot-regions training with chr20+21+22 scored
category: implementation
parent: t77
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-29T18:42:15"
updated: "2026-09-01T18:24:40"
---

# t79 — rewrite the eic regimes to chr19 and pilot-regions training with chr20+21+22 scored

Refs:: _(none)_

## Why

a regime must name its training loci and its eval loci; the current ones do not, and Avocado's joint fit sits on an eval chromosome

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The Pilot Regions in hg38, and the two live regimes](results/t79/G2_PILOT_HG38.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
