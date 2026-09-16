---
id: t78
type: task
title: rebuild the DNase p-value layer from alignments so all 40 DNase experiments are -log10 p
category: implementation
parent: t77
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-29T18:42:15"
updated: "2026-09-01T18:24:40"
---

# t78 — rebuild the DNase p-value layer from alignments so all 40 DNase experiments are -log10 p

Refs:: _(none)_

## Why

40 of 363 experiments carry read-depth normalized signal where the layer claims -log10 p; 34 of them are training tracks feeding CANDI's p-value head

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The DNase p-value layer, rebuilt from alignments](results/t78/G1_PHASE2_DNASE.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
