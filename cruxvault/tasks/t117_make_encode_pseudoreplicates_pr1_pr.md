---
id: t117
type: task
title: "make ENCODE pseudoreplicates (pr1, pr2: counts and MACS2 -log10 p) for all 130 counterfactual-arm products on Nibi"
category: data-acquisition
parent: 
blocked_by: None
refs: h1, h2
hypothesis_refs: 
status: done
created: "2026-09-23T14:20:03"
updated: "2026-09-23T22:57:09"
---

# t117 — make ENCODE pseudoreplicates (pr1, pr2: counts and MACS2 -log10 p) for all 130 counterfactual-arm products on Nibi

Refs:: [[h1_conditioning_on_the_recorded_sequencing_\|h1]], [[h2_conditioning_on_the_recorded_run_type_pr\|h2]]

## Why

oracle for the counterfactual-mapping f: a second measurement of the same material under the same settings

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [pseudoreplicate checks, 130/130 on all four](results/t117/CHECKS.md)
- [per-half manifest](results/t117/MANIFEST.tsv)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
