---
id: t305
type: task
title: Onion: The variance layers on real data
category: implementation
parent: t152
blocked_by: None
refs: h37, q18
hypothesis_refs: 
status: open
created: 2026-11-14T00:35:53
updated: 2026-11-14T00:35:53
---

# t305 — Onion: The variance layers on real data

Refs:: [[h37_on_real_data_the_cell_type_layer_is_the_\|h37]], [[q18_how_large_is_each_layer_of_the_variance_\|q18]]

## Why

Iteration 1 of Amendment 9 (ships 2026-11-15 09:00 UTC): collect the layers job 27164943 into the first Results

## Decisions and runs

- Partly done (first slip, 2026-11-14): results/t305/LAYERS_REAL.md gives two layers per assay (replicate and cell type) and the telescoping check (0 unexplained of 2,775 groups). The pseudoreplicate pairs were dropped by evalset.py:581 (a != b), so the sampling layer is missing; the builder reruns the layers on a derived pair list with the training and validation pseudoreplicate pairs.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
