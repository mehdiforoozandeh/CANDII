---
id: t23
type: task
title: teach train.py to open a CANDI_STORE, not only a baked h5
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-19T23:57:04
updated: 2026-08-19T23:57:04
---

# t23 — teach train.py to open a CANDI_STORE, not only a baked h5

Refs:: _(none)_

## Why

train.py takes --h5 and reads ds.num_cells, ds.num_assays, ds.context_bins and ds.resolution off CandiKitH5Dataset, so a store cannot be trained from at all today - the store's whole point is unreachable from the training entrypoint. StoreDataset already emits the training-path batch dict bit-identically (0 differing elements of 143.2 M against the old bake) and loads 4.80x faster, so this is a plumbing gap, not a data question. Scoped to training; scoring a store-backed imputation run still needs t14.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
