---
id: t29
type: task
title: decide what a store-backed h74 reference table is, so --reference on stops being refused under --store
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-21T01:03:39
updated: 2026-08-21T01:03:39
---

# t29 — decide what a store-backed h74 reference table is, so --reference on stops being refused under --store

Refs:: _(none)_

## Why

candi.reference.ReferenceTable pins itself to a baked h5's fingerprint, so train.py refuses --reference on when the data came from a store, and StoreDataset therefore cannot emit log_ref - the sixth key on t14's list. t14 deliberately left it out rather than fabricate one. This is a design question (what does a leave-one-out reference mean over a regime's declared train split, and what does it pin itself to instead of an h5 fingerprint) before it is code.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
