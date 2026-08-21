---
id: t14
type: task
title: emit the eval-only batch keys from StoreDataset (y_data_imp, y_pval_imp, y_peaks_imp, y_meta_imp, imp_biosample_name, log_ref)
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T12:49:05"
updated: "2026-08-21T10:11:22"
---

# t14 — emit the eval-only batch keys from StoreDataset (y_data_imp, y_pval_imp, y_peaks_imp, y_meta_imp, imp_biosample_name, log_ref)

Refs:: _(none)_

## Why

eval.py:329-354 and healthcheck.py:215-223 read all six through batch.get(...), so a store-backed eval does not raise - the imputation arm and the h74 reference arm silently degrade to nothing. Blocks scoring any imputation run off the store. Training is unaffected.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- results/t14/DELIVERABLE.md — merged 16fca65

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
