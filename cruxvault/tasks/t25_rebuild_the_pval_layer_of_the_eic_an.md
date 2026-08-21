---
id: t25
type: task
title: rebuild the pval layer of the EIC and MERGED stores under the arcsinh codec
category: implementation
parent: 
blocked_by: t24
refs: 
hypothesis_refs: 
status: done
created: "2026-08-20T23:47:18"
updated: "2026-08-21T10:11:22"
---

# t25 — rebuild the pval layer of the EIC and MERGED stores under the arcsinh codec

Refs:: _(none)_

## Why

the codec change is inert until the 289 GB pval layer is rebuilt from source; the 3.05M already-clipped bins are unrecoverable from the current files

Plan: [PVAL_CODEC_PLAN.md](../PVAL_CODEC_PLAN.md) — decisions D25-D29, the file-by-file code
change, the doc updates, and the Fir jobs.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- results/t25/DELIVERABLE.md — 455/455 rebuilt, gate 55911214 rc=0; fixes merged 6d81f64

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
