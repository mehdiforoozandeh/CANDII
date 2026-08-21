---
id: t26
type: task
title: make the signal head's target transform a training-loop option (D30), not the loader's job
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-20T23:57:31"
updated: "2026-08-21T10:11:22"
---

# t26 — make the signal head's target transform a training-loop option (D30), not the loader's job

Refs:: _(none)_

## Why

the bake pre-arcsinhs y_pval and the store does not, so the Gaussian head silently trains in two different target spaces

Plan: [PVAL_CODEC_PLAN.md](../PVAL_CODEC_PLAN.md) D30 and §1.1. Independent of t24/t25 — it can
land first.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- results/t26/DELIVERABLE.md — merged abd9b31

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
