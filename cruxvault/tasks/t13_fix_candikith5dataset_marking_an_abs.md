---
id: t13
type: task
title: fix CandiKitH5Dataset marking an absent control as available and feeding the model a channel of -1
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T12:49:05"
updated: "2026-08-21T10:11:22"
---

# t13 — fix CandiKitH5Dataset marking an absent control as available and feeding the model a channel of -1

Refs:: _(none)_

## Why

dataset.py:406 computes control_avail = 1.0 if (control_data != 0).any(), and the bake fills an absent control with MISSING = -1, so -1 != 0 makes it available. Measured on 16 of 89 EIC biosamples. Every past training run on a control-free biosample trained with a bogus all-minus-one control input marked present, so this touches the interpretation of prior results, not just future ones. Found by the store-vs-bake equivalence run.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- results/t13/DELIVERABLE.md — merged 2b7b723

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
