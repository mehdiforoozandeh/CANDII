---
id: t37
type: task
title: decide whether the bench harness sharing one thinning seed between input and target is a paired depth sweep or the same identity-copy leak
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-21T15:19:56"
updated: "2026-08-22T17:18:11"
---

# t37 — decide whether the bench harness sharing one thinning seed between input and target is a paired depth sweep or the same identity-copy leak

Refs:: _(none)_

## Why

harness.py:519 and harness.py:539 build identical draw_seed tuples; the store's _thin now passes a side but bench deliberately does not, so bench numbers are unmoved until this is answered

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Verdict: paired — one ladder read twice](results/t37/FINDING.md) — not the t27 leak; docstrings at both sites + two tripwire tests landed in 2f56cb1

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
