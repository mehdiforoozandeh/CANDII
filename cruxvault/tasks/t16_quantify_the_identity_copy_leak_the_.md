---
id: t16
type: task
title: quantify the identity-copy leak the old bake's materialized DSF ladder gives the model
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-19T13:41:44
updated: 2026-08-19T13:41:44
---

# t16 — quantify the identity-copy leak the old bake's materialized DSF ladder gives the model

Refs:: _(none)_

## Why

When x_dsf == y_dsf the old bake hands the model an input column that is the SAME array as the target column - measured 72/72 on matching pairs. With a 4-level ladder that is a free identity copy on roughly 1 in 4 available columns. The store reproduces it only at DSF 1 (2/2, and 0/7 at 2/4/8), so it is ~1 in 16 there. This changes the objective's irreducible noise and is the prime suspect for the old path descending faster over 356 steps. Every pre-store number was measured under the leak, so it bears on how AGENTS.md section 7 results are read. Found by the store-vs-bake A/B run.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
