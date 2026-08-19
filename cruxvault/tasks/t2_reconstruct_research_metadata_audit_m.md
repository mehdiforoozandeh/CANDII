---
id: t2
type: task
title: reconstruct research/METADATA_AUDIT.md, which is 0 bytes
category: data-acquisition
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-18T17:12:11
updated: 2026-08-18T17:12:11
---

# t2 — reconstruct research/METADATA_AUDIT.md, which is 0 bytes

Refs:: _(none)_

## Why

AGENTS.md §7 ranks it authority 3 and says it "ships **in this repo**", but the file is 0 bytes.

The numbers left standing on nothing are in §7.2 rule 5: `H(run_type | assay_id, read_length) = 0.000`
bits on the shipped 8-assay panel (n=26 `T_` records) against 0.551 bits on the full EIC panel. That
pair is the whole argument that a run_type steering demonstration is impossible without re-selecting
the biosample panel. Either rebuild the audit from the metadata it was computed over, or mark the
claim unsourced — §7 is frozen, so the fix is the file, not the section.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
