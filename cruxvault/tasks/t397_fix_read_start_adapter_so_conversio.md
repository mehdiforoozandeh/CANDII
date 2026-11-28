---
id: t397
type: task
title: fix read-start adapter so conversion does not depend on forward-batch boundaries
category: implementation
parent: 
blocked_by: None
refs: q94
hypothesis_refs: 
status: open
created: 2026-11-28T05:06:08
updated: 2026-11-28T05:06:08
---

# t397 — fix read-start adapter so conversion does not depend on forward-batch boundaries

Refs:: [[q94_is_the_read_start_or_the_read_count_form\|q94]]

## Why

rs_adapter.py stitches each forward batch separately; touches <=0.1% of bins with no fixed sign

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
