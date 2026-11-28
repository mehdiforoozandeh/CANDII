---
id: t398
type: task
title: fix the testbed half-split nesting check in the onion build: depth-thinned and keep-duplicates views draw their own random read halves, so their halves exceed or fall below the base halves (7 blind-retest rows fail verify, base rows pass)
category: implementation
parent: 
blocked_by: None
refs: q17, q90
hypothesis_refs: 
status: open
created: 2026-11-28T05:26:37
updated: 2026-11-28T05:26:37
---

# t398 — fix the testbed half-split nesting check in the onion build: depth-thinned and keep-duplicates views draw their own random read halves, so their halves exceed or fall below the base halves (7 blind-retest rows fail verify, base rows pass)

Refs:: [[q17_what_is_the_dataset_and_what_did_it_cost\|q17]], [[q90_can_a_count_space_model_like_minicandi_b\|q90]]

## Why

lets the 7 blind-retest rows pass verify; blind entry uses their base rows meanwhile (decision 2026-11-28)

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
