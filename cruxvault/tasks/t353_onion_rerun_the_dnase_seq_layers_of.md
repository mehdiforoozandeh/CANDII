---
id: t353
type: task
title: "Onion: rerun the DNase-seq layers of the replicate ladder and of the second-experiment layers without the 15,201-read track ENCSR000EKG (or with a minimum common depth)"
category: implementation
parent: t152
blocked_by: t352
refs: q59, q58
hypothesis_refs: 
status: done
created: "2026-11-18T09:48:04"
updated: "2026-11-18T21:14:52"
---

# t353 — Onion: rerun the DNase-seq layers of the replicate ladder and of the second-experiment layers without the 15,201-read track ENCSR000EKG (or with a minimum common depth)

Refs:: [[q59_do_the_nested_replicate_ladder_and_the_c\|q59]], [[q58_can_the_second_experiments_separate_a_la\|q58]]

## Why

every DNase-seq layer value is now computed at 15,201 reads and is not interpretable

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [DNase-seq layers without ENCSR000EKG](results/closeout_E4/ladder_excl_ENCSR000EKG_2026-11-19/ladder.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
