---
id: t126
type: task
title: Build and run the encoder-decoder rung A (shared per-side map phi, exact inverse; beta learned and beta=1) against a retrained old rung A, counts only, depth 7.5M held out of training (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-10-19T23:23:29"
updated: "2026-10-20T19:00:36"
---

# t126 — Build and run the encoder-decoder rung A (shared per-side map phi, exact inverse; beta learned and beta=1) against a retrained old rung A, counts only, depth 7.5M held out of training (exploratory)

Refs:: _(none)_

## Why

tests whether a composable g fixes the never-trained-pair generalisation failure (depth law and other knobs) without arm-to-arm training pairs

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Lab deck: encoder-decoder g](presentations/t126-encdec/DECK.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
