---
id: t368
type: task
title: Onion: pointwise MLP under the corrected composite: composite version 3, same-view pass, unscored seeds
category: implementation
parent: t152
blocked_by: None
refs: q48, q39
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:33
updated: "2026-11-18T23:23:12"
---

# t368 — Onion: pointwise MLP under the corrected composite: composite version 3, same-view pass, unscored seeds

Refs:: [[q48_which_minicandi_design_choices_move_impu\|q48]], [[q39_do_the_context_rungs_2_to_6_convolution_\|q39]]

## Why

ready queue entry 19; the pointwise MLP has no score under the corrected composite

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 19 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A10 pointwise MLP corrected score.md`. Not built.
- Step 1: composite version 3 from stored files (under 1 CPU-hour, no job). Steps 2 and 3: the same-view pass and the unscored seeds; they need the pointwise MLP checkpoints copied from Nibi to Fir, or a Nibi GPU. No script exists.
- Cost: metrics about 11.5 CPU-hours per seed; CPU prediction on Fir not measured.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
