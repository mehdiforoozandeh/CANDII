---
id: t360
type: task
title: Onion: learned cell-type embedding arm on the three models, with the zero-shot control
category: implementation
parent: t152
blocked_by: t357
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:32
updated: "2026-11-18T23:23:12"
---

# t360 — Onion: learned cell-type embedding arm on the three models, with the zero-shot control

Refs:: [[q48_which_minicandi_design_choices_move_impu\|q48]]

## Why

ready queue entry 12; budget of 728.6 CPU-hours waits for the PI

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 12 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A6 learned cell-type embedding.md`.
- Commands (Fir): `submit_cell_embedding.sh` with `MODE=train`, then `audit`, `derive`, `zeroshot`; each first as a dry run.
- Cost: 728.6 CPU-hours (173 jobs). PI decision open: this budget.
- Prerequisite: t357; a fresh Fir kit.
- Decision table: item A6 file, section 9.7. The zero-shot control is required before any "helps" conclusion.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
