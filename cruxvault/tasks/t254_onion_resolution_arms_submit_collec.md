---
id: t254
type: task
title: Onion: resolution arms, full design (5 to 125 bp, 75 runs, 396 jobs)
category: implementation
parent: t152
blocked_by: t354
refs: q60
hypothesis_refs: 
status: open
created: 2026-11-11T23:05:11
updated: "2026-11-18T23:23:12"
---

# t254 — Onion: resolution arms, full design (5 to 125 bp, 75 runs, 396 jobs)

Refs:: _(none)_

## Why

reports the resolution arms at four table sizes, reported only (chunks RMa, RMb)

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 9 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A7 resolution.md`.
- PI decision 6 of 2026-11-17: full design, next session.
- Command: `ssh -o BatchMode=yes nibi 'KIT_ID=<kit> DESIGN=full DRY_RUN=0 bash -s -- arms' < slurm/onion_data/closeout_a7_submit.sh` (dry run first). The recount, the tables and the variance by bin size are done.
- Cost: about 2,265 CPU-hours for the arms (about 2,340 with the recount).
- Prerequisite: the verified 1 bp tables and the resolution eval set (t354).
- Decision rule: item A7 file, section 9.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
