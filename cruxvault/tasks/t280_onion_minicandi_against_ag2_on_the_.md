---
id: t280
type: task
title: Onion: AlphaGenome 2 against miniCANDI: box-spread signal rescoring of the 9 base runs and their AlphaGenome 2 board rows
category: implementation
parent: t152
blocked_by: None
refs: q26, q57
hypothesis_refs: 
status: open
created: 2026-11-12T18:39:35
updated: "2026-11-18T23:23:12"
---

# t280 — Onion: AlphaGenome 2 against miniCANDI: box-spread signal rescoring of the 9 base runs and their AlphaGenome 2 board rows

Refs:: _(none)_

## Why

PI task of 2026-11-13 (AlphaGenome 2 as a competitor): submits the jobs that write the frozen final design's method track files on the validation tracks

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 7 (`closeout/ready queue for scaling.md`); item file `closeout/closeout E2 p-value and peaks from the predicted NB.md`. This task absorbs t281.
- Steps: (1) `KIT_ID=<fresh Fir kit> bash slurm/onion_eval/e2_submit.sh fir`; (2) `KIT_ID=CO8 bash slurm/onion_eval/e2_submit.sh nibi_baselines`; (3) Globus `$O/e2/tracks_box` Fir to Nibi; (4) `... e2_submit.sh nibi_board`; (5) repeat with `SIGNAL_BOX_DEPENDENCE=independent`. Each first as a dry run.
- Cost: about 434 CPU-hours for both dependence forms, plus 96 at most for the board jobs.
- Prerequisite: Fir back from its outage and a fresh Fir kit.
- Decision rule: none for the formula (ruled, box spread). Report both dependence bounds; the predicted-control route is a headline row.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
