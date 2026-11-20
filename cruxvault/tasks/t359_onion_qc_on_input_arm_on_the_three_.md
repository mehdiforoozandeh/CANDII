---
id: t359
type: task
title: Onion: QC-on-input arm on the three models (read-derived QC values with presence flags and the leak guard)
category: implementation
parent: t152
blocked_by: t357
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:31
updated: "2026-11-18T23:23:12"
---

# t359 — Onion: QC-on-input arm on the three models (read-derived QC values with presence flags and the leak guard)

Refs:: [[q48_which_minicandi_design_choices_move_impu\|q48]]

## Why

ready queue entry 11; PI decision 7 of 2026-11-17 accepted the arm

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 11 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A4 QC metrics on the input side.md`. PI decision 7 of 2026-11-17: design accepted as built.
- Command (Fir): `CLUSTER=fir K=<kit> MODE=train bash <kit>/tools/minicandi/configs/ablations/qc_on_input/submit_qc_on_input.sh` (dry run first).
- Cost: 404.4 CPU-hours (training 82.2 + scoring 322.2), 90 jobs.
- Prerequisite: the QC side tables (done, Nibi 27678089) copied to Fir by Globus; t357; a fresh Fir kit.
- Limit: do not run the foreign-covariate scoring path on this arm.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
