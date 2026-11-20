---
id: t285
type: task
title: Onion: data set version with DNase-seq in the correction term: 4 row rebuilds, version build, baselines
category: data-acquisition
parent: t152
blocked_by: None
refs: q17
hypothesis_refs: 
status: open
created: 2026-11-13T01:31:43
updated: "2026-11-18T23:23:12"
---

# t285 — Onion: data set version with DNase-seq in the correction term: 4 row rebuilds, version build, baselines

Refs:: _(none)_

## Why

PI ruling 38 (2026-11-13, freeze eic_v1 at a fixed time, then versioned updates): resubmits a failed tail row once and builds the next dataset version with its change notes

## Decisions and runs

- Not started. Since the freeze, rows 46 and 52 (validation histone) and 388 and 390 (DNase-seq training) verified, so a first update (eic_v1_1) is possible (run Log 2026-11-14 05:20 UTC).

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 15 (`closeout/ready queue for scaling.md`); item file `closeout/closeout D3 dataset update with DNase-seq.md`. Candidate `eic_v1_3_candidate`, 2,812 rows.
- Commands (Nibi): `ssh -o BatchMode=yes nibi 'KIT_ID=CO8 bash -s -- reruns' < slurm/onion_data/closeout_d3_submit.sh`, then with `DRY_RUN=0` (rows 56, 132, 280, 396; one 7-day rerun per row, rule B.5); after they end `-- version`, then `-- baselines`.
- Cost: reruns about 4,992 CPU-hours expected, 10,752 at most; version 64 at most; baselines about 123. If the PI picks this version, the 9 base runs train again (82.2).
- Prerequisite: writable Nibi scratch; the FASTQs of the 4 rows are downloaded again.
- Decision: the PI chooses the data version of the final model (t364).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
