---
id: t354
type: task
title: Onion: verify the 1 bp tables with more memory per worker and build the resolution eval set (job 27692994 ran out of memory)
category: implementation
parent: t152
blocked_by: None
refs: q60
hypothesis_refs: 
status: open
created: 2026-11-18T09:48:04
updated: "2026-11-18T23:23:12"
---

# t354 — Onion: verify the 1 bp tables with more memory per worker and build the resolution eval set (job 27692994 ran out of memory)

Refs:: [[q60_how_much_of_the_1_bp_count_variance_of_a\|q60]]

## Why

the 1 bp region store is not verified end to end; the resolution arms need the eval set

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 22 (`closeout/ready queue for scaling.md`). Nibi job 27692994 reached its 16-hour limit with 6 out-of-memory kills (`recount.py verify --all --no-rescan --workers 8` in 32000M). The tables `CO-A7/tables/eic_res_v1` are built and NOT verified; `CO-A7/eval/eic_res_v1` does not exist. Never resubmit 27692994.
- Command: the `tables` part of `slurm/onion_data/closeout_a7_submit.sh`, split into a verify job (for example `--workers 2` with 64000M, or 8 workers with 128000M) and an eval set build and check job. Measure the peak memory of one worker first.
- Cost: not measured. Blocks the resolution arms (t254).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
