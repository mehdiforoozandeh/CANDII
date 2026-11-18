---
id: t361
type: task
title: Onion: second experiments: score the base recipes and the final candidates on the add-on (cross-lab correction, metric 7)
category: implementation
parent: t152
blocked_by: t357
refs: q58
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:32
updated: "2026-11-18T23:23:12"
---

# t361 — Onion: second experiments: score the base recipes and the final candidates on the add-on (cross-lab correction, metric 7)

Refs:: [[q58_can_the_second_experiments_separate_a_la\|q58]]

## Why

ready queue entry 13

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 13 (`closeout/ready queue for scaling.md`); item file `closeout/closeout D2 second experiments and cross-lab pairs.md`.
- Command: `KIT_ID=<kit> bash slurm/onion_data/closeout_d2_submit.sh models fir`, then with `DRY_RUN=0` (`RUNS=` names the final candidates).
- Cost: base recipes 144 CPU-hours expected (252 at most); final candidates the same.
- Prerequisite: runs on eic_v1_2 (t357); for the final candidates, the final model (t223).
- Decision table: item D2 file, section 9.5. Lab is reported together with donor (PI decision 3).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
