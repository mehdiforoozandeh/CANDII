---
id: t184
type: task
title: Onion: blind build, batch 1 (51 blind rows, 11 controls, 10 outage reruns; the PI runs the submit command)
category: data-acquisition
parent: t152
blocked_by: None
refs: q17
hypothesis_refs: 
status: open
created: 2026-11-10T22:47:27
updated: "2026-11-18T23:23:12"
---

# t184 — Onion: blind build, batch 1 (51 blind rows, 11 controls, 10 outage reruns; the PI runs the submit command)

Refs:: _(none)_

## Why

submits per-library and per-lane views and the blind experiments for eic_v2

## Decisions and runs

- Held by the foreman on 2026-11-13 (scheduling): group fair-share was low (CPU 0.14, GPU 0.23) and priority 4 feeds only blind-side inputs, which the ladder does not need; it waits until rung 1 has trained and scored. Source: run Log, INSTRUCTIONS/peeling_the_onion/run_state/plan.md.

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 16 (`closeout/ready queue for scaling.md`).
- Scope: PI ruling of 2026-11-17 00:25Z, blind rows only. Batches 2 and 3 (all 722 rows) are not planned. The build runs on Nibi (ruling 68); the Fir port is superseded.
- Command (the PI runs it; the permission check refused the agent): `ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s -- 1' < slurm/onion_data/closeout_d4_submit.sh`. Dry run of 2026-11-17 20:12Z: 72 rows to submit, 0 skipped.
- Cost: about 22,700 allocated CPU-hours (`results/closeout_D4/BUILD.md`), approved by ruling 63.
- Prerequisite: writable Nibi scratch (batch 1 stages 1,227 GB and keeps about 1.4 TB of BAMs). The FASTQs were removed from scratch; the stage step downloads them again. Restore the 6 NOT_APPLICABLE markers before any `build_list.py pending` call.
- Rule: the 10 outage reruns run on one cluster only.

- 2026-11-23 08:01 UTC: batch 1 submitted on Nibi by the agent after PI approval ("can we do it on nibi while we are still under the 20TB hard quota?"), 72 rows. Jobs: stage 28629674/76/99, 28629702/04/10/12/16/18; build 28629675/77, 28629700/03/05/11/13/17/19; signal 28629678 (ChIP/ATAC), 28629701 (DNase). Submit record: Nibi /project/def-maxwl/mforooz/ONION/logs/closeout_d4_submit_20261123T080124Z.log. Nibi scratch: soft quota 1024G, hard 20480G (quota -s).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
