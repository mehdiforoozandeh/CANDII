---
id: t212
type: task
title: Onion: crossed mixed model of the replicate ladder, after a residual variance per library exists
category: implementation
parent: t152
blocked_by: None
refs: q59
hypothesis_refs: 
status: open
created: 2026-11-11T11:20:29
updated: "2026-11-18T23:23:12"
---

# t212 — Onion: crossed mixed model of the replicate ladder, after a residual variance per library exists

Refs:: _(none)_

## Why

the onion layers report (chunks E8a, E8b, E8c; E8c also waits for the blind pass BPa)

## Current scope (close-out reconciliation, 2026-11-18)

- Done elsewhere: the variance layers on real data (t305), the replicate ladder in the two-sided form (t351), the variance by bin size (close-out item A7). Layers with the blind tracks wait for the blind build (t184).
- Left: the stage `mixed` of `slurm/onion_eval/closeout_e4_submit.sh` (4 jobs, 128 CPU-hours at most) must not run until the model has a residual variance per library. That code does not exist (`results/closeout_E4/E4_MIXED_DIAG.md`; ready queue entry 3).
- Decision rule: none written; write the residual term first, then run the stage.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
