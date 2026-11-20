---
id: t286
type: task
title: "Onion: Training regions, exclusion and per-window loss weights"
category: implementation
parent: t152
blocked_by: None
refs: h33, h34, h36
hypothesis_refs: 
status: done
created: "2026-11-13T12:42:51"
updated: "2026-11-14T02:20:36"
---

# t286 — Onion: Training regions, exclusion and per-window loss weights

Refs:: [[h33_at_80_000_windows_training_on_r_blocks_o\|h33]], [[h34_at_a_limited_budget_20_000_windows_a_wei\|h34]], [[h36_seeing_the_evaluation_loci_in_training_d\|h36]]

## Why

PI questions of 2026-11-13 on region budget and selection (Amendment 7): the sampler keys train_regions and exclude_regions that every region arm trains with

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged 34f7711, 2cfe3a3: training regions sampler](INSTRUCTIONS/peeling_the_onion/run_state/plan.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
