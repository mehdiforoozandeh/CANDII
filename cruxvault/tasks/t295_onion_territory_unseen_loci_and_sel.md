---
id: t295
type: task
title: Onion: territory, unseen-loci and selection arms: settled scoring, report and the confirmation of the best strategy
category: implementation
parent: t152
blocked_by: t294
refs: h33, h34, h36
hypothesis_refs: 
status: open
created: 2026-11-13T12:42:53
updated: "2026-11-18T23:23:12"
---

# t295 — Onion: territory, unseen-loci and selection arms: settled scoring, report and the confirmation of the best strategy

Refs:: [[h33_at_80_000_windows_training_on_r_blocks_o\|h33]], [[h34_at_a_limited_budget_20_000_windows_a_wei\|h34]], [[h36_seeing_the_evaluation_loci_in_training_d\|h36]]

## Why

PI questions of 2026-11-13 on region budget and selection (Amendment 7): the 45 runs of the territory, unseen-loci and selection arms

## Current scope (close-out reconciliation, 2026-11-18)

- State: the trainings ran (selection arms at 20,000 windows, the unseen-loci arm `rg_no_eval`, 3 seeds each) and have screening scores on 16 regions (results/metrics_screening/scores_selection_arms.md; finding f80: no selection arm passes the settling rule). This task absorbs t299 and t300.
- Left: scores on all 256 pair regions, the report, and, only if a strategy wins, its confirmation at the full budget.
- Cost: per 256-region scoring of one run-seed about 30 to 52 CPU-hours (ready queue, section 0); not counted for this arm set.
- PI: the number and selection of training regions is follow-up work (2026-11-16).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
