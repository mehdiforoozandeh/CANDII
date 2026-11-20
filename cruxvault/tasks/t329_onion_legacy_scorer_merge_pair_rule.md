---
id: t329
type: task
title: "Onion: legacy scorer: merge, pair rules for cross-cell pairs, per-region pair sums (SF0)"
category: implementation
parent: t327
blocked_by: None
refs: q27
hypothesis_refs: 
status: done
created: "2026-11-14T02:21:31"
updated: "2026-11-18T23:23:12"
---

# t329 — Onion: legacy scorer: merge, pair rules for cross-cell pairs, per-region pair sums (SF0)

Refs:: [[q27_are_the_onion_run_s_estimators_scorers_a\|q27]]

## Why

no scoring on eic_v1 could finish

## Decisions and runs

- In progress on the SM builder's branch (not merged by 2026-11-14 per the run Log). Its kit SF0 (5d64dab) finished a full marginal_nb scoring on the real evaluation set (job 27256987, 1:30:27, 16 CPUs) with the 589 cross-cell pairs marked not applicable (results/scorer_memory/NOTES.md).

## Close-out note (2026-11-18)

Chunk SF0 accepted and merged 2026-11-14 (run Log, 12:45Z).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Run Log, SF0 accepted 2026-11-14 12:45Z](INSTRUCTIONS/peeling_the_onion/run_state/plan.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
