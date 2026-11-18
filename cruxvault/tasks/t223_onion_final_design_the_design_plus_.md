---
id: t223
type: task
title: Onion: final model and EIC board row: freeze, train, check on validation, single blind pass
category: implementation
parent: t152
blocked_by: t363, t364, t220, t362, t285, t184
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-11T11:20:31
updated: "2026-11-18T23:23:12"
---

# t223 — Onion: final model and EIC board row: freeze, train, check on validation, single blind pass

Refs:: _(none)_

## Why

freezes the final design before gate G4 (chunks FDa, FDb)

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 17 (`closeout/ready queue for scaling.md`); item file `closeout/closeout D4 blind tracks final model EIC board.md`. Not built: no freeze file, no final-model config, no blind tables, no `evalset_blind.json`, no `BLIND_PASSES.tsv`, no `final_board.tsv`.
- Steps: freeze file; 3 seeds on training plus validation and 3 seeds training-only (6 trainings, about 11 to 15 CPU-hours each by the item file); D14 condition 2 on validation; build the blind tables from the blind build; one ledgered blind pass; board row (seed 0, spread beside it); the compute tally after the pass. This task absorbs t224, t225 and t226.
- Prerequisite: the architecture (t363) and data version (t364) decisions; the mix-and-match recipe (t220); the covariate set (t362) if the PI wants it in the final model; the data version build (t285); the blind build (t184).
- Scorers exist: `tools/onion_eval/score.py`, `ag2_score.py` (board reproduction matches all 25 published ranks, `results/t166/BOARD_REPRO.md`).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
