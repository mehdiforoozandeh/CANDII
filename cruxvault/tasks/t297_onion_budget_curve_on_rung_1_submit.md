---
id: t297
type: task
title: Onion: budget curve on the pointwise MLP: score the 42 trained budget runs and report (U-Net curve optional)
category: implementation
parent: t152
blocked_by: t294
refs: h32
hypothesis_refs: 
status: open
created: 2026-11-13T12:42:53
updated: "2026-11-18T23:23:12"
---

# t297 — Onion: budget curve on the pointwise MLP: score the 42 trained budget runs and report (U-Net curve optional)

Refs:: [[h32_a_smaller_budget_of_windows_ranks_the_ar\|h32]]

## Why

PI questions of 2026-11-13 on region budget and selection (Amendment 7): the scores of the 42 rung-1 budget runs

## Current scope (close-out reconciliation, 2026-11-18)

- State: the 42 budget runs (7 arms x 20,000 and 40,000 windows x 3 seeds) trained on 2026-11-14 (t291) and have no score. This task absorbs t298, t302 and t303.
- Left: screening scores, then the report; optionally the same curve on the U-Net (old chunks RB4a, RB4b).
- Cost: the screening scoring of the pointwise MLP was about 2.7 GPU-slice-hours + 10.6 CPU-hours per run (Amendment 9), so about 42 x 10.6 = 445 CPU-hours + 113 GPU-slice-hours.
- PI: the number of training windows is follow-up work (2026-11-16).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
