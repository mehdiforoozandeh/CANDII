---
id: t213
type: task
title: "Onion: Rung 1: base arm and per-rung ablations, collect and settle"
category: implementation
parent: t152
blocked_by: t200, t186, t195
refs: 
hypothesis_refs: 
status: dropped
created: "2026-11-11T11:20:29"
updated: "2026-11-18T23:23:12"
---

# t213 — Onion: Rung 1: base arm and per-rung ablations, collect and settle

Refs:: _(none)_

## Why

settles rung 1 and its seed spread before gate G2b (chunks L1a, L1b)

## Decisions and runs

- Partly done: all 24 rung-1 runs (8 arms x 3 seeds) trained to DONE on 2026-11-14 00:19 UTC after the staging fix (kit L1fix; ONION_STAGE=1). No run is scored: the old-scorer jobs were cancelled by the PI on 2026-11-14; scoring waits for the fast scorer (Amendment 8) and the screening tier (t312). Notes: results/t213/SUBMITTED.md.

## Dropped (close-out reconciliation, 2026-11-18)

The rung-by-rung ladder (chunks L1a to L6b, LCa, LC) was deferred by Amendment 9, section D, and then replaced by the tuning programme (results/ladder_tuning/ROUND2_READ.md), the final round (results/final_round/FINAL_ROUND_EQUAL.md) and the close-out ablation items A1, A2, A3, A8 and A11 (results/ablations_closeout/ABLATIONS_READ.md). The 256-region rescoring of the decisive arms is t350. The pointwise MLP is rescored under the corrected composite in t368.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
