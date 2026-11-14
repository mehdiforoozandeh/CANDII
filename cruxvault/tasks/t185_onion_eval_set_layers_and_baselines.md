---
id: t185
type: task
title: Onion: Eval set, layers and baselines on eic_v1 (submit)
category: implementation
parent: t152
blocked_by: t183, t172, t179
refs: 
hypothesis_refs: 
status: open
created: 2026-11-10T22:47:27
updated: 2026-11-10T22:47:27
---

# t185 — Onion: Eval set, layers and baselines on eic_v1 (submit)

Refs:: _(none)_

## Why

submits the eval set, layers and baselines on eic_v1

## Decisions and runs

- Partly done (2026-11-13): the evaluation set on eic_v1 is built (256 regions, 63 imputation items, 1,480 correction pairs, 3,134 replicate pairs; evalset_sha256 b5f7a4cf...0ce8; check 'problems: 0'); layers job 27164943 and the average table 27164944 COMPLETED. The four baseline scorings ran out of memory, were resubmitted at 128000M, and were cancelled by the PI on 2026-11-14 (old scorer too slow; cross-cell pairs crash it). The baselines moved to t306. Notes: results/t185/SUBMITTED.md.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
