---
id: t366
type: task
title: Onion: PI decision on the 14 code kits packed in onion_kit_archive on 2026-11-17 (restore some, or keep them packed)
category: admin
parent: t152
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:33
updated: "2026-11-18T23:23:12"
---

# t366 — Onion: PI decision on the 14 code kits packed in onion_kit_archive on 2026-11-17 (restore some, or keep them packed)

Refs:: _(none)_

## Why

scripts whose default kit is archived need KIT_ID=CO8 or a restored kit

## Current scope (close-out reconciliation, 2026-11-18)

- On 2026-11-17 the 14 kits CO7, COA12, COA4, COA6, COA6b, COA7, COD2, COD2b, COD3, COE2d, COE2e, COE2f, COE4, COMIX were packed to `/project/def-maxwl/mforooz/onion_kit_archive` (PI ruling (iii)); restore: `tar -xf <kit>.tar -C /project/def-maxwl/mforooz/onion_kit`.
- Scripts whose default kit is archived run with `KIT_ID=CO8` or a fresh kit. A kit holds about 2,500 files; group `/project` was at 428K of 500K files on 2026-11-17.
- Decision: restore some kits, or keep them packed and push fresh kits. Recommendation (agent's intent): keep them packed.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
