---
id: t103
type: task
title: "build the in-vitro covariate testbed: encoder/decoder over paired tracks, with log depth pinned as a fixed offset of coefficient exactly 1"
category: implementation
parent: 
blocked_by: t101, t102
refs: h1, h2
hypothesis_refs: 
status: dropped
created: "2026-09-01T02:19:03"
updated: "2026-10-25T00:22:02"
---

# t103 — build the in-vitro covariate testbed: encoder/decoder over paired tracks, with log depth pinned as a fixed offset of coefficient exactly 1

Refs:: [[h1_conditioning_on_the_recorded_sequencing_\|h1]], [[h2_conditioning_on_the_recorded_run_type_pr\|h2]]

## Why

the apparatus both h1 and h2 are read on; the unit that ships, so every child lands on its branch

## Progress

- 2026-10-24: not built. The depth and run-type questions it served were asked instead with the generator testbed of [[t118_learn_the_covariate_conditioned_tra|the generator task]] (design of 2026-09-25), so no run of this encoder–decoder exists. Whether to drop the task is the PI's call.
- 2026-10-25: dropped. PI ruling 2026-10-25: h1 and h2 now run on the counterfactual arms and the g → f generator of t118, so the encoder–decoder testbed is not needed ([[t130_score_the_one_warp_check_set_blind_|the new scoring task]]).

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
