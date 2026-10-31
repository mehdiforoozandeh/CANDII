---
id: t104
type: task
title: the four baseline rungs (blind, onewarp, model, oracle), with the point-to-distribution spread device so blind and onewarp are CRPS-scorable
category: implementation
parent: t130
blocked_by: None
refs: h1, h2, h3, h4, h5, h6, h7, h8, h9, h10
hypothesis_refs: 
status: open
created: 2026-09-01T02:19:16
updated: 2026-09-01T02:19:16
---

# t104 — the four baseline rungs (blind, onewarp, model, oracle), with the point-to-distribution spread device so blind and onewarp are CRPS-scorable

Refs:: [[h1_conditioning_on_the_recorded_sequencing_\|h1]], [[h2_conditioning_on_the_recorded_run_type_pr\|h2]], [[h3_conditioning_on_the_recorded_read_length\|h3]], [[h4_conditioning_on_the_recorded_duplicate_h\|h4]], [[h5_conditioning_on_the_recorded_mapping_qua\|h5]], [[h6_conditioning_on_the_recorded_antibody_ef\|h6]], [[h7_conditioning_on_the_recorded_control_to_\|h7]], [[h8_conditioning_on_the_recorded_control_ide\|h8]], [[h9_conditioning_on_the_recorded_control_dep\|h9]], [[h10_conditioning_on_the_recorded_fragment_ex\|h10]]

## Why

gapclosed has no denominator without them, and the headline is measured from onewarp, not from blind

## Progress

- 2026-10-25: re-parented from the dropped testbed to [[t130_score_the_one_warp_check_set_blind_|the scoring task]]. Under g → f, `blind` is the no-covariates twin and `model` is g → f, both already trained in t118 and both NB-valued. `onewarp` is t118's per-pair QuantileMatching, and `oracle` is two NB draws at the target depth; only these two need the spread device.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
