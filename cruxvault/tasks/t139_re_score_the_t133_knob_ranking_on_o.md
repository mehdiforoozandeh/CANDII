---
id: t139
type: task
title: re-score the t133 knob ranking on one shared peak set to remove the regression-to-the-mean bias
category: implementation
parent: t118
blocked_by: None
refs: q9
hypothesis_refs: 
status: done
created: "2026-10-29T20:08:16"
updated: "2026-10-30T04:26:34"
---

# t139 — re-score the t133 knob ranking on one shared peak set to remove the regression-to-the-mean bias

Refs:: [[q9_which_recorded_processing_covariates_mov\|q9]]

## Why

settles the middle of the knob ranking, which picks what to augment if augmentation is needed

## Decisions and runs

- PI 2026-10-29: re-score on one shared peak set per track, and quote every knob beside the floor at the arm's depth. Plan definition (2026-10-29): per track and space, the top 1 % of the mean within-product rank over the track's products. The default `--peakset base` stays bit-exact with t133 (golden test).
- Foreman amendment A3 (2026-10-29): the floor at depth X is the same-track base product's curve read at X; "above floor" = value > floor + 2 × floor seed spread. p-space rows have no floor (the p ladder was not run).
- Knob re-score job 25389617 (Nibi, kit `/project/def-maxwl/mforooz/t137/code/K1`, 1f634e1), COMPLETED: 192 records, as in t133. Output `/project/def-maxwl/mforooz/t139/out`. The floor join ran locally (`tools/t139/join_floor.py`). Never resubmit.
- Caveat: the thinned floor rows behind the 7.5M and 3.75M joins are being rebuilt by whole-read thinning (PI ruling 2026-10-30).
- Digest: `results/t139/DIGEST.md`.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [t139 digest](results/t139/DIGEST.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
