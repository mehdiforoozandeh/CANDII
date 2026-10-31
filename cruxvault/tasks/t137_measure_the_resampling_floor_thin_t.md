---
id: t137
type: task
title: "measure the resampling floor: thin the t117 pr1/pr2 halves along a depth ladder and read peak height, 1 kb shape and per-bin divergence per depth, extrapolated to 30M"
category: implementation
parent: t118
blocked_by: None
refs: q10
hypothesis_refs: 
status: done
created: "2026-10-29T20:08:16"
updated: "2026-10-30T04:26:22"
---

# t137 — measure the resampling floor: thin the t117 pr1/pr2 halves along a depth ladder and read peak height, 1 kb shape and per-bin divergence per depth, extrapolated to 30M

Refs:: [[q10_how_much_does_resampling_alone_move_a_tr\|q10]]

## Why

the floor every q1 reading is quoted beside: the oracle for non-nested pairs, context for nested ones (PI 2026-10-30)

## Decisions and runs

- PI ruling 2026-10-29: thin the t117 halves along ratios 1, 0.5, 0.25, 0.125 (seeds 0, 1 below ratio 1); fit the depth curve per product and read it at 30M; −log10 p floor at 15M only, from the rungs_v2 oracle (plan Decision ii).
- Foreman decision 2026-10-29 (within the ladder ruling): reads per half = ratio × the product's `n_lines`, not a constant 15M (halves are 15M, 25M for DNase, and 7.5M / 3.75M / 1.875M for the depth arms).
- Kit Nibi `/project/def-maxwl/mforooz/t137/code/K1` (1f634e1). Smoke 25389044 PASS (14 rows, 1:59 wall, 4.6 GB). Array 25389757 (0–129 %30) and aggregate 25389759, COMPLETED: 1820 rows, 650 curve rows, `floor_check` 0 FAIL. Output `/project/def-maxwl/mforooz/t137/floor`; job ledger `/project/def-maxwl/mforooz/t137/jobs.tsv`. Never resubmit.
- PI ruling 2026-10-30: the floor is the "fresh replicate" context column, not the oracle for nested (depth) pairs.
- PI ruling 2026-10-30: the thinned points (ratio < 1) used 25 bp unit thinning, which makes 1 kb sums about √k too smooth (k = 4–5 units per read). They are rebuilt by whole-read thinning in a follow-up task; the ratio-1 rows and per-bin metrics stand.
- Superseded by the whole-read rebuild (t142, 2026-10-30): the thinned shape and peak-height points below 15M and the 30M extrapolation. The ratio-1 rows, the CRPS split and the per-bin metrics stand.
- Digest: `results/t137/DIGEST.md`.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [t137 digest](results/t137/DIGEST.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
