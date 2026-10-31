---
id: t142
type: task
title: rebuild the thinned floor, its 1 kb readings and the t141 depth null by whole-read thinning, and check against the real depth-arm halves
category: implementation
parent: t118
blocked_by: None
refs: q10, q11
hypothesis_refs: 
status: done
created: "2026-10-30T00:28:24"
updated: "2026-10-30T04:26:34"
---

# t142 — rebuild the thinned floor, its 1 kb readings and the t141 depth null by whole-read thinning, and check against the real depth-arm halves

Refs:: [[q10_how_much_does_resampling_alone_move_a_tr\|q10]], [[q11_where_along_the_genome_does_each_covaria\|q11]]

## Why

PI 2026-10-30: 25 bp unit thinning makes 1 kb sums about sqrt(k) too smooth (k = 4-5 count units per read); whole reads fix the floor's thinned points, the 3.75M map and the depth control

## Decisions and runs

- PI ruling 2026-10-30: (i) rebuild the thinned floor points (ratio < 1) and their 1 kb readings (shape, peak height, the 3.75M map) by thinning whole reads from the split read files on Nibi `/scratch/mforooz/t112_cf/pseudoreps/_readsets/`; (ii) redo the depth-arm null of the covariate map the same way; (iii) compare the real depth-arm halves (7.5M, 3.75M, 1.875M) with the corrected thinned floor.
- Foreman amendment A10 (2026-10-30): the code is chunk G23, which first reproduces the t117 25 bp counting bit-exact. The ratio-1 rows and the per-bin metrics of the floor stand.
- Kit Nibi `/project/def-maxwl/mforooz/t142/code/K1` (8392847). Smoke 25416193 PASS (bit-exact against the t117 counts). Read-thinning array 25416644, floor 25416645, floor aggregate 25416646 (1820 rows, 0 FAIL), depth arms 25416647 (21 arms), all COMPLETED. Real-halves comparison on the login node: `/project/def-maxwl/mforooz/t142/compare/`.
- Floor map rerun 25425796, COMPLETED, output `/project/def-maxwl/mforooz/t138/out_readthin`. Covariate map rerun 25425799 FAILED (it expects all 123 arms); the 102 unchanged non-depth arms were linked in from `t141/arms1k`, resubmitted as 25426840, COMPLETED. Never resubmit.
- Readings: real halves match whole-read thinning (shape 1.002, unit thinning 1.226; 21 arms); the depth control passes. Digest: `results/t142/DIGEST.md`.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [t142 digest](results/t142/DIGEST.md)
- [whole-read covariate-map tables](results/t142/t141_readthin/knob_classes.tsv)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
