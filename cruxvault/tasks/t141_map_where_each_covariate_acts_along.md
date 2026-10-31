---
id: t141
type: task
title: "map where each covariate acts along the genome: arm vs base per 1 kb after the one-warp, against mappability, GC, repeats and peak context"
category: implementation
parent: t118
blocked_by: None
refs: q11
hypothesis_refs: 
status: done
created: "2026-10-29T21:25:42"
updated: "2026-10-30T04:26:34"
---

# t141 — map where each covariate acts along the genome: arm vs base per 1 kb after the one-warp, against mappability, GC, repeats and peak context

Refs:: [[q11_where_along_the_genome_does_each_covaria\|q11]]

## Why

PI 2026-10-29: tells CANDI which covariates a genome-wide scale map can correct and which need a position-aware correction

## Decisions and runs

- PI ruling 2026-10-29: the residual after the one-warp; all knobs in both spaces, control knobs in −log10 p only; the depth arm is the control, read against the base thinned to the arm's depth.
- Foreman amendment A8 (2026-10-30): the hg38 v2 blacklist is masked in this map and in the floor map.
- Foreman decision 2026-10-30 (G18 review): each reading sits beside the floor at the nearest available depth, labelled "exact" or "nearest".
- Kit `/project/def-maxwl/mforooz/t137/code/K2` (6743ab6). Smoke 25402087 PASS. Arms array 25403149 (0–122 %12) and map job 25403150, COMPLETED. Output `/project/def-maxwl/mforooz/t141`. Never resubmit.
- Finding: the depth control failed (arm 1.9–2.6× its null), because the null thinned 25 bp count units, not reads. PI ruling 2026-10-30: redo the depth null by whole-read thinning (follow-up task).
- Depth control redone by whole-read thinning (t142, 2026-10-30): it passes, the depth-arm residual equals its null in every class at 15M and 3.75M. Tables: `results/t142/t141_readthin/`.
- Digest: `results/t141/DIGEST.md`.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [t141 digest](results/t141/DIGEST.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
