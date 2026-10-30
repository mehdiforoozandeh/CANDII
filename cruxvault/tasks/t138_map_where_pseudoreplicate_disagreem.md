---
id: t138
type: task
title: map where pseudoreplicate disagreement concentrates: fit signal level as the null, then read mappability/repeats, GC and peak-vs-domain context on the residual
category: implementation
parent: t118
blocked_by: t137
refs: q10
hypothesis_refs: 
status: open
created: 2026-10-29T20:08:16
updated: 2026-10-29T20:08:16
---

# t138 — map where pseudoreplicate disagreement concentrates: fit signal level as the null, then read mappability/repeats, GC and peak-vs-domain context on the residual

Refs:: [[q10_how_much_does_resampling_alone_move_a_tr\|q10]]

## Why

says whether the floor is uniform or regional, per mark class

## Decisions and runs

- PI 2026-10-29: read all four features (mappability, repeats, GC, peak-vs-domain context) on the residual after the signal-level null; download the annotations to Nibi (plan Decision iii).
- Foreman decision 2026-10-29 (G5 review): the level null is the binned conditional mean, not the median (the median null gave R² −0.30 at 3.75M).
- Annotation job 25389040 COMPLETED: `hg38.fa.gz`, `rmsk.txt.gz`, Umap `k36.umap.bedgraph.gz`, `hg38.chrom.sizes` in Nibi `/project/def-maxwl/mforooz/t138/annot` (3.7 GB, not the 1.2 GB estimated; sha256 in `ANNOT.tsv`).
- Foreman amendment A8 (2026-10-30): the hg38 v2 blacklist is masked in the map (195 800 1 kb bins dropped).
- Kit `/project/def-maxwl/mforooz/t137/code/K2` (6743ab6). Map job 25402043 COMPLETED; `map_check` 0 FAIL (260 of 260). Output `/project/def-maxwl/mforooz/t138/out`. Never resubmit.
- PI ruling 2026-10-30: the 3.75M map used 25 bp unit thinning and is rebuilt by whole-read thinning in a follow-up task.
- Digest: `results/t138/DIGEST.md`.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
