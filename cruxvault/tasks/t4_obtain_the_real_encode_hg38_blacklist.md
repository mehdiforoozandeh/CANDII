---
id: t4
type: task
title: obtain the real ENCODE hg38 blacklist v2
category: data-acquisition
parent: t3
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:28"
updated: "2026-08-19T02:08:36"
---

# t4 — obtain the real ENCODE hg38 blacklist v2

Refs:: _(none)_

## Why

EpiDenoise/data/hg38_blacklist_v2.bed is 8 bytes containing '# Empty' — the valid-bin mask cannot be built without a real one

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [hg38-blacklist.v2.bed, verified](results/t4/hg38-blacklist.v2.bed) — hg38-blacklist.v2.bed (Amemiya 2019, Boyle-Lab github) at /project/def-maxwl/mforooz/CANDI_STORE/genome/, sha256 31c69342df43bbc19dd8ef2886611a8150cb53e90f341de14d30e742c9251737, 26898 B. Verified independently: 636 intervals over chr1-22,X,Y, already merged and non-overlapping, 0 intervals past a chrom end, 227,162,400 bp = 7.356% of the primary genome (6.441% over the 23 chroms EIC uses). chrom_sizes.json written beside it from hg38.chrom.sizes; n_bins[chr1]=9958256 matches STORE_PLAN section 2 exactly. TWO CORRECTIONS TO THE PLAN: (1) ENCFF356LFX is NOT a mirror of the Boyle-Lab file but a different, 3.2x less aggressive list (910 intervals, 2.32% coverage, only 5/636 intervals agree) - Boyle-Lab kept because cruxvault/wiki/read-processing-and-artifact-regions.md cites it, but choosing ENCODE instead would move the D11 mask by ~5 points of the genome, so it is a live PI call; (2) the '8-byte # Empty' premise is false - four pre-existing copies on Fir and locally are byte-identical to this fresh download, independent evidence the corpus was already processed against this exact list. Bed is stored verbatim as decompressed upstream so the sha256 is checkable against the source; it is ordered lexicographically, not karyotypically, so t7 must group by chromosome rather than assume order. Evidence in cruxvault/results/t4/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
