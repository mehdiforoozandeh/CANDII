---
id: t3
type: task
title: build the whole-genome CANDI_STORE corpus for EIC and MERGED
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:28"
updated: "2026-08-19T06:45:04"
---

# t3 — build the whole-genome CANDI_STORE corpus for EIC and MERGED

Refs:: _(none)_

## Why

replaces the window-materialized h5 bake; 12 GB EIC / 81 GB MERGED vs ~1 TB / ~4 TB naive, ~10k windows/s

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [CANDI_STORE, built — 461 GB on Fir, nine children done](results/t3/DELIVERABLE.md) — CANDI_STORE is built and complete at /project/def-maxwl/mforooz/CANDI_STORE, all nine children t4-t12 done. THE STORE: genome/ 883,787,968 B (dna.h5 877 MB, mask.h5 6.7 MB, chrom_sizes.json, hg38-blacklist.v2.bed), shared by both corpora; eic/ 54,942,948,927 B = 54.94 GB (89 biosamples, 436 tracks, 35 assays); merged/ 406,058,086,308 B = 406.06 GB (361 biosamples, 3026 tracks, 47 assays); plus eic_slice/ 4.99 GB, the 5-biosample evidence store from t9. All three kinds everywhere, 23 chroms, 121,241,684 bins per track, verify OK on both corpora. 900 SLURM tasks across six arrays, ZERO failures. THE CODE: src/candi/store/ - layout, writer, manifest, cli (t6), genome (t7), reader, regime, dataset (t8) - on branch implementation/t6-candi-store, PR #11, with STORE.md as the contract doc. Merge gate held at every commit: pytest 497 passed (321 baseline + 176 new), tools/golden.py bit-exact at 0 ULP, params=2,353,634 sd=472362cea987, so the old bake path is provably untouched per D21. FIVE CORRECTIONS TO THE PLAN'S SECTION 2, all measured: EIC has 89 biosample directories not 91 and MERGED has 361 not 367 - both were raw ls counts sweeping in aliases.json, navigation.json and the hg38 reference files; the store is 461 GB not the predicted ~410 GB, because section 2's counts and pval totals were built on different column sets (the 12 GB figure omits the 76 EIC control columns and dna+mask despite reading 'counts + peaks + dna') and because D7 uint32 is not an edge case - 22 of 89 EIC and 57 of 361 MERGED biosamples exceed the uint16 ceiling, real DNase-seq reaching 448,686 reads in one 25 bp bin; loader throughput is 404 windows/s single-threaded, not the ~10k the plan cites, because that figure was a raw h5 slice-read rate that never included DNA one-hot, binomial thinning or batch assembly; /localscratch is 7.0 TB per node, not 561 GB; and the plan's section 5 list of 13 batch-dict keys is incomplete, the real CandiKitH5Dataset emitting 18. TWO GENUINE BUGS FOUND AND FIXED: layout.py::load_chrom_sizes could not read the wrapped genome/chrom_sizes.json its own CLI fallback points at, and manifest.py::read_file_metadata raised on any dict-valued field with more than one key, which one EIC track's free-text notes dict would have turned into a corpus-wide build-manifest failure. Both have regression tests. TWO METADATA RECOVERIES the plan scoped at one: t5 built control_metadata.csv for MERGED (361 rows) and t10 built eic_control_metadata.csv for EIC (76 rows, a gap t9 discovered), together closing 437 control columns that would otherwise have trained as all-MISSING; depth, absent from every file_metadata.json, was recovered from the per-DSF metadata.json and proven identical to the signal CSVs' depth at 58/58 on MERGED and 363/363 on EIC. THE D9 CEILING IS SAFE: across 3030 pval tracks the maximum clipped fraction is 0.003747, about 2.7x under the 0.01 threshold at which top-of-signal flattening would matter. WHAT IS STILL OPEN AND NEEDS THE PI: whether to keep the Boyle-Lab blacklist or the ENCODE portal's ENCFF356LFX, which is a different and 3.2x less aggressive list and would move the mask by ~5 points of the genome; the loader throughput gap, since 404 windows/s against 18,209 non-overlapping 6144-bin windows genome-wide is the real training-scale constraint; cell_cond, which needs a cell identity D16 forbids parsing off a biosample name; the MERGED train/test split, deferred by D17; and retiring prep/bake.py, which stays until a real training run has used the store.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
