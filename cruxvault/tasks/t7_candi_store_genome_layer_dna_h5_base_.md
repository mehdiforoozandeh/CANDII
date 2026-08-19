---
id: t7
type: task
title: "candi.store genome layer: dna.h5 base codes and mask.h5 valid-bin mask"
category: implementation
parent: t3
blocked_by: t4
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:40"
updated: "2026-08-19T02:52:08"
---

# t7 — candi.store genome layer: dna.h5 base codes and mask.h5 valid-bin mask

Refs:: _(none)_

## Why

one shared DNA copy instead of 91, FASTA hash in attrs so a build mismatch is loud; the mask removes ~5% all-N windows and puts the blacklist on tiling for the first time

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- genome.py (1016) + tests/test_store_genome.py (408, 37 tests) + real build-genome in cli.py, merged 19497dd on implementation/t6-candi-store. BUILT WHOLE-GENOME ON FIR: SLURM 55534677, COMPLETED, elapsed 00:02:32, MaxRSS 3.15 GB, exit 0:0. Outputs at /project/def-maxwl/mforooz/CANDI_STORE/genome/: dna.h5 877,062,209 B (0.28 B/bp over 3,088,269,832 bp) and mask.h5 6,662,180 B (0.054 B/bin), verified present by the orchestrator. FASTA /project/6014832/mforooz/EpiDenoise/data/hg38.fa sha256 5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51; the DATA_CANDI_MERGED copy is byte-identical. Blacklist 636 intervals / 227,162,400 bp, sha 31c69342. IUPAC letters folded to N: ZERO - hg38 primary carries only ACGTN. GENOME-WIDE MASK COVERAGE 0.916786 = 113,251,272 valid of 123,530,780 bins; 6,026,108 invalid by N, 9,086,496 by blacklist, 4,833,096 by both. Per-chrom valid fraction spans chrY 0.4332 and chr22 0.6815 to chr4 0.9813; chr19 (the usual train chrom) 0.9366, chr21 (the usual eval chrom) 0.7371. ELIGIBLE WINDOWS AT min_valid_frac=0.9, reported TILED (stride=L, what a window_plan type=tile epoch actually draws from): 146,978 genome-wide at L=768 and 18,209 at L=6144; every-start (stride=1, the random-start sampling space) 112,880,377 and 111,883,465. Per-chrom at L=768 tiled: chr1 11,753, chr19 2,848, chr21 1,786. WORTH THE PI'S EYE: 2,353,634 model parameters against 18,209 non-overlapping 6144-bin windows. Independently verified: samtools faidx hg38.fa chr1:1000001-1000050 and dna.h5 decode to the identical sequence with the soft-masked run kept as real bases; verify_genome clean; a wrong fasta_sha256 raises. DECISIONS NOT IN THE PLAN: IUPAC ambiguity codes fold to N=4 and are tallied per letter into an iupac_counts attr so a large fold would be visible; chromosome set is chrom_sizes.json verbatim, chr1-22 + chrX + chrY, excluding chrM, alts, randoms and chrUn, with --chroms overriding; the fasta_sha256 check lives in GenomeLayer.__init__ before a single base is read, checking mask.h5's hash against dna.h5's always and the caller's expected hash when supplied, with build_dna verifying --fasta-sha256 BEFORE writing anything; mask.h5 carries fasta_sha256 too so a mask rebuilt against a different genome is detectable; mask N-flags come from dna.h5 rather than a second FASTA parse so the two cannot disagree; genome root attrs are their own set because layout.py::root_attrs requires biosample/kind/tracks; the eligibility threshold uses count >= frac*L - 1e-9 so an exactly-integer threshold is inclusive; dna.h5 chunk is 25600 bp = CHUNK_BINS * resolution so one DNA chunk spans one counts chunk; a gzipped FASTA is refused because build_dna mmaps. ORCHESTRATOR FIXES ON TOP: (1) t7 found a REAL SEAM BUG between t4 and t6 - genome/chrom_sizes.json is a wrapped provenance object, not the flat map layout.py::load_chrom_sizes expected, so build-biosample --chrom-sizes died with int('GRCh38'); fixed in layout.py with a regression test, which unblocks t9; (2) removed regime.py's now-unreachable _fallback_eligible_starts, a second copy of the D12 rule free to drift; (3) __init__.py exports every module lazily so layout, which needs only numpy, does not drag h5py behind it. Gates verified by the orchestrator: pytest 496 passed, golden 0 ULP params=2,353,634 sd=472362cea987.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
