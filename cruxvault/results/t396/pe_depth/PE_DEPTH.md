# What `log2_depth` counts for paired-end rows (h61, R45)

Hand-written; not an output. Source files: `rows.tsv`, `summary.tsv`, `listed_rows.tsv`,
`tag_fragments.tsv`, from `tools/t396/audits.py` (Fir job 6731380, kit SC5). Inputs: tables
`eic_v1_2` (2,740 rows), the read-start store `/scratch/mforooz/rs392/store` (`provenance.json`
per row), the read-count store (`counts.h5`, chr1), and the tagAlign tars of rs392. The 78 rows
with `in_eic` = `blind` (71 SE, 7 PE) keep their table columns; none of their store files was read.

## Answer

`log2_depth` = log2(`depth_reads`) (`CovariateEncoder.meta` in `tools/minicandi/covariates.py`,
recomputed per row in `rows.tsv`). For every PE row, `depth_reads` counts tagAlign lines, i.e. one
per mate, so it counts both mates of each fragment. The stored overlap counts and read starts count
the same lines. So, for PE rows, `log2_depth` is about 1 above log2(fragments), and it agrees with
what our count and start stores hold.

## Evidence

| run type | rows (not blind) | table depth_reads / depth_fragments | store lines_per_fragment | store lines = depth_reads | plus strand = minus strand exactly |
|---|---|---|---|---|---|
| single-ended | 2,540 (2,469) | 1.0 on 2,540 | 1 on 2,469 | 2,469 of 2,469 | 0 of 2,469 |
| paired-ended | 128 (121) | 2.0 on 115, 1.0 on 13 | 2 on 108, 1 on 13 | 121 of 121 | 111 of 121 |
| mixed | 72 (72) | 2.0 on 58, 1.0 on 14 | 2 on 58, 1 on 14 | 72 of 72 | 0 of 72 |

A ratio within 1e-4 of 2 counts as 2: an odd line count gives `depth_fragments` =
`depth_reads` // 2 (36 mixed rows, for example ENCSR327OGS crop36, 66,663,271 / 33,331,635).

1. The table and the store agree on every row: `lines_per_fragment` equals the table ratio on
   2,662 of 2,662 rows read, and the store's line count equals `depth_reads` on all of them.
2. ChIP-seq and ATAC-seq PE rows (82 + 26 = 108): ratio 2, and the per-strand read counts are
   exactly equal on all 108. Each fragment gives one plus-strand and one minus-strand line.
3. The 13 PE rows at ratio 1.0 are all `dnase-seq-pipeline` rows (ENCSR000FJH, ENCSR700ILE,
   ENCSR813CKU). Their source command is `bedtools bamtobed` on the filtered BAM without `-bedpe`,
   which writes one line per mapped mate. On ENCSR813CKU base, mapq0 and nodedup, the strands are
   exactly equal (27,943,592 each on base). On the other 10, the strand imbalance |plus - minus| /
   lines is 2e-5 to 2.5e-4; the depth views of these rows subsample lines, not fragments. So these
   rows also count both mates, and their `depth_fragments` (= `depth_reads`) is about 2x the
   fragment count. This part is moderate evidence: no read names exist in these tagAligns.
4. tagAlign check (`tag_fragments.tsv`, pr1 halves; the name column is `N`, so fragments were
   counted as adjacent line pairs on one chromosome, opposite strands, within 2 kb). ChIP PE,
   ENCSR539TKY depth0.125 and depth0.25: 100% of lines are in mate pairs, 2 lines per pair
   (478,151 pairs from 956,302 lines). DNase PE (coordinate-sorted, mates not adjacent) gave 46%
   and 46%; the SE control ENCSR486GWT gave 23% from chance neighbours. The adjacency count cannot
   decide the DNase rows. Line counts equal the store's pr1 count on all 5 files.
5. Overlap per line: chr1 overlap total / main-chromosome lines x 25 / (read length + 25) has
   median 0.088 (SE), 0.092 (PE, 2 lines per fragment), 0.087 (PE, DNase), 0.098 and 0.098 (mixed).
   Each line adds its own read span, so the count store counts each mate over its own length.
6. Mixed rows (58 at ratio 2, 14 at 1) are never strand-exact (median imbalance 1.2e-4). Their
   units differ in run type; this audit does not split a mixed row by unit.

## What MACS2 received for PE

Not determined. No source `provenance.json` records the pipeline's peak-calling command; the
pipeline work folders (`/scratch/mforooz/onion/work`) are no longer on Fir. h61's claim about the
depth "MACS2 sees" is therefore not tested here.

## For h61

The expectation (PE recorded depth about 2x the fragments in the tagAligns) holds for 121 of 121 PE
rows read: 108 by the table and exact strand balance, 13 DNase rows by strand balance (moderate).
The h61 premise that "our counts see" half that depth does not hold: the stores count the same
mates that `depth_reads` counts.

Cost: jobs 6727207 (18 min 59 s) and 6731380 (14 min 32 s), 1 CPU each, about 0.56 CPU-h
(`sacct -o Elapsed`; `TotalCPU` read 0 on this cluster).
