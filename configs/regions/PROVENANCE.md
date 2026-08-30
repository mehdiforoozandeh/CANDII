# configs/regions — provenance

## `encode_pilot_hg19.bed` — the source

The 44 ENCODE 2007 pilot-phase regions, UCSC's `encodeRegions` track for **hg19**.

| field | value |
|---|---|
| fetched from | `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/encodeRegions.txt.gz` |
| schema | `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/encodeRegions.sql` — `chrom, chromStart, chromEnd, name`, i.e. already BED4, 0-based half-open |
| fetched on | 2026-08-29 |
| sha256 of the `.txt.gz` as downloaded | `719df21250d925ee6b95563f7c42bddde3d87fb40ce00b6da09bf3977cad910b` |
| UCSC dump date in the `.sql` header | 2012-03-30 |
| transform applied | sorted by chromosome (chr1…chr22, chrX) then start. No field was edited. |
| sha256 of `encode_pilot_hg19.bed` | `50f17f8c7783a6d83a14f8c4526be074945581eab02a5f1d846216aa8cef8bc2` |

Content check, all reproduced from the file itself: 44 regions, **29,955,196 bp**, 21
chromosomes, 14 `ENm*` = 14,955,138 bp, 30 `ENr*` = 15,000,058 bp.

**There is no hg38 `encodeRegions` track.** Verified 2026-08-29:
`goldenPath/hg38/database/encodeRegions.txt.gz` returns HTTP 404 (the hg19 URL returns 200), and
the hg38 `database/` index lists no `encodeRegions*` file. This is why we own the liftover.

## `encode_pilot_hg38.bed` — the lift

| field | value |
|---|---|
| tool | UCSC `liftOver`, Kent tools **486** (`module load kent_tools/486` on Fir; `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/kent_tools/486/bin/liftOver`) |
| chain | `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz` |
| chain sha256 (gzipped, as downloaded) | `5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d` |
| chain size / server Last-Modified | 227,698 bytes / Wed, 01 Jan 2014 07:08:46 GMT |
| command | `liftOver encode_pilot_hg19.bed hg19ToHg38.over.chain out.bed unmapped` — **default options**, so `-minMatch=0.95`, no `-multiple` |
| run on | 2026-08-29, Fir login node |
| transform after liftOver | sorted, as above. No coordinate was edited. |
| sha256 of `encode_pilot_hg38.bed` | `13e11a198fdee08edb7797d1e402b5d985846b5a7d973ade91e8511462acb7a3` |

**Outcome: 44 in, 44 out, 0 unmapped, 0 split, 0 chromosome changes.** Every region is one
contiguous hg38 interval on the same chromosome it had in hg19.

hg38 total **29,984,074 bp** (+28,878, +0.096 % vs hg19). Of that, +37,180 bp is ENm006 alone.

Both files carry the same 44 `name` values, so a row joins across assemblies on `name`.

Full per-region lift outcome, the fine-grained (25 bp tile) audit behind it, and the independent
cross-check against Ensembl's GRCh37→GRCh38 mapper: `cruxvault/results/t79/G2_PILOT_HG38.md`.

## What these files are not

Neither file has the chr20/21/22 cut applied. The cut is Rule 2's, it belongs to the regime that
declares the eval chromosomes, and baking it into the BED would silently make the file wrong for
any other split. A consumer intersects the BED with the regime's own chromosome list.
