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

## `eval_random450_seed890217.bed` — the mid-training SELECTION scope

**This is not a training scope and not a published region set.** It is 450 windows drawn by a seed
from the full-coverage eval window plan, and its only job is to make the mid-training check cheap
enough to run every `--eval-every` epochs without changing which checkpoint gets selected.

### Why a seeded draw and not the Pilot Regions

`encode_pilot_hg38.bed` was tried first, on the argument that a named, fixed, published set chosen
in 2007 is one nobody here could be accused of picking to flatter a method. Measured against full
coverage on an 8-checkpoint ladder (32 tracks, chr21), it **flipped 4 of 15 checkpoint
comparisons**, at gaps up to 0.069 — because its bias *drifts with the model* (+0.3044 … +0.4144
across six checkpoints) instead of cancelling in a difference. **Pedigree is not faithfulness.**
Who chose the loci governs whether the choice is corrupt; it says nothing about whether the loci
track the genome-wide number, which is the only property selection needs. `EVAL.md` has the table.

Subsetting at this size was never the problem: 200 random matched subsets centre on the
full-coverage value to a tenth of a standard deviation, and 194/200 reproduce the full checkpoint
ordering exactly. So the loci here are chosen by a seed, and the seed is committed before any
number is produced with it.

### The draw, exactly

| field | value |
|---|---|
| generator | `tools/make_eval_scope_bed.py make` |
| seed | **890217** |
| windows | **450**, drawn uniformly WITHOUT replacement, **no stratification** |
| pool | `harness.full_tiling` of the eval chromosomes — **8,437** starts, the same chromosome-0-anchored grid the full-coverage pass walks |
| chromosomes and bin counts | `chr20=2577766, chr21=1868399, chr22=2032738` (from `CANDI_STORE/eic`) |
| context_bins / resolution | 768 / 25 bp (one BED interval = one 19,200 bp window) |
| draw split, measured | chr20 199, chr21 117, chr22 134 |
| scored bins | **345,600 of 6,478,903 = 5.334 %** |
| sha256 of the BED | `24d9cb9cf6f5db696e4a47e85960f4cee7cc1c98a5bdbd75f1976f755b214ea7` |

**No stratification, deliberately.** Not by chromosome, not by signal, not by anything. The uniform
draw is what the evidence covers, and every stratification is a choice we made — which is the thing
the seed exists to remove. The Pilot Regions are what a well-motivated non-uniform choice looks like
when it goes wrong.

**450 rather than 225.** 225 (the Pilot-sized scope) already resolves the 0.03 margin with headroom.
450 was taken because the headroom exists inside the one-epoch budget and it buys insurance against
a comparison turning on a gap tighter than any in the ladder that was measured.

### Regenerating it

Reproducible from the recorded numbers alone — no store needed, because the bin counts are in the
table above:

```bash
python tools/make_eval_scope_bed.py check --windows 450 --seed 890217 --context-bins 768 \
    --n-bins chr20=2577766,chr21=1868399,chr22=2032738 \
    --bed configs/regions/eval_random450_seed890217.bed
```

`tests/test_store_regime.py` runs that same re-derivation, so a BED edited by hand fails the suite.

### What it is not

Not a training scope — the train split's region rule is D32's `regime.regions`, a different key on a
different split. Not a leaderboard scope — the end-of-run check and every published number stay full
coverage. Not a substitute for full coverage in any quoted result: a scoped curve carries
`eval_scope` in its provenance, and two runs are comparable on it only if that block matches.
