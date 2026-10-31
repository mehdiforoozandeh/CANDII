---
id: t111
type: task
title: re-run 7 EIC blind tracks paired-end under the single-end recipe (bwa, 30M reads, same pipeline images) on Nibi, one per assay, chosen for the largest before-vs-after reprocessing effect, and keep the filtered BAMs for both arms
category: data-acquisition
parent: t112
blocked_by: None
refs: h2
hypothesis_refs: 
status: open
created: 2026-09-16T21:53:46
updated: 2026-09-16T21:53:46
---

# t111 — re-run 7 EIC blind tracks paired-end under the single-end recipe (bwa, 30M reads, same pipeline images) on Nibi, one per assay, chosen for the largest before-vs-after reprocessing effect, and keep the filtered BAMs for both arms

Refs:: [[h2_conditioning_on_the_recorded_run_type_pr\|h2]]

## Why
one track per assay is enough for h2 if the run-type effect is large there; the finished single-end arm on Nibi has no depth- and aligner-matched paired-end partner for any of them (PI ruling 2026-09-16: not all 48, one per assay, the largest before/after effect)

**Selection (PI criterion: largest difference before vs after reprocessing, one per assay).** Ranked by
the organizers' published effect (`results/REPRO41_track_rank.tsv`, `msefam_effect` = median over the six
mse-family measures of the MOESM3→MOESM4 change), cross-checked against our own reprocessing
(`results/delta_ours_scores*.csv`: change in gwspear/gwcorr/mse of the 10 scored submissions when the
truth is swapped from the 2019 track to our single-end track, all 48 covered). The ATAC tracks (M01)
are excluded (paired-end only).

| assay | pick | published effect | ours: rank by Δgwspear | runner-up if the PI prefers ours |
|---|---|---:|---|---|
| M02 DNase-seq | **C12M02** | 1.322 (largest of all 51) | 1st (+0.158) | — |
| M16 H3K27ac | **C19M16** | 0.537 | 2nd (−0.096) | C40M16 (ours 1st, −0.104; published 0.377) |
| M17 H3K27me3 | **C40M17** | 0.691 | 3rd (−0.082) | C06M17 (ours 1st, −0.090; published 0.168) |
| M18 H3K36me3 | **C40M18** | 0.677 | 4th | C39M18 (ours 1st, −0.119; published 0.670) |
| M20 H3K4me1 | **C07M20** | 0.727 | 4th | C05M20 (ours 1st, −0.095; published 0.573; no BAM kept) |
| M22 H3K4me3 | **C19M22** | 0.260 | 3rd (−0.048) | C28M22 / C38M22 (ours tie, −0.056; published 0.048 / 0.091) |
| M29 H3K9me3 | **C07M29** | 0.356 | 5th | C31M29 (ours 1st, −0.108; published 0.079) |

All 7 picks are campaign-built, so their single-end `filter_shard0_*.nodup.bam` and control BAMs already
exist in `/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results/<track>/`; nothing single-end needs
redoing. Controls are shared per cell line (C19 ×2, C40 ×2, C07 ×2, C12 ×1), so 4 control alignments.

**Caveat recorded up front.** The before/after effect this selects on is dominated by the 30M subsample,
not the run type (`docs/02_FINDINGS.md` §8: the residual is the subsampling; the SE-vs-PE contrast there
varies aligner and run type together). Once the paired-end arm is built at the same 30M and aligner, the
run-type gap is measured for the first time; it may be smaller than the effect used to choose these tracks.

**Hold fixed between the two arms:** pipeline images (`003_pipeline/sif/`), `bwa`, `genome_tsv/v3/hg38`,
`chip.subsample_reads` 30M (the pipeline counts reads, not pairs, in paired-end mode), control
depth-limiting off, the same control accession per experiment. The only difference is the run-type
handling: mate concatenation into one single-end replicate versus paired-end alignment and the paired-end
dedup rule. Recipe and driver: `docs/04_RUNBOOK_reprocessing.md`, `003_pipeline/scripts/run_batches.py`
(`--mode paired-end`, `--aligner bwa`, and an explicit `--results`, since the default is guarded to the
single-end recipe). Then bin both arms to 25 bp raw counts plus MACS2 −log10 p with control
(`/scratch/mforooz/PVALCTRL/run.sh`, `bin_all.py` is the chr21 prototype) and land them in CANDI_STORE.

**Cost basis.** Single-end campaign: 50 leader jobs, median 13.4 h, max 30.6 h, ~22 GB FASTQ per
experiment; disk-gated at 3 concurrent. 7 leaders can run at once (PI: no 3-job limit), so about 1–2 days
wall. Scratch: PI states a 20 TB soft quota; `diskusage_report` on 2026-09-16 showed `244GiB/1024GiB` and
the earlier campaign hit a wall near 1 TiB — 7 experiments fit either way.

Overlaps t102, whose single-end half this supersedes.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
