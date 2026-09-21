---
id: t112
type: task
title: "build the counterfactual arms for the in-vitro testbed: 10 processing knobs on the 7 selected EIC tracks, one knob per arm, FASTQ re-runs where the knob sits before alignment, BAM-level re-derivation otherwise, all launched concurrently on Nibi"
category: data-acquisition
parent: 
blocked_by: None
refs: h1, h2
hypothesis_refs: 
status: done
created: "2026-09-16T23:25:09"
updated: "2026-09-21T16:31:57"
---

# t112 — build the counterfactual arms for the in-vitro testbed: 10 processing knobs on the 7 selected EIC tracks, one knob per arm, FASTQ re-runs where the knob sits before alignment, BAM-level re-derivation otherwise, all launched concurrently on Nibi

Refs:: [[h1_conditioning_on_the_recorded_sequencing_\|h1]], [[h2_conditioning_on_the_recorded_run_type_pr\|h2]]

## Why

gives h1 and h2 matched pairs that differ in exactly one recorded processing covariate, and gives the testbed p-only arms that separate the count head from the p head

## What we want

For each of the 7 tracks (C12M02 DNase, C19M16 H3K27ac, C40M17 H3K27me3, C40M18 H3K36me3, C07M20 H3K4me1,
C19M22 H3K4me3, C07M29 H3K9me3; selection in t111) a **base** and, per knob, one or more **arms** that differ
from the base in exactly one processing setting. Every base and arm yields the same two products over the
same 25 bp bins, genome-wide: raw treatment read-start counts from the filtered BAM, and the MACS2 −log10 p
against control (`macs2 bdgcmp -m ppois`, raw-count scale, as the ENCODE pipeline does). Each product carries
a covariate record: the four CANDI covariates plus the knob's value. Base-to-arm pairs are the testbed's
source→target pairs; the knob value is the covariate the testbed is conditioned on.

Knob list and evidence: `[[wiki/digest-processing-knobs-that-move-chip-seq-signal]]` (survey in
`raw/claude-2026-processing-knob-survey.md`). PI rulings 2026-09-16: FASTQ re-run wherever the knob needs it;
launch everything at once, no concurrency cap; Nibi scratch soft quota is 20 TB (PI) — `diskusage_report`
showed `244GiB/1024GiB` the same day, so confirm before staging ~30 FASTQ sets.

## The 10 arms

Base = the finished single-end recipe (bwa, 30M reads, Picard dedup, MAPQ ≥ 30, matched control, xcor
fraglen), whose filtered BAMs already exist at
`/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results/<track>/`. Levels are proposals; fix them before launch.

| # | knob | route | setting | levels | moves counts / p |
|---|---|---|---|---|---|
| 1 | treatment depth | BAM thinning (binomial, seeded) | reads kept | 15M, 7.5M, 3.75M | both (h1 arm) |
| 2 | run type | FASTQ re-run | `--mode paired-end`, mates as R1/R2 | PE | both (h2 arm) |
| 3 | duplicate removal | FASTQ re-run | `chip.no_dup_removal = true` | off | both |
| 4 | antibody / IP efficiency proxy | BAM mixing | replace fraction f of treatment reads with control reads, total reads fixed | f = 0.5, 0.9 | both |
| 5 | read length | FASTQ re-run | `chip.crop_length` | 36, 50 | both |
| 6 | MAPQ | FASTQ re-run | `chip.mapq_thresh` | 0, 10 | both |
| 7 | control-to-treatment scaling | BAM-level MACS2 | `--ratio` at k × the default ratio | k = 0.5, 2 | p only |
| 8 | control identity | BAM-level MACS2 | matched → another cell line's control → no control | 2 levels | p only |
| 9 | control depth | BAM thinning of the control | fraction of control reads | 0.5, 0.25 | p only |
| 10 | fragment extension | BAM-level MACS2 | `--extsize` at k × xcor fraglen | k = 0.5, 2 | p only |

Arms 7–10 leave the counts bit-identical: the testbed's count head must predict no change and its p head
must predict one. C12M02 is DNase, called with no control, so arms 7–9 do not apply to it. Arm 1 joins
CANDI's existing DSF ladder in kind. The planted-warp apparatus check is t106, not an arm here.

## Plan

1. **Stage.** Confirm scratch quota. Download FASTQs for the 7 experiments and their 4 controls
   (~22 GB each, measured) to `/scratch/mforooz/EIC_REPRO/cf/fastq/`.
2. **FASTQ arms (2, 3, 5, 6): 7 × 6 = 42 Caper leaders**, all submitted at once. Build input JSONs with
   `003_pipeline/scripts/build_input_jsons.py`, patch the one field per arm, and give each arm its own
   `--results` (the default results dir is guarded to the base recipe). Keep `filter_*.nodup.bam` (and the
   unfiltered BAM for arm 6) in every arm. Median leader wall 13.4 h, max 30.6 h measured, so ~1–2 days if
   the queue admits them together.
3. **BAM arms (1, 4, 7–10): ~90 MACS2 + binning jobs**, a SLURM array, ~1 h each measured (37–55 min for
   the signal step at 30M). Reuse `/scratch/mforooz/PVALCTRL/run.sh` and `bin_all.py`, generalised from
   chr21 to genome-wide and from one setting to the arm table. Seeded thinning; record seeds.
4. **Products.** Per (track, arm, level): `counts25.npz`, `pval25.npz`, `covariates.json`, `provenance.json`
   (input JSON or MACS2 command, md5s, job id). One manifest TSV over all products. Land in CANDI_STORE layout
   so the testbed's loader reads them unchanged.
5. **Checks before use.** Base products rebuilt from the kept BAMs must match the pipeline's own
   `pval.signal.bigwig` on chr21 (the PVALCTRL prototype is this check); arms 7–10 must show bit-identical
   counts to base; arm 1 must reproduce the store's `thin_counts` law in expectation.
6. **Storage.** FASTQ ~1 TB transient; unfiltered + filtered BAMs for 42 arm runs ~1 TB retained until binned;
   products small. Everything on scratch (60-day purge) until the products are copied to `/project`.

Children: t111 (arm 2). Every other arm lands on this task's branch.


## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [manifest of all 130 products](results/t112/MANIFEST.tsv)
- [counterfactual arms write-up](results/t112/COUNTERFACTUAL_ARMS.md)
- [checks](results/t112/CHECKS.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
