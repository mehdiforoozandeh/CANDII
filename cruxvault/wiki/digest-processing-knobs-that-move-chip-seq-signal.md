---
type: wiki
title: Digest — which technical covariates and processing knobs move ChIP-seq / DNase-seq signal, in counts and in −log10 p?
summary: Filed-back answer from a commissioned survey — depth, the single-/paired-end dedup rule, duplicate removal and antibody efficiency move raw counts hardest; control scaling, control identity, control depth and fragment extension move only the −log10 p track. Aligner, blacklist and the control cap do not earn an arm.
category: digest
sources: raw/claude-2026-processing-knob-survey.md, raw/jung-2014-sequencing-depth-chip-seq.xml, raw/koh-2017-coda.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/angelini-2015-chip-seq-normalization-diagnostic.xml, raw/zhang-2008-macs.xml, raw/landt-2012-chip-seq-guidelines.xml, raw/amemiya-2019-encode-blacklist.xml, raw/teng-2021-chip-seq-batch-effects.xml
created: 2026-09-16T22:55:55
updated: 2026-09-16T22:55:55
---

# Digest — which technical covariates and processing knobs move ChIP-seq / DNase-seq signal, in counts and in −log10 p?

The two signals must be kept apart: a knob can move the binned read count (a) and the MACS2 −log10
Poisson p-value against control (b) differently, and four knobs move (b) while leaving (a) bit-identical.
The survey in `raw/claude-2026-processing-knob-survey.md` is the source for the pipeline facts and for
every number whose paper is not in `raw/`; where the paper is in `raw/`, it is cited directly.

## What the ENCODE pipeline actually does (read from `chip.wdl` and the task wrappers, per the survey)

- ChIP peak/signal call is fixed: `macs2 callpeak ... -p 0.01 --nomodel --shift 0 --extsize {fraglen} --keep-dup all -B --SPMR`; all deduplication is upstream in Picard.
- The p-value track is `macs2 bdgcmp -m ppois -S {sval}` with `sval = tagAlign lines / 1e6`, so the Poisson test runs on the raw count scale — depth is **not** normalised away before (b) is computed ([[peak-calling-and-signal-tracks]]).
- DNase-seq is called with **no control**; every control-side knob below moves (b) for the six histone marks and does nothing for DNase.
- Blacklist filtering is applied to peak calls only, never to the BAM or the signal track ([[read-processing-and-artifact-regions]]).
- Defaults: bowtie2, MAPQ ≥ 30, Picard dedup, pooled control, `--slocal 1000 --llocal 10000`, `--scale-to small` (the larger library is scaled down), `subsample_reads` counts reads not pairs in paired-end mode.

## The knobs

Strength: **M** measured with numbers, **A** asserted, **I** inferred by the survey's author. "Second-hand" = the paper is not in `raw/`.

| knob | moves (a)? | moves (b) beyond (a)? | measured size | strength | settable in a pipeline re-run? |
|---|---|---|---|---|---|
| treatment depth | yes, everywhere, not a scale factor | yes: −log10 p ≈ linear in depth at fixed fold-enrichment (I) | 30M→1M: genome-wide r 0.36–0.51, within-peak r 0.18 for H3K27me3 (`raw/koh-2017-coda.xml` Tables 2–3); sufficient depth 25/35/40 M for H3K4me3/H3K36me3/H3K27me3, not reached by 55 M for H3K9me3 (`raw/jung-2014-sequencing-depth-chip-seq.xml`) | M | `chip.subsample_reads` |
| single- vs paired-end via the dedup rule | yes, site-specific, non-monotone | adds a peak-shape change quantile normalisation cannot undo | Spearman SE vs PE processing of the same experiments 0.453 mean, 0.037 for repressive marks; methods beating the naive baseline 2 → 16 of 23 after correction (`raw/schreiber-2023-encode-imputation-challenge.pdf`) | M | yes, from FASTQ |
| duplicate removal on/off | yes, concentrated in peaks | small on peak sensitivity | duplicates <10% genome-wide, 20–40% inside peaks; 20–55% for H3K4me3 vs ~5% for broad marks (second-hand: Chen 2012, Dozmorov 2015, Tian 2019) | M | `chip.no_dup_removal` |
| antibody / IP efficiency | yes, not a scale factor | peak detection collapses | 90% of reads replaced by control: r 0.42–0.65; AUPRC 0.14 for H3K27me3 (`raw/koh-2017-coda.xml` Table 5); 22% of 147 histone antibodies failed ChIP (second-hand: Egelhofer 2011) | M | only as a proxy: mix control reads into treatment |
| read length | yes, position-dependent | peak count insensitive; boundaries move in low-mappability regions | alignment rate 83.0% (36 bp) → 88.7% (101 bp) (second-hand: Zhang Q 2016) | M | `chip.crop_length`, from FASTQ |
| MAPQ / multimappers | yes, in repeats | +11–36% peaks | +17–25% effective depth (second-hand: Chung 2011); **TF ChIP only** | M | `chip.mapq_thresh` |
| control-to-treatment scaling | no | **largest pure (b) effect** | ×10 mis-scaling → ~500% more peaks (second-hand: Diaz 2012); histone peak counts move 2.5–10.5% with the constant alone (`raw/angelini-2015-chip-seq-normalization-diagnostic.xml` Table 4) | M | via control swap or subsample |
| control identity (matched / pooled / none) | no | yes | 24,422 vs 26,892 vs 91,113 peaks on the same ChIP BAMs (second-hand: Awdeh 2021); FDR 0.4% → 3.8% → 41.2% as the control and local λ are removed (`raw/zhang-2008-macs.xml`) | M | `chip.always_use_pooled_ctl` |
| control depth | no | yes, through Poisson noise in λ; the mean largely cancels under `--scale-to small` | **no paper measures it** | I | `chip.ctl_subsample_reads` — the cleanest (b)-only arm |
| fragment length / `--extsize` | no (if (a) is binned from read starts) | peak width, not summit | direction only (second-hand: Chen 2012) | M for direction | `chip.fraglen` — second (b)-only arm |
| aligner bwa vs bowtie2 | ~4% fragments | 99.8% peak overlap, below replicate noise | second-hand: Zhang, Song 2021 | M | `chip.aligner`; **not worth an arm** |
| blacklist | no | no | applied to peaks only (`raw/amemiya-2019-encode-blacklist.xml` for what the regions hold) | I | masking choice, not a counterfactual |
| control depth cap | no | inert below 200 M control reads | read from source | I | not worth an arm |
| `--slocal/--llocal` windows | no | structurally decisive, **unquantified** | — | — | MACS2 flags, not exposed as `chip.*` |
| GC bias | yes | cross-lab disagreement 24.3% → 16.9% after correction, **TF only** (`raw/teng-2021-chip-seq-batch-effects.xml`; second-hand: Teng & Irizarry 2017) | M | not switchable |
| lab / protocol | site-specific | site-specific | high-variability sites cluster by laboratory (`raw/teng-2021-chip-seq-batch-effects.xml`) | M | not switchable |
| platform / flowcell | via duplicate inflation | inherited | no peer-reviewed ChIP measurement | A | not switchable |
| fixation, sonication, cell input, DNase digestion | yes (wet-lab) | inherited | qPCR or TF-only measurements; no genome-wide histone number | M/A | new wet-lab |

## What is and is not established

Established with numbers: depth, the SE/PE dedup rule, duplicate removal, antibody efficiency, control
scaling and control identity. Established as direction only: `--extsize`. Not established at all: control
depth with the treatment held fixed, the local-λ window sizes, patterned-flowcell duplicate inflation in
ChIP, fixation time genome-wide, MAPQ and GC bias on broad histone marks.

> **Gap note.** Eleven of the papers behind the second-hand numbers above (Chen 2012, Diaz 2012,
> Liang & Keleş 2012, Awdeh 2021, Zhang Q 2016, Chung 2011, Zhang & Song 2021, Dozmorov 2015, Tian 2019,
> Egelhofer 2011, Rothbart 2015) are not in `raw/`; their numbers rest on the survey's reading of fetched
> full texts. Adding them to `raw/` would make those rows first-hand.

> **Gap note.** The claim that −log10 p is approximately linear in depth at fixed fold-enrichment is the
> survey author's Chernoff-bound derivation, not a literature result.

What would settle the open part: a re-run holding the treatment BAM bit-identical while subsampling the
control, and a sweep of `--extsize` and `--slocal/--llocal` on the same BAMs — none of these exists in
the literature.

## See also

Related:: [[sequencing-depth-and-coverage]], [[read-processing-and-artifact-regions]], [[peak-calling-and-signal-tracks]], [[distributional-shift-and-batch-effects]], [[chip-seq-assay-and-controls]], [[signal-normalization-in-epigenomics]], [[digest-depth-as-covariate-vs-divisor]], [[encode-imputation-challenge]]
