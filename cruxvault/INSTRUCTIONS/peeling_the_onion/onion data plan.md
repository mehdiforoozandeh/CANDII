---
title: onion data plan
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion caveats]]", "[[onion variation and objectives]]", "[[onion eval protocol]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: background information, plus one open to-do (accession check)
---

# onion data plan

This file extends the "data" section of [[peeling the onion]]. Every entry is background. Each was checked against its source on 2026-11-10. Source-path conventions are defined in [[onion caveats]]. The files `cruxvault/results/t10/manifest.json`, `t11/manifest.merged.json` and `t54/DATASET3_GAP.md` are gitignored. They exist only in the local main checkout.

## Index

- [eic-measurement-axis-imbalance](#eic-measurement-axis-imbalance)
- [collection-criteria-no-matched-cells](#collection-criteria-no-matched-cells)
- [identifiability-pointer](#identifiability-pointer)
- [per-replicate-counts-and-duplicate-files](#per-replicate-counts-and-duplicate-files)
- [raw-read-cost](#raw-read-cost)
- [qc-metrics-as-covariates](#qc-metrics-as-covariates)
- [metadata-audit](#metadata-audit)
- [what-each-source-supplies](#what-each-source-supplies)
- [bulk-rna-seq-biosamples](#bulk-rna-seq-biosamples)
- [split-by-group](#split-by-group)
- [control-availability](#control-availability)
- [spike-ins-and-personal-genome-mapping](#spike-ins-and-personal-genome-mapping)
- [exact-eic-truth](#exact-eic-truth)
- [eic-accession-facts](#eic-accession-facts) (includes a to-do)
- [counting-rule](#counting-rule)

---

## eic-measurement-axis-imbalance

The EIC training and blind splits differ along every recorded measurement axis. The table counts the 363 non-control tracks.

| | Training (T_) | Validation (V_) | Blind (B_) |
|---|---|---|---|
| tracks | 267 | 45 | 51 |
| paired-end | 26 (9.7%) | 2 (4.4%) | 51 (100%) |
| distinct read lengths | 9 (20–101 bp) | 6 (28–83 bp) | 3 (76, 100, 101 bp) |
| sequencing platforms | 9 | 6 | 1 (HiSeq 4000) |
| labs | 6 | 5 | 2 |
| median log2 depth | 24.34 | 24.38 | 26.63 |

- 39 of the 51 blind tracks are deeper than the deepest training track of their assay. In the validation split, 7 of 45 are.
- H3K27me3, H3K4me3 and H3K9me3 have blind tracks but no paired-end training track.
- **Source:** `CANDII: cruxvault/results/t10/manifest.json` (local), recomputed 2026-11-10.
- **Related:** [[onion caveats#eic-processing-mismatch]].

## collection-criteria-no-matched-cells

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In both EIC and the MERGED corpus, each (biosample, assay) cell holds exactly one experiment. None of 363 EIC cells and none of 2,684 MERGED cells has a second experiment.
- So no existing pair holds biosample and assay fixed while one covariate varies.
- On Dataset 1, Max found that 53% of 1,563 training tracks of 7 assays exceed 30M reads. 30M reads is the deepest histone level in the counterfactual testbed. A 240-track sample of the same pool gave 52%.
- **Source:**
  - `CANDI: cruxvault/q21_is_experimental_metadata_learnable_from_.md`, `h59_matched_pair_track_contrasts_measure_cov.md`.
  - Max s03: `notes/covariates.md`, `output/covariates/covariate_ranges_all.tsv`.
- **Related:** [[onion variation and objectives#replicate-ladder]].

## identifiability-pointer

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In the old CANDI data, no covariate's effect on the likelihood cleared its own noise floor. The largest effect was 1.17 times its floor.
- Run type is nearly fixed by (assay, read length). Only 6 of 83 EIC cells and 15 of 165 MERGED cells are discordant.
- A new dataset can only support covariate correction for covariates that vary independently of cell type and of each other.
- **Source:** `CANDI: cruxvault/h60_a_single_conditioning_dropout_trained_mo.md`, `q21_...`.
- **Related:** [[onion eval protocol#identifiability-table]].

## per-replicate-counts-and-duplicate-files

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In Dataset 1, the two replicate biosamples of one ENCODE experiment share one p-value track. Max found this for all 50 same-experiment pairs (Pearson above 0.99999), although their count layers differ.
- The likely source is ENCODE's replicate-pooled signal file. This was not checked against the portal.
- In MERGED, 133 file accessions occur in more than one cell, and the baked tracks are byte-identical. Within replicate groups, 112 (replicate pair, assay) entries name the same file twice.
- A replicate floor measured on these pairs reads as zero noise.
- **Source:**
  - Max s03: `notes/metrics.md`, `notes/model.md`.
  - `CANDII: cruxvault/results/t11/manifest.merged.json` (local), recomputed 2026-11-10.
  - `CANDI: cruxvault/q21_...`.

## raw-read-cost

- Any covariate other than depth needs raw reads, because the CANDI corpora hold processed signal only.
- A BAM-level change, e.g. thinning, control mixing or fragment extension, took about 1 h per product.
- A FASTQ-level rerun of the ENCODE pipeline, e.g. run type, read-length crop, deduplication or MAPQ, took a median of 13.4 h per product and up to 30.6 h.
- Each experiment's FASTQs were about 22 GB.
- **Source:** `CANDII: cruxvault/tasks/t112_build_the_counterfactual_arms_for_t.md`.

## qc-metrics-as-covariates

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- ENCODE histone standards ask for 20 million usable fragments per replicate for narrow marks and 45 million for broad marks. H3K9me3 is an exception.
- ENCODE prefers NRF above 0.9, PBC1 above 0.9 and PBC2 above 10.
- The ENCODE API also exposes NSC, RSC, JSD and estimated fragment length for each file.
- The old CANDI audit found that 17 of the 26 strongest API covariates are computed from the reads. It therefore labelled them derived features and kept them apart from looked-up metadata.
- A derived feature of the target track, given on the output side, carries information about the target itself. This is an inference, not a measured result.
- **Source:**
  - https://www.encodeproject.org/chip-seq/histone/ (read 2026-11-10).
  - `CANDI: cruxvault/q21_...`, `h66_the_api_reachable_covariate_pool_is_mate.md`.

## metadata-audit

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- The EIC metadata file has the wrong biosample accession on 214 of 363 rows.
- Seven DNase rows name an archived experiment. For two of these, the correct experiment comes from a different donor. A prepared fix of 14 cells was never applied.
- EpiClass, a classifier trained on EpiATLAS, found 23 mislabeled sex labels and 17 mislabeled life-stage labels. These were corrected in EpiATLAS. EpiClass also found 11 datasets with suspect assay labels.
- In CANDII, `research/METADATA_AUDIT.md` is an empty file, although `AGENTS.md` ranks it third in authority. The full audit is only in the old CANDI archive, `CANDI: candi_kit/research/METADATA_AUDIT.md`.
- **Source:**
  - `CANDI: cruxvault/h66_...`.
  - Raby et al. bioRxiv (2025), doi:10.1101/2025.09.04.670545.
  - `CANDII: research/METADATA_AUDIT.md` (origin/main).

## what-each-source-supplies

- **EpiATLAS:** 2,897 individuals, with uniformly processed ChIP-seq of histone marks, RNA-seq and WGBS. It has no ATAC-seq or DNase-seq. It also distributes imputed ChIP signal files, so a training or truth set must keep them apart from observed files. That last rule is our inference.
- **ChIP-Atlas:** runs every experiment through one pipeline (Bowtie 2 and MACS2). It concatenates all runs of an experiment before alignment, and it scales coverage tracks to reads per million.
- **ENCODE:** since Dataset 1 was built (2025-08-19), ENCODE released 2,079 more experiments in the queried assays: 1,680 DNase-seq and 399 RNA-seq. None of them is ChIP-seq or ATAC-seq.
- **Source:**
  - https://ihec-epigenomes.org/epiatlas (read 2026-11-10).
  - Raby et al. bioRxiv (2025).
  - https://github.com/inutano/chip-atlas/wiki ("Primary processing").
  - Max s07: `output/floors/prospective_summary.txt`.

## bulk-rna-seq-biosamples

- **Onion section:** data. The PI proposed a gene-expression vector as a cell-type input (see [[miniCANDI design notes#gene-expression-cell-type-input]]). It needs biosamples that have bulk RNA-seq.
- ENCODE has bulk RNA-seq for many Dataset 1 biosamples. Max matched it at four levels, from the same biosample accession down to the same term and donor.
- RNA was found for 37 of 45 held-out and 194 of 316 training biosamples. Most matches came from the same cell line or donor tissue, but from a different culture or tissue piece.
- 3 of 140 stranded libraries had inverted strand labels.
- ENCODE RNA signal is scaled per million reads, including mitochondrial and ribosomal reads. So nuclear coverage differed up to 8-fold between libraries.
- Added as an input coverage track, RNA raised r_spec by 0.024. Unstranded and polyA libraries gave almost no gain.
- **Source:** Max s03: `notes/rna.md`, `output/rna/rna_match.tsv`.

## split-by-group

- Max's Benchmark A v3 treats biosamples as near-duplicates when they share an ENCODE term, a donor or a store base name. Such biosamples are held out together.
- Each target is labelled seen experiment, seen cell type or held-out.
- Of 283 targets, 50 non-ATAC targets are held out in the strict sense. 18 of these come from six motor-neuron donors of a single ENCODE term.
- Uncertainty is therefore bootstrapped over the 40 terms, not over biosamples.
- **Source:** Max s03: `notes/model.md`, `notes/bzmap.md`, `output/model/benchmark_A_v3.tsv`, `scripts/metrics/leakage_square.py`.
- **Related:** [[onion eval protocol#inference-unit-and-seed-floors]].

## control-availability

- Controls exist only for ChIP-seq by design, because ATAC-seq and DNase-seq use no input control.
- 73 of the 89 EIC biosamples have a control track. Of the 16 without one, 14 have only accessibility tracks. The other two are blind biosamples (Caco-2 and SJSA1) with histone ChIP but no control.
- So the "control as an assay" ablation in [[peeling the onion]] applies to ChIP targets only.
- **Source:** `CANDII: cruxvault/results/t10/manifest.json` (local), recomputed 2026-11-10.

## spike-ins-and-personal-genome-mapping

- Bonhoure and colleagues (2014) added a fixed amount of foreign chromatin before immunoprecipitation. This spike-in gives an internal reference, so it detects a uniform genome-wide change in occupancy. An input control or a quantile step cannot detect such a change.
- Kasowski and colleagues (2013) mapped ChIP-seq reads to each individual's phased genome to reduce mapping bias. So the choice of reference genome is one more processing covariate. That paper did not measure its effect.
- **Source:**
  - Bonhoure N et al. Genome Res 24:1157 (2014), doi:10.1101/gr.168260.113.
  - Kasowski M et al. Science 342:750 (2013), doi:10.1126/science.1242510.

## exact-eic-truth

- Jacob Schreiber deposited the organizers' corrected blind truth in two Zenodo records:
  - Record 14736608 holds single-end reprocessed tracks.
  - Record 14736610 holds the same tracks after quantile normalization. Its `old/` folder holds an earlier normalization with values about 2.2 times larger.
- Max's scorer reproduced the published board against record 14736610 (top level). The median relative difference was 3e-10, and every rank was identical.
- **Source:**
  - https://zenodo.org/records/14736608, https://zenodo.org/records/14736610.
  - Max s04: `output/compare_summary.tsv`, `notes/01_handoff.md`.
- **Related:** [[onion eval protocol#score-on-normalized-and-unnormalized-truth]].

## eic-accession-facts

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- None of the 363 challenge tracks matches an ENCODE file by md5.
- 40 EIC DNase tracks hold read-depth-normalized signal, not p-values.
- The challenge scores on a ceil(length/25) grid with blacklisted bins removed. The CANDI store uses a floor(length/25) grid with the blacklist masked in place.
- Thus matching accessions does not mean matching data.
- **Open to-do:** the repository has no script that checks its accessions against the challenge's own published list.
- **Source:**
  - `CANDII: cruxvault/results/t54/DATASET3_GAP.md` (local).
  - `CANDII: DATA.md`, `src/candi/store/layout.py` (D13).

## counting-rule

- Each read adds 1 to every 25 bp bin from its start bin to its end bin, inclusive.
- A read of length L therefore adds on average (L + 25)/25 counts. So raw counts scale with read length as well as with depth.
- A 101 bp read adds about 5 counts, and a 36 bp read adds about 2.4 counts.
- **Source:** `CANDII: tools/t112/bin25.py` (docstring), `src/candi/store/layout.py` (D13); Max s03: `notes/covariates.md`.
