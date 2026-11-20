---
type: wiki
title: AlphaGenome 2
summary: Gayoso et al. 2026 (Google DeepMind): a condition-conditioned 1-Mb sequence model whose shared output channels let it impute unmeasured assays for a cell type, benchmarked against a retrained Avocado with EIC-style metrics.
category: method
sources: raw/gayoso-2026-alphagenome2.pdf, raw/avsec-2026-alphagenome.xml, raw/schreiber-2020-avocado.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/perez-2018-film.pdf, raw/hingerl-2025-scooby.xml
created: 2026-11-12T18:00:00
updated: 2026-11-12T18:00:00
---

# AlphaGenome 2

AG2 is the first model in this wiki that sits in both lineages at once: it reads DNA like the [[sequence-conditioned-epigenome-models]], and it fills missing (cell type, assay) entries like the [[epigenome-imputation]] methods, but its only cell-type input is a learned embedding, never the observed tracks themselves.

## What goes in, what comes out

Input is a 1-Mb DNA sequence (2^20 bp), a species flag (human or mouse) and one **condition** (`raw/gayoso-2026-alphagenome2.pdf`, p.2, p.20). A condition is either a **reference condition**, a row of a learned embedding table indexed by a 5-tuple of ontology CURIEs (CLO, UBERON, CL, EFO, NTR), or an **expression-conditioned** embedding, made by passing scRNA-seq counts through a separately trained cell encoder and a 2-layer adapter (p.15, p.21). The conditioning vector (256-d) drives [[film-conditioning]] after every normalisation layer of the encoder, transformer and decoder (p.21-22).

Outputs per condition (p.3 Fig. 1c; p.15-16; p.25):
- DNase, ATAC, scATAC at 1 bp: Tn5/DNase 5' insertion counts normalised to a genome-wide sum of 10^8. Not fold change.
- RNA-seq and CAGE coverage at 1 bp, stranded, same 10^8 scale.
- Histone and TF ChIP-seq at 128 bp: ENCODE **fold-change-over-control** tracks inherited from the AlphaGenome 1 corpus (`raw/avsec-2026-alphagenome.xml`), with replicates averaged per condition and channel. Linear, not log.
- CpG methylation fraction at 1 bp, contact maps at 2048 bp, splicing, and gene-level expression.

Single-end and paired-end experiments are separate output channels (p.25), a design choice the paper justifies by the EIC's single-end vs paired-end shift (p.41; see [[encode-imputation-challenge]], [[distributional-shift-and-batch-effects]]). Histone ChIP has 43 channels and TF ChIP 911 (p.25).

## How it imputes

The heads predict a fixed set of channels shared across conditions, so querying a condition returns every channel, including ones never measured there; that is the whole imputation mechanism (p.27). Unmeasured channels are masked out of the loss in training (p.28). A reference condition can be imputed only if at least one of its modalities was observed in training (p.20), because the observed assays are what shape its embedding (p.6). An unseen cell type is reachable only through expression conditioning, i.e. scRNA-seq from that cell type (p.7-9).

Contrast with the tensor methods: [[avocado]] also learns a per-cell-type embedding from that cell type's observed tracks (`raw/schreiber-2020-avocado.xml`), so the two share the "embedding fitted from observed assays" design. AG2 adds DNA sequence as the positional axis in place of Avocado's learned genomic factors. Neither takes observed signal as a forward-pass input, unlike [[edice]] and [[chromimpute]]. The expression-conditioned route is the same idea as scooby's cell-embedding decoder (`raw/hingerl-2025-scooby.xml`), but enters the trunk through FiLM rather than a generated decoder.

## The imputation benchmark

Design (p.6-7, p.20, p.41-43): an all-genome model trained with 184 test tracks (181 after merging stranded RNA pairs) and 26 validation tracks masked, across 32 human conditions; the other assays of each condition stayed in training. Genomic intervals were **not** held out for this model; scoring used fold-5 intervals (1749 x 196,608 bp, about 344 Mb). This is held-out **tracks**, not held-out cell types, the regime [[cross-cell-type-generalization-pitfall]] warns flatters within-matrix evaluation.

Metrics follow the EIC rationale and drop MSE as scale-dependent (p.41; compare [[imputation-evaluation-measures]]): genome-wide Pearson at 32 bp; **consensus-peak AUPRC**, where per channel the ENCODE narrowPeak calls of all conditions are merged, positives are peaks called in the held-out condition and negatives are peaks called only in other conditions, weighted by each track's positive fraction; the same AUPRC stratified by how many conditions share a peak; gene log-fold-change Pearson for RNA-seq.

Baselines (p.42-43): Avocado retrained at 32 bp on arcsinh signal, the channel mean (the [[average-activity-baseline]]), and a "closest" oracle that averages the five training tracks most correlated with the held-out track.

Results (p.6-7): the mean baseline reached genome-wide Pearson 0.47-0.72 across chromatin assays, "similar to Avocado (0.39–0.73)", which reproduces the EIC lesson that genome-wide correlation mostly rewards shared signal (`raw/schreiber-2023-encode-imputation-challenge.pdf`). On peak AUPRC AG2 scored 0.69-0.79 and beat all three baselines for each chromatin assay; RNA gene LFC r was 0.74 vs 0.58 (Avocado) and 0.69 (closest); AG2 beat Avocado on 166 of 181 tracks. The gain concentrates on rare peaks (called in at most 12.5% of conditions), 1.7- to 2.9-fold over the better of Avocado and closest; on near-ubiquitous peaks it is negligible.

Not compared: ChromImpute, PREDICTD, eDICE, Enformer, Borzoi, or AG1 as imputers. No EIC tracks were scored.

## Caveats the source states

- Imputed TF binding in a condition where the TF is not expressed is expected to give false-positive peaks, because training never shows such negatives (p.7).
- One forward pass per condition makes it costlier than fixed-track multitask models (p.14).
- The track-level split lists (Supplementary Table 4) and the bulk accession manifest (Table 2) are cited but not included in the PDF, which carries only Supplementary Table 1 (p.75).

## See also

Related:: [[sequence-conditioned-epigenome-models]], [[epigenome-imputation]], [[avocado]], [[film-conditioning]], [[encode-imputation-challenge]], [[imputation-evaluation-measures]], [[average-activity-baseline]], [[cross-cell-type-generalization-pitfall]]
