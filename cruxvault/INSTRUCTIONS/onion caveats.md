---
title: onion caveats
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion variation and objectives]]", "[[onion eval protocol]]", "[[onion data plan]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: background information, not a plan or a requirement
---

# onion caveats

This file holds evidence that qualifies specific claims in [[peeling the onion]]. Each entry quotes the claim in the onion text, gives the evidence, and gives its source. All entries are background information. Each one was checked against its source on 2026-11-10.

Source-path conventions (shared by all sibling files):
- `CANDII:` is a path in this repository. "(local)" marks a file that is untracked or only on local disk, i.e. not on `origin/main`.
- `CANDII-worktree-t146:` is `.claude/worktrees/t146-cac-final/cruxvault/`. It holds the results of the final CAC run, which git ignores.
- `CANDI:` is the read-only archive repository at `~/Desktop/research/libbrechteam@sfu/CANDI`.
- `Max sNN:` is Max's sprint NN in GitHub `mlibbrecht/01_2026-10-19_research`, folder `projects/03_2026-10-20_epi_imputation/sprints/`.

## Index

- [two-step-cac-per-track-vs-shared](#two-step-cac-per-track-vs-shared)
- [eic-processing-mismatch](#eic-processing-mismatch)
- [pseudoreplicates-lower-bound](#pseudoreplicates-lower-bound)
- [noise-floor-unit-rule](#noise-floor-unit-rule)
- [pvalue-from-nb-needs-lambda](#pvalue-from-nb-needs-lambda)
- [population-paper-and-edice](#population-paper-and-edice)

---

## two-step-cac-per-track-vs-shared

- **Onion section:** miniCANDI. The claim is that "this model generalizes to unseen C,C' pairs if we first go from (X, C) to (X_base, C_base) and from (X_base, C_base) to (X', C')".
- **Goal:** CAC.
- **Evidence:**
  - The CAC testbed holds 7 EIC tracks. Each track was reprocessed under controlled pipeline settings.
  - A two-step correction routes a track through the base processing instead of mapping it directly.
  - **Per-track models (supported):** on never-trained arm-to-arm pairs, the chained route gave a held-out CRPS of 0.545 and the direct route gave 0.662. The seed spread was 0.008. This run used 3 seeds of an older design.
  - **One model shared across all tracks (not yet supported):** the final 6-seed run gave a pooled CRPS of 0.633 chained against 0.660 direct.
    - The chained route was lower in 6 of 6 seeds.
    - It cleared the pre-registered bar of 0.154 (twice the seed spread) in only 2 of 98 source-target cells. The expectation needed 66.
- **Source:**
  - `CANDII: cruxvault/results/t128p3/analyze_out.txt` (local), metric 1, subgroup "never".
  - `CANDII: cruxvault/f28_routing_an_existing_model_through_the_ba.md`.
  - `CANDII-worktree-t146: results/t146/REPORT.md` §h30.
  - `CANDII: cruxvault/f31_routing_the_shared_model_through_the_bas.md`.
- **Related:** [[onion eval protocol#cac-evaluation-test-sets]], [[miniCANDI design notes#joint-imputation-and-cac-background]].

## eic-processing-mismatch

- **Onion section:** data. The claim is that "an imbalanced split along the M axis, ruined encode imputation challenge".
- **Goal:** data, CAC.
- **Evidence:**
  - The ENCODE Imputation Challenge (EIC) blind truth tracks were all paired-end (51 of 51). Only 9.7% of the training tracks were paired-end (26 of 267). These counts exclude control tracks.
  - The original pipeline processed the blind tracks as paired-end, so the two sets went through different duplicate-removal rules.
  - Max reports that Picard marks a read pair as a duplicate only when both mates match. So paired-end duplicate removal is less strict. This rule was not checked against the Picard documentation.
  - After the challenge, the organizers reprocessed 48 of the 51 blind tracks as single-end. They left the 3 ATAC-seq tracks paired-end.
  - This correction changed the broad-mark truth a lot. The median correlation between original and corrected tracks was:
    - H3K9me3: 0.37
    - H3K27me3: 0.46
    - H3K4me1: 0.56
    - H3K36me3: 0.58
    - DNase: 0.73
    - H3K27ac: 0.75
    - H3K4me3: 0.84
  - On the corrected board, the Average baseline fell from 4th to 18th of 26.
- **Source:**
  - `CANDII: cruxvault/results/t10/manifest.json` (local), field `run_type`.
  - Max s04: `output/track_pairs.tsv`, `output/candi_placement.tsv` (column `average_position`), `notes/00_sprint_log.md`.
  - Max s07: `notes/covariates.md` (Picard rule).
- **Related:** [[onion data plan#eic-measurement-axis-imbalance]], [[onion eval protocol#score-on-normalized-and-unnormalized-truth]].

## pseudoreplicates-lower-bound

- **Onion section:** eval. The claim is that "pseudoreplicates help establish the poisson-error component".
- **Goal:** eval, uncertainty.
- **Evidence:**
  - ENCODE defines a pseudoreplicate as a subsample of reads, drawn without replacement, from a single replicate.
  - So two pseudoreplicate halves come from one library, one biosample and one donor. They differ only by read sampling.
  - With duplicate removal on, the halves do not share PCR copies of a fragment. They still share the library's amplification and other library effects.
  - Pseudoreplicates therefore measure sampling noise only, and they give a lower bound on replicate noise. The "lower bound" wording is our inference; the ENCODE page does not say it.
  - Max found that true replicates sat a median 5% above the pseudoreplicate prediction (6 experiments).
- **Source:**
  - ENCODE Terms page, https://www.encodeproject.org/data-standards/terms/ (read 2026-11-10).
  - Max s07: `notes/floors.md`, `rewrites/processing_artifacts/processing_artifacts.md`.
- **Related:** [[onion variation and objectives#replicate-ladder]].

## noise-floor-unit-rule

- **Onion section:** eval.
- **Goal:** eval, uncertainty.
- **Evidence:**
  - Noise floors measured in asinh(−log10 p) do not carry over to NB counts.
  - MACS2 scores a bin as −log10 of the Poisson upper-tail probability of its pileup, given a local background rate λ.
  - In a background bin the pileup is at or below λ, so the score is near 0 in both pseudoreplicate halves. In that unit, background noise therefore reads as near zero. Background bins (asinh-p < 0.25) are 61–82% of bins.
  - In raw counts, most of the variance at 25 bp is background sampling noise.
  - A noise floor holds only in the unit that the model predicts. A count model therefore needs floors measured on counts.
- **Source:**
  - `CANDII: src/candi/store/pval_from_counts.py` (docstring).
  - Max s06: `report2.html` (units figure), `output/summary2/units_summary.tsv`.
  - Max s07: `rewrites/processing_artifacts/NOTES.md`.
- **Related:** [[onion eval protocol#max-error-decomposition-and-noise-floor]].

## pvalue-from-nb-needs-lambda

- **Onion section:** miniCANDI outputs. The claim is "we can use the exact formula to go from predicted counts to signal p-value and even binary peaks".
- **Goal:** CAC, uncertainty.
- **Evidence:**
  - MACS2 computes a signal p-value as a Poisson upper tail at a local background rate λ. With a control, λ comes from the control pileup, scaled by the treatment-to-control depth ratio.
  - A formula that derives p-values from a predicted NB therefore needs λ, i.e. a predicted or observed control.
  - On true counts, the log CRPS was:
    - fragment-window formula: 0.0865
    - formula plus a learned MLP correction: 0.0798
  - On counts predicted by a covariate-correction model, every p map gave a CRPS on P of 0.18 to 0.28. A model that predicted p directly gave 0.14 to 0.17. These are a different metric from the log CRPS above, so the two pairs are not comparable.
  - All of these results come from one track and one model seed.
  - The CANDII record holds no evaluation of peaks called by formula from a predicted NB.
  - On 2026-10-31 the PI ruled that p is derived from counts by the formula.
- **Source:**
  - `CANDII: cruxvault/q6_how_closely_does_a_map_h_reproduce_the_l.md`, `cruxvault/f18_clear_margins_over_seed_wobble_but_one_t.md`.
  - `CANDII: cruxvault/results/t125/h_vs_oldgrid.tsv` (local).
  - `CANDII: src/candi/store/pval_from_counts.py`, `EVAL.md` §B.
- **Related:** [[miniCANDI design notes#pvalue-and-peak-derivation-test]], [[miniCANDI design notes#control-rate-head]].

## population-paper-and-edice

- **Onion section:** objectives 1.2, population-level prediction.
- **Goal:** imputation (population axis).
- **Evidence:**
  - The cited preprint is Tu, Spiro, Chikina and Mostafavi (bioRxiv 2026), "A modality gap in personal-genome prediction by sequence-to-function models".
  - It is not an imputation method. It benchmarks AlphaGenome and Borzoi on predicting variation between individuals from their personal genomes.
  - For chromatin accessibility, the mean correlations were:
    - AlphaGenome: 0.627
    - Borzoi: 0.448
    - cis-genotype elastic-net baseline: 0.758
  - These were 1,852 lymphoblastoid ATAC peaks, chosen in advance as heritable. The paper therefore gives ceilings and baselines, not an imputer.
  - eDICE (Hawkins-Hooker et al., Nature Communications 2023) is the individual-level imputation precedent. Its individual-specific test used four EN-TEx donors and covered chromosome 21 only.
- **Source:**
  - Tu X, Spiro A, Chikina M, Mostafavi S. bioRxiv 2026, doi:10.64898/2026.02.01.702969 (local copy `CANDII: cruxvault/raw/mostafavi-2026-modality-gap.pdf`).
  - Hawkins-Hooker A et al. Nat Commun 14 (2023), doi:10.1038/s41467-023-40211-2 (`CANDII: cruxvault/wiki/edice.md`).
- **Related:** [[onion variation and objectives#population-variation-bounds]], [[onion eval protocol#within-locus-population-metric]].
