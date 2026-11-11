---
title: miniCANDI design notes
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion caveats]]", "[[onion variation and objectives]]", "[[onion eval protocol]]", "[[onion data plan]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: background information, not a plan or a requirement
---

# miniCANDI design notes

This file extends the "miniCANDI" section of [[peeling the onion]]. Every entry is background. Each was checked against its source on 2026-11-10. Source-path conventions are defined in [[onion caveats]].

Scope rule (PI, 2026-11-10): the miniCANDI architecture search starts from first principles. So this file holds no architecture evidence from our own old CANDI or CANDII ablations. Results from Max's sprints, the literature, and the CANDII covariate-correction (CAC) testbed are included.

## Index

Outputs:
- [pvalue-and-peak-derivation-test](#pvalue-and-peak-derivation-test)
- [control-rate-head](#control-rate-head)

Objectives and training:
- [joint-imputation-and-cac-background](#joint-imputation-and-cac-background)
- [loss-weighting-and-dispersion-background](#loss-weighting-and-dispersion-background)
- [masking-scheme](#masking-scheme)
- [checkpoint-selection-protocol](#checkpoint-selection-protocol)

Inputs:
- [retrieval-attention-over-reference-cells](#retrieval-attention-over-reference-cells)
- [average-baseline-input](#average-baseline-input)
- [gene-expression-cell-type-input](#gene-expression-cell-type-input)
- [mappability-and-input-dna-channels](#mappability-and-input-dna-channels)
- [covariate-vector-contents](#covariate-vector-contents)
- [decode-imputed-track-with-own-covariates](#decode-imputed-track-with-own-covariates)

---

## pvalue-and-peak-derivation-test

- **Onion section:** miniCANDI outputs. The onion text says "we can use the exact formula to go from predicted counts to signal p-value and even binary peaks".
- Max's processing-artifacts analysis found that peak-caller settings are the largest processing artifact left in p-value units. These settings are fragment extension, the control track and the control scaling.
- After a per-bin correction curve, these settings still left 3% to 41% of a track's variance unexplained, although the reads were identical. Fragment extension was the largest: ×2 left 0.31 of the variance after the curve, and ×0.5 left 0.41.
- A test of the formula route scores derived p-values and peaks against a head that predicts p-values directly, on the same targets.
- **Source:** Max s07: `rewrites/processing_artifacts/processing_artifacts.md` ("For CANDI", proposals 3 and 4).
- **Related:** [[onion caveats#pvalue-from-nb-needs-lambda]].

## control-rate-head

- The same analysis proposed a separate output for the control rate λ. The output is smooth at the 1–10 kb scale and supervised wherever a training experiment has a ChIP control.
- P-values would then follow from predicted counts and predicted λ by the MACS2 formula.
- Controls exist only for ChIP-seq (see [[onion data plan#control-availability]]).
- **Source:** Max s07: `rewrites/processing_artifacts/processing_artifacts.md`.

## joint-imputation-and-cac-background

- In the CAC testbed, a cross-assay imputation objective was added to one shared count-space correction model. The model was a FiLM CNN over 7 tracks, run with 3 seeds.
- At imputation weight 1, CRPS on never-trained covariate pairs moved by −0.025, against a pre-registered bar of 0.067 (twice the seed spread).
- At weight 4 it moved by −0.034 against a bar of 0.039. This was a near miss toward improvement, and it came mostly from scale_error.
- A masked-bin denoising objective moved CRPS by +0.007 against a bar of 0.103.
- The joint power was about 41%, so the null result is weak, and it covers one model form only.
- **Source:**
  - `CANDII-worktree-t146: results/t146/REPORT.md` §h29.
  - `CANDII: cruxvault/f30_no_objective_moves_never_trained_crps_be.md`, `h29_adding_a_cross_assay_imputation_or_maske.md`.

## loss-weighting-and-dispersion-background

- Max's sprints found that fitting the predicted mean with equal weight per bin gave the largest single training gain.
- The alternative, likelihood weighting, divides each bin's squared error by the model's own predicted variance.
- Equal weighting raised r_spec by +0.012 without RNA input and by +0.024 with it. These are asinh(−log10 p) units, on a Gaussian head.
- Under likelihood weighting, strong peaks held 36–63% of the effective loss scale, against 71% under plain asinh-p.
- In the CAC testbed, a Poisson-NLL mean, with the NB dispersion fitted on the detached mean, removed the scale bias of a misspecified log-linear form: scale_error fell from 0.253 to 0.005.
- For model forms that could follow the depth law, the gain was at seed-noise level. The NB log-likelihood and PIT calibration were unchanged in both cases.
- **Source:**
  - Max s07: `report_sections/02_model.md`.
  - Max s06: `output/summary2/rft_paired.tsv`, `output/summary2/units.tsv`.
  - `CANDII-worktree-t146: results/t127/report.md`.

## masking-scheme

- MultiMAE masks several input modalities at once. It draws the share of visible tokens per modality from a symmetric Dirichlet distribution, so some samples keep most of their tokens from one modality.
- Mitra and colleagues found that missingness in real data is often structured, i.e. it depends on which variables and sources are present. Standard methods that assume random missingness do not cover this case.
- In epigenome panels, whole assays are missing by lab and cell type, which is structured missingness.
- When a whole assay is masked, its input covariates C are also hidden.
- **Source:**
  - Bachmann R, Mizrahi D, Atanov A, Zamir A. ECCV 2022, arXiv:2204.01678.
  - Mitra R et al. Nat Mach Intell 5:13 (2023), doi:10.1038/s42256-022-00596-z.
  - `CANDII: cruxvault/wiki/set-conditioned-modelling-and-missingness.md`.

## checkpoint-selection-protocol

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- A checkpoint chosen on the validation panel (V_) gives an optimistic validation score. The blind panel (B_) gives the unbiased number.
- Background protocol: select the checkpoint and stop early on V_, and report on B_ only. Any V_ number carries the selection-bias caveat.
- **Source:** `CANDI: cruxvault/s1_deviation_training_does_not_unlock_imput.md` (protocol only).

## retrieval-attention-over-reference-cells

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- miniCANDI reads one cell type per window. Max's sprint 03 tested attention over individual reference cell types inside CANDII.
- The design followed eDICE and ChromImpute: at each 200 bp position, the target attended over training cell types that measured the target assay.
- The baseline design gave the model only the reference mean and SD as input channels. The attention design gained +0.056 r_spec over it, with 82% of tracks improving. The gain held at Borzoi test loci (+0.052).
- A trained control, with attention fixed to uniform over the same similarity-selected panel, gained +0.044. So about four fifths of the gain came from the choice of references and the per-reference pathway. The learned weights added the rest.
- **Source:** Max s03: `notes/design3.md`.

## average-baseline-input

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- **Onion section:** miniCANDI inputs, "optional/ablation: average baseline".
- The cross-cell-type average outperformed all but two of the 23 ENCODE Imputation Challenge submissions. After the single-end vs paired-end shift was corrected, more than half the entrants beat it.
- The CANDI result that ranked first on the challenge board used a per-assay blend fitted on training data, a·CANDI + b·Average + c. For H3K4me3 and ATAC-seq the CANDI weights were 0.103 and −0.067, so the blend was almost the average. This shows how much the average carries.
- An average fed to the model as an input must be built leave-one-out. Otherwise a training cell's own track is in the mean: a 1/51 share at 51 cells, and 50% for an assay that only 2 cells hold.
- **Source:**
  - `CANDII: cruxvault/wiki/average-activity-baseline.md`, `src/candi/reference.py` (module docstring).
  - Max s04: `output/candi_placement.tsv`; Max `notes/key_findings.md`.

## gene-expression-cell-type-input

- **Status:** idea proposed by the PI on 2026-11-10. A bulk RNA-seq gene-expression vector (about 20k genes) would serve as a cell-type specifier and a trans-biology input. It needs biosamples with bulk RNA-seq (see [[onion data plan#bulk-rna-seq-biosamples]]).
- Max's sprints tested two related inputs:
  1. RNA-seq coverage as an extra input track raised r_spec from 0.203 to 0.227 (+0.024). Coverage from another cell type scored 0.200, no better than no RNA. So the gain came from the cell's own transcription.
  2. A vector of 2,891 trans-regulator expression values, injected through FiLM as in Corgi+, carried no cell-specific information: own vector minus a swapped vector was +0.001. Its small training-time gain was regularization, because constant and shuffled vectors gave +0.011 and +0.024 (one seed).
- **Source:** Max s03: `notes/rna.md`; Max s07: `notes/trans.md`.

## mappability-and-input-dna-channels

- In the CAC testbed, a per-bin 36-mer mappability track was added as an extra input channel.
- It lowered CRPS in the lowest-mappability decile on all 6 histone tracks, for 36 bp read crops and for MAPQ 0.
- A copy of the channel shifted by 5 Mb kept almost none of the gain, so the effect needs the aligned track.
- Genome-wide CRPS did not change, because the gain was local to poorly mappable bins.
- A one-hot DNA input gave about a quarter of the mappability gain.
- A control that should not have moved (the depth arm) moved slightly, so the finding is weak.
- Input-DNA (ChIP control) coverage is a related candidate channel and was not tested.
- **Source:**
  - `CANDII-worktree-t146: results/t146/REPORT.md` §h31.
  - `CANDII: cruxvault/f32_the_mappability_channel_lowers_lowest_ma.md`.
- **Related:** [[onion variation and objectives#uniform-vs-position-dependent-knobs]].

## covariate-vector-contents

- Max's covariate analysis found that a correction map indexed by the arm's label was the wrong map for DNase. The DNase base had 50M reads, so its "15M" arm was a ratio of 0.3, not 0.5. The map had to be indexed by the depth ratio.
- For histone marks, a small set of monotone curves was shared across marks. The curves were indexed by depth ratio, run type, deduplication, read length and MAPQ.
- Two further options are untested ideas:
  - covariates kept per assay
  - an UNKNOWN token for lab and platform when they are not recorded
- **Source:** Max s07: `notes/covariates.md`.
- **Related:** [[onion data plan#qc-metrics-as-covariates]].

## decode-imputed-track-with-own-covariates

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- One evaluation of the old CANDI prompted the decoder with canonical covariates instead of the imputed target's own depth and processing.
- In that evaluation, imputed-count R² was near zero while Pearson correlation was about 0.5. On the same run, R² was −0.161 with canonical covariates and +0.122 with the target's own covariates.
- So a count prediction scored against a real track must be decoded with that track's own C'. Otherwise magnitude errors from the prompt are scored as model errors.
- This is an evaluation defect, not an architecture result.
- **Source:** `CANDI: cruxvault/q12_in_the_v2_backbone_why_is_imputed_count_.md`, `h23_autoresearch_e32_lifts_imputed_count_r2_.md`, `sandbox/ideas/synthesis_e32_imp_r2_autoresearch.md`.
