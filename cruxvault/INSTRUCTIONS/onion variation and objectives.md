---
title: onion variation and objectives
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion caveats]]", "[[onion eval protocol]]", "[[onion data plan]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: background information, not a plan or a requirement
---

# onion variation and objectives

This file extends two sections of [[peeling the onion]]: "sources of variation" and "objectives". Every entry is background. Each was checked against its source on 2026-11-10. Source-path conventions are defined in [[onion caveats]]. In Max's material, "Max" is the lab PI, Max Libbrecht.

## Index

Sources of variation:
- [replicate-ladder](#replicate-ladder)
- [read-sampling-noise-layer](#read-sampling-noise-layer)
- [uniform-vs-position-dependent-knobs](#uniform-vs-position-dependent-knobs)
- [artifact-source-ranking](#artifact-source-ranking)
- [population-variation-bounds](#population-variation-bounds)
- [copy-number-broad-band](#copy-number-broad-band)

Objectives:
- [calibration-terms-and-pit](#calibration-terms-and-pit)
- [single-head-aleatoric-ensemble-epistemic](#single-head-aleatoric-ensemble-epistemic)
- [preservation-vs-overcorrection](#preservation-vs-overcorrection)
- [nuisance-vs-target-covariates](#nuisance-vs-target-covariates)
- [steerability-vs-decodability](#steerability-vs-decodability)

---

## replicate-ladder

- **Onion section:** sources of variation, type 1 (data generation).
- ENCODE defines a ladder of replicate types. Each step adds one source of variation.

| Replicate type (ENCODE term) | What differs | Variation it adds |
|---|---|---|
| pseudoreplicate | a subsample of reads from one replicate | read sampling only |
| sequencing replicate | the same library run again | run or lane |
| technical replicate | the same biosample, treated identically | library preparation |
| isogenic replicate | a separate biosample from the same donor | biosample or culture |
| anisogenic replicate | a biosample from a different donor | donor |
| same assay target, other experiment (our extension, not an ENCODE term) | another experiment | lab and protocol |

- "Library preparation" for the technical replicate is our reading of ENCODE's "same biosample, treated identically".
- **Source:** ENCODE Terms and Definitions, https://www.encodeproject.org/data-standards/terms/ (read 2026-11-10).
- **Related:** [[onion caveats#pseudoreplicates-lower-bound]], [[onion data plan#collection-criteria-no-matched-cells]].

## read-sampling-noise-layer

- **Onion section:** sources of variation. This is a candidate innermost artifact layer.
- Read-sampling noise was measured as the per-bin disagreement between two pseudoreplicate halves of one library.
- The signal level of a bin explained much of this disagreement (class medians):
  - 0.18 to 0.38 at 15M reads per half
  - 0.21 to 0.32 at 3.75M reads per half
- GC content, mappability, repeats and peak/domain context together added at most 0.0003 of explained variance beyond level.
- Thus read-sampling noise follows signal level, not genomic region.
- At low depth, the worst 1% of bins concentrated in peaks and domains, at 1.2 to 1.8 times their share. This tail result was not judged.
- **Source:**
  - `CANDII: cruxvault/results/t138/DIGEST.md`, `map_classes.tsv`.
  - `CANDII: cruxvault/results/t142/DIGEST.md` (local).
- **Related:** [[onion eval protocol#max-error-decomposition-and-noise-floor]].

## uniform-vs-position-dependent-knobs

- **Onion section:** sources of variation, type 2 (data processing).
- Processing knobs fall into two groups.
- **Uniform knobs** shift every bin through one genome-wide monotone map:
  - depth
  - duplicate removal in histone marks
  - paired-end reads in histone marks
  - fragment extension, which leaves counts unchanged
- **Position-dependent knobs** change which reads map, so they act most where mapping is uncertain:
  - With the MAPQ 0 filter, the lowest-mappability decile read +0.45 to +0.48 log2 above the genome-wide map.
  - Read-length crops to 36 bp acted through mappability (+0.36 to +0.47 log2) and repeats.
  - Paired-end reads in DNase acted through mappability (+0.39 log2).
- A later model test, at weak confidence, found that a mappability input channel lowered CRPS in the lowest-mappability decile. This held for the 36 bp crop and for MAPQ 0, on all 6 tracks.
- **Source:**
  - `CANDII-worktree-t146: results/t141/DIGEST.md` §2, §4.
  - `CANDII: cruxvault/f26_counts_readings_clear_the_shuffle_and_fl.md`, `cruxvault/f32_the_mappability_channel_lowers_lowest_ma.md`.
- **Related:** [[miniCANDI design notes#mappability-and-input-dna-channels]].

## artifact-source-ranking

- **Onion section:** sources of variation, types 1 and 2.
- A controlled reprocessing test ranked artifact sources by their effect on peak height (shared peak set, read counts):

| Assay class | 1st | 2nd | 3rd |
|---|---|---|---|
| narrow histone | depth at 3.75M: 2.78 log2 | antibody proxy: 1.53 | read length 36: 1.04 |
| broad histone | depth: 2.48 | read length 36: 1.15 | antibody proxy: 0.99 |
| DNase | depth: 3.58 | duplicate removal off: 2.55 | paired-end: 1.86 |

- Every knob moved peak height 33 to 763 times more than the resampling floor did.
- Max's sprint 07 found that a per-bin curve corrects read-level artifacts in histone marks to within 0.05 of track variance. Most were corrected to within 0.02.
- Peak-caller settings (fragment extension, control choice and scaling) left 3% to 41% of variance unexplained after a per-bin curve. With a control from another cell type, the curve left 67% unexplained.
- The class values are medians over 3 tracks, or 1 track for DNase.
- **Source:**
  - `CANDII-worktree-t146: results/t139/ranking.tsv`, `results/t139/DIGEST.md`.
  - `CANDII: cruxvault/f24_peak_height_clears_the_floor_by_a_wide_m.md`.
  - Max s07: `report_sections/00_summary.md`, `report_sections/04_covariates.md`.

## population-variation-bounds

- **Onion section:** biological variation, type 4 (population level).
- Population-level variation in histone marks has measured bounds:
  - In lymphoblastoid cell lines from 75 Yoruba individuals, local histone QTLs were found for 10% to 15% of H3K4me1, H3K27ac and H3K4me3 peaks (10% FDR). This count is limited by power. It is not a heritability estimate.
  - In lymphoblastoid lines from 19 individuals of diverse ancestry, ancestry explained less than 20% of the variance at a majority of regions, for all marks.
  - In four EN-TEx donors, individual-specific peaks were usually also specific to one or a few tissues. The authors of that study said that four donors were too few for a robust analysis.
- **Source:**
  - Grubert F et al. Cell 162:1051 (2015), doi:10.1016/j.cell.2015.07.048.
  - Kasowski M et al. Science 342:750 (2013), doi:10.1126/science.1242510.
  - Hawkins-Hooker A et al. Nat Commun 14 (2023), doi:10.1038/s41467-023-40211-2.
- **Related:** [[onion caveats#population-paper-and-edice]].

## copy-number-broad-band

- **Onion section:** biological variation, and its overlap with artifacts.
- Max's error decomposition of his best imputation model split the cell-type-specific variance into spatial bands.
- The band at 102 kb and above held 8% of target variance. Of this band, 6% was sampling noise, 27% was explained and 67% was unexplained.
- Max proposed three candidate causes:
  1. copy-number differences between cell lines
  2. large chromatin domains
  3. differences in control or processing
- A planned test would correlate this residual with each biosample's input-DNA coverage, smoothed to about 100 kb. As of sprint 07, the test had not been run.
- Copy number is biology that can look like a broad artifact.
- **Source:**
  - Max s06: `output/summary2/numbers.json`, `report2.html` §6, `notes/01_handoff.md`.
  - Max s07: `notes/floors.md`, `report_sections/12_next.md`.

---

## calibration-terms-and-pit

- **Onion section:** objectives, goal 3 (uncertainty).
- Calibration has two separate parts:
  - **Level error** is the error in the predicted mean and scale. Max's decomposition calls this part "level" or "calibration".
  - **Interval calibration** asks whether observations fall in predicted quantiles at the stated rate.
- For a discrete NB forecast, interval coverage cannot reach the nominal level at low counts. A perfectly calibrated NB at a mean of 1 scored about 0.25 on an interval-coverage ECE. The exact value ranged from 0.24 to 0.33, depending on dispersion and level grid.
- The non-randomized PIT of Czado, Gneiting and Held (2009) avoids this problem. It interpolates each observation's CDF from F(y−1) to F(y).
- CANDII's `ece` is the mean absolute gap between this PIT curve and the diagonal, on a 21-point grid. CANDII reports 95% coverage separately, as a descriptive number only.
- **Source:**
  - Czado C, Gneiting T, Held L. "Predictive Model Assessment for Count Data." Biometrics 65(4):1254–1261 (2009), doi:10.1111/j.1541-0420.2009.01191.x.
  - `CANDII: EVAL.md` §"`ece` is PIT, not interval coverage", `src/candi/metrics.py::calibration_pit_curve`.
- **Related:** [[onion eval protocol#headline-metric-set]] (metric 6).

## single-head-aleatoric-ensemble-epistemic

- **Onion section:** objectives, goal 3.
- One predictive-distribution head, such as one NB per bin, models aleatoric noise only, i.e. noise in the data.
- It does not show how much the prediction would change with another training run, i.e. epistemic uncertainty. A seed or fold ensemble estimates this second part.
- In Max's benchmark, the mean of two independently trained seeds lowered normalized MSE by 0.021 (95% CI 0.018 to 0.026, 282 tracks), about 3%.
- That model predicted asinh p-values with a Gaussian head, so the size of the gain for an NB count model is not known.
- **Source:** Max s06: `output/summary/v3_paired_differences.tsv`, `report2.html`.

## preservation-vs-overcorrection

- **Onion section:** objectives, goal 2 (CAC).
- Single-cell integration work scores batch removal and biological preservation side by side.
- sysVI (Hrovatin et al., 2025) used three scores:
  - batch mixing: iLISI
  - cell-type preservation: a modified NMI
  - sub-cell-type preservation: Moran's I
- In that study, a higher KL weight removed biological and batch variation without discrimination. Adversarial alignment tended to mix unrelated cell types.
- The authors noted that integration always removes some biological information.
- For covariate artifact correction, a parallel score is the cell-type signal that survives a correction, reported next to the artifact that the correction removed.
- **Source:** Hrovatin K et al. "Integrating single-cell RNA-seq datasets with substantial batch effects." BMC Genomics 26:974 (2025), doi:10.1186/s12864-025-12126-3.

## nuisance-vs-target-covariates

- **Onion section:** objectives, goals 1.2 and 2.
- MrVI (Boyeau et al., 2025) separates sample-level covariates into two kinds:
  - Nuisance covariates, such as batch, site or technology, are corrected out.
  - Target covariates, such as donor or condition, are kept. A target covariate can be swapped at inference to produce counterfactual representations.
- DeepDive (Møller and Madsen, 2025, preprint) disentangles known covariate effects in single-nucleus ATAC-seq for counterfactual analyses.
- In CANDI's terms, lab, platform and pipeline fit the nuisance tier, and donor and condition fit the target tier.
- **Source:**
  - Boyeau P et al. Nat Methods 22:2264 (2025), doi:10.1038/s41592-025-02808-x.
  - Møller AF, Madsen JGS. bioRxiv (2025), doi:10.1101/2025.09.30.679466.
  - `CANDII: cruxvault/wiki/covariate-conditioning-and-counterfactuals.md`.

## steerability-vs-decodability

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- **Onion section:** objectives, goal 2.
- Elazar et al. (2021) showed that a property can be decoded from a representation even when the model does not use it. A causal intervention on the property tests use; a probe does not.
- The same logic applies to covariate correction: success means that changing the covariate input moves the output to the right place.
- In an old CANDI test, an apparent assay-steering effect of 0.833 came from a sentinel value that the probe moved onto empty slots. With the sentinel removed, the real effect was 0.0023. This was 43 times below the pre-registered bar of 0.10.
- **Source:**
  - Elazar Y, Ravfogel S, Jacovi A, Goldberg Y. TACL 9:160–175 (2021), doi:10.1162/tacl_a_00359.
  - `CANDI: candi_kit/research/H48_REPORT.md` §F2.
- **Related:** [[onion eval protocol#cac-evaluation-gates]].
