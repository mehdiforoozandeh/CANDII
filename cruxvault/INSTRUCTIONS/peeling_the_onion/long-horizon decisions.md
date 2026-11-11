---
title: long-horizon decisions
kind: decisions
parent: "[[peeling the onion]]"
siblings: ["[[long-horizon prompt]]", "[[long-horizon notepad]]", "[[onion caveats]]", "[[onion variation and objectives]]", "[[onion eval protocol]]", "[[onion data plan]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: PI decisions, confirmed 2026-11-10. Binding for the long-horizon run.
---

# long-horizon decisions

The PI made these decisions on 2026-11-10, before the long-horizon run started. They close the methods-level forks that [[peeling the onion]] left open. They are binding: the agent does not reopen them. A decision that turns out to be infeasible goes to the PI as a question, with the evidence.

Precedence: [[peeling the onion]] first, then this file, then the background files. The background files hold evidence, not requirements.

## How decisions are made during the run

**D1. Delegation.** The selection rule (D13) settles the outcome of each miniCANDI ablation, e.g. rung, DNA source, covariate injection, depth offset, loss or preprocessing. These outcomes appear in the Methods section, but the PI authored the rule that settles them. The agent applies the rule and logs each outcome, with its evidence, in [[long-horizon notepad]]. The agent asks the PI only in two cases:
1. The rule gives a tie, i.e. the composite difference is within 2× the seed spread and the simpler arm is not clear.
2. The rule cannot be applied, e.g. a metric is undefined for an arm.

Any other new fork that belongs in the main Methods section goes to the PI. A fork that belongs in the code or the supplementary material is the agent's, and the agent logs it in the notepad.

**D3. Budget.** There is no compute or storage cap. The agent may use Fir and Nibi as needed. The agent reports CPU-hours, GPU-hours and storage used at each milestone. Storage quotas are physical limits: the agent reads the real quotas (`diskusage_report`) before large downloads.

## Scope

**D2. Population-level variation is out of scope.** The agent records donor identity as a covariate wherever a source gives it. There is no population evaluation and no population model arm.

**Fixed defaults.** The PI may reopen these, but the run assumes them:
- Genome build: hg38 (GRCh38).
- Bin size: 25 bp. The resolution question in [[peeling the onion]] is an analysis, i.e. a study of which variance sources live at which scales. It does not change miniCANDI's bin size in this run.
- Assay panel: the EIC 35-assay panel, so that the extended model can predict every EIC target. ChIP controls are kept for the control ablation and for λ (D9b).
- "EED" names the MERGED corpus, i.e. the extended ENCODE dataset that this run retires.

## Data

**D4. Sources.** The agent searches ENCODE, EpiATLAS (IHEC) and ChIP-Atlas, and any other portal that helps. No single source is the limit. The aim is as much data as meets the goals of [[peeling the onion]]:
- measurement counterfactuals, i.e. pairs where biology is fixed and only the measurement covariates vary
- a balanced distribution along the measurement axis
- every data view that helps separate one variation source from another

The agent logs, per source, what it adds and why, in the notepad.

**D5. Processing.** Counts are built in three layers:
1. **Uniform base.** Every experiment with public reads is reprocessed with the most standard ENCODE-DCC pipeline for its assay: chip-seq-pipeline2 for histone ChIP-seq, atac-seq-pipeline for ATAC-seq, dnase-seq-pipeline for DNase-seq. The agent records the repository, release tag and image digest.
2. **Native views.** Each source's own processed version is kept as an extra view of the same experiment, with its pipeline and version recorded as covariates.
3. **Designed counterfactual arms.** These are in-house reprocessing arms of the same reads, e.g. depth thinning, read-length crops, paired-end to single-end, duplicate removal off, MAPQ 0.

**D6. Covariate vector (two arms).** The measurement vector is C on the input side and C' on the output side.
- **Arm "metadata only" (default).** C and C' hold recorded metadata and processing settings only, e.g. depth, run type, read length, lab, platform, pipeline and version, duplicate removal, MAPQ filter, antibody, donor.
- **Arm "QC on input only".** As above, plus read-derived QC metrics (e.g. NSC, RSC, NRF, PBC, FRiP, fragment length) in C only.
- Read-derived QC never enters C'. A QC value computed from the target's own reads leaks the target.
- The other ablations assume the default arm until this comparison runs.

**D7. Use of counterfactual views.** The designed views are training data, i.e. augmentations. A held-out slice of views forms the controlled covariate-correction test set: never-trained covariate pairs, on held-out biosamples. Real cross-lab and replicate pairs are also scored.

**D11. Extended-dataset split.** The extended dataset tests generalization to new cell types:
- Whole biosample groups are reserved for validation or test. Biosamples that share an ENCODE term or a donor go to the same split.
- The split is stratified so that the measurement axis (metadata strata) and the assay axis have balanced distributions across train, validation and test.
- There is no chromosome split. Each biosample's whole genome sits in one split.
- Split fractions and the stratification search are the agent's choices, logged in the notepad.

**D11b. Test inputs on a held-out cell type.** Two regimes are scored:
1. **Leave-one-assay-out (headline).** Each target assay is imputed from all other observed assays of that cell. The headline metrics and the selection rule use this regime.
2. **Accessibility-only (second table).** Only DNase-seq or ATAC-seq is given, and the other assays are imputed. Cells with no accessibility track are left out of this regime.

## EIC comparison

**D8. EIC training data.** The EIC-comparison model trains only on the EIC training experiments, plus counterfactual views built from those same experiments' reads. A script checks every accession against the challenge's published list (this check does not exist yet; see [[onion data plan#eic-accession-facts]]). No other biosample or experiment enters EIC training.

**D9. EIC headline truth.** The headline is scored with the challenge's own metrics, against the organizers' corrected truth (Zenodo record 14736610, top level, not `old/`), for which Max's scorer reproduced the published board. Count metrics against the un-normalized single-end reprocessed tracks are secondary.

**D9b. Signal from predicted counts.** EIC entries are p-value tracks, so the agent derives signal from the predicted NB:
- **ChIP-seq.** P-values follow the MACS2 formula with λ from the blind biosample's observed control. Where no control exists (Caco-2, SJSA1), λ comes from a predicted control. Each track is labelled with its λ source.
- **ATAC-seq and DNase-seq.** These assays have no control experiment, and their ENCODE pipelines differ from the ChIP-seq pipeline. The agent follows each assay's own standard pipeline's count-to-signal route. It must not reuse the ChIP-seq route.

**D10. No output blending.** CANDI's outputs are never blended with the per-position average or with any other predictor, in the headline or elsewhere. Feeding the per-position average as an input (built leave-one-out) is an allowed ablation. PI expectation: the average-as-input arm is at least as strong as the old blended CANDI (a·CANDI + b·Average + c).

**D16. External methods.** External methods are compared on EIC only, against their published submissions. No external method is retrained on the extended dataset in this run. That retraining waits for the miniCANDI-to-CANDI scale-up.

## Evaluation

**D12. Onion layers.**
- **Primary.** Nested differences along the replicate ladder, in counts: pseudoreplicate, sequencing, technical, isogenic, anisogenic, other experiment (other lab), other cell type. Each layer is the extra variance between consecutive rungs. It is reported per assay, per region stratum and per Haar scale band.
- **Supplementary.** A crossed linear mixed model (e.g. cell type, lab, donor, depth) gives order-free variance fractions as a cross-check.
- **Data requirement.** A rung with few pairs gives wide intervals. Data collection targets enough pairs on every rung, for every assay, to give a usable interval. The agent reports the pair count and the interval width per rung.
- **Development evals.** Max's error budget (level, shared and specific, explained / unexplained / noise / added) and his explained and unexplained variance evaluation are computed for every miniCANDI arm. They guide design and optimization. They do not replace the headline metrics in [[onion eval protocol#headline-metric-set]].

**D13. Selection rule.** Every miniCANDI ablation is settled by a composite score over the three goals, i.e. a soft Pareto rule.
- Each goal's lead metric is put on a 0–1 "gap closed" scale:
  - imputation: headline metric 1, the recovered share of specific variance above the noise floor
  - covariate correction: headline metric 4, the gap closed against depth-only scaling on never-trained covariate pairs
  - uncertainty: 1 − (PIT deviation of the arm) / (PIT deviation of the marginal-NB baseline), using headline metric 6
- composite = 0.5 · imputation + 0.25 · correction + 0.25 · uncertainty
- An arm beats the incumbent when both hold:
  1. Its composite is higher by more than 2× the seed spread (at least 3 seeds).
  2. No single goal drops by more than 2× that goal's own seed spread.
- The copy-leak guardrail is a hard gate: an arm that fails it is out.
- Otherwise the simpler arm stays.
- Scores come from the validation split, leave-one-assay-out regime, in counts.

**D14. Done rule.** miniCANDI is done when both hold:
1. Rungs 1 to 6 of [[miniCANDI architecture ablations]] and their listed per-rung ablations have run with at least 3 seeds, and the selection rule has settled each comparison.
2. The final design beats the strongest training-free baseline (marginal NB, per-position average, depth-only scaling) on each of the three goals by more than 2× the seed spread. This is measured on validation, and then once on test.

If the ladder is complete and condition 2 fails, the run stops and reports. Then the run produces the /prezit deck and the 12-page /storify report that [[long-horizon prompt]] asks for.

## miniCANDI

**D15. Base covariate set for two-step correction (two arms).** Two-step correction routes (X, C) to (X_base, C_base) and then to (X', C'). C_base is the uniform pipeline of D5, at default settings.
- **Arm "native depth" (default).** The base keeps each experiment's own depth.
- **Arm "fixed reference depth".** The base is thinned to one reference depth per assay class.
- Whether two-step correction beats direct correction is a separate ablation, settled by D13.

## Run count added by the arm pairs

The two arm pairs (D6 and D15) are each tested once against the default, with the other forks held fixed. They add 2 comparisons × at least 3 seeds to the ladder.
