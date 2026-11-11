# Candidate additions to "peeling the onion" and "miniCANDI architecture ablations"

2026-11-10. This is a menu, not an edit. Neither source document was changed.

**Sources.** Five read-only research agents covered these sources:
1. Max's sprint repo (`mlibbrecht/01_2026-10-19_research`, sprints 02-08). Its tag is **[Max]**.
2. The CANDII CAC testbed (findings f1-f33, results t112-t150). Its tag is **[CAC]**.
3. CANDII eval, data and store docs, the leaderboard and `src/`. Its tag is **[C2]**.
4. The old CANDI archive (vault h1-h83, sandbox, METADATA_AUDIT). Its tag is **[C1]**.
5. The literature wiki and the web. Its tag is **[Lit]**.

**How to read an item.** Each item gives four things:
- **Target**: the document and section it belongs in. "Onion" is the peeling-the-onion document, and "Abl" is the ablations document.
- **Goal**: the goal it serves, i.e. imputation, CAC, uncertainty, eval, data, architecture or guardrail.
- The addition.
- The evidence, with its source path.

Item codes (K1, C4, ...) are menu numbers only.

**Caveats on the evidence.**
- Most of Max's numbers are in asinh(-log10 p) units at 25 bp, on Benchmark A v3. They do not carry over to NB counts directly.
- Most CAC numbers come from one simulated testbed with 7 tracks and 3-6 seeds.
- Most old-CANDI numbers are from a single seed, on an 8-assay panel, with training on chr19 only.

---

## K. Claims in the two documents that the evidence undercuts (decide these first)

**K1. Two-step CAC generalizes per track, but only weakly for one shared model.** Onion / miniCANDI CAC paragraph. Goal: CAC, guardrail.
- Per-track models, on never-trained pairs: chained 0.545 vs direct 0.662 CRPS, seed spread 0.008.
- One shared model (t146, 6 seeds): pooled 0.633 vs 0.660. The model was lower in 6 of 6 seeds, but it cleared the 0.154 bar in only 2 of 98 cells, and 66 were needed.
- Suggested wording: "per track, supported; shared model, not yet."
- Evidence [CAC]: f19, f28, f31; `results/t128p2`, worktree `results/t146/REPORT.md`.

**K2. EIC's failure was a mismatch between how the truth and the training data were processed. The published CANDI win is a blend with the Average baseline.** Onion / data. Goal: guardrail.
- The blind truth was 100% paired-end. The training tracks were 9.7% paired-end.
- The single-end correction moved the Average baseline from 4th to 18th place.
- Uncorrected CANDI ranks 14th, 17th and 16th of 26 on the three truths. "CANDI + tcfloor" ranks 1st, and it is a*CANDI + b*Average + c. For H3K4me3 and ATAC the CANDI weight is 0.10 and -0.07.
- The new paper claim needs a run with no post-hoc blend, quoted with its margin.
- Evidence [Max]: S04 `output/candi_placement.tsv`, `track_pairs.tsv`, key_findings item 16. [C2]: `leaderboard/eic/site/index.html`.

**K3. Replicate types will separate the variance layers only weakly.** Onion / eval and data. Goal: guardrail, data.
- True replicates sat a median 5% above the pseudoreplicate prediction (6 experiments).
- Experiment-specific variance beyond sampling noise is about 0.05 of the variance (46 same-term targets).
- Pseudoreplicates give only a lower bound on noise. Halves share the library and the PCR duplicates, and each half has half the depth.
- Evidence [Max]: S07 `notes/floors.md`, S06 `related_vs_sampling.tsv`. [Lit]: ENCODE terms page.

**K4. The noise floor is 0.28, not 0.38. Floors measured in asinh-p do not carry over to counts.** Onion / eval. Goal: guardrail.
- The 0.38 value came from one replicate's BAM record count. The target is the pooled track, which is about 1.86x deeper.
- The corrected unrecoverable share is 0.28 over 282 targets.
- In asinh-p, background bins have p = 1 in both halves, so the background noise reads as zero. This is an artifact of the unit.
- Max's handoff says he still has to tell you this.
- Evidence [Max]: S07 `output/floors/floors_headline.tsv`, `notes/01_handoff.md` §8.

**K5. "Depth is exactly additive in η" is an assumption, not a measured fact.** Abl / preprocessing #1. Goal: CAC, guardrail.
- In the old CANDII count head, the slope of predicted level on target depth was 1.18-1.29, not 1.
- The predicted level correlated +0.17 to +0.59 with the depth of the *input* track. The truth correlated 0.00.
- When the offset is on, depth steering is arithmetic: the slope is 1.0000 with no learning.
- Offset on vs off is a trade-off. Offset off learns steering, but macro CRPS rises 27-42%.
- In the CAC testbed, a g that reads covariates bent the depth law anyway. Design D read 0.87-1.45, and its no-covariate twin read 0.05-0.14.
- Evidence [Max]: S03 `v2_level_diagnostics.tsv`. [C1]: h41, h46, h47. [CAC]: f17, `t125/part1_digest.md`.

**K6. "Signal p-values and peaks by formula from the NB" needs the control rate λ, and it was never tested on model output.** Onion / miniCANDI outputs. Goal: guardrail.
- On true counts, the fragment-window formula gives 0.0865 log CRPS. A learned map gives 0.0798.
- On the model's NB, every formula variant gave 0.18-0.28, against 0.14-0.17 for a direct p head.
- No peak derived from the NB was ever scored.
- Peak-caller settings are the largest artifact left in p space. Fragment extension leaves 0.31-0.41 of Var(Y) after correction.
- Evidence [CAC]: f18, q6, `src/candi/store/pval_from_counts.py`. [Max]: S07 `rewrites/processing_artifacts/NOTES.md`.

**K7. Today's CANDI is not rung 3 as written, and it does not beat the per-position average on count CRPS.** Abl / rung 3. Goal: guardrail, architecture.
- The shipped model differs from the written rung 3 in four ways:
  1. It has grouped per-assay conv lanes (groups = 36).
  2. FiLM is applied after every conv and deconv block.
  3. A learned mask token stands in for missing tracks.
  4. The assay panel is fixed (slot indices).
- Count CRPS on blind held-out (B_): CANDI 2.360 vs Average 2.312. The seed band is 0.20.
- Evidence [C2]: `src/candi/README.md`, `leaderboard/rows/eic.19/`. [C1]: s1 (ties the average on V_, loses on B_).

**K8. Joint imputation + CAC: the first test found no effect, at about 41% power.** Onion / miniCANDI. Goal: CAC.
- At weight 1, cross-assay imputation moved CRPS -0.025 against a bar of 0.067.
- At weight 4, it moved -0.034 against a bar of 0.039, which is a near miss.
- This was one model form, a 7-track CNN.
- Evidence [CAC]: f30, h29.

**K9. The DNA options: a small trained tower is roughly no DNA, and the gLM arm is probably a null.** Onion / miniCANDI DNA bullet. Goal: architecture.
- r_spec values: own tower 0.204, no sequence 0.201, frozen Flashzoi 0.249.
- Flashzoi memorizes its own training loci, and it was trained on GM12878 tracks.
- In the literature, gLM embeddings were no better than one-hot DNA (Tang 2025; DART-Eval).
- Evidence [Max]: S03 report §4. [Lit]: `wiki/sequence-model-critiques.md`.

**K10. NB NLL alone probably will not give calibrated uncertainty. Equal bin weighting of the mean was Max's largest training gain.** Onion / miniCANDI objectives. Goal: uncertainty, imputation.
- About 78% of the old ERA candidates failed the ECE floor.
- Max's uniform-weight mean gained +0.012 to +0.024 r_spec over likelihood weighting. Under likelihood weighting, strong peaks take 36-63% of the loss scale.
- Evidence [C1]: h28. [Max]: S06 `rft_paired.tsv`, S07 `02_model.md`.

**K11. The identity shortcut is not limited to rung 4.** Abl / rung 4 cons. Goal: guardrail.
- The shortcut exists at any rung that sees the counts at the output bin. "Denoising" on available tracks was autoencoding: 23-25% of cells were bit-identical between input and target.
- The old CANDI archived its U-Net as a dead end.
- Evidence [C2]: `results/t16/REPORT.md`. [C1]: `manuscript2.0/ADDITIONS_AND_UPDATES.md` §1.8. [Lit]: Noise2Self.

**K12. The cited population paper is not an imputer.** Onion / objectives 1.2. Goal: data.
- bioRxiv 10.64898/2026.02.01.702969 is Tu et al., "A modality gap in personal-genome prediction". It benchmarks AlphaGenome and Borzoi from genotype.
- Its use for us is to set ceilings. For accessibility, AlphaGenome scored 0.627 vs a genotype baseline of 0.758, measured on peaks pre-selected as heritable.
- eDICE has 4 donors and covers chr21 only.
- Evidence [Lit].

**K13. The resolution question is partly answered already, in asinh-p.** Onion / interesting question. Goal: eval.
- In the fine band (200 bp or less), specific error is about 90% sampling noise, and 0% of it is recovered.
- Specific signal first exceeds noise at these scales: ATAC 400 bp, H3K4me1 3.2 kb, H3K4me3 and H3K27ac 6.4 kb, H3K27me3 25.6 kb, H3K9me3 102 kb.
- No study in count space exists. No study has split the source of variation (artifact vs biology) by scale.
- Evidence [Max]: S06 `summary2/perlevel.tsv`, `eue.tsv`.

**K14. Exact EIC accessions are necessary but not sufficient.** Onion / data. Goal: data, guardrail.
- 0 of 363 challenge tracks md5-match an ENCODE file.
- 40 DNase tracks are read-depth-normalized signal, not p-values.
- The challenge grid is ceil(len/25) with the blacklist removed. Our grid is floor(len/25).
- No repo script checks our accessions against the paper's list.
- Evidence [C2]: `results/t54/DATASET3_GAP.md`, DATA.md. [C1]: h72.

**K15. EIC quantile normalization is applied to the truth, and it hides depth and scale CAC.** Abl / preprocessing #7. Goal: guardrail.
- CANDI + tcfloor ranks 1st on the normalized truths and 6th on the un-normalized single-end truth.
- The exact normalization rule is reproduced at all 3.03e9 bases for 47 of 48 tracks.
- Evidence [Max]: S04 `scripts/qnorm/apply_bp.py`, `candi_placement.tsv`.

---

## A. Onion: sources of variation

**A1. A replicate ladder table.** Variation 1 and Eval. Goal: eval.
- Give one row per view, with the component it isolates:
  - sequencing replicate: run or lane
  - technical replicate: library preparation and sequencing
  - isogenic biological replicate: biosample, with the donor fixed
  - anisogenic replicate: adds the donor
  - pseudoreplicate: read sampling only (data processing)
  - same term, other experiment: lab and protocol
- [Lit] ENCODE terms page.

**A2. Make read-sampling noise its own layer, separate from data generation.** Goal: eval, uncertainty.
- This noise follows signal level, not region. Level explains 0.18-0.38 of per-bin disagreement. GC, mappability and repeats add at most 0.0003.
- [CAC] t138, t142 DIGEST.

**A3. Split processing variation into uniform knobs and position-dependent knobs.** Goal: CAC.
- Uniform knobs: depth, dedup in histone marks, fragment extension.
- Knobs that act through mappability and repeats: MAPQ 0 and read-length crops. MAPQ 0 adds +0.45 log2 in the lowest-mappability decile.
- [CAC] f26, f32.

**A4. A measured ranking of artifact sources.** Goal: CAC, data.
- Peak height:
  - histone marks: depth > antibody proxy > read length 36
  - DNase: depth > dedup off > paired-end
- Read-level artifacts can be corrected by a per-bin curve to within 0.05.
- Peak-caller settings leave 3-41% unexplained.
- [CAC] f24, t139. [Max] S07 README item 5.

**A5. Bounds on population-level variation.**
- At n = 75, 10-15% of peaks have a detectable hQTL.
- Ancestry explains less than 20% of variance at most regions.
- Tissue identity dominates donor identity.
- [Lit] Grubert 2015, Kasowski 2013, eDICE.

**A6. Copy number confounds the broad band.**
- The 102 kb-and-up band is 67% unexplained.
- Planned test: correlate that residual with input-DNA coverage smoothed to 100 kb.
- [Max] S06 handoff Q2.

---

## B. Onion: objectives

**B1. Define "calibrated" precisely, as two separate terms.** Goal: uncertainty.
- Use "level error" for mean and scale error, and "interval calibration" for PIT.
- Use the non-randomized PIT for discrete NB, not interval coverage. A perfect NB at μ = 1 scores 0.25 on interval-coverage ECE.
- [C2] EVAL.md. [Max] item 9. [Lit] Czado, Gneiting, Held 2009.

**B2. State that one NB head is aleatoric only.**
- Add a seed or fold ensemble for epistemic uncertainty. The mean of 2 seeds improved MSE by 0.021.
- [Max] S06 `v3_paired_differences.tsv`. [Lit] Borzoi, DEGU.

**B3. Make preservation part of the CAC goal.** Goal: CAC, guardrail.
- After an A-to-B correction, report how much cell-type signal survives, next to how much artifact was removed.
- [Lit] sysVI, Scheible 2026.

**B4. Split covariates into nuisance and target tiers.**
- Nuisance covariates are corrected: lab, platform, pipeline.
- Target covariates are kept and can be swapped: donor, condition.
- [Lit] MrVI, DeepDive.

**B5. Define CAC success as steerability.**
- Success means that changing C' moves the output to the right place. Decoding the covariate from the latent is not enough.
- [Lit] Elazar 2020. [C1] h48: the 0.833 "steering" result was a sentinel artifact.

---

## C. Onion: eval (the benchmark itself)

**C1. Adopt Max's exact error decomposition as the backbone, and port it to counts and NB.**
- MSE/V = offset + scale + nonlinear + residual.
- Split the residual into shared and specific parts with an unbiased Gram estimator, over 14 Haar bands from 50 bp to 410 kb.
- Best arm: 0.686 = 0.115 level + 0.030 shared + 0.540 specific.
- [Max] S06 `scripts/lib/errdecomp.py`. [C2] `tools/eic_decomp/errdecomp.py`.

**C2. Split each component four ways: explained, unexplained, unexplainable (noise) and added.**
- Best arm: explained 0.43, unexplained 0.19, noise 0.28 after correction, added 0.01, level 0.11.
- [Max] S06 `summary2/eue.tsv`.

**C3. The noise-floor recipe.**
- floor = 1 − kR/(1+(k−1)R), per Haar level.
- Measure depth in fragments of the pooled track.
- When Y's reads are a subset of X's, use the conditional floor Var(Y|X).
- [Max] items 3 and 35.

**C4. Always quote CRPS as oracle_scaled + scale_error.**
- Mark oracle_scaled as an in-sample upper bound. scale_error can come out slightly negative.
- [C2] EVAL.md. [C1] h48: the split cut the spread between 4 arms by 84%.

**C5. A nested baseline ladder, next to every score.**
- The ladder, in order:
  1. marginal NB (climatological)
  2. per-position average
  3. generic calibrated average
  4. oracle-level average
  5. kNN
  6. depth-only scaling
  7. another experiment of the same term
  8. a true replicate
  9. the noise ceiling
- Quote the margin over the average, not over the marginal.
- [Max] item 16. [C2] item 2. [C1] item 4.

**C6. r_spec, a level-adjusted specificity correlation.**
- Any rescaling of the average scores exactly 0. The metric is stable under the unit used.
- Sprint 02's "specificity ahead of kNN" result was mostly level.
- [Max] S03 `metrics_lib.py`.

**C7. A calibration battery.**
- The battery: PIT curve, marginal calibration, coverage at several levels, sharpness and CRPS.
- Stratify by mean decile, depth, region, assay and Haar band, and report the worst stratum.
- Marginal cov90 is already 0.89-0.91 for every arm, so on its own it says little.
- Count-arm PIT is implemented but is not on the board.
- [C2] items 3-4. [Max] item 8. [Lit] B5.

**C8. Check predicted variance against real replicate variance.**
- Predicted dispersion r should fall between the pseudoreplicate-pair dispersion and the biological-replicate-pair dispersion.
- Add an NLL budget: noise floor + calibration + sharpness + model error.
- [Lit] SAGAconf. [Max] item 8.

**C9. Region strata.**
- Strata: promoters, enhancers, gene bodies, heterochromatin, ChromHMM states, mappability decile, and facultative vs constitutive.
- Weight each stratum equally. Report the realized foreground fraction. A requested 2% came out as 23.9%.
- [C1] METADATA_AUDIT S3. [Max] `TrackScorer(strata=…)`.

**C10. Value-stratified error and recall of rare positives.**
- The top 10% of bins carry two thirds of the residual.
- At the top 0.1% of bins, the best arm loses to the average: squared error 4.0 vs 2.7.
- [Max] S06 `v3_value_strata.tsv`.

**C11. Score broad marks at domain or broad bands.**
- Top-1% agreement between pseudoreplicate halves is only 4% for H3K9me3 and 9% for H3K27me3.
- [Max] S06 `floors_t112_class.tsv`.

**C12. The resolution study, done properly.**
- Report noise share by band, per assay and per variation source, in counts.
- Thin whole reads, never 25 bp units. Unit thinning reads 1.226 instead of 1.002.
- Compare models on one common coarse scale.
- [CAC] t142. [Lit] Toneyan 2022.

**C13. An identifiability table before any CAC claim.**
- Give the mutual information between each covariate and cell type, and H(run_type | assay, read length).
- Add an architecture-free probe of I(y; c | x).
- In the old data, no covariate cleared its floor. The best reached 1.17x.
- [C1] q21, h60. [Lit] Leek 2010.

**C14. CAC test sets.**
- Use cross-covariate pairs as ground truth.
- Report trained pairs and never-trained pairs separately, always. Gains fall from 0.29 to 0.11-0.16 on never-trained pairs.
- Hold out covariate combinations, not only cell types.
- [CAC] f8, f19.

**C15. CAC baselines and controls in every table.**
- Baselines and controls: identity, counts × d'/d, best line, monotone curve, the no-covariate twin, and a no-change control on a fresh draw of the base's own reads.
- Learned CRPS 0.432 vs depth-only 0.550 on trained pairs, and 0.527 vs 0.677 on never-trained pairs.
- [CAC] f33, t150. [Max] item 35.

**C16. Standing CAC gates.**
1. Depth law on never-trained pairs: |log2 error| ≤ 0.10. It was met in 0 of 72 cells.
2. The swap and identity check, C' = C.
3. A planted-effect positive control. It was designed and never built.
4. Each null control must be shown to land at its null first. All three h26 runs failed this.
- [CAC] f17, f19, f22, h1.

**C17. A family of twins.**
- The family:
  - no-covariate twin
  - labels-as-ids twin
  - shuffled-bin twin
  - shuffled, constant and withheld inputs
- Score the scrambled twin on never-trained pairs, because on trained pairs it cannot fail.
- At one point a random vector beat the real one.
- [CAC] row-2 grid, t146 defect 1. [Max] S07 `trans.md`.

**C18. A truth-pipeline swap exam.**
- Score each method on all four EIC truth layers: original, quantile-normalized, single-end reprocessed and ENCODE4. The ranking must hold on every layer.
- [C2] BENCHMARK_DESIGN §6.

**C19. The unit of inference.**
- Use a cluster bootstrap over ENCODE terms or targets, never over positions. CIs widened 18x when this was done.
- Use a wild cluster bootstrap-t for 5-30 clusters.
- Panel seed floors: 0.0139 (V_breadth), 0.0115 (V_matched), 0.2033 (B).
- State the power before the run.
- [C2] t86. [C1] H48 F4. [Max] item 32.

**C20. Metric hygiene.**
- Macro-average per assay, and never pool correlation. Pooling inflated a gap 3x.
- Never headline Spearman for monotone maps.
- Use AUPRC, not AUROC.
- Run a redundancy check across metrics.
- Never subtract V_breadth from B_.
- [C1] h46. [CAC] item 20. [C2] item 16.

**C21. Report peak height and 1 kb shape separately, each against the floor.**
- Peak height cleared the floor 27 of 27 times. Shape cleared it 3 of 27 times.
- [CAC] f24.

**C22. A crossed mixed-model variance partition, per region class.**
- Fit (1|celltype)+(1|donor)+(1|lab)+(1|batch) on observed counts, on predictions and on residuals.
- Peeling order matters, so report two orders or Shapley values.
- [Lit] variancePartition, Teng 2021.

**C23. For any individual-level claim, add a within-locus, across-individual correlation and a genotype baseline.**
- [Lit] Tu et al. 2026: AlphaGenome 0.151 vs genotype baseline 0.670 for expression.

**C24. A magnitude-vs-shape diagnostic per window.**
- Depth artifacts should hit magnitude, and biology should hit shape.
- [Lit] Borzoi loss.

**C25. Instrument hygiene.**
- Version the scorer and mutation-test it.
- Re-score the anchors after any metric fix.
- Add a reproduction gate per harness.
- Six broken instruments were found only by audit.
- [C1] H48_REPORT. [Max] item 54.

**C26. A fast selection scope for the ladder.**
- Use fixed, seeded random windows, about 2.7% of bins. They ran 16x faster and reproduced the full ordering in 194 of 200 draws.
- Do not use the ENCODE Pilot Regions. They were biased by +0.30 to +0.41.
- [C2] EVAL.md `--eval-regions`.

---

## D. Onion: data

**D1. Quantify EIC's imbalance along the measurement axis.**

| | train | blind |
|---|---|---|
| paired-end | 9.7% | 100% |
| platform | 9 | 1 |
| median log2 depth | 24.34 | 26.63 |

- 39 of 51 blind tracks sit above their assay's training depth ceiling.
- H3K27me3, H3K9me3 and H3K4me3 have no paired-end training track.
- [C2] from `results/t10/manifest.json`.

**D2. Collection criteria.**
- Collect same-biosample, same-assay cells with one covariate varied. EIC has 0 of 363 such cells, and MERGED has 0 of 2,684.
- Break the within-assay collinearity of depth, read length and run type by design.
- Cover depths above the training maximum. 53% of tracks exceed the 30M maximum used in the CAC testbed.
- [C1] q21. [Max] item 40.

**D3. Keep per-replicate count tracks and the BAMs.**
- In Dataset 1, all 50 replicate pairs share an identical p-value track.
- In MERGED, 122 "replicate" pairs list the same file twice. Deduplicate them before measuring any floor.
- [Max] item 28. [C2] item 27.

**D4. Any non-depth covariate needs raw reads.**
- BAM route: about 1 h per product. FASTQ route: median 13.4 h per product.
- Budget storage and plan for scratch purges.
- [CAC] t112, t117.

**D5. QC metrics as covariates or strata.**
- Candidates: NRF, PBC, FRiP, NSC/RSC, ENCODE audit color.
- Keep read-derived covariates apart from looked-up ones. A read-derived covariate on the C' side leaks the target.
- [C1] h66. [Lit] ENCODE standards. [Max] item 39.

**D6. A metadata audit gate.**
- In EIC, bios_accession was wrong on 214 of 363 rows.
- EpiATLAS corrected 40 labels in v2.0.
- `research/METADATA_AUDIT.md` is empty, although AGENTS.md ranks it third in authority.
- [C1] h66. [Lit] EpiClass. [C2].

**D7. What each source can supply.**
- EpiATLAS: 6 histone marks, 2,897 donors, harmonized processing. It has no ATAC or DNase.
- ChIP-Atlas: one fixed pipeline, so it gives a real processing-axis pair. Its tracks are RPM-scaled and its runs are merged.
- ENCODE: 2,079 new experiments since Dataset 1. All are DNase or RNA, and none is ChIP.
- Never use imputed files.
- [Lit] C1-C6. [Max] item 30.

**D8. Split by group, never by replicate.**
- Group by ENCODE term, donor and experiment, and exclude twins.
- Label each target as seen experiment, seen cell type or held-out.
- Six motor-neuron donors from one term supply 18 of the 50 held-out ChIP targets.
- [Max] items 31-32. [C2] item 27.

**D9. Control availability.**
- Controls exist for 73 of 89 EIC biosamples. ATAC and DNase have none, so a "control as an assay" arm is confounded with assay type.
- [C2] item 29.

**D10. More views and inputs.**
- Spike-ins, where available, give a genome-wide reference.
- Reference-genome vs personal-genome mapping is a processing covariate.
- [Lit] Bonhoure 2014, Kasowski 2013.

**D11. The exact EIC truth.**
- Use Jacob's Zenodo tracks (14736610, top level) and Max's scorer. The board reproduces with a median difference of 3e-10.
- Write the accession-check script that is missing today.
- [Max] item 22.

**D12. Write down the counting rule.**
- One read adds 1 to every 25 bp bin it overlaps. So counts per read = (L+25)/25, and read length is a covariate of raw counts.
- [Max] item 40.

---

## E. Onion: miniCANDI design

**E1. Add a retrieval (reference-panel) axis.**
- Similarity-selected reference attention gained +0.056 r_spec over mean/SD channels. About 4/5 of the gain came from the selection itself.
- [Max] S03 `design3.md`.

**E2. The optional average-baseline input.**
- Build it leave-one-out. Without that, it hands the answer away: 1/51 per cell, and 50% for a 2-cell assay.
- Compute it after CAC to a reference covariate set, or it copies training-set processing.
- [C2] `src/candi/reference.py`. [Max] caution 8.

**E3. Optional extra inputs.**
- RNA coverage: +0.024 r_spec. Swapped RNA was no better than none.
- A per-bin mappability channel: CRPS improved in the lowest-mappability decile on 6 of 6 tracks.
- Input-DNA coverage.
- [Max] S03 `rna.md`. [CAC] f32.

**E4. What goes in the covariate vector.**
- Keep covariates per assay. Pooling them was 25x worse.
- Add lab and platform with an UNKNOWN token.
- Index depth as a ratio, not as an absolute value.
- [C1] h34, h44. [Max] item 36.

**E5. Decode an imputed track with the target's own C'.**
- The old eval prompted the decoder with the canonical depth. This gave imputed R² near 0 while Pearson was about 0.5.
- [C1] s1, h23.

**E6. Masking.**
- Draw a Dirichlet budget across assays.
- Train on structured, deployment-like missingness.
- Note that full-assay masking also hides the C of the masked assay.
- [Lit] MultiMAE, Mitra 2023. [C2] DataMasker. [Max] item 50.

**E7. Loss arms.**
- Arms: a Poisson-NLL mean with NB dispersion, a uniform-weight mean, a CRPS term, and beta-NLL.
- The Poisson-mean arm cut scale_error from 0.253 to 0.005, with calibration unchanged.
- [CAC] t127. [Max] item 41. [Lit] Seitzer 2022.

**E8. If p-values are derived by formula, add a smooth λ head.**
- Supervise it where a control exists.
- [Max] item 38.

**E9. Select checkpoints on V_ and report on B_, with early stopping.**
- V_ bottomed out at epoch 8-11 of 25.
- [C1] h74.

---

## F. Ablations: the ladder

**F1. Add rung 0.**
- It holds the per-position average, the marginal NB and depth-only scaling.
- Rung 1 is not the floor.
- [C1] s1. [Max] item 16.

**F2. State rung 1 as a set transformer or conditional neural process.**
- This matches eDICE: set attention over the tracks at one bin.
- Use eDICE as the comparator with no covariates.
- [Lit] E4-E5.

**F3. Correct rung 3 to describe the shipped model.**
- See K7. Name which checkpoint is the baseline: the 3.1M-parameter 8-assay model or the 2.35M-parameter 35-assay model.
- [C2] item 21.

**F4. Rung 4.**
- Add a copy-leak test and a skip-source arm. Note that the old U-Net was archived.
- [C2] `tools/dsf_leak.py`.

**F5. Rung 5.**
- Ablate each axis: position-only, track-only, and tied row attention.
- Start with a light cross-assay axial mixer. It was the only rewrite that helped in old CANDI.
- [C1] menu REPORT. [Lit] MSA Transformer.

**F6. Rung 6.**
- Sweep the latent count from 16 to 1024. Compare the power spectrum of predictions with the truth.
- Add a local conv feature at the query bin.
- "The bottleneck loses positional detail" is unmeasured.
- [Lit] E7.

---

## G. Ablations: at every rung

**G1. A J-invariance leakage test.**
- Perturb a masked bin's input count, and assert that the output does not change.
- Score denoising only on masked or thinned bins.
- [Lit] Noise2Self. [C2] t16.

**G2. DNA arms.**
- Arms: DNA off, zeros, dinucleotide-shuffled, wrong locus, and shifted by 5 Mb.
- Match parameter counts.
- For any frozen pretrained model, build a leakage table by its train and test loci.
- Grid caution: gcd(25, 128) = 1, so 128 bp grids do not aggregate into 25 bp grids.
- [Max] item 43. [C2] t48. [CAC] h31.

**G3. Covariate-injection arms.**
- Add adaLN-Zero and conditioning dropout.
- Inject FiLM before vs after normalization.
- Exempt the conditioning embedders from weight decay. Decay drove one of them to 6e-41.
- [Lit] D9-D10. [C1] h47.

**G4. A lane-isolation test for any "per assay by construction" claim.**
- A LayerNorm across channels leaked 1e-2 between lanes.
- [C1] h78.

**G5. Head arms.**
- Arms: NB, CRPS-trained NB, and a constant-variance head.
- Wrap any point-output head with a flat sigma, and badge it as such.
- [C1] item 14. [C2] item 5.

**G6. A loss-resolution arm.**
- Pool to 200 bp, 800 bp or 3.2 kb before the loss.
- A coarse loss never helped.
- [Max] item 42.

**G7. CAC loss arms.**
- Cycle consistency and adversarial invariance.
- Report adversary accuracy, a post-hoc probe and the imputation cost.
- [Lit] D6-D7.

**G8. Sweep the pseudocount ε in {0.01, 0.1, 1}, plus a pseudocount-free arm.**
- With a +1 pseudocount, a log-ratio follows signal level. This broke the swap probe.
- [CAC] f22, wiki.

---

## H. Ablations: comparison protocol

**H1. Matched windows seen, a convergence curve, and at least 2 seeds.**
- The seed gap is about 0.007. Treat a single-run difference under 0.01 as an observation.
- Use a seed ensemble as a free baseline.
- [Max] items 47-48.

**H2. A data-regime control.**
- Do not rank architectures on chr19-only training.
- [C1] h26.

**H3. Shared and specific error columns for every rung.**
- So far all gains have been in shared error.
- [Max] item 49.

**H4. Test additivity before stacking two wins.**
- [C1] CEILING.md.

**H5. A health gate per rung.**
- Imputation Spearman above a floor, imputation Pearson above 0, denoising at least as good as imputation, PIT-ECE ≤ 0.10, effective rank above 1.
- [C1] h40.

**H6. External rivals under the same eval.**
- ChromImpute, Avocado, eDICE, Corgi+, Borzoi direct, and a ridge "oracle head".
- On our panel, eDICE has a genome-wide correlation of 0.502 vs CANDI 0.347.
- [Max] items 44 and 53.

**H7. Freeze the score before any automated search.**
- [C1] h26.

---

## I. Ablations: count preprocessing

**I1. Transform 8: raw counts with depth as an input covariate.**
- The literature treats depth as an offset, never as a divisor.
- [Lit] D11.

**I2. Transform 9: Borzoi's count^(3/4) with soft clipping, on the input side only.**
- [Lit] E11.

**I3. Size factor.**
- Estimate it from background bins, not from total reads.
- [Lit] S3norm, SES.

**I4. Pearson residuals.**
- Name the dispersion estimator, and report the clipping rate per assay.
- [Lit] E12.

**I5. Sentinel collision.**
- Transforms that go negative collide with the MISSING = −1 and CLOZE = −2 sentinels.
- Use an availability-mask channel instead.
- [C1] item 37.

---

## Process notes (unrelated to the documents)

- The CLAUDE.md router path `/project/def-maxwl/maxwl/epi_imputation/notes_mirror/` does not exist on Fir. This was checked on 2026-11-10. Max has a sprint 09 on Fir that is not on GitHub.
- No GitHub Pages leaderboard was found for Max's repos. "Explained and unexplained variance" is in the S06 `report2.html`.
