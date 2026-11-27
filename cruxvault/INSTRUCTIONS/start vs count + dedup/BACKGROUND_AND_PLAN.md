# Read starts vs read counts, and deduplication / SE-PE correction: background and plan

Drafted 2026-11-27 from four read-only sweeps: (1) the read-start branch, (2) Max's GitHub sprints 02-18, (3) Max's sprint 19 files on Fir, (4) our correction results for each covariate. Every number below is copied from the file named beside it. "Read-count" means reads extended to the fragment span and counted per 25 bp bin by overlap. "Read-start" means the 5' base of each read, binned at 25 bp, per strand.

## 0. PI decisions (2026-11-27)

1. Main-text claim: CANDI corrects the covariate effects that broke the EIC (dedup, SE/PE). Dedup and the other covariate results support this claim. Some evidence can go to the supplement. Iterations 1-4 all run.
2. Read-start vs read-count: no decision rule is set before the data. Gather the headlines, the mechanism results and Max's slam-dunk test first, then decide.
3. Compute: the factorial (about 2,350 CPU-h). Under-training is a separate question, like scaling. Test it later on the winner of the factorial, over windows, epochs and other budget terms. The 320,000-window arms are removed from iteration 2.
4. The correction headline gets dedup and SE pairs (the same metric with a wider pair set and a per-covariate breakdown).

## 1. Where we are

### 1.1 Data

- The read-start store uses the same tagAligns as the read-count store. A gate rebuilt the overlap counts from the starts and found 0 differing bins on 8,220 of 8,220 files (`.claude/worktrees/readstart/cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/BUILD.md`).
- Store: Fir `/scratch/mforooz/rs392/store`, files `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5` (pseudoreplicate halves), shape [n_bins, 2 strands].
- Coverage: eic_v1_2 complete. eic_v1_2x second experiments: 71 of 71 rows, gate passed 2026-11-27. Blind rows: not built. They wait for the Nibi blind build.
- No 1 bp store exists for training. Exact duplicate positions are visible only at 1 bp.

### 1.2 The comparison that ran (t392, PR 66, merged into `exp/t370-minicandi-recipe`)

- The two models had the same U-Net, the same recipe, the same windows (80,000), and parameter counts of 993,695 vs 993,615. Only the input and target representation changed.
- The read-start model was converted to overlap counts by the per-measurement span kernel. The NB was moment-matched, with bins treated as independent. Both models were scored in overlap-count space. **This is already leg A of Max's proposal**: extend the read-start model, then score it against read counts.
- Results, 16 regions, 3 seeds, mean (max - min) (`cruxvault/results/t392/SIDE_BY_SIDE.md`, read-start worktree):

| metric | read-count | read-start |
|---|---|---|
| imputation | 0.4996 (0.0349) | 0.4568 (0.0070) |
| correction | 0.8156 (0.0105) | 0.7826 (0.0362) |
| calibration | 0.6852 (0.0924) | 0.6064 (0.0679) |
| deviance imputation skill | 0.5426 (0.0653) | 0.3746 (0.0870) |
| deviance calibration ratio (1 = calibrated) | 1.0584 (0.0398) | 1.3081 (0.0965) |

- The read-count model won on every term. Only the deviance skill gap (0.168) is clearly outside the seed ranges: every read-start seed (0.338-0.425) is below every read-count seed (0.510-0.575).
- H3K36me3 alone contributed 0.093 of the 0.168 macro skill gap (-0.428 vs 0.409). The read-start model won on H3K27ac (0.516 vs 0.317).

### 1.3 Why it lost: the investigation so far (t394, t395; all findings weak or moderate)

- Calibration: the independence assumption in the conversion understates the variance. Dividing the NB size by a measured correlation factor moved the ratio from 1.25 to 0.88 (over-corrected) on seed 0. So the conversion alone can explain the whole calibration gap. It is an evaluation artifact, not a model defect. (weak)
- Sparsity: the read-start targets are 88.6% zeros, against 62% for counts. The replicate correlation is 0.248 against 0.499. Sparsity does not predict the per-assay gap (Spearman 0.20, p 0.61). (mixed, weak)
- Loss weighting and input bugs: ruled out. (weak; moderate)
- Under-training: not answered. The 160,000-window read-start seed (about 260 CPU-h) was deferred by the PI on 2026-11-25.
- H3K36me3 is a broad level error: the residuals stay correlated past 400 bins, and the predicted totals are 6-28% above observed. Smoothing the starts does not recover skill. (weak; its expectation was written after the numbers)

### 1.4 Unfair parts of the comparison as run

1. The recipe was tuned on read-counts. The read-start model got no tuning.
2. Both had 80,000 windows. The final read-count recipe uses 160,000 (decision D55). Under-training would hurt the sparser target more.
3. Scoring was only in read-count space. The read-start model paid a conversion floor (nMSE 0.014-0.072 by assay) and the variance artifact above. No read-start-space score was made on equal terms.
4. Masked overlap bins leak through neighboring bins, but masked starts do not. This helps the read-count model on correction.
5. Only 16 regions and 3 seeds. The PI declined 256-region scoring.

### 1.5 Leg B of Max's proposal (read-count model scored in read-start space)

- No code exists. The overlap count is the starts convolved with a box of about (L+25)/25 bins, summed over strands. Max's "take the derivative" gives c(i) - c(i-1) = s(i) - s(i-w): the difference of two start bins, not one. Inverting it amplifies noise and gives negative values. The read-count model also has no strand information.
- Plan: recover non-negative starts by Richardson-Lucy deconvolution (the EM solution for Poisson deblurring) with the same per-measurement kernel. Score in strand-summed start space so that the read-count model is not charged for strand it never saw.

### 1.6 Deduplication and SE/PE: what Max found

- **Sprint 19 has no deduplication analysis.** It is not on GitHub (the last push is sprint 18, 2026-11-18). Its files on Fir (`/project/def-maxwl/maxwl/epi_imputation/s19/`) contain no notes and no dedup code. It studies depth: (1) CANDI's p-value head ignores the depth covariate; (2) the count head plus the MACS2 formula follows synthetic thinning near the noise floor (nerr 0.936 vs floor 0.809 at 1/8 depth); (3) on real replicate pairs, CANDI follows only about 28-30% of the depth difference, and AG2 plus the formula follows about 85% (my agent's recomputation from `s19/cov/t2out`, `s19/ag2head/t2`). Two depth-convention signs relate to SE/PE: the best fit is at about 0.7x the recorded depth, and halving the depth of PE tracks (mates counted twice) lowers PE nerr by about 0.07 (`s19/cov/t2r4sw`, `t2r4pe`). These are the agent's computations, with no note from Max to confirm them. The verbal dedup claim may come from unpushed notes.
- The dedup analysis is in sprints 07 and 10 (GitHub, `projects/03_2026-10-20_epi_imputation/`):
  - EIC shift: the blind set was 100% PE and training about 90% SE. At the tail of -log10 p (about 100), only 32% of the shift is depth, and the rest is SE deduplication (`notes/key_findings.md`, ROBUST mechanism, PRELIMINARY coverage).
  - Mechanism: a duplicate cap. SE deduplication keeps at most one read per start position and strand. So a 25 bp bin can hold at most about 2(L+25) overlapping reads after dedup. In testbed data the SE track plateaus near the predicted caps (H3K4me3 20.2 vs PE 120.4; DNase 34.2 vs 364.7) (`sprints/07_2026-11-04/rewrites/processing_artifacts/NOTES.md`).
  - The saturation law SE = K(1 - exp(-c_PE/K)) with K = kappa(L+25)/25 "did not help" genome-wide. It is real only in the top 0.01-1% of bins.
  - Max (sprint 10): SE/PE and dedup "should go in the 'could be corrected by CAC' category". Max (sprint 18): the covariates that "ought to help" do not, "top priority".
  - Max's paper plan (sprint 15, section 5) suggests retiring "CAC" from the paper and keeping it as a supplement negative result. Freeze target: about 2026-11-27.

### 1.7 Our correction for each covariate (final model `rcp_combo_w2x_unet`, 3 seeds)

- Covariate-prompt test, i.e. change one output covariate and check the predicted response against a rule set before the run (`cruxvault/results/t370/recal_study/covariate_prompt_*.md`, recipe worktree):

| prompt | targets that follow it |
|---|---|
| depth +/-1 log2 | 16/16 (total ratio 1.967-2.022) |
| read length 36 -> 100 | 12/16 |
| run type SE/PE | 4/16 |
| MAPQ 0 | 2/16 |
| dedup flip | 1/16 |

- The dedup flip leaves the predicted total unchanged and lowers the NB size. The model reads "dedup off" as "noisier", not as "taller peaks".
- Training views (`closeout_E4/.../tables_eic_v1_2x`, onion worktree): dedup-off 312 experiments; SE reprocess 38 rows (21 DNase, 4 ATAC, 13 ChIP; 23 of 36 assays have none); PE base experiments 12 against 229 SE. SE/PE is confounded with assay and lab.
- Dedup-off views share about 96% of reads with the base view. With these views, the U-Net becomes a covariate-gated copier (f141).
- **The headline correction metric contains no dedup pair and no SE pair.** It uses only the 18 read-length crop pairs. The composite cannot see a dedup or SE failure.
- Depth is the only covariate with a closed-form path: mu = s_out * 2^eta, with s_out = 2^(log2_depth - 24). Binomial thinning of NB(mu, n) gives NB(q mu, n), so the offset is exact for depth views.

## 2. The two formulations

| | read-start | read-count |
|---|---|---|
| information | strictly more: strand and exact start bin; overlap counts are a linear function of starts | loses strand and position within the span |
| target noise | 88.6% zeros, replicate r 0.248 | 62% zeros, replicate r 0.499; the box acts as a free smoothing prior |
| dedup | the cap is per position, so it is closed form in this space (see 3.2) | the cap mixes about 2(L+25) positions; needs a fitted kappa |
| read length, extension, SE/PE | a known kernel maps starts to any overlap track | baked into the data: a crop changes counts by +0.68-0.80 (Max, sprint 09) |
| p-values | the MACS2 formula is exact from 1 bp positions (Max, ROBUST) | needs a pileup re-spread |
| cost | scoring about 2.7x slower; conversion needs a covariance term | the tuned recipe, current winner |

Because starts carry more information, a large enough and well-trained read-start model should not lose. If it loses, the cause is statistical efficiency (noisier target, smaller effective sample, no smoothing prior), under-training, tuning, or the scoring space. The plan separates these causes.

## 3. Plan

Each iteration takes 2-4 days and ends with a result.

### Iteration 1: no training (about 30 CPU-h, all with existing predictions and data)

1. **Leg B.** Deconvolve the existing read-count predictions (t370 reference, 16 regions, 3 seeds) to starts. Score both models in (a) strand-summed start space and (b) a neutral coarse space (1 kb sums, where the two representations differ only by a known factor). Normalize each score to its own pseudoreplicate floor (`starts_pr1/pr2` and the count halves). The slam dunk holds if the read-count model also wins in start space.
2. **Fix leg A.** Add the covariance term to the start-to-count conversion. If the calibration gap closes, record it as an evaluation artifact.
3. **Dedup mechanism, data only.** On the 312 (dedup-off, dedup-on) view pairs, predict the dedup-on track from the dedup-off track by a closed-form cap:
   - Start space, 25 bp bin, per strand: E[kept] = 25 (1 - (1 + lambda/(25a))^(-a)), i.e. the expected number of occupied positions when the positional rates within a bin are Gamma(a). K = 25 positions is known. Only a is fitted, one per assay.
   - Count space: Max's law with fitted kappa.
   - Measure which space explains the dedup view better, overall and in the top 1% of bins, against the half-vs-half floor.
4. **SE/PE mechanism, data only.** Picard marks a PE pair a duplicate only when both ends match. So the PE cap is per (start, fragment length) and is much higher. Test whether the cap plus the fragment kernel predicts the SE view from the PE view on the 12 PE base experiments and the 38 SE views.
5. **Depth convention.** Check whether our `log2_depth` counts PE mates twice (Max's factor 2). This is a correctness check, and I will fix it if it is wrong.

### Iteration 2: the fair comparison (training)

- Arms (a 2 x 2 design, to separate information from target smoothing):
  1. count input, count target (exists: the final recipe at 160,000 windows, 3 seeds)
  2. start input, start target
  3. start input, count target through a fixed kernel layer inside the model ("kernel head": the model predicts start intensity, the layer convolves it to the overlap mean, and the loss is on counts)
  4. count input, start target
- Equal terms: the same tagAligns, windows, parameters and seeds; the same small tuning sweep (learning rate x 3) for each arm, picked on validation in the arm's own space; 256 regions; scores in all three spaces from iteration 1.
- All arms at 160,000 windows. The under-training test runs later, on the winner only (PI decision 3).
- Readings: arm 3 matches arm 1 -> the gap is the target (smoothing), not the information. Arm 2 wins in start space but loses in count space -> the scoring space decides it.

### Iteration 3: mechanistic processing layer (if iteration 1, step 3 or 4 holds)

- Build the cap as an output layer, as the depth offset was built: the model predicts the pre-dedup start intensity, depth scales it, and the dedup and SE/PE flags apply the closed-form cap. A fragment kernel then gives any overlap track.
- Test it on the covariate-prompt test (target: dedup flip 1/16 -> most targets) and on dedup and SE pairs.
- This layer requires the read-start (or 1 bp) representation inside the model, even if the output is scored as counts. So the formulation choice and the dedup result are linked.

### Iteration 4: data layer

- Data-limited test: retrain with 25 / 50 / 100% of the dedup-off and SE views. If correction rises with views, more views help.
- More views are cheap where raw reads exist. A dedup view is a deterministic function of the dedup-off reads. An SE view is R1 of any PE experiment. PE views cannot be made from SE data. EIC training has only 12 PE base experiments, so the SE/PE fix needs the PE-rich extended dataset (ENCODE, EpiATLAS, ChIP-Atlas).

## 4. Hypotheses (recorded in the notebook when the branch is cut)

- The read-count advantage comes from target smoothing (statistical efficiency), not from information. Expect: the kernel-head arm is within the seed range of the read-count model.
- At 160,000 windows with equal tuning, the read-start skill gap to the read-count model is smaller than the 0.168 seen at 80,000 windows. (Full under-training test: later, on the winner.)
- The read-count model still wins in start space (Max's slam dunk). Expect: unknown before the run; this is the decisive test.
- Dedup is a per-position cap that is closed form in start space. Expect: the start-space cap explains the dedup-on view in the top 1% of bins better than the count-space law, beyond the half-vs-half floor.
- The dedup failure is a mechanism gap, not a data gap. Expect: the views-fraction curve is flat, and the mechanistic layer moves the dedup prompt from 1/16 to most targets.
- The SE/PE failure is a data gap first. Expect: the 38 SE views cover too few assays, and correction rises with more views.

## 5. Estimated compute

Base figure: one read-start seed at 160,000 windows is about 260 CPU-h (deferred h54 estimate, read-start worktree vault). The read-count arm 1 already exists.

- Minimal (arm 2 at 160,000 windows, 3 seeds, 256-region scoring): 3 x 260 = about 800 CPU-h.
- Factorial (arms 2, 3, 4 at 160,000, 3 seeds): 9 x 260 = about 2,350 CPU-h.
- Chosen: the factorial. The under-training test on the winner is costed later.
- The tuning sweep adds about 30%. These are estimates; the read-start scoring cost is 2.7x the read-count cost.
