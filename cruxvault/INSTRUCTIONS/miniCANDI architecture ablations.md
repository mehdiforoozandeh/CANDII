Every rung has the same contract. The inputs are the observed counts, DNA one-hot, input covariates and output covariates. The output is a Negative Binomial (n, p), with the same loss, masking and data. 

### Rung 1: Pointwise MLP

**Core idea.** The model predicts each bin from that bin alone. A set encoder reads the observed tracks at the bin, a short DNA vector is added, and the covariates condition an MLP that emits the Negative Binomial parameters.

**What it adds.** Nothing. This rung is the floor.

**Why it is interesting.** It measures how much of the task is cross-assay inference at one locus. If it comes close to CANDI, genomic context buys little.

**Pros.** It is fast, simple and easy to read. It has no window-edge effects.

**Cons.** It cannot use peak shape or neighbouring bins. Its DNA input is limited to a short window.

### Rung 2: Convolutional autoencoder

**Core idea.** Conv towers downsample the counts and the DNA to a coarse latent. A transposed-conv decoder upsamples back to full resolution. Context is local, and the receptive field grows with depth.

**What it adds.** Neighbouring bins and local signal shape.

**Why it is interesting.** It tests whether local context alone is enough, which would make CANDI's attention redundant.

**Pros.** It is cheap and stable. Its strong local bias suits peaks.

**Cons.** The receptive field is fixed and limited. The bottleneck loses fine detail.

### Rung 3: Conv + attention autoencoder

**Core idea.** Rung 2, plus a transformer encoder on the downsampled latent.

**What it adds.** Context over the whole window, at low cost, because attention runs on the coarse sequence.

**Why it is interesting.** The step from rung 2 to rung 3 measures what attention buys.

**Pros.** It gets global context cheaply.

**Cons.** Attention mixes positions only, and the tracks mix once, in one linear fusion. The decoder must rebuild fine detail from the latent alone.

### Rung 4: U-Net

**Core idea.** Rung 3, plus skip connections from each encoder level to the matching decoder level.

**What it adds.** Fine detail bypasses the bottleneck.

**Why it is interesting.** The skip source shows where the detail lives. Skips from the DNA tower, the count tower, or both are the three arms.

**Pros.** Outputs are sharper at full resolution, and U-Nets train reliably.

**Cons.** Count skips create an identity shortcut, so the model can denoise by copying its input. The output covariates must also steer the skip features, or the target depth is ignored.

### Rung 5: Axial transformer

**Core idea.** Each track becomes a row of tokens over position patches, and DNA is one more row. Attention alternates along positions and across rows. The output covariates form query rows that read the grid.

**What it adds.** Tracks mix at every layer, not once. A missing track is an absent row, so the model needs no mask token.

**Why it is interesting.** Cross-assay relations at a locus probably carry most of the imputation signal, and this is the most direct design for them. This is my bet.

**Pros.** Tracks and positions are symmetric axes. Variable assay panels need no special handling.

**Cons.** Cost grows with tracks times positions. Patching blurs fine resolution unless a conv decoder or skips restore it. It has a weaker local bias, so it probably needs more data.

### Rung 6: Perceiver IO

**Core idea.** Observed (track, patch) tokens and DNA tokens cross-attend into a small fixed latent array, i.e. a set of learned summary vectors. Self-attention runs on the latents. Queries built from the output covariates and the position read out any track at any bin.

**What it adds.** Cost no longer scales with input size. It needs no mask tokens, and any observed subset maps to any target.

**Why it is interesting.** It is the closest fit to the aim "any assays in, any assay out".

**Pros.** Missing data is handled by construction. It scales to large panels.

**Cons.** The latent bottleneck can lose positional detail. It is harder to train, and it has the weakest local bias on the ladder.

### Ablations at every rung

- DNA off.
- Scrambled covariates, as a control for covariate use.
- Covariate injection: concatenation, FiLM, or tokens.
- Matched parameters, then matched compute.
- Mask tokens, or dropping the missing tracks.
- Copy-leak guardrail (background, see [copy-leak-guardrail](#copy-leak-guardrail)).

### Comparisons that answer one question each

| Comparison | Question                                           |
| ---------- | -------------------------------------------------- |
| Rung 1 → 2 | Does local context help?                           |
| Rung 2 → 3 | Does global attention help?                        |
| Rung 3 → 4 | Do skips help, and from which source?              |
| Rung 3 → 5 | Does attention across tracks help?                 |
| Rung 5 → 6 | Does a latent bottleneck with query decoding help? |

---

## Input preprocessing of counts

These are candidate transforms for the count inputs. Each one lists how it removes depth on the input side and how depth is added back on the output side. Only the size-factor offset makes depth an exact additive shift in η.

**Notation.** y is the count in a bin. D is the sequencing depth, and d = log₂ D. D₀ = 2^c is the reference depth, and s = D/D₀ is the size factor. The model is y ~ NB(μ, r) with μ = s·λ, where λ is the depth-free rate.

### 1. Size-factor offset (NB GLM with a log-depth offset)

- Remove depth (input): η̂ = log₂((y + ε)/s) = log₂(y + ε) − (d − c).
- Add depth back (output): μ = s·2^η and p = r/(r + μ).
- Depth is exactly additive in η. Binomial thinning keeps r fixed, i.e. NB(μ, r) → NB(qμ, r).
- Background note: see [depth-offset-is-an-assumption](#depth-offset-is-an-assumption).

### 2. log(1 + count)

- Remove depth: x = log(1 + y/s).
- Add depth back: y = s·(eˣ − 1).
- Depth is additive only where y ≫ s. Near zero, x ≈ y/s, so depth acts as a scale factor.

### 3. arcsinh(count)

- Remove depth: x = asinh(y/(b·s)), with cofactor b.
- Add depth back: y = b·s·sinh(x).
- Large y: x ≈ log(2y/(b·s)), so depth is additive. Small y: x ≈ y/(b·s), so depth acts as a scale factor.
- The exact NB variance-stabilizing transform has this form: g(y) = 2√r·asinh(√(y/r)) stabilizes Var = μ + μ²/r. Dividing by s first breaks the stabilization.

### 4. Anscombe (Poisson variance-stabilizing transform)

- Remove depth: x = 2√(y/s + 3/8).
- Add depth back: y = s·((x/2)² − 3/8). This simple inverse is biased at low counts.
- The variance is near 1 for Poisson data at s = 1. Depth acts as a √s scale factor and never as a shift. The transform does not stabilize NB overdispersion.

### 5. NB Pearson residuals (as in sctransform)

- Remove depth: z = (y − s·λ̂ₐ) / √(s·λ̂ₐ + (s·λ̂ₐ)²/r). Here λ̂ₐ is the genome-wide mean count of assay a at the reference depth. sctransform clips z at ±√N.
- Add depth back: y = s·λ̂ₐ + z·√(s·λ̂ₐ + (s·λ̂ₐ)²/r).
- Depth is removed from both the mean and the variance by construction. The residual is a standardized deviation and not a log-rate, so η is not directly readable from it.

### 6. Local-background ratio (MACS2-style)

- Remove depth: x = log₂((y + ε)/(λ_local + ε)), with λ_local = max(λ_genome, λ_w₁, λ_w₂, …). Each λ is a mean count from the same sample over windows such as 1 kb and 10 kb.
- Add depth back: y = λ_local·2ˣ.
- s cancels without a depth estimate, and the ratio also removes broad GC and copy-number trends. The inverse needs the target's own λ_local, which an imputed target does not have. Use it on the input side only.

### 7. Quantile normalization per assay

- Remove depth: x = F_ref⁻¹(F_sample(y)). F_sample is the sample's empirical distribution function, and F_ref is the reference distribution of that assay.
- Add depth back: there is no per-sample inverse for an imputed target, because F_sample is unknown. Use it on the input side only.
- Background note: the EIC applied a quantile normalization to its truth tracks. See [eic-quantile-normalization-on-truth](#eic-quantile-normalization-on-truth).
- This transform is robust to depth and to outliers. Ties at zero need a rule, such as averaging the tied ranks.

### Common caveat

Under all seven transforms, the noise around the normalized value still depends on depth (relative variance 1/μ + 1/r). The encoder therefore still needs d as a covariate.

---

# Background notes (added 2026-11-10)

Everything below is background information, not a decision. Each entry was checked against its source on 2026-11-10. Source-path conventions are defined in [[onion caveats]]. Scope rule (PI, 2026-11-10): no architecture evidence from our own old CANDI or CANDII ablations appears here, so that the search starts from first principles. Results from Max's sprints, the literature, and the CAC testbed are included.

Related files: [[peeling the onion]], [[miniCANDI design notes]], [[onion eval protocol]], [[onion caveats]].

## Index of background notes

- [design-choices-to-ablate](#design-choices-to-ablate)
- Ladder: [rung-0-baselines](#rung-0-baselines), [rung-1-as-set-model](#rung-1-as-set-model), [name-the-candi-comparator](#name-the-candi-comparator), [axis-ablations-for-axial-rung](#axis-ablations-for-axial-rung), [perceiver-latent-sweep](#perceiver-latent-sweep)
- Every rung: [copy-leak-guardrail](#copy-leak-guardrail), [dna-arms-and-pretrained-leakage](#dna-arms-and-pretrained-leakage), [covariate-injection-arms](#covariate-injection-arms), [lane-isolation-test](#lane-isolation-test), [head-arms-and-flat-sigma-badge](#head-arms-and-flat-sigma-badge), [loss-resolution-arm](#loss-resolution-arm), [cac-loss-arms](#cac-loss-arms), [pseudocount-sweep](#pseudocount-sweep)
- Comparisons: [seeds-and-compute-matching](#seeds-and-compute-matching), [shared-and-specific-error-per-rung](#shared-and-specific-error-per-rung), [external-rivals](#external-rivals), [comparison-protocol-ideas](#comparison-protocol-ideas)
- Preprocessing: [depth-offset-is-an-assumption](#depth-offset-is-an-assumption), [eic-quantile-normalization-on-truth](#eic-quantile-normalization-on-truth), [depth-as-covariate](#depth-as-covariate), [borzoi-soft-clip](#borzoi-soft-clip), [size-factor-from-background](#size-factor-from-background), [pearson-residuals-details](#pearson-residuals-details), [sentinel-collision](#sentinel-collision)

## design-choices-to-ablate

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

These are the PI's open design questions (2026-11-10). They are listed with candidate options from the literature, from [[peeling the onion]], and from Max's design notes. No option is chosen.

- **DNA–signal fusion.** Options:
  1. Concatenate DNA features with count features before the trunk.
  2. Add DNA as one more token row in axial attention (rung 5).
  3. Use DNA as a FiLM conditioner on count features (Perez et al. 2018).
  4. Feed DNA tokens as extra inputs to a Perceiver latent (Jaegle et al. 2022).
  - The DNA source can be an own tower, frozen Flashzoi or a genomic language model (onion doc).
- **Attention axis.** Options:
  1. Per-position attention only (rung 3).
  2. Per-assay set attention at each bin, as in eDICE's factorized attention over assays and over cell types (Hawkins-Hooker et al. 2023).
  3. Both, alternating, as in axial or MSA-Transformer attention (Rao et al. 2021).
  4. A latent bottleneck over all (track, position) tokens (Perceiver IO).
- **Average-baseline input and its fusion.** Options:
  1. An input channel per assay, built leave-one-out.
  2. A GLM offset on the log mean, so the network predicts the log-deviation (`CANDII: src/candi/reference.py`).
  3. Retrieval attention over individual reference cells, with the mean as fallback (Max s03, design 3).
  4. A post-hoc per-assay blend with the model output (Max s04).
  - See [[miniCANDI design notes#average-baseline-input]] and [[miniCANDI design notes#retrieval-attention-over-reference-cells]].
- **Covariate injection.** Options:
  1. Concatenation.
  2. FiLM at every block.
  3. adaLN-Zero with a zero-initialized gate.
  4. Covariate tokens or query rows read by attention (rungs 5–6).
  - See [covariate-injection-arms](#covariate-injection-arms).
- **C and C' for truly missing assays, in training and at inference.** Options:
  1. A learned null or UNKNOWN token, trained by conditioning dropout (Ho and Salimans 2022).
  2. Drop the missing track and its C entirely, as set models do (Lee et al. 2019). MultiMAE also drops masked tokens.
  3. Fix C' to a reference covariate set and decode to that standard (two-step CAC).
  4. Fill with the assay median (Max s03 `notes/metrics.md`).
  - Whatever is chosen, an imputed target scored against a real track is decoded with that track's own C' ([[miniCANDI design notes#decode-imputed-track-with-own-covariates]]).
- **C_base for two-step CAC.** Options:
  1. One fixed reference processing per assay family, with corrections indexed relative to it (Max s07 `rewrites/processing_artifacts/processing_artifacts.md`).
  2. The most standard ENCODE-DCC pipeline per assay.
  3. The deepest or modal covariate set in the training data.
  4. A learned latent base with no observed counterpart. This option gives up the observed-pair supervision that two-step CAC relied on.
  - See [[onion caveats#two-step-cac-per-track-vs-shared]].
- **Three variants taken from the current CANDI design:**
  1. **Per-assay grouped conv lanes.** Options: grouped convolution with one group per assay; one shared conv with an assay embedding; separate lanes plus a [lane-isolation-test](#lane-isolation-test).
  2. **FiLM at every block.** Options: FiLM at every conv and deconv block; FiLM once at the bottleneck; adaLN-Zero gating; modulation applied to weights rather than to normalized features (StyleGAN2 demodulation).
  3. **Learned mask token.** Options: a learned mask token in the input; drop masked tracks in the encoder and use mask tokens only in the decoder (He et al. 2022, MAE, arXiv:2111.06377); zero fill plus an availability-mask channel ([sentinel-collision](#sentinel-collision)).

## rung-0-baselines

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- A rung 0 holds the baselines that need no training on the target cell type:
  - the marginal NB per assay
  - the per-position cross-cell-type average
  - depth-only scaling of a related track
- In the imputation challenge, the per-position average beat all but two of 23 entries. In Max's re-scoring, uncorrected CANDI did not beat it.
- So rung 1 is a learned model above a floor, not the floor itself.
- **Source:** `CANDII: EVAL.md` (`marg_crps`), `plan/BENCHMARK_DESIGN.md` §7, `cruxvault/wiki/average-activity-baseline.md`; Max `notes/key_findings.md`.
- **Related:** [[onion eval protocol#nested-baseline-ladder]].

## rung-1-as-set-model

- A pointwise rung reads a set of observed tracks at one bin, so it is a set model.
- Conditional neural processes embed each observation, pool the embeddings with a permutation-invariant operation, and decode targets from the pooled value.
- The Set Transformer replaces the pooling with attention blocks, including inducing points and pooling by attention.
- eDICE applies Set Transformer blocks at each locus: one over the set of cell types and, separately, one over the set of assays. It is the closest published rung-1 comparator.
- **Source:**
  - Garnelo M et al. ICML 2018, arXiv:1807.01613.
  - Lee J et al. ICML 2019, arXiv:1810.00825.
  - Hawkins-Hooker A et al. Nat Commun 14 (2023), doi:10.1038/s41467-023-40211-2.

## name-the-candi-comparator

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Two CANDI checkpoints exist:
  - about 3.1 million parameters, with an 8-assay panel
  - 2,353,634 parameters, with a 35-assay panel plus a control
- Any comparison with "current CANDI" names which checkpoint it uses.
- **Source:** `CANDII: src/candi/README.md`; Max `notes/key_findings.md`.

## axis-ablations-for-axial-rung

- The MSA Transformer alternates attention along each row (sequence positions) and down each column (aligned sequences).
- In its tied row attention, the attention logits are summed across rows before the softmax, so all rows share one attention map. In its ablations, tied attention beat untied attention.
- For an axial rung, the matching arms are:
  - position-only attention
  - track-only attention
  - tied position attention across tracks
- **Source:** Rao RM et al. ICML 2021 (PMLR 139:8844), https://proceedings.mlr.press/v139/rao21a.html.

## perceiver-latent-sweep

- Perceiver IO maps inputs into a fixed latent array by cross-attention, processes the latents with self-attention, and decodes with output queries.
- Its encoder and decoder costs scale linearly with input and output size.
- The paper used latent counts from about 32 to 2,048, depending on the task.
- The claim that the bottleneck loses positional detail is unmeasured here. A sweep over the latent count, with the prediction's power spectrum compared with the truth's, would measure it.
- **Source:** Jaegle A et al. ICLR 2022, arXiv:2107.14795.

## copy-leak-guardrail

- Noise2Self showed that a denoiser can be trained without clean targets if its output at a coordinate does not depend on the input at that coordinate (J-invariance). Without that constraint, the identity map minimizes the self-supervised loss.
- So any rung that sees a track's counts at the output bin has this shortcut, not only the U-Net.
- CANDII's leak measurement found that 23.4% of available input columns in the old baked data were bit-identical to their targets. The structural rate for a 4-level depth ladder is 25%. The store reproduced the leak (21.1%) under the evaluation random-number generator.
- A guardrail has two parts:
  1. Perturb a masked bin's input and check that its output does not change.
  2. Score denoising only on masked or thinned bins.
- **Source:**
  - Batson J, Royer L. ICML 2019, arXiv:1901.11365.
  - `CANDII: cruxvault/results/t16/REPORT.md` (local), `tools/dsf_leak.py`.

## dna-arms-and-pretrained-leakage

- Arms for the DNA input, each with matched parameter counts:
  - no DNA
  - zeros
  - dinucleotide-shuffled sequence
  - sequence from the wrong locus
  - sequence shifted by 5 Mb
- A frozen pretrained model such as Flashzoi was trained on ENCODE tracks at known loci, so its gain can be leakage.
- Max's leakage square tested this. It compared the gain at the pretrained model's training loci with the gain at its test loci, separately for target experiments the model had seen and had not seen.
- 128 bp grids do not nest into 25 bp grids (gcd(25, 128) = 1). So pretrained embeddings need explicit resampling to the 25 bp grid.
- **Source:** Max s03: `scripts/metrics/leakage_square.py`, `notes/metrics.md` ("Phase 2"), `notes/design3.md`.

## covariate-injection-arms

- FiLM conditions a network by a feature-wise scale and shift computed from the conditioning input.
- In DiT, adaLN-Zero regresses the scale, shift and a residual gate from the condition, and initializes the gate at zero, so each block starts as the identity. It gave a lower FID than in-context conditioning and cross-attention.
- Classifier-free guidance replaces the condition with a null token in 10–20% of training steps. So one network learns both conditional and unconditional outputs.
- StyleGAN2 found that normalizing features after style modulation (AdaIN) caused artifacts. It replaced explicit normalization with weight demodulation. This is evidence that the order of modulation and normalization matters.
- **Source:**
  - Perez E et al. AAAI 2018, arXiv:1709.07871.
  - Peebles W, Xie S. ICCV 2023, arXiv:2212.09748.
  - Ho J, Salimans T. arXiv:2207.12598 (2022).
  - Karras T et al. CVPR 2020, arXiv:1912.04958.

## lane-isolation-test

- **Status:** idea, no evidence.
- A claim that a model keeps assays separate "by construction" can be tested directly. Perturb one assay's input lane and measure the change in every other lane's activations, before any designed mixing layer.
- Normalization across channels is one known route for leakage between lanes.

## head-arms-and-flat-sigma-badge

- Head arms:
  - an NB head trained by likelihood
  - an NB head trained by CRPS
  - a constant-variance head
- The benchmark design wraps a point-only prediction in a Gaussian whose width is fitted on training residuals only. Each such cell carries the badge "fitted flat σ", as opposed to "native heteroscedastic".
- A flat width cannot follow a varying truth, so point-only methods lose on PIT and coverage partly by construction.
- **Source:** `CANDII: plan/BENCHMARK_DESIGN.md` §7.

## loss-resolution-arm

- Max fine-tuned the model with the loss computed on signal pooled to 200 bp, 800 bp or 3.2 kb.
- None of these coarse losses helped. Each was worse than the 25 bp loss in normalized MSE at every scoring resolution from 25 bp to 3.2 kb, mostly through calibration.
- These are single fine-tunes, in asinh-p units.
- **Source:** Max s06: `output/summary2/rft_paired.tsv`; Max `notes/key_findings.md`.

## cac-loss-arms

- CPA trains an adversary so that its basal latent carries no information about perturbations and covariates. It then adds covariate embeddings back to predict unseen combinations.
- sysVI found that adversarial alignment can mix unrelated cell types when their proportions differ across systems. A latent cycle-consistency loss with a VampPrior kept more biological signal.
- So a CAC loss arm reports three things:
  1. adversary accuracy
  2. a post-hoc probe of the latent
  3. the cost to imputation
- **Source:**
  - Lotfollahi M et al. Mol Syst Biol 19:e11517 (2023), doi:10.15252/msb.202211517.
  - Hrovatin K et al. BMC Genomics 26:974 (2025), doi:10.1186/s12864-025-12126-3.

## pseudocount-sweep

- A per-bin log ratio with a +1 pseudocount has an expected value that follows the bin's signal level. It is near 0 in empty bins and near log2 f in strong bins.
- In CANDII's split-half test, the two halves of a pure depth-thinning pair therefore correlated at 0.34–0.63, with no real per-bin effect.
- This defect voided the outcome-neutral controls of a covariate-swap probe.
- A sweep over ε in {0.01, 0.1, 1}, plus an arm with no pseudocount (a count model with a depth offset), shows how much any log-ratio result depends on ε.
- **Source:** `CANDII: cruxvault/wiki/pseudocount-log-ratios-carry-a-signal-level-curve.md`, `cruxvault/f22_all_three_runs_failed_their_outcome_neut.md`.

## seeds-and-compute-matching

- In Max's sprint 07, two runs that differed only in seed differed by 0.001 to 0.011 r_spec (mean 0.007). Max treated single-run contrasts under about 0.01 as observations, not findings.
- At the same number of windows seen, batch 32 for 60k steps recovered about two thirds of the gain that batch 8 got from 60k to 240k steps. So step count and windows seen are reported together.
- GPU-slice and bf16 training were within the seed gap of CPU fp32 training.
- A two-seed ensemble lowered normalized MSE from 0.668 to 0.664 with no extra design work, so it serves as a free baseline.
- **Source:** Max `notes/key_findings.md`; Max s07: `report_sections/02_model.md`, `output/compute/equivalence.md`, `PLANNING.md`.

## shared-and-specific-error-per-rung

- Max's error decomposition splits residual error into a shared part (common to cell types) and a specific part (cell-type-specific). See [[onion eval protocol#max-error-decomposition-and-noise-floor]].
- Between architectures, most gains were in shared error. The Flashzoi sequence input cut shared error by 0.126 and specific error by 0.025.
- Against the cross-cell-type average, the best model's gain was all specific (−0.043), with no change in shared error.
- So a rung's gain is reported in both columns. A gain in shared error can come only from catching up with the average.
- **Source:** Max s06: `output/summary/v3_paired_differences.tsv`.

## external-rivals

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Rivals under the same evaluation:
  - ChromImpute, Avocado, eDICE and Corgi+
  - Borzoi's own output head
  - a ridge head fitted on the target, as an upper bound
- On the blind panel, with training on chr19 only, the genome-wide correlations were eDICE 0.502, the average 0.438 and CANDI 0.347. CANDI had the best CRPS (0.698 vs 1.28 to 2.31).
- These rivals learn a paired training cell during training. So a true new-cell benchmark needs them retrained with only accessibility observed.
- **Source:** Max s07: `notes/mehdi_check_2026-11-04.md`, `lit_recommendations.md`; Max `notes/key_findings.md`.

## comparison-protocol-ideas

- **Status:** ideas, no evidence.
  1. Architectures are not ranked on a single-chromosome training regime.
  2. Two separate wins are tested for additivity before they are stacked.
  3. Each rung passes a health gate before comparison: imputation correlation above zero, denoising at least as good as imputation, and PIT calibration within a set bound.
  4. The score is frozen before any automated search.

## depth-offset-is-an-assumption

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Under preprocessing #1, "depth is exactly additive in η" is a property of the offset construction.
- Max's diagnostics of a CANDII count head with the offset on found a slope of log2 μ on target log2 depth of 1.18–1.29, not 1. So the learned η also responded to depth.
- The same diagnostics found that the predicted p-value level correlated +0.17 to +0.59 with the depth of the input track, while the truth correlated 0.00.
- Two unit tests follow from this (proposals, not existing code):
  1. Change only the target depth, and require a slope of exactly 1.
  2. Change only the input depth, and require no change in the output level.
- **Historical note (PI, 2026-11-10):** earlier CANDI runs predicted the NB with the offset off on purpose. Depth steering is the easiest covariate effect to measure. With the offset off, the model need not learn the exact slope of 1. A learned slope close to 1 still showed that the model steered by depth, so the slope served as a check that CAC was happening.
- **Source:** Max s03: `output/metrics/v2_counts_depth_check.tsv`, `v2_level_diagnostics.tsv`, `notes/metrics.md`; `CANDI: cruxvault/h41_depth_output_steering_is_present_distrib.md`.
- **Related:** [[onion eval protocol#headline-metric-set]] (metric 5), [[onion eval protocol#cac-evaluation-gates]].

## eic-quantile-normalization-on-truth

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Under preprocessing #7: the ENCODE Imputation Challenge applied a quantile normalization to the truth tracks.
- It ranked each track's bases separately within its own peaks and its own background, and mapped them to per-assay reference levels.
- Max recovered the rule exactly. It reproduced the normalized truth at all 3.03 × 10⁹ bases for 47 of 48 tracks.
- This normalization removes depth and scale differences from the truth, so a score on it cannot show depth or scale correction. The blended CANDI ranked 1st of 26 on the normalized truth and 6th on the un-normalized single-end truth.
- **Source:** Max s04: `scripts/qnorm/apply_bp.py`, `notes/00_sprint_log.md`, `output/candi_placement.tsv`.
- **Related:** [[onion eval protocol#score-on-normalized-and-unnormalized-truth]].

## depth-as-covariate

- Candidate transform 8: raw counts, with depth passed as an offset or as a covariate.
- DESeq models a count as NB with mean equal to a size factor times a depth-free rate. It estimates the size factor as the median ratio of counts to per-gene geometric means. It does not divide counts by depth before the fit.
- sctransform regresses each gene's counts on log10 of the cell's total count, with NB errors and regularized parameters. So depth enters with a fitted slope.
- **Source:**
  - Anders S, Huber W. Genome Biol 11:R106 (2010), doi:10.1186/gb-2010-11-10-r106.
  - Hafemeister C, Satija R. Genome Biol 20:296 (2019), doi:10.1186/s13059-019-1874-1.

## borzoi-soft-clip

- Candidate transform 9, for the input side only.
- Borzoi raised each coverage bin to the power 3/4. Where the result exceeded 384, it applied a square root to the excess: y = 384 + √(y^(3/4) − 384). This limited the loss share of very highly expressed genes.
- The description is from Borzoi's RNA-seq coverage methods, and it was not checked for every assay.
- **Source:** Linder J et al. Nat Genet 57:949 (2025), doi:10.1038/s41588-024-02053-6 (Methods).

## size-factor-from-background

- S3norm matches a dataset to a reference by two factors: one fitted on common peaks and one on common background. So it normalizes depth and signal-to-noise together.
- SES scales the input control to the ChIP sample using only bins it classifies as background. In its example, the background-based factor was 0.34, against 1.05 from total read counts.
- Both show that a size factor from total reads can be far from one estimated on background.
- **Source:**
  - Xiang G et al. Nucleic Acids Res (2020), doi:10.1093/nar/gkaa105.
  - Diaz A, Park K, Lim DA, Song JS. Stat Appl Genet Mol Biol 11(3) (2012), doi:10.1515/1544-6115.1750.

## pearson-residuals-details

- For preprocessing #5: analytic Pearson residuals use a fixed offset model, μ = depth × gene share, with slope 1 and an overdispersion θ = 100 by default. θ = 10, θ = 100 and Poisson gave only small differences.
- Both this method and sctransform clip residuals at ±√N, where N is the number of cells.
- So a Pearson-residual transform names its dispersion estimator and reports how often clipping fires per assay.
- **Source:** Lause J, Berens P, Kobak D. Genome Biol 22:258 (2021), doi:10.1186/s13059-021-02451-7.

## sentinel-collision

- **Status:** idea, no evidence.
- Transforms 2 to 7 can give negative values. Negative values collide with negative sentinel codes for missing or masked entries.
- An explicit availability-mask channel avoids the collision.
