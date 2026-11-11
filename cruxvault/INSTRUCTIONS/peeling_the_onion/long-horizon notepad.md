# Long-horizon notepad

One entry per PI ruling, agent-level call, settled ablation or logged agent decision. Each entry is a `## <date> <title>` section. The PI rulings of 2026-11-10 are binding and supersede the first plan's decisions list.

## 2026-11-10 PI ruling 1: the new EIC dataset comes first
The PI rescoped the programme to EIC first. The new EIC dataset is the dataset for the whole miniCANDI ladder. The extended 4-axis dataset keeps only its catalog and census work now. Its build is a gate after the done rule (D14). The onion layers use the replicate rungs that the EIC experiments' own replicates and libraries give, and report pair counts and interval widths.

## 2026-11-10 PI ruling 2: the selection split is EIC's own split
Models train on the EIC training tracks. Ablations are scored on the EIC validation tracks. The blind tracks are touched once, at the end. Headline metric 4 (correction gap closed versus depth-only scaling) is scored on never-trained covariate pairs of the validation tracks.

## 2026-11-10 PI ruling 3: measure the cost of the uniform base
The alignment-only FASTQ cost is measured on the EIC reprocessing and reported as CPU-h per experiment and as wall time at the concurrency reached. This cost is not a blocker.

## 2026-11-10 PI ruling 4: the scorable set
An assay enters the headline macro averages and the gate thresholds only when validation holds enough tracks and cell types, and its replicate rungs give a usable interval. The other assays stay in training and are reported in a separate table.

## 2026-11-10 PI ruling 5: no EpiATLAS controlled access
EpiATLAS controlled access is refused. EpiATLAS is catalogued, and only public data is used.

## 2026-11-10 PI ruling 6: the floor of headline metric 4 is half against half
The floor predicts the target's pseudoreplicate half from its other half. The model and the depth-only baseline are scored on the same half.

## 2026-11-10 PI ruling 7: seed spread
The seed spread of a quantity is its maximum minus its minimum across seeds. A comparison uses the larger spread of the two arms compared.

## 2026-11-10 PI ruling 8: withheld never-trained covariate combinations
The agent chooses the withheld never-trained covariate combinations and logs the choice here. Pure depth-subset pairs do not count toward correction validity.

## 2026-11-10 PI ruling 9: cluster and pipeline environment
The PI has no preference for the cluster that runs the FASTQ processing and holds the pipeline images. The environment change is approved on the chosen cluster: a Caper venv, chip-seq-pipeline2, atac-seq-pipeline, and dnase-seq-pipeline v3.0.0 as a new pull. Wave 0 checks the Nibi scratch quota discrepancy, and the cluster is picked on that evidence.

## 2026-11-10 PI ruling 10: Flashzoi and genomic-language-model venv
A separate venv for Flashzoi and the genomic language models is approved. Max's setup is reused if it exists.

## 2026-11-10 PI ruling 11: the Nibi testbed scratch copy
The PI runs the Globus transfer of Nibi `/scratch/mforooz/t112_cf` to Fir. The foreman then deletes the Nibi scratch copy after it verifies the sizes.

## 2026-11-10 PI ruling 12: superseded by ruling 13
Ruling 12 asked that the space of headline metric 1 be a parameter, with both spaces computable. Ruling 13 replaces it.

## 2026-11-10 PI ruling 13: headline metric 1 has two fixed spaces
The PI confirmed two spaces for headline metric 1 (specific variance recovered above noise). The counts space (log1p of counts at the target's depth, with the pseudoreplicate noise floor) serves the selection rule D13. The challenge's signal space (D9b signal for miniCANDI, published entries as they are) serves the comparison with external EIC methods. The scorer takes no option to change the selection space.

## 2026-11-10 PI ruling 14: architecture autonomy
The agent may try architecture ablations beyond the listed ones without asking. The PI's examples: the control as a maskable 36th channel and an output-only smooth lambda head, as two arms of the control ablation; the rung 5 axis arms; the rung 6 latent arms. Such arms stay outside D14 condition 1 unless they are listed. The selection rule D13 settles them like any arm, and each is logged here with its evidence. The EIC model keeps a control output in all cases (D9b).

## 2026-11-10 PI ruling 15: data movement autonomy
The agent moves data between Nibi and Fir as it sees fit, under these constraints. (1) Cluster-to-cluster moves go through Globus, over an already-authenticated route only. The agent never authenticates for the PI; if no such route exists, the chunk stops and says so. (2) Datasets never transit the laptop. (3) Sizes and checksums are verified at the destination. (4) The foreman deletes Nibi `/scratch/mforooz/t112_cf` only after that verification. The purge of that scratch tree is noted for 2026-11-20.

## 2026-11-10 Agent call: the scorable-set rule
An assay is imputation-scorable when validation holds at least 3 tracks from at least 3 distinct cells, each with pseudoreplicate halves, and training holds at least 5 tracks. An assay is correction-scorable when at least 2 validation tracks each have at least one never-trained pair that is not a subset pair. A replicate rung is usable for an assay with at least 5 pairs from at least 3 cells; otherwise the rung is reported pooled by assay class with a "few pairs" flag. From the Fir manifest of 2026-11-10, the expected imputation-scorable set has 10 assays: 11 assays have 3 validation tracks, but H4K20me1 has only 3 training tracks. Every other assay stays in training and goes in `non_scorable_assays.tsv`.

## 2026-11-10 Agent call: subset pairs are excluded from headline metric 4
A subset pair is a pair where the reads of one side are a subset of the other side's reads, i.e. pure depth thinning. For such a pair, depth-only scaling is exact in expectation. Subset pairs never count toward headline metric 4 or its gates. They are reported apart as `subset_pairs_gap_closed`.

## 2026-11-10 Agent call: fixed seeded random regions are the scoring scope
Scoring uses fixed regions of 16384 bins, drawn at random with a fixed seed, each with a mask fraction of at least 0.9. The eval set (`evalset.json`) stores them, so every arm is scored on the same regions.

## 2026-11-10 Agent call: the simplicity order lives in each arm config
Each arm config declares `simpler_than_base`: `true` for `dna_off` and for an arm that removes a component, `false` for an arm that adds one, `null` when neither is clearly simpler. The selection rule reads it as its `--simpler` input. When the difference is within the seed-spread bars, the simpler arm stays (D13); `null` gives a tie that goes to the PI.

## 2026-11-10 Agent call: matched compute means equal training FLOPs
Two arms have matched compute when their training FLOPs are equal, i.e. windows seen times FLOPs per window. FLOPs per window are measured on one batch with `torch.utils.flop_counter.FlopCounterMode` (torch 2.6, no new dependency). Matched parameters means a parameter count within 10% of the budget (default 1.0e6). The pilot collect chunk sets the two common budgets and logs them here.

## 2026-11-10 Agent call: assay-class members
`accessibility_dnase` is DNase-seq. `accessibility_atac` is ATAC-seq. `broad_histone` is H3F3A, H3K27me3, H3K36me3, H3K4me1, H3K79me1, H3K79me2, H3K9me1, H3K9me2, H3K9me3 and H4K20me1. `narrow_histone` is every other histone mark of the panel. `control` is chipseq-control. The source is the ENCODE histone standards page; the EIC accession ledger chunk verifies the broad list and reports any change. The panel has no TF ChIP-seq.

## 2026-11-10 Agent call: metadata strata
Every imbalance and balance report uses these metadata strata: (1) run type; (2) read length bin (36 or less, 37 to 75, 76 or more); (3) log2 depth tercile within assay; (4) lab; (5) platform.

## 2026-11-10 Agent call: sources of the region strata annotations
The region strata masks use these sources. Promoters and enhancers come from ENCODE cCRE v4 for hg38 (PLS, pELS and dELS). Gene bodies come from GENCODE basic, the release that cCRE v4 used. Heterochromatin comes from the full-stack ChromHMM universal annotation for hg38 (Vu and Ernst 2022, states HET1 to HET9). Mappability deciles come from the 36-mer single-read Umap track for hg38, the same track that the testbed used. The builder verifies each URL by a web lookup before coding and records URL, md5 and version in the masks file.

## 2026-11-10 Agent call: the pilot on the existing EIC store is plumbing only
The pilot on `CANDI_STORE/eic` proves the whole path and gives two of the three goal terms. It is labelled "pilot, not a D13 settlement" and settles nothing. It is not a valid D13 scorer for three reasons. (1) Headline metric 1 needs a pseudoreplicate noise floor per validation track, and the store holds no reads. (2) Headline metric 4 needs never-trained, non-subset covariate pairs on validation tracks, and the store has none. (3) The store's tracks went through mixed native processing, so training on them and scoring on uniform validation tracks would repeat the EIC defect.

## 2026-11-10 Agent call: the default cluster and its rule
Nibi is the default ONION cluster for the FASTQ processing, the new EIC dataset, training and scoring. Fir serves the pilot on the existing `CANDI_STORE/eic`, the catalogs if Nibi nodes lack internet, and overflow training. The kit and cluster-probe chunk confirms or overturns the default on evidence. It overturns it to Fir only if a 1 GiB write test on Nibi `/scratch/mforooz` fails or `quota -s` shows an expired grace. It writes the name to `.orchestrate/cluster` and the evidence to `.orchestrate/cluster_evidence.md`.

## 2026-11-10 Agent call: eic_v1 is frozen for the whole ladder
The tables `eic_v1` hold build priorities 1 to 3: the validation and training experiments with their controls, base views, pseudoreplicate halves, per-replicate views, alignment-level arms, and FASTQ-level arms (on all validation experiments and on a stratified training subset). Once built and checked, `eic_v1` is frozen for the whole miniCANDI ladder. Priority 4 (per-library and per-lane views, blind experiments) goes in `eic_v2` and never changes the ladder's data.

## 2026-11-10 Agent call: the error budget is ported from Max's repository
The error budget (level, shared and specific; explained, unexplained, noise and added) is ported from Max's repository at commit 3e7478a1: sprint 06 `errdecomp` and `aggregate2` `eue`, and the sprint 07 `eue_rep` floor. The port differs from Max's code in four points. (1) It works in the caller's space; the onion scorer calls it in counts (log1p at the target's depth). (2) The noise comes from the target's own pseudoreplicate halves by the Spearman-Brown correction with depth ratio 2, which is approximate under log1p. (3) It uses regions of 16384 bins. (4) An assay with a single cell gives NaN for the shared and specific parts. Evidence: the plan Log entry of the error-budget merge (commit 3b5f799).

## 2026-11-10 Agent call: the copy-gain threshold is 10% of the hidden-bin CRPS
The same-view copy gain, i.e. CRPS on hidden bins minus CRPS on visible bins, is flagged when it is more than 0.10 of the hidden-bin CRPS. The threshold was calibrated on synthetic tables over 5 seeds. An honest model that never reads a visible bin's own value gave a relative gain of -0.006 to 0.012. Models that blend in a visible bin's own value gave 0.16 to 0.43. The check fails closed when it has no entries, and the wiring tolerance is 1e-6. The gain on a constant model moved by about 15% across mask seeds on synthetic tables, so a caller must use enough batches and report the spread. Since PI ruling 17 the gain is a diagnostic and does not gate. Evidence: `tests/test_onion_eval_guardrail.py` and the plan Log entry of the guardrail merge (commit 5a3d47e).

## 2026-11-10 Agent call: onion layers thin every pair to the assay's shallowest pair depth
The onion-layer code thins every pair of an assay to the depth of that assay's shallowest pair, by binomial thinning per bin. So all replicate rungs of one assay share one depth, and a layer difference is not a depth difference. Evidence: the plan Log entry of the onion-layer merge (commit a6f532e).

## 2026-11-10 Agent call: onion-layer intervals are a cluster bootstrap over cells
The interval of a layer is a cluster bootstrap over cells: SE x t(0.975, n_cells - 1) x sqrt(n / (n - 1)). Its measured coverage was 0.953 over 12 seeds at 8 cells. With 1 cell the interval is plus or minus infinity. A rung with few pairs, or an empty rung, also gets a row pooled over its assay class. The code was not yet run on store counts or on the real region masks. Evidence: the plan Log entry of the onion-layer merge (commit a6f532e).

## 2026-11-10 Agent call: headline metric 1 and the correction gap closed are ratios of means
Per assay, headline metric 1 (specific variance recovered above noise) and the correction gap closed are ratios of means over the targets, not means of per-target ratios. A per-target ratio with a small denominator would otherwise dominate the mean. Evidence: the plan Log entry of the metrics library, accepted after one retry (commit 29e82b9).

## 2026-11-10 Agent call: the shared and specific split calibrates the prediction and reports the level beside it
The shared and specific split calibrates the prediction by default, as in Max's sprint 06. So the imputation term excludes the level and scale error. The level is reported beside headline metric 1 for every target in every scores file. Evidence: the plan Log entry of the metrics library (commit 29e82b9).

## 2026-11-10 Agent call: pseudoreplicate halves are rescaled to the target depth before log1p
Each pseudoreplicate half holds about half the reads of its target. The halves are multiplied by 2, i.e. rescaled to the target's depth, before the log1p transform. The noise floor and the error budget then compare like with like. Evidence: the plan Log entry of the metrics library (commit 29e82b9).

## 2026-11-10 Agent call: an undefined metric is a NaN that carries its reason
An undefined metric value is a subclass of NaN that carries the reason why it is undefined. The scorer writes the reason beside the null value, so a missing number is never silent. Evidence: the plan Log entry of the metrics library (commit 29e82b9).

## 2026-11-10 Agent call: the variance ratio thins the deeper replicate
The ratio of predicted to replicate variance needs integer counts. It matches the depths of two replicates by thinning the deeper one with a seeded binomial draw. Evidence: the plan Log entry of the metrics library (commit 29e82b9).

## 2026-11-10 Agent call: one fixed base depth of 2^23 reads
The miniCANDI batch contract uses one fixed base depth of 2^23 reads (about 8.4 million) for every assay class. Evidence: the plan Log entry of the miniCANDI batch contract, accepted after one retry (commit dc32049, 134 tests).

## 2026-11-10 Agent call: validation items rotate over three tasks
The validation items of the miniCANDI dataset alternate imputation, correction and reconstruction tasks, so a validation loss covers all three. Evidence: the plan Log entry of the miniCANDI batch contract (commit dc32049).

## 2026-11-10 Agent call: the control is an input and an output by default
The configuration keys `control_input` and `predict_control` are true by default. So the ChIP-seq control enters the model as an input, and the model predicts a control. Evidence: the plan Log entry of the miniCANDI batch contract (commit dc32049).

## 2026-11-10 Agent call: a withheld row blocks a pair across the whole window
A withheld row blocks any input-target pair in a training window, across tracks too. A row for one assay class applies when the source's or the target's class matches; an empty or `*` class applies to every class. This rule is conservative and removes some cross-track training windows. Evidence: the "choices to log" list of the review of the miniCANDI batch contract (`.orchestrate/review_M1.md`).

## 2026-11-10 Agent call: withheld matching reads only the arm knob and its value
Withheld matching uses only `arm_knob` and `arm_value`. Units (pooled, biological replicate, technical replicate) and layers (native, uniform) all count as "base". Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: covariates are raw values with log2 depth first
Covariates enter the network as raw values. Column 0 is log2(depth in reads), standardized inside the network by fixed constants (centre 24, scale 2). Depth counts reads, not fragments, while the noise floor of headline metric 1 counts fragments. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: covariate file layout and categorical ids
The covariate encoding is fixed in `covariates.json`. Categorical ids are capped at 1024; an unknown or overflow value maps to id 0. A rung that cannot run an arm raises `NotApplicable`. The batch contract's `wiring_check` uses an elementwise relative change, while the guardrail normalizes by the maximum. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: masking defaults
The masking defaults are: `mask_prob_range` 0.2 to 0.8, `same_view_prob` 0.5, `bin_mask_frac` 0.25, `min_valid_frac` 0.5 with 8 tries, and `exclude_chroms` chrY and chrM. At least one track of a window stays unmasked, and a cell with one track is never masked. A track with no compatible candidate becomes an imputation target or is dropped. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: all controls of a cell are views of one control track
All control experiments of a cell are treated as views of one control track (track 35). So two different controls of one cell form a correction pair. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: the shuffled-DNA arm permutes 25 bp blocks
The `dna=shuffled` arm permutes 25 bp blocks, with one permutation per batch. The permutation is fresh in training and comes from `shuffle_seed` in evaluation. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: count transforms add depth back as an offset
The `local_ratio` and `quantile` count transforms add depth back in the offset form, so the NB output stays on the count scale. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: the NB loss is reused with fixed clamps
The NB negative log-likelihood reuses `tools/t118/ladder/model.py`: float64 inside, the mean clamped to [1e-4, 1e6], the size clamped to [1e-3, 1e4], and zero gradient outside the clamps. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: correction training uses every bin, subset pairs included
Correction pairs are trained on every bin, including subset pairs, where depth-scaled copying is the correct answer. Subset pairs still never count toward headline metric 4. Evidence: `.orchestrate/review_M1.md`, "choices to log".

## 2026-11-10 Agent call: rungs do not validate their batch inside forward
No rung calls `validate_batch` inside `forward`, because the guardrail feeds nonzero counts into hidden bins to test the wiring. Every rung must pass the batch contract's wiring check and part 1 of the guardrail. Evidence: `.orchestrate/review_M1.md`, "choices to log", and the brief addendum.

## 2026-11-10 Agent call: scored correction pairs share no reads
A scored correction pair takes its source view from the pr1 half of the reads and its target from the pr2 half of the target arm. The two halves come from one read-level split, so the source and the target share no read. This applies PI ruling 6 (half against half). Evidence: the amendment in the plan Log entry of the first review of the miniCANDI batch contract, and `cruxvault/results/t170/withheld_design.md`.

## 2026-11-10 Agent call: counts files are chunked at 4096 bins or less
The counts h5 files of the new dataset use chunks of at most 4096 bins, so a training window reads few chunks. Evidence: the amendment in the plan Log entry of the first review of the miniCANDI batch contract.

## 2026-11-10 Agent call: per-assay sums are precomputed for the leave-one-out average
The new dataset holds a precomputed per-assay sum and count h5, so the leave-one-out per-position average (decision D10, average as input) is a subtraction and not a pass over all tracks. Evidence: the amendment in the plan Log entry of the first review of the miniCANDI batch contract.

## 2026-11-10 Agent call: the cluster choice, on evidence
The ONION cluster is Nibi. The rule kept the Nibi default unless a 1 GiB write test on Nibi `/scratch/mforooz` failed or `quota -s` showed an expired grace. The write test passed on node c405 (job 26809968). `quota -s` showed 1826G used against a soft quota of 1024G and a hard limit of 20480G, with no grace timer. Compute nodes on both clusters reached the ENCODE portal and S3. Nibi `/project` was at 4181 of 4657 GiB, so bulk data goes to Nibi `/scratch`. FairShare was cpu 0.233 and gpu 0.211 on Nibi, cpu 0.148 and gpu 0.117 on Fir. Two Fir nodes failed the venv step (fc30601, fc30568). Evidence: `.orchestrate/cluster_evidence.md`; jobs Nibi 26810638, 26810017 and Fir 3423981 (cancelled), 3425132 (failed), 3425384, 3422681.

## 2026-11-10 Agent call: a chunk may create the Nibi nearline directory
`/nearline/def-maxwl/mforooz` does not exist on Nibi, and `/nearline/def-maxwl` is writable. The foreman allowed a chunk that needs it to create the directory. Evidence: the plan Log entry of the cluster probe merge (commit 4072cfa) and `.orchestrate/cluster_evidence.md`.

## 2026-11-10 Agent call: two-step correction is a stage-2 ablation on the incumbent
Rungs 3, 5 and 6 mark two-step correction as not applicable at the rung level, and rung 1 raised NotImplementedError, to be made not applicable. The foreman accepted this: two-step correction (decision D15) is a stage-2 ablation on the incumbent design, and the planner adds it to the incumbent's rung at gate G3. If the axial rung becomes the incumbent, G3 must also add the average-as-input arm (decision D10). Evidence: the plan Log entries of the rung 6 and rung 5 merges (commits 60ee145, e96472d).

## 2026-11-10 Agent call: the scrambled-covariates rule and a second control
The scrambled-covariates control scrambles what the covariate encoder reads and keeps each track's true depth in the count-transform offset. So it measures the use of covariates beyond the depth offset. A second control, `scramble_all`, also scrambles the depth offset. The rung-consistency work defined the scramble once in `covariates.py`, applied in the model exactly once; the training loop no longer scrambles. Evidence: the foreman call in the plan Log entry of the rung 2 merge (commit 48f37b2) and the rung-consistency merge (commit 2465b48, 460 tests).

## 2026-11-10 Agent call: every rung's wiring test perturbs a zero-initialized head first
A wiring check on an untrained model passes trivially when the NB head is zero-initialized, because no output can move. So every rung's wiring test perturbs the model's parameters first and includes a power test. Under this real wiring check no rung leaked. Evidence: the plan Log entries of the rung 4 merge (commit 50f41ed) and the rung-consistency merge (commit 2465b48).

## 2026-11-10 Agent call: the Nibi-to-Fir transfer of the testbed is dropped
No authenticated Globus route existed on the laptop, Nibi or Fir (no CLI, no `~/.globus`). A loose copy would also overfill Fir `/project`: the source is 9.46 TiB when hard links are expanded (1.55 TiB unique), and Fir `/project` was at 20 of 28 TiB. Under ruling 15 the foreman dropped the transfer, because the ONION cluster is Nibi and the testbed stays there. The valuable parts are the products (40 G) and pseudoreplicates (80 G) on Nibi scratch, and the tagAligns (75.4 GB, 265 files, already on Nibi `/project`, sizes match). They are to be imported into the ONION layout on Nibi before the purge date of 2026-11-20. The FASTQ cache and the Caper localized inputs can be downloaded again and are left as they are. Nothing was deleted, because the PI approved deletion only after a verified transfer. Evidence: the plan Log entry of the Globus transfer tool merge (commit 9d476ff).

## 2026-11-10 Agent call: region strata sources as built
The region strata masks were built from these sources. Gene bodies: GENCODE v40 basic, the release that cCRE V4 used, protein-coding genes from TSS to TES. Promoters: cCRE V4 class PLS. Enhancers: cCRE V4 classes pELS and dELS. Heterochromatin: full-stack ChromHMM states HET1 to HET9. Mappability: ten boolean decile strata by the earlier 1 kb-block rule. Evidence: `cruxvault/results/t161/STRATA.md` and Nibi job 26813690.

## 2026-11-11 Agent call: the mappability deciles tie at the top
The mappability strata are not ten equal groups. Fully mappable 1 kb blocks tie at the top value, so they all fall in decile 8 (35% of all bins), and deciles 7 and 9 are empty; eight strata hold data. This is kept as it is: the deciles agree with the arrays of the earlier testbed with 0 mismatched bins. The other fractions of all bins are promoters 0.0047, enhancers 0.1610, gene bodies 0.3983 and heterochromatin 0.0687. Evidence: `cruxvault/results/t161/STRATA.md`, job log of Nibi job 26813690.

## 2026-11-11 Agent call: EIC DNase-seq rows are rebuilt from the published experiments
The accession check matched all 363 published EIC tracks (train 267, validation 45, blind 51) by experiment accession and role, against Schreiber 2023 Genome Biology Additional file 5 (md5 71d6a11f...). It found 10 mismatches. In 7 of them a training DNase-seq row's store file comes from another experiment, because the published experiment was archived or superseded. Under decision D8 the new EIC dataset builds every track from the published experiment's own reads, so these 7 rows are rebuilt from the published experiments' FASTQs. If an archived experiment has no retrievable FASTQ, the list goes to the PI. Evidence: `cruxvault/results/t158/LEDGER_CHECK.md`, `mismatches.tsv`, Nibi job 26815344.

## 2026-11-11 Agent call: controls enter training only when they serve a training experiment
A control enters EIC training only when it is the control of at least one training experiment. So C19CTL, C28CTL and C39CTL, the controls of blind-only experiments, leave training (the other 3 of the 10 mismatches). 51 ChIP-seq rows have a store control that is not among ENCODE's possible controls, so the build picks controls from the accession ledger, not from the store. Evidence: `cruxvault/results/t158/LEDGER_CHECK.md`, Nibi job 26815344.

## 2026-11-11 Agent call: the testbed tracks are blind tracks
All 7 counterfactual tracks of the earlier testbed are EIC blind tracks. Under decision D8 they are never training data. They serve only the final blind evaluation and plumbing tests. Evidence: the foreman call in the plan Log entry of the accession-ledger merge (commit ce56a09). The same check measured how far training is from blind: platform total variation distance 0.970 (blind 51 of 51 HiSeq 4000, training 8 of 267), lab 0.922 (blind 48 of 51 from the Snyder lab, training 5 of 267), run type 0.903 (blind 100% paired-end, training 9.7%). Validation resembles training (distances 0.012 to 0.131). Evidence: `cruxvault/results/t158/eic_imbalance_summary.tsv`.

## 2026-11-11 Agent call: comparisons must be comparable or the rule refuses
Every scores file in a D13 comparison, and every baseline in D14 condition 2, must carry the same non-empty eval-set and tables hashes, the leave-one-assay-out regime, the counts selection space, the pseudoreplicate noise floor and the expected split. Otherwise the selection rule refuses and writes no row. A pilot scored without halves is refused. Evidence: finding 1 of the scorer review (`.orchestrate/review_E6.md`) and the fix merge (commit 1936fdc, 130 tests).

## 2026-11-11 Agent call: a zero seed spread refuses the comparison
When a quantity has a seed spread of exactly 0 over 3 or more seeds, the seeds do not vary and the 2x-spread bars are void. The selection rule then refuses and escalates. Without this, a difference of 1e-9 would win. Evidence: the foreman call in the plan Log entry of the scorer review, finding 5 of `.orchestrate/review_E6.md`.

## 2026-11-11 Agent call: a vetoed win counts as a tie
When an arm's composite wins by more than 2x the spread but one goal drops by more than 2x that goal's own spread, the win is vetoed and the comparison counts as a tie. The simpler arm then stays, or the tie goes to the PI when neither arm is simpler. Evidence: the foreman call in the plan Log entry of the scorer review.

## 2026-11-11 Agent call: leave-one-assay-out inputs are the cell's training tracks only
In the leave-one-assay-out regime the inputs of a validation cell are its training tracks only. The cell's other validation assays are also withheld. Evidence: the foreman call in the plan Log entry of the scorer review.

## 2026-11-11 Agent call: tolerances of the goal terms
The selection rule escalates a goal term whose denominator is too small. The uncertainty denominator, the marginal-NB PIT deviation, must be at least 0.005 (absolute). The correction denominator, (depth-only CRPS minus floor CRPS) / depth-only CRPS, must be at least 0.01 in every correction-scorable assay. Before PI ruling 18 a term outside the band [-1, 1.5] also escalated; with clipping on, that band no longer escalates. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: the copy-gain spread must be well under its threshold
The same-view copy gain's spread across mask seeds must be at most 0.5 x the 0.10 gain threshold, else the gain is marked unreliable. This matters only if the copy gain gates again; under PI ruling 17 it is a reported diagnostic. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: the predicted control is asked for at the target's own covariates
When a ChIP-seq target has no observed control, the scorer asks the model for a control at the target's own output covariates, with the control depth and read length set to the treatment's. The signal track labels its lambda source `predicted_control`. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: the blind eval set path
The blind eval set is `eval/eic_v2/evalset_blind.json`. A blind pass records `run_tables_sha256`, because the blind tables `eic_v2` are a superset of the training tables. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: blind passes are ledgered
Every blind scoring pass is written to `BLIND_PASSES.tsv`, so a second look at the blind tracks is visible and needs a stated reason. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: pilot scoring without halves is never a selection input
The pilot on the existing EIC store has no pseudoreplicate halves. It is scored in a `--no-halves` mode, whose scores carry a key that the selection rule refuses. Evidence: the plan Log entry of the scorer fix merge (commit 1936fdc).

## 2026-11-11 Agent call: the scorer's cost
The oracle rescale in the scorer costs about 1 s per target. On the pilot, scoring cost about 1.7 min per target. Evidence: the plan Log entries of the scorer merge (commit 12aa154) and of the pilot collect interim.

## 2026-11-11 Agent call: the guardrail gate is the PI's wiring check
The hard gate of D13 is the guardrail as the PI's document defines it. (1) Perturbing a masked bin's input must not change its output; this is the gate. (2) Denoising is scored only on masked or thinned bins, or against an independent draw; this is a scoring rule. The same-view copy gain with its 10% threshold became a reported diagnostic, not a gate. A mask-trained model that sees a visible bin's own value is expected to use it, and copying is penalized by the scored metrics against the other pseudoreplicate half. The pilot on real data gave this evidence: part 1 was exactly 0.0 on every rung, while the relative copy gain over 5 mask seeds was 0.57 to 0.71 on rung 4, 0.14 to 0.20 (train) and 0.064 (validation) on rung 3, 0.08 to 0.18 (train) and 0.044 (validation) on rung 2, and -0.01 to 0.08 on rung 6, with a mask-seed spread of 0.06 to 0.10. The PI confirmed this as ruling 17. Evidence: the plan Log entries of the pilot submit merge (commit 2e48606) and of the foreman call; the code change merged at 16eb9cb (Fir check job 3436642).

## 2026-11-11 Agent call: the withheld never-trained combinations of eic_v1
Under PI ruling 8, `eic_v1` withholds 18 directed (source view, target view) combinations per assay class, two knobs each and never a whole knob: nodedup to mapq0 and back (broad histone, narrow histone, DNase-seq), crop36 to crop50 and back, and se to crop50 and back (broad histone, narrow histone, ATAC-seq). DNase-seq has no crop in training (all its training reads are 36 bp or shorter), and ATAC-seq has no mapq0 arm. Every validation track gets at least 2 never-trained non-subset pairs: 38 tracks get 2, 4 tracks get 4, and 3 tracks get 6. Pure depth pairs are subset pairs and never counted. `se` to `base` stays trained on all 43 paired-end training experiments, because the blind set is 100% paired-end. The arms can imitate paired-end to single-end and long to short reads only. They cannot imitate platform, lab, or single-end to paired-end. Evidence: `cruxvault/results/t170/withheld_design.md`, Nibi check job 26824524.

## 2026-11-11 Agent call: what ENCODE adds (decision D4)
The ENCODE catalog on Nibi (`ONION/catalog/encode`) holds 8494 experiments and 291436 files. The replicate-ladder census gives these pair counts, over the whole portal and over the EIC experiments (in brackets, the number of assays with at least 20 pairs): pseudoreplicate 14652 (15) and 729 (9); sequencing run with a known flowcell and lane 121778 (8) and 110 (1); technical 4784 (8) and 29 (0); isogenic 4645 (13) and 301 (7); anisogenic 273 (6) and 4 (0); other experiment 94773 (11) and 21 (0, 15 of them control pairs); other cell type 6371582 (25) and 5427 (12). 706 of 1411 EIC FASTQs have no flowcell or lane. The EIC isogenic rung exists for 34 of 35 assays and has at least 20 pairs for 7. So ENCODE gives the uniform base and the replicate structure, but the EIC experiments alone populate few rungs. Evidence: `cruxvault/results/t156/CENSUS.md`, about 5.9 CPU-h.

## 2026-11-11 Agent call: what EpiATLAS adds (decision D4)
EpiATLAS has 5339 observed experiments of 6 histone marks; 894 (16.7%) have public reads. 9539 of its 14878 panel rows are imputed; they are flagged and never enter training or truth. 1207 observed experiments are ENCODE or Roadmap data re-run through the IHEC pipeline, so each gives a pipeline counterfactual on the same reads. Its `noseq.bam` alignments are public, so counts, depth thinning and pseudoreplicate halves can be built without controlled access (ruling 5). It adds labs and protocols for the 6 core marks at a fixed pipeline, and nothing for ATAC-seq, DNase-seq or the other 29 marks. It has no isogenic or technical rung. It feeds gate G5 (the extended dataset), not the EIC-first work. Evidence: `cruxvault/results/t157/source_report.md`, Nibi job 26815002 (0.14 CPU-h).

## 2026-11-11 Agent call: what ChIP-Atlas adds (decision D4)
ChIP-Atlas holds 91441 non-control panel experiments and 18504 input controls on hg38, for 34 of 35 panel assays (no H3K23me2). Its reads are public (300 of 300 in a spot check), so every FASTQ-level arm and the uniform base can be built. For the 23 rare panel marks it holds 1442 experiments, against ENCODE's 190. 4823 (cell type, assay) groups hold at least 2 experiments; these are candidates for the cross-lab rung that EIC cannot populate, but their replicate status is not confirmed. It feeds gate G5, not the EIC-first work. Evidence: `cruxvault/results/t157/source_report.md`, Nibi job 26815002.

## 2026-11-11 Agent call: the DNase-seq pipeline runs as two calls of the release's own tasks
dnase-seq-pipeline v3.0.0 (commit c1082e63) has no alignment-only switch. So the setup wraps the release's own tasks in a two-call WDL. The README's reference bucket `gs://dnase` is gone, so the references are the ENCODE reference set ENCSR083JOX. The setup is on Nibi `ONION/pipelines/dnase` (+11 files, 6.1 GB on `/project`). Evidence: `cruxvault/results/t169/DNASE_PIPELINE.md` and the plan Log entries of the DNase-seq pipeline work (commit 68f8c12).

## 2026-11-11 Agent call: download routes from Nibi and Fir were measured
Downloads from ENCODE's S3 bucket (us-west-2) to Nibi run at about 0.14 MB/s per TCP flow, linear in the number of flows, 9 MB/s at 64 flows (31 h per TB), from login and compute nodes alike. The path is the limit: Nibi to us-east-1 S3 runs 54 to 74 MB/s on one flow. Fir compute nodes reach 5.5 MB/s per flow, 70 MB/s at 32 ranges and 129.5 MB/s with 4 files of 16 ranges each (2.1 h per TB), md5 verified. Nibi has no data-transfer node, and there is no Globus collection for ENCODE or AWS. The Fir to Nibi Globus leg is unmeasured; it needs a PI login. At 5.7 MB/s the full build list (2791.6 GB) takes 136 h, and the validation bases of all 10 scorable assays (229.9 GB) take 11.2 h. The plan Log records the measurement but no decision on the route yet: it says only that the staging, not CPU, bounds the build if throughput stays near 6 MB/s. Evidence: probe jobs Nibi 26823539, 26824325 and Fir 3440471, 3441260, 3442621; helper Fir `/scratch/mforooz/onion/netprobe/rangefetch.sh`; the build-list numbers from the plan Log entry of the EIC build-list merge (commit b2ae530).

## 2026-11-11 Agent call (draft): the pilot budget
This entry is a draft until the pilot collect work finalizes it. The pilot on Fir trained 40000 windows per rung (equal windows, not matched FLOPs). Training FLOPs per window are: rung 1 9.77e10, rung 2 2.11e9, rung 3 2.54e9, rung 4 1.66e9, rung 5 2.25e10, rung 6 2.70e9. Matched FLOPs anchored at rung 4 x 40000 windows is 1.66e9 x 40000 = 6.6e13, which gives rung 1 only 6.6e13 / 9.77e10 = 680 windows. That is too few: rung 1's validation loss was still 1.34 at 2000 windows. The draft rule is: equal windows (40000) within a rung, and matched compute F of about 2.4e15 for cross-rung comparisons only. Validation loss, first value to mean of the last 3: rung 1 1.34 to 1.02 (seeds 1.026, 1.019, 1.021), rung 2 1.128 to 0.984, rung 3 1.134 to 0.920, rung 4 1.197 to 0.914, rung 5 1.129 to 0.971 (at 32000 windows), rung 6 1.124 to 1.067. Rung 1's validation-loss seed spread was 0.027 at the last point and 0.007 on the mean of the last 3. The Log gives no arithmetic for F = 2.4e15; that figure is quoted as it stands. Evidence: the plan Log entries of the pilot submit merge (commit 2e48606), the rung-consistency merge and the pilot collect interim; `cruxvault/results/t181/SUBMITTED.md`.

## 2026-11-11 PI ruling 16: restricted EIC experiments use ENCODE's redacted alignments
The PI answered "yes to all six, go with your recommendations". The question: 46 EIC experiments, i.e. all tracks of the Roadmap cells C04, C09, C10, C13, C29 and C36 (6 validation, 31 training, 9 controls), serve no public FASTQ (129 files return HTTP 403), so decisions D5 and D8 cannot be met for them from reads. The approved answer: they use ENCODE's redacted unfiltered GRCh38 alignments (read names and MAPQ kept), with alignment-level arms only, and the pipeline is recorded as a covariate.

## 2026-11-11 PI ruling 17: the hard gate of D13 is the wiring check only
The PI answered "yes to all six, go with your recommendations". The question: is the hard gate of D13 the guardrail as the PI's document defines it, with the same-view copy gain as a diagnostic? The approved answer: the hard gate is the wiring check only (perturbing a masked bin's input must not change its output). The visible-versus-hidden copy gain is a reported diagnostic and does not gate. The code already had this (`GATE_ON_COPY_GAIN = False`).

## 2026-11-11 PI ruling 18: clip goal terms inside the composite
The PI answered "yes to all six, go with your recommendations". The question: does D13's "0-1 gap closed scale" mean clipping each goal term to [0, 1] in the composite? The approved answer: each goal term is clipped to [0, 1] inside the composite; the per-goal veto is applied on raw values; the comparison escalates when a denominator is near zero. The code now clips by default (`CLIP_GOAL_TERMS = True`), and every settlement row and the selection table show the raw and the clipped terms.

## 2026-11-11 Agent call: which values the clipped rule uses where
PI ruling 18 leaves open which values some steps use. The selection rule settles them as follows. (1) The composite's seed spread, the bar of D13 condition 1, is the spread of the clipped composite, because it measures the noise of the quantity being compared. (2) The per-goal veto and its seed spreads use the raw terms, as ruled. (3) Done-rule condition 2 compares the final design with the baselines goal by goal on raw terms, because a goal term above 1 or below 0 against a baseline is information, not noise; the clipped values are printed beside them. (4) With clipping on, the raw band [-1, 1.5] no longer escalates; a denominator under its tolerance still escalates. Evidence: `tools/onion_eval/select_rule.py` and `tests/test_onion_eval_select_rule.py` (the ruling-18 tests).

## 2026-11-11 PI ruling 19: DNase-seq signal for EIC scoring uses MACS2 without control
The PI answered "yes to all six, go with your recommendations". The question: decision D9b's own-pipeline route for DNase-seq (hotspot2 density) is in other units than the EIC DNase-seq truth, which is a MACS2 -log10 p track (reproduced at median r = 0.9851 on 34 training tracks, `plan/BENCHMARK_DESIGN.md` section 10). The approved answer: EIC signal scoring uses the MACS2 no-control route for DNase-seq, and the hotspot2 density stays available as the native DNase-seq signal view. The signal adapter's default is now `macs2_no_control`.

## 2026-11-11 PI ruling 20: signal from the predicted NB
The PI answered "yes to all six, go with your recommendations". The question: should the signal derived from the predicted NB be the expectation of the MACS2 score under the NB, or the score of the NB mean (the plug-in)? The approved answer: the expectation is the headline, and the plug-in is reported beside it. The scorer writes the plug-in under keys with the suffix `_plug_in`.

## 2026-11-11 PI ruling 21: the final EIC-comparison model trains on training plus validation
The PI answered "yes to all six, go with your recommendations". The question: should the final EIC-comparison model that predicts the blind tracks train on the training tracks only, or on training plus validation tracks? The approved answer: it trains on training plus validation tracks, as challenge entrants did, with the design and budgets frozen on validation first. The training-only score is reported beside it.

## 2026-11-11 Compute and storage used so far (decision D3)
This account is partial. It sums only the figures that the plan Log or the result files state; jobs without a stated CPU-h are listed but not counted. CPU-h stated: the cluster probe about 0.6 (`.orchestrate/cluster_evidence.md`), the EpiATLAS and ChIP-Atlas catalogs 0.14 (Nibi job 26815002), the ENCODE catalog and census about 5.9, and the DNase-seq pipeline setup about 7.8 so far: 0.6 + 0.14 + 5.9 + 7.8 = 14.4 CPU-h. GPU: rung 1 of the pilot ran 3 seeds on a Fir GPU slice; the Log says "DONE in ~32 min" and does not say whether that is per seed or in all. Not counted (no CPU-h stated): the region strata build (Nibi job 26813690, 6 min 59 s wall), the pilot training on Fir, and the pilot scoring jobs. Still running at the last Log line that names them (the queue was not checked for this entry): the EIC board reproduction, Nibi 26816540, 26818325, 26818330, 26818332 (each 56 CPU, 224 GB, 10 h requested, so at most 4 x 56 x 10 = 2240 CPU-h); the ChIP-seq and ATAC-seq smokes, Nibi 26816043 and 26817587; the DNase-seq smoke, Nibi 26820304; the pilot's rung 1 CPU twin, Fir 3432010; and the pilot scoring jobs, Fir 3438391, 3438393, 3438395, 3439926, 3438399, 3438400, 3438401, 3439674, 3438402, 3438404, 3438406. Storage stated: Nibi `/project` holds the genome layer (843M), the kit (34M), the hg38 pipeline references (8.5 GB), the DNase-seq pipeline (6.1 GB) and the strata masks (44 MB); Nibi `/scratch` holds the EIC truth copy (53 GB), left there with Max's copy as the backup; Fir holds its kit (37M) and the pilot tables (size not stated).

## 2026-11-11 Agent call: the read download stays on Nibi, with more parallel connections

The download route for the EIC reads stays on Nibi. The staging step uses about 256 connections in total: 4 files at once, with 64 byte ranges each.

- The measured rate from Nibi to ENCODE's storage is about 0.14 MB/s per connection. The rate rose linearly up to 64 connections (8.6 to 9.3 MB/s) and showed no cap.
- If the linear scaling holds, 256 connections give about 36 MB/s. The first priority of the build list (291.4 GB) then takes about 2.2 hours, and the whole list (2791.6 GB) takes about 22 hours.
- The scaling beyond 64 connections is an assumption. The first staging jobs measure the total rate.
- The fallback, if the scaling fails: download on Fir (129.5 MB/s measured with 4 files and 16 ranges each) and move the reads to Nibi with Globus. That route needs one Globus action from the PI, because no authenticated route exists.
- Reason for staying on Nibi: the pipelines, the references and the processing are on Nibi, Fir's environment setup failed on 12 nodes in two days, and the fallback needs the PI.

Evidence: the download-route measurement entry above (probe jobs on Nibi 26823539 and 26824325, on Fir 3440471, 3441260 and 3442621).
