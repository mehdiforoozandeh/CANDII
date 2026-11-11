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
