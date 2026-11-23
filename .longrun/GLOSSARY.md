# Glossary

Every term and metric in use. A new entry lands before its first use in anything the PI reads.

| Term | Gloss (10 words or fewer) | Origin (PI doc, PI, agent) | Row or decision |
|---|---|---|---|
| Pareto dominance | Best-or-tied on every goal term, strictly better on one | agent, PI chose | D1 |
| imputation term | variance recovered on held-out tracks (`imputation_raw`), 256 regions | onion close-out | D1 |
| correction term | gap closed on crop pairs vs oracle reference (`correction_v3_raw`) | onion close-out | D1, D3 |
| calibration term | uncertainty score of the predicted NB (`uncertainty_raw`) | onion close-out | D1 |
| seed range | max minus min over 3 seeds of a metric | onion close-out | D1 |
| conditioned skips | U-Net skip tensors modulated by FiLM from (C, C') | agent | D4 |
| pre/post CAC layers | covariate-only g sets count map f before and after network | PI | D5 |
| monotone curve f | invertible monotone map on log counts, knots from g | CAC testbed rung B | D6 |
| conditioned CNN f | small conv count map, parameters from g | CAC testbed rung D | D6 |
| FiLM | per-channel scale and shift computed from covariates | onion close-out | D4 |
| two-step CAC | (X,C) -> base view -> (X',C'), shared weights | onion close-out | D8 |
| leave-control-out evaluation | hide observed control, score the predicted control and its p-value | agent, PI chose | D9 |
| pair regions | 16 genome regions where correction pairs are scored | onion close-out | D3 |
| read-disjoint training pairs | input and target drawn from separate read halves | agent proposal, status proposed | D29, D35 |
| halves | short arm name for read-disjoint training pairs | agent, status proposed | D29 |
| budget checkpoint | spend level where spend is reported; work continues | PI | D20 |
| Avocado | published tensor-factorization imputation method; EIC comparison baseline | PI doc | D10, D33 |
| conv+attn | convolutional model with attention layers (rung 3) | onion close-out | D2 |
| ssh | remote shell connection to a cluster | standard tool | none |
| rsync | file copy tool that sends only changed files | standard tool | none |
| bigWig | indexed binary file format for genome signal tracks | standard format, status proposed | D33 |
| mix_avg | adds cross-cell-type average signal as an extra input | agent (arm label, mix_arms.py average_input); proposed | D39 |
| mix_pearson | inputs fed as NB Pearson residuals, not log counts | agent (arm label, count_transform pearson); proposed | D39 |
| headskip_film | covariates rescale the input features skipped to the output head | agent (arm label, unet.py head_skip film); proposed | D39 |
| average-signal input | same as mix_avg: cross-cell-type average track as input | agent; proposed | D39 |
| Pearson-residual input | same as mix_pearson: (count minus NB mean) over NB spread | agent; proposed | D39 |
| conditioned head skip | same as headskip_film | agent; proposed | D39 |
| GPU | graphics processor used to train neural networks fast | agent; proposed | R9 |
| H100 | NVIDIA's current data-centre graphics processor model on Fir | agent; proposed | R9 |
| shard | one of 8 equal slices of the evaluation set, predicted in parallel | agent; proposed | D39 |
| encoder | network half that downsamples the input into compact features | agent; proposed | R3 |
| decoder | network half that upsamples features back to 25 bp outputs | agent; proposed | R3 |
| bottleneck | lowest-resolution layer between the two network halves | agent; proposed | R3 |
| transformer | attention layers relating every position in the window | agent; proposed | R3 |
| pr1 / pr2 | the two pseudoreplicates: disjoint random halves of one experiment's reads | agent (ENCODE convention); proposed | D29 |
| config | training settings file read at launch | agent; proposed | D29 |
| artifact | signal distortion caused by protocol, not biology | agent; proposed | R4 |
| coarser | at lower resolution, i.e. larger bins | agent; proposed | R3 |
| widen | make a margin larger | agent; proposed | D1 |
| computed | calculated | agent; proposed | R6 |
| agnostic | independent of, not specific to | agent; proposed | D39 |
| calibrated | predicted uncertainty matches observed error | agent; proposed | R2 |
| caveats | limits on how a result may be read | agent; proposed | D39 |
| inference | running a trained model to make predictions | agent; proposed | R9 |
| parentheses | the bracket marks ( ) | agent; proposed | none |
| pr1 | first pseudoreplicate: random half of one experiment's reads | agent (ENCODE convention); proposed | D29 |
| pr2 | second pseudoreplicate: the other, disjoint half of the reads | agent (ENCODE convention); proposed | D29 |
| antibody | reagent that pulls down the ChIP target protein | agent; proposed | D5 |
| crop36 | view with reads trimmed to 36 bp | agent (view name in tables); proposed | D3 |
| decoded | turned from features into output values | agent; proposed | R2 |
| dedup | removal of PCR-duplicate reads | agent; proposed | D5 |
| downsample | reduce resolution by pooling neighbouring bins | agent; proposed | R3 |
| upsamples | raises resolution back toward 25 bp bins | agent; proposed | R3 |
| embedding | learned vector standing for a category or input | agent; proposed | R3 |
| log2 | base-2 logarithm | agent; proposed | none |
| MAPQ | read mapping-quality score; threshold filters ambiguous alignments | agent; proposed | D5 |
| MLP | small stack of fully connected layers | agent; proposed | R3 |
| GELU | smooth nonlinear activation, close to ReLU | agent; proposed | R3 |
| stem | first layer of a tower; mixes all tracks per bin | agent; proposed | R3 |
| size factor | depth scale s = 2^(log2_depth - 24) vs 16.8 M reads | agent; proposed | R3 |
| pseudocount | constant added to counts so log of zero is defined | agent; proposed | R3 |
| 1x1 convolution | per-position linear map across channels, no neighbour mixing | agent; proposed | R3 |
| broadcast | reuse one value across a tensor axis without copying | agent; proposed | R3 |
| append | add to the end | agent; proposed | R3 |
| arithmetically | by a fixed formula, with no learned weights | agent; proposed | R3 |
| bool | true/false data type | agent; proposed | R3 |
| D_TRACK | per-track feature width in the U-Net code (8) | agent; proposed | R3 |
| flattens | merges two array axes into one | agent; proposed | R3 |
| initialization | starting weight values before training | agent; proposed | R3 |
| leakage | target information reaching the model's input by mistake | agent; proposed | R3 |
| log1p | log(1 + x), natural logarithm | agent; proposed | R3 |
| log2_depth | covariate column: base-2 log of the experiment's read count | agent; proposed | R3 |
| normalized | put on a common scale | agent; proposed | R3 |
| padded | extended with zeros to a required length | agent; proposed | R3 |
| tensor | multi-dimensional array of numbers | agent; proposed | R3 |
| transposed | with two axes swapped | agent; proposed | R3 |
| vis | code name of the visible-bin flag: real count present | agent; proposed | R3 |
| footer | small text line at the bottom of a figure | agent; proposed | R3 |
| noisier | with a wider spread across seeds | agent; proposed | R3 |
| convolutional | using sliding learned filters over neighbouring bins | agent; proposed | R3 |
| eic_v1 | earlier data version of the EIC tables, before eic_v1_2 | agent; proposed | R3 |
| measurable | large enough to detect above seed noise | agent; proposed | R3 |
| axial attention | attention along one axis at a time: position, then track | agent; proposed | R3 |
| Perceiver IO | attention model through a small set of learned latents | agent; proposed | R3 |
| position average | baseline predicting each bin from other cell types' mean | agent; proposed | R3 |
| D1 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D18 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D20 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D31 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D39 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D41 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| D42 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| bj140el0j | id of a laptop watch job (agent-facing) | agent; proposed | none |
| eic_val | EIC validation-track prediction and scoring step (agent-facing) | agent; proposed | R9 |
| halves_headskip_off | arm: read-disjoint pairs U-Net with the head skip removed | agent; proposed | R3 |
| headskip_off | arm: U-Net with the head skip removed | agent; proposed | R3 |
| Nibi | second Alliance Canada cluster, used for training overflow | agent; proposed | none |
| p16 | scoring with correction on 16 pair regions (screen) | agent; proposed | R1 |
| p256 | scoring with correction on 256 pair regions (confirm) | agent; proposed | R1 |
| params | model parameters, i.e. learned weights | agent; proposed | R22 |
| pareto_select | script applying the Pareto win rule to arm summaries | agent; proposed | R1 |
| RCP6 | code snapshot id pushed to Fir (agent-facing) | agent; proposed | none |
| t370 | crux task id of this recipe run (agent-facing) | agent; proposed | none |
| worktree | separate checkout of one git branch in its own folder | agent; proposed | none |
| denoisers | models that remove noise from an observed signal | agent; proposed | R3 |
| dispersion | NB spread parameter: how far counts vary around the mean | agent; proposed | R2 |
| dropout | randomly zeroing a component during training | agent; proposed | R3 |
| post hoc | applied after training, with no retraining | agent; proposed | R2 |
| shortcut | easy solution that lowers loss without learning the task | agent; proposed | R3 |
| b904xdk65 | id of a laptop watch job (agent-facing) | agent; proposed | none |
| bx8rkzhqf | id of a laptop watch job (agent-facing) | agent; proposed | none |
| crux | research-notebook tool that keeps the vault (agent-facing) | agent; proposed | none |
| D43 | row id of a decision in DECISIONS.md (agent-facing) | agent; proposed | none |
| lint | automatic check of a file against style rules | agent; proposed | none |
| prezit | slide-deck builder skill used for lab decks | agent; proposed | none |
| python3 | the Python interpreter command | agent; proposed | none |
| calibrate | adjust predicted uncertainty to match observed error | agent; proposed | R2 |
| glossary | list of terms with short definitions | agent; proposed | none |
| imputes | predicts a missing track from the other inputs | agent; proposed | R2 |
| schematic | simplified labelled diagram of a design | agent; proposed | none |
| schematics | simplified labelled diagrams of a design | agent; proposed | none |
| enhancer | DNA region that raises a distant gene's transcription | genomics | proposed | 2026-11-22 EIC slide |
| MSE | mean squared error: average squared prediction difference | statistics | proposed | 2026-11-22 EIC slide |
| clipped | values beyond a bound are set to that bound | agent; proposed | 2026-11-22 Pearson residual |
| deviations | distances of observed values from their expected values | agent; proposed | 2026-11-22 Pearson residual |
| emptied | all stored files removed after use | agent; proposed | 2026-11-22 prediction store |
| G5 | pre-set pass rule for held-out control prediction (plan C21) | agent; proposed | 2026-11-22 |
| ledger | GOALS.md table: one row per PI requirement | longrun skill; proposed | 2026-11-22 |
| P0 | top priority tier: the run fails without it | longrun skill; proposed | 2026-11-22 |
| Poisson | count distribution whose variance equals its mean | agent; proposed | 2026-11-22 count-score discussion |
| deviance | twice the log-likelihood gap to a perfect fit | agent; proposed | 2026-11-22 count-score discussion |
| binomial | count of successes in n independent yes/no trials | agent; proposed | 2026-11-22 count-score discussion |
| thinning | keep each read independently with fixed probability | agent; proposed | 2026-11-22 count-score discussion |
| CRPS | squared distance between predicted and step CDF, summed | agent; proposed | 2026-11-22 count-score discussion |
| CRPS_depth-only | that score for counts scaled by target-over-source depth | agent; proposed | 2026-11-22 count-score discussion |
| CRPS_floor | that score of a perfect predictor (oracle floor) | agent; proposed | 2026-11-22 count-score discussion |
| E_sp | specific error: error on the cell-type-specific signal part | agent; proposed | 2026-11-22 count-score discussion |
| r_spec | correlation after removing a smooth function of the average | agent; proposed | 2026-11-22 count-score discussion |
| deviation (PIT) | mean gap between average predictive CDF and uniform | agent; proposed | 2026-11-22 count-score discussion |
| invariant | unchanged when a quantity, e.g. depth, changes | agent; proposed | 2026-11-22 count-score discussion |
| additive | parts sum exactly to the total; also "additivity" | agent; proposed | 2026-11-22 metric-family table |
| additivity | property that component errors sum to the total error | agent; proposed | 2026-11-22 metric-family table |
| AlphaGenome | DeepMind sequence-to-function genomics model (AG2 = version 2) | agent; proposed | 2026-11-22 metric-family table |
| invariance | score stays the same when depth changes | agent; proposed | 2026-11-22 metric-family table |
| outlier | a bin with a value far from the rest; plural "outliers" | agent; proposed | 2026-11-22 metric-family table |
| outliers | bins with values far from the rest | agent; proposed | 2026-11-22 metric-family table |
| quadrature | combine as square root of summed squares | agent; proposed | 2026-11-22 metric-family table |
| penalizes | raises the error score for | agent; proposed | 2026-11-22 metric-family table |
| predictor's | belonging to a model or baseline that outputs predictions | agent; proposed | 2026-11-22 metric-family table |
| unifies | puts two metric families on one scale | agent; proposed | 2026-11-22 metric-family table |
| comparability | ability to set our numbers beside published or earlier ones | agent; proposed | 2026-11-22 six-metric proposal |
| D9 | long-horizon decision: EIC headline uses the challenge's own metrics | long-horizon decisions.md; proposed | 2026-11-22 six-metric proposal |
| D_baseline | normalized Poisson deviance of the baseline prediction | agent; proposed | 2026-11-22 six-metric proposal |
| D_floor | normalized Poisson deviance a perfect predictor gets from read noise | agent; proposed | 2026-11-22 six-metric proposal |
| D_model | normalized Poisson deviance of the model's predicted mean | agent; proposed | 2026-11-22 six-metric proposal |
| deduplicated | with repeated copies of one sequenced fragment removed | agent; proposed | 2026-11-22 six-metric proposal |
| PCR | polymerase chain reaction; amplification that copies DNA fragments | agent; proposed | 2026-11-22 six-metric proposal |
| stratum | one subgroup of bins, e.g. by assay or count level | agent; proposed | 2026-11-22 six-metric proposal |
| testbed | small fixed training-and-scoring setup for comparing miniCANDI arms | agent; proposed | 2026-11-22 six-metric proposal |
| deployable | the version meant for real use, not an evaluation device | agent; proposed | 2026-11-22 control arms |
| RCP8 | code snapshot id pushed to Fir (agent-facing) | agent; proposed | 2026-11-22 |
| rcp_halves_mix_pearson_unet | run name of the chosen recipe (agent-facing) | agent; proposed | 2026-11-22 |
| local_2eb1d23b | id prefix of a linked Claude session (agent-facing) | app; proposed | 2026-11-22 |
| b735 | fragment of a linked session id (agent-facing) | app; proposed | 2026-11-22 |
| baf | fragment of a linked session id (agent-facing) | app; proposed | 2026-11-22 |
| prefix | fixed text at the start of a name | agent; proposed | 2026-11-22 job names |
| genomic | of the DNA sequence and its positions | agent; proposed | 2026-11-22 calibration regions |
| reproducible | can be rebuilt exactly from saved inputs | agent; proposed | 2026-11-22 prediction stores |
| caveat | a limit on how far a result can be trusted | agent; proposed | 2026-11-23 reply |
| D44 | decision row: train and confirm the full combination | agent (DECISIONS.md); proposed | 2026-11-23 reply |
| D51 | decision row: final recipe keeps control weight 1 | agent (DECISIONS.md); proposed | 2026-11-23 reply |
| D52 | decision row: dispersion recalibration ships with the final model | agent (DECISIONS.md); proposed | 2026-11-23 reply |
| p2x | recipe arm with twice the parameter budget (2 M) | agent (recipe_arms.py); proposed | 2026-11-23 reply |
| p4x | recipe arm with four times the parameter budget (4 M) | agent (recipe_arms.py); proposed | 2026-11-23 reply |
| timeout | job stopped by the cluster at its time limit | agent; proposed | 2026-11-23 reply |
| deletion | removal of files from disk | agent; proposed | 2026-11-23 reply |
| ENCODE | public consortium database of functional genomics experiments | field term; proposed | 2026-11-23 reply |
| MOESM2 | EIC supplementary truth table, original uncorrected tracks | EIC paper supplement; proposed | 2026-11-23 reply |
| t184 | task id: build the blind-experiment input rows | crux taskhub; proposed | 2026-11-23 reply |
| FASTQ | raw sequencing-read file format with base qualities | field term; proposed | 2026-11-23 reply |
| GiB | gibibyte, 2^30 bytes | unit; proposed | 2026-11-23 reply |
| TiB | tebibyte, 2^40 bytes | unit; proposed | 2026-11-23 reply |
| Globus | cluster-to-cluster file transfer service used by Alliance Canada | tool; proposed | 2026-11-23 reply |
| hg38 | human reference genome assembly GRCh38 | field term; proposed | 2026-11-23 reply |
| outage | period when a cluster was down and jobs died | agent; proposed | 2026-11-23 reply |
| nearline | Alliance tape-backed archive storage, slow to recall | cluster term; proposed | 2026-11-23 reply |
| ATAC | ATAC-seq: assay for open chromatin by transposase insertion | field term; proposed | 2026-11-23 reply |
| ids | identifiers, e.g. cluster job numbers | agent; proposed | 2026-11-23 reply |
| r_a | per-assay negative-binomial dispersion fit on training data | components.py:394; proposed | 2026-11-23 reply |
| tightest | smallest range across seeds | agent; proposed | 2026-11-23 reply |
| local_b5c55e32 | id prefix of the foreman Claude session (agent-facing) | session id; proposed | 2026-11-23 reply |
| b4b7 | fragment of a linked session id (agent-facing) | session id; proposed | 2026-11-23 reply |
| edc | fragment of a linked session id (agent-facing) | session id; proposed | 2026-11-23 reply |
| froze | stopped pushing changes to given files for a time | agent; proposed | 2026-11-23 reply |
