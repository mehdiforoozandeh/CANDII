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
