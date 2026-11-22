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
