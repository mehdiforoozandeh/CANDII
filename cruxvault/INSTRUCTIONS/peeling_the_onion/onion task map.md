# Onion task map

Every chunk of the peeling-the-onion plan maps to one crux child task of t152. `results/<T:X>/` in the plan means `cruxvault/results/<task id of chunk X>/`. A submit chunk and its collect chunks share one task (for example L1a and L1b), so a task id can repeat in the table. Chunk V1 created every task from V1 on; builders never run `crux task add` or `crux finding`, and the foreman records findings and hypotheses.

| chunk id | task id | title |
|---|---|---|
| V0 | t153 | Crux tasks, notepad, package files, golden reference |
| K0 | t154 | Kit tooling, cluster probes, cluster choice |
| GT | t155 | Globus transfer of the testbed scratch tree, Nibi to Fir |
| D1 | t156 | ENCODE catalog and replicate-ladder census |
| D2 | t157 | EpiATLAS and ChIP-Atlas catalogs |
| D3 | t158 | EIC accession ledger and check script |
| E1 | t159 | Headline metric library |
| E2 | t160 | Onion layers along the replicate ladder |
| E3 | t161 | Region strata masks |
| E4 | t162 | Training-free baselines and copy-leak guardrail |
| E5 | t163 | Development evals: error budget in counts |
| M1 | t164 | miniCANDI contract, dataset and shared components |
| P1 | t165 | Pilot tables over the existing EIC store |
| X1 | t166 | EIC scoring route against the corrected truth |
| D5 | t167 | Alignment-level view kit |
| D7 | t168 | FASTQ-level uniform kit for histone ChIP-seq and ATAC-seq |
| D7d | t169 | DNase-seq pipeline setup and kit |
| D4 | t170 | EIC build list and withheld design |
| DT | t171 | Import the counterfactual testbed as views |
| E6 | t172 | Eval set builder, scorer and selection rule |
| T1 | t173 | Train loop, registry, rung 1 (pointwise set model) |
| R2 | t174 | Rung 2: convolutional autoencoder |
| R3 | t175 | Rung 3: conv plus attention autoencoder |
| R4 | t176 | Rung 4: U-Net with skip-source arms |
| R5 | t177 | Rung 5: axial transformer |
| R6 | t178 | Rung 6: Perceiver IO |
| X2 | t179 | Signal from predicted counts |
| D6a | t180 | EIC build, priorities 1 to 3 (submit) |
| P2a | t181 | Pilot: all six rungs on pilot_eic (submit) |
| P2b | t182 | Pilot: collect |
| D6b | t183 | Collect priorities 1 to 3, build and check eic_v1 |
| D6c | t184 | EIC build, priority 4 (submit) |
| E7a | t185 | Eval set, layers and baselines on eic_v1 (submit) |
| E7b | t186 | Collect: scorer validity on eic_v1 |
| V1 | t187 | Crux tasks and task map rows for every remaining chunk |
| A1 | t188 | Two-step correction, loss arms, model wrappers |
| A2 | t189 | Dataset arms |
| A3 | t190 | Component arms: covariate embedding and injection, DNA tower mode, transforms, head |
| A4 | t191 | ChIP control arms: the output-only lambda head |
| A5 | t192 | DNA source arms: precompute tool for frozen sequence-model embeddings |
| SC | t193 | Scorer and eval-set additions |
| SR | t194 | Selection rule: the blind state after a validation failure |
| RP | t195 | Arm report: headline table, development evals, accessibility-only table |
| RS | t196 | Resolution study tool |
| MM | t197 | Crossed mixed-model cross-check |
| D6t | t198 | Tables builder and dataset check |
| QCt | t199 | Read-derived QC fill tool |
| LK | t200 | Ladder kit: arm configs for all six rungs and the submit script |
| PG | t201 | Genome-wide prediction and EIC export |
| CT | t202 | Compute and storage tally tool |
| EVt | t203 | Proxy scorer for the evolution step |
| FZ | t204 | Which EIC experiments are among the Borzoi and Flashzoi training targets |
| RX | t205 | Rung support for the stage-2 arm keys |
| XE | t206 | External EIC entries in signal space |
| A5x | t207 | Frozen DNA embeddings: environment, smoke and genome-wide jobs (submit) |
| A5c | t207 | Frozen DNA embeddings: collect |
| S2K | t208 | Stage-2 arm configs and the all-arms integration test |
| AS | t209 | Arm smoke on the pilot tables (Fir) |
| D6f | t183 | Build, check and freeze the eic_v1 tables |
| X1c | t166 | Collect the EIC board reproduction |
| D6d | t210 | Collect priority 4: verify and resubmit |
| D6g | t210 | Build and check eic_v2 |
| QCa | t211 | Fill the read-derived QC side table (submit) |
| QCb | t211 | Fill the read-derived QC side table (collect) |
| E8a | t212 | Blind eval sets (built, not scored); onion layers, resolution study and mixed model on training and validation tracks (submit) |
| E8b | t212 | Collect: the onion layers report (training and validation tracks) |
| E8c | t212 | Onion layers, resolution study and mixed model with the blind tracks |
| L1a | t213 | Rung 1: base arm and per-rung ablations (submit) |
| L1b | t213 | Rung 1: collect and settle |
| L2a | t214 | Rung 2: base arm and per-rung ablations (submit) |
| L2b | t214 | Rung 2: collect and settle |
| L3a | t215 | Rung 3: base arm and per-rung ablations (submit) |
| L3b | t215 | Rung 3: collect and settle |
| L4a | t216 | Rung 4: base arm and per-rung ablations (submit) |
| L4b | t216 | Rung 4: collect and settle |
| L5a | t217 | Rung 5: base arm and per-rung ablations (submit) |
| L5b | t217 | Rung 5: collect and settle |
| L6a | t218 | Rung 6: base arm and per-rung ablations (submit) |
| L6b | t218 | Rung 6: collect and settle |
| LCa | t219 | Combined configurations within a rung, and second scorings for one device type (submit) |
| LC | t219 | Cross-rung comparison and the selected rung |
| MXp | t220 | Mix-and-match combinations: the proposal |
| MXh | t220 | Mix-and-match: build the combinations (one builder per hybrid, in parallel) |
| MXa | t220 | Mix-and-match: train and score (submit) |
| MXb | t220 | Mix-and-match: settle and name the winning recipes |
| EVa | t221 | Evolution of the winning recipes: set up and start the autopilot runs |
| EVb | t221 | Evolution: collect the runs and submit the confirmations |
| EVc | t221 | Evolution: confirm under D13 and name the design |
| S2a | t222 | Stage 2 on the design (submit) |
| S2b | t222 | Stage 2: collect and settle |
| FDa | t223 | Final design: the design plus the stage-2 winners (submit) |
| FDb | t223 | Final design: settle, freeze, and the done rule on validation |
| FEa | t224 | The EIC model: train on training plus validation tracks (submit) |
| BPa | t225 | The single blind pass (submit) |
| BPb | t225 | The done rule on the blind tracks |
| XCb | t226 | EIC board and the comparison with external entries in signal space |
| G5x | t227 | The gate before the extended-dataset step |
| CTf | t228 | Final compute and storage tally |
| RD | t229 | The /prezit deck |
| RSy | t230 | The 12-page /storify report |
| VC | t231 | Vault close-out |
| TV1 | t232 | The total-variance note |
| TV2 | t233 | The audit of the estimators against the identity |
| TVf | t234 | Fix chunk for real defects of the audit (conditional) |
| MS | t235 | Read Max's result files |
| BRt | t236 | Retrieval baseline |
| XRd | t237 | RNA-seq input tool |
| XM | t238 | Extra-input model components |
| XD | t239 | Extra-input readers and the reverse-complement transform |
| RC0 | t240 | Probe: are the read sets there, and what does a recount cost |
| RCt | t241 | Recount tool at 1 bp with the point rule |
| RVt | t242 | Variance-by-scale tool |
| LS | t243 | Loss arms |
| XI | t244 | Integration of the extra inputs in the dataset, the contract and the six rungs |
| XRf | t245 | Fetch and bin the RNA-seq inputs (Nibi) |
| S3K | t246 | Stage-3 arm configs |
| RM | t247 | Model code for another bin size |
| AS3 | t248 | Smoke of the Amendment 3 arms on the pilot tables (Fir) |
| RCa | t249 | Recount the subset at 1 bp (submit and collect) |
| RVa | t250 | Variance lost below 25 bp (run and report) |
| BRa | t251 | Retrieval baseline on eic_v1 |
| YS | t252 | Run-to-run yardstick |
| PW | t253 | Does the NB head down-weight peaks |
| RMa | t254 | Resolution arms (submit) |
| RMb | t254 | Resolution arms (collect and report) |
| XIm | t255 | Extra inputs in a mix or evolved design (conditional) |
| S3a | t256 | Stage 3 on the design (submit) |
| S3b | t256 | Stage 3: collect and settle |
| OXs | t257 | Select the second experiments |
| OXd | t258 | Dataset guard for evaluation-only second experiments |
| OXe | t259 | Eval set add-on and add-on scoring |
| OXl | t260 | Layers by pair class |
| OXr | t261 | Arm report: metric 7 per rung with two bounds |
| OXb | t262 | Build list of the second experiments, and kit KX |
| OXt | t263 | Tables add-on and the ledger exception |
| OXa | t264 | Build the second experiments (submit) |
| OXc | t264 | Collect the second experiments |
| OXx | t265 | Build and check the eic_v1x tables |
| OXy | t266 | Add-on eval set, the experiment layer and the baselines' two bounds |
| AG0 | t267 | The AG2 metadata tables and the remaining API facts |
| AGf | t268 | Fetch tool with a restartable cache and quota backoff |
| AGl | t269 | Leakage audit tool |
| AGs | t270 | Space maps: counts to AG2's space, recalibration, peak probability |
| AGk | t271 | Peaks: ENCODE peak files, tiles, consensus AUPRC |
| AGb | t272 | Signal-space baselines of Max: training mean and closest-5 oracle |
| AGa | t273 | Run the leakage audit on the EIC ledger |
| AGc | t274 | The same-track scorer and the ag2_board table |
| AGd | t275 | Dump hook: method track files from the scorer |
| AGm | t276 | Reproduction gate against Max's numbers |
| AGx | t277 | Fetch AG2 for the EIC cells on the eval regions |
| AGva | t278 | AG2 against the baselines on the validation tracks (submit) |
| AGvb | t279 | AG2 against the baselines on the validation tracks (collect) |
| AGwa | t280 | miniCANDI against AG2 on the validation tracks (submit) |
| AGwb | t281 | miniCANDI against AG2 on the validation tracks (collect) |
| AGz | t282 | AG2 on the blind tracks, inside the single blind pass |
| ERg | t283 | The eval regions, written before the tables exist |
| AGxf | t284 | AlphaGenome 2 fetch on the regions file |
| D6u | t285 | Update the dataset from a watchdog report |
| TR | t286 | Training regions, exclusion and per-window loss weights |
| ERx | t287 | Evaluation set from an explicit regions file |
| STt | t288 | Strata and region-list tool, from training data only |
| RBk | t289 | Configs of the region experiments |
| SAx | t290 | submit_arms.sh: extra scoring arguments and a nice value |
| RB1a | t291 | Budget curve on rung 1: submit the trainings |
| STa | t292 | Strata tables and region lists: submit (Nibi) |
| STb | t293 | Strata tables and region lists: collect |
| PRg | t294 | Per-region output of the scorer and the region bootstrap |
| RTa | t295 | Territory, unseen-loci and selection arms: submit |
| ECa | t296 | Evaluation-region count: the 1,024-region set and its scorings (submit) |
| RB1s | t297 | Budget curve on rung 1: submit the scorings |
| RB1b | t298 | Budget curve on rung 1: report |
| RTb | t299 | Territory, unseen loci and selection: report |
| RTc | t300 | Confirmation of the best selection strategy at the full budget |
| ECb | t301 | Evaluation-region count: analysis |
| RB4a | t302 | Budget curve on rung 4: submit |
| RB4b | t303 | Budget curve on rung 4: report |
| RGs | t304 | The answers to the PI's region questions |
| I1a | t305 | The variance layers on real data |
| I1b | t306 | The baselines on the eic_v1 evaluation set (submit) |
| I1c | t307 | The baselines: collect, and the room for a model |
| I1d | t308 | AlphaGenome 2 against the baselines on the validation tracks |
| I1e | t309 | Methods: the dataset and the evaluation protocol |
| I1f | t310 | Deck 1 and the paper note of iteration 1 |
| I2a | t311 | The screening evaluation set and its baselines |
| I2b | t312 | Screening scorings of rung 1 and of the budget curve (submit) |
| I2c | t313 | Rung 1: the model results |
| I2d | t314 | The budget curve on rung 1 |
| I2e | t315 | AlphaGenome 2 against miniCANDI rung 1 on the fair tracks |
| I2f | t316 | Deck 2 and the paper note of iteration 2 |
