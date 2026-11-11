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
