# Onion task map

Each chunk of the peeling-the-onion plan (waves 0 to 6) is one crux child task of t152. `results/<T:X>/` in the plan means `cruxvault/results/<task id of chunk X>/`. Ladder tasks (wave 7 on) are created later by the chunk that submits them.

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
