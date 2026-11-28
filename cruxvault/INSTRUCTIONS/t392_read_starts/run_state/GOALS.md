# Requirements ledger (t392, Phase 1)

Source: ../INSTRUCTIONS.md (cited as I:<section>) and ../HANDOFF_2026-11-23_read-start-minicandi.md (H:<section>).
Tiers are the agent's recommendation; the PI's word through the foreman overrides.

| id | tier | requirement (quoted or close) | source | status | evidence |
|---|---|---|---|---|---|
| R1 | P0 | Recount with a strand channel: ChIP stranded (+/- starts), DNase unstranded | I:2, I:3 item 1 | open | |
| R2 | P0 | Carry the strand channel through the store, the input contract (36 fixed tracks) and average_h5 | I:3 item 1 | open | |
| R3 | P0 | res_dataset accepts average_h5 | I:3 item 1 | open | |
| R4 | P0 | Conversion step: means by box convolution, NB by moment matching (Poisson when var<=mu), between predict and metrics; scorers unchanged | I:3 Fair scoring | open | |
| R5 | P0 | Tests first, against INSTRUCTIONS (crux-tests agent); no new failures vs base | I:3 item 1, H:DoD 1 | open | |
| R6 | P0 | Recall reads from Nibi nearline; recount all 2,740 eic_v1_2 measurements in every view (base, crop, SE, depth, MAPQ-0, no-dedup); build the read-start store | I:3 item 2 | open | |
| R7 | P0 | GATE: convolved observed read starts reproduce eic_v1_2 overlap counts, 0 differing bins, every measurement; logged | I:3 item 2, H:DoD 2 | open | |
| R8 | P0 | Train read-start model, config halves_mix_avg_pearson, seeds 0,1,2, same windows/budget | I:3 item 3 | open | |
| R9 | P0 | Reuse read-count reference results; never retrain | I:3 | open | |
| R10 | P0 | Score both on 16 regions: imputation, correction, calibration, with seed ranges | I:3 item 4 | open | |
| R11 | P1 | Deviance metrics (PR 65) on 16 regions, once PR 65 merges into exp/t370 | I:3 Metrics | blocked: PR 65 open | |
| R12 | P1 | Read-start model vs read-start truth, sanity check only | I:3 Fair scoring | open | |
| R13 | P0 | SIDE_BY_SIDE.md: one table, seed ranges, one figure, one-paragraph reading; runall.sh, MANIFEST.tsv, no file >1 MB | I:3 Deliverable, H:DoD 4 | open | |
| R14 | P1 | Crux finding on q75 citing SIDE_BY_SIDE.md | H:DoD 5 | open | |
| R15 | P0 | Budget: stop and report at >600 CPU-h (plan ~470) | I:3 Budget | open | CPU-h so far: 0 |
| R16 | P0 | Status table to foreman at each milestone; never address the PI | I:5 | ongoing | |
| R17 | P3 | Phase 2 transforms; 256-region scoring (PI's word only) | I:4 | do not build | |
