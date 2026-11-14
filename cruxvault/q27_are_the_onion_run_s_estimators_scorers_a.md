---
id: q27
type: question
schema: 3
title: Are the onion run's estimators, scorers and training pipeline correct, fast and reproducible on the clusters?
parent: root
created: "2026-11-14T02:17:33"
updated: "2026-11-14T02:17:33"
---

# q27 — Are the onion run's estimators, scorers and training pipeline correct, fast and reproducible on the clusters?

Parent:: [[candii]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Methods and infrastructure of the onion run (crux task t152): the data path of training, the speed and numerical stability of the scorer, the checks of the variance estimators and of the EIC scorer, and the frozen DNA embeddings. Operational facts of the run that bear on trust in its numbers are recorded here.

## Bears on

Every miniCANDI number of the onion run is only as good as the tools that produce it; a defect or a slow step here changes what can be measured and how fast the feedback loop of architecture screening runs.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**0 children** · open 0, leaning 0, settled 0, parked 0 · 7 finding(s) on this question

- `f39` finding (moderate) [[f39_the_alphagenome_2_side_is_sound_our_ag2_|The AlphaGenome 2 side is sound. Our AG2 scorer reproduces Max's score_f1.py…]]
- `f40` finding (moderate) [[f40_our_eic_scorer_reproduces_the_published_|Our EIC scorer reproduces the published board against the corrected truth:…]]
- `f41` finding (moderate) [[f41_training_data_path_every_hdf5_open_on_ni|Training data path: every HDF5 open on Nibi's NFS /scratch takes about 0.21…]]
- `f42` finding (moderate) [[f42_scorer_the_old_scorer_s_cell_type_specif|Scorer: the old scorer's cell-type-specific correlation (r_spec) depended on…]]
- `f43` finding (moderate) [[f43_variance_estimators_a_simulation_audit_o|Variance estimators: a simulation audit of the layer estimators found 10…]]
- `f44` finding (moderate) [[f44_frozen_dna_embeddings_exist_genome_wide_|Frozen DNA embeddings exist genome-wide for both sources and verify (23…]]
- `f55` finding (moderate) [[f55_scorer_validity_on_the_training_free_bas|Scorer validity on the training-free baselines: the scorer (version 1,…]]
<!-- crux:ledger:end -->
