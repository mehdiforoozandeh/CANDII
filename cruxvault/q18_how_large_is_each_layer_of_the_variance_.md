---
id: q18
type: question
schema: 3
title: How large is each layer of the variance of a track on real data (sampling, technical, biological replicate, cell type), per assay?
parent: root
created: "2026-11-14T00:35:07"
updated: "2026-11-14T00:35:07"
---

# q18 — How large is each layer of the variance of a track on real data (sampling, technical, biological replicate, cell type), per assay?

Parent:: [[candii]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Iteration 1 of Amendment 9 of the onion plan (PI directive of 2026-11-14: iterative deployment); results go to INSTRUCTIONS/peeling_the_onion/onion paper.md.

## Bears on

Sets, per assay, how much variance imputation must explain (the cell-type layer) and the floor below which calibration cannot go (the sampling layer), before any model.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**3 children** · open 2, leaning 1, settled 0, parked 0 · 4 finding(s) on this question

- `h37` [[h37_on_real_data_the_cell_type_layer_is_the_|On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks]] — *leaning · moderate*
- `q24` _(Q)_ [[q24_how_much_of_each_variance_layer_does_the|How much of each variance layer does the model explain (the explained fraction per layer, beside the data layers of iteration 1)?]] — *open*
- `q31` _(Q)_ [[q31_how_do_the_variance_layer_shares_change_|How do the variance layer shares change at each assay's full depth and on a variance-stabilized scale, instead of log1p counts at pseudoreplicate-half depth?]] — *open*
- `f35` finding (weak) [[f35_partial_answer_on_eic_v1_no_model_involv|Partial answer on eic_v1 (no model involved): only two layers resolve, a…]]
- `f51` finding (moderate) [[f51_read_from_the_rerun_with_the_pseudorepli|Read from the rerun with the pseudoreplicate pairs added (Nibi job…]]
- `f52` finding (moderate) [[f52_read_from_layers_by_assay_tsv_of_the_rer|Read from layers_by_assay.tsv of the rerun (job 27265045); moderate because…]]
- `f53` finding (moderate) [[f53_read_from_layers_by_assay_tsv_and_layers|Read from layers_by_assay.tsv and layers_by_stratum.tsv of the rerun (job…]]
<!-- crux:ledger:end -->
