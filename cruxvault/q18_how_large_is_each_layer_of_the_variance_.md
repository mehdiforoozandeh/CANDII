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
**6 children** · open 2, leaning 4, settled 0, parked 0 · 6 finding(s) on this question

- `h37` [[h37_on_real_data_the_cell_type_layer_is_the_|On real data the cell-type layer is the largest layer for the narrow marks and for accessibility, and the sampling layer is the largest at 25 bp for the broad marks]] — *leaning · moderate*
- `q24` _(Q)_ [[q24_how_much_of_each_variance_layer_does_the|How much of each variance layer does the model explain (the explained fraction per layer, beside the data layers of iteration 1)?]] — *open*
- `q31` _(Q)_ [[q31_how_do_the_variance_layer_shares_change_|How do the variance layer shares change at each assay's full depth and on a variance-stabilized scale, instead of log1p counts at pseudoreplicate-half depth?]] — *leaning · moderate*
- `q33` _(Q)_ [[q33_can_the_negative_binomial_be_written_wit|Can the negative binomial be written with the depth-corrected eta (mean = depth x exp(eta)) throughout the model design, the variance decomposition and the law of total variance, and does decomposing the variance in eta space instead of on counts help, and how?]] — *leaning · moderate*
- `q60` _(Q)_ [[q60_how_much_of_the_1_bp_count_variance_of_a|How much of the 1 bp count variance of a track lies below the 25 bp bin of the model, and how much of that variance is reproducible signal and not read-sampling noise?]] — *leaning · moderate*
- `q61` _(Q)_ [[q61_does_duplicate_removal_which_makes_the_t|Does duplicate removal, which makes the two read halves of one library exclude each other, bias the read-half (sampling) layer of the earlier variance ladders at 25 bp?]] — *open*
- `f35` finding (weak) [[f35_partial_answer_on_eic_v1_no_model_involv|Partial answer on eic_v1 (no model involved): only two layers resolve, a…]]
- `f51` finding (moderate) [[f51_read_from_the_rerun_with_the_pseudorepli|Read from the rerun with the pseudoreplicate pairs added (Nibi job…]]
- `f52` finding (moderate) [[f52_read_from_layers_by_assay_tsv_of_the_rer|Read from layers_by_assay.tsv of the rerun (job 27265045); moderate because…]]
- `f53` finding (moderate) [[f53_read_from_layers_by_assay_tsv_and_layers|Read from layers_by_assay.tsv and layers_by_stratum.tsv of the rerun (job…]]
- `f64` finding (moderate) [[f64_measured_on_the_same_3_908_pairs_as_f51_|Measured on the same 3,908 pairs as f51 to f53 with intervals; moderate…]]
- `f77` finding (weak) [[f77_weak_on_purpose_this_qualifies_one_cavea|Weak on purpose: this qualifies one caveat of the earlier layers findings…]]
<!-- crux:ledger:end -->
