---
id: q48
type: question
schema: 3
title: Which miniCANDI design choices move imputation, correction or calibration beyond the seed range on the conv autoencoder, conv + attention and the U-Net at about 1.0 M parameters?
parent: q22
created: "2026-11-16T22:22:44"
updated: "2026-11-16T22:22:44"
---

# q48 — Which miniCANDI design choices move imputation, correction or calibration beyond the seed range on the conv autoencoder, conv + attention and the U-Net at about 1.0 M parameters?

Parent:: [[q22_does_the_smallest_model_beat_the_baselin]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Which miniCANDI design choices move imputation, correction or calibration beyond the seed range on the conv autoencoder, conv + attention and the U-Net at about 1.0 M parameters?

## Bears on

The answer fixes the settled miniCANDI design that goes to scaling; a choice with no effect takes its simpler setting.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**14 children** · open 6, leaning 8, settled 0, parked 0 · 1 finding(s) on this question

- `q49` _(Q)_ [[q49_does_the_chip_seq_control_help_minicandi|Does the ChIP-seq control help miniCANDI as an input assay and as a predicted output?]] — *leaning · weak*
- `q50` _(Q)_ [[q50_does_the_leave_one_out_average_of_the_tr|Does the leave-one-out average of the training cell types, given as an input, help miniCANDI?]] — *leaning · moderate*
- `q51` _(Q)_ [[q51_does_two_step_correction_through_a_base_|Does two-step correction through a base covariate set help miniCANDI, with native or fixed base depth?]] — *leaning · moderate*
- `q52` _(Q)_ [[q52_do_the_dna_source_and_the_way_dna_featur|Do the DNA source and the way DNA features join the count features matter for miniCANDI?]] — *leaning · weak*
- `q53` _(Q)_ [[q53_does_the_form_of_the_covariate_embedding|Does the form of the covariate embedding (numeric only, per-value numeric, dropout) matter for miniCANDI?]] — *leaning · weak*
- `q54` _(Q)_ [[q54_does_another_loss_poisson_mean_nb_crps_e|Does another loss (Poisson mean, NB CRPS, equal bin weight, imputation weight 4) help one of the three goals of miniCANDI?]] — *leaning · moderate*
- `q55` _(Q)_ [[q55_does_a_count_transform_on_the_input_beat|Does a count transform on the input beat the depth offset in miniCANDI?]] — *leaning · moderate*
- `q56` _(Q)_ [[q56_do_the_ablation_conclusions_drawn_from_t|Do the ablation conclusions drawn from the 16 pair regions hold on all 256 evaluation regions?]] — *leaning · moderate*
- `q64` _(Q)_ [[q64_which_combination_of_the_switches_that_q|Which combination of the switches that qualified one at a time gives the best miniCANDI on the conv autoencoder, conv + attention and the U-Net, on all 256 regions (the mix-and-match round)?]] — *open*
- `q65` _(Q)_ [[q65_which_covariate_fields_can_be_removed_fr|Which covariate fields can be removed from miniCANDI before a goal term drops beyond the seed range (the minimum useful covariate set and its exclusion curve)?]] — *open*
- `q66` _(Q)_ [[q66_does_a_learned_cell_type_embedding_add_i|Does a learned cell-type embedding add information beyond the input tracks of a cell, as shown by a gain that disappears when the cell id is forced to unknown at scoring?]] — *open*
- `q67` _(Q)_ [[q67_does_a_bin_size_other_than_25_bp_1_to_12|Does a bin size other than 25 bp (1 to 125 bp) change miniCANDI's goal terms, beyond what the change of context length explains?]] — *open*
- `q68` _(Q)_ [[q68_do_read_derived_qc_metrics_nsc_rsc_fragm|Do read-derived QC metrics (NSC, RSC, fragment length, NRF, PBC) given as model inputs improve miniCANDI beyond the metadata covariates on two or more models?]] — *open*
- `q69` _(Q)_ [[q69_does_bulk_rna_seq_of_the_cell_given_as_a|Does bulk RNA-seq of the cell, given as an aligned track or as an expression vector, improve miniCANDI imputation?]] — *open*
- `f90` finding (moderate) [[f90_each_claim_is_a_difference_of_means_larg|Each claim is a difference of means larger than the larger seed range;…]]
<!-- crux:ledger:end -->
