---
id: q90
type: question
schema: 3
title: Can a count-space model like miniCANDI be scored fairly on the EIC board of signal p-value entrants, through an exact count-to-p-value pipeline or a change of the model's output?
parent: q6
created: "2026-11-22T13:58:48"
updated: "2026-11-22T13:58:48"
---

# q90 — Can a count-space model like miniCANDI be scored fairly on the EIC board of signal p-value entrants, through an exact count-to-p-value pipeline or a change of the model's output?

Parent:: [[q6_how_closely_does_a_map_h_reproduce_the_l]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Can a count-space model like miniCANDI be scored fairly on the EIC board of signal p-value entrants, through an exact count-to-p-value pipeline or a change of the model's output?

## Bears on

an exact, tested conversion shows how much of the chain's p-value error is the map and how much is the predicted counts

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**3 children** · open 0, leaning 2, settled 1, parked 0 · 12 finding(s) on this question

- `h90` [[h90_max_s_quantile_normalization_rule_and_hi|Max's quantile-normalization rule and his 1 bp MACS2 formula together reproduce the organizers' signal p-value pipeline exactly, so a count route can be converted without a conversion error of its own]] — *settled · strong*
- `h91` [[h91_peaks_kept_by_the_encode_naive_overlap_r|Peaks kept by the ENCODE naive-overlap rule across two halves of the reads (an entrant-side analogue of the pseudoreplicate-overlap set) bring the count route's normalized score close to the score it gets with the truth's own peaks]] — *leaning · moderate*
- `q91` _(Q)_ [[q91_why_does_our_chip_reprocessing_of_the_bl|Why does our ChIP reprocessing of the blind set differ from Jacob Schreiber's SE tracks before the p-value step, when the DNase track is byte-identical?]] — *leaning · moderate*
- `f391` finding (moderate) [[f391_3_experiments_only_one_per_assay_class_t|3 experiments only, one per assay class; the model-side test needs blind…]]
- `f393` finding (moderate) [[f393_3_experiments_reproduction_exact_placeme|3 experiments; reproduction exact, placements are observed-read rungs, not…]]
- `f395` finding (moderate) [[f395_3_experiments_observed_reads_how_to_make|3 experiments, observed reads; how to make two halves from a model's…]]
- `f396` finding (moderate) [[f396_3_experiments_observed_reads_nmse_vs_the|3 experiments, observed reads, nMSE vs the exact formula; no model retrained]]
- `f397` finding (moderate) [[f397_a_one_command_pipeline_score_entry_py_ru|A one-command pipeline (score_entry.py + run_entry.sh) takes predict.py's NB…]]
- `f398` finding (moderate) [[f398_the_count_to_signal_code_assumed_rl_24_2|The count-to-signal code assumed (rl+24)/25 bins per read; the store rule…]]
- `f399` finding (moderate) [[f399_the_entry_pipeline_also_runs_on_the_eic_|The entry pipeline also runs on the EIC validation board (42 of 45…]]
- `f400` finding (moderate) [[f400_first_minicandi_scores_on_the_eic_valida|First miniCANDI scores on the EIC validation board (final model, seed 0, not…]]
- `f401` finding (moderate) [[f401_all_8_minicandi_validation_predictions_f|All 8 miniCANDI validation predictions (final model, seed 0, not…]]
- `f403` finding (moderate) [[f403_wide_grid_recalibration_n_divided_by_a_p|Wide-grid recalibration (n divided by a per-assay factor) made the…]]
- `f404` finding (weak) [[f404_interim_30_of_42_predicted_validation_ex|INTERIM, 30 of 42 predicted validation experiments: miniCANDI seed 0 raw…]]
- `f405` finding (moderate) [[f405_the_validation_error_of_minicandi_s_mean|The validation error of miniCANDI's mean predictions is a scale error, not a…]]
<!-- crux:ledger:end -->
