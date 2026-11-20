pair regions: 256 (none of 8, 16, 32 passes)
16-region screening scores settled: no (p = 16 fails a condition)

The check as amended on 2026-11-16 ran on 12 runs of 4 arms (rung2_base_heldout, rung3_base_heldout, rung4_base_heldout, final_rung4_mech_skip_dna) with 2000 region-bootstrap draws over 245 clusters of 256 regions.

| p | (1) | (2) | (3) | (4) | (5) | (6) | all |
|---|---|---|---|---|---|---|---|
| 8 | fail | fail | pass | fail | pass | fail | fail |
| 16 | fail | fail | pass | fail | pass | fail | fail |
| 32 | fail | fail | pass | fail | pass | pass | fail |

Amended constants: `{"amended": "2026-11-16", "boot_seed": 0, "comparisons": ["rung2_base_heldout vs rung3_base_heldout", "rung2_base_heldout vs rung4_base_heldout", "rung3_base_heldout vs rung4_base_heldout", "rung4_base_heldout vs final_rung4_mech_skip_dna"], "correction_only": "score.correction_only_block", "correction_term": "gap closed, oracle reference, headline pairs with no duplicate-keeping side (composite version 3)", "inside_rtol": 1e-09, "interval_percentiles": [2.5, 97.5], "metric7": "keys equal", "min_defined_share": 0.5, "n_boot": 2000, "per_assay_share_min": 0.95, "rank_quantities": ["composite_v3", "correction_v3", "correction_only"], "rank_rule": "sign of the subset difference equals the 256-region sign when the 256-region difference of arm means exceeds the larger of the two seed ranges", "screening_rtol": 1e-06, "sizes": [8, 16, 32], "spread_ratio_max": 1.5, "verdict": "equal on subset and all regions", "verdict_simpler": "none"}`.
D13 verdicts on all regions: rung2_base_heldout vs rung3_base_heldout: TIE_ASK_PI; rung2_base_heldout vs rung4_base_heldout: TIE_ASK_PI; rung3_base_heldout vs rung4_base_heldout: TIE_ASK_PI; rung4_base_heldout vs final_rung4_mech_skip_dna: TIE_ASK_PI.
Limit: CPU-trained models of three architectures; rungs 1, 5 and 6 are not tested.
The per-row values are in `check_v3.tsv`.
