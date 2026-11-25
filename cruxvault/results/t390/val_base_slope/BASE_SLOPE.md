# Count slope of the base model beside the final model (30 validation experiments, chr1, chr9, chr21)

Base model: eic_v1_2 rung-4 base, seed 0 (Fir `/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/eic_v1_2_rung4_base_heldout_s0`). Final model: rcp_combo_w2x_unet seed 0. Both predict the validation experiments without having trained on them: the trainer's train mode reads only rows with `in_eic = train` (kit COD1b `tools/minicandi/dataset.py:10-14`), and all 42 validation experiments have `in_eic = val` (`tables/eic_v1_2/measurements.tsv`). "heldout" in the run name is the held-out genome windows of the training loss.

Count slope = SD(predicted mean) / (r x SD(observed counts)); 1 = calibrated, above 1 = the prediction's contrasts are too large. Per track: `SLOPE_FINAL_VS_BASE.tsv`. Diagnosis: `tools/eic_entry/diag_scale.py`, Fir jobs 6030581-6030610.

| | final | base |
|---|---|---|
| median count slope | 1.34 | 1.20 |
| median correlation with observed counts | 0.627 | 0.563 |
| median p-value error, squared error / variance of truth (no depth cap) | 1.82 | 1.04 |

On the 16 tracks where the final model's count slope is above 1.3, the base slope is between 0.8 and 1.2 on 2 and below 1.3 on 6. The base model is less over-dispersed overall but correlates less with the observed counts; it is not near 1 where the final model overshoots.
