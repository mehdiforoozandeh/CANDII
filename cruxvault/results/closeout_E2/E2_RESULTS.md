# E2: signal p-value and peaks from the predicted NB (partial, 2026-11-17)

## 1. Formula check on observed counts (Nibi job 27533881, 4 min, 1 CPU)
The formula of `signal_from_counts` (plug-in mode) was run on the OBSERVED counts of the 42 validation targets with a pipeline p-value track. The result was compared with the pipeline's own -log10 p track (`signal.h5`). File: `formula/formula_check.tsv`, summary `formula/formula_check.json`.
- ChIP-seq, 41 targets: median Pearson 0.831 (range 0.583 to 0.993), median slope 0.749. Targets with the table fragment length: median Pearson 0.887 (9 targets). Targets with an estimated fragment length: 0.830 (32 targets).
- DNase-seq, 1 target: Pearson 0.943.
- Peaks at -log10 p >= 2 on 500 bp tiles (ChIP-seq): recall 0.957, precision 0.421, F1 0.573. The formula calls about twice as many tiles as the pipeline.
- One target (ENCSR521AIV.br1, H3K9ac) has no pipeline signal track.
- Meaning: this agreement is a ceiling for any model in the signal space. The formula uses each bin's own read-span count and no fragment extension across bins, so it is noisier than the pipeline pileup and over-calls peaks.

## 2. Conv autoencoder, seed 0 (Fir job 4448330, 11 h 44 min, 2 CPUs)
`score.py --lambda-source predicted_control --fraglen-table --dump-signal` on `evalset_p16.json`. Count-space headline 1 was 0.3637, equal to the earlier `p16_sv` score of the same run. Files: `scores/conv_ae_s0_e2.scores.json`, `table_rung2_s0/`.
- Headline 1 (variance recovered above noise), leave-one-assay-out, column `macro_common` of `signal_space_macro.tsv`, i.e. the mean over the 8 scorable assays defined in every space (H3K9ac is left out of all spaces): counts 0.331, signal (expectation) 0.178, signal (plug-in) 0.164, signal with the predicted control 0.143. The scorer's signal macro is undefined because H3K9ac lacks one truth track.
- r_spec on the same 8 assays: counts 0.263, signal with the predicted control 0.267 (other signal values in `signal_space_macro.tsv`).
- Lambda source per track: 41 `observed_control`, 1 `local_background` (DNase-seq), 1 undefined (no truth). Predicted-control variant: 41 `predicted_control` (`signal_space_lambda_sources.tsv`).
- In regime accessibility_only, 1 scorable assay (H3K79me2) has no record (`missing_scorable`), and headline 1 is defined for 1 assay only.
- One seed only: no seed range exists yet.

## 3. Predicted-control lambda against observed-control lambda (Fir job 4565715)
File: `lambda_rung2_s0/lambda_check.tsv`.
- Control hidden (regime accessibility_only, 20 targets, 9 cells): median of the per-target median ratio 0.919; 15 of 20 targets have a median ratio within 10%; median fraction of bins within 10% is 0.424.
- Control visible (leave_one_assay_out, 41 targets, 24 cells): median ratio 0.968; 35 of 41 within 10%; bins within 10% 0.427.

## 4. Predicted-control route for ChIP-seq (headline row beside the observed-control route; PI comment 2026-11-17)
Signal p-value from the predicted treatment NB and the predicted control (no observed control), conv autoencoder seed 0 (`table_rung2_s0/signal_space_macro.tsv`, `macro_common`, 8 assays):

| regime | metric | observed-control route | predicted-control route | counts |
|---|---|---|---|---|
| leave_one_assay_out (control is an input) | r_spec | 0.234 | 0.267 | 0.263 |
| leave_one_assay_out | headline 1 | 0.178 | 0.143 | 0.331 |
| accessibility_only (control hidden) | r_spec | 0.106 | 0.200 | 0.192 |

The lambda ratio of the predicted control to the observed control (section 3): median 0.919 with the control hidden (15 of 20 targets within 10%), 0.968 with the control visible (35 of 41).
Peaks on this route: not scored yet. The AG2 scorer's `control_free` table (Nibi, step 2 of `slurm/onion_eval/e2_submit.sh`, after the dumps move by Globus) ranks tiles by `signal_pc` in regime accessibility_only. No existing job scores peaks from `signal_pc` in leave_one_assay_out: `ag2_score.py --table headline` reads `signal.npy`, and the dump's `peakprob.npy` uses the observed control.
All signal values above use today's `inbin` pileup. Section 2 of `FORMULA_EXTENSION.md` shows that `box_d` is closer to the pipeline.
