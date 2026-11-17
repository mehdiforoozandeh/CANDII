---
title: "E2: signal p-value and peaks from the predicted NB; AlphaGenome 2 board row; signal-space score"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# E2: signal p-value and peaks from the predicted NB; AlphaGenome 2 board row; signal-space score

## 1. What the PI asked

- `peeling the onion.md`:147: "based on CAC findings, we can use the exact formula to go from predicted counts to signal p-value and even binary peaks -- so we don't need dedicated decoders for signal p-value and peaks. we can optionally generate them from our predicted NB"
- `peeling the onion.md`:148: "if we are also predicting chip control, we'd be relying on our own NB predictions of both signal and control"
- `long-horizon decisions.md`:75 (D9b): "P-values follow the MACS2 formula with λ from the blind biosample's observed control. Where no control exists (Caco-2, SJSA1), λ comes from a predicted control. Each track is labelled with its λ source."
- `long-horizon notepad.md`:42 (ruling 13): "The challenge's signal space (D9b signal for miniCANDI, published entries as they are) serves the comparison with external EIC methods."
- `long-horizon notepad.md`:267 (ruling 20): "the expectation is the headline, and the plug-in is reported beside it."
- `AlphaGenome 2.md`:19 (main checkout only): "similarly for predicting Peak (instead of signal) -- some of AG2's reported metrics are classification metrics on AUPRC. we should be able to process our count data into binary peaks and compare."
- `AlphaGenome 2.md`:23: "it'd be great if we can bring AG2's results into our leaderboard and make them comparable with us (CANDI), our baselines, and other competitors."

Success criterion served: 3 (evals well defined and tested). Audit rows: D14, B18, C5.

## 2. Question

Turn the predicted NB of trained miniCANDI models into signal p-value tracks and binary peaks, score them in the challenge's signal space and on the AlphaGenome 2 board, and report the source of λ (the local background rate) for every track.

## 3. State today

- Code exists: signal from predicted counts (`tools/onion_eval/signal_from_counts.py`: `macs2_lambda`, `signal_track`, `signal_from_nb`), the DNase-seq MACS2 no-control default (ruling 19), expectation and plug-in keys (ruling 20), peak tiles in the AlphaGenome 2 scorer, and the key `predicted_control` in `scores.json` (audit row D14).
- No report shows a signal track, a peak AUPRC or a predicted-control λ for a miniCANDI model (audit row D14). Signal-space scores were undefined for every baseline until the fragment-length fix (`results/fraglen_single_end/FRAGLEN.md`: 231 ChIP-seq base tracks had an empty `frag_len`).
- AlphaGenome 2 board (`results/t308/AG2_VS_BASELINES.md`), fair set of 11 tracks in 9 cells unseen by both: rspec_cal (r_spec after per-track recalibration) AlphaGenome 2 0.438, position average 0.222, marginal NB 0.121, noise floor 0.333 [0.276, 0.382]; AUPRC on ENCODE peaks 0.454, 0.271, 0.132, floor 0.571. `onion paper.md`:137: "The board has no row for our trained model yet."

## 4. Prior leads

- Ablation spec 1 (`ablation_specs/1_control_input.md`, trap 4): `scores.json` `predicted_control` holds 1 control target (cell C02). Prediction: the predicted-control λ is testable on one target only on validation; report it as a diagnostic.
- Question `q40_why_does_alphagenome_2_score_above_the_p.md`: AlphaGenome 2 scores above the pseudoreplicate floor on the fair set. Prediction: rspec_cal rewards a smooth predictor; a count model near the floor can still rank below AlphaGenome 2 on this metric.
- `results/t308`: 32 of 42 ChIP-seq headline targets had no fragment length before the side table; the rerun used an estimate for 32 of them. Prediction: signal-space scores carry a fragment-length error that count-space scores do not.

## 5. Work

1. Models: the reference and the final-round winner of conv autoencoder, conv + attention and U-Net, 3 seeds each (Fir `runs_fir`), on the validation leave-one-assay-out targets. Repeat on the final model in D4.
2. Signal: for each target, the expectation of the MACS2 p-value under the predicted NB (headline) and the plug-in (beside it). λ from the cell's observed control; from the predicted control where the cell has none. DNase-seq and ATAC-seq use the MACS2 no-control route. Label each track with its λ source.
3. Peaks: binary peaks from the predicted counts on the 500 bp tiles of the AlphaGenome 2 scorer; AUPRC against ENCODE peaks and pipeline-derived peaks.
4. Score: headline metric 1 in signal space (ruling 13) beside the count-space value; the AlphaGenome 2 board on the fair set (11 tracks) and the labelled trained set (30 tracks); report per track and macro.
5. Diagnostic: the predicted-control λ against the observed-control λ on every validation cell that has a control (hide the control, predict it, compare λ).

## 6. How the result answers the question

Acceptance checks:
- `ag2_board_val.tsv` holds a miniCANDI row per architecture, with rspec_cal, pearson_cal, Spearman and both AUPRC columns on the fair set.
- Every signal track written has a λ-source label (`observed_control`, `predicted_control`, `no_control_macs2`).

| outcome | conclusion |
|---|---|
| miniCANDI rspec_cal and AUPRC above the position average beyond the seed range on the fair set | the NB-to-signal route works; no dedicated signal or peak decoder is needed (constitution L147) |
| above the position average in counts but not in signal space | the route loses information (fragment length, λ); fix the route before the blind pass |
| predicted-control λ within 10% of observed λ on most cells | the predicted control is usable for Caco-2 and SJSA1 in the blind pass |

## 7. Dependencies, cost, risks

- Depends on nothing for validation. The blind pass (D4) reuses the route.
- Cost: one signal pass per seed; not estimated. Time one job first (notepad L531). About 2 agent-days.
- Risk 1: the fair set has 11 tracks and 9 of 10 marks have one track (`onion paper.md`:137), so the board shows a direction, not a per-mark rank.
- Risk 2: AlphaGenome 2 outputs are evaluation-only (notepad L439); no selection may read them.

## 8. Done when

The AlphaGenome 2 board and a signal-space metric-1 table hold rows for 3 architectures x 3 seeds, with expectation and plug-in values and a λ source per track, and the predicted-control diagnostic is reported.

## 9. Specification as worked (E2 agent, 2026-11-16)

- State today, checked: the base recipes exist only on Fir (`runs_fir/rung{2,3,4}_base_heldout/seed{0,1,2}`; the Nibi seed directories are empty). The AG2 pack, the AG2 audit, the ENCODE peak files and the t308 method tracks exist only on Nibi. Fir had no fragment-length side table (E2 copied the t308 table, sha256 4f2efdb6..., to Fir `e2/eic_v1_fraglen.tsv`). The prediction stores of the base recipes are gone, so a new forward pass is needed.
- Fair-set numbers, recomputed from `results/t308/ag2_per_target_val.tsv` (11 tracks, 9 cells): rspec_cal AG2 0.438, noise floor 0.333, position average 0.222.
- Formula check (new, step 0): the formula on the OBSERVED counts against the pipeline's p-value track (`tools/onion_eval/nb_signal_checks.py formula`). A model can do no better than this agreement in the signal space.
- Predicted-control diagnostic: in regime `leave_one_assay_out` the observed control is a model input, so its predicted control is a reconstruction. The hidden-control test is regime `accessibility_only` (inputs: the accessibility track only). Both are reported (`nb_signal_checks.py lambda`).
- Scoring: `score.py --lambda-source predicted_control --fraglen-table --dump-signal` (legacy path); the dump now also holds `ctl_pred.npy`. Eval set: the 256 regions of the AG2 comparison without pair jobs (`e2/evalset_e2_targets_only.json`). Board: `ag2_score.py` on Nibi, tables `headline` and `control_free`. Submit script: `slurm/onion_eval/e2_submit.sh` (dry run by default).
- Formula check result (2026-11-17): today's in-bin pileup agreed with the pipeline's -log10 p track on observed counts at a median Pearson of 0.831 (41 ChIP-seq validation targets) and over-called peak tiles (precision 0.42). A centered box spread of the fragment length d raised this to 0.921, and the tile F1 rose from 0.573 to 0.782 (`cruxvault/results/closeout_E2/FORMULA_EXTENSION.md`). The box was chosen among 4 variants on 42 validation targets. The strand and the bp positions are lost at binning, so no formula on 25 bp counts reproduces the pipeline exactly.
- PI ruling 2026-11-17, "Adopt the box spread": `box_d` is the pileup of every signal route (observed control and predicted control) in `signal_from_counts`, `ag2_space`, `score.py`, `ag2_baselines.py`, `minicandi/predict.py` and `nb_signal_checks.py`. `--signal-pileup inbin` keeps the old rule. The form is written as `signal_pileup` in scores.json and tracks.json. `select_rule` refuses a mix unless `--allow-mix signal_pileup` is given. `ag2_score` refuses a board that mixes pileup forms.
- Model side of `box_d` (corrected after review, 2026-11-17): the spread pileup's NB has the box mean of the predicted means and a size set by the bin dependence, an explicit option recorded as `signal_box_dependence`. `comonotone` (default) is the upper variance bound: n_s = m^2 / (sum w mu / sqrt n)^2. `independent` is the lower bound when neighboring bins are not negatively correlated: n_s = m^2 / sum w^2 mu^2 / n. The NB sits on the pileup scale (mean s * m, s = d / (rl + 24)), so its Poisson part is the fragment count's s * m (second review; checked by simulation). The `inbin` form keeps its old count-scale expectation, whose Poisson part is too large by the factor s. The -log10 p score is convex in the count, so the comonotone default gives the LARGER expected scores. Both are reported; for non-negative correlation the truth lies between them. The plug-in mode needs no assumption. The box is zero-padded at region ends, and the `box_reach(d)` end bins are masked from the signal-space vector metrics and the peak tiles. Headline 1 (Haar levels over whole regions) keeps them.
- PI comment 2026-11-17: for ChIP-seq, signal p-value and peaks from the predicted signal AND the predicted control is a headline row beside the observed-control route.
- Owed: rescore the 9 runs with box_d on Fir, rebuild the count-derived board references with box_d on Nibi, move the dumps by Globus, then score the board (`slurm/onion_eval/e2_submit.sh fir | nibi_baselines | nibi_board`).
