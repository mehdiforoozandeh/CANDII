# Benchmark programme — tracked state memo

Tracked copy of the unit table in `cruxvault/results/t81/PROGRAMME_STATE.md` (gitignored, per-worktree).
The gitignored file is the live one; this memo is refreshed when a phase closes. Brief:
`plan/HANDOFF_FINISH_BENCHMARK.md`. Branch: `implementation/t81-finish-benchmark` (the PI merges it to main).

## Decisions recorded 2026-09-01

| # | decision |
|---|---|
| D1 | `marginal`, `knn1`, `knn5` run once per regime; `avg`, `avg-arcsinh` once. 18 method-regime units. §12.2/§12.3 to be corrected; identity assertion for `avg`/`avg-arcsinh` to be added. |
| D2 | Build the training-residual σ pass; repoint all four rivals' `fit_sigma.py` and `avg-arcsinh` at it. |
| D3 | Truth toggle does not apply to `V_`. Exclusion recorded on the board. |
| D4 | `eic.19` band sized from a fresh measured epoch rate, `--early-stop-epochs 3`, 25-epoch ceiling. |
| D5 | The 40 `counts.h5.pre_t78p3` backups stay until every retrain is scored. |
| ops | Fir code: fresh clone `/project/def-maxwl/mforooz/CANDII_main` on this branch. Venv: `/project/def-maxwl/mforooz/EpiDenoise/candi_venv`. t90: rsync via a SLURM CPU job, checksum-verified, scratch copy left to purge. |

## Unit table

Rendered 2026-09-02 01:42 PDT from 67 row files.

Copied from `cruxvault/results/t81/PROGRAMME_STATE.md`. The convention and the one-line
renderer are in `cruxvault/results/t81/state/README.md`.

| unit | phase | slurm job | state | artifact path | number | date |
|---|---|---|---|---|---|---|
| anchor.pred.Aug2019Imputation | anchor | 57788584_1 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Aug2019Imputation/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Average | anchor | 57788584_24 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Average/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Avocado_p0 | anchor | 57788584_25 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Avocado_p0/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.BrokenNodes | anchor | 57788584_2 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/BrokenNodes/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.BrokenNodes_v2 | anchor | 57788584_3 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/BrokenNodes_v2/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.BrokenNodes_v3 | anchor | 57788584_4 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/BrokenNodes_v3/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.CUImpute1 | anchor | 57788584_5 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/CUImpute1/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.CUWA | anchor | 57788584_6 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/CUWA/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.CostaLab | anchor | 57788584_7 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/CostaLab/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Guacamole | anchor | 57788584_8 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Guacamole/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Hongyang_Li_and_Yuanfang_Guan | anchor | 57788584_9 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Hongyang_Li_and_Yuanfang_Guan/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Hongyang_Li_and_Yuanfang_Guan_v1 | anchor | 57788584_10 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Hongyang_Li_and_Yuanfang_Guan_v1/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Hongyang_Li_and_Yuanfang_Guan_v2 | anchor | 57788584_11 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Hongyang_Li_and_Yuanfang_Guan_v2/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.ICU | anchor | 57788584_12 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/ICU/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.KKT-ENCODE-Impute | anchor | 57788584_13 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/KKT-ENCODE-Impute/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Lavawizard | anchor | 57788584_14 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Lavawizard/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.LiPingChun | anchor | 57788584_15 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/LiPingChun/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.NittanyLions | anchor | 57788584_16 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/NittanyLions/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.NittanyLions2 | anchor | 57788584_17 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/NittanyLions2/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Song_Lab | anchor | 57788584_18 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Song_Lab/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Song_Lab_2 | anchor | 57788584_19 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Song_Lab_2/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.Song_Lab_3 | anchor | 57788584_20 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/Song_Lab_3/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.UIOWA_Michaelson | anchor | 57788584_21 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/UIOWA_Michaelson/B_ | 50 of 51 declared tracks converted (chr20-22; C38M18 never submitted, in skipped_tracks) | 2026-09-01 |
| anchor.pred.imp | anchor | 57788584_22 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/imp/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| anchor.pred.imp1 | anchor | 57788584_23 | done | /project/def-maxwl/mforooz/t81_pred_B/anchor/imp1/B_ | 51 of 51 declared tracks converted (chr20-22, floor 25 bp grid) | 2026-09-01 |
| ops.candi_probe_eic_19 | probe | 57787183_0 | done | /project/def-maxwl/mforooz/CANDII_main/slurm-logs/t81_train_57787183_0.out + /scratch/mforooz/t81_candi/runs/probe_eic_19_s0_3000.json | sampled 66.13 win/s; full-coverage epoch 87.2 min (projected, FC_FACTOR 0.45); scoped check 8.27 min; final full-coverage check 91-94 min/dial; MODE=full eic_19 --time=47:00:00 | 2026-09-01 |
| ops.fir_clone | 1 | - | done | /project/def-maxwl/mforooz/CANDII_main@4d77a6d23d025f152f0cb35f6955b85fac484c21 | - | 2026-09-01 |
| ops.fir_pull | 1 | - | done | /project/def-maxwl/mforooz/CANDII_main@b4cfc90 (was 4d77a6d, ff-only, clean) | - | 2026-09-01 |
| ops.fir_pull_2 | ops | - | done | /project/def-maxwl/mforooz/CANDII_main@e9cf729 (was b4cfc90, ff-only, clean) | - | 2026-09-02 |
| ops.t90_move | 1 | 57782741 | running | /project/def-maxwl/mforooz/t54_submissions_round2 | - | 2026-09-01 |
| ops.truth_root_B | ops | 57788584_0 | done | /project/def-maxwl/mforooz/t81_truth_challenge/B_ | 51 of 51 declared tracks converted (12 B_ pairs, chr20-22, floor 25 bp grid) | 2026-09-01 |
| pred.V_.ChromImpute.eic_19 | pred | 57806197,57806198,57806199,57806200 | queued | /scratch/mforooz/t81_pred/ChromImpute/eic_19/V_ (pending; run dir /scratch/mforooz/t81_chromimpute/eic_19/V_) | - | 2026-09-02 |
| pred.V_.ChromImpute.eic_pilot | pred | 57806201,57806202,57806203,57806204 | queued | /scratch/mforooz/t81_pred/ChromImpute/eic_pilot/V_ (pending; run dir /scratch/mforooz/t81_chromimpute/eic_pilot/V_) | - | 2026-09-02 |
| pred.V_.avg-arcsinh.eic_19 | pred | 57804705,57804707 | running | /scratch/mforooz/t81_pred/avg-arcsinh/eic_19/V_ | - | 2026-09-02 |
| pred.V_.avg.eic_19 | pred | 57804705,57804707 | running | /scratch/mforooz/t81_pred/avg/eic_19/V_ | - | 2026-09-02 |
| pred.V_.eDICE.eic_19 | pred | 57806922 | queued | /scratch/mforooz/t81_pred/eDICE/eic_19/V_.genomewide (pending; SCOPE=genomewide appends .genomewide to the panel root) | - | 2026-09-02 |
| pred.V_.eDICE.eic_pilot | pred | 57806379 | queued | /scratch/mforooz/t81_pred/eDICE/eic_pilot/V_.genomewide (pending; SCOPE=genomewide appends .genomewide) | - | 2026-09-02 |
| pred.V_.knn1.eic_19 | pred | 57804705 | running | /scratch/mforooz/t81_pred/knn1/eic_19/V_ | - | 2026-09-02 |
| pred.V_.knn1.eic_pilot | pred | 57804706 | queued | - | - | 2026-09-02 |
| pred.V_.knn5.eic_19 | pred | 57804705 | running | /scratch/mforooz/t81_pred/knn5/eic_19/V_ | - | 2026-09-02 |
| pred.V_.knn5.eic_pilot | pred | 57804706 | queued | - | - | 2026-09-02 |
| pred.V_.marginal.eic_19 | pred | 57804705 | running | /scratch/mforooz/t81_pred/marginal/eic_19/V_ | - | 2026-09-02 |
| pred.V_.marginal.eic_pilot | pred | 57804706 | queued | - | - | 2026-09-02 |
| score.store.V_.eDICE.eic_19 | score | 57806922 | queued | /project/def-maxwl/mforooz/t81_scores/eDICE/eic_19/store.V_.json (pending; same job as the predict, --held-out-chroms chr20,chr21,chr22) | - | 2026-09-02 |
| score.store.V_.eDICE.eic_pilot | score | 57806379 | queued | /project/def-maxwl/mforooz/t81_scores/eDICE/eic_pilot/store.V_.json (pending; same job as the predict, --held-out-chroms chr20,chr21,chr22) | - | 2026-09-02 |
| sigma.ChromImpute.eic_19 | sigma | - | blocked | /project/def-maxwl/mforooz/t81_sigma/ChromImpute/sigma_eic_19.json (not written) | - | 2026-09-02 |
| sigma.eDICE.eic_19 | sigma | 57806921 | queued | /project/def-maxwl/mforooz/t81_sigma/eDICE/sigma_eic_19.json (pending; train_pred /scratch/mforooz/t81_sigma/eDICE/eic_19/train_pred) | - | 2026-09-02 |
| sigma.eDICE.eic_pilot | sigma | 57806378 | queued | /project/def-maxwl/mforooz/t81_sigma/eDICE/sigma_eic_pilot.json (pending; train_pred /scratch/mforooz/t81_sigma/eDICE/eic_pilot/train_pred) | - | 2026-09-02 |
| train.Avocado.eic_19.s0 | train | 57788169,57788170,57788171 | queued | - | - | 2026-09-01 |
| train.Avocado.eic_pilot.s0 | train | 57788188,57788189,57788190 | queued | - | - | 2026-09-01 |
| train.CANDI.eic_19.s0 | train | 57788373_0 | running | - | - | 2026-09-01 |
| train.CANDI.eic_19.s1 | train | - | blocked | - | - | 2026-09-01 |
| train.CANDI.eic_pilot.s0 | train | 57674899_1 | done | /project/def-maxwl/mforooz/t81_checkpoints/t81_eic_pilot_s0.best.ckpt (md5 742f4251…) + t81_eic_pilot_s0.arch.json | impute macro CRPS 0.5386 (selection, n=45, chr20-22 full coverage, epoch 5; eval_curve row, not a result) | 2026-09-01 |
| train.ChromImpute.eic_19.s0 | train | 57788380,57788381,57788382,57788383,57788384,57788559 | done | /project/def-maxwl/mforooz/t81_checkpoints/ChromImpute/eic_19/PREDICTORDIR (+DISTANCEDIR, inputinfofile.txt, chrominfo.txt, chrominfo.train.txt, regime.eic_19.V_.json); run dir /scratch/mforooz/t81_chromimpute/eic_19/V_ | 556 classifier files; every stage prepare/convert/dist/gtd/train/ckpt_train COMPLETED exit 0:0; training grid chr19 2,344,704 bins | 2026-09-02 |
| train.ChromImpute.eic_pilot.s0 | train | 57788389,57788390,57788391,57788392,57788393,57788560 | done | /project/def-maxwl/mforooz/t81_checkpoints/ChromImpute/eic_pilot/PREDICTORDIR (+DISTANCEDIR, inputinfofile.txt, chrominfo.txt, chrominfo.train.txt, regime.eic_pilot.V_.json); run dir /scratch/mforooz/t81_chromimpute/eic_pilot/V_ | 556 classifier files; every stage prepare/convert/dist/gtd/train/ckpt_train COMPLETED exit 0:0; training grid 40 pilot loci over 18 chroms, 1,023,489 bins | 2026-09-02 |
| train.Lavawizard.eic_19.s0 | train | 57788192,57788194,57788195,57788196 | queued | /project/def-maxwl/mforooz/t81_checkpoints/Lavawizard/eic_19/ (pending; guacamole_shared.pt + guacamole_chr{20,21,22}.best.pt); caches 57788192,57788194 COMPLETED | - | 2026-09-01 |
| train.Lavawizard.eic_pilot.s0 | train | 57788197,57788198,57788199,57788200 | queued | /project/def-maxwl/mforooz/t81_checkpoints/Lavawizard/eic_pilot/ (pending; guacamole_shared.pt + guacamole_chr{20,21,22}.best.pt); cache 57788197 COMPLETED, 57788198 queued | - | 2026-09-01 |
| train.avg-arcsinh.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.avg.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.eDICE.eic_19.s0 | train | 57788364 | done | /project/def-maxwl/mforooz/t81_checkpoints/eDICE/eic_19/model.selected.pt | selection V_ pval mse 176.178 @ep2 (launcher metric, eval_curve row, not a result); early stop ep8; 3h44; MaxRSS 11.4G | 2026-09-02 |
| train.eDICE.eic_pilot.s0 | train | 57788365 | done | /project/def-maxwl/mforooz/t81_checkpoints/eDICE/eic_pilot/model.selected.pt (23827650 bytes) + select.json + train.json | selected epoch 2 of 8; V_ macro pval mse 222.701429 (n=45); EARLY STOP at epoch 8, patience 3; rc=0, elapsed 02:44:55, MaxRSS 28.3G of 48G | 2026-09-02 |
| train.knn1.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn1.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn5.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn5.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.marginal.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.marginal.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |

## Summary

Counts by phase and state, as the rows stand at the render time above. Rows are written by
many chunks; some units were still being launched while this table was rendered, so a
`queued` or `running` row may already have moved on.

- `train`: 5 done, 1 running, 4 queued, 9 blocked (19 rows)
- `sigma`: 2 queued, 1 blocked (3 rows)
- `pred`: 5 running, 7 queued (12 rows)
- `score`: 2 queued (2 rows)
- `anchor`: 25 done (25 rows)
- `ops`: 2 done (2 rows)
- `probe`: 1 done (1 rows)
- `1`: 2 done, 1 running (3 rows)

All 67 rows: 35 done, 7 running, 15 queued, 10 blocked.
