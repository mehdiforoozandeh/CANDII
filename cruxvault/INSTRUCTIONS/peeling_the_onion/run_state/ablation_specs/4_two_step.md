# Ablation 4: two-step covariate artifact correction through the base

Written 2026-11-16 by the ablation agent for `two_step` (branch `chunk/onion-AB-4`, base fb57fc1). Phase 1: specification only. No code was changed.

## 1. Question

Does miniCANDI correct covariate artifacts better on never-trained (C, C') pairs when it routes every prediction through the base covariates, i.e. (X, C) -> (X_base, C_base) -> (X', C')? Does it matter if the base keeps its native depth or is thinned to one reference depth? The PI's line (`peeling the onion.md`, # miniCANDI): "we also showed that this model generalizes to unseen C,C' pairs if we first go from (X, C) to (X_base, C_base) and from (X_base, C_base) to (X', C') -- i.e. a two step CAC."

## 2. Arms

| arm | keys changed from `final/rung<r>/reference_equal.json` | compared with |
|---|---|---|
| reference (base recipe) | none (`two_step: false`) | -- (existing runs `rung{2,3,4}_base_heldout`, Fir, already scored in `p16_sv`) |
| `two_step_native` | `two_step: true` (`base_depth: native`) | reference |
| `two_step_fixed` | `two_step: true`, `base_depth: fixed` | `two_step_native` (`stage2_arms.py` `COMPARISON`) |

What the code does (`tools/minicandi/two_step.py`, `registry.build_model`, `dataset.py` `_base_leg`, `losses.py`):
- `registry.build_model` builds the rung without the key `two_step`, then wraps it in `TwoStep`. Leg 1 is the rung on the batch's own inputs, decoded at C_base; it predicts the base view of every target track. Leg 2 is the same rung (shared weights) and reads leg 1's predicted mean (not detached) as its input counts, at C_in = C_base, decoded at the target's C'. The output is leg 2's NB. No parameter is added. The forward pass runs the rung twice.
- The loss adds the NB negative log-likelihood of the real base view `y_base` under leg 1's NB, on the same masked bins, weight 1 (`loss_weights.base`).
- C_base is the base view of the target's own (cell, assay): a measurement with `arm_knob` empty or `base`, preferring layer `uniform`, then unit `pooled` (`_base_rank`). This is D15's "uniform pipeline of D5, at default settings". A target with no base view uses its own C' in leg 1. An `evalonly` target is its own base view.
- `base_depth: fixed` binomially thins the base counts `y_base` to `FIXED_BASE_LOG2_DEPTH` = log2 23.0 (2^23 = 8,388,608 reads) for every assay class, and C_base states that depth. A base below it is not changed. The value is an agent decision (comment at `dataset.py` line 246), not the PI's. On Fir `eic_v1/measurements.tsv`, 758 of 785 non-evaluation-only base rows are deeper than 2^23 (mean log2 depth 24.55 narrow histone, 24.72 broad histone, 25.24 DNase, 26.22 ATAC, 24.92 control), so the fixed arm thins about 97% of the bases by a factor of about 3 to 9.

One reading only. "Fixed reference depth" could also mean that the input is thinned (`input_depth: fixed`); that is ablation-scope of the depth-control arm, not D15. I take the code's reading, which matches D15's text ("The base is thinned to one reference depth per assay class").

## 3. Prior leads

1. **CAC test bed, per-track model, chained through the base** (`CANDII/cruxvault/results/t128p3/analyze_out.txt`, rows `D per_track never` and `Dviabase per_track never`; finding f28): held-out CRPS on never-trained arm-to-arm pairs 0.5447 (oracle_scaled 0.5367, scale_error 0.0079, seed spread 0.0077) chained, against 0.6624 (0.6059 / 0.0565, spread 0.0433) direct. Gain 0.118, of which 0.069 is oracle_scaled. This is the PI's "going through the base helps a lot".
2. **Same, one model shared across tracks** (same file, `D across never` 0.6667 / spread 0.0195; `Dviabase across never` 0.6364 / spread 0.0642): gain 0.030, within noise.
3. **Final CAC test bed, shared model, 6 seeds** (`CANDII/.claude/worktrees/t146-cac-final/cruxvault/results/t146/REPORT.md` and `pooled.tsv`; finding f31, direction against, moderate): pooled never-trained CRPS 0.6333 chained against 0.6599 direct, bar 0.154, chained beats direct beyond the bar in 2 of 98 cells. On the crop knob alone (`pooled.tsv`, `alone chain never crop` / `alone direct never crop`): 0.4900 (spread 0.0033) chained against 0.5171 (spread 0.0238) direct; the difference 0.027 is beyond the larger range 0.024. Ceiling (leg 2 fed the real base) on crop 0.3718.
4. **Depth law** (finding f19, `results/h19/report.md`): chaining two trained legs through the base meets the depth law (0.07 to 0.16 log2) where the direct route misses it (0.29 to 0.34).
5. **Rung-1 miniCANDI run** (`/project/def-maxwl/mforooz/ONION/runs/rung1_two_step_native/seed0`, Nibi; COVERAGE.md row 21): seed 0 DONE (job 27272462, 01:56:10); seeds 1 and 2 never ran; not scored. `flops_per_window` 195,329,046,528 against 97,652,431,872 for `rung1_base/seed0`: ratio 2.0002. Parameter count 1,001,418 for both. Training wall on the GPU slice 6,626 s against 5,897 s (ratio 1.12). Last logged step: `main` 0.6491, `base` 0.8014; base run loss 0.6462 (one training step; not a held-out metric).
6. The archive vault (`CANDI/cruxvault`) has no two-step CAC result; its "two steps" hits (q22, h77) are about the EIC's own blind-set correction.

What the leads predict here. The test bed decoupled correction from imputation: one source track in, one target track out, chr19 + chr21, CRPS. The transfer is partial.
- Transfers: the 18 crop pairs of the correction metric are exactly the withheld view-to-view pairs (`eic_v1/withheld.tsv`: crop 36 <-> crop 50, single-end <-> crop 50; 14 crop-crop + 2 crop-se + 2 se-crop in `evalset_p16.json`), while base <-> view pairs are trained. That is the test bed's "never-trained arm-to-arm" setting, with the same base definition. miniCANDI is one shared model, so leads 2 and 3 (small, mostly within noise) apply, not lead 1. The crop-only reading of lead 3 predicts a small gain.
- Does not transfer: (a) the test bed chained a model trained on direct pairs at inference only; here both legs train jointly, with a base loss, and leg 2 can learn to undo leg 1's errors. (b) Leg 1 here also imputes. (c) No lead exists for the imputation or calibration terms. (d) The fixed-depth arm has no test-bed lead.

## 4. Expectation (before any run)

| metric | `two_step_native` vs reference | `two_step_fixed` vs `two_step_native` |
|---|---|---|
| imputation (variance recovered) | flat to down, 0 to -0.04 (leg 2 sees one track at evaluation, item 6.1) | flat, within range |
| correction (gap closed, crop pairs) | up, +0.00 to +0.05 on conv autoencoder and conv + attention; up to +0.10 on U-Net, whose correction-only term is low (0.318) | flat, within range |
| calibration | within range (ranges 0.108 to 0.178) | within range |
| composite | within range | within range |

## 5. How the result answers the question

Judge each architecture separately; "beyond" = difference of means larger than the larger of the two seed ranges.

| outcome (native vs reference) | conclusion |
|---|---|
| correction up beyond, imputation not down beyond | Two-step routing transfers to miniCANDI for unseen pairs; candidate for the design. |
| correction up beyond, imputation down beyond | Routing helps correction but costs imputation; read the composite; consider two-step for correction only. |
| correction within range | No evidence that routing helps at this scale; matches lead 3 (shared model). Two-step does not join the design. |
| correction down beyond | Joint two-leg training hurts; check item 6.1 before concluding. |
| same pattern on all three architectures | The answer is about routing, not one architecture. |

`two_step_fixed` vs `two_step_native`: beyond on correction -> the base depth matters; the better one is the D15 default. Within range -> keep `native` (no thinning, simpler).

## 6. Controls and traps

1. **Leg 2 input set differs between training and evaluation.** Leg 2 reads only the target tracks (`x_avail := y_avail`). A training window makes every assay of the cell a target (`dataset.plan`), so leg 2 trains on all of the cell's tracks. Every scoring window has one target (`score.py` line 1445, `_pair_window`), so leg 2 sees one track at evaluation. This shift can lower both imputation and correction. Control: report it beside the result; if native loses beyond the range, do not read it as "routing fails" without a test of this shift. A Phase-2 option (not in the arm count): score the same checkpoints by leg 1 only, decoded at C' (the direct route of the same weights, as test-bed lead 1 compared D and Dviabase).
2. **Parameter count.** Unchanged by construction; checked (section 7).
3. **Reference reuse.** The reference runs were trained from an earlier kit. Control: the off-switch is the base itself (`two_step: false` builds the unwrapped rung); Phase 2 checks bit-identity of the base under the new kit with `golden.py`.
4. **Scorer input.** The scorer builds `OnionDataset` with the run's config, so `c_base` is present; no two-step run has ever been scored. Control: Phase-2 dry run plus the first-seed metrics job log.
5. **Compute.** About 2x the training FLOPs per window; the composite does not price compute. Report FLOPs beside the result.
6. **Fixed depth value.** 2^23 is an agent choice; a null result for `fixed` says nothing about other reference depths.

## 7. Readiness

Per architecture: **ready**, no code. On the laptop (candii env, CPU) I wrote both arm configs with `stage2_arms.py write --arms two_step_native,two_step_fixed` onto each `final/rung{2,3,4}/reference_equal.json` (both arms status `write`; keys changed only `two_step`, `base_depth`, `simpler_than_base`). `registry.build_model` gives `TwoStep` with parameter counts equal to the reference: conv autoencoder 1,042,514; conv + attention 1,038,018; U-Net 993,599. On synthetic tables (64 bins, budget 3e5) each rung's two-step forward returns `n, mu, n_base, mu_base`, `losses.total_loss` returns the `base` part, leg 2 differs from leg 1, and `contract.wiring_check` passes. The repo test `tests/test_minicandi_two_step.py` covers rungs 1, 2 and 5 only; rungs 3 and 4 were checked by my script only. Data: base views exist for the training cells (785 base rows, Fir `/project/def-maxwl/mforooz/ONION/tables/eic_v1/measurements.tsv`); the rung-1 run read them with staging on Nibi.

Training time limit. `ladder_arms.plan` ignores `two_step`: limit = 3 x 80,000 x 0.0244 s = 5,856 s (conv autoencoder, conv + attention), 6,000 s (U-Net). Projected 2x walls: 2 x 2,521 = 5,042 s; 2 x 1,772 = 3,544 s; 2 x 2,660 = 5,320 s. The spare time is only 814 s (14% of the limit) for the conv autoencoder and 680 s (11%) for the U-Net. Phase 2 sets `TRAIN_TIME=03:00:00` in the submit script (an existing option; `train.py` also resumes from its 15-minute checkpoint).

Cost (Fir, seed 0 of each reference run, `sacct` AllocCPUS x Elapsed; two-step = 2 x training + 2 x prediction + 1 x metrics; the factor 2 is the measured FLOPs ratio 2.0002, an upper bound on wall):
- conv autoencoder: training 12 x 2,521 s, predict 8 x 4,105 s, metrics 16 x 2,108 s -> 2 x 8.40 + 2 x 9.12 + 9.37 = 44.4 CPU-h.
- conv + attention: 12 x 1,772 s, 8 x 4,097 s, 16 x 1,988 s -> 2 x 5.91 + 2 x 9.10 + 8.84 = 38.9 CPU-h.
- U-Net: 12 x 2,660 s, 8 x 4,535 s, 16 x 2,077 s -> 2 x 8.87 + 2 x 10.08 + 9.23 = 47.1 CPU-h.
- Per arm: 3 seeds x (44.4 + 38.9 + 47.1) = 391 CPU-h. Two arms: 18 runs, 782 CPU-h. The reference costs nothing new. (COVERAGE.md row 21 estimated 2 x 100 CPU-h for U-Net only; it predates the per-job numbers above.)
