# Plan: the miniCANDI recipe run (task t370, branch `exp/t370-minicandi-recipe`)

Written by the planner on 2026-11-21. Horizon: 2026-12-01. First measured screen result due: 2026-11-24.
Binding inputs: `.longrun/GOALS.md` (rows R1 to R22), `.longrun/DECISIONS.md` (D1 to D20), `.longrun/STATE.md`.
Where a document and the code disagree, the code is right.

## Goal

Find one miniCANDI recipe that Pareto-dominates the three base models (conv autoencoder, conv + attention, U-Net) on imputation, correction and calibration on the tables `eic_v1_2` (win rule D1). Get ChIP-control prediction to work (D9). Report the EIC comparison (D10) and the PI's metric panel (D15). Scope: P0 and P1 rows only (D18). Priority 1 is the recipe. Priority 2 is the control prediction.

**Out of this run.** The items that D12 places after the recipe (minimum covariate set, QC metrics on the input, learned cell-type embedding, resolution) are not in this plan. Scope is the P0 and P1 rows only (D18). The blind build (task t184) is also out; see Decisions needed, item 3.

### Terms used in this plan

- **Screen**: the scoring sub-directory `p16_sv`. It scores imputation and calibration on all 256 regions and correction on the 16 pair regions (`tools/onion_eval/score.py`; settled design, section 0).
- **Confirm**: the scoring sub-directory `p256_sv`. It scores all three terms on 256 regions.
- **The three terms**: `imputation_raw`, `correction_v3_raw`, `uncertainty_raw` of `tools/onion_eval/select_rule.py arm-summary`.
- **Arm**: one config. **Arm-model**: one arm on one model. Every arm-model runs seeds 0, 1, 2.
- **Kit**: a code snapshot under `/project/def-maxwl/mforooz/onion_kit/<id>` on the cluster, pushed with `tools/onion_kit/push_kit.sh`.
- **Trunk**: the layers of a model between the per-track input lift and the per-track output head.
- **Composite**: `composite_v3`, i.e. 0.5 x imputation + 0.25 x correction + 0.25 x calibration of the clipped terms. D1 does not use it; it is reported only.
- **Copy-leak**: a path that lets the content of a hidden bin or of a missing track reach the output. `contract.wiring_check` tests that no such path exists.
- **Same view** (the suffix `_sv`): a target that has the same covariates as its input track (C' = C), scored on hidden bins only. The same-view pass gives the correction-only term.
- **Crop pairs**: the 18 pairs of one experiment at two read lengths (crop 36, crop 50) in 6 cells. The correction term `correction_v3_raw` is the gap closed on these pairs.
- **Plug-in signal**: the signal p-value computed from the predicted mean alone. The headline signal is the expectation of the p-value under the predicted Negative Binomial.
- **Scoring tier**: screen (`p16_sv`) or confirm (`p256_sv`). Two arms are compared only on one tier.

### Facts the plan rests on (each checked by the planner on 2026-11-21)

1. **Reference bar of D1, 256 regions, `eic_v1_2`** (source: `arm_summary_eic_v1_2_p256_sv.tsv` in the onion worktree, folder `cruxvault/results/eic_v1_2_base_p256/read_20261121T0520Z/`; the table and `READ.md` beside it agree; 9 of 9 `p256_sv/SCORED` and 9 of 9 `p16_sv/SCORED` exist on Fir under `runs_fir/eic_v1_2_rung{2,3,4}_base_heldout/seed{0,1,2}`). Mean (seed range):

   | model | imputation | correction | calibration |
   |---|---|---|---|
   | conv autoencoder | 0.3704 (0.0214) | 0.6636 (0.0312) | 0.5986 (0.1864) |
   | conv + attention | 0.3860 (0.0298) | 0.6750 (0.0193) | 0.6272 (0.1379) |
   | U-Net | 0.4452 (0.0380) | 0.5352 (0.0624) | 0.6460 (0.1577) |

   D1 floors (best base minus the seed range of that base): imputation 0.4452 - 0.0380 = 0.4072; correction 0.6750 - 0.0193 = 0.6557; calibration 0.6460 - 0.1577 = 0.4883. Distance of each base to the floors: the U-Net needs +0.1205 correction (0.6557 - 0.5352); conv + attention needs +0.0212 imputation (0.4072 - 0.3860); the conv autoencoder needs +0.0368 imputation (0.4072 - 0.3704). The selection tool recomputes these from the arm summary; these numbers are for orientation.
2. **Where covariates enter today.** All three models apply FiLM twice only: on the per-track input features with C, and on the per-track head features with C' (`rungs/conv_ae.py` `inj_in`, `inj_out`; `rungs/conv_attn_ae.py` `cov_in`, `cov_out`; `rungs/unet.py` `in_inj`, `out_inj`). No trunk block reads a covariate. `components.ConvBlock.forward` already takes an optional `film=(gamma, beta)` argument that no model uses.
3. **U-Net skips today** (`rungs/unet.py`). (1) Level skips: the count-tower features of each level join the decoder after a FiLM `steer[i]` whose input is the C' embedding averaged over the target tracks of the window. C does not steer them. (2) Head skip: the masked per-track input features `h_track` join each track's head at full resolution through `head_skip`, with no covariate condition of their own. `skip_source` (dna, counts, both) selects the level skips.
4. **Testbed count maps exist and are reusable.** Monotone curve: `tools/t118/ladder/fforms/form_b.py` (`FormB`, 12 knots, 24 parameters, identity at start). Conditioned CNN: `tools/t118/ladder/fforms/form_d.py` (`FormD`, 4 dilated conv layers, 32 channels, context 30 bins each side, FiLM parameters from g, identity at start). The generator is `tools/t118/ladder/model.py` class `G` (already imported by `rungs/unet.py`).
5. **Wrappers.** `registry.build_model` strips `WRAPPER_KEYS = ("two_step", "control_head")`, builds the model, then wraps (`control_head.wrap`, then `two_step.wrap`).
6. **Config key guard.** `train.check_config_keys` accepts a key only when some `.py` file under `tools/minicandi/` reads it as a string literal (subscript, `.get`, `.pop`, `.setdefault`, or `in`).
7. **The A12 key-guard failure is gone.** `tests/test_minicandi_recipe_keys.py` passed on 2026-11-21 (see "Planning checks").
8. **`tools/golden.py` covers `src/candi` only**, not `tools/minicandi`. So the plan adds a miniCANDI digest test (chunk C5), and no chunk may touch `src/`.
9. **Measured unit costs on Fir, `eic_v1_2`, allocated CPU-hours per run-seed** (sacct `CPUTimeRAW` of the 135 jobs listed for the 9 base runs in `runs_fir/SUBMITTED.tsv`; cross-check on one seed: U-Net seed 0 predict shard 4712905 = 8 CPUs x 5,883 s = 13.07 CPU-h, metrics 4712913 = 16 CPUs x 2,250 s = 10.00 CPU-h):

   | model | training (12 CPUs) | screen scoring (4 predict + 1 metrics) | confirm scoring (8 predict + 1 metrics) |
   |---|---|---|---|
   | conv autoencoder | 5.98 | 8.45 + 8.29 = 16.74 | 75.93 + 11.71 = 87.64 |
   | conv + attention | 14.34 | 8.40 + 8.16 = 16.56 | 81.43 + 11.46 = 92.89 |
   | U-Net | 7.06 | 10.78 + 8.27 = 19.05 | 100.87 + 13.26 = 114.13 |

   The confirm scoring cost is about 2.7 times the figure in the older documents (30.0 / 25.4 / 51.85; the handoff predicted 326.7 CPU-h for the 9 scorings and sacct shows 883.95). This plan uses the measured values.
10. **Control data.** `eic_v1_2` holds 88 base control rows (87 train, 1 validation) in 38 cells. The 256-region eval set has 43 leave-one-assay-out validation targets; 41 are ChIP-seq targets with an observed control; the control is among the inputs for 32 of them. Every such control except one is a training row. So a leave-control-out score on today's models is in-sample for the control.
11. **Scorer support that exists.** `score.py --lambda-source predicted_control` scores every ChIP-seq target with the model's predicted control. Foreign-covariate scoring exists (`FOREIGN_MODES`: `in|out|both` x `keep_depth|swap_depth`; `FOREIGN_COVARIATES=<mode>` in `slurm/minicandi/submit_arms.sh`). Genome-wide prediction in the challenge format exists (`tools/minicandi/predict.py`), and the EIC scorer exists (`tools/onion_eval/eic_score.py`).
12. **Avocado.** The retrained Avocado (experiment 005, Fir `/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/005_2026-07-29_avocado_cross_dataset`) trained on the 312 training plus validation experiments and was scored on blind tracks only (its `README.md`, "Design"). No training-only Avocado and no validation prediction exists. Nibi `/scratch/maxwl/epi_imputation/005/` holds only `scores` today.
13. **Blind rows are not built.** Task t184 is open; 721 of 722 rows are unbuilt; the build costs about 22,700 CPU-h and the PI runs its submit command (`closeout D4`, section 3; ready queue entry 16).
14. **The PI notes mirror is not at the path in the project CLAUDE.md.** `/project/def-maxwl/maxwl/epi_imputation/notes_mirror/` does not exist on Fir or Nibi. Sprint 10 is on GitHub (`mlibbrecht/01_2026-10-19_research`, `projects/03_2026-10-20_epi_imputation/sprints/10_2026-11-11_covariate_correction`) and on Fir (`/project/def-maxwl/maxwl/epi_imputation/s10`).
15. **Fir state.** Fair-share of `def-maxwl_cpu` is 0.164. One old job runs (`mc_final_rung4_lr3e-3_matched_s1`); no chunk touches it. `eval/eic_v1_2/{baselines,baselines_p16,evalset.json,evalset_p16.json}` exist. `/scratch/mforooz/onion/aux/eic_v1_average.h5` exists. Nibi has `tables/eic_v1_2` and no `eval/eic_v1_2`. So Fir is the cluster of this run.

### Planning checks (what the planner ran, and the result)

- `PYTHONPATH=$PWD/src:$PWD /Users/mforooz/miniforge3/envs/candii/bin/python -m pytest -q tests/test_minicandi_recipe_keys.py tests/test_minicandi_rung4.py tests/test_onion_eval_select_rule.py tests/test_minicandi_control_head.py tests/test_minicandi_two_step.py tests/test_minicandi_mix_arms.py tests/test_minicandi_submit_arms.py` from the worktree root: 250 passed in 71 s.
- `python tools/golden.py save <scratch>/ref.pt` then `python tools/golden.py check <scratch>/ref.pt` (same interpreter and `PYTHONPATH`): "OK 0 ULP", 2,353,634 parameters. `git diff --quiet origin/main -- src tools/golden.py` returned 0, so the recording equals the one of `origin/main`.
- `registry.build_model` on `tools/minicandi/configs/rung{2,3,4}/base_heldout.json`: 1,042,514 / 1,038,018 / 993,599 parameters, widths 60 / 104 / 25. These equal the settled design.
- `ssh -o BatchMode=yes fir` and `ssh -o BatchMode=yes nibi`: both worked. `sacct -j <a>-<b>` (a range) returned nothing on Fir; use a comma list.
- Not run: the full `pytest tests/` (it is a cluster job in chunk C11).

### Conventions for every chunk

- Work only in the worktree. Python: `/Users/mforooz/miniforge3/envs/candii/bin/python` with `PYTHONPATH=$PWD/src:$PWD`. Below, `$PY` means this interpreter and `T=` means the test command run from the worktree root with that `PYTHONPATH`.
- The shell is zsh. An unquoted `$VAR` does not word-split. There is no `timeout` command.
- No chunk edits `src/`, `tools/golden.py`, or any file outside its "Owns" list. No new dependency.
- Every new config key defaults to today's behavior. With the key absent, each model has the same `state_dict` keys, the same parameter count and the same output as before.
- Every new config key is read with a string literal, so `train.check_config_keys` accepts it.
- Cluster chunks: Fir only. Paths: `O=/project/def-maxwl/mforooz/ONION`, tables `$O/tables/eic_v1_2`, runs root `$O/runs_rcp` (new), logs `$O/logs`, prediction stores under `/scratch/mforooz/onion/pred_store`. Every job name starts with `rcp_`. Run directories are `rcp_<arm>_<model>` with `<model>` in `convae`, `convattn`, `unet` (the names of `mix_arms.ARCH`). Exclude nodes with the `ONION_EXCLUDE` list of `configs/base_on_version.sh`. Export comma-valued variables in the shell and pass `--export=ALL`. Never use `afterok` on a finished job. Cap any array at `%12`. Never resubmit a job id named in `cruxvault/INSTRUCTIONS/peeling_the_onion/run_state/plan.md` (Log) or `HANDOFF_NEXT_SESSION.md`. A cluster chunk submits, appends its job ids to its jobs file, and returns. The foreman waits.
- Outputs of this run go under `cruxvault/INSTRUCTIONS/t370_recipe/` (decision D25). No chunk writes under `cruxvault/INSTRUCTIONS/miniCANDI_closeout/`: it holds the PI's instructions and a hook denies writes there.
- No chunk waits inside itself. A cluster chunk submits, records ids and returns. The foreman checks the result, and a later chunk or a gate read submits what depends on it.
- `runs_rcp/SUBMITTED.tsv` is shared by chunks that run at the same time. `submit_arms.sh` appends to it under `flock` on `<runs root>/SUBMITTED.tsv.lock` (chunk C4 adds this).
- Job-id records are tracked files: `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_<chunk>.tsv` (columns: job id, job name, run, seed, phase, submitted UTC).

## Acceptance criteria

Each line is a command and its expected result. `S=cruxvault/INSTRUCTIONS/t370_recipe`.

1. `$PY -m pytest -q tests/test_minicandi_recipe_film.py tests/test_minicandi_cac_layers.py tests/test_onion_eval_pareto_select.py tests/test_minicandi_recipe_arms.py tests/test_minicandi_base_golden.py tests/test_onion_eval_control_eval.py tests/test_minicandi_control_arms.py tests/test_onion_eval_eic_val_compare.py tests/test_onion_eval_max_panel.py tests/test_minicandi_recipe_keys.py tests/test_minicandi_rung2.py tests/test_minicandi_rung3.py tests/test_minicandi_rung4.py tests/test_minicandi_tune_rung2.py tests/test_minicandi_tune_rung3.py tests/test_minicandi_tune_rung4.py tests/test_minicandi_two_step.py tests/test_minicandi_submit_arms.py` prints 0 failed.
2. `$PY tools/golden.py check <ref.pt>` prints "OK 0 ULP", with `<ref.pt>` saved on `origin/main` by `$PY tools/golden.py save <ref.pt>`.
3. `git diff --stat origin/main -- src tools/golden.py` prints nothing.
4. Full suite on Fir (job `rcp_tests`, chunk C11, log `/project/def-maxwl/mforooz/ONION/logs/rcp_tests_<job id>.out`): `ssh -o BatchMode=yes fir 'f=$(ls -t /project/def-maxwl/mforooz/ONION/logs/rcp_tests_*.out | head -1); tail -n 1 "$f" | grep -c -E "^[0-9]+ passed"; grep -c -E "[0-9]+ (failed|error)" "$f"'` prints 1 and then 0.
5. First measured screen result by 2026-11-24: `ssh -o BatchMode=yes fir 'ls /project/def-maxwl/mforooz/ONION/runs_rcp/rcp_*/seed*/p16_sv/SCORED | grep -v /rcp_probe_ | wc -l'` prints at least 57 (the 19 arm-models of chunk C12 x 3 seeds). All screen arms: at least 120 (57 of C12, 46 of C13, 17 of C27).
6. `$PY tools/onion_eval/pareto_select.py --summary <p256 arm summary of runs_fir> --summary <p256 arm summary of runs_rcp>` exits 0 and prints one overall winner (the goal is met), or exits 1 and prints the nearest arm (the goal is not met; gate G3 says what follows). Exit 3 is a failure of the run.
7. `$PY tools/onion_eval/control_eval.py summary --runs <dir of pulled control_eval.json files> --out $S/results/control_summary.tsv` exits 0, and `grep -c -E "^G5 outcome: (works|partial)" $S/results/control_read.md` prints 1. Both outcomes of gate G5 pass this criterion; `partial` makes R8 partial.
8. `test -s $S/results/eic_validation.tsv && test -s $S/results/max_panel.tsv && test -s $S/recipe.md` exits 0.
8b. `grep -c "^## Blind pass: not run" $S/recipe.md` prints 1.
9. `ls $S/analysis/unet_correction_cause.md $S/analysis/heldout_loss_vs_composite.md $S/analysis/views_nodedup_mapq0.md $S/analysis/layer_widths_sprint10_vs_ours.md` lists 4 files, and `grep -c "^## Answer" <each>` prints 1.
10. (Removed. The foreman updates `.longrun/GOALS.md`; no chunk owns it.)
11. Job hygiene: `ssh -o BatchMode=yes fir "sacct -X -n -P -u mforooz -S 2026-11-21T18:00 -o JobName" | grep -v "^rcp_" | grep -v -c "^mc_final_"` prints 0 (the only other jobs of this user are the old matched-compute trainings named `mc_final_*`, which no chunk touches).
12. Spend: `ssh -o BatchMode=yes fir "sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName,CPUTimeRAW" | awk -F'|' '$1 ~ /^rcp_/ {s += $2} END {print s / 3600}'` prints the CPU-hours used; the foreman reports it at each wave and at 6,000 (gate G7).

## Chunks

### C1 trunk FiLM placement and U-Net skip switches   wave 1

- **Goal.** Add three default-off switches to the models: FiLM in the trunk at a chosen placement (all three models), FiLM(C, C') on the U-Net level skips, and a conditioned or removed U-Net head skip.
- **Owns.** `tools/minicandi/trunk_film.py` (new), `tools/minicandi/rungs/unet.py`, `tools/minicandi/rungs/conv_ae.py`, `tools/minicandi/rungs/conv_attn_ae.py`, `tests/test_minicandi_recipe_film.py` (new).
- **Interfaces (pinned).**
  - `trunk_film.PLACEMENTS = ("ends", "encoder", "decoder", "bottleneck", "all")`.
  - `trunk_film.pooled_condition(e_in, x_avail, e_out, y_avail) -> Tensor[B, 2*d]`: `e_in`, `e_out` are `[B, A, d]` covariate embeddings; the result is the concatenation of the mean of `e_in` over available input tracks and the mean of `e_out` over target tracks (a track count of 0 gives zeros).
  - `class trunk_film.TrunkFiLM(nn.Module)`: `__init__(self, d_cond: int, channels: Sequence[int], hidden: int = 64)`; `forward(cond: Tensor[B, d_cond]) -> list[tuple[Tensor[B, ch_i, 1], Tensor[B, ch_i, 1]]]` (gamma, beta per site). One shared MLP `d_cond -> hidden -> hidden` (ReLU) and one linear head per site with weight 0 and bias (ones, zeros), so every site is the identity at the start.
  - Config key `film_placement` (rungs 2, 3, 4; default `"ends"` = today's model). The per-track input FiLM with C and the per-track head FiLM with C' stay in every value. `encoder`: every encoder ConvBlock of the count path gets `film=` from `TrunkFiLM` (U-Net: `count_tower.blocks`; not the DNA tower). `decoder`: every decoder ConvBlock (U-Net: `dec1`, `dec2`). `bottleneck`: conv autoencoder: the `mid` blocks; conv + attention and U-Net: `gamma * seq + beta` on the input of each transformer layer (loop over `self.attn.layers`; in the default value call `self.attn(...)` exactly as today). The sequence is batch-first `[B, L_b, d]`, so each `TrunkFiLM` pair `[B, ch, 1]` is transposed to `[B, 1, d]` (`gamma.transpose(1, 2)`) before it is applied. Every per-layer call keeps the padding mask of today's call: `layer(seq, src_key_padding_mask=pad)` wherever today's code passes `src_key_padding_mask=pad` (the conv + attention bottleneck, and the U-Net under `tokens`); the mask is never dropped. `all`: the three together. The condition is `pooled_condition(inj_in.embed(c_in), x_avail, inj_out.embed(c_out), y_avail)` with the model's existing injectors. `covariate_injection = tokens` with a non-default placement raises `NotApplicable`.
  - Config key `skip_film` (rung 4 only; default `"cprime"` = today's `steer` input). `"c_cprime"`: the input of `steer[i]` is the `pooled_condition` vector (width `2 * D_COV`).
  - Config key `head_skip` (rung 4 only; default `"plain"` = today). `"off"`: no `head_skip` layer and `h_track` does not join the head. `"film"`: before `head_skip`, the `D_TRACK` features of `h_track` of track a are modulated by a FiLM `G(2 * D_COV, 2 * D_TRACK, identity)` whose input is `cat(in_inj.embed(c_in)[:, a], out_inj.embed(c_out)[:, a])`; the visibility flag is not modulated.
  - A non-default `skip_film` or `head_skip` on rung 2 or 3 raises `NotApplicable`. `skip_film = c_cprime` or `head_skip != plain` with `skip_source = dna` raises `NotApplicable`.
  - Every new module is built inside `_build`, so the width refit to `param_budget` counts it.
- **Depends on.** None.
- **Serves.** R2, R3, R4, R6.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_recipe_film.py tests/test_minicandi_rung2.py tests/test_minicandi_rung3.py tests/test_minicandi_rung4.py tests/test_minicandi_tune_rung2.py tests/test_minicandi_tune_rung3.py tests/test_minicandi_tune_rung4.py tests/test_minicandi_rung_arms.py tests/test_minicandi_all_rungs.py`. The new test file must show: (1) with the keys absent or at their defaults, parameter counts 1,042,514 / 1,038,018 / 993,599 and `state_dict` keys and outputs equal to a model built from the base config; (2) each non-default value builds within 10% of `param_budget`, is the identity at the start where stated, changes the output after one optimizer step, passes `contract.wiring_check`, and is accepted by `train.check_config_keys`; (3) the `NotApplicable` cases.

### C2 pre and post covariate-correction layers (wrapper)   wave 1

- **Goal.** Add the PI's pre/post layers in the testbed form (D5): a generator g reads only covariates and outputs the parameters of a count map f. Pre: g(C) and f act on the input counts. Post: g(C') and f act on the predicted Negative Binomial.
- **Owns.** `tools/minicandi/cac_layers.py` (new), `tools/minicandi/registry.py`, `tests/test_minicandi_cac_layers.py` (new).
- **Interfaces (pinned).**
  - Config keys `cac_pre` and `cac_post`: absent or `null` (default, no layer), `"curve"` (monotone curve, `ladder.fforms.form_b.FormB`), `"cnn"` (conditioned CNN, `ladder.fforms.form_d.FormD`). Reuse these two classes and `ladder.model.G`; do not re-implement them.
  - `registry.WRAPPER_KEYS = ("two_step", "control_head", "cac_pre", "cac_post")`. Wrap order in `build_model`: model, then `control_head.wrap`, then `cac_layers.wrap` (only when one of the two keys is set), then `two_step.wrap`. So both legs of two-step pass through the layers. With both keys absent, `build_model` returns the object it returns today.
  - `cac_layers.wrap(inner: MiniCandi, config: dict) -> MiniCandi`; `class CacLayers(contract.MiniCandi)`.
  - Both sides work per track in `log1p` space with the fixed knots `KNOTS = numpy.linspace(0.0, 9.0, 12)` and `stats = {"knots_x": KNOTS, "n0": 1.0, "sigma0": 1.0}` (no fitted state).
  - Pre: `x = log1p(x_counts)`; positions that are not visible (`MaskHandler.visible(x_avail, x_bin_mask)` false) are set to 0 before f, so f cannot carry hidden-bin content; `theta = g_pre(CovariateEmbedding(c_in))` per track (the raw `batch["c_in"]`, K = `contract.k_input(config)`); `loc, _ = f(x, theta)`; the inner model receives a shallow copy of the batch with `x_counts = expm1(loc).clamp_min(0)` (float32). The CNN form gets 30 zero bins of halo on each side.
  - Post: `out = inner(batch')`; `z = log1p(out["mu"])`; `theta = g_post(CovariateEmbedding(c_out))` per track (K = `K_META`); `loc, disp = f(z, theta)`; `mu = expm1(loc)` clamped to the contract's mean range; `n = out["n"] * exp(disp)` clamped to the contract's size range. Every other output key passes through.
  - g never reads counts and never reads the track identity. Each side has its own embedding and its own g. Both layers are the identity at the start.
  - "Core FiLM off" (D7) needs no new key: the arm sets the existing `covariate_drop_fields` to the 11 `META_COLUMNS`, so the inner encoders read neutral covariates while the wrapper reads the true `batch["c_in"]` and `batch["c_out"]`. The count transform and the depth offset keep the true depth.
  - `param_count()` counts the layers. A config whose total is above 1.10 x `param_budget` raises `ValueError`.
- **Depends on.** None.
- **Serves.** R5, R2, R4.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_cac_layers.py tests/test_minicandi_two_step.py tests/test_minicandi_control_head.py tests/test_minicandi_recipe_keys.py`. The new test file must show: keys absent gives the same object type and output as today for rungs 2, 3, 4; each of `curve` and `cnn` on each side is the identity at the start (output equal to the inner model within 1e-5 relative); the copy-leak check `contract.wiring_check` passes with `cac_pre = cnn`; g's output does not change when the counts change; the layers compose with `two_step = true`; `check_config_keys` accepts both keys; the parameter rule.

### C3 Pareto selection tool   wave 1

- **Goal.** Apply D1 mechanically to arm-summary tables and print the winner per model and overall.
- **Owns.** `tools/onion_eval/pareto_select.py` (new), `tests/test_onion_eval_pareto_select.py` (new).
- **Interfaces (pinned).**
  - CLI: `python tools/onion_eval/pareto_select.py --summary <tsv> [--summary <tsv> ...] [--ref convae=eic_v1_2_rung2_base_heldout --ref convattn=eic_v1_2_rung3_base_heldout --ref unet=eic_v1_2_rung4_base_heldout] [--arm-prefix rcp_] [--min-seeds 3] [--depth-native-summary <tsv>] [--out <tsv>]`. The `--ref` values shown are the defaults. Input: the TSV of `select_rule.py arm-summary` (columns `arm`, `n_seeds`, `<term>.mean`, `<term>.range`).
  - Python: `select(rows: list[dict], refs: dict[str, str], min_seeds: int = 3) -> dict` with keys `floors`, `arms` (list of per-arm dicts), `winner_by_model`, `winner_overall`.
  - Terms: `imputation_raw`, `correction_v3_raw`, `uncertainty_raw`. For term k: `best_k` = the reference row with the highest mean; `floor_k = best_k.mean - best_k.range`. An arm is **eligible** when its mean is at least `floor_k` on all three terms. Term k is **beyond** when `arm.mean - best_k.mean > max(arm.range, best_k.range)`. An arm **wins** when it is eligible and at least one term is beyond. `z_k = (arm.mean - best_k.mean) / best_k.range`; `score = min_k z_k`. Winner of a model: the winning arm of that model with the highest score. Overall winner: the same over all models. The model of an arm is its name suffix (`_convae`, `_convattn`, `_unet`).
  - Each arm row also reports, for information only, the difference to its own model's base per term and whether it is beyond the larger of the two ranges.
  - `--depth-native-summary`: for an arm that has a row in this table (fixed-input-depth arms scored at native depth), the eligibility test uses that row.
  - Output columns: `arm`, `model`, `n_seeds`, three means, three ranges, three `z`, `eligible`, `beyond_terms`, `wins`, `score`, `class` (`PASS` = wins; `NEAR` = eligible with no beyond term, or exactly one term below its floor by less than the arm's own range; `FAIL` otherwise).
  - Exit 0: an overall winner exists. Exit 1: none. Exit 3: a reference row is missing or has fewer than `--min-seeds` seeds, or a term is undefined in a reference. An arm with fewer than `--min-seeds` seeds is listed with class `WAIT` and never wins.
- **Depends on.** None.
- **Serves.** R1, R2, R4.
- **Test.** `T= $PY -m pytest -q tests/test_onion_eval_pareto_select.py tests/test_onion_eval_select_rule.py`. The new tests use synthetic tables, and one test feeds the three reference rows of fact 1 and checks the floors 0.4072, 0.6557, 0.4883.

### C4 arm configs and submit plumbing   wave 1

- **Goal.** Generate every screen config from one table, and give the submit script the `rcp_` job names and one entry point per stage.
- **Owns.** `tools/minicandi/recipe_arms.py` (new), `tools/minicandi/configs/recipe/` (new; generated configs, `arms.tsv`, `submit_recipe.sh`), `slurm/minicandi/submit_arms.sh`, `tests/test_minicandi_recipe_arms.py` (new).
- **Interfaces (pinned).**
  - `slurm/minicandi/submit_arms.sh`: the append to `<runs root>/SUBMITTED.tsv` runs under `flock <runs root>/SUBMITTED.tsv.lock` (two submissions at the same time never interleave a line; the line itself is unchanged).
  - `slurm/minicandi/submit_arms.sh`: new variable `JOB_PREFIX` (default `mc_`; unset gives every line byte for byte as today). The job name is `${JOB_PREFIX}<run dir name>_s<seed>` with the existing suffixes. Every place that matches a live job by name uses the same prefix.
  - `recipe_arms.py`: `MODELS = {2: "convae", 3: "convattn", 4: "unet"}`; `ARMS: dict[str, dict]` with `family`, `models`, `keys`; `config(rung: int, arm: str) -> dict` = `configs/rung<r>/base_heldout.json` plus the arm's keys; CLI `write` (writes `configs/recipe/<family>/rung<r>/<arm>.json` and `arms.tsv`) and `list`. Run name: `rcp_<arm>_<model>`.
  - Screen arms (family: arm = keys):
    - `skip` (U-Net only): `skipfilm_ccp` = `{"skip_film": "c_cprime"}`; `headskip_off` = `{"head_skip": "off"}`; `headskip_film` = `{"head_skip": "film"}`; `skipfilm_ccp_headskip_film` = both.
    - `placement` (all three): `film_encoder`, `film_decoder`, `film_bottleneck`, `film_all` = `{"film_placement": <value>}`.
    - `nocov` (all three): `nocov` = `{"covariate_drop_fields": [the 11 META_COLUMNS]}`.
    - `cac` (all three): `cac_curve` = `{"cac_pre": "curve", "cac_post": "curve"}`; `cac_cnn` = both `"cnn"`; `cac_curve_corefilm_off` and `cac_cnn_corefilm_off` = the same plus the `nocov` key.
    - `mix` (per the pool `mix_arms.POOL`, keys from `mix_arms.SWITCHES`): conv autoencoder `mix_svonly`; conv + attention `mix_concat`, `mix_fixdepth`, `mix_imp4`, `mix_crps`, `mix_svonly`; U-Net `mix_avg`, `mix_pearson`, `mix_fixdepth`.
    - `probe` (U-Net only): `probe_base` = no key (the base config).
  - `configs/recipe/submit_recipe.sh` (run on a Fir login node; `DRY_RUN=1` by default): `K=<kit dir> STAGE=probe|s1a|s1b|s1c|arms SUB=p16|p256|native|native256|foreign [ARMS="<arm> ..."] [MODELS="2 3 4"] [SEEDS="0 1 2"] [FOREIGN=<mode>] bash submit_recipe.sh`. `s1a` = families `skip`, `placement`, `nocov`. `s1b` = the curve arms of family `cac` (`cac_curve`, `cac_curve_corefilm_off`) and family `mix`. `s1c` = the CNN arms of family `cac` (`cac_cnn`, `cac_cnn_corefilm_off`). `arms` = the list in `ARMS`. It calls `submit_arms.sh` with `JOB_PREFIX=""`, runs root `$O/runs_rcp`, tables `$O/tables/eic_v1_2`, and the options of `configs/base_on_version.sh` stage `runs` for `SUB=p16` (4 shards, `p16_sv`, eval set `evalset_p16.json`, `REFS_FROM=$O/eval/eic_v1_2/baselines_p16`) and of `configs/ablations/covariate_fields/submit_a12.sh` for `SUB=p256` (`SCORE_ONLY=1`, 8 shards, `p256_sv`, `evalset.json`, `REFS_FROM=$O/eval/eic_v1_2/baselines`, `SCORE_TIME=06:00:00`). `SUB=native` = `SCORE_INPUT_DEPTH=native` into `p16_sv_native_depth` (screen options). `SUB=native256` = `SCORE_INPUT_DEPTH=native` into `p256_sv_native_depth` with the `SUB=p256` options (for fixed-input-depth finalists; gate G3 reads it). `SUB=foreign` = `FOREIGN_COVARIATES=$FOREIGN` into `p16_foreign_$FOREIGN`. `TRAIN_TIME=03:00:00`, and `08:00:00` for arms with `average_input` or a `cnn` layer. It refuses a run name that does not start with `rcp_`. It prints the number of jobs and the projected CPU-hours from the unit costs of fact 9.
- **Depends on.** None. It codes against the keys pinned in C1 and C2.
- **Serves.** R1, R5, R6, R14, R20.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_recipe_arms.py tests/test_minicandi_submit_arms.py tests/test_minicandi_submit_arms_extra.py tests/test_minicandi_submit_arms_phases.py tests/test_minicandi_mix_arms.py`. The new tests show: each generated config equals its base config except the arm's keys; the files on disk equal `write`; the dry run of each stage prints only job names that start with `rcp_`; with `JOB_PREFIX` unset the dry-run lines of `submit_arms.sh` are unchanged. Note: `tests/test_minicandi_recipe_keys.py::test_every_config_on_this_branch_passes_the_guard` cannot pass in this chunk's worktree before C1 and C2 merge; it is acceptance criterion 1 at the merge of wave 1.

### C5 base-model digests for miniCANDI   wave 1

- **Goal.** Record the three base models at the pre-change tree, so every later merge proves that the base configs still give the same model. `tools/golden.py` does not cover `tools/minicandi`.
- **Owns.** `tests/test_minicandi_base_golden.py` (new), `tests/data/minicandi_base_golden.json` (new).
- **Interfaces (pinned).** The JSON holds, per rung 2, 3, 4 with `configs/rung<r>/base_heldout.json` at `context_bins` 64 and `torch.manual_seed(0)`: the parameter count, the sorted `state_dict` key list hash, the SHA-256 of the concatenated `state_dict` tensors, and the SHA-256 of `mu` and `n` on one `minicandi.synth` batch in eval mode. Record it in this chunk's worktree before any other change (the worktree is at the pre-change tree). The test rebuilds and compares.
- **Depends on.** None.
- **Serves.** R4, R19.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_base_golden.py`.

### C6 leave-control-out evaluation tool   wave 1

- **Goal.** Score the predicted ChIP control when the observed control is hidden from the inputs (D9, first step).
- **Owns.** `tools/onion_eval/control_eval.py` (new), `slurm/onion_eval/control_eval.sh` (new), `tests/test_onion_eval_control_eval.py` (new).
- **Interfaces (pinned).**
  - CLI: `python tools/onion_eval/control_eval.py run --run <run dir> --evalset <json> --out <dir> [--regions 64] [--cells c1,c2] [--device cpu] [--infer-batch 8] [--stage-dir <dir>]`. For each cell of the eval set's validation ChIP-seq targets that has an observed base control: build the leave-one-assay-out inputs, remove the control from the inputs, ask for the control at the control's own C', and predict. It reuses `registry.load_run`, the scorer's batch building, `Scorer._average_control_rate` (the training-average control, the baseline to beat) and `signal_from_counts.signal_from_nb(..., predicted_control=...)`. It never writes into the scorer's directories.
  - Output `<out>/control_eval.json` and `<out>/CONTROL_EVAL_DONE`. Keys: `n_cells`, `n_control_targets`, `n_chip_targets`, `regions_n`, `tables_sha256`, `evalset_sha256`, `per_cell[<cell>].in_sample` (false only when the cell is in the run's `control_holdout_cells`), and for each of the groups `all`, `held_out`, `in_sample`: `control_count.{crps_skill_vs_training_average, log1p_pearson, log1p_r2}` and `signal.{pval_arcsinh_pearson_pred_vs_obs, pval_arcsinh_pearson_avg_vs_obs, fc_log2_pearson_pred_vs_obs, fc_log2_pearson_avg_vs_obs}`. "pred" uses the predicted control, "avg" the training-average control, "obs" the observed control; the signal p-value and the fold change are computed from the target's observed counts in the three cases. Fold change = (pileup + 1) / (lambda + 1).
  - CLI: `python tools/onion_eval/control_eval.py summary --runs <dir> [--prefix p] --out <tsv>`: per arm, `n_seeds` and mean and range (max minus min) of every key above.
  - `slurm/onion_eval/control_eval.sh <kit> <run dir> <evalset> <out dir>`: one CPU job (`--gres=none`, 8 CPUs, 32000M, 03:00:00), the environment block of `slurm/onion_eval/predict.sh`, skip when `CONTROL_EVAL_DONE` exists, `DRY_RUN=1` prints the command.
- **Depends on.** None.
- **Serves.** R8.
- **Test.** `T= $PY -m pytest -q tests/test_onion_eval_control_eval.py tests/test_onion_eval_score_avgctl.py`. The new tests run on the synthetic tables of `minicandi.synth`: the control never reaches the inputs (changing the control counts does not change the prediction); the keys exist; `summary` gives mean and range; the dry run of the slurm script.

### C7 analysis: why the U-Net loses correction   wave 1

- **Goal.** Answer the first analysis of D19 from existing scores and the code, with no new compute.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/analysis/unet_correction_cause.md` (new).
- **Interfaces.** The file has the sections `## Question`, `## Evidence` (every number with its file path and seed range), `## Answer`, `## What the screen arms of the skip family will show` (one predicted sign per arm of C4's `skip` family, written before any result). Sources to verify for itself: Fir `runs_fir/{rung4_base_heldout,final_rung4_mech_skip_dna_equal or the DNA-only-skip run named in settled design row 3,rung2_base_heldout,rung3_base_heldout}/seed*/p256_sv/{scores.json,pairs_per_region.npz,same_view_per_region.npz}`, the onion worktree `cruxvault/results/{region_sampling_checks,final_round,correction_metric_review}`, and `rungs/unet.py` (level skips steered by pooled C' only; head skip unconditioned; fact 3).
- **Depends on.** None.
- **Serves.** R12, R3.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/analysis/unet_correction_cause.md && grep -c "^## Answer" cruxvault/INSTRUCTIONS/t370_recipe/analysis/unet_correction_cause.md` prints 1.

### C8 analysis: why held-out loss and composite disagree   wave 1

- **Goal.** Answer the second analysis of D19 with no new compute (finding f88: 160,000 windows lowered the held-out loss and not the composite).
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/analysis/heldout_loss_vs_composite.md` (new).
- **Interfaces.** Same four sections as C7 (the fourth: `## What this means for selecting a recipe`). Sources: Fir `runs_fir/*/seed*/log.jsonl` (held-out loss parts, `main_no_control`) against `p16_sv/scores.json` of the same run-seeds; `tools/minicandi/losses.py` (what the loss averages over) against `score.py` (what each term averages over); the onion worktree `cruxvault/results/final_round/FINAL_ROUND_EQUAL.md`.
- **Depends on.** None.
- **Serves.** R12.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/analysis/heldout_loss_vs_composite.md && grep -c "^## Answer" cruxvault/INSTRUCTIONS/t370_recipe/analysis/heldout_loss_vs_composite.md` prints 1.

### C9 analysis: why the views hurt the no-dedup plus MAPQ-0 pair   wave 1

- **Goal.** Answer the third analysis of D19 with no new compute.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/analysis/views_nodedup_mapq0.md` (new).
- **Interfaces.** Same sections as C7 (the fourth: `## What this means for the recipe`). Sources: the per-knob and per-pair tables of the base runs and the views arms on Fir (`runs_fir/{abl_views_*,rung4_views_*}/seed*/p256_sv`), `score.py` (the key `correction_gap_closed_oracle_no_nodedup_side` and how pairs with a no-dedup side are treated), the onion worktree `cruxvault/results/ablations_closeout/ABLATIONS_READ.md` and `correction_metric_review/REVIEW.md`.
- **Depends on.** None.
- **Serves.** R12.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/analysis/views_nodedup_mapq0.md && grep -c "^## Answer" cruxvault/INSTRUCTIONS/t370_recipe/analysis/views_nodedup_mapq0.md` prints 1.

### C10 analysis: layer widths, the PI's sprint-10 model against ours   wave 1

- **Goal.** State the difference in layer widths between the model of the PI's sprint 10 and miniCANDI (R18).
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/analysis/layer_widths_sprint10_vs_ours.md` (new).
- **Interfaces.** Sections `## Question`, `## Evidence` (one table: layer, width in sprint 10, width in each of our three models, parameter count; each cell with its source file and line), `## Answer`. Sources: `gh api repos/mlibbrecht/01_2026-10-19_research/contents/projects/03_2026-10-20_epi_imputation/sprints/10_2026-11-11_covariate_correction` (read-only), Fir `/project/def-maxwl/maxwl/epi_imputation/s10` (read-only), and our widths from `registry.build_model` (60 / 104 / 25) and the rung constants. Send no project detail to an outside service beyond the read of that repository.
- **Depends on.** None.
- **Serves.** R18, R21.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/analysis/layer_widths_sprint10_vs_ours.md && grep -c "^## Answer" cruxvault/INSTRUCTIONS/t370_recipe/analysis/layer_widths_sprint10_vs_ours.md` prints 1.

### C11 kit, full test suite on Fir, and one probe at real scale   wave 2

- **Goal.** Prove on Fir that the merged code runs, that the base recipe on the new kit reproduces the reference, and that the unit costs hold.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C11.tsv` (new), `cruxvault/INSTRUCTIONS/t370_recipe/run_state/probe.md` (new).
- **Interfaces.** Steps: (1) `bash tools/onion_kit/push_kit.sh --cluster fir RCP1`. (2) Submit `rcp_tests`: `pytest tests/ -q` from the kit in a CPU job (8 CPUs, 32000M, 02:00:00, the environment block of `slurm/onion_eval/score.sh`). (3) `K=/project/def-maxwl/mforooz/onion_kit/RCP1 STAGE=probe SUB=p16 SEEDS=0 DRY_RUN=0 bash $K/tools/minicandi/configs/recipe/submit_recipe.sh` (run `rcp_probe_base_unet`, seed 0: 1 training, 4 predict, 1 metrics job). (4) Record the ids and return. `probe.md` is filled by the foreman with: the test result, sacct CPU-hours per phase against fact 9, and the three terms of the probe against `runs_fir/eic_v1_2_rung4_base_heldout/seed0/p16_sv`.
- **Depends on.** C1, C2, C3, C4, C5, C6 (merged).
- **Serves.** R19, R20, R4.
- **Test.** `ssh -o BatchMode=yes fir 'test -f /project/def-maxwl/mforooz/onion_kit/RCP1/tools/minicandi/trunk_film.py && grep -c "^rcp_probe_base_unet" /project/def-maxwl/mforooz/ONION/runs_rcp/SUBMITTED.tsv && sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName | grep -c "^rcp_tests"'` prints two numbers that are each at least 1 (true while the jobs run and after they end), and `wc -l < cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C11.tsv` prints at least 7.

### C12 screen, part 1: skip switches, FiLM placement, no covariates   wave 3

- **Goal.** Put the cheapest decisive arms on Fir: families `skip` (4 U-Net arms), `placement` (4 arms x 3 models), `nocov` (3 models). 19 arm-models, 57 run-seeds, 342 jobs.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C12.tsv` (new).
- **Interfaces.** `K=/project/def-maxwl/mforooz/onion_kit/RCP1 STAGE=s1a SUB=p16 bash $K/tools/minicandi/configs/recipe/submit_recipe.sh` first as a dry run (check 342 job lines, all `rcp_`), then with `DRY_RUN=0`. C12 and C13 run at the same time and both append to `runs_rcp/SUBMITTED.tsv`; the append is under `flock` (C4). Record ids and return.
- **Depends on.** C11 and gate G0.
- **Serves.** R1, R2, R3, R4, R6, R14.
- **Test.** `ssh -o BatchMode=yes fir 'cut -f1,2 /project/def-maxwl/mforooz/ONION/runs_rcp/SUBMITTED.tsv | sort -u | grep -c -E "^rcp_(skipfilm|headskip|film_|nocov)"'` prints 57.

### C13 screen, part 2: curve pre/post layers, mix-and-match switches, one CNN seed   wave 3

- **Goal.** Submit the curve arms of family `cac` (2 arms x 3 models) and family `mix` (9 arm-models): 15 arm-models, 45 run-seeds, 270 jobs. Submit the native-depth screen scoring of the two fixed-depth arms (6 run-seeds, 30 jobs; the script chains them to the live trainings). Submit one seed of one CNN arm as the cost probe of the CNN form.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C13.tsv` (new).
- **Interfaces.** Check that `/scratch/mforooz/onion/aux/eic_v1_average.h5` exists. Then, each as a dry run first: `STAGE=s1b SUB=p16`; `STAGE=arms ARMS="mix_fixdepth" MODELS="3 4" SUB=native`; `STAGE=arms ARMS=cac_cnn MODELS=4 SEEDS=0 SUB=p16` (run `rcp_cac_cnn_unet`, seed 0, `TRAIN_TIME=08:00:00`). C13 and C12 append to `runs_rcp/SUBMITTED.tsv` under `flock` (C4). Record ids and return. The chunk does not wait for any job. The foreman reads the CNN seed's training log at 200 steps or later and applies gate G0b; chunk C27 submits the other CNN run-seeds.
- **Depends on.** C11 and gate G0.
- **Serves.** R1, R2, R4, R5.
- **Test.** `ssh -o BatchMode=yes fir 'cut -f1,2 /project/def-maxwl/mforooz/ONION/runs_rcp/SUBMITTED.tsv | sort -u | grep -c -E "^rcp_(cac_|mix_)"'` prints 46 (45 + the one CNN seed).

### C14 first control read, probe on one base run   wave 3

- **Goal.** Submit the leave-control-out evaluation of one existing base run, to measure its cost. The control of each cell was a training target, so reads on the base runs are in-sample and are labelled so.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C14.tsv` (new).
- **Interfaces.** One job `rcp_ctleval_eic_v1_2_rung4_base_heldout_s0`: `sbatch --job-name=<that name> --output=/project/def-maxwl/mforooz/ONION/logs/%x_%j.out $K/slurm/onion_eval/control_eval.sh $K /project/def-maxwl/mforooz/ONION/runs_fir/eic_v1_2_rung4_base_heldout/seed0 /project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset.json /project/def-maxwl/mforooz/ONION/runs_rcp/control_eval/eic_v1_2_rung4_base_heldout/seed0` with `K` = kit RCP1. Outputs go under `/project/def-maxwl/mforooz/ONION/runs_rcp/control_eval/<run name>/seed<s>/`, never into `runs_fir`. Record the id and return. The foreman reads its CPU-hours from sacct; chunk C28 submits the other 8.
- **Depends on.** C11 and gate G0.
- **Serves.** R8.
- **Test.** `ssh -o BatchMode=yes fir 'sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName | grep -c "^rcp_ctleval_eic_v1_2_rung4_base_heldout_s0"'` prints at least 1.

### C15 control training switches   wave 3

- **Goal.** Add the switches of the control arms (D9, second step): hold out the controls of chosen cells from training, weight the control loss, and give the control its own head.
- **Owns.** `tools/minicandi/dataset.py`, `tools/minicandi/losses.py`, `tools/minicandi/rungs/unet.py`, `tools/minicandi/rungs/conv_ae.py`, `tools/minicandi/rungs/conv_attn_ae.py`, `tools/minicandi/recipe_arms.py`, `tools/minicandi/configs/recipe/control/` (new), `tools/minicandi/configs/recipe/arms.tsv`, `tests/test_minicandi_recipe_arms.py`, `tests/test_minicandi_control_arms.py` (new).
- **Interfaces (pinned).**
  - `control_holdout_cells` (dataset; list of cell ids; default absent = today): in training mode, the control rows of these cells are never an input and never a target. Scoring and `control_eval.py` are unchanged (they read the run's config only to label `in_sample`).
  - `control_loss_weight` (losses; float; default 1.0 = today, bit for bit): multiplies the per-bin loss of track `contract.CONTROL_INDEX` before the mean. `main_no_control` is unchanged.
  - `control_branch` (rungs 2, 3, 4; default `"shared"` = today): `"separate"` gives track 35 its own copy of the per-track output MLP and NB head (same shapes as the shared ones), fed by the same head features; tracks 0 to 34 keep the shared head.
  - `recipe_arms.CONTROL_HOLDOUT`: the cells chosen by this rule, computed once from `tables/eic_v1_2/measurements.tsv` and the eval set and then written as a literal list: sort the cells that have at least one validation leave-one-assay-out ChIP-seq target whose control is a training row; take every third (`[::3]`). The planner counted 24 such cells, so the list has 8.
  - Family `control` (all three models): `ctl_holdout` = `{"control_holdout_cells": CONTROL_HOLDOUT}`; `ctl_holdout_w4` = that plus `{"control_loss_weight": 4.0}`; `ctl_holdout_separate` = that plus `{"control_branch": "separate"}`.
- **Depends on.** C1, C4 (merged in wave 1). It edits files that C1 and C4 owned in wave 1.
- **Serves.** R8.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_control_arms.py tests/test_minicandi_base_golden.py tests/test_minicandi_recipe_film.py tests/test_minicandi_recipe_arms.py tests/test_minicandi_dataset.py tests/test_minicandi_losses.py tests/test_minicandi_recipe_keys.py`. The new tests show: defaults leave batches, loss and models unchanged; a held-out cell's control never appears in `x_avail` or `y_avail` in training mode over 200 batches; weight 4 multiplies the control part by 4; `separate` builds within 10% of the budget and changes only track 35.

### C16 EIC validation comparison tool   wave 3

- **Goal.** One command that turns a run's genome-wide validation predictions into the EIC scorer's table beside the reference methods (D10, R10).
- **Owns.** `tools/onion_eval/eic_val_compare.py` (new), `slurm/onion_eval/eic_val.sh` (new), `slurm/onion_eval/avocado_train_only.sh` (new), `tests/test_onion_eval_eic_val_compare.py` (new).
- **Interfaces (pinned).**
  - `slurm/onion_eval/eic_val.sh <kit> <run dir> <out dir>`: a CPU array over the validation cells (`--array=0-<n-1>%12`, `predict.py --split val --cell-index $SLURM_ARRAY_TASK_ID --tables $O/tables/eic_v1_2 --evalset $O/eval/eic_v1_2/evalset.json`); the variable `EIC_VAL_CHROMS` (for example `chr21`, the cost probe) is passed to `predict.py` as `--chroms $EIC_VAL_CHROMS`, and unset means every chromosome (the script defines this variable; no other code reads it); `DRY_RUN=1` prints.
  - `python tools/onion_eval/eic_val_compare.py score --pred <out dir>/signal --name <entry> --truth /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data --cache <dir> --out <tsv>`: scores one entry with `eic_score` (the nine measures, per track). `average --train /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/training_data --out <dir>`: builds the Average baseline of the training tracks only (per assay, as the challenge defines it) in the same format. `table --scores <tsv> [...] --out <tsv>`: one row per entry and measure: mean over the validation tracks, and the number of tracks on which the entry beats each other entry.
  - `slurm/onion_eval/avocado_train_only.sh` (`DRY_RUN=1` by default): the ready command for a training-only Avocado. It writes the list of training-only experiments (the files of `DATA_EIC_SYNAPSE/training_data`), and prints the `sbatch` lines that run the training and prediction scripts of experiment 005 (`scripts/hpc_train.slurm`, `scripts/hpc_predict.slurm` under `/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/005_2026-07-29_avocado_cross_dataset`) on that list with job names `rcp_avocado_*` and predictions for the validation tracks. The chunk reads those scripts and pins their real arguments. No chunk runs it with `DRY_RUN=0`; the foreman does after a PI yes (Decisions needed, item 2).
  - Entries: miniCANDI (expectation signal and plug-in signal), Average (training only), and Avocado only when the training-only Avocado has been run. The chunk verifies for itself which validation tracks have truth files (the planner counted 46 files in `validation_data`).
- **Depends on.** None (it uses `predict.py` and `eic_score.py` as they are).
- **Serves.** R9, R10.
- **Test.** `T= $PY -m pytest -q tests/test_onion_eval_eic_val_compare.py tests/test_onion_eval_eic_score.py tests/test_minicandi_predict.py`.

### C17 the PI's metric panel tool   wave 3

- **Goal.** One command that builds the panel of D15 from score files: R2 on test tracks, the gain from covariates, and the gain from real against wrong covariates.
- **Owns.** `tools/onion_eval/max_panel.py` (new), `tests/test_onion_eval_max_panel.py` (new).
- **Interfaces (pinned).** `python tools/onion_eval/max_panel.py --recipe <runs dir>/<arm> --nocov <runs dir>/<arm> --foreign-sub p16_foreign_out_keep_depth [--foreign-sub ...] --sub p256_sv --out <tsv>`. Rows: (1) R2 per validation leave-one-assay-out target = 1 - SSE / SST in `log1p` counts at 25 bp over the eval regions, mean over targets, with the seed range, read from the per-target sufficient statistics of `suffstats.npz` (the chunk verifies that they hold SSE and SST; if not, it reports which statistic is missing and uses the scorer's per-target `r_spec` squared, labelled so); (2) recipe minus no-covariates arm on R2 and on the three terms, with the larger seed range; (3) true covariates minus foreign covariates on the same quantities for each foreign mode, with `n_changed_by_column` from the score file. Every row carries `n_seeds` and the sub-directory it came from.
- **Depends on.** None.
- **Serves.** R11, R13, R14, R15.
- **Test.** `T= $PY -m pytest -q tests/test_onion_eval_max_panel.py tests/test_onion_eval_foreign_columns.py`.

### C18 control arms on Fir   wave 4

- **Goal.** Train and score the three control arms on the three models with held-out controls, and evaluate their control prediction.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C18.tsv` (new).
- **Interfaces.** (1) `bash tools/onion_kit/push_kit.sh --cluster fir RCP2`. (2) `K=.../RCP2 STAGE=arms ARMS="ctl_holdout ctl_holdout_w4 ctl_holdout_separate" SUB=p16` (dry run, then real): 9 arm-models, 27 run-seeds. (3) For each run-seed, one `rcp_ctleval_<run>_s<seed>` job submitted with `--dependency=afterok:<its live training job id>`. Record ids and return.
- **Depends on.** C15, C6, C14 (its measured cost), gate G0.
- **Serves.** R8.
- **Test.** `ssh -o BatchMode=yes fir 'cut -f1,2 /project/def-maxwl/mforooz/ONION/runs_rcp/SUBMITTED.tsv | sort -u | grep -c "^rcp_ctl_"'` prints 27.

### C19 EIC validation probe and Average baseline   wave 4

- **Goal.** Measure the cost of one genome-wide validation prediction and build the Average entry.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C19.tsv` (new).
- **Interfaces.** This chunk pushes its own kit, `bash tools/onion_kit/push_kit.sh --cluster fir RCP2e` (C18 pushes RCP2 in the same wave; two chunks never push one kit id). Then: (1) `EIC_VAL_CHROMS=chr21 sbatch --job-name=rcp_eicval_probe ... eic_val.sh $K $O/runs_fir/eic_v1_2_rung4_base_heldout/seed0 $O/runs_rcp/eic_val/probe`. (2) One CPU job `rcp_eic_average` that runs `eic_val_compare.py average` and then `score` for the Average entry. Record ids and return. The foreman then writes the projected CPU-hours of a full validation prediction (chr21 share of the genome by bins) into `jobs_C19.tsv`.
- **Depends on.** C16.
- **Serves.** R9, R10.
- **Test.** `ssh -o BatchMode=yes fir 'sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName | grep -c -E "^rcp_(eicval_probe|eic_average)"'` prints at least 2.

### C20 screen read, combination arms, and the first confirm   wave 5

- **Goal.** Read the screen with the selection tool, apply gate G1, generate the combination arms, and submit them and the 256-region scoring of the finalists.
- **Owns.** `tools/minicandi/recipe_arms.py`, `tools/minicandi/configs/recipe/combo/` (new), `tools/minicandi/configs/recipe/cac_sides/` (new), `tools/minicandi/configs/recipe/arms.tsv`, `tests/test_minicandi_recipe_arms.py`, `cruxvault/INSTRUCTIONS/t370_recipe/results/screen_read.md` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/screen_p16.tsv` (new), `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C20.tsv` (new).
- **Interfaces.** (1) On Fir: `select_rule.py arm-summary --runs $O/runs_rcp --prefix rcp_ --sub p16_sv`, the same for `--runs $O/runs_fir --prefix eic_v1_2_rung --sub p16_sv`, and for `p16_sv_native_depth`; copy the three small tables into `results/`. (2) `pareto_select.py` on them; write `screen_read.md`: per family and model the class, the three terms with ranges, and the outcome of each rule of gate G1, with the combination list and the finalist list that the rules give. (3) Add family `combo` to `recipe_arms.py` exactly as G1 lists it (arm name `combo_<a>+<b>[+<c>]`, keys = the union of the members' keys); regenerate; test. (4) When rule 4 of G1 asks for them, add family `cac_sides` (folder `configs/recipe/cac_sides/`): `cac_pre_<form>` = `{"cac_pre": <form>}` and `cac_post_<form>` = `{"cac_post": <form>}`, with `<form>` the leader's form, on the one model named by rule 4. (5) Push kit RCP3; submit the combination arms and the `cac_sides` arms (`SUB=p16`), the finalists (`SUB=p256`, score only), and for each fixed-input-depth finalist also `SUB=native256`. Record ids and return.
- **Depends on.** C12, C13, C27 (all 120 run-seeds `SCORED`, or the missing ones listed as failed), C3, C4.
- **Serves.** R1, R2, R3, R4, R5, R6.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_recipe_arms.py tests/test_minicandi_recipe_keys.py` and `test -s cruxvault/INSTRUCTIONS/t370_recipe/results/screen_read.md`.

### C21 control read   wave 5

- **Goal.** Read the control arms, apply gate G5, and name the control setting of the recipe.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/results/control_read.md` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/control_summary.tsv` (new).
- **Interfaces.** Pull the `control_eval.json` files (small) with `rsync -e 'ssh -o BatchMode=yes'`; run `control_eval.py summary`; run `pareto_select.py` on the `p16_sv` summary of the `rcp_ctl_*` arms; write the read: the in-sample values of C14 beside the held-out values, each rule of G5 and its outcome, and one line `G5 outcome: works` or `G5 outcome: partial`.
- **Depends on.** C14, C28, C18.
- **Serves.** R8.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/results/control_summary.tsv && grep -c "^## Gate G5" cruxvault/INSTRUCTIONS/t370_recipe/results/control_read.md` prints 1.

### C22 combination read, two-step arms, second confirm   wave 6

- **Goal.** Read the combination arms and the first confirm, apply gates G2 and G4, and submit two-step on the winners (D8) and the 256-region scoring of the combination finalists.
- **Owns.** `tools/minicandi/recipe_arms.py`, `tools/minicandi/configs/recipe/twostep/` (new), `tools/minicandi/configs/recipe/arms.tsv`, `tests/test_minicandi_recipe_arms.py`, `cruxvault/INSTRUCTIONS/t370_recipe/results/combo_read.md` (new), `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C22.tsv` (new).
- **Interfaces.** Family `twostep`: for the screen winner W of each model (after combinations), `twostep_native_<W>` = W's keys plus `{"two_step": true, "base_depth": "native"}` and `twostep_fixed_<W>` = W's keys plus every key in which `configs/ablations/two_step/rung<r>/two_step_fixed.json` differs from `two_step_native.json` of the same rung, copied from that file. This includes `base_ref_log2_depth`: the code default 23.0 is wrong for this use, so the value is never left to the default. The chunk reads the two files and a test asserts that each `twostep_fixed_*` config holds the same `base_ref_log2_depth` as its rung's `two_step_fixed.json`. The read `combo_read.md` also covers the `cac_sides` arms of C20. Training time 06:00:00. Push kit RCP4; submit two-step (`SUB=p16`) and the combination finalists of G2 (`SUB=p256`, and `SUB=native256` for a finalist with fixed input depth).
- **Depends on.** C20.
- **Serves.** R1, R2, R4, R7.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_recipe_arms.py tests/test_minicandi_two_step.py tests/test_minicandi_recipe_keys.py` and `test -s cruxvault/INSTRUCTIONS/t370_recipe/results/combo_read.md`.

### C23 EIC validation run on the finalists   wave 6

- **Goal.** Predict the validation tracks genome-wide for the finalists and score them with the EIC scorer.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C23.tsv` (new).
- **Interfaces.** For seed 0 of each finalist of G2 (at most 3 runs) and seed 0 of the three base runs: `sbatch --job-name=rcp_eicval_<run> ... eic_val.sh`, then one `rcp_eicscore_<run>` job with `--dependency=afterok` on the live array. If the projected cost of C19 is above 150 CPU-h per run, submit the finalists and the U-Net base only, and report.
- **Depends on.** C19, C20.
- **Serves.** R9, R10, R11.
- **Test.** `ssh -o BatchMode=yes fir 'sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName | grep -c -E "^rcp_eicval_(rcp_|eic_v1_2_)"'` prints at least 2.

### C24 final recipe runs   wave 7

- **Goal.** Freeze the recipe by gate G3 and run what the final tables need.
- **Owns.** `tools/minicandi/recipe_arms.py`, `tools/minicandi/configs/recipe/final/` (new), `tools/minicandi/configs/recipe/arms.tsv`, `tests/test_minicandi_recipe_arms.py`, `cruxvault/INSTRUCTIONS/t370_recipe/results/confirm_read.md` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/confirm_p256.tsv` (new), `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C24.tsv` (new).
- **Interfaces.** (1) Arm summaries at `p256_sv` of `runs_fir` (prefix `eic_v1_2_rung`) and `runs_rcp`, and at `p256_sv_native_depth` of `runs_rcp`; `pareto_select.py --depth-native-summary <the native-depth table>`; write `confirm_read.md` with each rule of G3 and its outcome. Read the two-step arms on `p16_sv` and apply G4; when G4 names a replacement, submit its confirm (`SUB=p256`); chunk C25 reads it. (2) Family `final`: `final_<model>` = the confirmed winner's keys of that model plus the control setting of gate G5 (only when G5 adopted one that changes training). If `final_<model>` differs from an already confirmed arm, train it (3 seeds) with `SUB=p16`, then `SUB=p256`. (3) For the overall winner and for the three base runs: `SUB=foreign FOREIGN=out_keep_depth` and `SUB=foreign FOREIGN=in_keep_depth` (screen form). (4) `SUB=p256` for the `nocov` arm of the winner's model. (5) `control_eval.sh` on the final runs. Push kit RCP5 first.
- **Depends on.** C20, C21, C22.
- **Serves.** R1, R2, R4, R7, R8, R11, R14, R15.
- **Test.** `T= $PY -m pytest -q tests/test_minicandi_recipe_arms.py tests/test_minicandi_recipe_keys.py` and `test -s cruxvault/INSTRUCTIONS/t370_recipe/results/confirm_read.md`.

### C25 final tables and the recipe document   wave 8

- **Goal.** Write the result of the run.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/recipe.md` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/eic_validation.tsv` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/max_panel.tsv` (new), `cruxvault/INSTRUCTIONS/t370_recipe/results/final_p256.tsv` (new).
- **Interfaces.** `recipe.md`: the recipe as a config table (key, value, the arm that tested it, the 256-region numbers with seed ranges, the noise floor beside every number as `AGENTS.md` section 7.2 requires); the D1 verdict from `pareto_select.py` (exit code and winner line); the control result (G5); the EIC validation table; the panel; a section `## Blind pass: not run` (the blind rows are not built; task t184, about 22,700 CPU-h, is the PI's to start; so D10's blind pass is not part of this run and R9 is partial for it); what is not known; the input to the scaling decision t363 (R22). Before it writes, the chunk reruns `pareto_select.py` on the `p256_sv` tables as they stand (this picks up a G4 replacement that C24 sent to confirm) and applies G3 and the date rule G8. `eic_validation.tsv` from `eic_val_compare.py table`; `max_panel.tsv` from `max_panel.py`. No number is typed by hand: each comes from a tool output file named beside it.
- **Depends on.** C23, C24, C17, C7, C8, C9, C10.
- **Serves.** R1, R2, R7, R8, R9, R10, R11, R13, R14, R15, R22.
- **Test.** `test -s cruxvault/INSTRUCTIONS/t370_recipe/recipe.md && test -s cruxvault/INSTRUCTIONS/t370_recipe/results/eic_validation.tsv && test -s cruxvault/INSTRUCTIONS/t370_recipe/results/max_panel.tsv && grep -c -E "^## (D1 verdict|Blind pass: not run)" cruxvault/INSTRUCTIONS/t370_recipe/recipe.md` prints 2.

### C27 screen, part 3: the CNN pre/post arms   wave 4

- **Goal.** Submit the other 17 run-seeds of the two CNN arms (`cac_cnn`, `cac_cnn_corefilm_off`, 3 models, 3 seeds, minus the seed that C13 submitted), with the training time that gate G0b set.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C27.tsv` (new).
- **Interfaces.** `K=/project/def-maxwl/mforooz/onion_kit/RCP1 STAGE=s1c SUB=p16 TRAIN_TIME=<value of G0b> bash $K/tools/minicandi/configs/recipe/submit_recipe.sh` (dry run: 17 new run-seeds, the existing one skipped; then real). Record ids and return.
- **Depends on.** C13 and gate G0b.
- **Serves.** R1, R2, R4, R5.
- **Test.** `ssh -o BatchMode=yes fir 'cut -f1,2 /project/def-maxwl/mforooz/ONION/runs_rcp/SUBMITTED.tsv | sort -u | grep -c "^rcp_cac_cnn"'` prints 18.

### C28 first control read, the other eight base runs   wave 4

- **Goal.** Submit the leave-control-out evaluation of the 8 base runs that C14 did not submit.
- **Owns.** `cruxvault/INSTRUCTIONS/t370_recipe/run_state/jobs_C28.tsv` (new).
- **Interfaces.** The `sbatch` line of C14 for each of `eic_v1_2_rung{2,3,4}_base_heldout`, seeds 0 1 2, except rung 4 seed 0; `--time` = 3 x the wall time that the foreman measured on C14's job, at least 01:00:00. Kit RCP1. Record ids and return.
- **Depends on.** C14 (its job ended; the foreman recorded its cost in `jobs_C14.tsv`).
- **Serves.** R8.
- **Test.** `ssh -o BatchMode=yes fir 'sacct -X -n -P -u mforooz -S 2026-11-21 -o JobName | sort -u | grep -c "^rcp_ctleval_eic_v1_2_rung"'` prints 9.

## Decision gates

All rules were written on 2026-11-21, before any run. "Range" is max minus min over 3 seeds. "Beyond" is a difference of means larger than the larger of the two ranges. "Own base" is the base run of the same model on `eic_v1_2`.

**G0, after the probe (C11), before any batch.**
- Pass when all hold: `rcp_tests` shows 0 failed; the probe run is `DONE` and `SCORED`; each of the three terms of the probe is inside [reference mean - reference range, reference mean + reference range] of `eic_v1_2_rung4_base_heldout` at `p16_sv`; the training cost is at most 2 x 7.06 CPU-h and the screen scoring at most 2 x 19.05 CPU-h.
- Pass: waves 3 and 4 start. Fail on tests or on the terms: one fix chunk on the files of the failing wave-1 chunk, a new kit (RCP1b), and G0 again, all within one day. Wave 1 is not restarted. The foreman logs the slip of the first-screen date (D14) in the Log and tells the PI. Fail on cost only: continue and recompute the Budget table with the measured costs.
- If the probe is still pending 12 hours after submission: the foreman reports the queue state to the PI and keeps the job. No move to Nibi without the PI (Nibi has no `eval/eic_v1_2`, and D1 needs arms and references scored on one cluster).

**G0b, the CNN cost read (the foreman, after C13).** From the training log of `rcp_cac_cnn_unet` seed 0 at 200 steps or later: projected training wall time = elapsed x 10,000 / steps done. At most 08:00:00: C27 runs with `TRAIN_TIME` = 1.5 x the projection, at least 03:00:00 and at most 12:00:00. Above 08:00:00: C27 runs only `cac_cnn` (not `cac_cnn_corefilm_off`) on the three models, and the foreman tells the PI the cost of the diagnostic arm; D7 keeps the core-FiLM-off arm on all three models unless the PI drops it. The seed that C13 submitted with 08:00:00 keeps running.

**G1, the screen read (C20), on `p16_sv` with the `p16_sv` reference rows.**
1. Class of every arm-model from `pareto_select.py` (`PASS`, `NEAR`, `FAIL`). A fixed-input-depth arm uses its native-depth score for eligibility.
2. An arm-model **qualifies** when it is better than its own base beyond the range on at least one term and worse beyond on none (the rule of `mix_arms.py qualify`).
3. Family leader per model: among the qualifying arms of a family, the one with the highest `score`. Families: `skip` (U-Net only), `placement`, `cac`, `mix`.
4. Pre/post layers: when the curve and the CNN both qualify on a model, the curve is the leader unless the CNN is beyond the curve on a term. A `corefilm_off` arm is a diagnostic: when it is within the range of its core-FiLM-on twin on all three terms, record "the layers carry the correction without core FiLM"; it becomes a leader only when it is `PASS`. When a pre/post arm qualifies, two follow-up arms (family `cac_sides`: pre only, post only, the leader's form) run on one model only: the model whose pre/post leader has the highest score. C20 generates and submits them; C22 reads them; a side-only arm replaces the two-sided leader in the recipe only when it is within the range of the leader on all three terms (the simpler arm stays).
5. Combination arms per model (at most 3): the two family leaders with the highest scores together; all family leaders together (when 3 or more qualify); and, when the `mix` family has two or more qualifying switches on the model, the best non-mix leader plus the two best mix switches. A model with fewer than 2 qualifying families gets no combination arm.
6. Finalists for the 256-region scoring (at most 2 per model, at most 6 in all): every `PASS` arm, highest score first; a model with no `PASS` arm sends its `NEAR` arm with the highest score; a model with neither sends the qualifying arm that most improves that model's weakest term (U-Net: correction; conv models: imputation), if one exists.
7. No screen number is quoted as settled (D3).
8. When no arm qualifies on any model: the foreman reports it at once with the screen table, and the next arms are chosen with the PI. The analyses C7 to C9 give the candidates.

**G2, the combination read (C22), on `p16_sv`.** A combination arm is a finalist when it is `PASS`, or when its score is higher than the score of every single-family finalist of its model. At most 1 combination finalist per model goes to the 256-region scoring. The screen winner of a model is its finalist with the highest screen score.

**G3, the confirm read (C24), on `p256_sv` with the `p256_sv` reference rows (fact 1).**
- `pareto_select.py` exit 0: the overall winner is the recipe. The winners of the other models are reported beside it.
- Exit 1: the goal is not met. `recipe.md` then gives the eligible arm with the highest score (or, when none is eligible, the arm with the highest score) as "best recipe, not dominant", with the term that fails and by how much. The foreman tells the PI at once.
- A fixed-input-depth arm is judged on its native-depth 256-region score (`p256_sv_native_depth`, submitted with `SUB=native256`, read through `--depth-native-summary`). When that score is not eligible, the arm is not deployable; the next arm takes its place and the PI is told.

**G4, two-step (C22 submits, C24 reads).** A two-step arm replaces the winner it was built on only when all hold on the scoring tier of the comparison: it is eligible; its `score` is higher; no term is lower than the winner's beyond the range. A replacement is confirmed on 256 regions before it enters the recipe: C24 submits that scoring (`SUB=p256`), C25 reads it and applies the same three conditions on `p256_sv`. A replacement whose confirm is not on disk at the date of G8 does not enter the recipe. Otherwise two-step stays off, and the read records the measured differences and the 2x FLOPs.

**G5, control (C21).** On the held-out-control cells (`held_out` group of `control_eval.json`), 3 seeds per arm-model.
- Control prediction **works** on a model when `crps_skill_vs_training_average` mean minus its range is above 0, and `pval_arcsinh_pearson_pred_vs_obs` is higher than `pval_arcsinh_pearson_avg_vs_obs` beyond the range.
- A control arm (`w4`, `separate`) is **adopted** when it raises `crps_skill_vs_training_average` over `ctl_holdout` beyond the range on at least 2 models, and it is not worse than `ctl_holdout` beyond the range on any of the three terms on any model. When both arms meet this, the one with the higher mean skill over the models wins. When none does, the shared head with weight 1 stays.
- The final recipe does not hold out any control (`control_holdout_cells` is an evaluation device); it takes only the adopted weight or head.
- The read ends with `G5 outcome: works` (the first rule holds on at least one model) or `G5 outcome: partial`.
- When control prediction does not work on any model with any arm: the foreman reports it with the in-sample values of C14 beside the held-out values (their gap is the over-fit of the control), and proposes the next control arm to the PI. R8 is then partial.

**G6, EIC validation (C23).** Report only. The table states, per measure, the number of validation tracks on which the recipe beats the Average and (when it exists) the training-only Avocado. No arm is selected on it.

**G7, budget (D20).** The foreman runs acceptance command 12 after each wave. When the sum first reaches 6,000 CPU-h, the foreman writes the checkpoint report (spend per wave, what each remaining wave costs and buys) and continues. The confirm scorings are the largest item; they are submitted only for the finalists that the gates name.

**G8, the date stop.** On 2026-11-29 the recipe freezes on what is confirmed at 256 regions by then. C24 and C25 run on that state. An arm whose 256-region scores are not complete (3 seeds) on that date is listed in `recipe.md` as "not confirmed in time" with its screen class, and it cannot be the recipe.

## Decisions needed

Only the PI can decide these. Each has a default that the run uses when no answer arrives, so nothing waits.

1. **The meaning of "beyond" in D1.** Default: the difference of means is larger than the larger of the two seed ranges (the definition of the settled design). Reason: it is the only definition already used for these scores. With it, a calibration term beats the best base only above about 0.80 (0.6460 + 0.1577), so in practice a recipe wins through imputation or correction.
2. **Avocado on validation.** The retrained Avocado trained on training plus validation tracks and has no validation prediction (fact 12). Default: the validation comparison uses the training-only Average. This does not block anything. The training-only Avocado retrain is a PI ask: C16 leaves the ready command (`slurm/onion_eval/avocado_train_only.sh`, dry run by default), and the foreman starts it only on a PI yes. Reason: its notebook gives about 86 GPU-hours for the experiment, the GPU fair-share is 0.158, and starting large compute is the PI's call. Recommended answer: yes.
3. **The blind pass.** The blind rows are not built (task t184; about 22,700 CPU-h; the PI runs the submit command; fact 13). Settled for this run: no blind pass. `recipe.md` states that it was not run and why, and R9 is partial for the blind part. Left to the PI: when to start the blind build (recommended: with the scaling run, where the final model is trained).
4. **Which control cells are held out.** Default: every third of the 24 validation cells whose control is a training row (8 cells; rule pinned in C15). Reason: it leaves 30 of 38 control cells in training and gives 8 cells of honest evaluation. The PI may want a different list.
5. **The definition of "R2 on test tracks" (the PI's note).** Default: R2 = 1 - SSE / SST per validation target in `log1p` counts at 25 bp over the 256 eval regions, mean over targets; "test tracks" means the validation tracks in this run. The same quantity in signal p-value space comes from the EIC table. Reason: it needs no new scoring. Only the PI knows the intended space and split.
6. **GPU slices for the 256-region prediction jobs.** Default: no; keep CPU prediction as in the references. Reason: the scorer documents GPU and CPU scores differing by up to 1.1e-5 and asks for one device per comparison. A yes would cut most of the 88 to 114 CPU-h per confirm run-seed and use the GPU allocation in its place.

## Budget

Unit costs: fact 9 (Fir sacct, `eic_v1_2`). Per arm-model with 3 seeds:
- Screen (train + `p16_sv`): conv autoencoder 3 x (5.98 + 16.74) = 68.16; conv + attention 3 x (14.34 + 16.56) = 92.70; U-Net 3 x (7.06 + 19.05) = 78.33; one arm on all three models = 239.19.
- Confirm (`p256_sv` only): 3 x 87.64 = 262.92; 3 x 92.89 = 278.67; 3 x 114.13 = 342.39; all three = 883.98.

| wave | item | arithmetic | CPU-h | cumulative |
|---|---|---|---|---|
| 2 | probe (C11) | 1 x (7.06 + 19.05) + tests job 8 CPUs x 0.5 h | 30.1 | 30.1 |
| 3 | screen part 1 (C12) | skip 4 x 78.33 = 313.32; placement 4 x 239.19 = 956.76; no covariates 239.19 | 1,509.3 | 1,539.4 |
| 3 | screen part 2 (C13) | curve pre/post 2 x 239.19 = 478.38; mix at base training cost 68.16 + 5 x 92.70 + 3 x 78.33 = 766.65; extra training of the mix arms at the A13 sacct figures (conv autoencoder 14.84, U-Net without the average input 17.94, U-Net with it 30.56 CPU-h per seed): 3 x (14.84 - 5.98) + 2 x 3 x (17.94 - 7.06) + 3 x (30.56 - 7.06) = 26.58 + 65.28 + 70.50 = 162.36; native-depth screen scoring 3 x 16.56 + 3 x 19.05 = 106.83; one CNN seed at the worst case, 12 CPUs x 8 h + 19.05 = 115.05 | 1,629.3 | 3,168.7 |
| 3 | control probe (C14) | 1 x 8.5 (assumed equal to one screen predict pass; this job measures it) | 8.5 | 3,177.2 |
| 4 | CNN pre/post arms (C27) | worst case, training at the 08:00:00 limit: 17 x 96 = 1,632.00; screen scoring 2 x 3 x (16.74 + 16.56 + 19.05) - 19.05 = 295.05 (the scoring cost of the CNN form is not measured) | at most 1,927.1 | 5,104.3 |
| 4 | control read, 8 base runs (C28) | 8 x 8.5 (assumed) | 68.0 | 5,172.3 |
| 4 | control arms (C18) | 3 arms x 239.19 = 717.57; control evaluation 27 x 8.5 = 229.5 | 947.1 | 6,119.4 |
| 4 | EIC probe and Average (C19) | one chromosome and one scoring job (assumed) | 10.0 | 6,129.4 |
| 5 | combination arms (C20) | at most 3 per model: 3 x 239.19 = 717.57 | at most 717.6 | 6,847.0 |
| 5 | pre-only and post-only arms (C20) | one model, 2 arms; worst case the CNN form on the U-Net: 2 x 3 x (96 + 19.05) = 690.30 (curve form on conv + attention: 2 x 92.70 = 185.40) | at most 690.3 | 7,537.3 |
| 5 | first confirm (C20) | at most 2 finalists per model: 2 x 883.98 = 1,767.96; native-depth 256-region scoring of at most 2 fixed-depth finalists (conv + attention, U-Net): 278.67 + 342.39 = 621.06 | at most 2,389.0 | 9,926.3 |
| 6 | two-step (C22) | 2 modes x [training 2 x 3 x (5.98 + 14.34 + 7.06) = 164.28; scoring 3 x 52.35 + second forward 3 x 27.63 = 239.94] | at most 808.4 | 10,734.7 |
| 6 | second confirm (C22) | at most 1 combination finalist per model | at most 884.0 | 11,618.7 |
| 6 | EIC validation (C23) | at most 6 runs; unit unknown until C19; cap 150 per run | at most 900.0 | 12,518.7 |
| 7 | final runs (C24) | final recipe, worst case all three models retrained and confirmed: 239.19 + 883.98 = 1,123.17; foreign scoring 2 modes x 12 run-seeds x 17.45 = 418.80; no-covariates confirm, one model, at most 342.39; control evaluation 9 x 8.5 = 76.50 | at most 1,960.9 | 14,479.6 |
| 7 | confirm of a two-step replacement (C24, gate G4) | worst case all three models, 256-region scoring with a second forward: 883.98 + 3 x (75.93 + 81.43 + 100.87) = 883.98 + 774.69 | at most 1,658.7 | 16,138.3 |

Reading of the table. The cumulative column takes the upper value of every line, so it is an upper bound and not a forecast. Three kinds of line are not measured: the CNN arms (costed at their time limit; gate G0b replaces this with the measured projection), the control evaluation (8.5 CPU-h assumed per run-seed), and the EIC validation prediction. With the CNN arms at their limit, the 6,000 CPU-h checkpoint (D20) falls at the end of wave 4, before any confirm scoring. If the CNN form trains at the base cost, wave 4 ends near 4,566 CPU-h (the 18 CNN trainings cost 3 x 2 x (5.98 + 14.34 + 7.06) = 164.28 in place of 18 x 96 = 1,728.00, so 6,129.4 - 1,563.7) and the checkpoint falls in wave 5. The 256-region scorings sum to at most 6,158 CPU-h (1,767.96 + 621.06 + 883.98 + 883.98 + 342.39 + 1,658.67); the gates shrink them (fewer finalists, no replacement), the date rule G8 cuts what is not confirmed by 2026-11-29, and item 6 of Decisions needed can move them to GPU slices. GPU: 0 GPU-hours unless the PI approves the Avocado retrain (about 86 GPU-hours) or item 6.

Wall-clock plan against the horizon: wave 1 and wave 2 on 2026-11-21; wave 3 submitted on 2026-11-22 after G0; wave 4 on 2026-11-22 to 2026-11-23 (it does not wait for screen scores); screen part 1 read on 2026-11-23 or 2026-11-24 (the first measured result, D14); wave 5 on 2026-11-24 to 2026-11-25; wave 6 on 2026-11-26 to 2026-11-27; the recipe freezes on 2026-11-29 (gate G8); wave 7 on 2026-11-28 to 2026-11-29; wave 8 on 2026-11-30 to 2026-12-01. The Fir fair-share (0.164) is the main risk to these dates; G0 names what the foreman does when the probe does not start.

## Log

- 2026-11-21T11:46:35Z wave 1: worktrees rcp-C1..C10 cut from 57f3c9e8; 10 builders (opus)
- 2026-11-21T11:50:04Z C5 accepted (foreman <20-line fix: platform-gated digests), merged; diff total 207
- 2026-11-21T11:50:16Z C3 accepted (rule logic read against D1/D21; 449 lines, tests not read line by line), merged; diff total 656
- 2026-11-21T11:51:52Z C7 accepted (analysis doc; answer: U-Net count skips carry a full-res copy, copy gain 0.67 vs 0.03-0.05), merged
- 2026-11-21T11:52:55Z C2 accepted (registry diff + module contract read; smaller g/f widths accepted as D28), merged; diff total ~1130
- 2026-11-21T11:55:32Z C1 accepted (base digests pass on C1+C2 merged tree: defaults bit-identical), merged
- 2026-11-21T11:55:32Z C6 accepted, merged; AMENDMENT: test commands use PYTHONPATH=$PWD/src:$PWD:$PWD/tests (score tests import e2_inbin from tests/)
- 2026-11-21T11:55:46Z C10 accepted (sprint-10 reference recipe 4,622,554 params, width 384), merged
- 2026-11-21T11:56:24Z C9 accepted (alignment views share ~96% reads with base -> covariate-gated copy; U-Net head skip copy gain 0.65), merged. CANDIDATE ARM for wave 3: pair alignment-view targets with read-disjoint halves (pr1 source / pr2 target)
- 2026-11-21T11:57:32Z C4 accepted (merged-tree tests: 202 passed incl. key guard on all 41 configs), merged; screen dry-run projection 2,155 CPU-h for 41 arm-models x 3 seeds. Wave 2 started before C8 (analysis only, touches no code) returns
- 2026-11-21T12:01:18Z C8 accepted (held-out loss dominated by read-sharing kind-2/3 bins; U-Net copy wins loss, loses disjoint-half correction), merged. Wave 1 complete.

### C29 read-disjoint training pairs   wave 2   (added by the foreman, D29)
- **Goal.** New config key `train_halves` (default absent = today's sampler, bit-identical). Value `pairs`: in training, every correction (different-view) and same-view pair reads its source from the source view's `pr1` file and its target from the target view's `pr2` file, each described at half depth (input_depth_scale = target_depth_scale = 0.5, exactly as the scored correction items do), so source and target share no reads. Imputation targets (kind 1) and context inputs keep reading `counts`. Optional `train_halves_prob` (default 1.0) applies it per pair. For same-view pairs under `pairs` the bin mask rule stays as today.
- **Owns.** tools/minicandi/dataset.py, tests/test_minicandi_train_halves.py, tools/minicandi/recipe_arms.py, tools/minicandi/configs/recipe/arms.tsv, tools/minicandi/configs/recipe/halves/ (new folder), tests/test_minicandi_recipe_arms.py.
- **Interfaces.** Key names above; arm names `halves_convae`, `halves_convattn`, `halves_unet` in arms.tsv, screen stage `s1a` (or the stage that holds the skip and placement arms), family `halves`.
- **Serves.** R1 R3 R4.
- **Test.** `PYTHONPATH=$PWD/src:$PWD:$PWD/tests python -m pytest -q tests/test_minicandi_train_halves.py tests/test_minicandi_dataset.py tests/test_minicandi_train_views.py tests/test_minicandi_recipe_arms.py tests/test_minicandi_recipe_keys.py tests/test_minicandi_base_golden.py`
- 2026-11-21T12:02:54Z C29 added to wave 2 (D29), worktree rcp-C29 cut from 345c0c7c
- 2026-11-21T12:06:49Z C29 accepted (145 tests), merged
- 2026-11-21T12:09:52Z C11 done by foreman: kit RCP1 (git 7b5dc8cd), rcp_tests 4863258, probe 4863259-4863264
- 2026-11-21T12:10:01Z wave 3 code chunks C15 C16 C17 started early (no dependency on G0); cut from 1da07953
- 2026-11-21T12:15:21Z C17 accepted (15 tests; R2 from level + Gram, verified vs direct), merged
- 2026-11-21T12:18:07Z C16 accepted (38 tests; 45 validation bigWigs, miniCANDI predicts 43), merged
- 2026-11-21T12:22:10Z C15 accepted (cell-level held-out control list C02 C10 C16 C20 C25 C31 C36 C46 per D22), merged
- 2026-11-21T12:33:41Z kit RCP2 pushed (all code through C17 + C29), used for the screen and control arms
- 2026-11-21 G0 verdict: terms pass, scoring cost pass, train cost fail (continue, x1.50 budget), tests timeout -> rerun 4888524. C12, C13, C14 submitted by foreman on RCP2 (no builder needed: submission only). C12 test 66 (57 + 9 halves), C13 test 46: pass.
