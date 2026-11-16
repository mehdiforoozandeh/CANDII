# Ablation 15: count transforms (`count_transforms`)

Status: specification and configs, 2026-11-16 (close-out item A8). Worktree base d65a980, branch `chunk/onion-CO-A8`. No model code changed.

## 1. Question

The ablations document (`miniCANDI architecture ablations.md`, section "Input preprocessing of counts", and the background notes "depth-as-covariate", "borzoi-soft-clip" and "pseudocount-sweep") names the candidate transforms of the count inputs. The PI ruled on 2026-11-16: all of them, on all 3 architectures (conv autoencoder, conv + attention, U-Net), base recipes, 3 seeds, 80,000 windows, no screen.

The ablation asks: does any transform impute, correct or calibrate better than the base transform, i.e. the size-factor offset x = log2((y + 1)/s)?

## 2. Arms

s = 2^(log2 depth - 24) is the size factor (`components.py`, `REF_LOG2_DEPTH = 24.0`). η is the head's per-bin output. The loss (`nb_nll`) and the head's parameters (shared Linear(d, 2) plus a per-track bias) are the same in every arm. Each config is `final/rung<r>/reference_equal.json` plus one key, written by `stage2_arms.py write --arms <the 9 arms>`.

| arm | key | input x (`CountTransform.forward`) | output μ (`CountTransform.add_back`) | input-only? | zero-count visible bin gives |
|---|---|---|---|---|---|
| reference | `count_transform: offset` | log2((y + 1)/s) | s·2^η | (base) | 24 - log2 depth (depth readable) |
| `tf_log1p` | `count_transform: log1p` | ln(1 + y/s) | s·pos(e^η - 1) | no: output link changes | 0 |
| `tf_asinh` | `count_transform: asinh` | asinh(y/s), cofactor 1 | s·pos(sinh η) | no: output link changes | 0 |
| `tf_anscombe` | `count_transform: anscombe` | 2·sqrt(y/s + 3/8) | s·pos((η/2)^2 - 3/8) | no: output link changes; the link is not monotone in η | 1.225 |
| `tf_pearson` | `count_transform: pearson` | (y - sλ_a)/sqrt(sλ_a + (sλ_a)^2/r_a), clipped at ±30 | pos(sλ_a + η·sqrt(sλ_a + (sλ_a)^2/r_a)) | no: output link changes and reads λ_a, r_a | -sqrt(sλ_a/(1 + sλ_a/r_a)) (depth readable) |
| `tf_local_ratio` | `count_transform: local_ratio` | log2((y + 1)/(λ_local + 1)), λ_local = max of window, 1 kb and 10 kb running means over visible bins | offset form, s·2^η | yes | -log2(λ_local + 1) (scales with depth) |
| `tf_quantile` | `count_transform: quantile` | per-window average-rank normal score over visible bins | offset form | yes | the tied-rank score of the zeros (weakly depth-dependent through the zero fraction) |
| `tf_raw` | `count_transform: raw` | y | pos(η) | no: the output loses the depth offset; depth reaches μ only through C' | 0 |
| `tf_soft_clip` | `count_transform: soft_clip` | z = (y/s)^(3/4); 384 + sqrt(z - 384) above 384 | offset form | yes | 0 |
| `offset_eps0.1` | `offset_eps: 0.1` | log2((y + 0.1)/s) | s·2^η (unchanged) | yes | 24 - log2 depth - 3.32 |

pos(z) = softplus(z, beta 10) + 1e-4 (`components.py` `_pos`). Every arm keeps the depth covariate in C and C' (FiLM), as the document's "Common caveat" asks.

What else each arm changes, beside the input path (the brief requires this list):
1. `tf_log1p`, `tf_asinh`, `tf_anscombe`, `tf_pearson` change the output link from η to μ. This follows the document, which pairs each transform with its own depth add-back. So these arms test the pair (input transform, output link), not the input transform alone. A clean input-only form of them needs a new key (the add-back fixed to the offset form); I did not add it, because nothing asks for it now.
2. `tf_raw` removes the depth offset from the output. Depth reaches μ only through C' in the decoder. The arm is therefore also an output-depth arm.
3. The start point differs. At initialization η = 0 (zero-initialized head), so μ_init = s (offset forms), 0.069·s (log1p, asinh), about 2.4e-3·s (anscombe), sλ_a (pearson) and 0.069 (raw). The synthetic check (section 6) showed first-step losses of 2.28 (offset), 6.54 (log1p, asinh), 13.78 (anscombe), 1.92 (pearson) and 6.74 (raw). A worse start can cost windows of the 80,000 budget.
4. `tf_pearson` reads per-track statistics λ_a (mean rate at 2^24 reads) and r_a (method-of-moments NB size) from 4 training batches (`train.py` `PEARSON_BATCHES = 4`, batch 8, i.e. 32 windows) at the start of training. A track not visible in those 32 windows keeps the defaults λ_a = 1, r_a = 10. The buffers are in the `state_dict` and are reloaded at scoring.
5. `tf_soft_clip` divides by s before the power, so it is the Borzoi squashing of the depth-normalized count, not of the raw count.

The arms not in this set:
- The ε = 0.01 point of the pseudocount sweep ("pseudocount-sweep": ε in {0.01, 0.1, 1}). It needs no code (`offset_eps: 0.01`). It is not written, because the item file and the PI's count name 9 arms. It is an open PI decision (3 x 87.2 = 262 CPU-h more).
- The "no pseudocount" arm of the sweep note (a count model with a depth offset) is `tf_raw` with an offset add-back; the code has no such key.
- An asinh cofactor sweep and a Pearson clip at ±sqrt(N): not asked.

## 3. Prior leads

1. U-Net split depth arms (`cruxvault/results/views_ablation/VIEWS_ABLATION.md`, scorer sub-directory `p16`, not `p16_sv`; base from Nibi). Base 0.444 (0.030). Depth covariate hidden on the input side: composite 0.382 (0.009), specific variance 0.401 (0.020). Hidden on the output side: 0.455 (0.013). Both sides hidden: 0.260 (0.019). In the input-hidden arm the offset transform still received the true depth (plan Log 2026-11-15T07:52Z: a zero-count bin gives x = 24 - log2 depth). Prediction: depth that reaches the encoder only through the input values is not enough; the depth covariate matters. Every arm here keeps the covariate, so the lead predicts no large loss from removing depth out of the values alone.
2. Arm smokes (`cruxvault/results/t209/ARM_SMOKE.md`; foreman worktree only): all 9 transform arms trained 2,000 windows on rungs 2 and 5 on pilot data, with no crash and no NaN. `tf_pearson` was the one outlier: its loss rose (-1.95% on rung 2, -2.78% on rung 5). The fix chunk (plan Log 2026-11-12, ASf) found no bug: it starts at a lower loss (late ratio to the base 0.986 and 0.946). Prediction: `tf_pearson` learns least per window.
3. Old vault, h32 (`CANDI/cruxvault/h32_invertibility_sets_difficulty_inverting_.md`, read-only, older CANDI): inverting a transform on the input was harder than applying one on the output (0.211 gap on the input side, about 0 on the output side), and output steering fell under every non-identity input transform. Prediction: input transforms that leave a non-linear depth effect (log1p, asinh and Anscombe near zero, where depth acts as a scale factor) cost correction more than imputation.
4. Pseudocount note (`cruxvault/wiki/pseudocount-log-ratios-carry-a-signal-level-curve.md`): a +1 pseudocount compresses low counts, so a log ratio follows the signal level. Prediction for `offset_eps0.1`: low-count bins spread out by up to log2(1.1/0.1) = 3.46 between 0 and 1 count, against 1 at ε = 1; more input noise at shallow depth, and a larger depth signal in zero bins.
5. Literature (wiki `count-distributions-for-sequencing-data.md`; archive vault `wiki/digest-normalization-assumptions-of-prior-imputation-methods.md`): asinh is the standard input transform for epigenomic counts; ENCODE Imputation Challenge entrants used arcsinh, log1p or quantile. The document line 98: only the offset makes depth an exact additive shift in η.
6. Ablation 8 (`8_depth_input.md`, input depth capped at 2^23) runs now; its result gives the direction for the transforms that remove depth from the input values (Pearson, local ratio, quantile).

## 4. Expectation (before any run)

Seed ranges of the composite (base recipes, `p16_sv`): conv autoencoder 0.045, conv + attention 0.048, U-Net 0.010.

| arm | composite | most affected term | reason |
|---|---|---|---|
| `tf_log1p`, `tf_asinh` | within the range on the conv models; down 0.01 to 0.03 on the U-Net | calibration (the output link changes) | standard transforms; depth is a scale factor near zero |
| `tf_anscombe` | down beyond the range on 2+ architectures | calibration | non-monotone link and a near-zero start (μ_init ≈ 2.4e-3·s) |
| `tf_pearson` | down beyond the range on 2+ architectures | correction | lead 2; statistics from 32 windows |
| `tf_local_ratio`, `tf_quantile` | within the range or down; correction down more than imputation | correction | depth only through the covariate; lead 1 |
| `tf_raw` | down beyond the range on all 3 | correction and calibration | unscaled inputs (counts in the thousands) and no output offset |
| `tf_soft_clip` | within the range | imputation (strong bins compressed) | the offset add-back is kept |
| `offset_eps0.1` | within the range | correction | lead 4 |

No arm is expected to beat the offset beyond the range on 2 or more architectures.

## 5. How the result answers the question

"Beyond" = the difference of seed means is larger than the larger of the two seed ranges (max minus min, 3 seeds). Judged per architecture, on `p16_sv`.

| outcome | conclusion |
|---|---|
| an arm beyond the offset on composite on 2+ architectures, no term down beyond 2 x its range | the transform replaces the offset in the design (D13), and it enters the mix-and-match step (ruling 23) |
| an arm up beyond on 1 architecture only | report per architecture; it enters the mix-and-match step for that architecture only |
| all arms within the range | the offset stays: it is the only exact additive form (document line 98), and it is the simplest |
| depth-removing arms (Pearson, local ratio, quantile) worse beyond the range | depth must reach the encoder as a value as well as a covariate; consistent with lead 1 |
| log1p / asinh / Anscombe / Pearson down, but local ratio / quantile / soft clip (offset link kept) within | the loss comes from the output link, not from the input; a follow-up arm with the offset link would settle it |
| `offset_eps0.1` beyond in either sign | the result depends on ε; the ε = 0.01 point is then needed before any log-ratio claim |

## 6. Controls and traps

- **The output link.** Five arms change the output as well as the input (section 2, list items 1 and 2). State this with every result.
- **Parameter count and the off switch.** Unchanged: every arm builds to the base count (conv autoencoder 1,042,514; conv + attention 1,038,018; U-Net 993,599; `registry.build_model`, seed 0). The keys at their defaults give the base path; `tests/test_ablation_configs.py` holds each config to the reference plus its one key.
- **Synthetic check (laptop, 2026-11-16).** All 27 configs (9 arms x 3 architectures), at small widths and a 64-bin context on `synth.py` tables: 3 training steps with finite losses, reload through `load_run`, `contract.wiring_check` passed with perturbed weights. The Pearson buffers were set for the visible tracks (e.g. λ = 2.21, 4.85, 2.59) and kept the defaults for the others. Script: `cruxvault/results/closeout_A8/tf_synth.py`.
- **References stay fixed.** As in the other ablations: `REFS_FROM = $O/eval/eic_v1/baselines_p16`, computed on native inputs.
- **Kit drift.** The reference runs used kit CM; the arms run on a newer kit. Control: `python tools/golden.py check` on the base config with the new kit before submission.
- **Held-out loss.** Its value depends on the output link, so it is not comparable across arms. Use it only within an arm.
- **Pearson coverage.** Read λ_a and r_a from each Pearson checkpoint and report how many of the tracks kept the defaults.

## 7. Readiness

| architecture | configs (9 each, with `arms.tsv`) | parameters |
|---|---|---|
| conv autoencoder | `tools/minicandi/configs/ablations/count_transforms/rung2/<arm>.json` | 1,042,514 |
| conv + attention | `.../count_transforms/rung3/<arm>.json` | 1,038,018 |
| U-Net | `.../count_transforms/rung4/<arm>.json` | 993,599 |

Smoke: the 9 U-Net configs (the never-run combination of transforms, U-Net skips and `eic_v1` tables), 400 windows, seed 0, Fir CPU, kit COA8, run root `/project/def-maxwl/mforooz/ONION/smoke_closeout_A8` (results in section 8).

Submission: `ABLATIONS=count_transforms bash tools/minicandi/configs/ablations/submit_all.sh` on a Fir login node, with `K` set to the merged kit. Training time 03:00:00 (the default branch of the script).

Cost (per-seed figures from `1_control_input.md` section 7, Fir `sacct` CPUTimeRAW of the base jobs: training + 4 predict shards + 1 metrics job): conv autoencoder 32.5, conv + attention 22.4, U-Net 32.3 CPU-h; sum 87.2 CPU-h per seed over the three architectures. Total: 9 arms x 3 seeds x 87.2 = 2,354 CPU-h for 81 runs (486 jobs: 81 x (1 training + 4 predict + 1 metrics)). The reference needs no new run.

## 8. Smoke results

Fir jobs 4446660 and 4446662 to 4446669 (one per arm, U-Net, seed 0, `--max-windows 400`, 12 CPUs, no scoring), kit COA8, run root `/project/def-maxwl/mforooz/ONION/smoke_closeout_A8/r4_<arm>/seed0`. All 9 jobs COMPLETED in 1 min 37 s to 2 min 34 s and wrote CAPPED and DONE. Allocated CPU time (`sacct` CPUTimeRAW): 1,164 + 1,176 + 1,176 + 1,332 + 1,848 + 1,200 + 1,164 + 1,164 + 1,176 = 11,400 s = 3.2 CPU-h.

- Every logged training loss was finite (6 logged steps per arm). 400 windows is 50 steps, so the losses show only that the path runs; they say nothing about learning.
- `tf_pearson`: 11 of the 36 tracks kept the default statistics (λ = 1, r = 10) after the 32 setup windows. The measured λ ran from 0.103 to 1, and r from 0.1 (the lower clamp) to 17.8. I did not check whether the 11 tracks ever appear as training inputs. If they do, the Pearson arm uses wrong statistics for them; read the checkpoints of the real runs (section 6).
