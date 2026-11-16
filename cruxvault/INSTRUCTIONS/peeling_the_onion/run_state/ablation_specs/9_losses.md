# Ablation 9 of 9: losses and regularization

Phase-1 specification, 2026-11-16. Worktree branch `chunk/onion-AB-9`, base fb57fc1.

## 1. Question

Does a change to the training loss improve one of the three goals of miniCANDI (imputation, covariate artifact correction, calibrated uncertainty) on the conv autoencoder, conv + attention and the U-Net, compared with the core loss?

The PI's lines (`peeling the onion.md`, "# miniCANDI", "objective(s)"): "core loss: NB NLL on the output predicted NB distribution -- on masked assays (imputation) -- on available assays (reconstruction/denoising)" and "ablations: any other loss/regularization/etc. that can help with our three main objectives: 1) imputation 2) covariate artifact correction 3) calibrated uncertainty estimation".

The base loss today (`tools/minicandi/losses.py`, `total_loss`): the NB negative log-likelihood (NLL) of `y_counts` under `(n, mu)`, averaged over the bins of `contract.loss_mask`. The mask holds every bin of an imputation target (kind 1) and a correction target (kind 2), and the hidden bins of a same-view target (kind 3). The weights `impute`, `correct`, `same` are all 1. The weight `base` acts only on the two-step base leg, and that leg is off in the base (`two_step: false`). No regularizer is in the loss; weight decay 0.01 and gradient clipping at norm 1.0 are recipe settings (`train.py`, `RECIPE_DEFAULTS`).

## 2. Arms

The PI's line allows three readings: (a) another per-bin loss in place of the NB NLL; (b) another weighting of the core terms; (c) a new auxiliary term (for example a covariate-pair consistency term or an adversarial covariate term). I take (a) and (b). I take no arm from (c), because no lead motivates one: the CAC test bed found that an added objective did not move correction beyond the bar (finding f30), and the adversarial CAC arms exist only as literature background with no config (`miniCANDI architecture ablations.md`, "cac-loss-arms"; audit row 79).

Reference arm: the base recipe, `final/rung{2,3,4}/reference_equal.json`, i.e. the existing runs `rung{2,3,4}_base_heldout` (3 seeds, scored in `p16_sv`). Each arm is the reference plus one key. No arm changes the model, the data path or the scorer input; each changes only `losses.total_loss`.

| arm | key = value | aimed at | what it changes |
|---|---|---|---|
| `loss_impute_x4` | `loss_weights = {impute 4, correct 1, same 1, base 1}` | imputation (and correction, see f30) | Each masked bin of an imputation target weighs 4 times a correction or same-view bin. The denominator stays the number of masked bins, so the loss scale rises. |
| `loss_bin_equal` | `loss_bin_equal = true` | imputation | The NB analogue of Max's equal-weight loss: per bin, NB NLL(n, mu detached, y) + 0.5 (y - mu)^2 / vbar, vbar = the detached mean predicted NB variance over the masked bins of one (window, track). The likelihood trains only the dispersion n; the mean gets equal weight per bin within a track. |
| `loss_poisson_mean` | `loss = poisson_mean_detached_size` | correction | The mean is trained by the Poisson NLL, and the dispersion by the NB NLL on the detached mean (the PPML form of the CAC test bed, task t127). The Poisson NLL forces the sum of mu to match the sum of y; the NB NLL matches them only with per-bin weights n / (n + mu). |
| `loss_nb_crps` | `loss = nb_crps` | calibration | The per-bin loss is the continuous ranked probability score (CRPS, a proper score of the whole predicted distribution) of NB(mu, n), exact sum over a +-10 sd window. |

Arms not taken: `head_shared_size` changes the head, not the loss, and has no lead toward a goal. `loss_multiscale`: Max's coarse-resolution losses were worse than the 25 bp loss at every scoring resolution, mostly through calibration (lead L6). `loss_assay_equal` / `loss_assay_learned` / `loss_shape_x5` balance assays or profile shape; no lead number ties them to a goal.

## 3. Prior leads

- **L1, imputation weight 4 helped correction a little (CAC test bed).** `cruxvault/f30_no_objective_moves_never_trained_crps_be.md` and `.../worktrees/t146-cac-final/cruxvault/results/t146/REPORT.md` line 30: a cross-assay imputation objective at weight 4 moved never-trained CRPS by -0.034 against a bar of 0.039, "mostly scale_error (-0.029)"; at weight 1, -0.025 against 0.067. Predicts: `loss_impute_x4` raises imputation and moves correction little or slightly better; a separate shared model, so only a weak prediction.
- **L2, Max's equal-weight loss.** `.../worktrees/onion/cruxvault/results/t235/MAX_NUMBERS.md` line 30: +0.0239 r_spec [0.0163, 0.0311] with RNA-seq input and +0.0123 [0.0042, 0.0200] without, Gaussian head, asinh-p units, two seeds each. `miniCANDI design notes.md`, "loss-weighting-and-dispersion-background": "the largest single training gain" in Max's sprints. Predicts: `loss_bin_equal` raises variance recovered by a small amount; the unit and the head differ, so the size does not transfer.
- **L3, PPML removed a scale bias.** `.../worktrees/ppml-vs-nb/cruxvault/results/t127/report.md` lines 14 to 19: in the misspecified rung A, scale_error 0.253 -> 0.005 (wobble 0.025); in law-capable forms, CRPS -0.021 (wobble 0.012) and -0.006 (0.006); NB log-likelihood |delta| <= 0.0024 nats/bin; PIT distance delta <= 0.002 (wobble about 0.01). Predicts: `loss_poisson_mean` helps correction only where the model's totals drift; calibration unchanged.
- **L4, the held-out loss is not a proxy for the goal terms.** `cruxvault/f88_3_seeds_per_recipe_at_equal_windows_winn.md`: recipes chosen on held-out NB loss did not raise the composite (conv autoencoder 0.528 -> 0.496, conv + attention 0.525 -> 0.510, U-Net 0.495 -> 0.511, all inside seed ranges). Predicts: a lower training loss need not raise any metric; read only the four metrics.
- **L5, the composite.** PI ruling 47 (`long-horizon notepad.md` line 563; `cruxvault/f68_...md`): gap closed uses the oracle floor as its top. The reference table below uses that composite.
- **L6, coarse losses.** `miniCANDI architecture ablations.md`, "loss-resolution-arm": pooled losses at 200 bp to 3.2 kb were worse than 25 bp "mostly through calibration" (single fine-tunes).
- **L7, smoke runs.** `.../worktrees/onion/cruxvault/results/t209/arm_smoke.tsv` (2,000 windows, conv autoencoder): `loss_impute_x4`, `loss_nb_crps`, `loss_poisson_mean` all trained, finite, wiring check 0.0, scored. `.../results/t248/ARM_SMOKE_3.md`: `loss_bin_equal` trained and scored; its logged loss rose (loss drop -1.93), because vbar shrinks as the model sharpens. Smokes compare nothing.
- **L8, calibration is noisy.** `.../worktrees/onion/cruxvault/results/final_round/FINAL_ROUND_EQUAL.md` and `f86`: calibration seed range 0.108 to 0.178 on the three bases; "Calibration ... ranks nothing".

Reference values (FINAL_ROUND_EQUAL.md, mean (range)): conv autoencoder 0.382 (0.028) / 0.718 (0.012) / 0.631 (0.178) / composite 0.528 (0.045); conv + attention 0.387 (0.012) / 0.673 (0.031) / 0.652 (0.157) / 0.525 (0.048); U-Net 0.432 (0.023) / 0.507 (0.071) / 0.610 (0.108) / 0.495 (0.010).

## 4. Expectation (before any run)

| arm | imputation | correction | calibration | composite |
|---|---|---|---|---|
| `loss_impute_x4` | up, +0.00 to +0.02, likely inside the range | flat to down 0.03 (fewer correction gradients) | inside the range | inside the range |
| `loss_bin_equal` | up, +0.01 to +0.03, beyond on at most 1 architecture | unknown, small | unknown; the dispersion still has its likelihood, so no large loss | inside the range |
| `loss_poisson_mean` | flat | flat to up 0.02; larger only where totals drift (U-Net has the lowest correction) | flat (L3) | inside the range |
| `loss_nb_crps` | flat to up (CRPS weighs high-count bins more) | flat | direction unknown; only an effect above about 0.18 is "beyond" (L8) | inside the range |

Overall expectation: no arm changes the composite beyond the seed range on 2 of 3 architectures. This is the null, i.e. the core NB NLL is adequate at this scale.

## 5. How the result answers the question

Per arm and architecture, an effect is "beyond" when the difference of means exceeds the larger of the two seed ranges (arm, reference).

| outcome | conclusion |
|---|---|
| The aimed-at term beyond-better on 2 or 3 architectures, and no term beyond-worse | The loss helps that goal. It is a candidate for the final design (stage-3 rule, notepad line 407: loss winners may join). |
| The aimed-at term beyond-better, and another term beyond-worse | A trade-off. Report both; the composite decides; the PI rules. |
| Composite beyond-better on 2 or 3 architectures | Adopt-candidate, whatever term carries it. |
| The effect on 1 architecture only | Architecture-specific; record, do not adopt. |
| All terms inside the ranges on all architectures | The loss choice does not matter at this scale; keep the NB NLL (the PI's core loss, the simplest). |
| `loss_nb_crps` calibration inside the range | No answer for goal 3: the test cannot resolve less than about 0.18. Say so; do not read it as "no effect". |

## 6. Controls and traps

- **Held-out loss.** `heldout_loss` in `log.jsonl` is `step_loss(model, b, cfg)` (`train.py` line 549), i.e. the arm's own loss: CRPS, Poisson + NB, a weighted NLL or the bin-equal form. It is not comparable across arms. No selection reads it: training runs to `windows_seen` 80,000, one `ckpt.pt` is saved, and the scorer reads that final checkpoint. Control: never rank arms by `heldout_loss` or `loss`; use the four metrics only (L4).
- **Loss scale and clipping.** On a synthetic batch the four arms give losses 6.68 / 2.90 / 5.42 / 1.47 against the base 2.67 (impute_x4, bin_equal, poisson_mean, nb_crps), and the mu-gradient norm 0.219 / 0.004 / 0.096 / 0.014 against the base 0.079. AdamW is nearly scale-invariant, but clipping at norm 1.0 is not: on Fir seed 0 the bases clip 18% (conv autoencoder), 19% (conv + attention) and 16% (U-Net) of logged steps (median grad norm 0.58 / 0.58 / 0.52). `loss_impute_x4` will clip more and `loss_nb_crps` less. Control: report the clipped fraction of logged steps per arm beside the metrics.
- **Reference reuse.** The reference runs were trained by an earlier kit. Control: (1) the off switch is exact (`losses.py` docstring: absent keys give `contract.nb_nll` bit for bit); Phase 2 checks it on a synthetic batch (loss and gradients equal). (2) Record the kit git sha of every arm run. A fresh base seed in the new kit costs about 26 CPU-h per architecture; the lead decides.
- **Parameter count and scorer input.** Unchanged: built at 1,042,514 / 1,038,018 / 993,599 for all four arms (section 7). Every arm emits the same `(n, mu)`; the scorer reads the same output.
- **CRPS runtime.** `nb_crps` loops over window sizes in float64. Smoke rung 2: 0.0198 s per window against 0.0165 for a plain NB-loss arm (t209 `arm_smoke.tsv`). `submit_arms.sh` gives 3 x the projected wall, so the time limit holds.
- **Bin-equal loss level.** Its logged loss can rise during training (L7). A rise is no failure; a NaN is.
- **Weights `base`.** Inert in every arm (`two_step: false`).

## 7. Readiness

All four keys are read by `losses.total_loss`, which `train.step_loss` calls for every architecture; no rung-specific path. Checked 2026-11-16: `stage2_arms.py write` wrote the three stage-2 arms and `stage3_arms.py write --n-cells 51` wrote `loss_bin_equal` on each `reference_equal.json` (status `write`); `registry.build_model` gave the base parameter count for every arm (1,042,514 / 1,038,018 / 993,599); `total_loss` on a synthetic batch was finite with finite gradients for every arm.

| architecture | state |
|---|---|
| conv autoencoder | ready, no code |
| conv + attention | ready, no code |
| U-Net | ready, no code |

Shared files Phase 2 must edit: none.

Cost per run (Fir). Training: 12 CPUs (`budgets.json`, `slurm.cpu.cpus`) x the base wall from `run.json` (`rung{2,3,4}_base_heldout/seed0`): 12 x 2,389 s = 7.96 CPU-h; 12 x 1,643 s = 5.48; 12 x 2,541 s = 8.47. Scoring, `p16_sv` (sacct CPUTimeRAW of the seed-0 jobs, 4 predict + 1 metrics): conv autoencoder 32,840 + 33,728 = 66,568 s = 18.5 CPU-h (jobs 4389803 to 4389807); conv + attention 32,776 + 31,808 = 64,584 s = 17.9 (4389818 to 4389823); U-Net 36,280 + 33,232 = 69,512 s = 19.3 (4389834 to 4389838). Per run: 26.5 / 23.4 / 27.8; per seed over the 3 architectures 77.7 CPU-h. Total: 4 arms x 3 seeds x 77.7 = 932 CPU-h for 36 runs, plus up to 20% on the `nb_crps` training (3 x 21.9 x 0.2 = 13 CPU-h): about 945 CPU-h. The reference costs nothing (existing runs).
