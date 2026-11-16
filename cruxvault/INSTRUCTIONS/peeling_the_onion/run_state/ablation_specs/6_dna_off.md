# Ablation 6: DNA off (`dna_off`)

Phase 1 specification, 2026-11-16. Worktree branch `chunk/onion-AB-6`, base fb57fc1.

## 1. Question

Does the DNA sequence input S add anything to imputation, correction or calibration, when the model has the same parameter budget and the same training windows? The PI's line, `miniCANDI architecture ablations.md`, section "Ablations at every rung", line 77: "DNA off." The constitution (`peeling the onion.md`, section "# miniCANDI") lists S as a model input ("S: DNA sequence for the context length W").

Relation to ablation 5 (DNA source). I did not coordinate with that agent. Ablation 5 asks which source of S is best (own tower, Flashzoi, nucleotide transformer, and the controls `dna_shuffled`, `dna_shift5mb` in `stage2_arms.ARMS`). This ablation asks whether any S is needed. `dna_off` is the zero point for every arm of ablation 5, and both use the same reference (`dna: own`). The two controls keep the tower and its parameters but break the alignment. So own minus shuffled measures aligned sequence at a fixed capacity, and shuffled minus off measures the tower without information. If `dna_off` equals own, a pretrained-source gain in ablation 5 needs the leakage check of PI ruling 26 before anyone believes it. If ablation 5 also plans a `dna_off` arm, the lead should run it once.

## 2. Arms

One arm per architecture, against the reference arm `reference_equal` (`tools/minicandi/configs/final/rung{2,3,4}/reference_equal.json`, `dna: "own"`).

| arm | change from the reference | `simpler_than_base` |
|---|---|---|
| `dna_off` | `"dna": "off"` | `true` (notepad line 60: "`true` for `dna_off`") |

The PI's line has one reading: no sequence input. The other "no sequence" variants (zeros, shuffled, shifted 5 Mb) belong to ablation 5. I did not add a fixed-width control (see section 6).

What `dna: "off"` does, read from the code and checked by `registry.build_model` at seed 0 (CPU, worktree):

| architecture | DNA parts removed | width before -> after | parameters, reference -> `dna_off` |
|---|---|---|---|
| conv autoencoder | `DnaTower` (Conv1d(4,32,15), Conv1d(32,32,1); 3,008 parameters) and the 32 DNA input channels of the 1x1 stem (32 x 60 = 1,920) | 60 -> 60 | 1,042,514 -> 1,037,586 (-4,928) |
| conv + attention | `DnaTower` (3,008) and the 32 DNA columns of the track-fusion Linear (32 x 104 = 3,328) | 104 -> 104 | 1,038,018 -> 1,031,682 (-6,336) |
| U-Net | `DnaTower`, the 5-level DNA tower `dna_tower`, the bottleneck `fuse` conv (139,477 parameters together), and the DNA half of every skip (decoder inputs and skip FiLM) | 25 -> 30 | 993,599 -> 996,232 |

- `param_budget` re-fits the width on every architecture (`conv_ae.choose_width`, `conv_attn_ae.fit_width`, `unet.fit_width`). The conv widths step by 4 and 8, so they did not move. The U-Net width steps by 1, so it grew from 25 to 30. All three counts stay within 10% of 1.0e6 (+3.8%, +3.2%, -0.4%).
- U-Net with `skip_source: both` and `dna: off`: `skip_dna = skip_source in ("dna","both") and dna != "off"` is false (`unet.py` line 183). Only the count skips remain. The docstring states the same (line 43). `skip_source: dna` with `dna: off` raises NotApplicable.
- Data path: the dataset still reads the sequence and puts `x_dna` in the batch (`dataset.py` line 406 only skips the file check). The model does not read it. The training stream, the scorer input and the evaluation set do not change.

The existing `configs/arms/rung{2,3,4}/dna_off.json` were built on `arms/rung<r>/base.json`. I compared all keys. They differ from the reference recipe in one key besides `dna` and `simpler_than_base`: `heldout_loss` (`{"batches": 16, "every": 20}`) is absent. This is not only a log key. It removes the held-out windows from the training sampler (`train.py` docstring, line 83 onward), so the training stream differs. I will not use those files. Phase 2 writes `configs/ablations/dna_off/rung{2,3,4}/dna_off.json` = reference + the two keys. `dna_off` is not in `stage2_arms.ARMS` (it is in `ladder_arms.ARMS`), so I will write the three files directly and check the diff is exactly two keys.

## 3. Prior leads

1. **Pointwise MLP, rung 1** (the only architecture with a DNA-off run). `COVERAGE.md` row 10 (foreman worktree, `cruxvault/results/ladder_audit/`): "composite +0.012 ... within". This is composite version 2. I re-read it on Nibi from `/project/def-maxwl/mforooz/ONION/runs/rung1_{base,dna_off}/seed{0,1,2}/p16/scores.json` with today's `select_rule.py arm-summary --sub p16`. Mean (range), base -> `dna_off`: composite v3 0.4736 (0.0039) -> 0.4908 (0.0232), +0.017, within; imputation 0.3431 (0.0029) -> 0.3400 (0.0035), -0.003, within (all 3 paired seeds negative, `metrics_screening/scores_rung1_arms.md`); correction (18 pairs) 0.5930 (0.0309) -> 0.6040 (0.0341), within; calibration 0.6152 (0.0327) -> 0.6792 (0.0677), within. Finding f73 (weak) records the imputation drop as "about one seed spread". Caveats: p16 is not the same-view `p16_sv` sub-directory, and the rung-1 recipe is not `reference_equal`. Prediction: at most a small imputation loss.
2. **U-Net skip mechanism**, `FINAL_ROUND_EQUAL.md` (foreman worktree, `cruxvault/results/final_round/`), mean (range): base 0.432 / 0.507 / 0.610 / composite 0.495 (0.010); skips carry counts only (`final_rung4_mech_skip_counts`: DNA reaches only the bottleneck) 0.432 / 0.518 / 0.563 / 0.486 (0.033); skips carry DNA only (`final_rung4_mech_skip_dna`) 0.380 / 0.734 / 0.640 / 0.533 (0.029). I checked on Fir that all 6 runs have `dna: own`, the `heldout_loss` key and 80,000 windows (parameters 1,003,482 and 1,002,138). Removing the DNA skips changed nothing beyond the range. The gain of the DNA-only skip comes from removing the count skips (the copy path), not from the DNA. U-Net `dna_off` is the counts-only skip arm minus the DNA at the bottleneck. Prediction: close to 0.486.
3. **Mappability lead**, `miniCANDI design notes.md` section "mappability-and-input-dna-channels" (source t146 `REPORT.md` section h31): "A one-hot DNA input gave about a quarter of the mappability gain", and that gain was local to poorly mappable bins, with no genome-wide CRPS change. The 18 correction pairs are crop and single-end pairs, which are mappability artifacts. Prediction: if DNA matters anywhere, it is on correction, and the effect is small.
4. **Archive vault and Max's notes.** I searched `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDI/cruxvault` (all md, raw excluded) and Fir `notes_mirror` for a DNA-off result. I found none. h81 (per-assay DNA prior) is an untested idea.

## 4. Expectation (before any run)

| architecture | imputation | correction | calibration | composite |
|---|---|---|---|---|
| conv autoencoder | -0.00 to -0.01, within | within | within | within |
| conv + attention | -0.00 to -0.01, within | within | within | within |
| U-Net | about 0.43, within | about 0.51, within | within | about 0.49, within |

The DNA tower in the two conv models is 0.3% of the parameters and joins once at the input. The calibration ranges (0.11 to 0.18) are too wide to resolve a small effect.

## 5. How the result answers the question

The ranges are the seed ranges of the reference (`FINAL_ROUND_EQUAL.md`) and of `dna_off`. "Beyond" = |difference of means| > larger range.

| outcome | conclusion |
|---|---|
| Composite and all three terms within, on all 3 architectures | At this scale (1e6 parameters, 80,000 windows, `eic_v1`), the sequence input gives no measurable benefit. D13 keeps the simpler arm, `dna_off`. This removes an input that the constitution lists, so the outcome goes to the PI with the scale caveat, before the recipe drops DNA. |
| Imputation lower beyond the range, on any architecture | Sequence carries information that the observed counts do not. Keep DNA. Ablation 5 (which source) is then worth its cost. |
| Correction lower beyond the range, imputation within | DNA helps the crop and single-end correction (the mappability lead). Keep DNA for correction. |
| Any term higher beyond the range with `dna_off` | The DNA path costs capacity or adds noise at a fixed budget. On U-Net, first rule out the width change (section 6). |
| Effect only on U-Net | DNA matters only with a deep DNA path (its own tower and skips). The conv models give it too small a path to judge. |

## 6. Controls and traps

- **Wrong recipe.** The old arm files lack `heldout_loss`, which changes the training stream. Control: the Phase-2 config differs from `reference_equal.json` in exactly `dna` and `simpler_than_base`, checked by a json diff.
- **U-Net width change.** The re-fit moves about 140,000 parameters from the DNA path to the count path (width 25 -> 30). A gain or loss can come from the wider count tower. A fixed-width control (`width: 25`) has 746,747 parameters, -25%, outside the 10% band, so I do not propose it. Control: compare with the counts-only skip run (lead 2), which kept DNA at width 27.
- **Initialization.** Removing modules changes the construction order, so seed k of `dna_off` does not share the reference's initial weights. Judge on ranges, not on paired differences.
- **Copy path.** The U-Net keeps its count skips, so its copy risk equals the base. Control: read `copy_leak` in each `scores.json`.
- **Scorer input.** It does not change (section 2, data path). Same scorer, `p16_sv`, same-view, as the reference runs.
- **Leakage.** None. The arm removes an input.
- **Off-switch bit-identity.** No code changes, so the reference is untouched. Phase 2 will still rebuild the three reference models and check 1,042,514 / 1,038,018 / 993,599.

## 7. Readiness

- Conv autoencoder: ready. Conv + attention: ready. U-Net: ready. `dna: off` is read on all three (`conv_ae.py` lines 150, 178-184, 245; `conv_attn_ae.py` lines 154-155, 173, 191; `unet.py` lines 165, 183, 210-215, 289), and all three build at seed 0 on the laptop.
- Configs: 3 new files, no code, no shared file. The cluster `eic_v1` genome holds the sequence, which the dataset still loads. `dna_off` itself needs no new data.
- Cost per run, read with `sacct` on Fir (train + predict + metrics, 10 jobs per seed, including one repeated prediction pass): reference runs `rung{2,3,4}_base_heldout`, 3 seeds each: 99.8, 68.1 and 99.7 CPU-h. One single-pass U-Net seed (`final_rung4_mech_skip_counts` seed 0: train 69,264 s + predict 41,504 s + metrics 35,104 s = 145,872 s) = 40.5 CPU-h.
- Total: 9 runs, about 99.8 + 68.1 + 99.7 = 268 CPU-h (23 to 41 CPU-h per run). Wall: about 1.5 h training and 1 h scoring per seed, in parallel.
