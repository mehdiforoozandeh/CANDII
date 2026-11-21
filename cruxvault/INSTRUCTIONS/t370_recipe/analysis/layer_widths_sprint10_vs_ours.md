# Layer widths: the model of the PI's sprint 10 against the three miniCANDI base models

Chunk C10 of the miniCANDI recipe run (task t370). Requirement served: R18 (and R21).
Written 2026-11-21.

## Question

The PI asked: "figure out the difference between the width of the layers of Sprint 10's onion and Mehdi onion".
Sprint 10 has no model of its own named "onion". "Peeling the onion" is our run (PI repo, sprint 10 `PLANNING.md` line 20; sprint 18 `report_sections/02_mehdi.md` line 7).
The model that sprint 10 trained and probed is the CANDII "reference recipe", i.e. CANDII at our commit `1391b36` with the Flashzoi DNA arm and three architecture plugins.
This note compares that model, layer by layer, with our three miniCANDI base models on `eic_v1_2`: the conv autoencoder (rung 2), conv + attention (rung 3) and the U-Net (rung 4).

## Sources

Abbreviations used in the table (call these the source keys):

- **S10**: PI repo `mlibbrecht/01_2026-10-19_research`, commit `fa70e877`, folder `projects/03_2026-10-20_epi_imputation/sprints/`. Read with a sparse, depth-1 clone in the session scratch directory. Files:
  - `10_2026-11-11_covariate_correction/scripts/candi/harness/train9.slurm` lines 47-50: the training command, `--heads count,signal,peak`, batch 8.
  - `10_2026-11-11_covariate_correction/scripts/candi/ctlaug_jobs.sh` lines 3-4, 14-19, 40: arm `flashzoi`, plugins `d3rd`, `rna`, `usig`, `ctlaug`, `dilfix`, 240,000 steps, "the reference recipe exactly" plus control augmentation (which adds no parameters).
  - `10_2026-11-11_covariate_correction/scripts/candi/harness/job_env9.sh` lines 4, 23: CANDII is sprint 02's pinned checkout `1391b36`.
  - `03_2026-10-24/scripts/model/s03_patch.py` lines 10-13, 131-142: Flashzoi adapter, `LayerNorm(1920) + Linear(1920 -> tower width)`, replaces the DNA tower.
  - `03_2026-10-24/scripts/rna/rnaplugin/rna.py` lines 16-19, 53-60: RNA module, 64 hidden channels, two kernel-5 convs (second dilated 2), gated add to the DNA-tower output.
  - `07_2026-11-04/scripts/s2a/plugins/d3rd/d3_patch.py` lines 54-55, 130-150: "design 3", i.e. attention from each 200 bp token over K = 32 reference biosamples per assay; 4 heads, dk 16, key-encoder width 32.
  - `09_2026-11-08/scripts/compute/plugins/dilfix/dilfix.py` and `usig/usig.py`: a faster dilated conv and a loss change; neither adds parameters.
- **CK**: Fir `/project/def-maxwl/maxwl/epi_imputation/s10/candi/harness/ckpt/cd_ctlaug_s1.ckpt` (2026-11-12), a sprint 10 training checkpoint. I loaded its state dict read-only and counted tensors per module.
- **C1**: our repo at `1391b36`, `src/candi/config.py` lines 22-35, 65-67 and `src/candi/model.py` (`CandiModel.__init__` defaults). I built `CandiModel(num_assays=47, heads=("count","signal","peak"))` and added the three plugin modules with their stated sizes. The rebuilt count is 4,622,554 parameters. The checkpoint has 4,622,586 state-dict elements; the difference of 32 is non-parameter buffers (the rebuilt model's state dict also has 32 more elements than its parameters). So the reconstruction matches the checkpoint exactly.
- **MC**: this worktree (base `57f3c9e8`), `tools/minicandi/`. I built each model with `registry.build_model` from `configs/rung{2,3,4}/base_heldout.json` (`param_budget` 1.0e6) and counted parameters per module. Constants: `rungs/conv_ae.py` lines 94-101, 122-125, 195-212; `rungs/conv_attn_ae.py` lines 84-94, 176-182; `rungs/unet.py` lines 98-106, 206-209, 225, 244, 259-261; `components.py` lines 539-545 (`ConvBlock`: Conv1d, GroupNorm, GELU, residual) and 560 (`Downsample`: strided conv).

## Evidence

Tracks: the sprint 10 model reads 48 tracks (47 assays of Dataset 1 plus the ChIP-seq control) and predicts 47 (CK: `d3.k_in` has 3,040 = 2 x 47 x 32 + 32 parameters; `decoder.input_proj` has 144,760 = 384 x 376 + 376 with 376 = 47 x 8). Our models read and predict 36 tracks (35 assays plus the control; MC `contract.py` line 64).

| layer | sprint 10 model (CANDII-dev + Flashzoi + design 3 + RNA) | conv autoencoder (rung 2) | conv + attention (rung 3) | U-Net (rung 4) |
|---|---|---|---|---|
| total parameters | 4,622,554 (C1; CK 4,622,586 elements) | 1,042,514 (MC) | 1,038,018 (MC) | 993,599 (MC) |
| width parameter | d_model 384 = 8 x 48 tracks, auto (C1 `config.py` line 65; CK) | W = 60, fitted to budget (MC) | W = 104, fitted to budget (MC) | width = 25, fitted to budget (MC) |
| window, resolution | 768 bins x 25 bp = 19.2 kb (C1 `config.py` line 24) | 1,024 bins x 25 bp = 25.6 kb (MC config `context_bins`) | same as rung 2 | same as rung 2 |
| per-track input features | grouped signal tower, 3 conv layers, kernel 3, pool 2 each: 2, 4, 8 channels per track (96, 192, 384 in total); 11,676 parameters (CK, C1) | lift to 16 per track (`D_TRACK`, `conv_ae.py` line 94) | lift to 16 per track (`conv_attn_ae.py` line 84) | MLP to 8 per track (`unet.py` line 101) |
| where tracks first mix | Linear fusion of signal (384) and DNA (384) features to 384 at 96 tokens; 295,296 parameters (CK) | 1x1 stem conv, 36 x 16 -> 60; 36,540 parameters (MC) | Linear fusion 36 x 16 + 32 DNA -> 104; 63,336 parameters (MC) | 1x1 stem conv, 36 x 9 -> 25; 8,125 parameters (MC) |
| DNA path | frozen Flashzoi (Borzoi) embedding, 1,920 per 200 bp token, LayerNorm + Linear -> 384; 741,504 parameters (CK; S10 `s03_patch.py`) | own one-hot DnaTower, 32 channels, kernel 15, to 32 per bin; 3,008 parameters (MC) | same tower; 3,008 parameters (MC) | same tower to 25, then its own level stack (25, 50, 50, 100, 100, as the count tower); 2,777 + 116,600 parameters (MC) |
| conv encoder channels per level | none beyond the 3 per-track layers; the sequence goes from 768 bins to 96 tokens (stride 8) in the signal tower | 60, 76, 88, 104, 120 over 4 downsamplings (multiples 1, 1.25, 1.5, 1.75, 2; `conv_ae.py` lines 122-125); 2 ConvBlocks per level, kernel 5, dilation 1 and 2; encoder 281,328 + down 66,148 parameters (MC) | 104 at every level, 4 downsamplings, 1 ConvBlock per level, kernel 5; encoder 217,568 + down 86,944 (MC) | 25, 50, 50, 100, 100 (multiples 1, 2, 2, 4, 4; `unet.py` line 99), 1 ConvBlock per level, kernel 5; count tower 116,600 (MC) |
| bottleneck | 2 transformer layers at 384, 4 heads (attention inner width 256), feed-forward 1,536, over 96 tokens of 200 bp; 3,151,872 parameters, i.e. 68% of the model (C1, CK) | 2 ConvBlocks at 120, dilation 2 and 4, at 64 positions of 400 bp; no attention; 144,720 parameters (MC) | 2 transformer layers at 104, 4 heads, feed-forward 208 (`FF_MULT` 2), over 64 positions of 400 bp; 175,344 parameters (MC) | 1x1 fuse of count and DNA (2 x 100 -> 100), then 2 transformer layers at 100, 4 heads, feed-forward 200, over 64 positions; 20,100 + 162,200 parameters (MC) |
| decoder channels | Linear 384 -> 47 x 8 = 376 lanes (144,760 parameters), then 3 per-lane transposed-conv blocks (8 channels per track, kernel 3, upsample 2 each, 96 -> 768 bins); 38,400 parameters (CK) | mirror of the encoder (120 -> 60), 2 ConvBlocks per level; up 66,088 + decoder 281,328 (MC) | mirror (104 at every level); up 86,944 + decoder 217,568 (MC) | per level: upsample, concatenate skips (count and DNA), 2 ConvBlocks (100 -> 25); up 37,725 + dec1 235,050 + dec2 78,800 (MC) |
| per-track output head | 8 per track; count (NB), signal (Gaussian) and peak heads; under 200 parameters (C1) | Linear 60 -> 36 x 16, then a small MLP and NBHead; 35,136 + 378 (MC) | Linear 104 -> 36 x 16 and NBHead; 60,480 + 106 (MC) | Linear to 32 per track (`D_HEAD`), head skip from the masked input features, NBHead; about 3,400 (MC) |
| trunk total (input lift to head, excluding DNA and covariates) | signal tower + fusion + transformer + decoder = 3,642,004 | stem through decoder = 876,152 | fusion through decoder = 847,704 | stem through dec2, with the DNA stack = 858,788 |
| how covariates enter | 4-row metadata embedding (depth, read length, assay, run type) to 32; FiLM on each of the 3 per-track signal-tower convs with C; per-lane FiLM before and after each decoder block (4 FiLM layers) with C'; 6,848-7,232 parameters per embedding (C1 torchinfo, CK) | FiLM twice only: on the per-track input features with C and on the per-track head features with C'; embedding width 32 (`D_COV`); 63,040 parameters each (MC; plan fact 2) | same as rung 2; `cov_in`, `cov_out` 63,040 each (MC) | same, plus a C' FiLM on every skip (`steer`); 62,000 + 65,120 + 83,588 (MC) |
| reference-panel path | design 3: attention from each token over 32 reference biosamples per assay, key encoder width 32 (convs kernel 5), gated into the decoder input; 174,345 parameters (C1, CK) | none | none | none |
| RNA path | RNA module, 64 channels, 2 convs kernel 5, gated add to the DNA features; 47,552 parameters (CK) | none (RNA enters as a track when present) | none | none |

## Answer

The sprint 10 model is CANDII-dev plus three add-ons, with 4,622,554 parameters (rebuilt from code `1391b36` and the sprint 10 plugins; it matches the Fir checkpoint `cd_ctlaug_s1.ckpt` exactly), which is 4.4 to 4.7 times our 0.99 M to 1.04 M base models (4,622,554 / 1,042,514 = 4.43; 4,622,554 / 993,599 = 4.65).
Its trunk is one wide layer stack: the shared width is 384 everywhere after the track fusion (d_model = 8 x 48 input tracks), against our trunk widths of 60 to 120 (conv autoencoder), 104 (conv + attention) and 25 to 100 (U-Net), so its attention layers are 384 / 104 = 3.7 times as wide as ours and their feed-forward layers 1,536 / 208 = 7.4 times as wide.
The parameter difference sits almost entirely in its transformer: 3,151,872 parameters, 68% of the model, at 96 tokens of 200 bp, against 175,344 (rung 3) and 162,200 (rung 4) at 64 positions of 400 bp, and the conv autoencoder has no attention.
Its other large blocks have no counterpart in our models: the Flashzoi adapter (741,504), the track-fusion layer (295,296), the reference-panel attention (174,345) and the RNA module (47,552).
In the opposite direction, our models spend more on the convolutional trunk at full and half resolution: the sprint 10 model has only 3 grouped per-track convs (2, 4, 8 channels per track) before the transformer and 3 per-track transposed convs (8 channels per track) after it, whereas each of our models has 4 levels of shared-channel ConvBlocks (kernel 5) on each side.
Covariates enter the sprint 10 model at 7 places (3 encoder FiLMs with C, 4 decoder FiLMs with C', all per track), against 2 places in our conv autoencoder and conv + attention models (input lift with C, head with C') and 3 kinds in the U-Net (adding the C' FiLM on the skips).

## Not checked

- I did not compute receptive fields. Our conv autoencoder has only local context (`conv_ae.py` line 5); the other four models attend over the whole window.
- I checked one sprint 10 checkpoint (`cd_ctlaug_s1`). The four sprint 09 reference-recipe runs (`sa_r2rd_s1..s4`) that sprint 10 also scored are documented as the same recipe without control augmentation (S10 `ctlaug_jobs.sh` line 3), which adds no parameters; I did not load their checkpoints.
- I did not compare training budgets (240,000 steps at batch 8 in sprint 10 against `windows_seen` 80,000 for our base configs); this note is about layer widths only.
