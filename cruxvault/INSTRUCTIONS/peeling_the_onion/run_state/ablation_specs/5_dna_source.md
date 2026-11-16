# Ablation 5: DNA source (`dna_source`)

Phase-1 specification, 2026-11-16. Worktree branch `chunk/onion-AB-5`, base `fb57fc1`.

## 1. Question

Does the source of the DNA features change imputation, correction or calibration? The three sources are our own DNA tower trained from scratch, frozen Flashzoi embeddings and frozen genomic-language-model embeddings. The PI's line (`peeling the onion.md`, # miniCANDI, important notes): "for DNA embeddings we have three options: our own DNA tower trained from scratch; using Flashzoi embeddings; using a genomic language model embeddings e.g. nucleotide transformer". A second question follows from it: does the own tower use aligned sequence at all, or only a locus fingerprint (i.e., a sequence pattern that identifies the locus, so the model can memorize the training cells' signal there)?

## 2. Arms

Each arm is one change from the base recipe of each architecture (`tools/minicandi/configs/final/rung{2,3,4}/reference_equal.json`). Every arm key is in `stage2_arms.py` `ARMS`.

| arm | keys | what changes |
|---|---|---|
| reference: own tower (base) | `dna: own` | No new run. The base runs `runs_fir/rung{2,3,4}_base_heldout`, seeds 0 to 2, are reused. |
| `dna_gnlm` | `dna: precomputed`, `dna_embedding_dim: 128`, `dna_embedding_h5: <D>/ntv2` | Nucleotide Transformer v2 500M (NT v2), last hidden layer, PCA to 128 dimensions, one float16 vector per 25 bp bin. The model reads `x_dna_emb` through LayerNorm(128) and Linear(128, d). The one-hot conv front of the tower is gone. |
| `dna_flashzoi` | as above, `<D>/flashzoi` | Flashzoi (a fast Borzoi replicate), final joined convs, PCA to 128. Ablation only (PI ruling 26). It is judged on Borzoi fold-3 loci. |
| `dna_shift5mb` | `dna_shift_bp: 5000000` | The own tower reads the sequence 5 Mb away on the same chromosome (wrapping). The shift is fixed, so each locus keeps a stable but wrong sequence. |

`<D>` is the directory that holds the embeddings on the training cluster (see section 7). Three new arms x 3 architectures x 3 seeds = 27 runs.

Readings and choices.
- "Genomic language model" has many candidates. I take NT v2, because ruling 22 approved it and its embeddings exist.
- Two leakage controls are declared: `dna_shuffled` and `dna_shift5mb`. I take `dna_shift5mb` only, in this round. It keeps the parameter count and the sequence statistics, and it keeps a locus fingerprint. So own minus shift is the value of correctly aligned sequence. `dna_shuffled` draws a fresh 25 bp block order per training step, so it removes both alignment and fingerprint. It is useful only when shift equals own (row C of section 5). Then it is the second stage (9 runs).
- `dna_off` (COVERAGE row 10) is not taken. In the U-Net it removes the DNA tower and the DNA skips. That is 119,377 parameters, i.e. 12% of the U-Net, so the arm would confound source with capacity.

## 3. Prior leads

1. Rung-1 DNA off, 3 seeds (`results/metrics_screening/scores_rung1_arms.md`, foreman's worktree): specific variance 0.3431 (0.0029) base against 0.3400 (0.0035) DNA off. Mean difference -0.0031. No column moved by more than the seed spread. Finding f73 (weak) gives the imputation direction as lower. Prediction: at the pointwise MLP, DNA carries almost nothing. Rung 1 has no sequence context, so the lead is weak for the three architectures here.
2. U-Net skip sources (`results/final_round/FINAL_ROUND_EQUAL.md`): skips carry counts only, imputation 0.432 (0.006), the same as base 0.432 (0.023). Skips carry DNA only: 0.380 (0.022), with correction 0.734 (0.063) against 0.507 (0.071). Prediction: the U-Net's DNA skips add no imputation when count skips are present. A new DNA source changes the U-Net's imputation only if it carries track-like information.
3. Flashzoi overlap (`long-horizon notepad.md`, ruling 26 result; `results/flashzoi_overlap/FLASHZOI_OVERLAP.md`): Flashzoi was trained on 44 of 45 EIC validation experiments and on 252 of 267 training experiments. Its training folds cover 68.95% of chr1-22 and chrX. Fold 3 holds 11.40%. Prediction: on training-fold loci the Flashzoi embedding encodes the validation tracks themselves. So its standard-set score is inflated, and only fold-3 loci are valid.
4. Embedding quality (`results/t207/DNA_EMBEDDINGS.md`): 128 PCA components explain 0.7746 of Flashzoi's 1,920 dimensions and 0.5499 of NT v2's 1,024. Receptive field is 524,288 bp (Flashzoi) and 12,276 bp (NT v2), against 15 bp for the own tower's first convolution. Prediction: NT v2 at 128 dimensions loses about half its variance. Its gain over the own tower is limited.
5. The old vault holds no DNA-source result. `h81` (archive vault) is an untested idea.

## 4. Expectation (before any run)

Against the base of the same architecture, 3 seeds, standard 16-region screening set (`p16_sv`):

| arm | imputation | correction | calibration | composite |
|---|---|---|---|---|
| `dna_gnlm` | within range; at most +0.02 | within range | within range | within range |
| `dna_flashzoi`, standard set | higher, beyond range (about +0.03 or more) on all three | within range | within range | higher |
| `dna_flashzoi`, fold-3 set | smaller gain than on the standard set; within range likely | within range | within range | within range |
| `dna_shift5mb` | conv autoencoder and conv + attention: within range; U-Net: within range | within range | within range | within range |

Correction is not expected to move. The DNA input is the same for the source and the target of a correction pair.

## 5. How the result answers the question

"Beyond" means the difference of means is larger than the larger of the two seed ranges. Each row is read per architecture.

| outcome | conclusion |
|---|---|
| A. `dna_gnlm` beyond own on imputation or composite, fold-3 too | A pretrained sequence model helps. NT v2 is a candidate for the final design (it joins the design). |
| B. `dna_gnlm` within range of own | The DNA source does not matter at this scale. Keep the own tower (simpler, no extra environment). |
| C. `dna_shift5mb` within range of own | The own tower adds nothing, or adds only a locus fingerprint. Run `dna_shuffled` (stage 2) to tell them apart. |
| D. `dna_shift5mb` below own, beyond range | Aligned sequence carries information that the model uses. |
| E. `dna_flashzoi` beyond own on the standard set, within range on fold 3 | The Flashzoi gain is leakage of its training tracks. It is not evidence for pretrained DNA. |
| F. `dna_flashzoi` beyond own on fold 3 too | Pretrained, track-trained sequence features help on unseen loci. This is the upper bound of the DNA input. Flashzoi still stays out of the final model (ruling 26). |
| G. any arm lowers correction beyond range | The DNA path interferes with correction. Report it per architecture. |

## 6. Controls and traps

- Flashzoi leakage. Control: score base, `dna_gnlm` and `dna_flashzoi` on a fold-3 eval set as well (section 7). Only 33 of the 256 current eval regions lie wholly inside fold 3, and only 2 of the 16 correction pair regions (indices 64 and 240). So the current set cannot be reused by a subset.
- Locus fingerprint. A stable wrong sequence can still act as a locus ID. `dna_shift5mb` keeps it; `dna_shuffled` (stage 2) removes it.
- Parameter count. No refit is needed; the trunk width does not change. Built at seed 0 with `registry.build_model`:

| architecture | own (base) | precomputed (both sources) | `dna_shift5mb` | trainable DNA parameters, own -> precomputed |
|---|---|---|---|---|
| conv autoencoder | 1,042,514 | 1,043,890 (+4.4% of 1.0e6) | 1,042,514 | 3,008 -> 4,384 |
| conv + attention | 1,038,018 | 1,039,394 (+3.9%) | 1,038,018 | 3,008 -> 4,384 |
| U-Net | 993,599 | 994,303 (-0.6%) | 993,599 | 119,377 -> 120,081 |

  The frozen model's own weights are not trained and not counted. The frozen embedding removes only the tower's one-hot front (Conv1d(4, 32, 15) and a 1x1 conv). Linear(128, d) replaces it, so the trainable count rises slightly.
- U-Net DNA skip. With `dna: precomputed`, the U-Net DNA level stack (`_Tower`, about 116,600 trainable parameters) still runs. It reads the projected frozen embedding and feeds both the bottleneck and the DNA skips (`skip_source: both`). The comparison is therefore "a different DNA front end under the same trainable stack", not "no trainable DNA path".
- Scorer input. The scorer and the predict jobs build `OnionDataset` from the run config (`score.py` line 943). So they read `x_dna_emb` from the same `dna_embedding_h5` path. That path must exist on the scoring cluster too. No other scorer input changes.
- One device per comparison. The base runs are on Fir CPU. The new arms must also train and score on Fir CPU.
- `aug_revcomp` is refused with a frozen embedding (`registry.py` line 180). The base does not use it.

## 7. Readiness

Embeddings (checked 2026-11-16, read-only `ls -la`):
- Nibi: `/scratch/mforooz/onion/dna_emb/{flashzoi,ntv2}/chr{1..22,X}.h5` exist (46 h5 files), with `folds/borzoi_{train,test}.bed`. `DNA_EMBEDDINGS.md` records a verify run: 23 chromosomes, bins match the genome layer, NaN 0. The size is 31,054,109,776 B per source.
- Fir: absent. `/scratch/mforooz/onion/dna_emb` does not exist. `/project/def-maxwl/mforooz/ONION/dna_emb*` and `.../ONION/dna_models` do not exist. A `find` for `projection.npz` under `ONION` and `/scratch/mforooz` (depth 4) found nothing.
- **The embeddings and the fold-3 BED exist on Nibi only.** Moving them to Fir (31.05 GB per source, by Globus) is the foreman's call.

Coverage. Training reads chr1-22 and chrX (`dataset.py` default `exclude_chroms` chrY, chrM). The 256 eval regions lie on the same chromosomes (no chrY). The embeddings cover all 23 chromosomes on the genome-layer grid. So they cover every training window and every eval region.

Per architecture (Phase 2, 2026-11-16):
- Configs written: `tools/minicandi/configs/ablations/dna_source/rung{2,3,4}/{dna_gnlm,dna_flashzoi,dna_shift5mb}.json` and `arms.tsv`. They were written by `stage2_arms.py write` with `--gnlm-emb /scratch/mforooz/onion/dna_emb/ntv2 --flashzoi-emb /scratch/mforooz/onion/dna_emb/flashzoi`. So `<D>` is `/scratch/mforooz/onion/dna_emb` on Fir, the path that mirrors Nibi. `dna_shuffled` is not written, because it is the conditional stage 2.
- Key diff against `reference_equal.json` (all 9 checked). The precomputed arms change `dna`, `dna_embedding_dim`, `dna_embedding_h5` and `simpler_than_base` (null -> false). `dna_shift5mb` changes `dna_shift_bp` only.
- Build (`registry.build_model`, seed 0): all 9 counts equal the table in section 6.
- `dna: precomputed` is read by `conv_ae.py` line 246, `conv_attn_ae.py` line 193 and `unet.py` line 290. `dna_shift_bp` is read by `dataset.py` lines 707 and 750. There is no code change and no test.
- **`dna_gnlm` and `dna_flashzoi` stay blocked on all three architectures.** They wait until the embeddings are on Fir at `/scratch/mforooz/onion/dna_emb/{ntv2,flashzoi}`. `dna_shift5mb` is not blocked.
- Submit options: none beyond the defaults.
- Fold-3 scoring: **a missing piece, and the blocker for the Flashzoi judgment.** It is not built in this chunk. The scorer pools imputation and calibration statistics over all regions. It writes per-region sums only for pair jobs. A fold-3 eval set needs four steps. First, copy `borzoi_test.bed` (Nibi, 206,040 B) to Fir. Second, run `evalset.py build --regions-bed borzoi_test.bed` with the build arguments of `evalset.json`. Third, run `derive` to 16 pair regions. Fourth, score the baselines (marginal NB, position average, oracle reference) on the new set, because the scorer refuses baselines from another eval set. 802 grid blocks of 409,600 bp lie wholly inside fold 3 before the genome mask. The current set drew 256 regions from 6,724 candidates, so 256 fold-3 regions are likely available.

Cost (Fir `sacct -X`, base runs, seed 0; allocated CPUs x ElapsedRaw):
- conv autoencoder: train 12 x 2,521 s, predict 8 x 4,105 s, metrics 16 x 2,108 s. Sum 96,820 CPU-s = 26.89 CPU-h.
- conv + attention: 12 x 1,772 + 8 x 4,097 + 16 x 1,988 = 85,848 CPU-s = 23.85 CPU-h.
- U-Net: 12 x 2,660 + 8 x 4,535 + 16 x 2,077 = 101,432 CPU-s = 28.18 CPU-h.
- One arm on 3 architectures x 3 seeds: 3 x (26.89 + 23.85 + 28.18) = 236.8 CPU-h. Three arms: 710.3 CPU-h.
- Fold-3 score-only, 3 arms (base, `dna_gnlm`, `dna_flashzoi`) x 9 runs: scoring per run is 18.49, 17.94 and 19.31 CPU-h. 3 x 3 x (18.49 + 17.94 + 19.31) = 501.7 CPU-h. This assumes a 256-region fold-3 set. The eval-set build and the baseline scoring on it are not measured.
- Stage 2, `dna_shuffled` only on outcome C: 236.8 CPU-h.
- Total, stage 1: 710.3 + 501.7 = 1,212.0 CPU-h. The h5 reads of the embedding (1,024 x 128 float16 = 256 KB per window) were not timed. Their cost is assumed small.
