# Avocado — retrained on our EIC store

Avocado (Schreiber, Bilmes & Noble 2020, *Genome Biology* 21:81) as a **tensor factorisation over
(cell, assay, position)**: a prediction is one MLP applied to the concatenation of a cell embedding,
an assay embedding and that position's genomic latent factors. Because the genomic factors are
per-position free parameters, the model cannot extrapolate to a position it never fitted — which is
why genome-wide prediction is 23 separate per-chromosome fits, not one pass.

This is Max Wallace's PyTorch port (`experiments/005_2026-07-29_avocado_cross_dataset`), **vendored
byte-identical** and re-trained on our data. The authors' own implementation is Keras/TF1 and does
not install on Fir.

---

## 1. Provenance — the vendored files

Source: `fir:/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/005_2026-07-29_avocado_cross_dataset/scripts/`
(Max's 005, byte-verified against his own Nibi manifest
`experiments/MD5SUMS.nibi.txt`; originals live at
`nibi:/scratch/maxwl/epi_imputation/repo/experiments/005_2026-07-29_avocado_cross_dataset/scripts/`).

| vendored file | md5 | what it is |
|---|---|---|
| `vendor/avocado.py` | `0eca3ad67d1854ff05ece0a81466173c` | **the model** — imported, never edited |
| `vendor/hpc_train.py` | `29ec22235acfbc9f5fcb4d4368f0e51b` | 005's trainer; `train.py` is derived from it |
| `vendor/hpc_predict.py` | `4806d6337764bca89326d136a8313683` | 005's predictor; `predict.py` is derived from it |
| `vendor/hpc_bin_tracks.py` | `627668c2264bc46267d39796eea5746d` | 005's binner; `bin_store.py` is derived from it |

`vendor/MD5SUMS` holds the same four lines and is what to re-check against, not this table.
`md5 -r vendor/*.py` on macOS, `md5sum` on Fir — both print the same digests.

**Nothing in `vendor/` is edited.** `train.py` and `predict.py` `import` `Avocado` and
`holdout_mask` out of `vendor/avocado.py`; they do not re-declare the architecture. The other three
vendored files are the record of what the adapters were derived from, so a reviewer can diff.

### What the adapters changed, and why

| file | changed from 005 | why |
|---|---|---|
| `bin_store.py` | reads `CANDI_STORE` through `candi.store.reader` instead of binning bigWigs | the store is already on the `floor(chr_len/25)` grid; there is nothing to bin |
| `index.py` | new — the (cell, assay) index space | 005 read a challenge `bridge.csv` of `C##`/`M##` codes; we have `T_`/`V_` biosample ids |
| `train.py` | columns from `tracks.csv`; **resume**; `--seed` a flag | the MIG slice may not finish a chromosome inside one job's walltime |
| `predict.py` | predicts the regime's declared tracks; writes §4.1 npz | 005 wrote a fixed 51-experiment `.npy` and bigWigs |
| `fit_sigma.py` | new — the §6.1 σ-table | 005 had no CRPS arm |

Untouched from 005: the architecture, the arcsinh target and `sinh`+clip inversion, the MSE
objective, Adam with two learning rates (`1e-3` shared / `1e-2` genomic), batch = 1024 positions ×
every track, the deterministic 1-in-50 held-out **entry** mask, and the two-stage
shared-then-per-chromosome scheme with chr20 refitted from a fresh init in stage 2.

---

## 2. Fairness — where §6.2 is enforced

> Rivals train on training-split biosamples only. No `V_` or `B_` track enters any rival's
> training, fine-tuning, correlation table, or calibration.

**The enforcement point is `index.py::train_columns`.** It is the only function in this directory
that produces a column list, and every stage downstream — binning, training, the σ fit — takes its
columns from it. It does two things:

1. Draws the biosample list from `regime["biosamples"]["train"]` and from nowhere else. On
   `configs/regime.eic_val.json` that is the 51 `T_` cells, giving **267 columns**.
2. **Raises** `FairnessError` if any train biosample also appears as an eval *target* in
   `eval_pairs`. It does not filter it out quietly: a rule that silently drops a column is a rule
   nobody can audit.

`bin_store.py` writes those columns and only those to `<chrom>.npy`, and records them in
`tracks.csv`. **A `V_` or `B_` track has no column in the matrix `train.py` loads, so there is
nothing for it to be trained on.** The σ-fit reads predictions and truth, never a second fit.

The other half of the rule is subtler and worth stating. Avocado needs a cell embedding for the
target cell of every pair, or it has nothing to impute from. `index.py::cell_index` gives `T_X` and
`V_X` **one** index — but it learns that they are the same cell from the regime's own `eval_pairs`
entry `["T_X", "V_X"]`, never by splitting a biosample name apart (`STORE.md` D16: a biosample name
is an opaque id). That embedding is fitted **entirely from `T_X`'s tracks**, which is exactly the
situation 005 was in: every EIC blind-test cell type has 1–11 train/validation experiments of
*other* assays, and that is where its embedding comes from.

Verified on this store: `T_X` and `V_X` share **zero** assays across all 26 declared pairs, so no
target track's assay is even present in the training matrix for that cell.

---

## 3. Recipe (§7.1, §6.3)

Paper defaults, **one seed** (`--seed 0`, the value 005 used). No tuning sweeps.

| | |
|---|---|
| cell factors | 32 |
| assay factors | 256 |
| genomic factors | 25 @ 25 bp, 40 @ 250 bp, 45 @ 5 kbp |
| head | 2048 – 2048 – 1, ReLU |
| target | `arcsinh(-log10 p)`; inverted with `sinh`, clipped at 0 |
| optimiser | Adam, lr `1e-3` shared / `1e-2` genomic |
| batch | 1024 positions × 267 tracks |
| **epochs** | **60** joint (chr20) / **30** per chromosome |
| held-out monitor | deterministic 1-in-50 of (position, track) entries |

**Why 60/30 and not 005's 120/60.** 005 measured its own budget as roughly twice what is needed.
On its d3 chr20 joint fit the held-out MSE falls 0.14495 → 0.07206 by epoch 15, bottoms out at
**0.07177 around epoch 34**, and then drifts *up* to 0.07572 by epoch 119 while the training MSE
keeps falling — mild overfitting of the genomic factors. Halving is not a saving on a converged
run; it is the better run. (Source: `005/output/train_logs/d3_train_log.jsonl`.)

Checkpoint selection follows 005 exactly — **the last epoch, not the best on held-out**. 005's own
Limitations section recommends best-on-held-out for a future run; adopting it here would make our
Avocado and 005's Avocado two different procedures and break the convergence-shape comparison that
is this task's gate. The full held-out curve is in `logs/train_*.jsonl` if the PI wants the other
choice made later.

---

## 4. Environment and paths

| | |
|---|---|
| cluster | Fir, account `def-maxwl` |
| checkout | `/project/def-maxwl/mforooz/CANDII_t50` (branch `implementation/t50-avocado`) |
| python | `/project/def-maxwl/mforooz/candi_venv` — torch 2.6.0, h5py 3.12.0, numpy 1.26.4 |
| store | `/project/def-maxwl/mforooz/CANDI_STORE/eic` (via the regime) |
| workspace | `/scratch/$USER/t50_avocado/{binned,ckpt,pred,logs}` |
| regime | `configs/regime.eic_val.json` — dev. **`regime.eic_test.json` (B pairs) is run once, at the end, and not by this task.** |

`slurm/_env.sh` sets all of it; every job sources it. The workspace is on `/scratch` because the
binned matrices are ~129 GB and the checkpoints another ~50 GB, and none of that is a result — what
comes back to `/project` is the training logs, the scores and the σ-table. `/scratch` purges after
60 days, which is why the commands below exist rather than living in somebody's shell history.

**GPU.** Every `#SBATCH --gres` here is `gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, per `AGENTS.md`
§3.13. 005 ran on a full H100 at ~46 steps/s and estimated a 1g.10gb slice at ~7× slower; that
estimate is 005's, not a measurement of this code on our 267-column panel — see the smoke-run note
in the task output.

---

## 5. Running it

```bash
cd /project/def-maxwl/mforooz/CANDII_t50 && mkdir -p slurm-logs
WS=/scratch/$USER/t50_avocado
A=competitors/avocado

# 1. bin the store  (23 tasks, CPU-bound)
sbatch --array=0-22 $A/slurm/bin.sh

# 2a. joint fit, chr20, 60 epochs, everything trainable
printf 'chr20\n' > $WS/chrom20.txt
sbatch --array=0-0 --export=ALL,MODE=shared,CHROMFILE=$WS/chrom20.txt,EPOCHS=60 $A/slurm/train.sh

# 2b. per-chromosome genomic factors, 30 epochs, shared frozen  (23 tasks)
sbatch --array=0-22 --export=ALL,MODE=genome,EPOCHS=30 $A/slurm/train.sh

# 3. the §4.1 prediction root, all 23 chromosomes
python $A/predict.py --regime configs/regime.eic_val.json --out $WS/pred/eic_val \
       --write-manifest --version 005-port --notes "60/30 epochs, seed 0, 267 T_ training tracks"
sbatch --array=0-22 $A/slurm/predict.sh

# 4. the §6.1 σ-table, fitted on the V panel's P1 chromosomes
python $A/fit_sigma.py --regime configs/regime.eic_val.json \
       --pred $WS/pred/eic_val --out $WS/sigma_avocado_v.json

# 5. score — P1 (declared pairs, the regime's eval_chroms)
python -m candi.bench.external --store configs/regime.eic_val.json \
       --pred $WS/pred/eic_val --sigma-table $WS/sigma_avocado_v.json \
       --out $WS/scores_avocado_P1.json

# 5b. score — P2 (genome-wide: the same root, every chromosome)
python -m candi.bench.external --store configs/regime.eic_val.json \
       --pred $WS/pred/eic_val --sigma-table $WS/sigma_avocado_v.json \
       --chroms "$(paste -sd, $A/chroms.txt)" --out $WS/scores_avocado_P2.json
```

A resumed `train.py` picks up its `.partial` automatically; re-submitting the same array command is
the whole resume procedure. A *finished* chromosome writes `ckpt/<mode>_<chrom>.pt` and the job
skips it, so re-submission is idempotent.

**Measured binning cost** (first run, 2026-08-25): chr21 **1444 s**, chr20 **2017 s**, MaxRSS
**2.9 GB** against a 32 GB ask. Close to linear in bins, so chr1 (9.96 M bins) projects to ~2.1 h —
which is why `bin.sh` asks for 6 h and not 3. Output sizes are exact: `chr20.npy` is
2,753,054,216 B = 2,577,766 bins × 267 tracks × 4 B + a 128 B npy header. If that arithmetic does
not close on a rebuild, the column set changed and §2 is the thing to re-check.

**Smoke probe** (writes nothing, prints steps/s and peak CUDA memory):

```bash
sbatch --array=0-0 --export=ALL,MODE=shared,CHROMFILE=$WS/chrom20.txt,EPOCHS=60,SMOKESTEPS=400 \
       --time=00:30:00 $A/slurm/train.sh
```

---

## 6. Reporting

Avocado emits a **point in `-log10 p` and nothing else**. So:

- pval arm only. No count arm — B1b forbids inventing a read depth to manufacture one.
- CRPS exists only through the §6.1 σ-table, and **V-pair CRPS is in-sample for σ**. Any table
  showing it says so. The B-pair run reuses the V-fitted table unchanged.
- The peak tier is **coverage ranking**: `signal_mu` used as the ranking score, `has_peak_head=False`
  recorded by the scorer. Every AUPRC row carries that label and its `peak_base_rate` (B3).
- Numbers from this method are **not quotable** until the t49 leaderboard exists.
