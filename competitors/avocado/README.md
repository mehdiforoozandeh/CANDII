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
| `bin_store.py` | reads `CANDI_STORE` through `candi.store.reader` instead of binning bigWigs; `--regions` | the store is already on the `floor(chr_len/25)` grid; and §3.1 restricts `eic.pilot`'s scope to a BED |
| `index.py` | new — the (cell, assay) index space, the `V_` selection panel, the BED scope | 005 read a challenge `bridge.csv` of `C##`/`M##` codes; we have `T_`/`V_` biosample ids |
| `train.py` | columns from `tracks.csv`; **resume**; `--seed` a flag; **`V_` checkpoint selection**; `--positions` | the MIG slice may not finish a chromosome inside one job's walltime; §5 asks every method to select on `V_` |
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

Checkpoint selection **no longer follows 005** — see §7. It used to: the last epoch, not the best on
held-out, so that our Avocado and 005's Avocado stayed one procedure and the convergence-shape
comparison below stayed readable. `plan/BENCHMARK_DESIGN.md` §5 overrides that, and the PI ruled on
2026-08-31 that Avocado gets a real selection loop rather than a "no selection" marker. The
convergence gate below was measured on a last-epoch run and stands as recorded; the shape it reads
is the whole held-out curve, which is still logged in full.

### The convergence gate, measured (§7.1)

The chr20 joint fit ran 60 epochs in 6:44:42 at 6.23 steps/s. Its curve against the 005 anchor
(`005/output/train_logs/d3_train_log.jsonl`, its own chr20 joint fit):

| | ours, 60 ep | 005 d3, 120 ep |
|---|---|---|
| held-out minimum | **0.05852 @ epoch 27** | **0.07153 @ epoch 22** |
| drift, minimum → epoch 59 | **+2.24 %** | **+2.14 %** |
| train MSE at epoch 59 | 0.02653, still falling | 0.03934, still falling |
| held-out normalised, ep 0 → 5 → 30 → 59 | 1.000 → 0.576 → 0.558 → 0.566 | 1.000 → 0.649 → 0.618 → 0.629 |

Same signature in both: a fast drop through epoch ~5, a shallow basin through the 20s, then mild
drift upward while the training MSE keeps falling — the genomic factors overfitting, not the run
undertraining. **The gate is on shape, and the shape reproduces.**

The absolute levels differ (0.0585 vs 0.0715) and are *not* expected to match: 005 fitted the
challenge's own Dataset-3 tracks over 312 columns, we fit our store's Dataset-2 `-log10 p` over 267.
Comparing the two absolute numbers would be exactly the Dataset-2→Dataset-3 translation 005 measured
at 12–66 % per-experiment error and told us not to do.

This also re-confirms the halved budget from our own data rather than on 005's authority: the
minimum lands at epoch 27, and epochs 28–59 bought **nothing** (+2.24 % worse at the end).

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

**GPU — the MIG slice was measured, not assumed.** Every `#SBATCH --gres` here is
`gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, per `AGENTS.md` §3.13. 005 ran on a full H100 at 46.4 steps/s
and *estimated* a 1g.10gb slice at ~7× slower with a working set too large to fit. Both halves of
that estimate were checked on our own panel before any budget was committed:

| | measured |
|---|---|
| throughput, chr20 joint fit | **6.23 steps/s** (600-step probe: 5.78) — **7.5×** slower than 005's full H100 |
| peak CUDA | **5.15 GiB** of the slice's 10 GiB — fits with room, because 267 columns is not 312 |
| chr20 joint fit, 60 epochs | 6:44:42 |
| throughput, genome mode (chr21) | **7.60 steps/s** — faster, because no gradient reaches the frozen shared params |
| peak CUDA, genome mode | **5.17 GiB** |

So the rule costs wall-clock and forbids nothing: the whole retrain is ~130 GPU-h on the slice
against ~22 on a full H100, and every job fits its walltime. **No exception to §3.13 was needed and
none was taken.**

Per-chromosome walltimes below were set from the conservative 6.0 steps/s figure *before* the
genome-mode probe measured 7.60. They were deliberately not re-tightened afterwards: chr1 carries
291 M genomic parameters against chr21's 54.6 M, so its Adam step is heavier and chr21's rate is an
upper bound, not a prediction. The asks are split into three walltime bins so a 2.5 h chromosome
does not queue behind a 20 h ask, and `MAXHOURS` plus the resume path absorbs any overrun at a cost
of at most one epoch of rework.

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

# 2b. per-chromosome genomic factors, 30 epochs, shared frozen  (23 tasks).
#     Split by projected wall-clock at 6.0 steps/s so a 2.5 h chromosome is not asking for 20 h:
#     chr1 13.5 h, chr2 13.1 h | chr3 10.8 h .. chr7 8.6 h | chr8 7.9 h .. chrX 8.5 h, chr21 2.5 h.
#     MAXHOURS always leaves the job ~1 h to load its matrix and write out.
sbatch --array=0-1  --time=20:00:00 --export=ALL,MODE=genome,EPOCHS=30,MAXHOURS=19.0 $A/slurm/train.sh
sbatch --array=2-6  --time=14:00:00 --export=ALL,MODE=genome,EPOCHS=30,MAXHOURS=13.0 $A/slurm/train.sh
sbatch --array=7-22 --time=11:00:00 --export=ALL,MODE=genome,EPOCHS=30,MAXHOURS=10.5 $A/slurm/train.sh

# 3. the §4.1 prediction root, all 23 chromosomes
python $A/predict.py --regime configs/regime.eic_val.json --out $WS/pred/eic_val \
       --write-manifest --version 005-port --notes "60/30 epochs, seed 0, 267 T_ training tracks"
sbatch --array=0-22 $A/slurm/predict.sh

# 4-5. σ-table + P1 + P2 scoring, one job (see slurm/score.sh for why it is a job)
sbatch $A/slurm/score.sh
```

A resumed `train.py` picks up its `.partial` automatically; re-submitting the same array command is
the whole resume procedure. A *finished* chromosome writes `ckpt/<mode>_<chrom>.pt` and the job
skips it, so re-submission is idempotent.

**Measured binning cost** (first run, 2026-08-25): chr21 **1444 s**, chr20 **2017 s**, MaxRSS
**2.9 GB** against a 32 GB ask. Close to linear in bins, so chr1 (9.96 M bins) projects to ~2.1 h —
which is why `bin.sh` asks for 6 h and not 3. Output sizes are exact: `chr20.npy` is
2,753,054,216 B = 2,577,766 bins × 267 tracks × 4 B + a 128 B npy header. If that arithmetic does
not close on a rebuild, the column set changed and §2 is the thing to re-check.

### The predict chain was validated before the genome fits were spent

A bug in `predict.py` would only surface after ~130 GPU-h of per-chromosome fits, so it was checked
first against a deliberately throwaway 1-epoch chr21 genome fit, written to a **separate** root so
it could never contaminate a real prediction:

```
[pred chr21] 45 declared tracks x 1868399 bins
[pred chr21] wrote 45 tracks in 19s; mean=0.2801 max=84.1
declared 45  found 45  missing 0  unknown 0
CONTRACT OK: 45 tracks, float32, len 1868399, finite, non-negative (min 0.0000 max 84.08)
```

Read back through `candi.bench.external.read_track_arrays` — the scorer's own loader — so the
directory names, the grid length, the dtype and the `sinh`+clip inversion are all checked by the
code that will consume them. Prediction is cheap: 19 s per chromosome for all 45 tracks, so the
whole genome is ~20 min and `predict.sh`'s 3 h ask is generous.

**Throwaway artifacts left on scratch by that check** — `ckpt/_smoke_genome_chr21.pt` and
`pred/_smoke/` (549 MB). Both are inert: `predict.sh` resolves `ckpt/genome_<chrom>.pt`, which the
`_smoke_` prefix does not match, and `score.sh` defaults `PRED` to `pred/eic_val`. Delete them at
will; `/scratch` purges them anyway.

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

### Results

`cruxvault/results/t50/` in the main checkout — `scores_avocado_P1.json`, `scores_avocado_P2.json`,
`sigma_avocado_v.json`, `FIR_PATH.txt`, and **`CAVEATS.md`, which must be read before any number
here is quoted**.

Both protocols scored 45 of 45 declared tracks with `missing_tracks: []` and no non-finite value
anywhere. P2 reuses P1's σ-table byte-identically — only `--chroms` differs.

| macro, pval arm | P1 (chr21) | P2 (genome-wide) |
|---|--:|--:|
| tracks | 45 | 45 |
| bins per track | 1,868,399 | 121,241,684 |
| `mse` | 6.6634 | 8.8983 |
| `gwcorr` | 0.5576 | 0.5456 |
| `gwspear` | 0.3486 | 0.3186 |
| `crps` | 0.6172 | 0.6565 |
| `pit_ks` | 0.3765 | 0.3592 |
| `coverage_95` | 0.9840 | 0.9803 |
| `gaussian_nll` | 1.7578 | 1.9293 |
| `c_index` | 0.5277 ± 0.0002 | 0.5275 ± 0.0002 |
| `auprc` | 0.4118 | 0.4186 |
| `peak_base_rate` | 0.0138 | 0.0169 |

Broad (H3K27me3/H3K36me3/H3K9me3) vs punctate medians, never pooled — P1: `mse` 0.700 / 1.161,
`gwcorr` 0.436 / 0.596, `auprc` 0.212 / 0.431. P2: `mse` 0.763 / 1.551, `gwcorr` 0.343 / 0.596,
`auprc` 0.217 / 0.457.

**The σ device is the thing to look at first.** `coverage_95` 0.98 against a nominal 0.95 and
`pit_ks` ~0.37 say the homoscedastic per-assay σ is far too wide almost everywhere — one constant
cannot cover residuals running from near-zero background to tall peaks. That is a property of the
B1a point→Gaussian device, not of Avocado's point predictions, and every point-only rival inherits
it. `mse`/`gwcorr`/`gwspear` and the partition MSEs are unaffected.

---

## 7. Checkpoint selection on `V_`, and the BED training scope

Two capabilities added 2026-08-31 on the PI's rulings. Both are held by
`tests/test_avocado_selection.py` and `tests/test_avocado_regions.py`.

### 7.1 The selection loop

Every `--select-every` epochs, `train.py` writes this model's `V_` predictions to a §4.1 root and
scores them with `candi.bench.external.score_external` — **the same function that scores CANDI**,
reading truth through the same window walk. Nothing here computes a metric of its own, which is what
makes §5's "by the same rule" checkable rather than asserted. The best weights are written the moment
the metric improves, so a run killed by walltime still leaves a selected checkpoint.

`--out` ends up holding the **selected** checkpoint, and the last epoch sits beside it as
`.last.pt`. Nothing downstream resolves that name, so no stage can predict from the unselected model
by accident.

**`B_` is never read.** `index.py::write_select_regime` derives a `V_`-only regime — 26 pairs kept,
12 `B_` pairs dropped — and writes it next to the checkpoint as `<out>.select_regime.json`, so the
panel a run selected on is auditable after the fact instead of being an argument.

**Where it can select, and one place it cannot.** `score_external` scores whole chromosomes, and
Avocado can only be scored where it has genomic factors:

| stage | scope | selects on |
|---|---|---|
| `MODE=genome`, chr20 / chr21 / chr22 | one whole eval chromosome | that chromosome's `V_` panel |
| `MODE=shared` under `eic.19` | chr19 | chr19's `V_` panel |
| `MODE=shared` under `eic.pilot` | 1,023,489 bins scattered over 18 chromosomes | **nothing — it cannot** |

The last row is a property of Avocado's shape, not a missing feature: a BED-scoped fit holds factors
only inside the regions, so it cannot produce the whole-chromosome array the scorer demands.
`train.py` refuses `--select-every` with `--positions` outright rather than downgrading in silence,
and `slurm/train.sh` forces `SELECTEVERY=0` for that one stage and says so. The stage that produces
the shipped predictions — `MODE=genome` — still selects under both regimes.

**Which number.** `--select-metric` defaults to `mse` on the pval arm. CANDI selects on `crps`, and
Avocado has none: it emits a point, and CRPS exists for it only through a §6.1 σ-table, which is
fitted *after* training. Asking for `crps` without `--select-sigma-table` is refused rather than
silently answered with a different key. **This is a real divergence from CANDI's selection rule and
is the PI's to close** — the same σ table would let every point-only rival select on `crps`.

### 7.2 What selection costs — measured, and it is not free

`score_external` was timed on a synthetic store at three sizes (50 k / 200 k / 800 k bins × 6
tracks); it is linear above ~200 k bins at **7.16 µs per (bin × track)** on a laptop CPU. The Fir
anchor for the same panel is §12.2's measured CANDI monitor — 5,446.6 s over chr20+21+22 × 45
tracks = **19.1 µs per (bin × track)** — a different estimator on different hardware, and the
pessimistic end. One check of the 45-track `V_` panel therefore costs:

| | laptop rate | Fir anchor |
|---|--:|--:|
| chr19 (2,344,704 bins) | 12.6 min | 33.6 min |
| chr20 (2,577,766) | 13.8 min | 36.9 min |
| chr21 / chr22 | ~10.1 min | ~27.0 min |

Prediction is negligible beside it: 45 tracks over one chromosome is ~26 s on the MIG slice, at the
rate §5 measured.

Against a regime's ~12.9 GPU-h of training (6.12 h shared on chr19 at the measured 6.23 steps/s,
plus 2.76 + 2.00 + 2.03 h of genome fits at 7.60 steps/s), `SELECTEVERY=10` buys 6 checks on the
shared fit and 3 on each genome fit:

| | added | total per regime |
|---|--:|--:|
| at the laptop rate | +2.95 h (+23 %) | 15.9 GPU-h |
| at the Fir anchor | +7.9 h (+61 %) | 20.8 GPU-h |

**It does not triple the run; it adds a quarter to two thirds.** `SELECTEVERY=10` is the default for
that reason, and it is a flag: lengthen it before reaching for a bigger walltime. Two consequences
worth stating —

- **`MAXHOURS` must leave room for one check.** `train.py` tests the deadline before starting a
  selection and skips it if it is past, so the overrun is bounded, but a 12 h job whose `MAXHOURS`
  is 11.5 has less slack than a 34-minute check wants.
- **Patience is coupled to the cadence.** `slurm/train.sh` defaults `SELECTPATIENCE` to
  `SELECTEVERY`; a patience below the cadence can never fire, which is the trap CANDI's launcher
  documents.

### 7.3 The BED scope

Under `configs/regime.eic_pilot.json` the shared fit trains on the ENCODE Pilot Regions. **The rule
is D32's containment at Avocado's own unit: a 25 bp bin counts only if the bin lies wholly inside a
region.** CANDI applies containment to a 768-bin window and gets 1,294 windows = 993,792 bins;
Avocado is per-position, so it applies it to a bin and gets **1,023,489** — the figure §3.1 pins for
the training scope. Both come from `RegionSet.bin_spans`, the same primitive, so the two scopes
cannot drift apart.

`bin_store.py --regions` writes those bins as one `regions.npy` over every train chromosome (1.1 GB,
against 2.7 GB for a single whole chromosome) plus a `regions_layout.csv` naming the absolute bin
behind every row. The rows are **packed**, because factors for 18 whole chromosomes would be ~11 G
parameters, and the packing keeps the grid anchored at chromosome bin 0 exactly as §3.1 requires:
each region's offset is a whole number of 5 kbp cells away from its own first bin, so `pos // 10`
and `pos // 200` — the 250 bp and 5 kbp factor grids — cut the genome where a whole-chromosome fit
would have cut it, and no coarse factor is shared by two regions. The alignment costs 7,775 unused
slots (0.75 %), which are never drawn and whose factors stay at their initialisation.

The genome stage is untouched by the BED. It fits one whole eval chromosome, because §4's eval scope
is the whole chromosome and Rule 2 cuts the four Pilot Regions on chr20/21/22 by the regime's
chromosome list, not by the BED.
