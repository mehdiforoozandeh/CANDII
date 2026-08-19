# CANDI_STORE — build plan and handoff

**Status:** approved by the PI (Mehdi) on 2026-08-19 after a full grilling session. Every decision
below is **settled**. Do not re-open them and do not stop to ask. Where this plan is silent, choose
the option most consistent with the decisions here, write down what you chose, and keep going.

**Crux tasks:** `t3` (parent) with children `t4`–`t12` in `cruxvault/tasks/`.
**Merge gate (non-experiment lane):** `pytest tests/ -q` green **and** `python tools/golden.py`
bit-exact. Both must hold at every commit. That is why the new code lands *beside* the old path.

---

## 1. What we are building, in one paragraph

An immutable, whole-genome **corpus store** for EIC (91 biosamples, 439 tracks) and MERGED
(367 biosamples, 3045 tracks) — replacing the window-materialized HDF5 bake. Chromosomes are stored
**whole and contiguous**; windows, context length, DSF factor, chromosome split and assay column
order all move **out of storage and into a manifest read at load time**. One store serves every
regime without a re-bake.

Measured outcome: **12 GB for EIC, 81 GB for MERGED** (counts + peaks + dna) against ~1 TB / ~4 TB
for a naive whole-genome version of the current bake, reading **~10k windows/s single-threaded**.

---

## 2. Measurements this plan rests on

All measured on Fir against real `T_DND-41` chr1 data. Re-derive if you doubt them; do not guess.

**Source corpus (`/project/6014832/mforooz/`)**

| | fact |
|---|---|
| `DATA_CANDI_EIC/` | 91 biosamples, 439 track dirs, **already whole-genome** (`chr1…chrX.npz`) |
| `DATA_CANDI_MERGED/` | 367 biosamples, 3045 track dirs, names like `A549_nonrep`, `A673_grp1_rep1` |
| npz shapes, chr1 | counts `(9958257,) int64` max 73, 31% nonzero, 42 unique |
| | pval `(9958256,) float32` max 161.2, **99.96% nonzero**, 48,182 unique |
| | peaks `(9958256,) int64` max 1, 0.9% nonzero |
| chr1 bins @25bp | 9,958,256 — **counts npz is one bin longer (ceil vs floor)** |
| metadata CSVs | `EpiDenoise/data/eic_metadata.csv` (363 rows), `merged_metadata.csv` (2684 rows) |
| quota | `/project` 15/28 TiB · `/scratch` 1/19 TiB (purged) · `/localscratch` **561 GB per node** |
| Fir Lustre PFL | <1 MB → 1 stripe SSD · 1 MB–1 GB → 1 · 1–10 GB → 4 · >10 GB → **8** (of 164 OSTs) |

**Encoding benchmark (gzip-1, 16k-bin chunks, 1-D)**

| encoding | B/bin | vs source |
|---|---|---|
| counts int64 → **uint16** | 0.441 → **0.273** | 1.6× |
| peaks int64 → **uint8** | 0.041 → **0.010** | 4× |
| pval float32 → float16 | 2.727 → 1.484 | 1.8× |
| pval float32 → **uint16 fixed-point 0.01** | 2.727 → **0.771** | **3.5×** |
| pval → uint16 @0.1 | → 0.404 | 6.7× (max err 0.05) |

Quantization error at 0.01: **max 0.005** on `-log10 p`. At 0.25/uint8, 0.13% of bins clip.

**2-D layout benchmark — `(9958256 bins × 9 tracks)` uint16, real T_DND-41**

| chunk (bins) | codec | size | B/value | read 768×9 | read 6144×9 |
|---|---|---|---|---|---|
| 512 | gzip1 | 24.3 MB | 0.271 | 0.07 ms | 0.25 ms |
| **1024** | gzip1 | 23.5 MB | 0.262 | 0.08 ms | 0.26 ms |
| 2048 | gzip1 | 23.1 MB | 0.258 | 0.10 ms | 0.27 ms |
| 8192 | gzip1 | 22.8 MB | 0.254 | 0.23 ms | 0.35 ms |
| 2048 | **gzip4** | **18.0 MB** | **0.201** | **0.10 ms** | 0.28 ms |
| 2048 | lzf | 40.6 MB | 0.453 | 0.08 ms | 0.20 ms |

gzip4 is 22% smaller at identical read speed → **use gzip4**. lzf rejected. **Blosc/zstd filters are
NOT registered on Fir** — the Compute Canada `hdf5plugin` wheel is a stub with no filters. Do not
plan around zstd.
*Caveat: page cache could not be dropped, so these are warm numbers — i.e. the staged-to-`/localscratch`
regime, which is the one we designed for.*

**DSF is exactly binomial thinning of DSF1.** Verified on 3 tracks × 3 levels:

```
sum(dsf_d)/sum(dsf1):    0.5002 / 0.2499 / 0.1250     (ideal 0.5 / 0.25 / 0.125)
dsf_d <= dsf1:           True everywhere
conditional on dsf1==k:  obs mean/var match Binomial(k, 1/d) to 3-4 decimals
   dsf1==20, d=4:        obs 4.9937 / 3.7215   Binom 5.0000 / 3.7500
log2 depth per doubling: exactly -1.0000
```

**Aux heads are off by default.** `decoder.py:504` builds `GaussianSignalHead` only if
`"signal" in self.heads`; `decoder.py:75` — *signal and peak are additional supervision, off unless
asked for*. `candi.eval` scores **count only**. They **will** come back on, so `pval.h5` is built,
just last.

---

## 3. Settled decisions

| # | decision |
|---|---|
| D1 | **Chromosomes stored whole and contiguous.** One dataset per chromosome, shape `(n_bins, n_tracks)`. **No windows in storage.** HDF5 chunking is an internal compression block, not a window; any `[start:stop]` is addressable. |
| D2 | **Chunk = `(1024 bins, all tracks)`, gzip level 4**, every kind. |
| D3 | **2-D `(bins × tracks)` per (biosample, kind, chromosome)** — one read serves a whole window across all assays. The OO tree (`biosample → experiment → kind → chrom`) lives in the **Python API**, not in HDF5 paths. |
| D4 | **One file per biosample per kind**: `biosamples/<NAME>/{counts,peaks,pval}.h5`. Enables one SLURM array task per biosample with no merge step, and lets staging leave `pval` behind. |
| D5 | **Training reads from `/localscratch`** (561 GB/node), staged at job start. The store itself lives on **`/project`**, never `/scratch` (purged). |
| D6 | **counts: DSF1 only.** DSF2/4/8 are dropped; the loader thins binomially. |
| D7 | **counts dtype: per-file**, `uint16` if the biosample's global max ≤ 65535 else `uint32`; recorded in attrs and manifest; reader upcasts. |
| D8 | **peaks: `uint8`.** |
| D9 | **pval: `uint16` fixed-point, scale 100** (`round(-log10p * 100)`, clip `[0, 65535]`). Stored in the **original space** — no arcsinh at bake time; every transform happens in the model. `scale` in attrs. |
| D10 | **DNA: one shared `genome/dna.h5`**, `uint8` base codes `A=0,C=1,G=2,T=3,N=4`, uppercase-folded, one array per chromosome of length `chr_len`. FASTA sha256 in attrs so a build mismatch is **loud**. |
| D11 | **`genome/mask.h5`** — one `uint8` 0/1 array per chromosome at bin resolution. A bin is **invalid if it contains any `N` or overlaps a blacklist interval**. |
| D12 | **Window eligibility:** a window of `L` bins is eligible iff `mask[s:s+L].mean() >= min_valid_frac`, default **0.9**, overridable in the regime file. |
| D13 | **All kinds use `n_bins = floor(chr_len / resolution)`.** The counts npz `ceil` extra bin is truncated. This kills the documented off-by-one trap outright. |
| D14 | **Assay column order is DECLARED in the regime file**, not derived. The unstable `sort_values(ascending=False)` and its bijection asserts are gone. The store validates that every declared assay exists. |
| D15 | **Store everything on disk**, RNA-seq included, whole long tail. The panel/regime selects. The hardcoded `if "RNA-seq" in exps: exps.remove(...)` at three sites does not carry over. |
| D16 | **Biosample names are opaque ids, verbatim.** No `T_`/`V_`/`B_` first-letter parsing — that filter silently accepts `vagina_nonrep`, `BJ_nonrep`, `BE2C_grp1_rep1`. EIC's prefixes remain meaningful **by convention only**. |
| D17 | **MERGED train/test split is deliberately deferred.** Build with current names. `EpiDenoise/data/merged_train_va_test_split.json` exists for later. |
| D18 | **Control is a normal column** in `counts.h5`, named `chipseq-control`, flagged by a `control_col` attr. |
| D19 | **Metadata: no fabricated values.** Missing or ambiguous → explicit `null` → the encoder's existing `readlen_missing_emb` / `depth_missing_emb`. The `read_length=50` and `run_type=single` fallbacks and the hardcoded replicate key `"2"` do not carry over. |
| D20 | **Metadata authority is the CSVs**; `file_metadata.json` becomes a cross-check that **fails loudly** on disagreement. |
| D21 | **New code lands beside the old.** `prep/bake.py` and `CandiKitH5Dataset` are untouched; `pytest` and `golden.py` stay green throughout. |
| D22 | **Thinning RNG.** Training: seeded from the DataLoader worker seed, free-running. **Evaluation: counter-based and deterministic** — the RNG for one draw is seeded from `SeedSequence([run_seed, hash(biosample), hash(assay), chrom_id, window_start, dsf_milli])`, so the same eval window yields identical counts regardless of batch order, worker count or shuffling. |
| D23 | **Default DSF policy stays discrete `{1,2,4,8}`** for continuity with existing configs; `--dsf-policy loguniform --dsf-min 1 --dsf-max 8` enables the continuous version. |
| D24 | **Build order:** slice → EIC → MERGED → pval. |

---

## 4. On-disk layout

```
/project/def-maxwl/mforooz/CANDI_STORE/
  genome/
    dna.h5              /chr1 … /chrX   (chr_len,)  uint8 codes 0-4, chunk 25600, gzip4
                        attrs: build="GRCh38", fasta_sha256, codes
    mask.h5             /chr1 … /chrX   (n_bins,)   uint8 0/1, chunk 1024, gzip4
                        attrs: blacklist_sha256, blacklist_source, rule
    chrom_sizes.json
  eic/
    manifest.json
    biosamples/
      T_DND-41/
        counts.h5       /chr1 … (n_bins, n_tracks) uint16|uint32, chunk (1024, n_tracks), gzip4
        peaks.h5        /chr1 … (n_bins, n_tracks) uint8
        pval.h5         /chr1 … (n_bins, n_tracks) uint16, attrs scale=100
      …
  merged/               same shape, 367 biosamples
```

**Per-file root attrs** (every kind): `schema=1`, `biosample`, `kind`, `resolution=25`,
`tracks` (JSON list, column order), `control_col` (int or -1), `dtype`, `n_bins` per chrom,
`source_root`, `kit_version`, `built_utc`. For `counts`: `dsf=1`. For `pval`: `scale=100`.

### `manifest.json` (corpus level, generated — never hand-edited)

```json
{
  "schema": 1,
  "corpus": "eic",
  "resolution": 25,
  "genome": {"build": "GRCh38", "fasta_sha256": "...", "n_bins": {"chr1": 9958256}},
  "assay_vocabulary": ["ATAC-seq", "CTCF", "DNase-seq", "..."],
  "kinds": ["counts", "peaks", "pval"],
  "biosamples": {
    "T_DND-41": {
      "dtype": "uint16",
      "control_col": 9,
      "tracks": [
        {"assay": "H2AFZ", "col": 0, "depth": 24684534, "read_length": 36,
         "run_type": "single-ended", "file_accession": "ENCFF764MJJ",
         "exp_accession": "ENCSR...", "bios_accession": "ENCBS...",
         "lab": "...", "platform": "...", "assembly": "GRCh38",
         "pval_clip_frac": 0.0}
      ]
    }
  },
  "source_root": "/project/6014832/mforooz/DATA_CANDI_EIC",
  "built": {"kit_version": "...", "utc": "..."}
}
```

### Regime file (`configs/regime.*.json`) — the authority that replaced `h5.attrs`

```json
{
  "store": "/project/def-maxwl/mforooz/CANDI_STORE/eic",
  "assays": ["ATAC-seq", "DNase-seq", "H3K4me3", "..."],
  "biosamples": {"train": ["T_..."], "eval": ["V_...", "B_..."]},
  "context_bins": 768,
  "train_chroms": ["chr19"],
  "eval_chroms": ["chr21"],
  "window_plan": {"type": "tile", "stride_bins": 768, "min_valid_frac": 0.9},
  "dsf": {"policy": "discrete", "levels": [1, 2, 4, 8]},
  "kinds": ["counts", "peaks"],
  "seed": 42
}
```

The regime's `assays` list **is** the column order. `run.json` must record the regime file verbatim
plus its sha256, and the store's manifest sha256.

---

## 5. Module skeleton

```
src/candi/store/
  __init__.py
  layout.py     # paths, attr names, dtype rules, the fixed-point pval codec — one place
  writer.py     # NPZ tree -> counts/peaks/pval h5 for ONE biosample  (t6)
  manifest.py   # metadata CSV + file_metadata.json cross-check -> manifest.json  (t6)
  genome.py     # FASTA -> dna.h5 ; FASTA + blacklist -> mask.h5  (t7)
  reader.py     # BiosampleStore / CorpusStore — the OO API  (t8)
  regime.py     # regime file parse + validate + window plan generation  (t8)
  dataset.py    # StoreDataset: window sampling, binomial thinning, batch assembly  (t8)
  cli.py        # python -m candi.store build-biosample|build-genome|build-manifest|verify
```

**Reader API** (this is the OO tree from the grilling session):

```python
corpus = CorpusStore("/…/CANDI_STORE/eic")
bs     = corpus["T_DND-41"]
counts = bs["H3K4me3"].counts("chr1", start_bin, end_bin)   # (L,) int32
block  = bs.counts("chr1", start_bin, end_bin, assays=[...]) # (L, F) int32, declared order
dna    = corpus.genome.dna("chr1", start_bp, end_bp)         # (Lbp,) uint8 codes
```

`StoreDataset` must emit **the same batch dict keys** as `CandiKitH5Dataset` so `train.py`,
`batch.py` and `eval.py` need no changes: `x_data, y_data, x_meta, y_meta, x_avail, y_avail,
y_pval, y_peaks, x_dna, control_data, control_meta, control_avail, region_type`, and the
`MISSING=-1` / `CLOZE=-2` sentinel semantics from `_vendored.py` are preserved exactly.

---

## 6. Tasks, in dependency order

`t4`, `t5` and `t6` are independent and start in parallel.

### t4 — obtain the real ENCODE hg38 blacklist v2  `[data-acquisition]`
`EpiDenoise/data/hg38_blacklist_v2.bed` is **8 bytes containing `# Empty`**. Download the real one:
- `https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz` (Amemiya 2019,
  the source `cruxvault/wiki/read-processing-and-artifact-regions.md` already cites), or the ENCODE
  portal file `ENCFF356LFX`.
Verify: >500 intervals, all chroms present in `hg38.chrom.sizes`, no interval past a chrom end.
Place at `/project/def-maxwl/mforooz/CANDI_STORE/genome/hg38-blacklist.v2.bed`, record sha256 and
total covered bp. **Do not overwrite the 8-byte file** — leave the old path alone.

### t5 — recover control-track metadata  `[data-acquisition]`
The 361 `chipseq-control` tracks are absent from both CSVs (3045 − 2684 = 361). Scan every
`<BIOSAMPLE>/chipseq-control/file_metadata.json`, take **whatever replicate key exists** (never
hardcode `"2"`), assert exactly one key and fail loudly on more, and emit
`CANDI_STORE/control_metadata.csv` with the same columns as the signal CSVs. Anything absent is
written as empty → `null` per **D19**. Report the gap counts per field.

### t6 — `candi.store` writer + manifest  `[implementation]`  ← **start here**
Implement `layout.py`, `writer.py`, `manifest.py`, `cli.py`. Per **D1–D4, D7–D9, D13, D15, D18–D20**.
Deliverables: the module, a `STORE.md` contract doc (sibling of `DATA.md`, same voice: anchors as
`file.py::symbol`, symptom-first traps), and tests. **Must not touch `prep/bake.py`, `dataset.py`,
or anything `golden.py` reads.**

### t7 — genome layer  `[implementation]`  ← needs t4
Implement `genome.py` + CLI. Per **D10–D12**. Report mask coverage and eligible-window counts per
chromosome at `context_bins ∈ {768, 6144}` — those numbers go in the task output.

### t8 — reader + data harness  `[implementation]`  ← needs t6
Implement `reader.py`, `regime.py`, `dataset.py`. Per **D3, D6, D12, D14, D22, D23**, plus the batch
dict contract in §5. Binomial thinning: `rng.binomial(counts, 1.0/d)` per element.

### t9 — 5-biosample whole-genome slice + benchmark  `[implementation]`  ← needs t6, t7, t8
Pick 5 EIC biosamples including `T_DND-41` and `B_DND-41` (the 52,051-count DNase track, which
exercises the dtype rule) and at least one `V_`. Build all three kinds whole-genome. Then on a GPU
node: stage to `/localscratch`, measure **windows/s** at `context_bins=768` with 1 and 8 DataLoader
workers, and confirm the store size matches the predictions in §2 within 20%. Record the numbers in
the task output — they are the evidence that the design holds.

### t10 / t11 / t12 — the builds  `[implementation]`
SLURM array, one task per biosample. **These are runs, not code — no branch** (`CLAUDE.md`: an
experiment that only changes flags needs no branch). t10 = EIC counts+peaks+dna (~12 GB, 91 tasks).
t11 = MERGED counts+peaks (~81 GB, 367 tasks). t12 = pval for both (+40 GB / +277 GB).
Follow `slurm/bake.sh`'s env preamble and its `--gres` note verbatim — a CPU-only job routed to
`def-maxwl_cpu` effectively never starts.

---

## 7. Tests — all of these must exist and pass

1. **Round-trip.** Synthetic biosample → write → read → counts and peaks exactly equal;
   pval within 0.005 of source.
2. **Thinning.** `E[thin(c, d)]/c → 1/d` and `Var` matches `c·(1/d)(1−1/d)` over ≥1e6 draws;
   `thin(c,d) <= c` always.
3. **Eval determinism (D22).** Two independent reads of the same eval window with different worker
   counts and batch orders produce **byte-identical** thinned counts.
4. **Mask.** No eligible window has `mask.mean() < min_valid_frac`; no eligible window is all-N.
5. **Declared order (D14).** The regime's assay list maps to the right columns; an assay absent
   from the store raises naming it; a permuted list produces permuted columns (not a silent pass).
6. **dtype rule (D7).** A synthetic value > 65535 forces `uint32` and round-trips exactly.
7. **pval codec (D9).** `-log10 p` of 0, 161.2 and 655.35 round-trip within 0.005; above 655.35
   clips and the clipped fraction is recorded.
8. **Batch contract (§5).** `StoreDataset` emits exactly the key set `CandiKitH5Dataset` emits,
   with matching shapes and `MISSING`/`CLOZE` semantics.
9. **Old path untouched.** `pytest tests/ -q` green and `python tools/golden.py` bit-exact after
   every commit.

---

## 8. Branch and crux discipline

- **One branch for the module:** `implementation/t6-candi-store`, carrying `t6`, `t7`, `t8`.
  `git push -u origin implementation/t6-candi-store` **at creation**, not at the end.
  The PR body says it serves t6–t8; note the multi-task branch as a deliberate deviation
  (they are one module and one review).
- `t4`, `t5`, `t10`, `t11`, `t12` need **no branch**.
- Close each task with `crux task done tN --output "<real output>"` — the engine refuses a `done`
  with no resolving output. `crux task accept` is **the PI's signature — never run it.**
- Run crux from **inside `cruxvault/`**: `python /Users/mforooz/.claude/skills/crux/scaffold/crux.py …`
- `crux validate --check=tree,tasks` after any task change.

## 9. Environment

- `conda activate candii` before `pytest` — base torch is broken by a duplicate `LC_RPATH` libgfortran.
- Fir/Nibi go through the `hpc` command. **Always `-o BatchMode=yes` on raw `ssh`/`rsync`.**
- Fir python for store work: `module load StdEnv/2023 python/3.11 scipy-stack` then
  `source ~/scratch/enctest_env/bin/activate` (already built, has h5py 3.16).
- **Datasets never come to the laptop.**

**If Fir is down (exit 69):** do **not** try to authenticate. Do every local task that does not need
the cluster — `t6`, `t8`, all tests, the `STORE.md` doc — commit and push them, and leave a clear
note saying which cluster tasks are queued and why. Report at the end; do not block the night on it.

---

## 10. Explicitly out of scope tonight

- **The MERGED train/test split** (D17) — the PI deferred it.
- **The genome-wide eval output policy** — overlap and edge handling when emitting a full imputed
  track. This is a *question* for the crux tree, not a task, and the tree is still empty.
- **Deleting the old bake path.** `prep/bake.py` stays until a real training run has used the store.
- **Installing a working `hdf5plugin`/zstd on Fir.** Nice to have, not tonight.
