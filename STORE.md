# STORE.md — the corpus store contract

Owns: the CANDI_STORE on-disk layout, the three codecs, the `n_bins` rule, the root attrs, the
manifest and its metadata authority, the CLI, and the traps — each with the symptom you will
actually see.
The **old** window-materialized bake → `DATA.md`. Metric contract → `EVAL.md`. Invariants, tasks,
gates → `AGENTS.md`. The decisions behind every choice here, numbered `D1`–`D24` → `STORE_PLAN.md`.

Anchors are `file.py::symbol`, never line numbers. Source of truth is the code; where this file
disagrees with it, the code is right and this file is the bug.

**The store does not replace the bake yet.** `prep/bake.py` and `dataset.py::CandiKitH5Dataset`
are the training path and are untouched (D21). `candi.store` lands beside them.

## What changed, in one sentence

The bake wrote **windows**; the store writes **chromosomes**, and everything that used to be frozen
into the h5 — window size, context length, DSF factor, chromosome split, assay column order — moves
into a regime file read at load time. One store, every regime, no re-bake.

```
ENCODE-style npz tree --[store.writer]--> CANDI_STORE --[store.reader + regime]--> batches
```

## On-disk layout

```
CANDI_STORE/
  genome/                      # shared across corpora (t7, not built yet)
    dna.h5                     /chr1 …  (chr_len,)  uint8 codes A=0 C=1 G=2 T=3 N=4
    mask.h5                    /chr1 …  (n_bins,)   uint8 0/1
    chrom_sizes.json           {"chr1": 248956422, …}
  eic/
    manifest.json              generated — never hand-edited
    biosamples/
      T_DND-41/
        counts.h5              /chr1 …  (n_bins, n_tracks)  uint16 | uint32
        peaks.h5               /chr1 …  (n_bins, n_tracks)  uint8
        pval.h5                /chr1 …  (n_bins, n_tracks)  uint16, scale 100
  merged/                      same shape
```

Every path in that tree comes from `store/layout.py` — `layout.py::kind_path`,
`::manifest_path`, `::biosample_dir`, `::genome_dir`, `::corpus_genome_dir`. Nothing in the package
joins a path by hand, and neither should anything downstream.

`corpus_root` is `CANDI_STORE/<corpus>`, e.g. `CANDI_STORE/eic`. That is the handle the reader takes
(`CorpusStore("/…/CANDI_STORE/eic")`); `genome/` is its **sibling**, not its child.

### One dataset per chromosome, whole and contiguous

`(n_bins, n_tracks)` per (biosample, kind, chromosome) — D1 and D3. HDF5 chunking is an internal
compression block, **not** a window: any `[start:stop]` is addressable at the same cost, so one read
serves a whole window across every assay at once.

The `biosample → experiment → kind → chrom` tree people think in lives in the **Python API**
(`reader.py`, t8), not in HDF5 group paths.

### Chunk and codec are fixed and measured

`(1024 bins, all tracks)`, gzip level 4, every kind (`layout.py::dataset_kwargs`). Measured on real
`T_DND-41` chr1: gzip4 is 22% smaller than gzip1 at identical read latency. lzf is 2.2× larger.

**Blosc and zstd are not available on Fir** — the Compute Canada `hdf5plugin` wheel is a stub with
no filters registered. Do not add a codec here that the cluster cannot read back.

## The three kinds and their codecs

| kind | source dir | dtype | rule |
|---|---|---|---|
| `counts` | `signal_DSF1_res25/chr*.npz` | `uint16` or `uint32` | **per FILE**, from the biosample's global max (D7) |
| `peaks` | `peaks_res25/chr*.npz` | `uint8` | D8 |
| `pval` | `signal_BW_res25/chr*.npz` | `uint16` | fixed point, `round(-log10p * 100)` (D9) |

`counts` is **DSF1 only** (D6). DSF2/4/8 are not stored; the loader thins binomially, which is
exactly what the source DSF levels were (verified to 3–4 decimals against `Binomial(k, 1/d)`).

### The counts dtype is one per file, chosen by a full pre-pass

`writer.py::_scan_counts_max` reads every counts npz of the biosample once **before** writing
anything, and `layout.py::counts_dtype_for_max` turns that global max into `uint16` (max ≤ 65535)
or `uint32`. The dtype goes in the `dtype` root attr and in the manifest; the reader upcasts, so a
`uint16` store and a `uint32` store are interchangeable downstream.

That pre-pass doubles the read cost of a build. It buys one dtype for the whole file instead of one
per chromosome, which is what keeps `reader.py` from special-casing every slice. `--counts-dtype`
skips the scan when a previous run already told you the answer.

`B_DND-41`/DNase-seq reaches **52,051** reads in one 25 bp bin — under `uint16`, but the reason the
rule exists rather than a hardcoded width.

### The pval codec is lossy by design, and the bound is 0.005

`layout.py::encode_pval` / `::decode_pval`. `round(-log10 p * 100)` clipped to `[0, 65535]`, so the
resolution is 0.01 and the ceiling is **655.35**. Measured quantization error on the real corpus:
max **0.005** on `-log10 p`, against a chr1 observed max of 161.2. Storage: 0.771 B/bin against
2.727 for the source float32, a 3.5× win.

Stored in the **original space** — no arcsinh at bake time. Every transform belongs to the model,
the same rule `DATA.md` states for counts. This is a deliberate departure from the old bake, which
arcsinh'd pval on the way in.

Clipping is counted per track and written to the `pval_clip_frac` root attr and into the manifest.
`+inf` is a real `-log10 p` (`p == 0`) and clips to the ceiling; **NaN is refused** — see the traps.

## `n_bins` is `floor`, for every kind, always

`layout.py::n_bins_for` — `n_bins = chr_len // resolution`, and `layout.py::fit_to_n_bins`
truncates a source array that is one bin longer.

The counts npz is `ceil(chr_len / resolution)` and the pval and peaks npz are `floor` — a real
one-bin difference measured on the reference corpus, and the source of a documented trap in
`DATA.md`. Storing the floor everywhere kills it outright: every kind, every chromosome, one length.

Anything that is neither `n_bins` nor `n_bins + 1` raises `layout.py::LengthMismatch` naming
`<biosample>/<track>/<kind>/<chrom>`. There is no tolerant mode.

## Root attrs — every file is self-describing

Written by `layout.py::root_attrs`, read back by `layout.py::read_root_attrs` (which decodes the
JSON-encoded ones — do not touch `.attrs` directly).

| attr | kinds | contents |
|---|---|---|
| `schema` | all | `1` |
| `biosample` | all | the opaque id, verbatim |
| `kind` | all | `counts` / `peaks` / `pval` |
| `resolution` | all | `25` |
| `tracks` | all | JSON list — **the column order of this file** |
| `control_col` | all | int column index of `chipseq-control`, or `-1` |
| `dtype` | all | e.g. `uint16` |
| `n_bins` | all | JSON dict `{chrom: n_bins}` |
| `source_root` | all | the npz tree it was built from |
| `kit_version`, `built_utc` | all | provenance |
| `dsf` | counts | always `1` (D6) |
| `npz_depth` | counts | JSON list aligned with `tracks`; `depth` as the npz tree reported it, `null` where absent |
| `scale` | pval | always `100` |
| `pval_clip_frac` | pval | JSON list aligned with `tracks` |

### Storage column order is not model column order

`layout.py::order_tracks` sorts assays alphabetically and puts `chipseq-control` last. That is
**storage** order only. The model's assay order is **declared in the regime file** (D14) and mapped
onto these columns by name; the old derived, availability-sorted order and its bijection asserts are
gone.

### The control is a normal column

`chipseq-control` is a column of `counts.h5` like any other, flagged by `control_col` (D18).
`peaks.h5` and `pval.h5` have no control column and carry `control_col = -1`, because the source
corpus gives the control neither peaks nor a p-value track.

### Biosample names are opaque ids

Used verbatim, everywhere (D16). Nothing in this package parses a `T_` / `V_` / `B_` prefix — the
old first-letter filter silently accepted `vagina_nonrep`, `BJ_nonrep` and `BE2C_grp1_rep1` as
train/val/blind biosamples. EIC's prefixes remain meaningful **by convention only**.

### Nothing is filtered

Everything on disk is stored, RNA-seq and the whole long tail included (D15). The
`if "RNA-seq" in exps: exps.remove(...)` that ran at three loading sites in the old handler does not
exist here. Selection is the regime file's job, at load time.

## `manifest.json` — the corpus-level record

Built by `manifest.py::build_manifest`, written by `::write_manifest`. **Generated, never
hand-edited**: regenerate it instead.

```json
{
  "schema": 1, "corpus": "eic", "resolution": 25,
  "genome": {"build": "GRCh38", "fasta_sha256": "…", "n_bins": {"chr1": 9958256}},
  "assay_vocabulary": ["ATAC-seq", "CTCF", "DNase-seq", "…"],
  "kinds": ["counts", "peaks", "pval"],
  "biosamples": {
    "T_DND-41": {
      "dtype": "uint16", "control_col": 9,
      "kinds": ["counts", "peaks"], "chroms": ["chr1", "…"],
      "tracks": [{"assay": "H2AFZ", "col": 0, "depth": 24684534, "read_length": 36,
                  "run_type": "single-ended", "file_accession": "ENCFF764MJJ",
                  "exp_accession": "ENCSR…", "bios_accession": "ENCBS…",
                  "lab": "…", "platform": "…", "assembly": "GRCh38",
                  "npz_depth": 24684534, "pval_clip_frac": 0.0,
                  "kinds": ["counts", "peaks"]}]
    }
  },
  "source_root": "…", "built": {"kit_version": "…", "utc": "…"},
  "metadata_csvs": ["…"], "metadata_gaps": [{"biosample": "…", "track": "…",
                                             "field": "read_length", "reason": "absent"}]
}
```

The **structural** facts — track order, `dtype`, `control_col`, `n_bins`, `pval_clip_frac` — are
read straight off the h5 root attrs, because the h5 is the record. The CSVs supply only the
per-track experimental metadata.

`assay_vocabulary` excludes `chipseq-control`: it is a column, not an assay, and the regime file's
`assays` list should not contain it.

### The CSVs are the authority; `file_metadata.json` must agree

D20. `manifest.py::read_metadata_csvs` merges one or more CSVs into a
`(biosample_name, assay_name) -> fields` table; `manifest.py::read_file_metadata` flattens the
per-track json; `build_manifest` compares `assay`, `accession`, `read_length` and `run_type` and
raises `manifest.py::MetadataConflict` listing every disagreement.

`--metadata-csv` is repeatable. That is the whole mechanism for t5's `control_metadata.csv` — it has
the same columns as the signal CSVs and joins the same table, with no code path of its own.

`--no-strict` downgrades a conflict to a warning plus a `metadata_gaps` entry. It exists for triage.
Do not build a store you intend to train on with it.

### Nothing is fabricated

D19. A field that is missing, unparseable, or contradicted by a second CSV row is written as an
explicit `null` and listed in `metadata_gaps`, so the encoder's `readlen_missing_emb` and
`depth_missing_emb` get the case they exist for. The old path's `read_length = 50` and
`run_type = single` fallbacks are gone, and so is the hard-coded replicate key `"2"` — whatever
replicate key a `file_metadata.json` field carries is the one used, and more than one raises.

## CLI

```bash
python -m candi.store build-biosample \
    --source-root /project/6014832/mforooz/DATA_CANDI_EIC \
    --corpus-root /project/def-maxwl/mforooz/CANDI_STORE/eic \
    --chrom-sizes /…/CANDI_STORE/genome/chrom_sizes.json \
    --biosample T_DND-41 --kinds counts,peaks

python -m candi.store build-manifest --corpus-root …/eic --corpus eic \
    --metadata-csv EpiDenoise/data/eic_metadata.csv \
    --metadata-csv …/CANDI_STORE/control_metadata.csv \
    --source-root /project/6014832/mforooz/DATA_CANDI_EIC

python -m candi.store verify --corpus-root …/eic
```

`--biosample` is repeatable and defaults to every biosample under `--source-root`, which is why one
SLURM array task per biosample needs no merge step. `--chrom-sizes` falls back to
`<corpus_root>/../genome/chrom_sizes.json`. `build-genome` is a stub that raises pointing at t7.

Every writer output goes through a `.tmp` and an `os.replace`, so a task killed mid-write leaves no
half-file for the next run to read.

## Traps

Each entry gives the symptom you will actually observe.

### A rebuild refuses instead of overwriting

`writer.py::_write_kind` raises `… already exists; pass --overwrite to replace it`. A SLURM array
re-run over an already-built corpus therefore fails on every task rather than silently rewriting a
store something else may be reading. Pass `--overwrite` when you mean it.

### One missing chromosome kills the whole biosample, on purpose

**Symptom:** `T_X/counts: 3 missing source npz … First few: [('DNase-seq', 'chr8'), …]`.

`writer.py::_resolve_chroms` requires every selected track to have every selected chromosome. A
ragged track set would give a matrix whose columns disagree about what a row means — a
`(n_bins, n_tracks)` block is only meaningful when every column covers the same interval. Fix the
source, or name the chromosomes you actually want with `--chroms`.

### A NaN in a pval track stops the build

**Symptom:** `N NaN (non-finite) values in a pval track; refusing to invent a number for them.`

There is no number that means "no p-value", and writing 0 would mean `p = 1` — a confident claim of
no signal. `--pval-nan zero` does it anyway, counts those bins in `pval_clip_frac`, and is the wrong
answer unless you have checked what produced the NaNs. `+inf` is **not** affected: `p == 0` is a real
measurement and clips to 655.35 like any other overflow.

### `pval_clip_frac` above zero means the ceiling is real for your data

The manifest records it per track. A nonzero value says some bins hit 655.35 and lost their true
magnitude. Everything below the ceiling is still good to 0.005, so a small fraction is harmless —
but a track at 0.01 or higher is a track whose top-of-signal is flattened, and any evaluation that
weights extreme p-values will read low with no other symptom.

### A missing `chrom_sizes.json` looks like a flag error, not a data error

**Symptom:** `--chrom-sizes is required (and …/genome/chrom_sizes.json does not exist).`

`n_bins` cannot be derived from an npz — the whole point of D13 is that the npz length is ambiguous
by one bin. Until t7 writes `genome/chrom_sizes.json`, pass `--chrom-sizes` explicitly; a
two-column `hg38.chrom.sizes` TSV works as well as the JSON (`layout.py::load_chrom_sizes`).

### Two chrom_sizes files produce a store that cannot be described

**Symptom, only at manifest time:** `n_bins[chr2] = 4001 but another file says 4000. The store was
built against two different chrom_sizes.`

Nothing at write time notices, because each biosample is built independently. `build_manifest` is
where it surfaces. Rebuild the odd biosample against the same chrom sizes as the rest.

### `peaks` above 255 is a source problem, not a dtype problem

**Symptom:** `… values span [0, 300] which does not fit uint8`.

`peaks_res25` is a 0/1 indicator. A peaks npz carrying scores rather than calls is the usual cause,
and widening the dtype would store the wrong quantity in the right container.

### The npz array key is ignored

`writer.py::_load_npz` reads `data.files[0]` and nothing else, exactly as `handler.py::_load_npz`
does. An npz holding two arrays silently stores the first one. Write one array per npz.

## What is not here yet

| module | task | what it owns |
|---|---|---|
| `genome.py` | t7 | FASTA → `dna.h5` (D10); FASTA + ENCODE hg38 blacklist v2 → `mask.h5` (D11, D12) |
| `reader.py` | t8 | `CorpusStore` / `BiosampleStore` — the OO API |
| `regime.py` | t8 | regime file parse, validate, window plan (D14, D12) |
| `dataset.py` | t8 | `StoreDataset`: window sampling, binomial thinning (D6), the batch dict |

`StoreDataset` must emit exactly the key set `dataset.py::CandiKitH5Dataset` emits, with the
`MISSING = -1` / `CLOZE = -2` semantics of `_vendored.py` preserved, so `train.py`, `batch.py` and
`eval.py` need no changes.
