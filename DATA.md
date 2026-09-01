# DATA.md — the input contract

Owns: the on-disk layout the bake expects, how the four covariates are derived, the sentinels, the
panel file, the HDF5 schema, and the traps — each with the symptom you will actually see.
Metric contract → EVAL.md.  Invariants, tasks, gates → AGENTS.md.  What CANDI is → README.md.

Anchors are `file.py::symbol`, never line numbers. Source of truth is the code; where this file
disagrees with it, the code is right and this file is the bug.

**No checkpoints ship.** You bake an HDF5 from your data and train from scratch.

## Scale flows one way and is never a train-time flag

```
ENCODE-style directory --[prep.bake]--> HDF5 (schema v2, self-describing) --[dataset]--> batches --[train]--> ckpt + run.json
```

`panel.json` → handler → **derived** assay order → `h5.attrs` → dataset → model. `num_assays`,
`context_bins`, `resolution`, `dsf_list`, the assay order and the train/eval chromosomes are
properties of the baked file (`dataset.py::CandiKitH5Dataset`). You cannot silently train a
35-assay model on a 12-assay file.

## Expected on-disk layout

```
<ROOT>/
  aliases.json                       # generated on first bake if absent
  navigation.json                    # generated on first bake if absent
  .candi_state/                      # kit-owned scratch (metadata.csv, split.json); auto-created
  <BIOSAMPLE>/                       # e.g. T_DND-41, V_DND-41, B_RWPE2  (prefix REQUIRED)
    <ASSAY>/                         # e.g. H3K4me3, DNase-seq, ATAC-seq
      file_metadata.json
      signal_DSF{1,2,4,8}_res25/     { chr*.npz, metadata.json }
      signal_BW_res25/               { chr*.npz }   # the assay's signal bigWig, binned — NOT
                                                    # -log10 p for DNase-seq; see below
      peaks_res25/                   { chr*.npz }   # 0/1 peak call mask
    chipseq-control/                 # optional but strongly recommended
      file_metadata.json
      signal_DSF{1,2,4,8}_res25/     { chr*.npz, metadata.json }
```

`res25` is literally `res{resolution}` and follows the `resolution` set in the panel.

### `<ROOT>` must be writable

The bake creates `aliases.json`, `navigation.json` and `.candi_state/` inside the data root on first
use, so pointing at a shared read-only reference corpus — the normal institutional setup — fails.
Copy or symlink into a writable location first. This is separate from the FASTA's own need for a
writable parent so pysam can build its `.fai`.

### Each `.npz` holds one whole-chromosome array

Only `data.files[0]` is read, so the array key is irrelevant (`handler.py::_load_npz`).

| directory | dtype on disk | length (bins) | contents |
|---|---|---|---|
| `signal_DSF{d}_res25/chr*.npz` | integer | **`ceil(chr_len / resolution)`** | **raw, unnormalized read counts** at downsampling factor `d` |
| `signal_BW_res25/chr*.npz` | `float32` | **`floor(chr_len / resolution)`** | whatever ENCODE's signal bigWig for that experiment holds, mean-binned and untransformed on disk. `-log10 p` for ChIP-seq and ATAC-seq; **`read-depth normalized signal` for DNase-seq** |
| `peaks_res25/chr*.npz` | integer | **`floor(chr_len / resolution)`** | 0/1 peak indicator |

### `signal_BW_res25` is not one quantity — 40 of EIC's 363 tracks are not a p-value

ENCODE ships **no** p-value track for DNase-seq. Swept per file against the portal for all 363
EIC signal bigWigs on 2026-08-29: 316 ChIP-seq and 7 ATAC-seq carry `signal p-value`, and 40
DNase-seq carry `read-depth normalized signal` — all released, bigWig, GRCh38. 34 of the 267
training tracks are therefore in different units from the rest of the layer, and any macro that
pools assays pools a probability with a scaled read count.

The accession and its `output_type` are now recorded per track in the store manifest
(`STORE.md`), from `configs/signal_provenance.eic.json`, and a build whose recorded `output_type`
disagrees with the corpus's declared signal semantics fails. The rebuild of the DNase layer from
alignments is `plan/BENCHMARK_DESIGN.md` §10 and crux task `t78`.

The ceil/floor difference is real, not a typo — measured on the reference corpus,
`T_DND-41/H3K27me3/chr19` gives counts `2,344,705 = ceil(58,617,616/25)` against `2,344,704` for BW
and peaks. The loader never validates length; it slices `[start // resolution : end // resolution]`.
Tiled and cCRE-sampled windows sit fully inside `floor(chr_len/resolution)`, so the off-by-one is
benign — but an array short by more than one bin yields a short slice and then a shape error deep
inside `make_region_tensor_*`.

Counts must be raw integers. Do not pre-transform them; `arcsinh` is applied inside the model.

### Biosample names carry their role in a `T_` / `V_` / `B_` prefix

The handler runs in EIC layout mode only (`self.eic = True`). `handler.py::_filter_navigation`
deletes every biosample directory whose name does not start with `T`, `V` or `B`, and
`panel.py` requires the full prefix.

| prefix | role |
|---|---|
| `T_` | training biosample; also the denoising ground truth at eval |
| `V_`, `B_` | held-out views of the *same* cell type; supply the imputation ground truth |

`V_<X>` and `B_<X>` match `T_<X>` by string suffix — `T_K562` ↔ `V_K562` / `B_K562`.

You need **≥ 2 `T_` biosamples, each with ≥ 2 available assays**. Below that the whole-assay cloze
masker skips the sample (`num_available <= 1`, `_vendored.py::DataMasker`) and the imputation loss
is identically zero. Both `panel.py` and `bake.py` assert it.

## Four covariate rows, from three metadata sources

Every assay column carries a length-4 covariate vector — `[4, F]` per biosample per DSF level,
`[B, 4, F]` in a batch. Assembled in `handler.py::make_bios_tensor_Counts`.

| row | field | dtype | source |
|---|---|---|---|
| 0 | `log2(sequencing depth)` | float | `signal_DSF{d}_res25/metadata.json` → `depth`, **per DSF level** |
| 1 | `assay_id` | float holding an integer | derived from `aliases.json` order; **not read from any file** |
| 2 | `read_length` | float (bp) | `<ASSAY>/file_metadata.json` → `read_length` |
| 3 | `run_type` | float, `0` = single-end, `1` = paired-end | `<ASSAY>/file_metadata.json` → `run_type` |

**These four rows are raw — no normalization, no z-scoring.** They reach the model as-is
(`encoder.py::MetadataEmbedding`).

### Source A — `signal_DSF{d}_res25/metadata.json`, per DSF level

```json
{"coverage": 0.3173361622168517, "depth": 24684534, "dsf": 1}
```

Only `depth` is read. Because depth is per-DSF, `meta_dsf{d}[0] == meta_dsf1[0] - log2(d)` holds to
a small tolerance, and `bake.py::_verify` asserts it at `< 0.05` for every available column. That is
the invariant catching a DSF-1 file masquerading as DSF-8.

### Source B — `<ASSAY>/file_metadata.json`, per experiment

**Every value is a dict keyed by replicate index, not a scalar:**

```json
{
  "assay":       {"2": "H3K4me3"},
  "accession":   {"2": "ENCFF764MJJ"},
  "read_length": {"2": 36},
  "run_type":    {"2": "single-ended"},
  "sequencing_platform": {"2": "Illumina HiSeq 2500"},
  "lab":         {"2": "Bradley Bernstein, Broad"}
}
```

The key is arbitrary; the code takes `list(d.keys())[0]`. `len(run_type) == 1` is asserted so that
read is well-defined. **`read_length` is not covered by that assert** — more than one replicate
under `read_length` picks one non-deterministically. Emit exactly one replicate per experiment.

`run_type` parsing is substring-based: the string must contain `single` or `pair`, or it raises.
`sequencing_platform` and `lab` are read and dropped.

For the **control** track the replicate key is hard-coded to `"2"`, with silent fallbacks — missing
`read_length` → `50`, missing `run_type` → single-ended → `0`. Control metadata keyed differently
gives you depth right and read_length/run_type wrong, with no warning.

### Source C — `aliases.json`, the assay_id and the column order

`assay_id` is the index of the assay in `aliases['experiment_aliases']` key order, and the **control
uses `assay_id == num_assays`**, one past the last real assay. That same order is the column order
of every `[*, *, F]` array in the HDF5.

`aliases.json` is generated on first bake if absent (`handler.py::_make_alias`) by listing
`<ROOT>/<BIOSAMPLE>/<ASSAY>/` and ordering assays by descending biosample availability.

## Sentinels, masking, and the `obs`/`imp` naming

| value | name | meaning |
|---|---|---|
| `-1` | `MISSING` | this assay is not available for this biosample |
| `-2` | `CLOZE` | this assay is available but was masked out for this training step |

Defined once in `_vendored.py` and imported everywhere. Written into `x_data`, `x_meta` and
`x_avail` alike.

### Each sentinel gets its own learned embedding, not the number minus one

In `encoder.py::MetadataEmbedding`:

- continuous fields (depth, read_length) — `depth_missing_emb`, `depth_cloze_emb`,
  `readlen_missing_emb`, `readlen_cloze_emb`, separate `nn.Parameter`s substituted *after* the
  linear projection;
- categorical fields (assay_id, run_type) — tables sized `num_assays + 3` and `num_runtypes + 2`,
  the last two rows reserved for MISSING and CLOZE.

So `assay_id` must lie in `[0, num_assays]` and `run_type` in `[0, num_runtypes-1]`. Values above
those bounds would alias onto the sentinel slots, so the encoder raises instead.

### `obs` and `imp` mean unmasked and masked, not biological

Throughout — losses, metric keys, run JSON:

- **`obs` / `observed_map` = positions NOT masked this step**
- **`imp` / `masked_map` = positions that WERE masked this step**

They do not mean biologically observed versus imputed. `imp` metrics score the model on whole assays
cloze-masked out of its input. EVAL.md spells `obs` as `den` in the M1 pools; same side, two
spellings.

### The control channel is index `A` and is structurally unmaskable

`batch.py::prepare_masked_batch` applies the masker to the `F` assay columns first and only then
concatenates the control:

```
x_data_in = cat([x_data_masked, control_data], dim=2)   # -> [B, L, F+1], control at index F
x_meta_in = cat([x_meta_masked, control_meta], dim=2)   # -> [B, 4, F+1]
```

The control is therefore always available to the model. Its covariate row 1 carries
`control_assay_id = num_assays`. The bake refuses a biosample whose control is not available in 100%
of windows unless you pass `--allow-missing-control`.

## The panel file

One flat JSON; every field is consumed; unknown keys are rejected (`panel.py`).

| key | type | default | notes |
|---|---|---|---|
| `assays` | `list[str]` | required | **order-insensitive** (below). Unique and non-empty. |
| `biosamples` | `list[str]` | required | full names with `T_`/`V_`/`B_` prefix; ≥ 2 `T_` |
| `dsf_list` | `list[int]` | `[1,2,4,8]` | load-bearing; a single level makes DSF sampling inert and deletes the depth signal |
| `resolution` | `int` | `25` | must be a perfect square — it is `dna_pool_size ** 2` |
| `context_bins` | `int` | `768` | must be divisible by `pool_size ** n_cnn_layers` |
| `train_chroms` | `list[str]` | `["chr19"]` | must be disjoint from `eval_chroms` |
| `eval_chroms` | `list[str]` | `["chr21"]` | |

Required DNA length per window is `context_bins * resolution` bp; the encoder asserts it at forward
time. The shipped panel is `configs/panel.eic_full.json` — 35 assays × 89 biosamples.

### The assay ORDER is derived from the data and never declared

The order you write in `panel.json` is **not** the column order of the resulting HDF5. The true
order is the post-filter key order of `aliases['experiment_aliases']`
(`reference_sample.py::resolve_column_order`), which `handler.py::_make_alias` establishes as
`data_matrix.count().sort_values(ascending=False)` — assays sorted by how many biosample directories
contain them.

That sort is data-dependent, unstable on ties (pandas defaults to quicksort), and seeded by
`os.listdir` order, so two bakes of the same directory on different filesystems can legitimately
disagree. The kit therefore derives the order, records it in `h5.attrs['assays']`, and asserts a
bijection against the requested panel in `reference_sample.py`:

```python
absent  = sorted(set(panel.assays) - present)     -> ValueError naming the assays
missing/extra vs resolve_column_order(h)          -> ValueError naming both sets
[h.assay_to_id[a] for a in resolved] != range(A)  -> ValueError
h.control_assay_id != len(resolved)               -> ValueError
```

**No assay-name list literal exists anywhere in `candi`.** Every consumer — dataset, eval, report —
reads `h5.attrs['assays']`.

Why the asserts matter: the model only ever sees the integer `assay_id`, so a permuted name list
does not crash and does not change the loss. It mislabels every per-assay number in the results and
every metadata join, and there are `A!` orderings with exactly one that joins cleanly. The bijection
assert is what turns that silent corruption into a loud failure.

## The bake

### Side files you must download

Not in this repo and not in the data directory. Standard public references from UCSC.

```bash
mkdir -p side && cd side

# 1. Chromosome sizes (~12 KB) — always required.
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes

# 2a. Reference genome, only the chromosomes you use. Enough unless you enable type2 loci.
for c in chr19 chr21; do
  wget -qO- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz | zcat
done > hg38_subset.fa
grep -E '^(chr19|chr21)\s' hg38.chrom.sizes > hg38_subset.chrom.sizes

# 2b. Whole genome (3.27 GB compressed) — required for --type2-ccre / --type2-non, which
#     sample loci from every chromosome EXCEPT your train/eval ones.
# wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip hg38.fa.gz

# 3. cCRE regions (~61 MB) — ONLY for --type2-ccre / --type2-non.
wget -O GRCh38-cCREs.bed https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.bed
```

A 2-chromosome FASTA is ~107 MB instead of 3.27 GB and bakes noticeably faster. The bake fails
loudly on a missing chromosome, so this is safe to try.

| flag | required | notes |
|---|---|---|
| `--chrom-sizes` | yes | two columns, `chrom\tlength`; must cover every chromosome in the panel |
| `--fasta` | yes | needs a `.fai` beside it, or a writable parent so pysam can build one. **There is no counts-only path that skips DNA** — the encoder always builds and calls the DNA tower. |
| `--blacklist` | no | plain BED, `chrom start end`; `#` comments skipped |
| `--ccres` | only with `--type2-ccre`/`--type2-non` | 6-column BED; requires a whole-genome FASTA |

**Genome build.** Everything assumes hg38 / GRCh38. Counts called against another build need the
matching FASTA and chrom-sizes — nothing detects a mismatch, and the DNA tower would silently read
the wrong sequence.

### CLI

```bash
python -m candi.prep.bake \
  --root        /path/to/ENCODE_STYLE_ROOT \
  --panel       configs/panel.eic_full.json \
  --out         /scratch/$USER/candi/eic_full.h5 \
  --fasta       /path/to/hg38.fa \
  --chrom-sizes /path/to/hg38.chrom.sizes \
  --blacklist   /path/to/hg38_blacklist_v2.bed \
  --seed 42
```

Optional: `--ccres FILE --type2-ccre N --type2-non N` (adds cCRE-centred and random non-cCRE loci,
writes a `<stem>_loci_type2.bed` sidecar); `--max-tile-per-chrom N` / `--max-windows N` for smoke
runs; `--allow-missing-control`. `configs/panel.example.json` is a 3-assay smoke panel that finishes
in minutes with `--max-tile-per-chrom 200`.

`--out` has no default and must point at `/scratch` or another large scratch filesystem.

### HDF5 schema v2

Let `n` = windows, `L = context_bins`, `Lbp = L * resolution`, `F = len(assay_order)`. All
per-biosample datasets are chunked `(1, ...)` with gzip level 1.

**Root attrs:** `version` (int `2` — the dataset refuses v1), `context_bins`, `resolution`,
`data_root`, `kit_version`, `assays` (**the derived column order, the authority**), `assay_ids`
(`[0..F-1]`, asserted), `requested_panel` (what you wrote), `num_assays`, `control_assay_id` (`F`),
`dsf_list`, `train_chroms`, `eval_chroms`, `panel_json` (the full panel, verbatim).

**Windows,** each `(n,)`: `/windows/chrom` (vlen utf-8), `/windows/start` and `/windows/end` (int64
bp, `end = start + L*resolution`), `/windows/region_type` (uint8: `0` cCRE, `1` non-cCRE, `255`
tiled).

**Biosamples.** `/biosamples` carries attr `order`, a json sorted list of full names. Per group
`/biosamples/<NAME>/`:

| dataset | shape | dtype | notes |
|---|---|---|---|
| `counts_dsf{d}` (one per `d`) | `(n, L, F)` | `int32` | **raw integer counts**, never arcsinh'd |
| `pval` | `(n, L, F)` | `float16` | `arcsinh(-log10 p)`, always at DSF 1 |
| `peaks` | `(n, L, F)` | `int64` | 0/1 mask |
| `control` | `(n, L, 1)` | `float32` | control counts, DSF 1 |
| `control_meta` | `(n, 4, 1)` | `float32` | `-1` when the control is absent |
| `dna` | `(n, Lbp, 4)` | `int8` | one-hot ACGT |
| `meta_dsf{d}` (one per `d`) | `(4, F)` | `float32` | `[log2 depth, assay_id, read_length, run_type]`; **`-1`, never `0`**, for an unavailable assay or absent level |

`meta_dsf{d}` has **no window axis** — there is exactly one natural depth per (biosample, assay).

### The post-bake gate runs before the file gets a real name

`bake.py::_verify` runs on the temp file, so a poisoned bake is never renamed into place.

| check | assertion |
|---|---|
| F3 | rows 0-2 of every `meta_dsf{d}` are not all-zero for any column |
| F4 | `abs(meta_dsf{d}[0,c] - (meta_dsf1[0,c] - log2(d))) < 0.05` for every available column |
| F7 | counts are non-negative and `max > 50` somewhere — a double-transform would collapse the range |
| F15 | control availability per biosample is printed; `< 100%` aborts unless `--allow-missing-control` |

`CandiKitH5Dataset.__init__` adds load-time checks: refuses a v1 file, raises when a regime produces
zero training windows, warns on a single-level DSF ladder, and verifies that every column marked
available in `meta_dsf1` actually carries a nonzero count over the first few windows — the tripwire
for the zero-vs-`-1` poisoning below.

### Cost, and why `/scratch`

`dna` dominates and does **not** scale with assay count; `counts_dsf*` do. `pval` and `peaks` are
carried whether or not the heads that consume them are enabled, and `peaks` is stored as `int64` for
a 0/1 mask — an 8× waste that is real but not on the critical path. The shipped 35-assay × 89-biosample
`eic_full.h5` bakes to roughly 24 GB.

The bake monkeypatches a per-`(biosample, chromosome)` memoization onto the chromosome-level loaders.
**This is load-bearing**: without it `reference_tensors` re-reads every `.npz` per window and the
bake goes from minutes to hours (millions of NPZ reads).

## Traps

Each entry gives the symptom you will actually observe.

### Counts are `int32`, and `int16` silently clipped real data

`counts_dsf{d}` and the control channel are `int32`. The research pipeline this came from used
`int16` and silently clipped every bin above 32,767 — a ceiling real data exceeds:
`B_DND-41`/DNase-seq reaches **52,051** reads in a single 25 bp bin in a cCRE-sampled window. Any
bake including such a window wrote corrupted counts with no warning.

`int32` removes the ceiling and keeps the overflow guard at the wider bound, so it can never again
be silent:

> `ValueError: count N exceeds int32 for <bios>/<assay>; re-bake with a wider dtype`

If you see that, you have a bin above 2,147,483,647 reads, which almost certainly means the `.npz`
is not raw per-bin counts.

`int32` doubles the counts footprint versus `int16`, but counts are sparse and gzip-compressed, so
the effect on the h5 is much smaller than 2×.

### `arcsinh` is applied inside the model, never by the counts loader

- `counts_dsf*` in the HDF5 and `y_data` out of the loader are **raw integers**.
- The encoder applies `arcsinh` to the signal tower input.
- The NB likelihood is evaluated against **raw integer targets**.
- `pval` **is** arcsinh'd at bake time — the one exception. **The store is not.** `STORE.md`'s pval
  layer holds `-log10 p` and its reader returns that, so the same `y_pval` key means two different
  quantities depending on which path filled it. That seam is why the signal head's target transform
  is a training-loop flag with a source-derived default (`--signal-target-transform`, D30) rather
  than a constant — see `EVAL.md` *Only the count head is scored*.

**Symptom of applying it twice:** nothing crashes, loss descends, macro CRPS collapses to a
plausible small number on a completely different scale, and every result is incomparable to anything
else. The F7 gate catches the gross case; a transform applied *downstream* of the bake is not caught
at all. Do not transform your counts before writing the `.npz`.

### RNA-seq is unconditionally excluded from the count matrix

At all three loading sites — `handler.py::_get_count_experiments`, `::load_bios_BW`,
`::load_bios_Peaks`:

```python
if "RNA-seq" in exps:
    exps.remove("RNA-seq")
```

`chipseq-control` is removed at the same sites and re-enters through the dedicated control path.

**Symptom:** listing `RNA-seq` in `panel.assays` fires the bijection assert with
`panel/alias mismatch: missing=['RNA-seq']`, or the column is never populated. There is no flag to
include it.

### `must_have_chr_access` is unreachable code that reads like a live filter

It drops any biosample lacking both ATAC-seq and DNase-seq. It defaults to `False` and is set only
inside `setup_datalooper`, which is never called. On the EIC-only path (`self.eic = True`) the
branch containing it is unreachable — the `T_`/`V_`/`B_` prefix filter takes the `if` arm first.
Recorded because the code is visible: re-enable the non-EIC path and it will silently delete
biosamples with no error message.

### An empty blacklist BED disables blacklist filtering silently

`data/hg38_blacklist_v2.bed` in the research repo is 8 bytes containing the single line `# Empty`.
`handler.py::_load_blacklist` skips `#` lines, so it parses to zero intervals and **no filtering
happens at all**. Omitting `--blacklist` does the same. The warning is printed once, on stdout:

```
WARNING: blacklist empty (None) -> 0 regions excluded
```

**Symptom if you miss it:** the bake succeeds, training succeeds, and a small number of windows sit
on high-signal artifact regions, inflating both loss and evaluation. Nothing else tells you.

The blacklist is consulted only through `handler.py::_is_region_allowed`, i.e. for **type-2 locus
sampling**. Plain chromosome tiling of `train_chroms` / `eval_chroms` does not consult it.

### `meta_dsf` must be `-1`, never `0`, for an absent level

The availability test is literally `float(meta[0]) != -1.0`. A zero-filled metadata row therefore
marks **every** assay available at `log2(depth) = 0`, `assay_id = 0`, `read_length = 0`, with
all-zero counts. The bake writes `-1` and the loader tripwires:

> `ValueError: T_X/H3K4me3 is marked available in meta_dsf1 (log2 depth 0.0) but counts_dsf1 is
> all-zero over 4 windows — the h5 is poisoned ...; re-bake.`

**Symptom without the tripwire:** training proceeds, loss descends, every metric is garbage. This is
why schema v1 files are refused outright.

### DSF only downsamples — there is no upsampling regime

Every `signal_DSF{d}` level is Bernoulli read-retention from the full BAM, so an assay's training
depth support is `[natural_min - 3, natural_max]` in `log2` units. Nothing in the data or the code
ever presents the model with a *deeper* target than it has observed. Two consequences for any
depth-steering claim:

- Held-out targets sitting above their per-assay training depth ceiling are **extrapolation in the
  untrained direction**. Check the ceiling per assay before reading a steering result.
- The DSF levels are **not a nested ladder** — `dsf4 <= dsf2` is False. They are independent draws,
  so `counts ∝ 1/dsf` holds only in expectation.

A `dsf_list` with a single level makes per-assay independent `x_dsf != y_dsf` inert and removes the
depth signal entirely. Both `panel.py` and `dataset.py` warn.

### The exposure covariates are collinear, so attribution among them is not identified

Depth, `read_length` and `run_type` are mutually collinear on a small biosample panel, and
`run_type` can be a deterministic function of `assay_id` and `read_length` — in which case a
`run_type` steering demo is impossible on that panel, as a property of the data rather than of the
architecture. If `run_type` steering matters to you, select biosamples that break the degeneracy and
measure the conditional entropy on **your** panel before designing the experiment.

The measurements behind this were made on the predecessor's 8-assay panel and are not reproducible
in this repository — treat the mechanism as established and the numbers as needing a re-measurement
on whatever panel you bake.

### An absent DSF level is skipped, not silently substituted

Upstream, an assay missing from an explicit DSF map loaded DSF-1 counts and paired them with DSF-1
depth metadata — a lie the depth objective then learned from. The kit skips the assay instead.
**Symptom:** an assay you expected does not appear at that DSF level and is marked `-1`. The F4 gate
would otherwise have caught the mismatch at bake time.

## Bringing your own data

1. **Directory shape** exactly as above. Biosample names start with `T_`, `V_` or `B_`; held-out
   views share the suffix with their `T_`.
2. **≥ 2 `T_` biosamples, each with ≥ 2 assays present.** Fewer and imputation supervision is
   identically zero.
3. **Counts:** one `.npz` per chromosome per DSF level, whole-chromosome, raw integers at
   `resolution` bp, length `ceil(chr_len / resolution)`, single array per file.
4. **DSF levels** produced by actual read subsampling, each with its own `metadata.json` carrying
   the **subsampled** `depth`. If `meta_dsf{d}[0] != meta_dsf1[0] - log2(d)` within 0.05, the bake
   aborts.
5. **`file_metadata.json`** per assay directory with `read_length` and `run_type` as single-entry
   dicts; `run_type` strings must contain `single` or `pair`.
6. **`signal_BW_res{r}/` and `peaks_res{r}/`** per assay, whole-chromosome, length
   `floor(chr_len / resolution)`. The bake reads them unconditionally.
7. **`chipseq-control/`** per biosample with its own `file_metadata.json` (replicate key `"2"`) and
   `signal_DSF{d}_res{r}/`. If you cannot supply it everywhere, pass `--allow-missing-control` and
   accept that the model loses a channel it can never mask.
8. **Do not hand-write `aliases.json` or `navigation.json`.** Let the first bake generate them, then
   never delete them — a checkpoint is only valid against an HDF5 whose recorded assay order matches.
9. **Side files:** real `--chrom-sizes`, real `--fasta` + `.fai`, and a **real** `--blacklist`.
10. **Write the panel JSON** and bake to `/scratch`.

### Assertions that fire when it doesn't

| message (abridged) | source | cause |
|---|---|---|
| `panel assays absent from the data root <ROOT>: [...]` | `reference_sample.py` | an assay in `panel.assays` has no directory anywhere |
| `panel/alias mismatch: missing=[...] extra=[...]` | `reference_sample.py` | resolved alias set differs from the requested panel |
| `assay_to_id is not 0..A-1 in alias order: [...]` | `reference_sample.py` | corrupt or hand-edited `aliases.json` |
| `control_assay_id N != num_assays M` | `reference_sample.py` | same |
| `need >=2 T_ biosamples with >=2 panel assays on disk, got N` | `bake.py` | too few training biosamples |
| `expected exactly one run_type entry for <assay>, got [...]` | `handler.py` | multi-replicate `file_metadata.json` |
| `unparseable run_type '<x>' for <assay>` | `handler.py` | contains neither `single` nor `pair` |
| `count N exceeds int32 for <bios>/<assay>` | `handler.py` | counts are not raw per-bin |
| `F3 <bios>: meta_dsf{d} zero-filled for [...]` | `bake.py::_verify` | metadata row of zeros |
| `F4 <bios>/<assay>: ... not the downsampled data they claim to be` | `bake.py::_verify` | DSF depth ladder violated |
| `F7 <bios>: max count N <= 50 (already transformed?)` | `bake.py::_verify` | counts were pre-transformed |
| `F15 <bios>: control available in only X% of windows` | `bake.py::_verify` | missing control; pass `--allow-missing-control` |
| `<file> is schema v1; candi requires v2` | `dataset.py` | old HDF5 |
| `... marked available in meta_dsf1 ... but counts_dsf1 is all-zero` | `dataset.py` | zero-vs-`-1` poisoning |
| `no training windows: regime=..., train_chroms=[...]` | `dataset.py` | regime/panel mismatch |
| `metadata must be 4 rows [log2_depth, assay_id, read_length, run_type]` | `encoder.py` | wrong covariate tensor |
| `assay_id N exceeds table bound M` / `run_type N exceeds table bound M` | `encoder.py` | out-of-range covariate that would alias onto a sentinel slot |
| `WARNING: blacklist empty (...) -> 0 regions excluded` | `paths.py` | **warning only** |

## What the model consumes

- **Counts drive the objective.** The decoder emits a Negative Binomial `(n̂, p̂)` per assay per bin,
  and the evaluation is NB end to end (EVAL.md). `pval` and `peaks` are baked and carried through
  the batch dict so the Gaussian signal head and Bernoulli peak head can consume them as auxiliary
  training losses; nothing scores them.
- **DNA is mandatory.** The encoder always builds and calls the DNA tower. There is no
  counts-without-sequence configuration.
- **Four covariates, raw:** `[log2 depth, assay_id, read_length, run_type]`, no normalization,
  through FiLM at every conv layer.
