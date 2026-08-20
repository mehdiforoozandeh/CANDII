# AGENTS.md — the agent-facing spec for CANDI

**Owns:** the file map, the invariants, the common tasks, the validation gates, the failure-mode
table, and the frozen pre-CANDII science in §7.

Everything else lives elsewhere and is not restated here: how we work → `CLAUDE.md`. What CANDI is
and the full tunable surface → `README.md`. The input contract and h5 schema → `DATA.md`. What the
metrics measure and the rules for quoting a number → `EVAL.md`.

**Where this file and the code disagree, the code is right and this file is the bug.** Anchors are
`file.py::symbol`, never `file.py:NN` — a symbol survives an edit above it, a line number does not.

---

## 1. ORIENTATION

CANDI is **one architecture, not a registry** (`model.py` module docstring). A grouped Conv1D tower
over the assay tracks plus a dense Conv1D tower over DNA, fused linearly, through a RoPE
transformer; then a grouped deconv mirror of the signal tower with per-assay per-layer FiLM and a
depth-offset log-linked Negative Binomial head. ~2.35 M parameters.

What it is **not**:

- **Not a pretrained model.** No weights ship. Training is from scratch.
- **Not a vendored kit.** Earlier documents in this lineage described `candi/` as a self-contained
  copy of a "q19 dual-conditioning" recipe built around `RealDualCondModel`, `DualCondDecoder`,
  `build_real_model` and a `compat.py` bit-exactness module. **None of those exist.** The model is
  `model.py::CandiModel` with `decoder.py::SymmetricDecoder`, and the gate is `tools/golden.py`. If
  you find a document naming the old symbols, it predates this repo.
- **Not the 56.2 M predecessor.** The decoder it replaces ran an *ungrouped* trunk at
  `num_assays * feat_per_assay * 2**3` channels — a vendoring accident that put 93.7% of the
  parameters in one dense conv tower. Mirroring the encoder instead dropped the model to 2.35 M and
  improved both imputation and denoising.

---

## 2. FILE MAP

### 2.1 `src/candi/` — the training path

| file | lines | owns |
|---|---|---|
| `model.py` | 263 | `CandiModel`, `build_model`, the FiLM-tap vocabulary (`parse_film_taps`), and the NB helpers `forward_full` / `nb_mean` / `nb_nll` / `encode_latent`. |
| `encoder.py` | 1252 | `MetadataEmbedding`, `SignalConvTower`, `MaskTokenInjector`, `DNAConvTower`, `LinearFusion`, `V2Encoder`. The `arcsinh` transform lives here, not in the loader. |
| `decoder.py` | 593 | `SymmetricDecoder` — grouped deconv trunk at constant lane width, four FiLM taps, the NB head, plus the optional `GaussianSignalHead` and `PeakHead`. |
| `dataset.py` | 497 | `CandiKitH5Dataset` and the depth-centre helper. Reads all scale from `h5.attrs`. |
| `batch.py` | 160 | masking, the obs/imp maps, the control append, the availability cross-check. |
| `train.py` | 1479 | the CLI, the training loop, and the run JSON. |
| `eval.py` | 1532 | the M1/M2/M3/S14 instrument. Contract in `EVAL.md`. |
| `metrics.py` | 146 | numeric primitives only — `nb_crps`, `calibration_pit_curve`, `ece`, the correlations. |
| `param_groups.py` | 418 | the decay/no-decay partition, asserted exhaustive and disjoint. |
| `precision.py` | 134 | the bf16 autocast fences. Evaluation is never autocast. |
| `reference.py` · `probes.py` · `meta_probe.py` · `healthcheck.py` · `compare_arms.py` · `verify_bake.py` · `report.py` · `report_h74.py` | | reference table, diagnostic probes, pre-training health check, arm comparison, bake verification, figures. |
| `_vendored.py` | 354 | `MISSING=-1`, `CLOZE=-2`, `DataMasker`. **RNG-load-bearing** — `apply_mask` draws from the global torch RNG every step. Do not reorder, do not add draws. |

### 2.2 `src/candi/prep/` — the bake path (needs the `[prepare]` extra)

`bake.py` (522) writes the h5 and runs the post-bake `_verify` gates. `handler.py` (2000) is the
ENCODE-side reader. `reference_sample.py` (191) derives and asserts the assay order.
`panel.py` validates the panel JSON and rejects unknown keys. Schema and layout → `DATA.md`.

### 2.3 `src/candi/store/` — the corpus store (needs no extra; h5py is a core dep)

`layout.py` owns every on-disk rule in one place — paths, chunking, the per-file counts dtype, the
fixed-point pval codec, the floor `n_bins` rule — and is the only module here that does not import
h5py. `writer.py` turns one biosample's npz tree into `counts/peaks/pval.h5`; `manifest.py` builds
`manifest.json` with the metadata CSVs as authority; `genome.py` builds `dna.h5` and `mask.h5` and
owns window eligibility; `reader.py` is the `CorpusStore → BiosampleStore → TrackView` API;
`regime.py` parses the regime file and generates the window plan; `dataset.py` is `StoreDataset`.
`cli.py` exposes the build half as `python -m candi.store`.

This lands **beside** the bake, not on top of it (D21). The contract is `STORE.md`; §4.8 and §4.9
are the recipes.

### 2.4 Tests, tools and jobs

`tests/` is CPU-only and needs no GPU and no data. `tools/golden.py` is the bit-exactness gate;
`tools/peak_check.py` checks the peak head. `slurm/` holds the sbatch scripts — the directory is
named `slurm/` and **not** `jobs/` because an unanchored `jobs/` rule in a parent `.gitignore` would
swallow it. Avoid `models/` and `logs/` for the same reason.

---

## 3. INVARIANTS — check yourself against this list after every edit

1. **The control channel is never masked.** `batch.py::prepare_masked_batch` masks first and
   concatenates the control column at index `A` afterwards, so it is structurally unmaskable. Inputs
   are `A+1` wide, targets `A` wide.
2. **`obs` / `imp` mean unmasked / masked**, never biological. `masked_map` is `x_data == CLOZE`;
   `observed_map` is neither sentinel. The loss adds the two means unweighted.
3. **`arcsinh` lives in the model, not the loader.** The loader yields raw counts; the transform is
   applied inside `encoder.py::V2Encoder`. Transforming in `dataset.py` or `batch.py` is a silent
   double-arcsinh.
4. **Counts stay raw non-negative integers as NB targets.** The bake's `_verify` F7 check rejects an
   already-transformed range; `tests/test_bake_gates.py` guards the loader side.
5. **Scale is a property of the baked file, never a CLI flag.** `num_assays`, `context_bins`,
   `resolution`, `dsf_list`, the assay order and the chromosome split all come from `h5.attrs`
   (`dataset.py`). Never add flags for them.
6. **The assay order is derived and asserted, never declared.** There is no assay-name string literal
   under `src/`. `prep/reference_sample.py` asserts a bijection against the requested panel and that
   `assay_to_id == range(A)` with `control_assay_id == A`, then records the order in `h5.attrs`.
   A hard-coded assay list anywhere is a defect.
7. **Covariates stay raw.** The four rows are `[log2_depth, assay_id, read_length, run_type]` with no
   normalization. Sentinel detection is exact equality against `-1` and `-2`, so any normalization
   scheme must be range-guaranteed to avoid both. The `assay_id` table is `Embedding(num_assays + 3)`
   — real ids, control, MISSING, CLOZE. **Do not "fix" the `+3` to `+2`.**
8. **Never write a `0`-filled metadata row.** An absent DSF level or unavailable assay is `-1`. A
   zero-filled row marks every assay available at `log2(depth) = 0` with all-zero counts: the loss
   descends and every metric is garbage. Guarded by the dataset's availability check and by bake
   `_verify` F3.
9. **`d_model = 0` means auto**, deriving the transformer width from the panel — so width silently
   tracks assay count. Attention inner dim is `nhead * 64` regardless of `d_model`, so capacity does
   **not** scale with the panel unless `--nhead` rises. Set `--d-model` explicitly when
   `num_assays != 8`.
10. **Determinism pattern:** `cudnn.deterministic=True`, `cudnn.benchmark=False`, then
    `torch.manual_seed(seed)` immediately before the build with nothing touching the RNG in between.
    `transformer_layer_drop` must remain `0.0` — it consumes global RNG inside `forward`.
    `torch.use_deterministic_algorithms` is deliberately **not** used; it crashes this encoder.
11. **Anything purely additive and default-off must be constructed LAST in its parent module.** An
    `nn.Linear` draws from the global RNG before any re-init overwrites it, so a module inserted
    mid-constructor re-samples everything built after it — and the arm then differs from its control
    in a re-sampled trunk as well as in the thing under test.
12. **No imports reaching back into a parent repo.** Nothing under `src/` may contain `from sandbox`,
    `from model import`, `from _utils import`, or `SANDBOX_ASSAYS`.
13. **The SLURM GPU spec is fixed.** Every `#SBATCH --gres` line reads
    `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`. The bake does no GPU compute; it requests the
    smallest MIG slice only to route through the GPU account.
14. **`--heads` changes the objective, not just the architecture.** `count` is required and is the
    only head `eval.py` scores. `signal` and `peak` add loss terms supervised on full-depth targets
    only, so a run naming either is **not** a control for a run that does not.

**Never infer what a tracker id means.** `h34`, `h45`, `q19` and friends appear in prose throughout
this lineage. Those nodes live in the read-only archive vault (§7); CANDII-era ids live in
`cruxvault/`. If an id resolves in neither, report it as unresolvable rather than reconstructing it
from context — confidently-interpolated provenance is how the retracted `Δη = 0.833` reached three
separate records.

---

## 4. COMMON TASKS

### 4.1 Install

```bash
pip install -e .                          # train-only deps
pip install -e ".[prepare,report,test]"   # + bake, figures, pytest
```

On Alliance Canada build the venv from `requirements-fir.txt` first — it documents the exact
`module load` → `--no-index` → pinned-`x-transformers` sequence — then `pip install -e . --no-deps`.

### 4.2 Run the tests

```bash
pytest tests/ -q
```

CPU only, no GPU and no data required. Run it from a directory that is not a parent repo; that is
part of the proof the package is cwd-independent.

### 4.3 Bake a panel

Copy `configs/panel.example.json`. Every key is consumed and unknown keys are **rejected**, so
`--type2-ccre`, `--type2-non` and `--seed` are bake **CLI flags**, not panel fields. Requirements:
≥2 `T_` biosamples each with ≥2 available assays, `context_bins` divisible by `pool_size**n_cnn_layers`,
`resolution` a perfect square, `train_chroms` disjoint from `eval_chroms`. Full schema → `DATA.md`.

### 4.4 Train an arm

```bash
python -m candi.train --h5 <panel.h5> --out-dir <runs> --epochs 10 --steps-per-epoch 2000
```

Writes `{out_dir}/{tag}.json` and `{out_dir}/{tag}.ckpt`. The full flag surface — geometry,
normalisation, FiLM taps and init, head sharing, LR schedule, clip, precision — is tabulated in
`README.md`, and every flag defaults to the shipped model.

### 4.5 Evaluate a checkpoint

```bash
python -m candi.eval --h5 <panel.h5> --ckpt <run.ckpt> --arch-from <run.json>
```

**Always pass `--arch-from`.** Every architecture flag changes the `state_dict`, and that file
carries the exact arguments the checkpoint was built from, so nothing has to be retyped and nothing
can be retyped wrong. A mismatch is a strict-load failure at best, and at worst a model that loads
and is quietly not the one you trained. `--reference` and `--depth-center` still need passing by
hand; see `EVAL.md`.

### 4.6 Add a knob

1. Add the keyword to `model.py::CandiModel.__init__` **with the current behaviour as its default**.
   The arch keys are read off the signature, so the knob joins `config.arch` — and therefore
   `--arch-from` — without a second edit.
2. Add the argparse flag in `train.py` and thread it through into the arch dict. Do not read globals.
3. Add it to `tests/test_flags.py`: to the defaults set (its default must be a bit-exact no-op) and
   to the non-defaults set (flipping it must change the model). **The second is the one that
   matters** — an exposed, documented, inert flag has shipped here before.
4. Run `python tools/golden.py check <ref.pt>`. It must report 0 ULP. If it cannot, the change is not
   a knob: it is a new model, and it needs its own recorded golden and a commit that says so.
5. If it is scale, it is not a knob — it belongs in the panel and the h5 attrs (invariant 5).

### 4.7 Change the panel size

Scale flows `configs/panel*.json` → handler → derived assay order → `h5.attrs` → dataset → `build_model`.
Edit `assays`/`biosamples` and re-bake; **nothing in `src/` changes.** Then train with an explicit
`--d-model` (invariant 9), and re-run `tests/test_model.py` and `tests/test_bake_gates.py`, which
already cover 3/8/16 assays and 384/768/1536 bins.

### 4.8 Read the store, and train off it

The corpora are built: `/project/def-maxwl/mforooz/CANDI_STORE` holds `eic/` (54.94 GB, 89
biosamples, 35 assays) and `merged/` (406.06 GB, 361 biosamples, 47 assays) beside a shared
`genome/`. A regime file replaces the h5 attrs; `configs/regime.eic_smoke.json` and
`configs/regime.equiv.json` are the two that exist.

```python
corpus = CorpusStore("/project/def-maxwl/mforooz/CANDI_STORE/eic")
counts = corpus["T_DND-41"].counts("chr1", 0, 768, assays=["H3K4me3"])   # (768, 1) int32
loader = DataLoader(StoreDataset("configs/regime.eic_smoke.json"), batch_size=None, num_workers=4)
```

`batch_size=None` is required — `StoreDataset` yields whole batches and shards itself.

`train.py` trains off it through ONE flag (t23) — the regime file, not the corpus root, because
the regime already carries `store` as a required key:

```bash
python -m candi.train --store configs/regime.eic_smoke.json --out-dir runs/smoke --d-model 32
```

`--h5` and `--store` (alias `--regime-file`) are **mutually exclusive and exactly one is required**;
argparse refuses both-or-neither. Every dataset on either path is built by the one factory
`train.py::make_dataset(source, mask_regime, …)`, and the path is picked once by
`train.py::DataSource.resolve`.

Three things that will bite, all in `STORE.md` *Using the store*, which owns the detail:

- **`--regime` is NOT `--store`.** `--regime` is the *masking* regime (`type1` / `type2_loci`) and
  predates the store. `type2_loci` is h5-only and is refused with `--store`.
- **A store run is training only.** `StoreDataset` does not emit `y_data_imp` / `y_pval_imp` /
  `y_peaks_imp` / `y_meta_imp` / `imp_biosample_name` / `log_ref`, and `eval.py` reads all six
  through `batch.get(...)` — so an imputation eval would score nothing and not say so (task `t14`).
  `--store` therefore skips `evaluate()`, forces `--eval-every` off, and writes a run json with no
  M1/M2/M3/S14 keys at all rather than empty ones.
- **`--reference on` is refused** on the store path: the table pins itself to an h5 fingerprint.

`cell_cond` is refused rather than defaulted (D16) — a ported training command that passes it stops,
and `num_cells` is 0, so the store trains the historical 4-row model.

The run json records `data_source`, and on the store path also `store`, `regime_file`,
`regime_json` (verbatim), `regime_sha256` and `store_manifest_sha256` (`STORE_PLAN.md` §4).

### 4.9 Build a store

Genome layer once, then one `build-biosample` per biosample, then the manifest, then verify. Full
commands and flags → `STORE.md` *Build a store for a corpus that does not exist yet*.

```bash
python -m candi.store build-genome    --store-root <CANDI_STORE> --fasta … --blacklist …
python -m candi.store build-biosample --source-root <npz tree> --corpus-root <…/eic> --biosample "$B"
python -m candi.store build-manifest  --corpus-root <…/eic> --corpus eic --metadata-csv …
python -m candi.store verify          --corpus-root <…/eic>
```

**Do not write a new array script.** `cruxvault/results/t10/t10_build_eic.sh` built EIC and
`cruxvault/results/t12/t12_build_pval.sh` is its parameterised form; both carry the fixed `--gres`
spec (invariant 13) and sizing justified against t9's measured worst case. Every store command needs
torch, `build-genome` included — `candi/__init__.py` imports the encoder eagerly, so a torch-free
venv cannot run any of them.

---

## 5. VALIDATION

### 5.1 The bit-exactness gate — every change clears it before the next one starts

```bash
python tools/golden.py save <ref.pt>     # record the tree in front of you
python tools/golden.py check <ref.pt>    # rebuild and assert 0 ULP
```

The model is built from one seed on fixed inputs, so **any refactor that only moves code must
reproduce the recording exactly**. A nonzero diff, a changed parameter count, or a changed
`state_dict` key is a failure, never noise. Its reference config is the 35-assay arm
(`A=35`, `d_model=288`), the run these numbers stay comparable to.

Recordings are machine-local by design and gitignored: the point is to record *your* tree and check
your next edit against it, so a committed one would be a stale claim about somebody else's checkout.

If the gate fails after a pure refactor, the cause is almost always construction-order drift — walk
invariant 11.

### 5.2 The test suite

`pytest tests/ -q`. CPU only. `tests/test_flags.py` is the largest single file (1061 lines) and holds
the tunable surface to two claims: every default is a bit-exact no-op, and every flag changes the
model when flipped.

### 5.3 Pre-training health check on real data

`healthcheck.py` runs the checks that need an actual h5 and would otherwise only fail hours into a
training run.

---

## 6. FAILURE MODES AND THEIR SIGNATURES

Every message below is a real string in the source.

| symptom | cause | where to look |
|---|---|---|
| `tools/golden.py check` reports a nonzero diff after a pure refactor | construction-order drift — a module inserted mid-constructor re-samples everything after it | invariant 11; `model.py`, `decoder.py` |
| `... is schema v1; candi requires v2` | h5 baked by an older pipeline | re-bake with `candi.prep.bake` |
| `panel/alias mismatch: missing=[...]` | a panel assay is absent from the data root, or the alias filter dropped it | `prep/reference_sample.py`; check the spelling against the directory |
| `F3 <bios>: meta_dsf<d> zero-filled` | an absent DSF level written as `0` rather than `-1` | invariant 8; bake `_verify` F3 |
| `F7 <bios>: negative counts on an available column` | already-transformed or corrupted counts | invariant 4 |
| `... not the downsampled data they claim to be` | a `counts_dsf{k}` dataset holds DSF-1 data | bake `_verify` F4 |
| `count ... exceeds int16` | counts wider than int16 in the source NPZ | re-bake with a wider dtype |
| `unparseable run_type ...` | a metadata value outside the expected vocabulary | `prep/handler.py` |
| `metadata must be 4 rows` / `assay_id ... exceeds table bound` / `run_type ... exceeds table bound` | a hand-built metadata tensor | `encoder.py::MetadataEmbedding`; deliberate — these previously aliased onto the sentinel slots |
| `DNA length G != context L x resolution` | `resolution`/`context_bins` mismatch between h5 and model | `encoder.py::V2Encoder` |
| `Availability mismatch between counts and peaks` / `... and signal` | the optional heads' targets disagree with the count availability map | `batch.py` |
| `depth_center mismatch: reference ... vs eval ...` | a reference table built at a different depth centre | `eval.py`; the two offsets would compose on different scales |
| `has no config.arch block — it predates --arch-from` | an old run JSON | re-score by passing the architecture by hand, or re-train |
| `parameter(s) landed in BOTH groups` / `in no group` | the decay/no-decay partition is no longer exhaustive and disjoint | `param_groups.py`; `tests/test_param_partition.py` |
| `[data] WARNING: single-DSF ladder` | `dsf_list` has one level, so per-assay independent DSF sampling is inert and the depth signal is gone | `prep/panel.py` |
| loss descends smoothly, metrics are garbage, everything "available" | zero-filled metadata rows | invariant 8 |
| imputation loss is ~0 | biosamples with ≤1 available assay are skipped by the masker, so no cloze is produced | `_vendored.py::DataMasker` |
| run-to-run numbers differ on the same seed | something touched the RNG between `manual_seed` and the build, or `cudnn.benchmark` was re-enabled | invariant 10 |

---

## 7. HOW TO READ THE SCIENCE

**This section is frozen.** Everything in it predates CANDII and is never appended to. Results
produced from CANDII onward live in `cruxvault/` and are never copied back here. See `CLAUDE.md`.

Authority order, highest first. The first two live only in the **read-only archive repo** at
`~/Desktop/research/libbrechteam@sfu/CANDI/`; paths below are relative to that root:

1. `sandbox/diagnostics/dual_conditioning_real/H48_REPORT.md` (2026-07-24/25,
   post-adversarial-verification) — also mirrored at `candi_kit/research/H48_REPORT.md`
2. `candi_kit/research/H48_SCORECARD.md`
3. `research/METADATA_AUDIT.md` — this one ships **in this repo**
4. everything else

Where an older document disagrees, H48_REPORT wins. The archive also holds the old crux vault at
`cruxvault/` (engine 1.3, 82 hypotheses); it is a historical record, not a live one, and no node
from it was migrated into CANDII.

### 7.1 The recorded four-arm results (single seed 0, full chr21 coverage)

| arm | macro CRPS | oracle-scaled (capability) | scale_error | macro Sp | pooled imp Sp | ECE | beats honest marginal | M3 ratio |
|---|---|---|---|---|---|---|---|---|
| `wd0_on_s0` (offset ON, wd=0) | 1.3413 | 1.3077 | 0.0336 | 0.5653 | 0.6372 | 0.0533 | 7/8 | 0.2154 |
| `main_s0_perassay` (offset ON, wd=1e-4) | 1.4950 | 1.4210 | 0.0740 | 0.5051 | 0.5327 | 0.0615 | 5/8 | 0.2443 |
| `offoff_s0_perassay` (offset OFF) | 1.9023 | 1.3871 | 0.5152 | 0.4647 | 0.4007 | 0.0968 | 2/8 | 0.1974 |
| `wd0_off_s0` (offset OFF, wd=0) | 2.0561 | 1.4026 | 0.6535 | 0.4641 | 0.3800 | 0.0782 | 1/8 | 0.2185 |

M1 columns are from the archive's `candi_kit/research/H48_SCORECARD.md` §M1. The **M3 ratio column is carried forward from a
pre-fix run** (the archive's `.BUILD_PLAN.md` anchor table): `H48_REPORT.md` §3 records that M3 was never re-scored,
so it still uses the old permuted labels and a `between` pool that admits same-region pairs. The kit
fixes that pool (`eval.py::eval_M3`), so **your M3 will not be comparable to this column** — quote it
only with that caveat, or not at all.

Steering, sentinel-free and target-clustered (H48_SCORECARD §M2, H48_REPORT F2):

| arm | total told-depth slope | assay real→real max\|Δη\| | run_type clustered CI (n=12) | supports direction | sign-test p |
|---|---|---|---|---|---|
| `wd0_on_s0` | 1.0000 | **0.0023** | [−0.00066, +0.000087] | no | 1.000 |
| `main_s0_perassay` | 1.0000 | 0.0000 | [0, 0] | no | n/a (all ties) |
| `offoff_s0_perassay` | 0.8869 | **4.1772** | [+0.1179, +2.1804] | YES | 0.039 |
| `wd0_off_s0` | 1.0325 | **9.7144** | [−0.2326, +9.4084] | no | 0.039 |

### 7.2 Rules an agent must follow when quoting any of this

1. **Never present the four-arm ordering as established.** H48_REPORT F1 says so explicitly. Under the
   oracle per-assay scale the spread compresses 0.7148 → 0.1133 (84%), and only
   "`wd0_on` is best on capability" survives inference (`offoff − wd0_on = +0.093`, 95% CI
   [+0.004, +0.217]). The other three are statistically indistinguishable; the target-level sign test
   `main` vs `offoff` is 7+/5−, p = 0.77, and P(`main` worst of four) = 0.54. The published 4-dp
   ordering is the modal bootstrap ordering at only 45% of replicates.
2. **Quote the noise floor with every number.** Effective replication is **12 held-out targets / 5
   biosample pairs / 4 cell types**, with `(T_RWPE2, B_RWPE2)` supplying 7 of the 12. Target-clustered
   bootstrap noise floor on macro CRPS is **~0.09**; per-comparison uncertainty ±0.13. A single **seed**
   change moves pooled imputation CRPS by 0.1195, Spearman by 0.0562, ECE by 0.0354, M3 ratio by
   0.0479 — comparable to or larger than several between-arm gaps. Sign-test resolution at n=12 is
   quantized: 10/12 → p=0.039, 11/12 → 0.0063, 12/12 → 0.00049.
3. **The offset head is a real Pareto, not a solved problem.** Offset ON gives the best imputation
   (macro CRPS 1.3413, macro Spearman 0.5653, ECE 0.0533, beating an honest per-assay marginal on 7/8
   assays) but its covariate steering is **functionally null**: sentinel-free real→real assay ablation
   **0.0023**, 43× below its own pre-registered 0.10 bar, and a run_type clustered CI of
   [−0.00066, +0.000087]. Its depth response is **arithmetically exact, not learned**: told-depth
   slope 1.0000 because `log2_mu = (depth − depth_center) + eta` (`decoder.py::SymmetricDecoder`) is a closed-form
   thinning identity — NB is closed under thinning with `n` preserved. Offset OFF gives genuine
   learned steering (assay real→real 4.1772; run_type CI [+0.118, +2.180], sign-test p=0.039) at +42%
   macro CRPS (1.9023) and 2/8 beats-marginal. **State both sides whenever you state either.**
4. **Read the clamp tail next to any slope.** `wd0_on`'s 1.0000 is read partly through the saturating
   `log2_mu` clamp on 16.9% of targets (p90 clamp fraction 0.475). Always emit
   `frac_targets_any_clamp` / `p90` / `max` alongside a slope.
5. **Known bounds — do not design around them without re-selecting the panel.**
   * `run_type` is **analytically unidentifiable** on the shipped 8-assay panel:
     `H(run_type | assay_id, read_length) = 0.000 bits` (n=26 `T_` records). The full EIC panel retains
     0.551 bits. A run_type steering demonstration is **impossible** here; it needs a re-selected
     biosample panel, not an architecture change.
   * DSF only **down**-samples, so the upward-depth regime is never trained, and **7 of 12** eval
     targets sit above their per-assay training depth ceiling (worst +1.43 log2). Depth steering on
     those targets is extrapolation into an untrained direction.
   * In this dataset `assay_id ≡ slot index`, and the decoder trunk emits a dedicated channel block per
     slot, so assay identity is carried structurally. Any assay-steering verdict bounds the **prompt
     pathway**, not assay-awareness.
   * Eval runs `dsf_sampling='off'` with `apply_mask=False`, so `x_data == y_data` for available
     assays and "denoising" is autoencoding. Nothing in the suite measures denoising as distinct from
     autoencoding.
   * `crps_oracle_scaled` is an **in-sample upper bound** — `c*` is fitted on the same 12 targets it
     scores — and `scale_error` can be slightly negative (−0.0008 observed).
6. **Capability of the model, not of this kit's outputs:** the model is counts-only NB. Do not claim
   peak calling, peak precision/recall/AUROC, or p-value tracks.
7. **Metric hygiene.** Only the default-emitted metrics may back a claim. Keys behind
   `--include-deprecated` ship with a verdict string in `eval.py::DEPRECATED_VERDICTS` and must never be quoted as
   evidence: the `read_length` flip arm (7/12 flips are out of training support), the shuffled-depth
   `null` (a mathematical no-op — one `[4,F]` tensor broadcast over the batch, so the permutation is
   bitwise identity), `frac_min_at_true` (0.25 is not chance; it is the deterministic value of
   argmin-always-at-told=1, and a perfect model caps at ~0.73), all position-level `_bootstrap_ci`
   outputs (~24× too narrow), and `median_eta_slope` / `offset_independent` (decided by the sign of
   ~1e-17 float noise under offset-ON). The condition-recoverability probe (S23) is **withdrawn** and
   is not shipped in any mode: its ordering is inverted against every other instrument. `pearson_log1p`
   and `spearman_raw` are computed in different spaces and must never be quoted as a pair.
8. **Float resolution.** GPU-vs-CPU on identical weights reproduces macro CRPS to 4 decimals; per-assay
   values move 2e-6…3.3e-4, and `nb_crps` itself is accurate to ~2.4e-5 relative. Anything at 1e-7 or
   below in `eta`/`mu` is float noise.
9. **Per-assay labels.** Earlier documents in the research repo label per-assay results with a
   *permuted* assay list. The corrected order is the one derived from the handler aliases and stored in
   `h5.attrs['assays']` (38/38 vs 5/38 on a metadata join). Under corrected labels, the largest CRPS
   contributor is **DNase-seq** (not ATAC-seq) and the collapse outlier is **H3K9me3** (not H3K27ac).
10. **Hybrid recovery was not demonstrated.** The upstream hypothesis that a hybrid (annealed /
    α-attenuated offset / learned scale head) recovers both magnitude and steering is recorded
    **REFUTED** in the project's research tracker (`h45`; see the decoder ring in
    the archive's `research/README.md`) — but refuted on its premises, with **no hybrid arm ever trained**, and the
    entry carries a 2026-07-28 flag that its refutation basis is itself under review. Do not present a hybrid
    as either available or ruled out by experiment.

### 7.3 Provenance of the numbers in §7.1

They are **single-seed** scores of four historical checkpoints trained in the research repo and
re-scored on CPU with the corrected instruments (608 eval units, 1215 target-records, 12 held-out
targets). They are anchors for validating that this kit reproduces the same computation — they are not
results this kit has produced, and no weights backing them ship here.
