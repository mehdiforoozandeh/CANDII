# PVAL_CODEC_PLAN — arcsinh fixed point for the store's pval layer

**Status:** **approved by the PI on 2026-08-21.** Every decision below is settled; do not re-open them.

**Progress.** **t26** and **t24** are implemented and pushed (`implementation/t26-signal-target-transform`
→ `be754bb`, `implementation/t24-arcsinh-pval` → `7e1d01f`), each with `pytest tests/ -q` green and
`tools/golden.py check` bit-exact. **t25**'s scripts and its verification gate are written and tested
(`implementation/t25-pval-rebuild` → `4562deb`) but **nothing has been submitted to Fir**: §5 blocks
on t24 merging to `main`. See §8 for the two things the implementation forced that this plan did not
anticipate.

The PI ruled on four points: the fixed-point scale is **2000**; the loader stays in raw `-log10 p`
space and the signal head's target transform moves into the training loop as a configurable option
(D30); **both** corpora are rebuilt in one pass; and the rebuild **overwrites in place**.

This plan amends `STORE_PLAN.md` **D9** only. Every other decision in that document stands, and this
file does not edit it — `STORE_PLAN.md` is the approved record of what was built, and the record of
what was wrong with it lives here and in the vault.

---

## 1. The defect, as measured

`STORE_PLAN.md` D9 encodes pval as `round(-log10 p * 100)` in `uint16`, so the representable ceiling
is **655.35**. Measured on the built EIC corpus (`eic/biosamples/*/pval.h5` root attrs, 2026-08-20):

| | |
|---|---|
| pval tracks in EIC | 363 |
| tracks with ≥1 clipped bin | **62 (17.1%)** |
| clipped bins | **3,046,724 of 44,010,731,292 (6.9e-05)** |
| worst single track | `B_SJCRH30/H3K4me3` — 0.371% of its bins |
| worst assays | H3K4me3 (1.95 M bins), H3K27ac (0.99 M), ATAC-seq (**7 of 7 tracks**, 91 k bins) |

The bin fraction is tiny; the placement is not. Every clipped bin is a peak summit, which is the most
informative position in the track. A truth value of 17,731 is stored as 655.35 — a 96% error on
exactly the bins a signal benchmark weights hardest.

Two things follow that are **not** defects, and are recorded here so they are not re-litigated:

- **The ceiling was never silent.** Every `pval.h5` root carries `scale`, `dtype` and a per-track
  `pval_clip_frac`, and `manifest.json` republishes them. `STORE_PLAN.md` §7 item 7 required the
  clipped fraction be recorded, and it was. A consumer that reads only datasets and not root attrs
  will conclude the units are undocumented; they are not.
- **CANDI's own eval never scored this.** `EVAL.md` — "Only the count head is scored." The Gaussian
  signal head is an auxiliary training loss. The cost of the defect is borne by anything that scores
  the pval track itself, and by the signal head's training target.

### 1.1 A second, independent defect found alongside it

The old bake stores **`arcsinh(-log10 p)`** (`prep/handler.py:1121`, via `prep/reference_sample.py`),
and `decoder.py::GaussianSignalHead` documents that contract: *"the BAKE already transformed … nothing
downstream may transform it again."* The store keeps the **raw** space per D9, and
`store/dataset.py:515` writes it into `y_pval` untransformed. Nothing bridges the two, so the same
head sees a target reaching ~10 off the bake and up to 655 off the store, and the loss at
`train.py:399` consumes whichever it gets without looking.

**This is not fixed by the codec change and does not depend on it.** D30 settles it in the training
loop, and it is tracked separately as **t26** — which can land before t24.

---

## 2. Decisions

| id | decision |
|---|---|
| **D25** | **pval is stored as `round(arcsinh(-log10 p) * scale)` in `uint16`, `scale = 2000`.** Ceiling `sinh(65535/2000)` ≈ **8.5e13** — unreachable. Quantization is `|ε| ≤ 1/(2·scale) = 2.5e-4` in arcsinh space. |
| **D26** | **The codec is invisible above the reader.** `BiosampleStore.pval()` returns `sinh(q/scale)` as float32 — ordinary `-log10 p` space. D9's "every transform happens in the model" is unchanged — see D30 for where that transform now lives. |
| **D27** | **`SCHEMA_VERSION` 1 → 2, with a `transform` root attr** (`"arcsinh"` \| `"linear"`). A schema-1 file has no `transform` and is read as `"linear"` at its own recorded `scale`. **The reader accepts both; the writer emits only schema 2.** |
| **D28** | **`pval_clip_frac` stays and must read exactly 0.0 everywhere after the rebuild.** It stops being routine bookkeeping and becomes an alarm. |
| **D29** | **Rebuild from source, both corpora in one pass, overwriting in place. Transcoding the existing `pval.h5` is forbidden** — it would faithfully preserve all 3,046,724 clipped bins. Both sources are present: `DATA_CANDI_EIC` and `DATA_CANDI_MERGED`. EIC, MERGED and the slice all move to schema 2 together, so the store never carries two codecs beyond the rebuild window. |
| **D30** | **The signal head's target transform is a training-loop option, never the loader's job.** The loader returns raw `-log10 p` from both paths. `--signal-target-transform {none,arcsinh,log1p}` is applied inside the loss when the signal head is on. Its default is derived from the data source — `none` for `--h5` (the bake pre-transforms) and `arcsinh` for `--store` — and the resolved value is recorded in the run config, so a run's target space is never inferred. |

### 2.1 Why `scale = 2000`, and what it costs

Quantize in arcsinh space, invert on read, and the Jacobian `cosh(y)` amplifies the error by exactly
the factor the compression removed. What survives is **constant relative precision** — an integer
code that behaves like a floating-point one:

```
q = round(arcsinh(x)·S)      x̂ = sinh(q/S)      |q/S − arcsinh(x)| ≤ 1/(2S)
x̂ − x ≈ ε·√(1+x²)   →   (x̂ − x)/x → ε   for x ≫ 1
```

At `S = 2000`, `ε = 2.5e-4`:

| truth `-log10 p` | new abs err | new rel err | today (linear ×100) |
|---|---|---|---|
| 0.1 | 0.00025 | 0.25% | 0.005 → **5%** |
| 1 | 0.00035 | 0.035% | 0.005 → 0.5% |
| 10 | 0.0025 | 0.025% | 0.005 → 0.05% |
| 100 | 0.025 | 0.025% | 0.005 → 0.005% |
| 655 | 0.16 | 0.025% | 0.005 → 0.0008% |
| **17,731** | **4.4** | **0.025%** | **clipped → 17,076 (96%)** |

So: ~20× finer below `-log10 p` of 1, about 2–5× coarser between 100 and 655, and the truncation
gone. Because the model re-applies `arcsinh`, its training target sees the flat ±2.5e-4 and no `cosh`
amplification at all — the round trip cancels.

**Size, measured** on real chr20 tracks re-encoded at the store's own `(1024, n_tracks)` chunking and
gzip4:

| codec | `B_SJCRH30` (6 tracks) | `T_H1-hESC` (22 tracks) |
|---|---|---|
| linear ×100 (today) | 13.23 MB | 53.76 MB |
| arcsinh ×1000 | +4.8% | +3.7% |
| **arcsinh ×2000** | **+7.8%** | **+6.2%** |
| arcsinh ×5000 | +11.7% | +9.8% |
| float16 | +6.7% | +4.1% |

`S = 2000` costs the same bytes as float16 and matches its 0.025% precision, while never clipping and
admitting no `inf`/`NaN` into the file. Corpus-wide that is roughly **+20 GB on 289 GB**.


### 2.2 Why a scale at all, when `arcsinh` already compresses

They are two different jobs and `uint16` needs both. **`arcsinh` compresses the range** — it folds
0 … 17,731 into 0 … 10.5, and that is what kills the truncation. **The scale quantizes** — HDF5
stores integers, `arcsinh(x)` is a real, and the scale is the ruler that maps one onto the other.
`scale = 2000` means 2,000 integer steps per unit of arcsinh.

Drop the scale and the file must hold a float instead. `arcsinh` + float16 works, but float16's
precision is relative, so its steps are uneven across the range — 0.0005 near arcsinh 1, 0.008 near
10.5 — where `uint16` at scale 2000 is a flat 0.00025 everywhere in the same two bytes, and cannot
carry `inf` or `NaN` into the file. `arcsinh` + float32 is uniform but doubles the pval layer from
289 GB to 578 GB.

So the scale is the precision-against-disk dial, spending 65,536 codes across ~10.5 units of range:

| scale | codes used of 65,536 | step (arcsinh) | measured size |
|---|---|---|---|
| 1000 | 10,500 | 0.0005 | +3.7 to +4.8% |
| **2000** | **21,000** | **0.00025** | **+6.2 to +7.8%** |
| 5000 | 52,500 | 0.0001 | +9.8 to +11.7% |

Spending more codes is not free, which is the non-obvious part: more distinct integers means higher
entropy per chunk, so gzip does worse. That is measured on chr20, not assumed. **2000 matches
float16's precision at float16's size, and never clips.**

**Rejected: float16.** Its precision is relative at 2⁻¹¹, so between 100 and 655 — where real peaks
live — it is coarser than what we have today. It also represents `inf`, and `p == 0` occurs; `arcsinh(inf)`
is a `NaN` loss. Integers cannot carry that hazard into the file.

---

## 3. The code change (t24)

Branch `implementation/t24-arcsinh-pval`, pushed at creation. **Must not touch** `prep/bake.py`,
`dataset.py` or anything `tools/golden.py` reads — D21 still holds, and the merge gate for a
non-experiment lane is `pytest tests/ -q` green **and** `python tools/golden.py check <ref.pt>`
bit-exact.

### `src/candi/store/layout.py`
- `SCHEMA_VERSION` 1 → 2.
- `PVAL_SCALE` 100 → 2000; add `PVAL_TRANSFORM = "arcsinh"` and `ATTR_TRANSFORM = "transform"`.
- Replace `PVAL_MAX_ENCODABLE` — under D25 it is `sinh(65535/scale)`, so it must be computed from
  `(scale, transform)`, not be a module constant tied to the linear codec.
- `encode_pval(values, scale=PVAL_SCALE, nan_policy="error", transform=PVAL_TRANSFORM)` — apply
  `np.arcsinh` before `np.rint`. `+inf` still clips (it is genuinely infinite); NaN policy unchanged.
- `decode_pval(encoded, scale=PVAL_SCALE, transform=PVAL_TRANSFORM)` — `np.sinh(q/scale)` for
  `"arcsinh"`, `q/scale` for `"linear"`, returning float32.
- `root_attrs(...)`: for `kind == "pval"`, write `ATTR_TRANSFORM` beside `ATTR_SCALE`.
- Update the D9 docstrings in place. They currently assert "no arcsinh at bake time" and a 0.005 error
  bound; both become wrong.

### `src/candi/store/reader.py`
- `BiosampleStore.pval()` (line 324) reads `transform` from the file's own attrs, defaulting to
  `"linear"` when absent, and passes it to `decode_pval`. **This is the whole of D27's back-compat**
  and the one place a schema-1 file is still understood.

### `src/candi/store/writer.py`
- No logic change; it already delegates to `encode_pval` and aggregates `n_clipped`. Confirm the
  clip accounting still totals correctly when nothing clips.

### `src/candi/store/manifest.py`
- Carry `transform` through into the per-track manifest records beside `pval_clip_frac`, so the
  manifest alone answers "what are the units".

### `src/candi/store/cli.py`
- Add `--pval-scale` (default `L.PVAL_SCALE`) and `--pval-transform` (default `arcsinh`, choices
  `arcsinh,linear`) to `build-biosample`, so a rebuild is a flag and not an edit.

### `src/candi/config.py`, `src/candi/train.py`, `src/candi/decoder.py` — **t26, separate branch**
Per D30, and **independent of the codec**. `signal_target_transform: Literal["none","arcsinh","log1p"]`
in the config — named apart from the existing `signal_transform`, which is the ENCODER'S INPUT
transform and is a different thing; `decoder.py:278` already warns the two are easy to conflate.
Apply it at `train.py:399`, inside the `has_signal` branch only. Update the `GaussianSignalHead`
docstring, which currently forbids exactly this.

**Golden stays bit-exact** because the `--h5` default is `none`: the bake path's arithmetic does not
change. That is the reason the default is source-derived rather than a single constant.

### Tests — `tests/test_store_writer.py`, `tests/test_store_reader.py`
1. Round trip at `-log10 p` ∈ {0, 0.1, 1, 655.35, 17731, 2.4e5}: relative error ≤ 2.5e-4 above 1,
   absolute error ≤ 2.5e-4 below 1.
2. A synthetic track reaching 17,731 encodes with `n_clipped == 0`.
3. A hand-built schema-1 fixture (`scale=100`, no `transform`) still decodes linearly through
   `BiosampleStore.pval()` — the regression that D27 exists to prevent.
4. `root_attrs("pval", …)` contains `transform`, and `SCHEMA_VERSION == 2`.
5. `+inf` clips to the ceiling; `NaN` still raises under `nan_policy="error"`.
6. **Existing test to fix:** `tests/test_store_writer.py:241` asserts the round trip is within
   `0.005` absolute. That bound is the linear codec's; under D25 it becomes a relative bound.

---

## 4. Documentation

- **`STORE.md`** — the contract doc, and the only one that must be exactly right. Update the pval
  codec section with D25–D28, both tables from §2.1 and §2.2, and the ceiling. Add a trap entry:
  *symptom — a pval track reads back with values above 655 where an older copy of the same track
  capped;* cause — schema 2 vs schema 1, and the `transform` attr is how you tell which you have.
- **`AGENTS.md`** — §2.3 (`src/candi/store/`) gains a line that pval is stored transformed and the
  reader inverts it. §4 gains a recipe for reading the codec off a file's attrs.
- **`DATA.md`** — the old bake stores `arcsinh` already; add the one sentence that says so explicitly
  and points at `STORE.md` for how the store differs. This is the seam that caused §1.1.
- **`AGENTS.md` §4 and `EVAL.md`** — D30's `--signal-target-transform`, what its two source-derived
  defaults are, and the rule that follows from them: **two runs are comparable on the signal head
  only if their resolved target transform matches**, exactly as `EVAL.md` already says of `--heads`.
- **`STORE_PLAN.md`** — **not edited.** D9 stays as the record of what was approved and built.
  Add one line under its status block pointing here.
- **`cruxvault/`** — record the outcome on t24/t25/t26 and link the verification report under
  `cruxvault/results/t25/`.

---

## 5. The Fir runs (t25)

Sync first — `hpc push CANDII fir` — so `KIT=/home/mforooz/projects/def-maxwl/mforooz/CANDII` carries
the merged t24 code. **Do not start any of this before t24 is merged**; a half-updated kit produces a
corpus in two codecs with no way to tell which biosample got which.

Reuse `$STORE/t12/t12_build_pval.sh` verbatim except for the job name and log paths — it already
takes `CORPUS`, `SRC` and `LIST` from the environment, already carries the MIG `--gres` line that
keeps a CPU-shaped job off `def-maxwl_cpu`, and already passes `--overwrite` (D29). Copy it to
`$STORE/t25/t25_build_pval.sh`.

Add `--pval-scale 2000 --pval-transform arcsinh` to the `build-biosample` call **explicitly**, even
though §3 makes them the defaults. A 455-file rebuild that silently inherited a default is a rebuild
whose codec you have to go and measure afterwards; passing them puts the choice in every task's
sbatch log.

| job | array | env | est. |
|---|---|---|---|
| A — EIC pval | `0-88%15` | `CORPUS=eic SRC=/project/6014832/mforooz/DATA_CANDI_EIC` | 89 tasks, ~2.1 node-h |
| B — MERGED pval | `0-360%15` | `CORPUS=merged SRC=/project/6014832/mforooz/DATA_CANDI_MERGED` | 361 tasks, ~8.4 node-h |
| C — slice pval | `0-4` | `CORPUS=eic_slice SRC=…DATA_CANDI_EIC` | 5 tasks, keeps the slice consistent |
| D — manifests | 2 tasks | `python -m candi.store build-manifest` for `eic` and `merged` | `scale`/`transform`/`clip_frac` are republished there |
| E — verify | 1 task | the scan in §6 | the gate |

Budget from t12's actuals: 450 tasks, mean 84 s, max 319 s, 4 CPUs, `--mem=32G`, **0 failures**, 10.57
node-hours total. Expect the same plus the ~7% larger writes. Wall clock at `%15` is **30–60 min**.

`--overwrite` replaces each file in place, so the transient disk cost is one file, not 289 GB; the
steady-state growth is ~+20 GB. `/project` has 12 TiB free of 28 TiB, so neither is a constraint.

**During the rebuild the corpus is mixed** — some biosamples schema 2, some schema 1. That is safe to
read (D27) but not safe to *train* on, because precision differs between biosamples. Do not start a
run against the store until E is green.

---

## 6. What "done" means

`t25` closes when a single scan over all 455 `pval.h5` files reports, for every file:

1. `schema == 2` and `transform == "arcsinh"` and `scale == 2000`.
2. `pval_clip_frac` is exactly `0.0` on every track (D28). **Any nonzero value fails the gate** —
   it means a source value exceeded 8.5e13, which would mean something is wrong with the source, not
   with the codec.
3. A spot round trip on ≥5 tracks, including `B_SJCRH30/H3K4me3` and one ATAC-seq track: decode the
   store and compare against the source npz, asserting relative error ≤ 2.5e-4 above 1 and the
   previously-clipped summits now present at full height.
4. `crux validate --check=tree,tasks` clean, `pytest tests/ -q` green, `tools/golden.py` bit-exact.

The report and the scan JSON go to `cruxvault/results/t25/`.

---

## 7. Risks

- **Half-built corpus.** Mitigated by gating on t24's merge and by §6's scan. The scan is what makes
  a partial rebuild visible instead of silent.
- **Schema-1 readers elsewhere.** Confirmed on 2026-08-20: `L.decode_pval` has exactly two call
  sites, `reader.py:325` and one test. Nothing bypasses `BiosampleStore.pval()`, so D27's
  back-compat has a single point of enforcement.
- **Scale is a one-way door.** Changing `PVAL_SCALE` again later means another 10.6 node-hours. 2000
  is chosen with a ceiling ~5e9× above the largest value anyone has reported, so headroom is not the
  thing that would force a second pass — precision would be, and 0.025% is already below the pval
  track's own noise.
- **The signal head's numbers move.** Every recorded pre-CANDII signal-head number was produced
  against the `arcsinh` bake. After this change the store path agrees with them, which is the point,
  but nothing store-trained before it is comparable. That is `AGENTS.md` §7.2 territory, not a bug.

---

## 8. What the implementation forced that this plan did not anticipate

Two decisions had to be made while building t24 and t14. Neither re-opens anything above; both are
recorded here because a reader of §2 would otherwise find code that does more than the plan says.

### 8.1 D27's schema bump needed a MEMBERSHIP check, not an equality one

`SCHEMA_VERSION` is stamped on **every** kind, not just pval, and `manifest.py::verify_store` and
`reader.py` both compared it for **equality**. Bumping it to 2 would therefore have made every
existing `counts.h5`, `peaks.h5`, `dna.h5` and `mask.h5` fail validation — and t25 rebuilds the pval
layer only, so a correct corpus is **mixed by construction**.

So `SUPPORTED_SCHEMA_VERSIONS = (1, 2)`: the writer emits only 2, the readers accept either. An
equality check would have turned a pval codec bump into a whole-corpus rebuild of every kind, which
is a far larger operation and buys nothing. `verify_store` gained one new rule in exchange — a
schema-2 pval file that carries no `transform` attr is a **problem**, because its units are then
unrecoverable, which is the entire point of D27.

### 8.2 D31 — the imputation pairing is declared (t14)

Not a codec decision, but it landed in the same pass and belongs on the record. `eval.py` scores
imputation by prompting with one biosample and comparing against a **different** one; the bake finds
that second biosample by string surgery on the first (`T_X` → `V_X` / `B_X`), and **D16 forbids that
here** — store biosample names are opaque ids.

**D31: the pairing is a declared `eval_pairs: [[input, target], …]` in the regime file.** Optional and
absent by default, so every regime written before t14 is unchanged. The regime refuses a pair naming
one biosample twice, a repeated pair, and a target that also appears in `biosamples.train`.

`log_ref` — the sixth key on t14's list — is deliberately **not** emitted, and `eval.py`'s own dataset
factory is still h5-only, so `train.py --store` still skips `evaluate()`. Both are filed: **t28**
(wire the factory) and **t29** (decide what a store-backed reference table even is).

### 8.3 A third defect, found while measuring t16

Unrelated to the codec, and recorded here only because it changes how the store-vs-bake comparison
must be read. `StoreDataset._thin` seeds each deterministic draw from
`(run_seed, biosample, assay, chrom, window_start, dsf_milli(dsf))` — with **no x/y term** — so under
D22's counter-based eval RNG, `x_dsf == y_dsf` yields the *same draw* and the store reproduces the
bake's full 1-in-4 identity-copy leak rather than the 1-in-16 it does while training. Latent today,
because `eval.py` pins `dsf_sampling="off"`. Filed as **t27**, PI-gated: fixing it moves every
deterministic eval number. Measurement in `cruxvault/results/t16/REPORT.md`.
