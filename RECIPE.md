# RECIPE — architecture, training procedure, and why each choice is what it is

This is the scientific core of the handoff. It documents the model in `candi` module by module with
exact tensor shapes, states which choices are **load-bearing** (changing them changes the result) versus
**free** (tune at will), and records what has already been tried and refuted so you do not repeat it.

**Read this first:**

- **No checkpoints ship.** You train from scratch. The bit-exactness gate is `tools/golden.py`, which
  records the current tree's forward outputs and `state_dict` digest to a `.pt` **you generate** and
  re-checks them at 0 ULP after an edit. Nothing is committed for it to compare against, so it proves
  self-consistency across your refactor, not agreement with any historical run.
- **The model is counts-only Negative Binomial.** There is no Gaussian p-value head and no Bernoulli
  peak head, therefore no peak precision/recall/AUROC and no p-value track. `y_pval` and `y_peaks` are
  carried through the batch dict (`src/candi/batch.py:152-153`) but never supervised.
- **⚠ Every recorded *number* in §5 and §8 predates this architecture.** They were measured on an
  8-assay q19 panel with an **ungrouped ~3.10 M-parameter decoder trunk** and a single FiLM tap. This
  repository ships a **grouped, constant-lane-width decoder with four FiLM taps at 2,353,634
  parameters** (`tests/test_invariants.py:207-216`). None of the source documents those numbers came
  from (`H48_REPORT.md`, `H48_SCORECARD.md`, `.BUILD_PLAN.md`, `TRADEOFF.md`) ship here, and
  `research/METADATA_AUDIT.md` is a **zero-byte placeholder**. `README.md:67-72` records that the
  current architecture was chosen by a nine-arm, two-seed ladder on the full 35-assay panel — and
  those runs are not reported anywhere in this repository either. Sections §1–§4, §6 and §7 describe
  *this* code and are verified against it; §5 and §8 are retained as inherited context and are marked
  as such.
- **Noise floor.** In the recorded q19 experiments, effective replication was **12 held-out targets /
  5 biosample pairs / 4 cell types**, with `T_RWPE2`/`B_RWPE2` supplying 7 of the 12. The
  target-clustered bootstrap noise floor on macro CRPS was **~0.09**; per-comparison uncertainty
  **±0.13**; a **single seed change** moved pooled imputation CRPS by **0.1195** and Spearman by
  **0.0562**. Do not read 4-decimal orderings as rankings. The full EIC panel this repo now targets
  has 96 target tracks over 38 eval pairs (`configs/panel.eic_full.json`); **its noise floor has not
  been measured.**

---

## 1. Input contract

One training sample is one genomic window. Shapes below use the full EIC panel
(`configs/panel.eic_full.json`: 35 assays, `context_bins=768`, `resolution=25`), which is the scale the
parameter anchor is taken at. `A` = `num_assays`, `L` = `context_bins`.

| tensor | shape (A=35, L=768) | dtype | meaning |
|---|---|---|---|
| `x_data` | `[B, 768, 36]` | float32 | raw integer counts, input side. Columns `0..A-1` = assays, column `A` = ChIP control (appended at `batch.py:128`) |
| `x_dna` | `[B, 19200, 4]` | float32 | one-hot DNA, `context_bins × resolution` bp. The encoder accepts `[B,4,G]` or `[B,G,4]` (`encoder.py:662-669`) |
| `x_meta` | `[B, 4, 36]` | float32 | covariates, input side, incl. the control column |
| `y_meta` | `[B, 4, 35]` | float32 | covariates, **target** side. Control-free by construction — the control column is appended to `x_meta` only (`batch.py:129`) |
| `y_data` | `[B, 768, 35]` | float32 | raw integer counts, target side |
| `observed_map` / `masked_map` | `[B, 768, 35]` | bool | loss masks. **`obs` = unmasked, `imp` = masked (cloze)** — these are *not* biological observed/imputed |
| `log_ref` | `[B, 768, 35]` | float32 | **optional** (h74). Present only under `--reference on`; the key is absent otherwise (`batch.py:156-159`) |

The 4 covariate rows are fixed by the ingestion contract and are **not** configurable:

| row | covariate | type | range |
|---|---|---|---|
| 0 | `log2(sequencing_depth)` | continuous | natural (DSF-1) ~22–28; *as prompted during training* it is lower, because DSF only down-samples (§4.4) |
| 1 | `assay_id` | categorical | `0..num_assays-1` real assays; `num_assays` = ChIP control |
| 2 | `read_length` | continuous | {30, 36, 76, 100, 101} bp on the EIC panel |
| 3 | `run_type` | categorical | {0 = single, 1 = paired} |

**An optional 5th row exists.** Under `--cell-cond {id,random}` the loader appends a `cell_id` row,
constant across the assay axis because it is a per-*sample* fact (`dataset.py:427-437`). It is
**off by default**, and the `off` path is bit-identical to the 4-row model, RNG stream included — the
`cell_embedding` table is constructed only when `num_cells > 0` (`encoder.py:148-153`). Availability is
inferred from **rows 0–3 only** (`encoder.py:755-770`), so a valid `cell_id` in an otherwise-absent
column changes no availability.

Two sentinels, kept distinct everywhere (`src/candi/_vendored.py:15-16`): **`MISSING = -1`** (assay does
not exist for this biosample) and **`CLOZE = -2`** (exists, masked, must be imputed).

`assay_id` embedding indices are `Embedding(num_assays + 3)`: `0..A-1` real, `A` = control, `A+1` =
MISSING, `A+2` = CLOZE (`encoder.py:144`, remapped at `encoder.py:219-228`). The `+3` is correct; the
comment at `encoder.py:143` omits the control slot and is wrong — do not "fix" it to `+2`.

`MetadataEmbedding.forward` **raises** rather than silently aliasing on a malformed input: a
wrong-row-count tensor (`encoder.py:190-196`), `assay_id > num_assays` (`encoder.py:212-218`),
`run_type >= num_runtypes` (`encoder.py:232-238`), and `cell_id >= num_cells` (`encoder.py:255-261`).

---

## 2. Architecture walkthrough

Shapes shown for A=35 (so `num_tracks = A+1 = 36`) at `d_model = 288`.

```
x_meta[B,4,36] ──► MetadataEmbedding #1 ──► meta_embed[B,36,32] ─┬─► FiLM after conv0
                                                                 ├─► FiLM after conv1
                                                                 └─► FiLM after conv2
x_data[B,768,36] ─► zero masked chans ─► SignalConvTower ─► [B,96,288] ─► MaskTokenInjector ─┐
x_dna[B,19200,4] ─────────────────► DNAConvTower (NO metadata) ─► [B,96,288] ───────────────┤
                                                                                            ▼
                                                          LinearFusion ─► [B,96,288] ─► 2× x-transformers (RoPE, NO FiLM)
                                                                                            ▼
                                                                                       z[B,96,288]
                                                                                            ▼
y_meta[B,4,35] ─► MetadataEmbedding #2 (SEPARATE, UNTIED) ─► memb[B,35,32] ──┐               │
                                                                            │      input_proj -> [B,96,35,8]
                                                                            ├──► PerLaneFiLM tap 0
                                                                            │      LaneDeconvBlock ×2 -> [B,192,35,8]
                                                                            ├──► PerLaneFiLM tap 1
                                                                            │      LaneDeconvBlock ×2 -> [B,384,35,8]
                                                                            ├──► PerLaneFiLM tap 2
                                                                            │      LaneDeconvBlock ×2 -> [B,768,35,8]
                                                                            └──► PerLaneFiLM tap 3
                                                                                            ▼
                                                          head_eta / head_n (weight-shared) ─► η, raw_n [B,768,35]
y_meta row 0 ────────────────► arithmetic depth offset (bypasses the embedder) ─────────► log2_mu ─► μ, n, p
```

**Two disjoint metadata pathways, two separate untied embedders.** The encoder embeds `x_meta`
("what the input measurement was"); the decoder embeds `y_meta` ("what measurement to produce"). They
are different `nn.Module` instances with independent weights (`model.py:62-64` vs `decoder.py:222-224`).

### 2.1 Metadata embedder (`encoder.py:104-275`)

`[B, n_rows, F] → [B, F, 32]`. Per-field lift, then late fusion:

| field | branch | line |
|---|---|---|
| `log2_depth` | `nn.Linear(1, 32)` | `encoder.py:135` |
| `read_length` | `nn.Linear(1, 32)` | `encoder.py:136` |
| `assay_id` | `nn.Embedding(A+3, 32)` | `encoder.py:144` |
| `run_type` | `nn.Embedding(num_runtypes+2, 32)` | `encoder.py:146` |
| `cell_id` *(only when `num_cells > 0`)* | `nn.Embedding(num_cells+2, 32)` | `encoder.py:153` |

then `concat(n_rows × 32) → Linear(n_rows·32, 32) → GELU → Linear(32,32) → LayerNorm(32)`
(`encoder.py:156-163`; concat order at `encoder.py:251` is depth, assay, readlen, runtype, with
`cell_id` appended at `:272`).

Sentinels are handled **per field, independently**. Continuous fields project everything and then
*overwrite* at sentinel positions with four learned vectors initialised `randn(32)*0.02`
(`encoder.py:137-140`, applied at `:165-180`). Categorical fields remap the sentinel to a reserved row
before lookup (`encoder.py:219-249`). This is what makes the imputation prompt work: you can hand the
decoder a target's `assay_id`, `depth` and `run_type` while its *signal* is absent.

The wasted `proj(-1)` / `proj(-2)` FLOPs do **not** leak gradient: the in-place `emb[mask] = ...` does
not break autograd (`index_put_` has a defined backward), and with all-sentinel depth the projection
receives no gradient while the sentinel vector does.

### 2.2 Signal conv tower — grouped, per-assay, FiLM after every conv (`encoder.py:505-615`)

`groups = num_tracks = A+1 = 36`, so each track's channels are processed independently. Channel
schedule with `expansion_factor=2`, `n_cnn_layers=3`, `pool_size=2` (`encoder.py:529-546`):

| conv layer | in→out channels | per-track C | sequence length | FiLM projection |
|---|---|---|---|---|
| 0 | 36 → 72 | 2 | 768 → 384 | `Linear(32 → 4)` |
| 1 | 72 → 144 | 4 | 384 → 192 | `Linear(32 → 8)` |
| 2 | 144 → 288 | 8 | 192 → 96 | `Linear(32 → 16)` |

Per-track C does not depend on the panel size, so the FiLM projection widths above hold at any `A`;
only the flat channel counts scale. `SignalConvTower.lane_shapes` (`encoder.py:568-579`) renders this
table at run time and `tests/test_lane_view.py:112` pins it to a real forward pass, so the schedule
cannot drift away from its own documentation.

Each `ConvTower` is `Conv1d(same) → Norm → + 1×1 residual → GELU → MaxPool` (`encoder.py:355-376`).
One **independently learned** `FiLMLayer` is applied after *each* block (`encoder.py:562-566`, applied
at `:605-608`); because each block pools, layer 2 conditions at 8× the receptive field of layer 0.
`pre_film` and `post_film` are `None` under `film_mode="per_conv"`.

FiLM is `x ← x·(1+scale) + shift` where `(scale, shift) = proj(meta_embed)` — **metadata-only** (the
activations are only ever the operand) and **broadcast over all positions** (`encoder.py:396-413`).
There is zero per-position specificity. This is textbook FiLM (Perez et al.).

> **Known caveat, not a bug.** `conv_norm="layer"` — the shipped value (`config.py:36`) — normalises
> over the *full* channel axis, mixing all `A+1` tracks' statistics at every layer. `ConvBlock`
> documents this at `encoder.py:285-300` and is explicit that a grouped tower with `norm="layer"` is
> **not** per-assay, and never has been. `norm="lane"` (`encoder.py:327-335`) is the exact per-lane
> analogue — same statistic, computed within a track — but the encoder's `conv_norm` is **not
> reachable from `build_model`**: `CandiModel` never passes it (`model.py:49-54`), so changing it
> means editing `model.py`. `--lane-norm` is a *decoder*-side flag and does not touch this.

### 2.3 Mask token injection (`encoder.py:460-498`)

After the conv tower and **before** DNA fusion, each track flagged CLOZE or MISSING has its
`d_per_assay=8`-wide channel slice replaced by a learned per-track vector
(`nn.Parameter(randn(A+1, 8)*0.02)`, `encoder.py:478-480`). Availability is derived from metadata rows
0–3 with `.any` (`encoder.py:755-770`), and `_prepare_signal` raises if metadata- and signal-derived
availability disagree (`encoder.py:1046-1051`) — this is the assertion an ad-hoc synthetic batch will
trip.

### 2.4 DNA tower (`encoder.py:618-672`)

Ungrouped, no metadata. `n_cnn_layers + 2 = 5` `ConvTower` blocks; channel schedule
`[4] + exponential_linspace_int(4, signal_dim, 5)` (`encoder.py:636-638`). Under
`dna_pool_order="late"` the first 3 blocks pool by `pool_size=2` and the last 2 by `dna_pool_size=5`
(`encoder.py:642-646`), giving a total stride of `8 × 25 = 200`: `19200 → 96`. Output `[B, 96, 288]`,
matching the signal tower's length and width.

There is **no counts-only path that skips DNA** — `V2Encoder` always builds and calls `dna_tower`.

### 2.5 Fusion and transformer (`encoder.py:679-714`, `encoder.py:996-1007`)

`LinearFusion`: `concat([signal 288, dna 288]) → Linear(576, d_model) → GELU → Identity → Dropout(0.1)`.
(`fusion_norm` defaults to `"none"`, so the norm is `nn.Identity`; `fusion_deep=False`, so
`hidden_projs` is an empty `ModuleList`.)

Transformer: **2 independent x-transformers `Encoder` blocks**, each `depth=1, heads=4,
rotary_pos_emb=True, ff_mult=4, pre_norm=True`, `attn_dropout = ff_dropout = 0.1`. **No FiLM on the
transformer** — `transformer_film_layers` is built only under `film_mode="per_conv_and_transformer"`
(`encoder.py:1017-1022`) and is `None` here, because `CandiModel` pins `film_mode="per_conv"`
(`model.py:53`). Note this differs from the `EncoderConfig` dataclass default (`config.py:45`); the
model's override is what ships. `pooled_meta = meta_embed.mean(dim=1)` is computed at `encoder.py:1105`
but consumed only by that dead branch. `output_norm` is `nn.Identity` (`output_rms_norm=False`).

Output `z: [B, 96, d_model]`.

> x-transformers `dim_head` is 64, so attention inner dim is `nhead × 64` **regardless of `d_model`**.
> Capacity does not scale with panel size unless you raise `--nhead`.

### 2.6 Decoder trunk — grouped, constant lane width (`decoder.py:163-268`)

`input_proj = Linear(d_model, A × lane)` with `lane = --decoder-lane` (default 8), viewed as
`[B, 96, A, 8]`. Then 3 `LaneDeconvBlock`s (`decoder.py:130-160`), each
`ConvTranspose1d(stride=2, groups=A) → LaneNorm → + 1×1 grouped transposed residual → GELU`. Length
`96 → 192 → 384 → 768`; **lane width stays 8 at every stage**.

Output `feat: [B, 768, A, 8]`.

**`groups=A` is the design, not an optimisation** (`decoder.py:144-148`): no channel of assay `a` ever
reaches assay `b`. `tests/test_invariants.py:106` asserts it. `LaneNorm` is per-assay in both of its
modes and neither mixes lanes (`decoder.py:56-97`, asserted at `tests/test_invariants.py:121`):

* `lane_norm="lane"` (default) normalises the C channels of one lane **at each position** — it cannot
  leak, but it fixes each lane's energy per bin, so amplitude survives only as a pattern across C.
* `lane_norm="group"` normalises the C channels of one lane **across all L positions** — it removes
  the lane's overall location and scale and leaves the profile along the genome intact.

This is an **arm, not a fix**: `ConvTower`'s residual branch is an un-normalised 1×1, so amplitude
reaches the next layer either way (`decoder.py:74-77`). Measure it.

**Why constant width rather than an 8→4→2→1 mirror** (`decoder.py:27-33`): the exact mirror shrinks
conditioning capacity precisely as resolution grows — at the last tap FiLM would modulate a single
channel per assay and the head would degenerate to `Linear(1,1)`. Once the trunk is grouped, width is
nearly free. **This is a reasoned choice, not a measured one — no run has yet varied it.**

### 2.7 Decoder FiLM — per-assay, adaLN-zero, FOUR TAPS (`decoder.py:106-125`, `:214-217`, `:247-250`)

```python
memb = self.meta_embedding(y_meta.float())          # [B, A, 32]  SEPARATE untied embedder
gamma, beta = self.proj(meta_embed).chunk(2, dim=-1)  # [B, A, 8] each, per tap
z = z * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # broadcast over all positions
```

`n_deconv_layers + 1 = 4` `PerLaneFiLM` taps: one right after `input_proj`, one after each deconv
block. Each is an `nn.Linear(32, 2*lane)` with **weight and bias both zero-initialised**
(`decoder.py:117-118`) — adaLN-zero, so every tap is a bit-exact identity at step 0 and steering grows
from there. `tests/test_invariants.py:187` asserts that at init the target metadata moves nothing.

**Why four taps and not one at the end** (`decoder.py:15-25`): FiLM is a *position-constant* affine, so
applied after the trunk it can raise, lower or rescale a finished profile but cannot move a peak,
narrow a peak, or turn a sharp ATAC profile into a broad H3K36me3 domain. Assay-specific **shape**
forms *inside* the deconv, so the conditioning has to be there while it forms. Tap 0 is not redundant
with tap 1: `input_proj` is a dense `Linear`, so the lanes carry no assay identity until FiLM gives
them one.

Cross-assay leak in the decoder is **exactly 0.0**, unconditionally: `PerLaneFiLM` computes `(γ, β)`
per assay column and never averages over the assay axis, and the deconv is grouped. Perturbing one
assay's `y_meta` column changes no other assay's output (`tests/test_invariants.py:163`). There is **no
`pool_meta` option** in this decoder — the pooling control arm the predecessor exposed does not exist
here.

### 2.8 Heads and the NB arithmetic (`decoder.py:218-268`)

Two weight-shared heads applied on the last dim of `[B, 768, A, 8]`:

```python
head_eta = Sequential(Linear(8,8), GELU(), Linear(8,1))
head_n   = Sequential(Linear(8,8), GELU(), Linear(8,1))
eta   = head_eta(feat).squeeze(-1)    # [B, 768, A]
raw_n = head_n(feat).squeeze(-1)      # [B, 768, A]
```

Then the depth-offset log-link, **exactly** as implemented at `decoder.py:255-267`:

```python
depth = y_meta[:, 0, :]                                  # [B, A]   depth_row = 0
valid = (depth != MISSING) & (depth != CLOZE)            # [B, A]

if use_offset:                                           # offset ON
    d_off   = (depth - depth_center).unsqueeze(1)        # [B, 1, A]
    log2_mu = torch.where(valid.unsqueeze(1), d_off + eta, eta)
else:                                                    # offset OFF
    log2_mu = eta

if log_ref is not None:                                  # h74 reference arm only
    log2_mu = log2_mu + log_ref

log2_mu = log2_mu.clamp(-15.0, 30.0)                     # log2_mu_clamp
mu      = torch.pow(2.0, log2_mu).clamp_min(1e-6)        # mu_eps
n       = F.softplus(raw_n) + 1e-6
p       = (n / (n + mu)).clamp(1e-6, 1.0 - 1e-6)
```

Returns `dict(p, n, eta, log2_mu, mu)`, each `[B, 768, A]`. The NB mean is `mu`; `eta` is the
offset-free mean statistic used by the steering diagnostics. This arithmetic is **unchanged from the
predecessor**, deliberately, so the objective and both offsets stay comparable across arms
(`decoder.py:171-176`).

**Read the clamp carefully.** `log2_mu.clamp(-15, 30)` saturates, so any depth-slope number quoted for
an offset-ON arm is partly read through it. Emit `frac_targets_any_clamp` / `p90` / `max` alongside
every slope.

### 2.9 Parameter budget

**2,353,634 parameters** at `num_assays=35, context_length=768, d_model=288, nhead=4,
n_transformer_layers=2, decoder_lane=8, embed_dim=32, dropout=0.1` — asserted, not estimated, by
`tests/test_invariants.py:207-216`. `train.py:706-708` prints the count at every build and
`train.py:888` records it in the run JSON.

The predecessor's per-module table is **deliberately not reproduced here**: it described a decoder that
held 91.2% of the model in a dense conv tower, which is exactly the property this architecture removes
(`model.py:14-20`). A current per-module breakdown has not been measured; get it from
`model.named_parameters()` rather than from memory.

---

## 3. Training recipe

One `python -m candi.train` invocation = one arm: train on the h5's `train_chroms`, then evaluate
on `eval_chroms`, writing `{out_dir}/{tag}.json` and `{out_dir}/{tag}.ckpt`.

```bash
python -m candi.train --h5 /scratch/$USER/mypanel.h5 --out-dir /scratch/$USER/runs --offset on --seed 0 --weight-decay 0.0 --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 --fg-frac 0.02 --n-boot 1000 --tag candi_on_s0
```

With `--eval-every N > 0` the driver also scores held-out imputation every N epochs, keeps the best
`V_` checkpoint as `{tag}.best.ckpt`, and scores *that* one at the end instead of the last
(`train.py:788-839`). Selection is on `V_` so `B_` stays clean.

### 3.1 Masking

`make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)` (`train.py:297-298`) — **whole-assay
cloze only**. Per sample, a uniform random `num_to_mask ∈ [1, num_available-1]` assays are set to
`CLOZE` in data, metadata and availability (`_vendored.py:114-160`). A sample with `num_available ≤ 1`
is skipped entirely (`_vendored.py:133-137`) — which is why a panel needs biosamples with ≥2 available
assays or the imputation loss is identically 0.

The **control channel is never masked**: it is concatenated onto `x_meta`/`x_data` *after*
`masker.apply_mask` runs (`batch.py:71` vs `batch.py:128-130`).

**`y_meta` never carries CLOZE** — the masker touches `x_meta` only. Decoder embedding rows for CLOZE
and control are therefore unreachable during training.

**Training uses no prompt builder.** The dataset's raw `T_` `y_meta` passes straight through. The honest
imputation prompt (`V_`/`B_` natural metadata for absent targets) is an *eval*-side construction
(`eval.py:75-99`).

**A fraction of steps can run fully unmasked.** `--unmask-frac` (default 0.0) draws its coin from a
**dedicated** generator (`train.py:333`), never the shared data stream, so two arms visit identical
batches in identical order. It exists because the encoder otherwise only ever sees inputs containing
CLOZE tokens while eval presents none (`train.py:323-332`).

### 3.2 Per-assay independent DSF sampling

`_sample_xy_dsf` (`dataset.py:41-63`) draws, **independently per assay**, an input downsampling factor
`x_dsf` and a target factor `y_dsf` from the h5's `dsf_list` (default `[1,2,4,8]`). Column `fi` of
`x_data` is then loaded from `counts_dsf{x_dsf[fi]}` with `meta_dsf{x_dsf[fi]}`, and column `fi` of
`y_data` from `counts_dsf{y_dsf[fi]}` with `meta_dsf{y_dsf[fi]}` (`dataset.py:407-425`). The
metadata's depth row moves with the counts, so the model is always told the truth about what it is
looking at and what it must produce.

### 3.3 Loss

Counts only. Element-wise NB NLL, then **obs + `imp_weight` × imp**, each term separately normalised
(`train.py:52-85`):

```python
elem = -NegativeBinomial(total_count=n, probs=1-p).log_prob(y_data)   # [B, L, A]
loss = elem[observed_map].mean() + imp_weight * elem[masked_map].mean()
```

Two properties to know:

1. Because the two terms are normalised **independently**, cloze positions are effectively up-weighted
   per element even at `imp_weight=1.0`, and the ratio drifts with how many assays the masker happens
   to pick. This is a property of the recipe you are inheriting, not an accident to "fix" casually — it
   changes the obs/imp gradient balance.
2. `--imp-weight` (default 1.0) raises the imputation term deliberately, because evaluation scores
   imputation only while the objective weighted denoising equally and denoising dominated the curve.
   The sum is left **unnormalized** on purpose: normalizing would partly undo the reweighting by
   shrinking the step (`train.py:59-76`). `grad/total_preclip` is logged so the interaction with the
   1.0 clip stays visible.
3. `imp` is **omitted** from the logged terms when no position is masked, rather than logged as `0.0`
   — structural zeros were previously averaged into the curve as if they were measured likelihoods.
   Only the logging changes; the loss still gets a real zero-gradient term.

### 3.4 Optimizer and schedule — FROZEN by default

These are constants in the code, not flags (`train.py:88-97`, `:247`, `:270`, `:305-320`, `:693-696`):

| item | value |
|---|---|
| optimizer | `torch.optim.Adam(model.parameters(), lr, weight_decay=wd)` — **plain Adam, coupled L2, NOT AdamW**, one implicit parameter group |
| lr | `5e-4` |
| weight_decay | **`0.0`** (default; see §4.5) |
| grad clip | global norm `1.0` (`CLIP_NORM`), every step |
| schedule | linear warmup over `0.1 × total_steps`, then cosine to `0.1 × lr` |
| batch size | 8 |
| epochs | 25 |
| determinism | `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.manual_seed(seed)` immediately before `build_model`. **Not** `torch.use_deterministic_algorithms` — it crashes this encoder |
| `transformer_layer_drop` | must be `0.0`; the model raises otherwise, because it consumes global RNG inside `forward` and destroys step-for-step determinism (`encoder.py:1030-1035`) |

**`--optimizer adamw` is the one escape hatch, and it does not relax the freeze.** On the default
`adam` path the optimizer line is the historical one character for character. `adamw` builds two
parameter groups via `candi.param_groups` — the conditioning pathway (embedding tables, FiLM
projections, every norm affine, every bias) pinned at `wd=0`, trunk and heads at `--trunk-wd`. At
`--trunk-wd 0` the `adamw` path is numerically the `adam` default (a decoupled shrink by `1 - lr*0` is
a multiply by 1.0), which is what makes it a clean control. It **cannot** be combined with a non-zero
`--weight-decay`; `validate_optimizer_config` refuses that combination before anything expensive runs
(`train.py:677-678`). Classification is by module **type and identity**, never by name substring, and
there is **no fallback branch** — an unrecognised module raises (`param_groups.py:26-29`).

### 3.5 `--full-coverage`

The default sampled path picks a random `T_` biosample per batch and runs `steps_per_epoch` batches.
`--full-coverage` (`train.py:370-434`) instead builds one dataset per `T_` biosample sharing a single
RAM buffer (`train.py:375-383`) and pulls them **round-robin**, so every epoch deterministically visits
all train windows of all `T_` biosamples. `--steps-per-epoch` is dead under `--full-coverage`.

Host RSS, not GPU memory, is the binding constraint — the h5 is slurped into one shared buffer. The
recorded ~85 min/arm envelope in `slurm/train.sh:5-6` was measured on the **8-assay** q19 panel; the
35-assay panel has no measured envelope.

---

## 4. The load-bearing choices

For each: the choice, the evidence, and what breaks without it.

> Evidence labelled `h9` / `h33` / `h34` / `h43` and similar comes from a research tracker that **does
> not ship**, and this repository contains **no decoder ring** for those ids
> (`research/METADATA_AUDIT.md` is empty). Cite them as unresolved provenance; do not reconstruct what
> they mean.

### 4.1 Per-assay conditioning, NOT pooled across assays

**Choice.** The decoder FiLM computes `(γ, β)` per assay column from `memb[B, A, 32]` and does **not**
average over the assay axis (`decoder.py:120-125`). Combined with `groups=A` in the deconv
(`decoder.py:144-148`), the per-assay property is now **structural** rather than optional.

**Evidence.** A controlled pooling arm on the predecessor collapsed distributional output-steering
(M2 0.50 → 0.022, ~25×) *and* degraded reconstruction (M1 ceiling gap 1.16 pooled vs 0.63 per-assay).
A refinement worth knowing: uniform-per-batch sampling alone did **not** reproduce the null (M2 = 0.53)
— the causal factor was across-assay **pooling** specifically. Recorded as `h34` under `q16`, verdict
*supported*.

**What breaks.** Output steering, and reconstruction with it.

**⚠ There is no longer a `pool_meta` control arm.** The predecessor exposed
`build_real_model(pool_meta=True)` as a Python-API-only control. `PerLaneFiLM` has no such switch, and
reintroducing pooling would mean writing a new module. Do not cite `pool_meta` as something you can
turn on here.

### 4.2 Raw, UNNORMALIZED covariates

**Choice.** `log2_depth` (~22–28) and `read_length` (30–101) go straight into `nn.Linear(1, 32)` with no
scaling (`encoder.py:135-136`, projected at `:175`).

**Evidence, empirical.** Normalization was pre-registered as load-bearing and **refuted**: ranking three
arms by distributional M2 gave **none 0.515 > z-score 0.483 > log-scale 0.448** — the exact reverse of
the hypothesis, with no family showing a gain and reconstruction comparable (M1 gap 0.59 / 0.63 / 0.70).
Recorded as `h33` under `q15`, verdict *refuted*.

**Evidence, structural.** Sentinel detection is **exact equality against `-1` and `-2`**
(`encoder.py:173-174`). A normalized covariate that happens to take the value `-1` would be silently
reinterpreted as MISSING. Any normalization scheme you add must be range-guaranteed to avoid `-1`
and `-2`, or must move sentinel detection to a separate mask tensor.

**What breaks.** Normalizing costs steering (measurably), and a careless scheme can corrupt the
missing-data semantics.

*Open counter-consideration, recorded but not tested:* because `LayerNorm` pins `‖memb‖` to `√32`, the
entire observable depth range moves the embedding only ~10% of its radius while a `read_length` flip
moves it several times that. `--meta-embed-layernorm off` (`train.py:1012-1017`) and `--meta-gain`
(`train.py:1018-1025`) exist to probe exactly this, and `candi.probes` measures the attenuation
directly (`probes.py:25`). Neither has been read out here.

### 4.3 Per-assay independent DSF sampling (`--dsf-sampling uniform`)

**Choice.** `x_dsf` and `y_dsf` are drawn **independently** per assay (`dataset.py:52`, the
`mode == "uniform"` branch), so the target depth is generally not the input depth, and the target is not
a copy of the input.

**Evidence.** The `x_eq_y` control — one DSF drawn per assay and used for *both* sides
(`dataset.py:53-55`) — makes the task copyable. It got **higher raw imputation Spearman (0.6386 vs
0.533 on the main recipe)** but **broke latent invariance**: the M3 within/between cos-distance ratio
was **0.3335, above the 0.3 bar**, versus 0.244–0.292 on the main recipe. Recorded as `h43` under `q19`.

**What breaks.** With `x_eq_y`, target depth is redundant with input signal magnitude, the depth
covariate receives ~zero gradient, and the encoder stops using metadata to normalise the measurement
condition. The higher Spearman is the *symptom*, not a win: it is what a copying model scores.

**`x_eq_y` is a control, not a recipe.** So are `--dsf-sampling off` and `upsample_only`
(`dataset.py:49-62`). And a **single-DSF `dsf_list` makes independent sampling inert and deletes the
depth signal the whole recipe rests on** — the panel warns at bake time (`prep/panel.py:44-48`) and the
dataset warns again at load (`dataset.py:209-211`).

### 4.4 `depth_center` derived from the data

**Choice.** `log2_mu = (depth - depth_center) + eta`. The trainer derives `depth_center` at build time
as the **median of finite `meta_dsf1[0]` over `T_` biosamples** (`dataset.py:92-106`) and prints it
(`train.py:688-691`). `--depth-center` overrides. The `SymmetricDecoder` constructor default is 25.1
(`decoder.py:182`), a historical q19 constant that is a *mean over all views at DSF 1*, not a median
over `T_` — **do not assume the two coincide; read the printed value.**

**Known defect you inherit with any DSF-1 statistic:** because DSF only ever *down*-samples, the depth
actually prompted during training is systematically lower than the DSF-1 median, so the offset term has
a negative mean rather than 0 — contradicting the head's own rationale. Deriving the centre from the h5
does not fix this, since it is also a DSF-1 statistic. Centring on the *prompt* mean is an untested
one-line change.

**Evidence.** An **uncentred raw `2^d` offset FAILS** — depth-controllability ratio ~1.0, i.e. no
depth response at all — while the depth-centred size factor `μ = 2^(d - center) · exp(η)` gives
DCR 3.99–4.02 from epoch 0. Recorded as `h9` under `q4`.

**What breaks.** Without centring, `2^d` at d≈25 is ~3.4e7, so `η` must carry a ~25-log2 offset that the
head is not parameterised to produce; the run collapses to a depth-insensitive solution.

**Derive it; do not copy 25.1 onto a different panel.**

**One consistency rule.** Under `--reference on` the h74 reference table is normalized to *its* centre
while the decoder subtracts the model's, so a disagreement silently biases every prediction by
`2^(difference)`. Both default to `h5_depth_center` on the same file and agree; `--depth-center` can
break that, so the trainer and the eval CLI both refuse a mismatch (`train.py:744-749`,
`eval.py:1457-1459`).

### 4.5 `weight_decay = 0.0`

**Choice.** The default is `--weight-decay 0.0` (`train.py:956`), which differs from the research
driver's `1e-4`.

**Evidence.** With `weight_decay=1e-4` — **L2-coupled inside plain Adam**, applied over tens of
thousands of updates to a pathway whose task gradient is ≈0 because the arithmetic offset already fits
the mean — the decoder's metadata embedder was *annihilated*: `assay_embedding` and
`runtype_embedding` absmax reached **~6e-41** (denormal), all assay-embedding row norms were exactly 0,
and `depth_proj.weight` was 1.6e-4. The FiLM projection itself stayed healthy (0.347), so the death was
specifically on the embedder's **input projections**. With adaLN-zero the entire decoder metadata
pathway also receives *exactly zero* gradient at init, so it starts dead and decay outruns anything it
might later acquire.

Setting `weight_decay=0` **prevents the annihilation**: the table stays full-rank and injective
(effective rank 6.8/8 on the 8-assay panel). This is the weight-level claim, and it is supported.

**⚠ State the limit honestly.** `weight_decay=0` does **not** buy functional steering. On the
predecessor's offset-ON `wd=0` arm the table sat at **random-init statistics** (element std 0.94 vs 0.97
for a fresh N(0,1) table; cosine 0.988 against an independently-trained table) — it was never destroyed
*and* never trained. See §5 for the corresponding steering numbers, and the banner above them.

**What breaks.** At `wd=1e-4` you get a decoder that is bit-exactly blind to its own target metadata
from the fusion's first `Linear` onward, which will read as an "honest null" in any steering probe.

**`--optimizer adamw --trunk-wd X` is the principled alternative** (§3.4): it decays the trunk while
pinning the whole conditioning pathway at zero, so you do not have to choose between decaying
everything and decaying nothing. `--trunk-wd 0.0` is its control arm.

### 4.6 adaLN-ZERO init on every decoder FiLM

**Choice.** `PerLaneFiLM.proj` weight *and* bias are zero-initialised **at construction**
(`decoder.py:117-118`), on **all four taps**, so decoder conditioning is a bit-exact identity at
step 0. `tests/test_invariants.py:187` asserts it.

**Why.** Reconstruction is stable from step 0 and steering grows from a clean identity, rather than the
model having to unlearn a random modulation. It is also what lets a run be compared to its own control:
the model is born un-conditioned, so it cannot lose by being born mis-conditioned.

**What it costs — know this.** Combined with the free arithmetic offset, adaLN-zero means the optimizer's
depth objective is *already satisfied* before the learned pathway is even alive. The encoder FiLM uses
the opposite init (Xavier weight, `N(0, 0.1)` bias, `encoder.py:393-394`); that asymmetry is
deliberate but it is also the mechanism behind the offset-ON steering null.

**The silent failure it creates, and how to see it.** A tap that never leaves its zero init produces
`gamma = beta = 0` forever — an exact identity — and the loss curve looks entirely normal. `train.py`
therefore logs, per tap, both the gradient norm and the parameter absmax: `grad ≈ 0 AND absmax ≈ 0`
means never switched on, `grad ≈ 0 AND absmax > 0` means converged, and only the pair separates them
(`train.py:212-224`). `candi.healthcheck` reads the same distinction at init and after an overfit
(`healthcheck.py:18-24`).

---

## 5. The offset head is a real Pareto — not a solved problem

> ### ⚠ HISTORICAL. These numbers are the **predecessor's**.
>
> They were measured on an 8-assay q19 panel with an ungrouped ~3.10 M-parameter decoder trunk and a
> single FiLM tap, on four checkpoints that **do not ship**, from source documents that **do not ship**
> (see §0). This repository's model is a 2.35 M-parameter grouped decoder with four taps. **Do not
> present any figure below as a property of the current model, and do not compare a current run
> against it.** The *structural* arguments in this section survive the architecture change because they
> are about the head arithmetic, which is unchanged; the numbers do not.

`--offset {on,off}` selects between two arms. **Both are first-class; neither dominates.** Read the
numbers against the noise floor stated in §0.

| | **offset ON** (`--offset on --weight-decay 0.0`) | **offset OFF** (`--offset off --weight-decay 0.0`) |
|---|---|---|
| macro CRPS | **1.3413** | 2.0561 |
| macro Spearman | **0.5653** | 0.4641 |
| pooled imputation Spearman | **0.6372** | 0.3800 |
| PIT-ECE | **0.0533** | 0.0782 |
| beats honest per-assay marginal | **7/8** | 1/8 |
| oracle-scaled CRPS (capability) | 1.3077 | 1.4026 |
| per-assay scale error | 0.0336 | 0.6535 |
| assay steering, sentinel-free real→real max\|Δη\| | **0.0023** | **9.7144** |
| run_type steering, target-clustered CI | [−0.00066, +0.000087] | [−0.2326, +9.4084] |
| told-depth total slope | 1.0000 | 1.0325 |

The `--offset off --weight-decay 1e-4` arm reads macro CRPS 1.9023 (+42% over the ON arm), assay
steering 4.1772, run_type clustered CI [+0.1179, +2.1804], sign-test p = 0.039 — the only arm whose
steering direction was target-clustered supported.

**What this means, stated plainly:**

1. **offset ON was the best imputer** and the best-calibrated. Its **covariate steering was functionally
   null**: sentinel-free real→real assay ablation **0.0023**, **43× below** its own pre-registered ≥0.10
   functional bar, and 1816×/4224× below the two offset-OFF arms on the *identical* probe. Its run_type
   response was bit-indistinguishable from zero.
2. **Its depth response is arithmetic, not learned — and this argument still holds.**
   `log2_mu = (depth − depth_center) + η` (`decoder.py:257-259`) is a closed-form thinning identity: NB
   is closed under binomial thinning with `n` preserved, and DSF downsampling *is* exactly that
   generative model. A told-depth slope of **1.0000** is therefore the arithmetically correct answer
   produced by a hardwired subtraction. It is **not** evidence of learned conditioning. Under
   offset-ON, `η` has analytically nothing left to learn about depth on the DSF axis.
3. **offset OFF had genuine learned steering** at a large magnitude cost. Its raw-CRPS deficit was
   **calibration, not capability**: under an oracle per-assay scale the four arms compressed from a
   0.7148 spread to **0.1133 (84% compression)**.
4. **A prior "steering 0.833" result for the offset-ON `wd=0` arm was retracted.** It was a
   MISSING-sentinel artifact: the probe permuted the whole `assay_id` row across all 8 slots, sliding
   the `-1` sentinel onto and off *unavailable* slots whose prompt columns are entirely `(-1,-1,-1,-1)`.
   Sentinel-free, the real→real value is 0.0023.
5. **`h45` — the hypothesis that a hybrid (offset warmup→anneal, α-attenuated offset, or a learned
   metadata-driven scale head) recovers both — is recorded REFUTED, 0/4 verifiables met.** Read that
   precisely: it was refuted **on its premises**, and **no hybrid arm was ever trained**. One leg of the
   refutation rested on the retracted 0.833. So the hybrid is **neither demonstrated nor experimentally
   excluded** — do not re-run it as posed, and do not cite it as a negative result. Note the offset is a
   **boolean all the way down** (`train.py:1073` → `build_model(use_offset=…)` → `decoder.py:257`), so a
   β-schedule arm is a real code change plus a per-step schedule threaded into `_train_step`, not a flag.

**⚠ The 4-arm ordering was NOT established.** Under oracle per-assay scale the four arms sat in a 0.113
band against a ~0.09 target-clustered noise floor. Only "the offset-ON `wd=0` arm is best on capability"
survived inference (paired bootstrap Δ = +0.093, 95% CI [+0.004, +0.217]); the other three were
statistically indistinguishable (all pairwise CIs cover 0). The published ordering was the modal
bootstrap ordering at only 45% of replicates.

### Bounds you inherit with the 8-assay panel

These are properties of the **data**, not the model, so they still apply to any run on that panel.

- **`run_type` is analytically unidentifiable** on the q19 panel:
  `H(run_type | assay_id, read_length) = 0.000 bits` (n=26). It is a deterministic function of the other
  two covariates, so weight decay has a free ride on its embedding and *any* function of it is
  reproducible at zero loss cost. **A run_type steering demo is impossible on that panel** — it needs a
  re-selected biosample panel. The full EIC panel retains **0.551 bits** after conditioning on assay, so
  this is a property of the 5-biosample slice, not of the field. `--meta-probe` (§7.3) is the bracket
  built for exactly this question.
- **DSF only ever DOWN-samples**, so the upward-depth regime is never trained. Training support for
  assay *a* is `[natural_min − 3, natural_max]`, and **7/12 q19 eval targets sat above their per-assay
  training ceiling** (worst +1.43 log2). Depth steering evaluated there is extrapolation in the
  untrained direction.
- **Depth, read_length and run_type are mutually collinear** at n=38 (assay-centred
  `corr(d, log2 rl) = 0.763`, `corr(d, rt) = 0.590`, `corr(log2 rl, rt) = 0.697`). Attribution among the
  three exposure covariates is **not identified** on this dataset. Any claim of the form "covariate X
  carries Y% of the exposure signal" is unsupportable here.
- **9/12 q19 eval targets were OOD in (assay × read_length)**.
- **`assay_id ≡ slot index`**, and the decoder emits a dedicated lane per slot, so assay identity is
  carried structurally. Any assay-steering verdict bounds the **prompt pathway**, not assay-awareness —
  and this is *more* true of the current grouped decoder than it was of the predecessor.

---

## 6. Construction order and the RNG stream

`CandiModel.__init__` draws from the **global torch RNG**, so reordering module construction in
`src/candi/model.py`, `encoder.py` or `decoder.py` **silently changes every initial weight**. Nothing
crashes; the model just trains from different initial conditions.

The sequence, and the one line that is easy to delete by accident:

| # | draw | file:line |
|---|---|---|
| 1 | `V2Encoder(cfg)` **in full** — `metadata_embedding` → `signal_tower` → `mask_injector` → `dna_tower` → `fusion` → `transformer_blocks` → (`transformer_film_layers`, not built here) → `output_norm` | `encoder.py:916-1027` |
| 2 | a **replacement** `MetadataEmbedding` overwriting `encoder.metadata_embedding` — the one built at step 1 is discarded | `model.py:62-64` |
| 3 | `SymmetricDecoder` — `input_proj` → 3 × `LaneDeconvBlock` → 4 × `PerLaneFiLM` → `head_eta` → `head_n` → `meta_embedding` | `model.py:65-70`, built at `decoder.py:203-224` |

**Step 2 is the trap.** The replacement is *functionally* a no-op — `V2Encoder` already built an
identically-configured `MetadataEmbedding` from the same cfg — but it consumes a block of RNG draws, so
every module built afterwards (the whole decoder) lands on different weights without it. The comment at
`model.py:56-61` is explicit that removing it is a **legitimate cleanup**, but one that re-samples the
model and must be its own labelled change with a fresh `tools/golden.py` recording, not a side effect of
moving code between files.

Inside `MetadataEmbedding`, the `cell_embedding` table is created **only** when `num_cells > 0`
(`encoder.py:148-153`). Do not hoist it above `assay_embedding`/`runtype_embedding` and do not create a
zero-size table unconditionally — that is what keeps `--cell-cond off` bit-identical to the 4-row model.

Other things that move the stream:

- **an x-transformers version other than the pinned `2.11.23`** (`pyproject.toml:22`) — its per-module
  init order is part of the RNG stream *and* it fixes the `encoder.transformer_blocks.*` `state_dict`
  key names. The package has no `__version__` attribute; `train.py:912` records
  `importlib.metadata.version` in the run JSON.
- `PerLaneFiLM` must be **constructed then zeroed** (`decoder.py:116-118`), not created as a zero
  tensor.
- The masker draws from the global stream at every step (`_vendored.py:272-274`), so any
  construction-time RNG change shifts the entire masking sequence too.

**If you intend to change the architecture, that is fine.** Re-record with `tools/golden.py save` and
state that your run is a new lineage. There is no `--compat-q19` flag to invalidate — that pin was
removed with the predecessor model.

---

## 7. The full knob table

**Single source of truth rule:** scale is declared **once** in the bake panel, written into the h5
attrs, and read back by the trainer. `num_assays`, `context_length`, `resolution`, `dsf_list`,
`num_cells`, assay order, and train/eval chromosomes are **never** train-time CLI flags. You cannot
silently train a 35-assay model on a 12-assay file.

### 7.1 Bake-time (`configs/panel*.json`)

| knob | type | default | status | note |
|---|---|---|---|---|
| `assays` | list[str] | required | load-bearing (via derived order) | order-**insensitive**; the resolved column order is derived from the handler's `aliases.json` and asserted bijective against this list |
| `biosamples` | list[str] | required | **LOAD-BEARING** | full names with `T_`/`V_`/`B_` prefix; ≥2 `T_` biosamples, each ideally with ≥2 available assays |
| `dsf_list` | list[int] | `[1,2,4,8]` | **LOAD-BEARING** | a single-DSF ladder makes independent `x_dsf ≠ y_dsf` inert and deletes the depth signal |
| `resolution` | int | 25 | semi-knob | it is `dna_pool_size**2`, so only perfect squares are reachable without editing the encoder's block-count formula |
| `context_bins` | int | 768 | FREE | must be divisible by `pool_size**n_cnn_layers = 8` |
| `train_chroms` | list[str] | `["chr19"]` | FREE | |
| `eval_chroms` | list[str] | `["chr21"]` | FREE | must be disjoint from `train_chroms` |

Those **seven keys are the whole schema** (`prep/panel.py:13-21`). `load_panel` rejects anything else
(`prep/panel.py:60-65`), so `type2_ccre`, `type2_non` and `seed` are **not** panel fields — putting them
in `panel.json` raises `ValueError: unknown panel keys [...]`. They are **bake CLI flags**
(`prep/bake.py:498-500`): `--type2-ccre N` / `--type2-non N` (both default 0; `>0` also requires
`--ccres`) and `--seed` (default 42, type-2 locus sampling only). Keys beginning with `_` are ignored
as comments (`prep/panel.py:59`).

### 7.2 Train-time: DERIVED from the h5, not configurable

`num_assays`, `context_length`, `resolution`, `dsf_list`, `num_cells`, assay order, `train_chroms`,
`eval_chroms` ← `h5.attrs` (`dataset.py:176-192`, consumed at `train.py:698-705`).

`depth_center` ← median of finite `meta_dsf1[0]` over `T_` biosamples, **printed at build**;
`--depth-center` overrides.

### 7.3 Train-time: LOAD-BEARING flags

| flag | default | why |
|---|---|---|
| `--offset {on,off}` | `on` | the two shipped arms; §5 |
| `--dsf-sampling` | `uniform` | per-assay independent `x_dsf ≠ y_dsf`; §4.3. `x_eq_y` / `off` / `upsample_only` are controls |
| `--p-full-assay` | `1.0` | whole-assay cloze is the **only** source of imputation supervision |
| `--weight-decay` | `0.0` | §4.5 |
| `--optimizer {adam,adamw}` | `adam` | §3.4. `adamw` is the two-group split; cannot be combined with a non-zero `--weight-decay` |
| `--trunk-wd` | `0.0` | decay on the trunk+heads group under `adamw`; **rejected** under `adam` |
| `--lane-norm {lane,group}` | `lane` | how every per-assay normaliser in the decoder pools its statistic; §2.6 |
| `--decoder-lane` | `8` | channels per assay in the deconv trunk, held constant across all three stages |
| `--imp-weight` | `1.0` | loss is `obs + w·imp`; eval scores imputation only; §3.3 |
| `--unmask-frac` | `0.0` | fraction of fully-unmasked steps, from a dedicated RNG; §3.1 |
| `--reference {off,on}` | `off` | h74 deviation-from-average-reference arm. Strict no-op at `off` |
| `--lr` | `5e-4` | |
| `--batch-size` | `8` | |
| `--epochs` | `25` | |
| `--full-coverage` | set it | deterministic all-windows × all-`T_`-biosamples round-robin |

Frozen constants (not flags): grad-clip 1.0; cosine warmup_frac 0.1 / min_ratio 0.1; plain Adam with
coupled L2 on the default path.

### 7.4 Train-time: FREE, or diagnostic

| flag | default | note |
|---|---|---|
| `--d-model` | `0` = auto | **see §7.6** |
| `--nhead` | `4` | attention inner dim is `nhead × 64` regardless of `d_model` |
| `--embed-dim` | `32` | metadata embedding width, both towers |
| `--n-transformer-layers` | `2` | |
| `--dropout` | `0.1` | |
| `--seed` | `0` | but see the seed sensitivity in §0 |
| `--regime` | `type1` | or `type2_loci` |
| `--steps-per-epoch` | `200` | **dead under `--full-coverage`** |
| `--eval-batch-size` | `4` | |
| `--eval-max-batches` | `0` (=all) | |
| `--eval-budget` | `200000` | max eval points for CRPS/correlation; set very high for no subsampling |
| `--m3-regions` | `8` | |
| `--fg-frac` | `0.02` | foreground fraction for the scored positions |
| `--n-boot` | `1000` | |
| `--eval-every` | `0` (off) | mid-training eval every N epochs; keeps and scores the best `V_` checkpoint |
| `--eval-batches-per-pair` | `4` | mid-training eval only; thins windows while keeping **all** targets. Never use `--eval-max-batches` for this — it truncates the pair cycle and drops whole targets |
| `--include-deprecated` | off | emits legacy metric keys with their verdict strings attached |
| `--cell-cond {off,id,random}` | `off` | 5th metadata row. `random` is the null control (same channel, same parameters, zero information) |
| `--meta-embed-layernorm {on,off}` | `on` | h75. `on` is the historical architecture and a strict no-op |
| `--meta-gain` | `1.0` | h76. Fixed non-learnable scalar on `memb` before the decoder FiLM. `1.0` skips the multiply entirely |
| `--meta-probe {off,shuffled,planted}` | `off` | h64 covariate-gradient bracket: `shuffled` = negative control, `planted` = positive control. **Not applied in the final eval** — see the warning `train.py` prints and `config.final_eval_meta_probe_applied` |
| `--meta-probe-delta` | `1.0` | planted shift in log2 units |
| `--reference-path` | `<h5 stem>.reference.h5` | |
| `--reference-pseudocount` | `0.25` | `c` in `log2(R + c)` |
| `--wandb-project` / `--wandb-run-name` | `None` | logging is never fatal; a dead dashboard cannot kill a run |
| `--tag` | `candi_{lane_norm}_ep{epochs}_seed{seed}` | output basename |

### 7.5 INERT and NOT EXPOSED

- `--mask-fraction` (`0.2`) is **inert** under `p_full_assay=1.0` — `_mask_full_assay` never reads it.
  The trainer warns if you set a non-default (`train.py:294-296`).
- `num_metadata_rows` (4) and `num_runtypes` (2) are **not** config fields (`config.py:6-7`). They are
  fixed by the ingestion contract and fenced by raises in `MetadataEmbedding.forward`.
- **Not reachable from `build_model` at all** — editing `model.py` is the only way in. `CandiModel`
  pins `film_mode="per_conv"`, `signal_transform="arcsinh"`, `missing_data_mode="mask_token"`
  (`model.py:49-54`), and never passes `conv_norm`, `n_cnn_layers`, `expansion_factor`,
  `conv_kernel_size`, `pool_size`, `dna_pool_size`, `dna_pool_order`, `fusion_mode`, `fusion_norm`,
  `fusion_deep`, `transformer_type`, `attn_qk_norm`, `transformer_layer_drop` or `output_rms_norm`, all
  of which take their `config.py:19-68` defaults. On the decoder side it never passes
  `n_deconv_layers`, `upsample`, `conv_kernel_size`, `log2_mu_clamp`, `mu_eps` or `in_lane_width`
  (`decoder.py:179-185`).
- **`pool_meta` no longer exists.** See §4.1.

### 7.6 ⚠ The `d_model` / `num_assays` coupling

With `--d-model 0` (the default), the transformer width is **derived from the panel**
(`encoder.py:952-954`):

```
d_model = signal_tower.out_channels = num_tracks × expansion_factor**n_cnn_layers
        = (num_assays + 1) × 2**3
```

So 35 assays → `d_model = 288`; 8 assays → 72; 3 assays → 32; 16 assays → 136.
`tests/test_model.py:180-188` pins this.

`--d-model` is an **independent override**: set it and the transformer width follows the flag, not the
panel (`encoder.py:953`, pinned by `tests/test_model.py:190-197`). **Set it explicitly whenever you
change the panel**, or your transformer capacity silently tracks how many assays you happened to
include.

**Changing the assay count invalidates any checkpoint** — it moves the transformer width, the grouped
conv schedule, the mask-token table, the decoder's `A × lane` trunk, and the `assay_id` embedding table
size. A checkpoint is valid only against an h5 whose recorded assay **order** matches; compare against
the order stored in the run JSON (`config.assays`, `train.py:906`), not merely against the panel, and
never delete `aliases.json` once a bake exists.

---

## 8. What was tried and did NOT work

> ⚠ Same historical caveat as §5: every verdict below was reached on the **predecessor** architecture,
> and the tracker ids are unresolvable from this repository. They are listed so you do not re-run them
> as posed — not as measurements of the current model.

| idea | status | what actually happened |
|---|---|---|
| **`h45` — a hybrid recovers both magnitude and steering** (offset warmup→anneal-off; α-attenuated offset β ∈ {0.25,0.5,0.75}; learned metadata-driven scale head) | **recorded REFUTED, 0/4 verifiables met — but on its premises only: NO HYBRID ARM WAS EVER TRAINED** | The premises that fell: offset-OFF's "recovered depth steering" (η-slope 0.88) was a measurement artifact — `eta_slope` scores the *offset-free residual*, which is ~0 by construction under a correct offset. Its run_type steering does not survive target-clustered CIs, and run_type is analytically unidentifiable on that panel. One leg rested on the retracted 0.833. **Do not report the hybrid as tested.** |
| **`h33` — covariate normalization is load-bearing** | **REFUTED** | none 0.515 > z-score 0.483 > log 0.448; no family showed a gain. Keep raw. §4.2 |
| **Across-assay covariate pooling** (`h34` / `q16`) | **refuted as a design; supported as a diagnosis** | Pooling cost ~25× steering (M2 0.50 → 0.022) *and* hurt reconstruction (M1 gap 1.16 vs 0.63). The current decoder has no pooling switch at all. §4.1 |
| **Uncentred raw `2^d` offset** (`h9`) | **FAILS** | depth-controllability ratio ~1.0 vs 3.99–4.02 for the depth-centred size factor. §4.4 |
| **`S23` condition-recoverability probe** | **WITHDRAWN — do not cite in either direction** | Its ordering is inverted against every other measurement: the arm carrying ~5,900× more feature energy scored *below* chance, while the near-zero-signal arm scored 2.5× higher. Leave-one-target-out nearest centroid on within-target-centred features penalises a target-*adaptive* response whose direction flips sign between targets. Reliable only as a bit-exactly-dead detector. |
| **"`wd=0` revives functional assay steering"** | **RETRACTED** | The 0.833 that supported it was a MISSING-sentinel artifact. Sentinel-free real→real is 0.0023. The *weight-level* claim (`wd=0` prevents annihilation) stands; the *function-level* claim does not. §4.5, §5 |
| **The 16×-wide ungrouped decoder trunk** | **REPLACED** | It ran at `num_assays × feat_per_assay × 2**3` channels against production's `num_assays × 2**3` — a vendoring accident that put 93.7% of a 56.2 M-parameter model in a dense conv tower. Mirroring the encoder instead (grouped by assay, constant lane width) drops the model to 2.35 M **and improves both imputation and denoising** (`model.py:14-20`). This is the change that produced the current architecture. |

### Ranked but NOT yet run

Both target the same finding: **most of the apparent ON/OFF magnitude difference was per-assay
scale-calibration error, not capability** (84% compression under an oracle per-assay scale). That
compression result is the solid one; the 4-arm reordering is not. **Neither idea has been re-assessed
against the current grouped decoder, which changes the premise of the first one materially** — see the
note under `h50`.

**`h50` — an explicit per-assay output factor.** Add metadata-**independent** per-assay `η` scale + bias
and a per-assay `log n` offset, indexed by **slot**, not by `y_meta`.

*Why it was proposed:* the head is `Linear(C,C)+GELU+Linear(C,1)` **weight-shared across assays** with a
single scalar output bias, so on the predecessor the only per-assay knob was a low-rank FiLM. The
oracle correction that produces the 84% compression is *literally one scale per assay*.

*⚠ Re-check the premise first.* The current decoder is **grouped**, so every assay already owns a
private lane through the entire trunk and four private FiLM taps. The "one shared head, no per-assay
capacity" diagnosis that motivated `h50` is substantially weaker here. Measure the residual per-assay
scale error on a current run before adding parameters.

*How to judge it:* the gain must show up in the **scale-error term** (`CRPS − CRPS_oracle_scaled`
shrinks), not in shape; hold macro Spearman and ECE at least flat; and beat capability **by more than
the noise floor**, not by a 4-decimal margin.

**`h49` — `read_length` as a fixed-coefficient physical exposure term.** The NB head is a size-factor
GLM, but its exposure term is **incomplete**: it counts *reads*, not read *footprint*. A length-R read
at 25 bp resolution covers ~`R/25 + 1` bins, so a second exposure factor `log2(R/25 + 1)` spans ~1.2
log2 over the 30–101 bp range.

*Why it should help:* `read_length` carries a ~0.48–0.61 coefficient on log2 mean count and **is** the
"excess depth slope" — once `log2(rl)` enters, the depth coefficient returns to 0.975. Today that job is
left to a FiLM that cannot extrapolate, and 9/12 q19 eval targets were OOD in (assay × read_length); an
arithmetic offset extrapolates by construction, and it frees the FiLM to spend its rank on assay
identity instead.

*Constraint:* fix the coefficient at the physical value **1**. Because depth, read_length and run_type
are mutually collinear at n=38, attribution among them is not identified on this data — do not fit and
then claim credit. An optional second arm may learn a no-decay coefficient and check it converges
to ~1.

---

## 9. Where to look next

| question | file |
|---|---|
| how to build the h5 from ENCODE-style data | `DATA.md` |
| the file map, invariants and failure-mode table | `AGENTS.md` |
| quickstart and the layout in one screen | `README.md` |
| the bit-exactness gate | `tools/golden.py` |
| the construction-order rule | `src/candi/model.py:56-61` |
| the decoder invariants, as executable assertions | `tests/test_invariants.py` |
| pre-training health checks on real baked data | `python -m candi.healthcheck` |
| comparing two trained arms | `python -m candi.compare_arms` |
