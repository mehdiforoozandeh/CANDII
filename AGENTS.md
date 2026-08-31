# AGENTS.md — machine-actionable spec for `candi/`

Audience: an LLM coding agent (and its human) working inside this directory with no prior context.
Read this file top to bottom before editing anything. Every technical claim below cites `file:line`
inside this repository.

Companion docs in this directory: `README.md` (quickstart), `RECIPE.md` (architecture and training
recipe), `DATA.md` (input layout + h5 schema).

---

## 1. ORIENTATION

`candi/` is **one architecture, not a registry**: a counts-only Negative-Binomial epigenome
imputer/denoiser conditioned on 4 experimental covariates on both the input and the output side. It
contains everything needed to go from an ENCODE-style directory of per-biosample count files to a
trained checkpoint and a scored results JSON: ingestion/bake (`src/candi/prep/`), dataset + masking,
model, training driver, evaluation stack, tests, and SLURM scripts. It has **no imports back into the
parent EpiDenoise repo** and runs from any working directory once `pip install -e .` has been run.

The shipped model is ~**2.35 M parameters** — 2,353,634 on the 35-assay EIC panel at
`context_length=768, d_model=288, nhead=4, n_transformer_layers=2, decoder_lane=8, embed_dim=32`,
pinned by `tests/test_invariants.py:207-216`. The encoder is a Conv1D tower grouped by track with
per-assay FiLM after every block, a dense DNA tower, a linear fusion and two RoPE transformer layers
(`encoder.py:891`). The decoder is a **grouped deconv mirror at constant lane width** with a FiLM tap
after the input projection and after each deconv block (`decoder.py:163`).

What it is **not**:

* **Not a pretrained model.** No weights ship. Training is from scratch. The bit-exactness gate is
  `tools/golden.py`, which records and re-checks forward outputs + `state_dict` digests of the
  *current tree* against a reference `.pt` you generate yourself — no reference tensors are committed.
* **Not full CANDI.** This is the counts-only NB head. There is no Gaussian p-value head and no
  Bernoulli peak head (`model.py:39-87`); therefore no peak precision/recall/AUROC and no p-value
  track. `y_pval` / `y_peaks` are carried through the loader and batch untouched
  (`batch.py:51-52,152-153`) purely as the extension hook.
* **Not the predecessor.** The decoder this replaces ran an **ungrouped** deconv trunk at
  `num_assays * feat_per_assay * 2**3` channels — 93.7% of a 56.2 M-parameter model in a dense conv
  tower. That model, its `--compat-q19` gate, its committed golden tensors and its `candi.compat`
  module are **all gone**; the rationale for the replacement is `model.py:14-20` and
  `decoder.py:15-33`. Nothing in this tree loads a historical q19 checkpoint.
* **Not a general framework.** It is one recipe with a load-bearing construction order and a frozen
  optimizer/loss/step order. Most "improvements" an agent might reflexively make (reordering module
  construction, renaming state_dict keys, "cleaning up" unreachable branches, normalizing covariates)
  are regressions here. See §3.

---

## 2. FILE MAP

Paths are relative to the repo root. "DO NOT TOUCH" = editing changes recorded numerics or breaks the
`tools/golden.py` bit-exactness gate; change only with an explicit instruction and a re-run of §5.

### 2.1 `src/candi/` — training path (importable without the `[prepare]` extra)

| file | lines | responsibility | do not touch |
|---|---|---|---|
| `__init__.py` | 21 | Exports only `EncoderConfig`, `V2Encoder`, `CandiModel`, `build_model`, `forward_full`, `nb_nll`; defines `__version__`. | Do not add re-exports that pull the bake path or the eval stack — a bare `import candi` must stay light. |
| `_vendored.py` | 354 | The severance module: `MISSING=-1`, `CLOZE=-2`, `exponential_linspace_int`, `DataMasker`. | `DataMasker` is RNG-load-bearing: `apply_mask` (`_vendored.py:248`) draws from the **global** torch RNG every step (`:272-274`, and `_mask_full_assay` at `:140,:143`). Do not reorder, do not add draws. |
| `config.py` | 68 | `EncoderConfig` dataclass only (all encoder knobs, incl. `d_model=0` auto and `num_cells=0`). | Do not add/remove fields or change defaults. Note the dataclass defaults are **not** the shipped values — `CandiModel` overrides `film_mode`, `signal_transform` and `missing_data_mode` at `model.py:49-54`. |
| `encoder.py` | 1129 | `lanes_from_channels`/`channels_from_lanes` (`:68`,`:79`), `MetadataEmbedding` (`:104`), `ConvBlock` (`:282`), `ConvTower` (`:355`), `FiLMLayer` (`:383`), `MaskTokenInjector` (`:460`), `SignalConvTower` (`:505`), `DNAConvTower` (`:618`), `LinearFusion` (`:679`), `V2Encoder` (`:891`). Raises at `:190-196` (row count), `:212-218` (`assay_id` bound), `:232-238` (`run_type` bound), `:255-261` (`cell_id` bound), `:1030-1035` (`transformer_layer_drop` must be 0.0), `:1046-1051` (signal-vs-meta availability), `:1075-1080` (DNA length). | Blocks marked `# UNREACHABLE on the shipped path` (`:85`, `:416`, `:441`, `:717`, `:796`, `:808`, `:1009`) are kept **verbatim on purpose** — deleting them is a refactor with checkpoint-compat risk and zero benefit. |
| `decoder.py` | 268 | `LANE_NORMS` (`:53`), `LaneNorm` (`:56`), `PerLaneFiLM` (`:106`), `LaneDeconvBlock` (`:130`), `SymmetricDecoder` (`:163`). Head arithmetic at `:252-268`. | The deconv is `groups=num_assays` (`decoder.py:144-148`) — that grouping *is* the design. Lane width is held **constant** across all three upsample stages on purpose (`decoder.py:27-33`). |
| `model.py` | 122 | `CandiModel` (`:39`), `build_model` (`:85`), and the batch helpers `forward_full` (`:94`), `encode_latent` (`:104`), `nb_mean` (`:109`), `nb_nll` (`:113`). | **CONSTRUCTION ORDER IS LOAD-BEARING** — the `metadata_embedding` replacement at `model.py:62-64` is functionally a no-op but consumes RNG, so the whole decoder lands on different weights without it. The comment at `model.py:56-61` states the rule; read it before any edit. |
| `batch.py` | 160 | `make_masker` (`:15`) + `prepare_masked_batch` (`:33`): masking, obs/imp maps, control append, availability cross-check (`:85-90`). | Batch-key contract (`:140-160`) must stay stable. Control is concatenated **after** masking, at `batch.py:128-130` vs the mask at `:71`. |
| `dataset.py` | 497 | `_sample_xy_dsf` (`:41`), `base_cell_type` (`:69`), `cell_id_map` (`:82`), `h5_depth_center` (`:92`), `CandiKitH5Dataset` (`:109`). Reads scale from `h5.attrs` (`:176-192`); refuses schema v1 (`:171-175`). | `__iter__`'s emitted dict keys/shapes (`dataset.py:439-458`) must not move. The availability test is `float(xm[0]) != -1.0` (`:413`, `:421`). |
| `metrics.py` | 137 | Numeric primitives only: `nb_crps`, `nb_quantile`, `calibration_pit_curve`, `ece`, `spearman`, `pearson`, `r2`, `_cos_dist`, `_steering_index`, `P_EPS`, `PIT_GRID`. | `nb_crps` is a closed-form NB CRPS (validated to ~2.4e-5 relative against exact discrete sums by `tests/test_metrics_primitives.py:40`). Do not "simplify". |
| `eval.py` | 1474 | The measurement stack: `_cluster_bootstrap_ci` (`:190`), `build_eval_units` (`:276`), `eval_M1` (`:523`), `eval_M2` (`:950`), `eval_M3` (`:1068`), `_dsf_counterfactual` (S14, `:1122`), `reference_only_baseline` (`:1217`), `quick_eval` (`:1286`), `evaluate` (`:1354`) and a checkpoint-scoring CLI (`:1392`). Owns `DEPRECATED_VERDICTS` (`:49`). | Assay labels are threaded in as an argument, never declared. Deprecated keys must keep their verdict strings. |
| `train.py` | 1124 | Training driver + CLI: NB loss (`:59`), cosine warmup (`:88`), `_grad_norms` (`:165`), `_train_step` (`:250`), full-coverage round-robin (`:370-434`), `train_and_eval` (`:660`), config echo (`:872-912`), argparse (`:948-1071`). | `train.py:257-283` (step order), `:310` (`Adam`, coupled L2, positional lr), `:693-696` (cudnn determinism pair immediately before `torch.manual_seed`) are FROZEN. Never add `torch.use_deterministic_algorithms` — it crashes this encoder. |
| `param_groups.py` | 418 | h51: the no-decay conditioning group vs the decayed trunk, classified by module **type and identity**, never by name substring. `build_adamw`, `validate_optimizer_config`. Only imported inside `train.py`'s `adamw` branch (`train.py:317`). | There is **no fallback branch**: an unrecognised module raises `ParamClassificationError`. Adding an `else: decay` arm silently changes the experiment. |
| `meta_probe.py` | 323 | h64 arm switch (`off`/`shuffled`/`planted`) for the covariate-gradient meter. Transforms the **assembled batch**, never the architecture. `make_meta_probe`, `META_PROBE_MODES`, `DEFAULT_META_PROBE_DELTA`. | At `off` it returns `None`, so no object is built and no RNG is seeded — that is what makes the control bit-identical. |
| `probes.py` | 221 | h75 read-only forward hooks around the metadata-fusion LayerNorm and the decoder FiLM. `MetaPathProbe`, `make_meta_path_probe`, `effective_rank`. | Nothing here may enter the graph, draw RNG or touch the loss. Built only when `log_fn` is set (`train.py:342-345`). |
| `reference.py` | 412 | h74 per-assay, per-position average reference over `T_` biosamples as a **log-mean offset**. `ReferenceTable`, `reference_path_for`, `h5_fingerprint`, `REFERENCE_PSEUDOCOUNT`, and a `build`/`verify` CLI (`:386`). | Leave-one-out is mandatory (the reference otherwise leaks the answer) and `R` must stay depth-free, or the two offsets compose on different scales. |
| `healthcheck.py` | 539 | Pre-training G0/W1–W4/H1–H4 checks for the cell-identity arm on a **real** baked h5. CLI at `:507`. | A check that cannot run must report SKIP, never pass silently. |
| `verify_bake.py` | 530 | Independent h5-vs-npz verification that deliberately does **not** import the bake. CLI at `:501`. | V2 (column mapping pinned by content) is the check that matters; sharing a reader with `handler.py` would let a reader bug cancel itself out. |
| `compare_arms.py` | 402 | Paired arm-vs-arm comparison over `(T_bios, imp_bios, assay)` targets. CLI at `:367`. | **Two opposite sign conventions live here**: `control − case` for CRPS (a loss), `case − control` for the M2 ablation (a magnitude). Both are pinned by `tests/test_compare_arms.py`. |
| `report.py` | 352 | Regenerates a markdown scorecard + matplotlib figures from a results JSON alone (`[report]` extra, needs `MPLBACKEND=Agg`). CLI: `python -m candi.report <results.json> [--outdir DIR]`. | Assay labels come from the results JSON `config.assays` (`report.py:40-43`), never a literal. |
| `report_h74.py` | 824 | h74 residual-vs-raw report: one figure per verifiable plus `report.md`, from two run JSONs only. CLI at `:811`. | Everything must come from the JSONs — a W&B round-trip makes the report un-reproducible from artifacts. |

### 2.2 `src/candi/prep/` — bake path (needs the `[prepare]` extra: pysam, intervaltree, pandas)

| file | lines | responsibility | do not touch |
|---|---|---|---|
| `panel.py` | 66 | `Panel` dataclass (`:13`) + `load_panel` (`:51`) with strict unknown-key rejection (`:60-65`); validates ≥2 `T_` biosamples, `context_bins % 8 == 0`, square `resolution`, disjoint train/eval chroms, and warns on a single-level `dsf_list` (`:44-48`). | The `%8` and perfect-square rules encode the encoder's pool/block arithmetic; loosening them produces shape errors deep in the tower. Keys starting with `_` are ignored as comments (`:59`). |
| `paths.py` | 42 | `SideFiles(chrom_sizes, fasta, blacklist=None, ccres=None)` + `validate()`; loud warning when the blacklist is empty or absent. | — |
| `handler.py` | 2000 | `CANDIDataHandler`, vendored verbatim from repo `data.py` with 5 surgical edits (E1 side files/panel injection `:95-98,:146`, E2 explicit `includes` `:95`, E3 DSF fallback now raises `:1017`, E4 **int32** counts + overflow guard `:1066-1075`, E5 unparseable `run_type` raises `:1262`). Sole definition of assay→id ordering, covariate assembly, NPZ/FASTA/blacklist IO. | Do not reformat and do not "modernize". Edit E1 must stay inside `__init__` before `_load_blacklist`. |
| `reference_sample.py` | 191 | `make_handler` (`:20`) — builds the handler and runs the **panel/alias bijection gate** (`:43-52`); `resolve_column_order` (`:56`), `assert_panel_bijection` (`:61`), `reference_tensors` (`:90`), `fixed_dsf_pair_maps` (`:170`). | The bijection gate is the fix for the assay-order landmine. Never bypass it. |
| `bake.py` | 522 | ENCODE-style dir → HDF5 schema v2. Window tiling, optional type2 cCRE loci, per-(bios,chrom) bake cache (`:155`), attrs, and the post-bake `_verify` gate (`:426-486`, checks F3 zero-filled meta / F4 DSF depth ladder / F7 raw counts / F15 control availability) which runs **before** the temp file is renamed into place (`:421-422`). CLI at `:489`. | `_enable_bake_cache` is load-bearing for runtime (minutes vs hours). `-1` (never `0`) is the sentinel written for an absent DSF level (`:405-407`). Counts are **int32** (`:349-351`). |

### 2.3 Tests, tools and jobs

| path | responsibility |
|---|---|
| `tools/golden.py` (112) | The bit-exactness gate. `save <ref.pt>` records params / `state_dict` sha / output tensors of the current tree at seed 0 on fixed inputs; `check <ref.pt>` rebuilds and asserts **0 ULP**. The reference file is **not committed** — you generate it before a refactor and check it after. |
| `tests/test_model.py` (18 fns) | Synthetic-tensor model tests: sentinel handling, offset reads row 0, `offset=off ⇒ log2_mu == eta`, per-assay locality, latent invariance to `y_meta`, out-of-range `assay_id`/`run_type` raise, scale axes (`num_assays ∈ {3,8,16}` × `context ∈ {384,768,1536}`), `--d-model` decoupling, DNA-length mismatch raises. No h5 needed. |
| `tests/test_invariants.py` (10 fns) | The three claims that define the decoder: the grouped deconv never mixes lanes (`:106`), conditioning is per-assay and not pooled (`:163`), every decoder FiLM is adaLN-zero so init is an exact no-op (`:187`). Plus the parameter-count anchor (`:207`). |
| `tests/test_lane_view.py` (10 fns) | `lanes_from_channels`/`channels_from_lanes` are pure views, `FiLMLayer` computes the identical product, and `SignalConvTower.lane_shapes` is pinned to a real forward pass (`:112`) so the documented channel schedule cannot drift. |
| `tests/test_bake_gates.py` (10 fns) | Ingestion landmines on a synthetic ENCODE-layout fixture in `tmp_path`: bijection error naming, `-1` vs `0` meta, DSF-ladder forgery detection, **int32** storage past the int16 ceiling (`:305`), unparseable run_type, 3-assay/384-bin round-trip into the model, raw-integer counts out of the loader. |
| `tests/test_param_partition.py` (31 fns) | h51: the no-decay/decay partition is exhaustive, disjoint, has no silent fallback, and `AdamW` at all-zero decay is bit-identical to the frozen `Adam` default. |
| `tests/test_meta_probe.py` (22 fns) | h64: `off` is a strict no-op, the negative control preserves every sentinel, the no-op detector has power, the positive control never leaks into `x_data`. |
| `tests/test_reference.py` (17 fns) | h74 gates L1 (exact leave-one-out), L2 (no `V_`/`B_` contributor), L3 (depth-free), G0, W1–W4, on a synthetic in-process panel. |
| `tests/test_probes.py` (13 fns) | Every probe scalar checked against an independent brute-force loop; the probe must not change the forward output. |
| `tests/test_compare_arms.py` (23 fns) | Both sign conventions pinned from both directions. |
| `tests/test_cell_cond.py` (19 fns) | The 5th metadata row: survives cloze, `T_X`/`V_X` share a row, `cell_cond="off"` is bit-identical, the null arm does not perturb the data stream. |
| `tests/test_metrics_primitives.py` (12 fns) | `nb_crps` vs exact sums and Monte Carlo, PIT uniformity, ECE sensitivity, correlations vs scipy. Data-free. |
| `slurm/bake.sh` (84) | q19-panel bake. Pure CPU *work*, but it still requests `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (`slurm/bake.sh:23`) — deliberately, to route through the GPU account; the reason is in the header comment at `:7-13`. |
| `slurm/bake_eic_full.sh` (76) | The same bake at full EIC scale (35 assays × 89 biosample entries); differs only in panel, output name, walltime and memory. |
| `slurm/build_reference.sh` (62) | One-off h74 reference build + `candi.reference verify`. CPU work; the MIG slice is requested because the venv's Python needs the `x86-64-v4` node class. |
| `slurm/train.sh` (66) | 2 arms × 3 seeds array (`--array=0-5`, `:19`). `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (`:16`). |
| `slurm/example.sh` (65) | The smallest end-to-end run: bake + 3-epoch train + report, ~15 min. Use it to prove an install. |
| `configs/panel.q19.json` | The 8-assay / 10-biosample q19 panel. |
| `configs/panel.eic_full.json` | The full 35-assay / 89-biosample-entry EIC panel — the scale the shipped anchors are measured at. |
| `configs/panel.example.json` | Minimal 3-assay template, annotated. |
| `configs/panel.gatec.json` | 5-assay / 512-bin panel, kept as a non-default-scale smoke panel. **No job script drives it** — run it by hand (§4.7). |
| `pyproject.toml` / `requirements-fir.txt` | Portable pins / literal Alliance-Canada (`+computecanada`) rebuild list. |
| `research/METADATA_AUDIT.md` | **Currently a zero-byte placeholder.** Any citation to it in this repo's prose is unresolvable until it is filled in. |

The directory is named `slurm/`, **not** `jobs/`, because the parent repo's `.gitignore` has an
unanchored `jobs/` rule that would silently swallow the sbatch scripts (`.gitignore:10-13`). Avoid
`models/` and `logs/` for the same reason.

---

## 3. INVARIANTS — check yourself against this list after every edit

1. **Construction order is load-bearing.** `CandiModel.__init__` builds `V2Encoder` **first**, then
   *replaces* `encoder.metadata_embedding` with an identically-configured `MetadataEmbedding`, then
   builds `SymmetricDecoder` (`model.py:55-70`). The replacement is a functional no-op that consumes
   a block of global-RNG draws; removing it re-samples every decoder weight. That may be a legitimate
   cleanup, but it must be its own labelled change with a fresh `tools/golden.py` recording — never a
   side effect of moving code between files. Inside `MetadataEmbedding`, the `cell_embedding` table is
   constructed **only** when `num_cells > 0` (`encoder.py:148-153`), which is what keeps the 4-row
   model bit-identical.
2. **No imports reaching back into the parent repo.** Nothing under `src/` may contain `from sandbox`,
   `from model import`, `from _utils import`, or `SANDBOX_ASSAYS`. Import-time third-party top-levels
   must stay within `{torch, numpy, h5py, x_transformers, einops, einx, loguru, sympy, scipy}` plus
   the `[prepare]`/`[report]` extras in `prep/`/`report*.py`.
3. **The assay order is derived and asserted, never declared.** There is **no assay-name string
   literal anywhere under `src/`**. Bake derives the order from the handler's filtered
   `experiment_aliases` keys (`prep/reference_sample.py:56-58`), asserts a bijection against the
   requested panel (`:61-65`), asserts `assay_to_id == range(A)` and `control_assay_id == A`
   (`prep/reference_sample.py:48-52`), and records it in `h5.attrs['assays']`. Everything downstream
   reads it back (`dataset.py:176-177`) and threads it as an argument (`eval.py`, `report.py`).
   Adding a hard-coded list of assay names anywhere is a defect.
4. **`arcsinh` lives in the model, not the loader.** The loader yields **raw counts**; the transform is
   applied inside `V2Encoder.encode` via `_apply_signal_transform` (`encoder.py:783-793`, called at
   `:1074`), with `signal_transform="arcsinh"` set at `model.py:51`. Never transform in `dataset.py`
   or `batch.py` — that is a silent double-arcsinh. (Note `EncoderConfig.signal_transform` defaults to
   `"log1p"` at `config.py:68`; the shipped value is the model's override, not the dataclass default.)
5. **Counts stay raw non-negative integers as NB targets.** `y_data` is the NB target
   (`train.py:77`). The bake writes **int32** raw counts (`prep/bake.py:349-351`, `prep/handler.py:1066-1075`)
   and `_verify`'s F7 check rejects an already-transformed range (`prep/bake.py:455-472`);
   `tests/test_bake_gates.py:370` guards the loader side.
6. **The control channel is never masked.** `prepare_masked_batch` masks first (`batch.py:71`), then
   concatenates the control column at index `A` (`batch.py:128-130`), so it is structurally
   unmaskable. Model inputs are therefore `F = num_assays + 1` wide on the input side and
   `num_assays` wide on the target side.
7. **`obs` / `imp` mean unmasked / masked**, not biological observed/imputed. `observed_map` =
   unmasked-and-available, `masked_map` = cloze (`batch.py:73-82`); the loss adds the two means with
   the `imp` term weighted by `--imp-weight` (`train.py:77-85`, default 1.0).
8. **Covariates stay raw.** The 4 rows are `[log2_depth, assay_id, read_length, run_type]` with **no
   normalization**. Sentinels are `MISSING=-1` and `CLOZE=-2` and are kept distinct everywhere
   (`_vendored.py:15-16`, consumed at `encoder.py:165-180`, `:219-249`). `assay_id == num_assays` is
   the control slot; the embedding table is `Embedding(num_assays + 3)` (`encoder.py:144`) = real ids
   + control + MISSING + CLOZE. Do not "fix" the `+3` to `+2`.
9. **The optional 5th row is cell identity, and it does not vote on availability.**
   `_infer_availability_from_meta` slices `meta[:, :4, :]` (`encoder.py:764`) because availability is
   a per-assay fact while `cell_id` is a per-sample one. `num_cells` comes from the h5's own biosample
   list, never a flag (`train.py:701`).
10. **Scale is a property of the baked file, never a CLI flag.** `num_assays`, `context_bins`,
    `resolution`, `dsf_list`, `num_cells`, assay order and train/eval chromosomes come from `h5.attrs`
    (`dataset.py:176-192`) and are passed into `build_model` from the dataset (`train.py:698-705`).
    Never add flags for them.
11. **`d_model=0` means auto** `= (num_assays+1) * expansion**n_cnn_layers` (`encoder.py:952-954`) —
    i.e. transformer width silently tracks panel size (35 assays → 288; 8 assays → 72). When you
    change the panel, set `--d-model` explicitly. `tests/test_model.py:180-197` pins both halves.
12. **Determinism pattern:** `cudnn.deterministic=True`, `cudnn.benchmark=False`, then
    `torch.manual_seed(seed)` immediately before `build_model` with nothing touching the RNG in
    between (`train.py:693-705`). `transformer_layer_drop` must remain 0.0 (`encoder.py:1030-1035`) —
    it consumes global RNG inside forward.
13. **Every default-off arm must be a *strict* no-op, not an inert one.** `--reference off` builds no
    table and emits no `log_ref` key (`train.py:732-733`, `batch.py:156-159`); `--meta-probe off`
    constructs no object and seeds no RNG (`train.py:723-727`); `--meta-gain 1.0` skips the multiply
    rather than performing it (`decoder.py:235-236`); `--cell-cond off` builds no embedding table
    (`encoder.py:152-153`); `--optimizer adam` never imports `param_groups` (`train.py:317`). When you
    add an arm, follow this pattern — an "inert but present" default silently moves the RNG stream.
14. **SLURM GPU spec is fixed:** every `#SBATCH --gres` line in this repo reads
    `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, and all five scripts carry one — `bake.sh:23`,
    `bake_eic_full.sh:32`, `build_reference.sh:24`, `example.sh:16`, `train.sh:16`. The bake and the
    reference build do no GPU compute; they request the smallest MIG slice only to route through the
    GPU account (rationale at `bake.sh:7-13`). Never write any other `--gres` spec on any line.
15. **Never write a `0`-filled metadata row.** An absent DSF level or unavailable assay is `-1`
    (`prep/bake.py:405-407`; availability test is `float(xm[0]) != -1.0`, `dataset.py:413,421`). A
    zero-filled row marks every assay available at `log2(depth)=0` with all-zero counts: loss
    descends, metrics are garbage. `dataset._check_available_columns_nonzero` (`dataset.py:215`) and
    bake `_verify` F3 (`prep/bake.py:437-441`) both guard this.

**Never infer what a tracker id means.** `h9`, `h33`, `h34`, `h45`, `h48`, `h51`, `h64`, `h74`, `h75`,
`h76`, `q19` and friends appear throughout the prose of these docs and in the source comments. **The
tracker does not ship, and this repository contains no decoder ring for those ids** —
`research/METADATA_AUDIT.md` is an empty placeholder. Treat every such id as unresolvable and report
it as such rather than reconstructing its meaning from context. Confidently-interpolated provenance is
how the retracted `Δη = 0.833` reached three separate records.

---

## 4. COMMON TASKS

Set once (adapt to your machine):

```bash
export KIT=/project/6014832/mforooz/EpiDenoise/candi
export VENV=/project/6014832/mforooz/EpiDenoise/candi_venv
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4.1 Install

```bash
pip install -e "$KIT"
```

```bash
pip install -e "$KIT[prepare,report,test]"
```

The first is train-only deps; the second adds bake, figures and pytest. On Alliance Canada, build the
venv from `requirements-fir.txt` first (it documents the exact `module load python/3.10.13` →
`--no-index` → `pip install x-transformers==2.11.23` sequence), then `pip install -e "$KIT" --no-deps`.

### 4.2 Run the tests

```bash
cd /tmp && python -m pytest "$KIT/tests" -q
```

Run from a directory that is **not** the parent repo — that is part of the proof that the tree is
cwd-independent. The suite is 185 test functions across 11 files, expanding to ~213 cases under
`@pytest.mark.parametrize`; it is CPU-only and needs no h5 and no GPU.

### 4.3 Bake a new panel

1. Write a panel JSON (copy `configs/panel.example.json`). Every key is consumed; unknown keys are
   rejected (`prep/panel.py:60-65`), except keys starting with `_`, which are treated as comments
   (`:59`). Required: ≥2 `T_` biosamples, `context_bins % 8 == 0`, `resolution` a perfect square,
   `train_chroms` disjoint from `eval_chroms`.
2. Bake:

```bash
python -m candi.prep.bake --root /project/6014832/mforooz/DATA_CANDI_EIC --panel "$KIT/configs/panel.q19.json" --out /scratch/$USER/candi/q19.h5 --fasta /project/6014832/mforooz/EpiDenoise/data/hg38.fa --chrom-sizes /project/6014832/mforooz/EpiDenoise/data/hg38.chrom.sizes --type2-ccre 0 --type2-non 0 --seed 42
```

Optional: `--blacklist BED`, `--ccres BED` (required iff `--type2-ccre`/`--type2-non` > 0),
`--max-tile-per-chrom N` and `--max-windows N` (smoke bakes), `--allow-missing-control`. `--out` has
no default; point it at scratch. Note `--type2-ccre` / `--type2-non` / `--seed` are **bake CLI flags,
not panel keys** — `load_panel` rejects unknown keys. On SLURM: `sbatch "$KIT/slurm/bake.sh"` for the
q19 panel or `sbatch "$KIT/slurm/bake_eic_full.sh"` for the full EIC panel (no GPU compute, but both
scripts do request the MIG slice — see invariant 14).

To check the result against the source tree rather than only against itself:

```bash
python -m candi.verify_bake --h5 /scratch/$USER/candi/q19.h5 --root /project/6014832/mforooz/DATA_CANDI_EIC --fasta /project/6014832/mforooz/EpiDenoise/data/hg38.fa --n 300
```

### 4.4 Train an arm

```bash
python -m candi.train --h5 /scratch/$USER/candi/q19.h5 --out-dir /scratch/$USER/candi/runs --offset on --seed 0 --tag candi_on_s0 --weight-decay 0.0 --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 --fg-frac 0.02 --n-boot 1000
```

`--offset off` (alias `--arm offset_off`) trains the other arm. Writes `{out_dir}/{tag}.json` and
`{out_dir}/{tag}.ckpt`; with `--eval-every N > 0` it also writes `{tag}.best.ckpt` and scores that one
instead of the last (`train.py:832-839`). Both arms × 3 seeds as one array:
`sbatch "$KIT/slurm/train.sh"`.

`--d-model` is the flag you must not forget when the panel is not 8 assays (invariant 11). There is
**no `--compat-q19` flag** — that architecture pin was removed with the predecessor model.

### 4.5 Evaluate an existing checkpoint

```bash
python -m candi.eval --h5 /scratch/$USER/candi/q19.h5 --ckpt /scratch/$USER/candi/runs/candi_on_s0.ckpt --out /scratch/$USER/candi/runs/candi_on_s0_rescore.json --offset on --m3-regions 40 --n-boot 1000 --eval-budget 50000000
```

The architecture flags must match those the checkpoint was trained with, or the `strict=True` load
fails loudly (`eval.py:1448`). That includes `--decoder-lane`, `--lane-norm`, `--reference` and
`--meta-embed-layernorm`, each of which changes the parameter set (`eval.py:1413-1433`). Add
`--include-deprecated` to also emit the audited-and-rejected metric keys, each carrying its verdict
string. Figures + markdown from a results JSON:

```bash
python -m candi.report <results.json> --outdir DIR
```

### 4.6 Add a knob

1. Add the argparse flag in `train.py:948-1071` (and/or `eval.py:1402-1434` if it affects scoring).
2. Thread it through `train_and_eval` (`train.py:660-672`) into `build_model` / `train` / `evaluate` —
   do not read globals.
3. Add it to the `config` echo dict (`train.py:872-912`) so every results JSON is self-describing.
4. Make the default a **strict** no-op, per invariant 13, and re-run §5.
5. If it is scale (`num_assays`, `context_bins`, `resolution`, `dsf_list`, `num_cells`, chroms), it is
   **not** a knob — it belongs in the panel and the h5 attrs (invariant 10).

### 4.7 Change the panel size

Panel size flows: `panel.json` → handler → derived assay order → `h5.attrs` → dataset → `build_model`.
To go from A assays to N:

1. Edit `assays` / `biosamples` in the panel JSON and re-bake (§4.3). Nothing in `src/` changes.
2. Train with an explicit `--d-model` (invariant 11). Attention inner dim is `nhead * 64` regardless
   of `d_model`, so capacity does not scale with the panel unless `--nhead` also rises.
3. Expect the parameter anchor in `tests/test_invariants.py:216` to be meaningless for the new panel —
   it is an anchor for the 35-assay build only.
4. Re-run `tests/test_model.py::test_scale_axes_independent` and `tests/test_bake_gates.py` — they
   already cover 3/8/16 assays and 384/768/1536 bins.

`configs/panel.gatec.json` (5 assays, 512 bins) is kept for exactly this smoke test. There is no job
script for it; bake and train it by hand.

---

## 5. VALIDATION

### 5.1 The bit-exactness gate (CPU, seconds, no SLURM, no data)

**What it proves:** that a refactor which claims to only move code really did only move code. It
constructs the model from one seed on fixed inputs and compares parameter count, `state_dict` key set,
`state_dict` value digest and all five output tensors at **0 ULP**.

Record **before** you start editing, check **after**:

```bash
python tools/golden.py save /tmp/candi_ref.pt
```

```bash
python tools/golden.py check /tmp/candi_ref.pt
```

The recorded configuration is 35 assays / 768 bins / `d_model=288` / `decoder_lane=8`
(`tools/golden.py:20-25`), and the inputs deliberately drive assays 3, 11 and 27 to `CLOZE` so the
mask-token and sentinel branches are exercised (`tools/golden.py:50-52`).

**Read the failure mode it reports.** `state_dict keys changed` means the module tree moved;
`state_dict VALUES changed — the RNG stream moved` means construction order changed (invariant 1);
a nonzero `max|diff|` with matching weights means the *arithmetic* changed. None of the three is
noise.

> **The reference `.pt` is not committed**, so this gate proves *self-consistency across your edit*,
> not agreement with any historical run. There is no external copy-proof in this repository — the
> predecessor's `candi.compat` gate, its committed `goldens/*.pt` and `tests/test_compat_q19.py` were
> all removed along with the model they anchored.

### 5.2 The test suite

```bash
cd /tmp && python -m pytest "$KIT/tests" -q
```

The load-bearing cases, if you are triaging rather than running the lot: the parameter anchor
(`tests/test_invariants.py:207`), the three decoder invariants (`:106`, `:163`, `:187`), the lane-shape
pin (`tests/test_lane_view.py:112`), and the partition exhaustiveness proof
(`tests/test_param_partition.py`).

### 5.3 Pre-training health check on real data

```bash
python -m candi.healthcheck --h5 /scratch/$USER/candi/eic_full.h5 --device cuda
```

Unlike `tests/`, this touches a real baked h5 and answers "does the conditioning signal actually reach
the model and move anything". Read its two legs differently: the encoder FiLM is Xavier-initialised
and a dead leg at init is a real failure, while the decoder FiLM is adaLN-zero and is **exactly 0.0 at
init by construction** — there the question is whether it grows once gradients flow
(`healthcheck.py:18-24`).

> **Gates A, B and C are gone.** The old §5 described a three-tier gate built on `candi.compat`,
> `slurm/gate.sh`, `slurm/gatec.sh` and a `.BUILD_PLAN.md` acceptance-band table. None of those files
> exists in this repository, and the acceptance bands they carried were set on the predecessor
> architecture at 8 assays. Do not reconstruct them from memory; §5.1–5.3 are the gates this tree
> actually has.

---

## 6. FAILURE MODES AND THEIR SIGNATURES

| symptom | likely cause | where to look |
|---|---|---|
| `tools/golden.py check`: wrong param count or `state_dict` values changed | construction-order drift — most often the `encoder.metadata_embedding` replacement was removed or moved | `model.py:55-70` |
| `tools/golden.py check`: keys unchanged, values unchanged, nonzero `max|diff|` | the arithmetic changed, not the init; diff the forward path | `model.py:72-74`, `decoder.py:226-268` |
| `state_dict` reports unexpected `encoder.transformer_blocks.*` keys | x-transformers version drift | `pyproject.toml:22` pin (`x-transformers==2.11.23`) |
| `ValueError: ... is schema v1; candi requires v2` | h5 baked by the old repo pipeline | re-bake with `candi.prep.bake` (`dataset.py:171-175`) |
| `ValueError: panel/alias mismatch: missing=[...]` | a panel assay is absent from the data root, or the alias filter dropped it | `prep/reference_sample.py:43-52,61-65`; check the assay name spelling against the directory |
| Loss descends smoothly, metrics are garbage, everything "available" | zero-filled metadata rows (absent DSF level written as 0 rather than -1) | `dataset.py:215-243`; bake `_verify` F3 (`prep/bake.py:437-441`) |
| `AssertionError: F4 ... not the downsampled data they claim to be` | a `counts_dsf{k}` dataset holds DSF-1 data | `prep/bake.py:443-450`; handler edit E3 (`prep/handler.py:1017`) |
| `ValueError: count ... exceeds int32` | counts wider than int32 in the source NPZ | handler edit E4 (`prep/handler.py:1066-1075`); re-bake with a wider dtype |
| `ValueError: unparseable run_type ...` | a metadata value outside the expected vocabulary (previously inherited the previous assay's value silently) | handler edit E5 (`prep/handler.py:1262`) |
| `ValueError: metadata must be 4 rows ...` / `assay_id ... exceeds table bound` / `run_type ... exceeds table bound` / `cell_id ... exceeds table bound` | a hand-built or externally-supplied metadata tensor | `encoder.py:190-196,212-218,232-238,255-261`; these are deliberate — previously they aliased onto the MISSING/CLOZE slots |
| `ValueError: DNA length G != context L x resolution` | panel `resolution`/`context_bins` mismatch between h5 and model | `encoder.py:1075-1080` |
| `ValueError: Signal vs metadata assay availability mismatch` | masking config other than assay-only (`p_full_loci`/`p_chunks` nonzero) under `missing_data_mode='mask_token'` | `encoder.py:1046-1051` |
| `ValueError: Availability/supervision mismatch` | the loader produced `y_avail` that disagrees with the derived supervision maps — a data pipeline bug | `batch.py:85-90` |
| `ValueError: no training windows: regime=...` | `type2_loci` regime on an h5 baked with `--type2-ccre 0 --type2-non 0` | `dataset.py:205-208` |
| `[data] WARNING: single-DSF ladder` | `dsf_list` has one level; per-assay independent DSF sampling is inert and the depth-steering signal is absent | `dataset.py:209-211`, `prep/panel.py:44-48` |
| `[train] WARNING: --mask-fraction is INERT` | `--mask-fraction` set while `--p-full-assay 1.0`; `DataMasker._mask_full_assay` never reads it | `train.py:294-296`, `_vendored.py:114-160` |
| Training runs but imputation loss is ~0 | biosamples with ≤1 available assay are skipped by the masker, so no cloze is produced | `_vendored.py:133-137`; the panel gate requires ≥2 `T_` biosamples but not ≥2 assays each |
| `ParamClassificationError: ...` under `--optimizer adamw` | a module type `param_groups` does not positively recognise | `param_groups.py:26-29`; decide where its parameters belong, do not add a fallback |
| `ValueError: trunk_wd=... is inert under optimizer='adam'` | `--trunk-wd` passed without `--optimizer adamw` | `train.py:305-309` — deliberate, so a run cannot record a decay it never applied |
| `ValueError: depth_center mismatch: the reference was built at ...` | `--depth-center` overridden on one side only, so the h74 reference offset and the decoder offset sit on different scales | `train.py:744-749`, `eval.py:1457-1459` |
| Host OOM at ~N×1.6 GB | the shared RAM buffer was not shared across the per-biosample datasets | `train.py:375-383` (`ds._ram_buf = shared`) |
| `raise ValueError("missing_data_mode='mask_stem' not shipped ...")` / `transformer_type='production_dual' not shipped` | a config branch that depends on parent-repo modules | intentional; use `mask_token` / `xtransformers` (`encoder.py:939-943`, `:1008-1012`) |
| Run-to-run numbers differ on the same seed | `torch.use_deterministic_algorithms` added, `cudnn.benchmark` re-enabled, or something touched the RNG between `manual_seed` and `build_model` | `train.py:693-705` |
| `MPLBACKEND` / display errors from `report.py` | headless node without `MPLBACKEND=Agg` | §4 env block |

---

## 7. HOW TO READ THE SCIENCE

> ### ⚠ PROVENANCE WARNING — READ BEFORE QUOTING ANYTHING IN THIS SECTION
>
> Everything below was recorded on the **predecessor architecture**: an 8-assay q19 panel with an
> ungrouped ~3.10 M-parameter decoder trunk and a single decoder FiLM tap. **That model is not the
> model in this repository.** The current model is 2,353,634 parameters at 35 assays, with a grouped
> deconv trunk and four FiLM taps (§1).
>
> Every source document these numbers were drawn from — `H48_REPORT.md`, `research/H48_SCORECARD.md`,
> `.BUILD_PLAN.md`, `TRADEOFF.md` — is **absent from this repository**, and
> `research/METADATA_AUDIT.md` is a zero-byte placeholder. None of it can be verified from this tree,
> and no checkpoint that would let you re-score it ships here.
>
> `README.md:67-72` records that the current architecture was selected by a **nine-arm, two-seed
> ladder on the full 35-assay panel**, and that only the winning arm ships. **Those runs are not
> reported anywhere in this repository.** Treat §7 as inherited context about a related system, not as
> a description of this one. If you need current numbers, train and score an arm (§4.4–4.5).

### 7.1 The findings that are architecture-independent

These are properties of the *data and the measurement design*, not of the model, so they carry over.
Quote them with the noise floor attached; do not quote the arm-level numbers in §7.2 without the
banner above.

1. **Quote the noise floor with every number.** Effective replication in the recorded q19 experiments
   was **12 held-out targets / 5 biosample pairs / 4 cell types**, with `(T_RWPE2, B_RWPE2)` supplying
   7 of the 12. Target-clustered bootstrap noise floor on macro CRPS was **~0.09**; per-comparison
   uncertainty ±0.13. A single **seed** change moved pooled imputation CRPS by 0.1195, Spearman by
   0.0562, ECE by 0.0354. Sign-test resolution at n=12 is quantized: 10/12 → p=0.039, 11/12 → 0.0063,
   12/12 → 0.00049. The full EIC panel this repo now targets has 96 target tracks across 38 eval
   pairs (`configs/panel.eic_full.json`), so the floor there is different and has not been measured.
2. **The unit of comparison is the target, never the position.** Position-level bootstrap intervals
   ran ~24× too narrow because positions within one target are not independent draws. This is why
   `eval.py:190` exists and why every position-level `_bootstrap_ci` key is deprecated
   (`eval.py:57-61`), and why `compare_arms.py` reuses the clustered helper verbatim rather than
   reimplementing it.
3. **Known bounds of the 8-assay q19 panel — do not design around them without re-selecting the
   panel.**
   * `run_type` was **analytically unidentifiable** on that panel:
     `H(run_type | assay_id, read_length) = 0.000 bits` (n=26 `T_` records). The full EIC panel retains
     0.551 bits. A run_type steering demonstration was impossible there; it needs a re-selected
     biosample panel, not an architecture change.
   * **DSF only down-samples**, so the upward-depth regime is never trained (`dataset.py:41-63`), and
     7 of 12 q19 eval targets sat above their per-assay training depth ceiling (worst +1.43 log2).
     Depth steering on those targets is extrapolation into an untrained direction. This is a property
     of the sampler and still holds.
   * Depth, `read_length` and `run_type` were mutually collinear at n=38. Attribution among the three
     exposure covariates is not identified on this dataset; any claim of the form "covariate X carries
     Y% of the exposure signal" is unsupportable.
   * In this dataset `assay_id ≡ slot index`, and the decoder emits a dedicated lane per slot
     (`decoder.py:144-148`), so assay identity is carried **structurally**. Any assay-steering verdict
     bounds the **prompt pathway**, not assay-awareness. This is *more* true of the current grouped
     decoder than it was of the predecessor.
   * Eval runs `dsf_sampling='off'` with `apply_mask=False`, so `x_data == y_data` for available
     assays and "denoising" is autoencoding. Nothing in the suite measures denoising as distinct from
     autoencoding.
   * `crps_oracle_scaled` is an **in-sample upper bound** — `c*` is fitted on the same targets it
     scores — and `scale_error` can be slightly negative (−0.0008 observed).
4. **Capability of the model, not of this repo's outputs:** the model is counts-only NB. Do not claim
   peak calling, peak precision/recall/AUROC, or p-value tracks.
5. **Metric hygiene.** Only the default-emitted metrics may back a claim. Keys behind
   `--include-deprecated` ship with a verdict string in `eval.py:49-68` and must never be quoted as
   evidence: the `read_length` flip arm (7/12 flips are out of training support), the shuffled-depth
   `null` (a mathematical no-op — one `[4,F]` tensor broadcast over the batch, so the permutation is
   bitwise identity), `frac_min_at_true` (superseded by S14), all position-level `_bootstrap_ci`
   outputs (~24× too narrow), `frac_direction` (strict `>` reports ties as 0% correct), and
   `median_eta_slope` / `offset_independent` (decided by the sign of ~1e-17 float noise under
   offset-ON). The condition-recoverability probe (S23) is **withdrawn** and is not shipped in any
   mode: its ordering is inverted against every other instrument. `pearson_log1p` and `spearman_raw`
   are computed in different spaces and must never be quoted as a pair.
6. **Float resolution.** GPU-vs-CPU on identical weights reproduces macro CRPS to 4 decimals;
   per-assay values move 2e-6…3.3e-4, and `nb_crps` itself is accurate to ~2.4e-5 relative. Anything
   at 1e-7 or below in `eta`/`mu` is float noise.
7. **Per-assay labels.** Earlier research-repo documents labelled per-assay results with a *permuted*
   assay list. The corrected order is the one derived from the handler aliases and stored in
   `h5.attrs['assays']`. Under corrected labels, the largest q19 CRPS contributor was **DNase-seq**
   (not ATAC-seq) and the collapse outlier was **H3K9me3** (not H3K27ac). Invariant 3 is what keeps
   this from recurring.

### 7.2 The recorded predecessor results — historical only

**These describe the 8-assay ungrouped-trunk model, not the model in this repository.** They are
retained because the offset ON/OFF tradeoff they document is the reason `--offset` exists, and because
several of them are explicitly *retractions* an agent should not re-derive. They are single-seed scores
of four historical checkpoints that do not ship, re-scored on CPU (608 eval units, 1215 target-records,
12 held-out targets). **Do not present any of these as a property of the current model, and do not
compare a current run against them.**

| arm | macro CRPS | oracle-scaled (capability) | scale_error | macro Sp | pooled imp Sp | ECE | beats honest marginal |
|---|---|---|---|---|---|---|---|
| `wd0_on_s0` (offset ON, wd=0) | 1.3413 | 1.3077 | 0.0336 | 0.5653 | 0.6372 | 0.0533 | 7/8 |
| `main_s0_perassay` (offset ON, wd=1e-4) | 1.4950 | 1.4210 | 0.0740 | 0.5051 | 0.5327 | 0.0615 | 5/8 |
| `offoff_s0_perassay` (offset OFF) | 1.9023 | 1.3871 | 0.5152 | 0.4647 | 0.4007 | 0.0968 | 2/8 |
| `wd0_off_s0` (offset OFF, wd=0) | 2.0561 | 1.4026 | 0.6535 | 0.4641 | 0.3800 | 0.0782 | 1/8 |

The M3 ratio column that used to accompany this table has been **removed**: it was carried forward
from a pre-fix run against a `between` pool that admitted same-region pairs, was never re-scored, and
the fix cited for it (`eval.py:917-932`) no longer points at the code it described.

Steering, sentinel-free and target-clustered:

| arm | total told-depth slope | assay real→real max\|Δη\| | run_type clustered CI (n=12) | supports direction | sign-test p |
|---|---|---|---|---|---|
| `wd0_on_s0` | 1.0000 | **0.0023** | [−0.00066, +0.000087] | no | 1.000 |
| `main_s0_perassay` | 1.0000 | 0.0000 | [0, 0] | no | n/a (all ties) |
| `offoff_s0_perassay` | 0.8869 | **4.1772** | [+0.1179, +2.1804] | YES | 0.039 |
| `wd0_off_s0` | 1.0325 | **9.7144** | [−0.2326, +9.4084] | no | 0.039 |

Four readings of that table survive as *guidance*, because each is a structural argument rather than a
measured constant:

1. **Never present the four-arm ordering as established.** Under the oracle per-assay scale the spread
   compressed 0.7148 → 0.1133 (84%) against a ~0.09 noise floor, and only "`wd0_on` is best on
   capability" survived inference. The published 4-dp ordering was the modal bootstrap ordering at only
   45% of replicates.
2. **The offset head's depth response is arithmetic, not learned.**
   `log2_mu = (depth − depth_center) + eta` (`decoder.py:257-259`) is a closed-form thinning identity —
   NB is closed under binomial thinning with `n` preserved, and DSF downsampling *is* exactly that
   generative model. A told-depth slope of 1.0000 is the arithmetically correct answer produced by a
   hardwired subtraction, **not** evidence of learned conditioning. This argument is about the head
   arithmetic, which is unchanged, so it still applies.
3. **Read the clamp tail next to any slope.** `log2_mu` is clamped to `(-15, 30)` (`decoder.py:264`)
   and it saturates: 16.9% of q19 targets clamped somewhere on the offset-ON `wd=0` arm (p90 clamp
   fraction 0.475). Always emit `frac_targets_any_clamp` / `p90` / `max` alongside a slope.
4. **`h45` — the hybrid hypothesis — is recorded REFUTED but was never tested.** It was refuted **on
   its premises**, with **no hybrid arm ever trained**, and one leg of the refutation rested on a
   retracted 0.833 steering figure that turned out to be a MISSING-sentinel artifact. Do not present a
   hybrid as either available or ruled out by experiment. Note that `--offset` is a boolean all the way
   down (`train.py:1073` → `build_model(use_offset=…)` → `decoder.py:257`), so a β-schedule arm is a
   real change, not a flag.
