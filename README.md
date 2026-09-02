# CANDI

**C**onfidence-**A**ware **N**eural **D**enoising **I**mputer — self-supervised epigenome imputation
and denoising from raw sequencing read counts plus experimental covariates.

CANDI consumes **raw counts**, not normalized signal, and emits **probability distributions** rather
than point estimates, so it handles batch effects directly, reports calibrated uncertainty, and does
zero-shot imputation and denoising on cell types it has never seen.

## The model

**→ [`src/candi/README.md`](src/candi/README.md) — the diagram, the shapes, and every architecture
flag at its default.**

That page is *generated*, by `tools/arch_diagram.py`, from a real forward pass through
`build_model()`. Nothing on it is typed by hand and nothing on it can drift:
`tests/test_arch_readme.py` fails the suite the moment the page and the model disagree. This section
stays a pointer for exactly that reason — a second hand-written copy is a second thing to go stale.

In one paragraph: a Conv1D tower grouped by track (`groups = A+1`), so each assay's channels stay
disjoint, with per-assay FiLM after every block; a dense Conv1D tower over DNA; a linear fusion; two
RoPE transformer layers; then a grouped deconv mirror of the signal tower at constant lane width,
with a FiLM tap after the input projection and after each deconv block, and a weight-shared
Negative Binomial head per assay. The mean uses a depth-offset log link, so telling the model a
different sequencing depth *scales* the prediction rather than making it relearn scale.

## Two conventions that trip everyone up

- **`obs` and `imp`** in every loss and metric mean **unmasked** and **masked** positions. They do
  *not* mean biologically observed and imputed.
- The **control channel** is index `A` in `x_data` and is never masked.

## Layout

```
src/candi/
  README.md      the generated architecture page — diagram, shapes, every default
  arch/          what generates it reads and writes: arch.json, the graph, the layer table
  model.py       the model, and the batch helpers train/eval/healthcheck share
  encoder.py     conv towers, metadata embedding, FiLM, mask token, transformer
  decoder.py     grouped deconv trunk, per-lane norm and FiLM, NB head
  dataset.py     the H5 reader        batch.py    masking and batch prep
  train.py       training + the run JSON      eval.py / metrics.py   the M1/M2/M3/S14 instrument
  prep/          bake an H5 from ENCODE tracks
  store/         the whole-genome corpus store — reader, regime, StoreDataset
tools/golden.py  bit-exactness gate — every change must clear it before the next one starts
tools/arch_diagram.py  writes src/candi/README.md from a real forward pass; `build` / `check`
tests/           the suite; no GPU required
```

## Running it

```bash
pip install -e .
python -m candi.train --h5 <panel.h5> --out-dir <runs> --epochs 10 --steps-per-epoch 2000
python -m candi.bench --h5 <panel.h5> --ckpt <run.ckpt> --arch-from <run.json> --out <scores.json>
pytest tests/ -q
```

Always re-score with `--arch-from <run>.json`. Every architecture flag changes the `state_dict`, and
that file carries the exact arguments the checkpoint was built from — so nothing has to be retyped,
and nothing can be retyped wrong.

## The corpus store

There is a second data path. `CANDI_STORE` holds whole chromosomes rather than pre-cut windows, so
window size, context length, the DSF ladder, the chromosome split and the assay column order stop
being frozen into an h5 and move into a regime file read at load time — one store, every regime, no
re-bake. It is built and lives on Fir at `/project/def-maxwl/mforooz/CANDI_STORE`: `eic/` is
54.94 GB (89 biosamples, 35 assays), `merged/` is 406.06 GB (361 biosamples, 47 assays), beside a
shared 884 MB `genome/`.

```python
CorpusStore("/project/def-maxwl/mforooz/CANDI_STORE/eic")["T_DND-41"].counts("chr1", 0, 768)
```

`train.py` trains off it: `--store <regime.json>` instead of `--h5 <baked.h5>`, exactly one of the
two, and everything the h5's attrs used to freeze — context length, DSF ladder, chromosome split,
assay column order — comes from the regime file. Evaluation is still h5-only (task `t14`), so a
store run writes a training curve and checkpoints and no M1/M2/M3. The store lands beside the bake
rather than replacing it. `STORE.md` is the contract and the recipes.

## What is tunable

Scale is **never** a flag: `num_assays`, `context_bins`, `resolution`, the assay order, the DSF
ladder and the chromosome split all come from the H5's own attributes. Everything below is a
deliberate choice, and every one of them defaults to the shipped model.

| group | flags | default |
|---|---|---|
| objective | `--offset` `on\|off` — whether the depth term enters the log link at all | on |
| geometry | `--n-cnn-layers` `--conv-kernel-size` `--pool-size` `--expansion-factor` `--n-deconv-layers` `--deconv-upsample` `--deconv-kernel-size` `--decoder-lane` | 3 · 3 · 2 · 2 · 3 · 2 · 3 · 8 |
| width | `--d-model` (`0` = derive from the panel) · `--nhead` · `--n-transformer-layers` · `--embed-dim` | 0 · 4 · 2 · 32 |
| metadata pathway | `--meta-embed-layernorm` `on\|off` · `--meta-gain` · `--depth-center` (default: derived from the h5) | on · 1.0 · derived |
| regularisation | `--dropout` | 0.1 |
| normalisation | `--conv-norm` `layer\|lane\|group\|batch\|instance` · `--deconv-norm` `lane\|group` · `--transformer-norm` `layer\|rmsnorm\|simple_rmsnorm\|scalenorm` · `--transformer-norm-placement` `pre\|post\|sandwich` · `--attn-qk-norm` | layer · lane · layer · pre · off |
| conditioning | `--film-taps`, a comma set over `pre_conv` `per_conv` `post_conv` `per_transformer` `pre_deconv` `per_deconv` `post_head` · `--film-init-encoder` · `--film-init-decoder` | `per_conv,pre_deconv,per_deconv` · xavier · zero |
| heads | `--head-sharing` `shared\|per_assay` · `--head-hidden` · `--heads`, a comma set over `count` `signal` `peak` | shared · 0 (match the lane) · `count` |
| optimisation | `--lr-schedule` `cosine\|linear\|constant` · `--warmup-frac` · `--lr-min-ratio` · `--clip-norm` | cosine · 0.1 · 0.1 · 1.0 |
| precision | `--precision` `fp32\|bf16` | fp32 |

**`--offset` is the one flag above that changes what the model is asked to do.** With it on, the
mean carries a hardwired `(d − depth_center)` term, so the depth response is a closed-form thinning
identity rather than something learned; with it off the model must learn depth from the covariate.
Both arms are first-class and neither dominates — see `AGENTS.md` §7.2 before quoting either.
`--reference` also changes the objective — it trains on the deviation from a per-assay average
rather than on raw signal — but the table it needs is loaded outside the model, so it is absent
from `config.arch` and must be passed identically at train and eval time. `EVAL.md` owns it.
`--d-model 0` derives the transformer width from the panel, so **set it explicitly whenever
`num_assays != 8`** or capacity silently tracks how many assays you happened to include.

`--precision` is the one entry above that is **not** an architecture flag: it builds no module and
changes no `state_dict` key, so it is absent from `config.arch` and lands in the run config instead.
It buys **memory, not speed** — activations halve, master weights and optimizer state stay fp32 — so
a null wall-clock result is the expected result and not a bug. The metadata embedders, every FiLM
tap, `LaneNorm`, the NB head arithmetic, the loss, and the whole of `eval.py` are fenced back into
fp32 whatever it is set to; **evaluation is never autocast**, so every recorded number stays
comparable to every other. fp16 is deliberately not offered: the `log2_mu` ceiling below is 30, fp16
tops out at 2**15.99, and `p = n / (n + mu)` would launder the resulting `inf` into a finite,
plausible-looking probability rather than raise.

Two guards run before anything is built. `pool_size ** n_cnn_layers` must equal
`deconv_upsample ** n_deconv_layers`, or the decoder would not undo the encoder's downsampling. And
the DNA tower's large pool is **derived** as `isqrt(resolution)`, never chosen — the panel's
resolution and the tower geometry are the same fact.

Three things stay hardcoded on purpose: the `log2_mu` clamp `(-15, 30)`, the `mu` floor `1e-6`, and
the 4-row covariate contract. They are part of the objective, not settings.

`--heads` is the one flag that changes the **objective** rather than the architecture alone. `count`
is required and is the only head the evaluation scores; `signal` (Gaussian over the arcsinh
log-p-value track) and `peak` (Bernoulli over the peak calls) add terms to the loss with coefficient
1.0 and are supervised on full-depth targets only. A run naming either is therefore **not** a control
for a run that does not — it descends a different total loss and is then scored on the NB head alone.

`tests/test_flags.py` holds this surface to two claims: naming every flag at its default is
bit-identical to naming none of them, and flipping any flag off its default changes the model. The
second is the one that matters — an exposed, documented, inert flag has shipped here before.

## Provenance

The architecture here was selected by a nine-arm, two-seed ladder on the full 35-assay ENCODE
Imputation Challenge panel. Only the winning arm ships; the ladder's other rungs, including the
56.2 M-parameter predecessor it replaced, live in the frozen snapshot that produced those runs and
are not carried in this repository.
