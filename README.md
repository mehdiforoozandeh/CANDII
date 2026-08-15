# CANDI

**C**onfidence-**A**ware **N**eural **D**enoising **I**mputer — self-supervised epigenome imputation
and denoising from raw sequencing read counts plus experimental covariates.

CANDI consumes **raw counts**, not normalized signal, and emits **probability distributions** rather
than point estimates, so it handles batch effects directly, reports calibrated uncertainty, and does
zero-shot imputation and denoising on cell types it has never seen.

## The model

~2.35 M parameters. Per assay, per 25 bp bin, it outputs a Negative Binomial `(n̂, p̂)` over raw counts.

```
x_data  [B, 768, A+1]     counts: A assays + 1 ChIP control   (CLOZE = -2, MISSING = -1)
x_dna   [B, 4, 19200]     one-hot DNA over the same 30 kb
x_meta  [B, 4, A+1]       log2(depth), assay_id, read_length, run_type — INPUT tracks
y_meta  [B, 4, A]         the same four rows — TARGET tracks
   -> {p, n, eta, log2_mu, mu}   each [B, 768, A]
```

**Encoder.** A Conv1D tower grouped by track (`groups = A+1`), so each assay's channels stay
disjoint, with per-assay FiLM after every block; a dense Conv1D tower over DNA; a linear fusion; and
two RoPE transformer layers.

**Decoder.** A grouped deconv mirror of the signal tower at constant lane width, with a per-assay
FiLM tap after the input projection and after each deconv block, then a weight-shared head per assay.
The mean uses a depth-offset log link, so telling the model a different sequencing depth scales the
prediction rather than requiring it to relearn scale:

```
log2_mu = (d - depth_center) + eta        # d = log2 depth, sentinel-guarded
mu      = 2 ** clamp(log2_mu, -15, 30)
n       = softplus(raw_n) + 1e-6
p       = n / (n + mu)
```

## Two conventions that trip everyone up

- **`obs` and `imp`** in every loss and metric mean **unmasked** and **masked** positions. They do
  *not* mean biologically observed and imputed.
- The **control channel** is index `A` in `x_data` and is never masked.

## Layout

```
src/candi/
  model.py       the model, and the batch helpers train/eval/healthcheck share
  encoder.py     conv towers, metadata embedding, FiLM, mask token, transformer
  decoder.py     grouped deconv trunk, per-lane norm and FiLM, NB head
  dataset.py     the H5 reader        batch.py    masking and batch prep
  train.py       training + the run JSON      eval.py / metrics.py   the M1/M2/M3/S14 instrument
  prep/          bake an H5 from ENCODE tracks
tools/golden.py  bit-exactness gate — every change must clear it before the next one starts
tests/           the suite; no GPU required
```

## Running it

```bash
pip install -e .
python -m candi.train --h5 <panel.h5> --out-dir <runs> --epochs 10 --steps-per-epoch 2000
python -m candi.eval  --h5 <panel.h5> --ckpt <run.ckpt> --arch-from <run.json>
pytest tests/ -q
```

Always re-score with `--arch-from <run>.json`. Every architecture flag changes the `state_dict`, and
that file carries the exact arguments the checkpoint was built from — so nothing has to be retyped,
and nothing can be retyped wrong.

## What is tunable

Scale is **never** a flag: `num_assays`, `context_bins`, `resolution`, the assay order, the DSF
ladder and the chromosome split all come from the H5's own attributes. Everything below is a
deliberate choice, and every one of them defaults to the shipped model.

| group | flags | default |
|---|---|---|
| geometry | `--n-cnn-layers` `--conv-kernel-size` `--pool-size` `--expansion-factor` `--n-deconv-layers` `--deconv-upsample` `--deconv-kernel-size` | 3 · 3 · 2 · 2 · 3 · 2 · 3 |
| normalisation | `--conv-norm` `layer\|lane\|group\|batch\|instance` · `--deconv-norm` `lane\|group` · `--transformer-norm` `layer\|rmsnorm\|simple_rmsnorm\|scalenorm` · `--transformer-norm-placement` `pre\|post\|sandwich` · `--attn-qk-norm` | layer · lane · layer · pre · off |
| conditioning | `--film-taps`, a comma set over `pre_conv` `per_conv` `post_conv` `per_transformer` `pre_deconv` `per_deconv` `post_head` · `--film-init-encoder` · `--film-init-decoder` | `per_conv,pre_deconv,per_deconv` · xavier · zero |
| heads | `--head-sharing` `shared\|per_assay` · `--head-hidden` | shared · 0 (match the lane) |
| optimisation | `--lr-schedule` `cosine\|linear\|constant` · `--warmup-frac` · `--lr-min-ratio` · `--clip-norm` | cosine · 0.1 · 0.1 · 1.0 |
| precision | `--precision` `fp32\|bf16` | fp32 |

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

`tests/test_flags.py` holds this surface to two claims: naming every flag at its default is
bit-identical to naming none of them, and flipping any flag off its default changes the model. The
second is the one that matters — an exposed, documented, inert flag has shipped here before.

## Provenance

The architecture here was selected by a nine-arm, two-seed ladder on the full 35-assay ENCODE
Imputation Challenge panel. Only the winning arm ships; the ladder's other rungs, including the
56.2 M-parameter predecessor it replaced, live in the frozen snapshot that produced those runs and
are not carried in this repository.
