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
tests/           214 tests, no GPU required
```

## Running it

```bash
pip install -e .
python -m candi.train --h5 <panel.h5> --out-dir <runs> --epochs 10 --steps-per-epoch 2000
python -m candi.eval  --h5 <panel.h5> --ckpt <run.ckpt>
pytest tests/ -q
```

## Provenance

The architecture here was selected by a nine-arm, two-seed ladder on the full 35-assay ENCODE
Imputation Challenge panel. Only the winning arm ships; the ladder's other rungs, including the
56.2 M-parameter predecessor it replaced, live in the frozen snapshot that produced those runs and
are not carried in this repository.
