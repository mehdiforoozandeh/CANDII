"""Lavawizard / Guacamole, ported to PyTorch — `plan/RIVALS_PLAN.md` §7.4.

Upstream is `github.com/ccchang0111/ENCODE_imputation_2019` at commit `d638b204`, Keras 2.2.4 on
TensorFlow 1.15. `README.md` beside this file records the provenance, the model, and the three
upstream defects the port rules on rather than inherits.

**Nothing here is imported by `src/candi`** (`RIVALS_PLAN.md` decision E). The dependency runs one
way: this package imports `candi.bench.external` for the §4.1 directory-name rule, so the
prediction-track contract keeps a single definition.

Four modules, in the order the port uses them:

- `model` — the architecture, `Precamole` (stage 1) and `Guacamole` (stage 2).
- `keras_weights` — their `.h5` into a torch module. `h5py` only; no TensorFlow anywhere.
- `features` — the cross-cell average and variance, read from our store.
- `emit` — §4.1 prediction roots.
"""

__all__ = ["model", "keras_weights", "features", "emit"]
