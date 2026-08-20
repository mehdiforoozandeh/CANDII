"""candi.bench — the CANDII evaluation suite.

Five blocks under one CLI, replacing `candi.eval`. `EVAL_PLAN.md` is the build plan; `EVAL.md`
becomes the contract at cutover (t22).

- **E** the nine ENCODE Imputation Challenge measures, faithful to the organizers' code
- **P** the four post-hoc measures the challenge's own retrospective recommends instead
- **D** distributional: CRPS with its scale split, PIT/ECE, C-index, coverage, the marginal bar
- **B** binary/peak: AUPRC, peak overlap, correspondence curve. No AUROC, deliberately
- **C** covariate sensitivity: use, share, direction, specificity, invariance, and the guard

Everything is imported lazily, for the reason `candi.store.__init__` gives: `annotations` needs
only numpy, while the harness drags in h5py and the store. A consumer wanting a pinned bed should
not pay for either.
"""
# NO `from __future__ import annotations` in THIS file, and that is a fix rather than an omission.
# The future import binds the name `annotations` in the package namespace, so `from candi.bench
# import annotations` would resolve to the `_Feature` object and never reach the `__getattr__`
# below. Submodules may use it freely; the package `__init__` that lazily exports a submodule of
# that name may not.

__all__ = ["annotations", "binary", "distributional", "eic", "partitions"]


def __getattr__(name: str):
    if name in __all__:
        import importlib
        return importlib.import_module(f"candi.bench.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
