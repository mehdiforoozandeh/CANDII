"""candi.bench — the CANDII evaluation suite.

Five blocks under one CLI. It replaced `candi.eval`, which is now deleted (D15); `EVAL_PLAN.md`
is the build plan and `EVAL.md` is the contract.

- **E** the nine ENCODE Imputation Challenge measures, faithful to the organizers' code
- **P** the four post-hoc measures the challenge's own retrospective recommends instead
- **D** distributional: CRPS with its scale split, PIT/ECE, C-index, coverage, the marginal bar
- **B** binary/peak: AUPRC, peak overlap, correspondence curve. No AUROC, deliberately
- **C** covariate sensitivity: `covuse`, `covshare`, `depthdir`, `depthcounterfact`, `covspec`,
  and `depthblind` with the `biokeep` guard it is never reported without

Plus the **loss tier**, which is not a block: `harness.loss_block` scores the three NLLs the training
objective is built from (`distributional.nb_nll` / `gaussian_nll` / `bernoulli_nll`) on the eval
panel, so `val_loss` and `test_loss` are the same quantity as `train_loss`. They are pure numpy
copies because this package may not import `candi.train`, and the equivalence is pinned by test.

Everything is imported lazily, for the reason `candi.store.__init__` gives: `annotations` needs
only numpy, while the harness drags in h5py and the store. A consumer wanting a pinned bed should
not pay for either.
"""
# NO `from __future__ import annotations` in THIS file, and that is a fix rather than an omission.
# The future import binds the name `annotations` in the package namespace, so `from candi.bench
# import annotations` would resolve to the `_Feature` object and never reach the `__getattr__`
# below. Submodules may use it freely; the package `__init__` that lazily exports a submodule of
# that name may not.

__all__ = ["annotations", "binary", "cli", "covariate", "distributional", "eic", "harness",
           "partitions", "ranking"]


def __getattr__(name: str):
    if name in __all__:
        import importlib
        return importlib.import_module(f"candi.bench.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
