"""Import the vendored ENCODE scorer without touching a byte of it.

`encode_score_metrics_vendored.py` is upstream's `score_metrics.py`, unmodified — its git blob hash
is `5dcb343139a037e9a528e8e4ababf9cd00163ac1`, identical to the one GitHub serves. That property is
the whole value of the fixture, and `test_bench_reference.py` asserts it, so the file must not be
edited to make it importable.

It imports two things this repo does not have: `logger` (a module of the organizers' repo we did not
vendor) and `sklearn.metrics.roc_auc_score` (imported at the top and never used by any of the nine
measures — adding scikit-learn as a dependency to satisfy a dead import would be absurd). Both are
supplied here as stand-ins installed into `sys.modules` before the import runs, which leaves the
vendored bytes alone.

If either stub is ever *called*, it raises. A silent no-op would let a future upstream change start
depending on one without the test noticing.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

VENDORED = Path(__file__).resolve().parent / "encode_score_metrics_vendored.py"

# The upstream git blob hash. `git hash-object <file>` must still print this.
UPSTREAM_BLOB_SHA1 = "5dcb343139a037e9a528e8e4ababf9cd00163ac1"


def _stub_logger() -> types.ModuleType:
    mod = types.ModuleType("logger")

    class _Log:
        def __getattr__(self, name):
            def _refuse(*a, **k):
                raise AssertionError(
                    f"the vendored scorer called logger.{name}(). It did not when vendored, so "
                    f"upstream changed or a new code path was reached; the stub is deliberately "
                    f"loud rather than silent."
                )
            return _refuse

    mod.log = _Log()
    return mod


def _stub_sklearn() -> types.ModuleType:
    sk = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")

    def roc_auc_score(*a, **k):
        raise AssertionError(
            "the vendored scorer called roc_auc_score(). None of the nine measures uses it; the "
            "import is dead upstream. If this fires, upstream changed."
        )

    metrics.roc_auc_score = roc_auc_score
    sk.metrics = metrics
    return sk


def load_reference() -> types.ModuleType:
    """Import the vendored module with its two missing dependencies stubbed. Cached by sys.modules."""
    name = "encode_score_metrics_vendored"
    if name in sys.modules:
        return sys.modules[name]

    sys.modules.setdefault("logger", _stub_logger())
    sk = _stub_sklearn()
    sys.modules.setdefault("sklearn", sk)
    sys.modules.setdefault("sklearn.metrics", sk.metrics)

    spec = importlib.util.spec_from_file_location(name, VENDORED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
