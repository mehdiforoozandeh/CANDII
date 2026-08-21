"""CANDI_STORE — the immutable whole-genome corpus store.

Chromosomes are stored whole and contiguous, one `(n_bins, n_tracks)` dataset per chromosome,
one file per biosample per kind. Windows, context length, DSF factor, chromosome split and assay
column order are **not** in the store: they live in a regime file read at load time, so one store
serves every regime without a re-bake. The contract is `STORE.md`; the plan is `STORE_PLAN.md`.

This package lands **beside** the old bake (D21). `prep/bake.py` and `dataset.py` are untouched
and stay the training path until a real run has used the store.

Modules: `layout` (the contract), `writer` (t6), `manifest` (t6), `cli` (t6),
`genome` (t7), `reader` / `regime` / `dataset` (t8).

**Everything but `layout` is imported lazily**, and that is load-bearing rather than tidy.
`layout` is the one module that defines the on-disk contract and it needs nothing but numpy;
eager exports would make `import candi.store` drag in h5py through `genome`/`writer`/`manifest`
and torch through `dataset`. So the names below all resolve, but only the module you actually
name gets imported.

Torch is the weaker half of that argument today: `candi/__init__.py` imports it unconditionally,
so it is already loaded before this package runs. h5py is not, and `layout` is the module a
consumer wanting only a path helper or the pval codec should be able to reach for free. Should
the parent ever stop importing torch eagerly, this file does not need revisiting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from candi.store import layout
from candi.store.layout import (
    CONTROL_TRACK,
    DEFAULT_RESOLUTION,
    KINDS,
    PVAL_SCALE,
    PVAL_TRANSFORM,
    SCHEMA_VERSION,
    StoreError,
    decode_pval,
    encode_pval,
    kind_path,
    manifest_path,
    n_bins_for,
    read_root_attrs,
)

# name -> the submodule that defines it. A submodule maps to itself.
_LAZY = {
    "writer": "writer",
    "manifest": "manifest",
    "genome": "genome",
    "reader": "reader",
    "regime": "regime",
    "dataset": "dataset",
    # genome (t7)
    "DEFAULT_MIN_VALID_FRAC": "genome",
    "DNA_CODES": "genome",
    "N_CODE": "genome",
    "GenomeLayer": "genome",
    "build_genome": "genome",
    "count_eligible": "genome",
    "eligible_starts": "genome",
    "eligible_window_mask": "genome",
    "genome_report": "genome",
    "load_genome_chrom_sizes": "genome",
    "verify_genome": "genome",
    "window_valid_counts": "genome",
    # reader / regime / dataset (t8)
    "CorpusStore": "reader",
    "BiosampleStore": "reader",
    "TrackView": "reader",
    "GenomeView": "reader",
    "Regime": "regime",
    "RegimeError": "regime",
    "DsfPolicy": "regime",
    "WindowPlan": "regime",
    "StoreDataset": "dataset",
    "thin_counts": "dataset",
    "stable_hash": "dataset",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from candi.store import dataset, genome, manifest, reader, regime, writer
    from candi.store.dataset import StoreDataset, stable_hash, thin_counts
    from candi.store.genome import (
        DEFAULT_MIN_VALID_FRAC,
        DNA_CODES,
        N_CODE,
        GenomeLayer,
        build_genome,
        count_eligible,
        eligible_starts,
        eligible_window_mask,
        genome_report,
        load_genome_chrom_sizes,
        verify_genome,
        window_valid_counts,
    )
    from candi.store.reader import BiosampleStore, CorpusStore, GenomeView, TrackView
    from candi.store.regime import DsfPolicy, Regime, RegimeError, WindowPlan


def __getattr__(name: str):
    """PEP 562 lazy export. See the module docstring for why this is not just neatness."""
    mod_name = _LAZY.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{mod_name}")
    return mod if name == mod_name else getattr(mod, name)


def __dir__():
    return sorted(__all__)


__all__ = [
    # eager — layout only, so this import stays free of h5py and torch
    "layout",
    "SCHEMA_VERSION",
    "DEFAULT_RESOLUTION",
    "KINDS",
    "CONTROL_TRACK",
    "PVAL_SCALE",
    "PVAL_TRANSFORM",
    "StoreError",
    "n_bins_for",
    "encode_pval",
    "decode_pval",
    "kind_path",
    "manifest_path",
    "read_root_attrs",
    # lazy — resolved through __getattr__
    *sorted(_LAZY),
]
