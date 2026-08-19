"""CANDI_STORE — the immutable whole-genome corpus store.

Chromosomes are stored whole and contiguous, one `(n_bins, n_tracks)` dataset per chromosome,
one file per biosample per kind. Windows, context length, DSF factor, chromosome split and assay
column order are **not** in the store: they live in a regime file read at load time, so one store
serves every regime without a re-bake. The contract is `STORE.md`; the plan is `STORE_PLAN.md`.

This package lands **beside** the old bake (D21). `prep/bake.py` and `dataset.py` are untouched
and stay the training path until a real run has used the store.

Modules: `layout` (the contract), `writer` (t6), `manifest` (t6), `cli` (t6),
`genome` (t7), `reader` / `regime` / `dataset` (t8).
"""
from __future__ import annotations

from candi.store import layout
from candi.store.layout import (
    CONTROL_TRACK,
    DEFAULT_RESOLUTION,
    KINDS,
    PVAL_SCALE,
    SCHEMA_VERSION,
    StoreError,
    decode_pval,
    encode_pval,
    kind_path,
    manifest_path,
    n_bins_for,
    read_root_attrs,
)

__all__ = [
    "layout",
    "SCHEMA_VERSION",
    "DEFAULT_RESOLUTION",
    "KINDS",
    "CONTROL_TRACK",
    "PVAL_SCALE",
    "StoreError",
    "n_bins_for",
    "encode_pval",
    "decode_pval",
    "kind_path",
    "manifest_path",
    "read_root_attrs",
]
