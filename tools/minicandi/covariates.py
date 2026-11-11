"""miniCANDI covariates: the row encoder (tables -> raw vectors) and the network embedding.

`covariates.json` (one per tables directory; pinned interface B, layout defined here):

  {"schema": 1,
   "meta_columns": [the 11 META_COLUMNS, in order],      # checked when present
   "qc_columns":   [the 7 QC_COLUMNS, in order],          # checked when present
   "vocab": {"pipeline": [...], "pipeline_version": [...], "lab": [...], "platform": [...],
             "antibody": [...], "donor": [...]}}

A categorical value's id is 1 + its index in the vocabulary; an unknown or empty value has id 0.

Vectors hold RAW values: log2_depth = log2(depth_reads x depth_scale), 0/1 flags, read length in
bp, MAPQ threshold, categorical ids as floats. An empty numeric cell is written as the column's
fixed centre (so it standardises to 0). `CovariateEmbedding` standardises the numeric columns with
fixed constants (no fitted state, so a checkpoint never depends on a fit) and embeds each
categorical column through its own `nn.Embedding`.

D6: read-derived QC (`QC_COLUMNS`) goes into C only when `qc_on_input` is set, never into C'.
`c_out` therefore has exactly K_META columns, always.

Standardisation follows `tools/t118/ladder/encoding.py` in spirit (z-scores, unknown -> neutral);
its column set is the testbed's, not interface B's, so it is not imported.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi.contract import K_META, K_QC, META_COLUMNS, QC_COLUMNS  # noqa: E402

#: categorical column -> the measurements.tsv field it reads
CATEGORICAL = {"pipeline_id": "pipeline", "pipeline_version_id": "pipeline_version",
               "lab_id": "lab", "platform_id": "platform", "antibody_id": "antibody",
               "donor_id": "donor"}
#: fixed (centre, scale) per numeric column; centre is also the value of an empty cell
NUMERIC = {"log2_depth": (24.0, 2.0), "run_type_pe": (0.5, 0.5), "read_length": (75.0, 50.0),
           "dedup_on": (0.5, 0.5), "mapq_threshold": (20.0, 15.0),
           "nsc": (1.1, 0.1), "rsc": (1.0, 0.5), "nrf": (0.8, 0.2), "pbc1": (0.8, 0.2),
           "pbc2": (5.0, 5.0), "frip": (0.2, 0.15), "frag_len": (200.0, 100.0)}
#: rows in every categorical embedding table; ids at or above it map to 0 (unknown)
CAT_TABLE = 1024
_TRUE = {"1", "true", "yes", "on", "t", "y"}
_PE = {"paired-ended", "paired-end", "paired", "pe", "1"}


def load_covariates_json(path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if "meta_columns" in spec and tuple(spec["meta_columns"]) != META_COLUMNS:
        raise ValueError(f"{path}: meta_columns {spec['meta_columns']} != {list(META_COLUMNS)}")
    if "qc_columns" in spec and tuple(spec["qc_columns"]) != QC_COLUMNS:
        raise ValueError(f"{path}: qc_columns {spec['qc_columns']} != {list(QC_COLUMNS)}")
    spec.setdefault("vocab", {})
    return spec


def _num(v, col: str) -> float:
    if v is None or str(v).strip() == "":
        return NUMERIC[col][0]
    return float(v)


class CovariateEncoder:
    """measurements.tsv row (dict of str) -> raw float32 vectors for C and C'."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.ids = {field: {v: i + 1 for i, v in enumerate(spec.get("vocab", {}).get(field, []))}
                    for field in CATEGORICAL.values()}

    def cat_id(self, field: str, value) -> int:
        i = self.ids.get(field, {}).get("" if value is None else str(value), 0)
        return i if i < CAT_TABLE else 0

    def meta(self, row: dict, depth_scale: float = 1.0) -> np.ndarray:
        """float32[K_META]. `depth_scale` rescales the depth (0.5 for a pseudoreplicate half)."""
        d = row.get("depth_reads", "")
        if str(d).strip() == "":
            log2_depth = NUMERIC["log2_depth"][0]
        else:
            log2_depth = math.log2(max(float(d), 1.0) * float(depth_scale))
        v = [log2_depth,
             1.0 if str(row.get("run_type", "")).strip().lower() in _PE else 0.0,
             _num(row.get("read_length"), "read_length"),
             1.0 if str(row.get("dedup", "")).strip().lower() in _TRUE else 0.0,
             _num(row.get("mapq_threshold"), "mapq_threshold")]
        v += [float(self.cat_id(field, row.get(field))) for field in CATEGORICAL.values()]
        out = np.asarray(v, dtype=np.float32)
        assert out.shape == (K_META,)
        return out

    def qc(self, row: dict) -> np.ndarray:
        return np.asarray([_num(row.get(c), c) for c in QC_COLUMNS], dtype=np.float32)

    def c_in(self, row: dict, qc_on_input: bool = False, depth_scale: float = 1.0) -> np.ndarray:
        m = self.meta(row, depth_scale)
        return np.concatenate([m, self.qc(row)]) if qc_on_input else m

    def c_out(self, row: dict, depth_scale: float = 1.0) -> np.ndarray:
        """C' — metadata only, by construction (D6)."""
        return self.meta(row, depth_scale)


class CovariateEmbedding(nn.Module):
    """raw covariates [..., K] (K = 11 or 18) -> [..., d_out].

    Numeric columns are standardised with the fixed `NUMERIC` constants; each categorical column
    has its own `nn.Embedding(CAT_TABLE, d_cat)`; the concatenation goes through a 2-layer MLP.
    `scramble` (the scrambled-covariates control) is applied by the caller, not here.
    """

    def __init__(self, k_in: int, d_out: int, d_cat: int = 8, hidden: int = 64):
        super().__init__()
        if k_in not in (K_META, K_META + K_QC):
            raise ValueError(f"k_in {k_in}; want {K_META} or {K_META + K_QC}")
        cols = list(META_COLUMNS) + (list(QC_COLUMNS) if k_in > K_META else [])
        self.k_in = k_in
        self.num_idx = [i for i, c in enumerate(cols) if c in NUMERIC]
        self.cat_idx = [i for i, c in enumerate(cols) if c in CATEGORICAL]
        self.register_buffer("centre", torch.tensor([NUMERIC[cols[i]][0] for i in self.num_idx]))
        self.register_buffer("scale", torch.tensor([NUMERIC[cols[i]][1] for i in self.num_idx]))
        self.tables = nn.ModuleList(nn.Embedding(CAT_TABLE, d_cat) for _ in self.cat_idx)
        n_feat = len(self.num_idx) + d_cat * len(self.cat_idx)
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.shape[-1] != self.k_in:
            raise ValueError(f"covariates have K = {c.shape[-1]}, embedding built for {self.k_in}")
        c = c.float()
        num = (c[..., self.num_idx] - self.centre) / self.scale
        feats = [num]
        for j, i in enumerate(self.cat_idx):
            ids = c[..., i].round().long()
            ids = torch.where((ids >= 0) & (ids < CAT_TABLE), ids, torch.zeros_like(ids))
            feats.append(self.tables[j](ids))
        return self.mlp(torch.cat(feats, dim=-1))


def scramble_covariates(c: torch.Tensor, avail: torch.Tensor, generator=None) -> torch.Tensor:
    """The scrambled-covariates control: permute covariate rows across the windows of a batch.

    c [B, A, K], avail [B, A]. Each track's rows are permuted among the windows where that track
    is present, so a covariate vector stays a real one but no longer belongs to its counts.
    """
    out = c.clone()
    for a in range(c.shape[1]):
        idx = torch.nonzero(avail[:, a], as_tuple=False).flatten()
        if idx.numel() > 1:
            perm = idx[torch.randperm(idx.numel(), generator=generator)]
            out[idx, a] = c[perm, a]
    return out
