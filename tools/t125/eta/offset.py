"""t125 depth offset: eta_in = log(1 + X) - log d, log mu' = log d' + eta_out.

d is a depth in millions of reads (`D_UNIT`), read from the manifest column `depth`. The count
offset is `EPS = 1` (log1p). A zero count, and the zero pad outside a chromosome, maps to -log d.
"""
from __future__ import annotations

import math

import numpy as np

D_UNIT = 1e6
EPS = 1.0


def log_depth(depth_reads: float) -> float:
    """log(depth_reads / 1e6); ValueError for a non-positive depth."""
    d = float(depth_reads)
    if not d > 0:
        raise ValueError(f"depth must be positive, got {depth_reads!r}")
    return math.log(d / D_UNIT)


def depth_table(rows: list[dict]) -> dict[str, float]:
    """pid -> log d from the manifest column `depth`, for every row (not only the usable ones)."""
    out = {}
    for r in rows:
        if r["pid"] in out:
            raise ValueError(f"duplicate pid {r['pid']!r} in the manifest")
        out[r["pid"]] = log_depth(r["depth"])
    return out


def transform_eta(X: np.ndarray, log_d: float) -> np.ndarray:
    """eta_in = log1p(X) - log_d, computed in float64, returned as float32 (shape kept)."""
    return (np.log1p(np.asarray(X).astype(np.float64)) - float(log_d)).astype(np.float32)


def offset_loc(loc: np.ndarray, log_d_tgt: float) -> np.ndarray:
    """log mu' = loc + log d' (float64 in, float64 out)."""
    return np.asarray(loc, dtype=np.float64) + float(log_d_tgt)
