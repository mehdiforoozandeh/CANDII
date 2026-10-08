"""t118 ladder — the covariate encoder: one product's row of `tools/t118/covariates.tsv` -> float32[20].

**Vector** (pinned order, `FIELDS`):
  0 depth_log2 z          5 control_fraction z*     10-12 ctl_identity one-hot (matched, other, none)
  1 read_length z         6 ratio_k z*              13-19 assay one-hot (DNase-seq, H3K27ac,
  2 run_type_pe           7 control_depth_log2 z*         H3K27me3, H3K36me3, H3K4me1, H3K4me3,
  3 dedup_on              8 extsize_k z                   H3K9me3)
  4 mapq_thresh z         9 has_control
z = (v - mean) / std over the fit pids (population std; std < 1e-6 -> 0). z* = the same, but the
mean and std are taken over the fit pids with has_control = 1, and a product without a control
gets 0. Binary columns are 0/1 as tabled. g's input is concat(encode(src), encode(tgt)) = 40 —
never a difference.

The encoder keeps the raw values of every row it was fitted from (not only the fit pids), so a
checkpoint can encode any product of the table — law-test and shuffle pids included.
"""
from __future__ import annotations

import csv

import numpy as np

Z_COLS = ("depth_log2", "read_length", "mapq_thresh", "extsize_k")
ZSTAR_COLS = ("control_fraction", "ratio_k", "control_depth_log2")
BIN_COLS = ("run_type_pe", "dedup_on", "has_control")
CTL_CATS = ("matched", "other", "none")
ASSAYS = ("DNase-seq", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3")
FIELDS = (("depth_log2 z", "read_length z", "run_type_pe", "dedup_on", "mapq_thresh z",
           "control_fraction z*", "ratio_k z*", "control_depth_log2 z*", "extsize_k z",
           "has_control")
          + tuple(f"ctl_identity={c}" for c in CTL_CATS) + tuple(f"assay={a}" for a in ASSAYS))
N_COV = len(FIELDS)
assert N_COV == 20
STD_EPS = 1e-6
_NUM_COLS = Z_COLS + ZSTAR_COLS + BIN_COLS


def read_covariates(path) -> list[dict]:
    """covariates.tsv as a list of str dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _raw(row: dict) -> dict:
    out = {c: float(row[c]) for c in _NUM_COLS}
    for c in BIN_COLS:
        if out[c] not in (0.0, 1.0):
            raise ValueError(f"{row['pid']}: {c} = {row[c]!r} is not 0/1")
    if row["ctl_identity"] not in CTL_CATS:
        raise ValueError(f"{row['pid']}: ctl_identity {row['ctl_identity']!r} not in {CTL_CATS}")
    if row["assay"] not in ASSAYS:
        raise ValueError(f"{row['pid']}: assay {row['assay']!r} not in {ASSAYS}")
    out["ctl_identity"] = row["ctl_identity"]
    out["assay"] = row["assay"]
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std())


class Encoder:
    """`Encoder.fit(cov_rows, fit_pids)`, then `encode(pid) -> float32[20]`."""

    def __init__(self, rows: dict, norm: dict, fit_pids: list[str]):
        self.rows = rows            # pid -> raw values (floats + ctl_identity + assay)
        self.norm = norm            # column -> [mean, std]
        self.fit_pids = list(fit_pids)

    @classmethod
    def fit(cls, cov_rows: list[dict], fit_pids) -> "Encoder":
        rows = {}
        for r in cov_rows:
            if r["pid"] in rows:
                raise ValueError(f"duplicate pid {r['pid']!r} in the covariate table")
            rows[r["pid"]] = _raw(r)
        fit_pids = sorted(fit_pids)
        missing = [p for p in fit_pids if p not in rows]
        if missing:
            raise KeyError(f"fit pids missing from the covariate table: {missing}")
        fit = [rows[p] for p in fit_pids]
        norm = {c: list(_mean_std([r[c] for r in fit])) for c in Z_COLS}
        ctl = [r for r in fit if r["has_control"] == 1.0]
        norm.update({c: list(_mean_std([r[c] for r in ctl])) for c in ZSTAR_COLS})
        return cls(rows, norm, fit_pids)

    @staticmethod
    def _z(v: float, mean_std) -> float:
        mean, std = mean_std
        return 0.0 if std < STD_EPS else (v - mean) / std

    def encode(self, pid: str) -> np.ndarray:
        if pid not in self.rows:
            raise KeyError(f"pid {pid!r} not in the covariate table")
        r = self.rows[pid]
        has = r["has_control"] == 1.0
        zs = {c: (self._z(r[c], self.norm[c]) if has else 0.0) for c in ZSTAR_COLS}
        v = [self._z(r["depth_log2"], self.norm["depth_log2"]),
             self._z(r["read_length"], self.norm["read_length"]),
             r["run_type_pe"], r["dedup_on"],
             self._z(r["mapq_thresh"], self.norm["mapq_thresh"]),
             zs["control_fraction"], zs["ratio_k"], zs["control_depth_log2"],
             self._z(r["extsize_k"], self.norm["extsize_k"]),
             r["has_control"]]
        v += [1.0 if r["ctl_identity"] == c else 0.0 for c in CTL_CATS]
        v += [1.0 if r["assay"] == a else 0.0 for a in ASSAYS]
        return np.asarray(v, dtype=np.float32)

    def encode_pair(self, src_pid: str, tgt_pid: str) -> np.ndarray:
        """float32[40] = concat(encode(src), encode(tgt))."""
        return np.concatenate([self.encode(src_pid), self.encode(tgt_pid)])

    def state(self) -> dict:
        return {"fields": list(FIELDS), "fit_pids": list(self.fit_pids),
                "norm": {c: list(v) for c, v in self.norm.items()},
                "rows": {p: dict(r) for p, r in self.rows.items()}}

    @classmethod
    def from_state(cls, state: dict) -> "Encoder":
        if list(state["fields"]) != list(FIELDS):
            raise ValueError("encoder state has a different field order")
        return cls({p: dict(r) for p, r in state["rows"].items()},
                   {c: list(v) for c, v in state["norm"].items()}, state["fit_pids"])
