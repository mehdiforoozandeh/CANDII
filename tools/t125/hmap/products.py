"""Per-product numbers h reads: depths, read length, fragment length, control, ratio V.

Source: `tools/t118/covariates.tsv` plus the t112 MANIFEST.tsv, never `covariates.json` (absent
locally and in synthetic products). The manifest is the check: its `depth`, `fraglen` and
`read_length` must equal the covariates', its `control_counts25_md5` is non-empty exactly when the
product has a control, and on the ratio arms its `knob_value` (MACS2's `--ratio`) must equal
V = k * N_T / N_C.

`control_read_length` is the track base's treatment read length: the control's own length is not
recorded (planner's assumption, approved by the PI 2026-10-11; the Nibi smoke reports the
formula-vs-real-P error that would reveal a problem).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from ladder import pairs as lpairs

#: the manifest knob of the ratio arms (`knob_value` = MACS2 --ratio)
RATIO_KNOB = "macs2_ratio"
RATIO_REL_TOL = 1e-8


@dataclass(frozen=True)
class ProductInfo:
    pid: str
    track: str
    assay: str
    mark_class: str
    arm: str
    level: str
    depth_reads: int            # N_T: all tagAlign lines
    read_length: int
    run_type_pe: int
    fraglen: int                # ChIP: FRAGLEN (extsize); DNase: the smooth window (150, 75, 300)
    has_control: int
    control_reads: int          # N_C; 0 without a control
    ratio_V: float | None       # k * N_T / N_C on the ratio arms, else None
    ctl_identity: str
    control_read_length: int    # = the track base's read_length (not recorded for the control)


def _read_tsv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _int(v: str) -> int:
    f = float(v)
    if f != int(f):
        raise ValueError(f"{v!r} is not an integer")
    return int(f)


def load(manifest_tsv, covariates_tsv) -> dict[str, ProductInfo]:
    """pid -> ProductInfo for every manifest row (130 on t112); raises on any inconsistency."""
    man = _read_tsv(manifest_tsv)
    cov = {r["pid"]: r for r in _read_tsv(covariates_tsv)}
    base_rl = {r["track"]: _int(r["read_length"]) for r in cov.values()
               if r["arm"] == "base" and r["level"] == "base"}
    out: dict[str, ProductInfo] = {}
    for m in man:
        pid = m["pid"]
        if pid in out:
            raise ValueError(f"duplicate pid {pid} in {manifest_tsv}")
        if pid not in cov:
            raise KeyError(f"{pid}: no row in {covariates_tsv}")
        c = cov[pid]
        track, assay = c["track"], c["assay"]
        if (m["track"], m["assay"]) != (track, assay):
            raise ValueError(f"{pid}: manifest track/assay {m['track']}/{m['assay']} != "
                             f"covariates {track}/{assay}")
        if track not in base_rl:
            raise KeyError(f"{pid}: track {track} has no base__base row in the covariates")
        depth = _int(c["depth_reads"])
        rl = _int(c["read_length"])
        fraglen = _int(c["fraglen_bp"])
        for name, got, want in (("depth", depth, m["depth"]), ("read_length", rl, m["read_length"]),
                                ("fraglen", fraglen, m["fraglen"])):
            if got != _int(want):
                raise ValueError(f"{pid}: covariates {name} {got} != manifest {want}")
        has_ctl = _int(c["has_control"])
        if has_ctl != int(m["control_counts25_md5"] != ""):
            raise ValueError(f"{pid}: has_control {has_ctl} but manifest control_counts25_md5 "
                             f"{m['control_counts25_md5']!r}")
        if assay == "DNase-seq" and has_ctl:
            raise ValueError(f"{pid}: a DNase product with a control")
        n_c = _int(c["control_reads"]) if has_ctl else 0
        if has_ctl and n_c <= 0:
            raise ValueError(f"{pid}: has a control but control_reads = {c['control_reads']}")
        ratio_v = None
        if c["arm"] == "ratio":
            if not has_ctl:
                raise ValueError(f"{pid}: a ratio arm without a control")
            ratio_v = float(c["ratio_k"]) * depth / n_c
            if m.get("knob") == RATIO_KNOB and m.get("knob_value", ""):
                mv = float(m["knob_value"])
                if not math.isclose(ratio_v, mv, rel_tol=RATIO_REL_TOL):
                    raise ValueError(f"{pid}: V = k*N_T/N_C = {ratio_v} != manifest --ratio {mv}")
        out[pid] = ProductInfo(
            pid=pid, track=track, assay=assay, mark_class=lpairs.MARK_CLASS[assay],
            arm=c["arm"], level=c["level"], depth_reads=depth, read_length=rl,
            run_type_pe=_int(c["run_type_pe"]), fraglen=fraglen, has_control=has_ctl,
            control_reads=n_c, ratio_V=ratio_v, ctl_identity=c["ctl_identity"],
            control_read_length=base_rl[track])
    return out
