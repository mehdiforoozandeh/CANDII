"""SP2c -- compare two score directories leaf by leaf (Amendment 10, pinned interface M).

    python tools/onion_eval/compare_scores.py <dir a> <dir b> [--rtol R] [--ignore <key> ...]

A score directory holds `scores.json` and `suffstats.npz` (score.py). The tool walks:
  - every leaf of `scores.json`: a dict key adds `.<key>` to the path, a list item `[i]`. A number,
    a bool or null is compared by value; a string leaf is compared only for presence (shas, paths
    and labels are not scores). A leaf path on one side only is a difference.
  - every array of `suffstats.npz`: a name on one side only, another shape or another dtype is a
    difference; else the values are compared (NaN equals NaN; a non-numeric array must be equal).
Two numbers a, b are equal when |a - b| <= rtol * max(|a|, |b|) (`--rtol`, default 0: exact).
`--ignore <key>` (repeatable) drops every leaf or array whose path has `<key>` as one component,
or starts with `<key>` followed by `.` or `[` (e.g. `wall_seconds`, `timing_seconds`,
`scorer_settings`, `refs`, `scorer_version`, `scorer_settings.device`).
A file present in neither directory is skipped; in one only, it is a difference. Prints
`differences: <n>` and then one line per difference. Exit 0 when n is 0, 1 when n > 0, 2 when
neither directory holds a scores.json or suffstats.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SCORES = "scores.json"
SUFF = "suffstats.npz"
_COMPONENT = re.compile(r"[^.\[\]]+")


def leaves(x, path: str = "") -> dict:
    """{path: leaf} of a JSON value."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(leaves(v, f"{path}.{k}" if path else str(k)))
        return out
    if isinstance(x, list):
        out = {}
        for i, v in enumerate(x):
            out.update(leaves(v, f"{path}[{i}]"))
        return out
    return {path: x}


def ignored(path: str, ignore) -> bool:
    comps = set(_COMPONENT.findall(path))
    return any(k in comps or path == k or path.startswith(k + ".") or path.startswith(k + "[")
               for k in ignore)


def _is_num(v) -> bool:
    return v is None or isinstance(v, (bool, int, float))


def num_equal(a, b, rtol: float) -> bool:
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
        return a == b and type(a) is type(b)
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def compare_json(a: dict, b: dict, rtol: float = 0.0, ignore=()) -> list[str]:
    la = {k: v for k, v in leaves(a).items() if not ignored(k, ignore)}
    lb = {k: v for k, v in leaves(b).items() if not ignored(k, ignore)}
    out = []
    for k in sorted(set(la) | set(lb)):
        if k not in lb:
            out.append(f"{SCORES} {k}: only in a ({la[k]!r})")
        elif k not in la:
            out.append(f"{SCORES} {k}: only in b ({lb[k]!r})")
        else:
            va, vb = la[k], lb[k]
            if isinstance(va, str) and isinstance(vb, str):
                continue
            if not (_is_num(va) and _is_num(vb)):
                out.append(f"{SCORES} {k}: type {type(va).__name__} vs {type(vb).__name__}")
            elif not num_equal(va, vb, rtol):
                out.append(f"{SCORES} {k}: {va!r} vs {vb!r}")
    return out


def compare_arrays(a, b, rtol: float = 0.0, ignore=()) -> list[str]:
    """a, b: {name: ndarray}."""
    out = []
    for k in sorted(set(a) | set(b)):
        if ignored(k, ignore):
            continue
        if k not in b:
            out.append(f"{SUFF} {k}: only in a (shape {a[k].shape})")
            continue
        if k not in a:
            out.append(f"{SUFF} {k}: only in b (shape {b[k].shape})")
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out.append(f"{SUFF} {k}: shape {x.shape} vs {y.shape}")
            continue
        if x.dtype != y.dtype:
            out.append(f"{SUFF} {k}: dtype {x.dtype} vs {y.dtype}")
            continue
        if x.dtype.kind in "biufc":
            xf, yf = x.astype(np.complex128 if x.dtype.kind == "c" else np.float64), \
                y.astype(np.complex128 if y.dtype.kind == "c" else np.float64)
            both_nan = np.isnan(xf) & np.isnan(yf)
            same = (xf == yf) | both_nan
            if rtol > 0:
                with np.errstate(invalid="ignore"):
                    same |= np.abs(xf - yf) <= rtol * np.maximum(np.abs(xf), np.abs(yf))
            n = int((~same).sum())
            if n:
                d = np.abs(xf - yf)[~same]
                den = np.maximum(np.abs(xf), np.abs(yf))[~same]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.nanmax(np.where(den > 0, d / den, np.inf)) if d.size else 0.0
                out.append(f"{SUFF} {k}: {n} of {x.size} values differ (max relative {rel:.3g})")
        elif not np.array_equal(x, y):
            out.append(f"{SUFF} {k}: values differ")
    return out


def _npz(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def compare_dirs(a, b, rtol: float = 0.0, ignore=()) -> list[str]:
    a, b = Path(a), Path(b)
    present = [f for f in (SCORES, SUFF) if (a / f).is_file() or (b / f).is_file()]
    if not present:
        raise FileNotFoundError(f"neither {a} nor {b} holds {SCORES} or {SUFF}")
    out = []
    for f in present:
        if not (a / f).is_file():
            out.append(f"{f}: only in b")
        elif not (b / f).is_file():
            out.append(f"{f}: only in a")
        elif f == SCORES:
            out += compare_json(json.loads((a / f).read_text(encoding="utf-8")),
                                json.loads((b / f).read_text(encoding="utf-8")), rtol, ignore)
        else:
            out += compare_arrays(_npz(a / f), _npz(b / f), rtol, ignore)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--ignore", action="append", default=[])
    a = ap.parse_args(argv)
    if a.rtol < 0:
        ap.error("--rtol must be >= 0")
    try:
        diffs = compare_dirs(a.a, a.b, a.rtol, a.ignore)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"differences: {len(diffs)}")
    for d in diffs:
        print(d)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
