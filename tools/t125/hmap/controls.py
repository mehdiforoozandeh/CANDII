"""The control-count cache: each product's `control_counts25.npz` in the t118 ladder cache format.

`<cache_dir>/<pid>__counts.npy` + `<pid>__counts.json` exactly as `ladder.data.build_cache` writes
a counts space, so `ladder.data.Corpus(cache_dir).get(pid, "counts", chrom)` returns the CONTROL
counts of `pid` (uint32, chromosomes in MAIN_CHROMS order). The npz md5 is checked against the
manifest column `control_counts25_md5`; an empty column means the product has no control
("absent", nothing written). An existing json means "skipped". The json is written last.

    python tools/t125/hmap/controls.py cache <MANIFEST.tsv> <products_dir> <cache_dir> <index>

`index` runs over the manifest rows sorted by pid (0..129 on t112), as `ladder.train cache`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_T125 = Path(__file__).resolve().parents[1]
for _p in (_T125.parent / "t118", _T125):          # script mode: ladder and hmap importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data as ldata  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402

CONTROL_NPZ = "control_counts25.npz"
MD5_COLUMN = "control_counts25_md5"
SPACE = "counts"
DTYPE = np.dtype(np.uint32)


def _as_counts(a: np.ndarray, where: str) -> np.ndarray:
    if a.ndim != 1:
        raise ValueError(f"{where}: ndim {a.ndim}, need 1-D")
    if a.dtype == DTYPE:
        return a
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{where}: dtype {a.dtype} is not an integer count")
    if a.size and (a.min() < 0 or int(a.max()) > np.iinfo(DTYPE).max):
        raise ValueError(f"{where}: counts outside the uint32 range")
    return a.astype(DTYPE)


def build_control_cache(products_dir, cache_dir, pid: str, manifest_row: dict) -> dict:
    """Cache one product's control counts; returns `{"counts": "built" | "skipped" | "absent"}`."""
    if manifest_row["pid"] != pid:
        raise ValueError(f"manifest row is {manifest_row['pid']!r}, not {pid!r}")
    want = manifest_row.get(MD5_COLUMN, "")
    if not want:
        return {SPACE: "absent"}
    products_dir, cache_dir = Path(products_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    js = cache_dir / f"{pid}__{SPACE}.json"
    if js.is_file():
        return {SPACE: "skipped"}
    npz = products_dir / pid / CONTROL_NPZ
    md5 = ldata.md5_file(npz)
    if md5 != want:
        raise ValueError(f"{npz}: md5 {md5} != manifest {MD5_COLUMN} {want}")
    with np.load(npz, allow_pickle=False) as z:
        present = {f[:-4] if f.endswith(".npy") else f for f in z.files}
        chroms = [c for c in lpairs.MAIN_CHROMS if c in present]
        if not chroms:
            raise ValueError(f"{npz}: no main chromosome among {sorted(present)}")
        lens = {c: ldata._npz_member_len(z, c, npz) for c in chroms}
        offsets, n = {}, 0
        for c in chroms:
            offsets[c] = [n, n + lens[c]]
            n += lens[c]
        tmp = cache_dir / f"{pid}__{SPACE}.npy.tmp"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=DTYPE, shape=(n,))
        for c in chroms:
            s, e = offsets[c]
            mm[s:e] = _as_counts(z[c], f"{npz}:{c}")
        mm.flush()
        del mm
    os.replace(tmp, cache_dir / f"{pid}__{SPACE}.npy")
    meta = {"pid": pid, "space": SPACE, "dtype": DTYPE.name, "chroms": chroms,
            "offsets": offsets, "n_bins": n, "source_npz_md5": md5, "source": CONTROL_NPZ}
    jtmp = cache_dir / f"{pid}__{SPACE}.json.tmp"
    jtmp.write_text(json.dumps(meta, indent=1) + "\n")
    os.replace(jtmp, js)
    return {SPACE: "built"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache", help="cache one product's control counts (index over pid order)")
    for a in ("manifest", "products_dir", "cache_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    args = ap.parse_args(argv)
    rows = sorted(lpairs.read_manifest(args.manifest), key=lambda r: r["pid"])
    if not 0 <= args.index < len(rows):
        raise IndexError(f"cache index {args.index} outside 0..{len(rows) - 1}")
    row = rows[args.index]
    status = build_control_cache(args.products_dir, args.cache_dir, row["pid"], row)
    print(f"{row['pid']}: " + " ".join(f"{s}={v}" for s, v in status.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
