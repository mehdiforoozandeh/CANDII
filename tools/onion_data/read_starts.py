"""t392 C3 -- stranded 25 bp read-start counts, the read-level gate, and the read-start store.

    python read_starts.py run --measurement <id> --measurements <eic_v1_2 measurement dirs> \\
        --work-root <prefix> --out <store root> [--reach 8]
    python read_starts.py verify --out <store root>/<id>

READ-START RULE. The 5' base of a tagAlign line is `start` on strand `+` and `end - 1` on strand
`-` (`recount.point_positions(start, end, strand, 0)`). Its start bin is that base `// 25`. The grid
of a chromosome is the shape of its dataset in the measurement's `counts.h5` (`len // 25` bins), for
the chromosomes of `bin25.MAIN_CHROMS`. A line whose start bin is not on the grid adds no start and
is counted in `n_off_grid`. Each line is one read, as in `bin25.counts_from_tagalign`, so a file
with 2 lines per fragment counts both lines.

OVERLAP RULE AND THE GATE (D9). `records.COUNT_RULE` with the edge handling of
`bin25.counts_from_tagalign`: bins `start // 25 .. end // 25` inclusive on a grid of `len // 25 + 1`,
a line whose first bin is past that grid dropped, the end bin clamped, the grid cut to `len // 25`.
In the same scan, each line's span is placed as a box of offsets `[lo, hi]` from its own start bin;
the boxes rebuild the overlap counts, and `gate` compares them with the stored `eic_v1_2` file on
every bin. A span that leaves `reach` bins of its start bin raises. `kernel_counts[s, reach + k]`
counts the kept bins at offset k that the strand-s lines kept on the grid cover; divided by the
strand's `n_reads_strand` it is the span kernel.

STORE. `<out>/<id>/starts.h5`, `starts_pr1.h5`, `starts_pr2.h5`; one `[n_bins, 2]` dataset per
chromosome (column 0 plus, column 1 minus), uint16 or uint32, chunks `(4096, 2)`, gzip 4 + shuffle.
`provenance.json` holds the read files, every output's sha256, per file the gate and the conversion
floor, `wall_s` and `peak_rss_kb`. A differing bin on any file gives exit 1, after the files are
written.

numpy, pandas and h5py only (no torch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS, TOOLS / "t112"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bin25 import MAIN_CHROMS  # noqa: E402
from records import COUNT_RULE  # noqa: E402
from onion_data.recount import point_positions  # noqa: E402

RES = 25
REACH = 8
SCHEMA = 1
CHUNKSIZE = 5_000_000
#: the prefix of the `eic_v1_2` read files that `--work-root` replaces (plan C2: the tree extracts
#: to `<work root>/onion/work/<experiment>/...`)
WORK_PREFIX = "/scratch/mforooz"
#: the prefix of the testbed read files that `--testbed-root` replaces
TESTBED_PREFIX = "/project/def-maxwl/mforooz/ONION/testbed/readsets"
#: store file -> (half key, reference overlap file)
FILES = {"starts.h5": ("counts", "counts.h5"),
         "starts_pr1.h5": ("pr1", "pr1.h5"),
         "starts_pr2.h5": ("pr2", "pr2.h5")}
HALVES = ("pr1", "pr2")


# --------------------------------------------------------------------------- scan

def scan(paths, sizes: dict, reach: int = REACH, chunksize: int = CHUNKSIZE) -> dict:
    """One pass over the read files: stranded starts, the rebuilt overlap and the span kernel.

    `sizes`: {chrom: length in bp}; the grid is `len // 25`. Returns `starts` {chrom: uint32
    [n_bins, 2]}, `overlap` {chrom: uint32 [n_bins]}, `kernel_counts` int64 [2, 2 reach + 1],
    `n_reads_strand` int64 [2], `n_lines`, `n_main`, `n_off_grid`.
    """
    reach = int(reach)
    width = 2 * reach + 1
    nb = {c: int(sizes[c]) // RES for c in sizes}
    starts = {c: np.zeros((nb[c], 2), np.uint32) for c in sizes}
    diff = {c: np.zeros(nb[c] + 2, np.int64) for c in sizes}
    kdiff = np.zeros((2, width + 1), np.int64)
    n_lines = n_main = n_off = 0
    for path in paths:
        reader = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 5],
                             names=["chrom", "start", "end", "strand"], chunksize=chunksize,
                             dtype={"chrom": str, "start": np.int64, "end": np.int64,
                                    "strand": str})
        for chunk in reader:
            n_lines += len(chunk)
            for chrom, g in chunk.groupby("chrom", sort=False):
                if chrom not in starts:
                    continue
                n_main += len(g)
                n_bins = nb[chrom]
                n_grid = n_bins + 1
                s = g["start"].to_numpy()
                e = g["end"].to_numpy()
                strand = g["strand"].to_numpy()
                sb = point_positions(s, e, strand, 0) // RES
                minus = strand.astype(str) == "-"
                # starts: one per line whose start bin is on the grid
                on = sb < n_bins
                n_off += int((~on).sum())
                for col, sel in ((0, on & ~minus), (1, on & minus)):
                    if sel.any():
                        starts[chrom][:, col] += np.bincount(
                            sb[sel], minlength=n_bins).astype(np.uint32)
                # overlap: the bin25 edge rule, each box placed from its own start bin
                b0 = s // RES
                keep = b0 < n_grid
                b1 = np.minimum(e // RES, n_grid - 1)
                lo = b0 - sb
                hi = b1 - sb
                bad = keep & ((lo < -reach) | (hi > reach) | (lo > reach) | (hi < -reach))
                if bad.any():
                    i = int(np.flatnonzero(bad)[0])
                    raise ValueError(f"{path}: read {chrom}:{s[i]}-{e[i]} spans bins {lo[i]}..{hi[i]} "
                                     f"from its start bin, outside reach {reach}")
                d = diff[chrom]
                d += np.bincount((sb + lo)[keep], minlength=d.shape[0])
                d -= np.bincount((sb + hi + 1)[keep], minlength=d.shape[0])
                # span kernel: kept bins covered by lines kept on the grid
                kk = keep & on
                hik = np.minimum(hi, n_bins - 1 - sb)
                for col, sel in ((0, kk & ~minus), (1, kk & minus)):
                    if sel.any():
                        kdiff[col] += np.bincount(lo[sel] + reach, minlength=width + 1)
                        kdiff[col] -= np.bincount(hik[sel] + reach + 1, minlength=width + 1)
    overlap = {}
    for c in sizes:
        v = np.cumsum(diff[c][: nb[c] + 1])[: nb[c]]
        if v.size and int(v.max()) > np.iinfo(np.uint32).max:
            raise ValueError(f"{c}: bin count {int(v.max())} overflows uint32")
        overlap[c] = v.astype(np.uint32)
        del diff[c]
    kernel_counts = np.cumsum(kdiff, axis=1)[:, :width]
    n_reads_strand = np.array([sum(int(starts[c][:, k].sum()) for c in starts) for k in (0, 1)],
                              np.int64)
    return {"starts": starts, "overlap": overlap, "kernel_counts": kernel_counts,
            "n_reads_strand": n_reads_strand, "n_lines": n_lines, "n_main": n_main,
            "n_off_grid": n_off}


def kernel_of(kernel_counts, n_reads_strand) -> np.ndarray:
    """The span kernel: each strand's `kernel_counts` row over its `n_reads_strand` (0 if none)."""
    kc = np.asarray(kernel_counts, np.float64)
    n = np.asarray(n_reads_strand, np.float64)[:, None]
    return np.divide(kc, n, out=np.zeros_like(kc), where=n > 0)


# --------------------------------------------------------------------------- gate and floor

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gate(overlap: dict, reference_h5) -> dict:
    """The read-level gate (D9): the rebuilt overlap against one `eic_v1_2` file, every bin.

    A chromosome on one side only, or of another length, counts all its bins as differing.
    """
    n_bins = n_diff = 0
    with h5py.File(reference_h5, "r") as f:
        chroms = list(f.keys()) + [c for c in overlap if c not in f]
        for c in chroms:
            ref = np.asarray(f[c], np.int64) if c in f else None
            got = np.asarray(overlap[c], np.int64) if c in overlap else None
            if ref is None or got is None or ref.shape != got.shape:
                n = max(0 if ref is None else ref.size, 0 if got is None else got.size)
                n_bins += n
                n_diff += n
                continue
            n_bins += ref.size
            n_diff += int(np.count_nonzero(ref != got))
    return {"reference": str(reference_h5), "reference_sha256": sha256_file(reference_h5),
            "n_bins": int(n_bins), "n_differing": int(n_diff),
            "result": "pass" if n_diff == 0 else "fail"}


def conversion_floor(starts: dict, kernel, overlap: dict) -> float:
    """nMSE = mean((a - b)^2) / var(b) over all bins: a = starts convolved with the span kernel
    (zero outside the chromosome), b = the overlap counts."""
    kernel = np.asarray(kernel, np.float64)
    reach = (kernel.shape[1] - 1) // 2
    n = 0
    sb = 0.0
    for c in overlap:
        b = np.asarray(overlap[c], np.float64)
        n += b.size
        sb += float(b.sum())
    if n == 0:
        return float("nan")
    mean_b = sb / n
    sse = ssb = 0.0
    for c in overlap:
        b = np.asarray(overlap[c], np.float64)
        st = np.asarray(starts[c], np.float64)
        m = b.size
        a = np.zeros(m)
        for s in range(2):
            for k in range(-reach, reach + 1):
                w = kernel[s, reach + k]
                if w == 0 or abs(k) >= m:
                    continue
                if k >= 0:
                    a[k:] += w * st[: m - k, s]
                else:
                    a[: m + k] += w * st[-k:, s]
        sse += float(np.sum((a - b) ** 2))
        ssb += float(np.sum((b - mean_b) ** 2))
    return float(sse / ssb) if ssb > 0 else float("nan")


# --------------------------------------------------------------------------- store

def write_starts(path: Path, starts: dict, attrs: dict) -> None:
    """One `[n_bins, 2]` dataset per chromosome; uint16, or uint32 when a value exceeds 65,535."""
    big = any(v.size and int(v.max()) > np.iinfo(np.uint16).max for v in starts.values())
    dtype = np.uint32 if big else np.uint16
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for c, v in starts.items():
            f.create_dataset(c, data=np.asarray(v).astype(dtype), chunks=(min(4096, len(v)), 2),
                             compression="gzip", compression_opts=4, shuffle=True)
        for k, val in attrs.items():
            f.attrs[k] = val
    os.replace(tmp, path)


def read_strands(path, chrom: str, start: int, n: int) -> np.ndarray:
    """Bins `start .. start + n` of one chromosome as int32 `[n, 2]`; zeros past the grid end."""
    out = np.zeros((int(n), 2), np.int32)
    with h5py.File(path, "r") as f:
        v = f[chrom][int(start): int(start) + int(n)]
    out[: len(v)] = v
    return out


def _swap(p: str, prefix: str, root: str | None) -> str:
    if root and (p == prefix or p.startswith(prefix + "/")):
        return str(root).rstrip("/") + p[len(prefix):]
    return p


def halves_paths(prov: dict, work_root: str | None, testbed_root: str | None = None) -> dict:
    """`provenance.json::halves` as {half: [read files]}; both the list form and the nested
    `{"tagalign": [...]}` form. A path under WORK_PREFIX moves to `work_root`, one under
    TESTBED_PREFIX to `testbed_root`; others stay."""
    out = {}
    for h in HALVES:
        v = prov["halves"][h]
        if isinstance(v, dict):
            v = v["tagalign"]
        if isinstance(v, str):
            v = [v]
        paths = []
        for p in v:
            q = _swap(str(p), TESTBED_PREFIX, testbed_root)       # at most one swap per path
            paths.append(q if q != str(p) else _swap(q, WORK_PREFIX, work_root))
        out[h] = paths
    return out


def grid_sizes(reference_h5) -> dict:
    """{chrom: n_bins * 25} from the reference's dataset shapes, MAIN_CHROMS order."""
    with h5py.File(reference_h5, "r") as f:
        return {c: int(f[c].shape[0]) * RES for c in MAIN_CHROMS if c in f}


def _peak_rss_kb() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r // 1024) if sys.platform == "darwin" else int(r)


def _depth(path) -> float | None:
    with h5py.File(path, "r") as f:
        v = f.attrs.get("depth_reads")
    return None if v is None else float(v)


def run(measurement: str, measurements, work_root, out, reach: int = REACH,
        testbed_root=None) -> int:
    t0 = time.time()
    mdir = Path(measurements) / measurement
    prov_in = json.loads((mdir / "provenance.json").read_text())
    reads = halves_paths(prov_in, work_root, testbed_root)
    sizes = grid_sizes(mdir / "counts.h5")
    odir = Path(out) / measurement
    odir.mkdir(parents=True, exist_ok=True)

    res = {h: scan(reads[h], sizes, reach=reach) for h in HALVES}
    res["counts"] = {
        "starts": {c: res["pr1"]["starts"][c] + res["pr2"]["starts"][c] for c in sizes},
        "overlap": {c: res["pr1"]["overlap"][c] + res["pr2"]["overlap"][c] for c in sizes},
        **{k: res["pr1"][k] + res["pr2"][k]
           for k in ("kernel_counts", "n_reads_strand", "n_lines", "n_main", "n_off_grid")}}

    outputs = {}
    n_diff = 0
    for name, (which, ref) in FILES.items():
        r = res[which]
        kernel = kernel_of(r["kernel_counts"], r["n_reads_strand"])
        floor = conversion_floor(r["starts"], kernel, r["overlap"])
        g = gate(r["overlap"], mdir / ref)
        n_diff += g["n_differing"]
        depth = _depth(mdir / ref)
        attrs = {"schema": SCHEMA, "counting_rule": "read_start", "resolution": RES,
                 "strand_columns": "plus,minus", "measurement_id": measurement,
                 "depth_reads": -1.0 if depth is None else depth,
                 "n_reads_strand": np.asarray(r["n_reads_strand"], np.int64),
                 "n_off_grid": int(r["n_off_grid"]),
                 "kernel_counts": np.asarray(r["kernel_counts"], np.int64),
                 "kernel": kernel, "conversion_floor_nmse": floor}
        write_starts(odir / name, r["starts"], attrs)
        outputs[name] = {"sha256": sha256_file(odir / name), "gate": g,
                         "conversion_floor_nmse": floor,
                         "n_lines": int(r["n_lines"]), "n_main": int(r["n_main"]),
                         "n_off_grid": int(r["n_off_grid"]),
                         "n_reads_strand": [int(x) for x in r["n_reads_strand"]],
                         "kernel_counts": np.asarray(r["kernel_counts"]).tolist()}
    prov = {"schema": SCHEMA, "measurement_id": measurement, "measurements": str(measurements),
            "source_provenance": str(mdir / "provenance.json"),
            "lines_per_fragment": prov_in.get("lines_per_fragment"),
            "read_files": reads, "reach": int(reach), "resolution": RES,
            "count_rule": COUNT_RULE, "outputs": outputs,
            "gate_passed": n_diff == 0, "wall_s": round(time.time() - t0, 1),
            "peak_rss_kb": _peak_rss_kb(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = odir / "provenance.json.tmp"
    tmp.write_text(json.dumps(prov, indent=1) + "\n")
    os.replace(tmp, odir / "provenance.json")
    print(f"[read_starts] {measurement} wall_s={prov['wall_s']} peak_rss_kb={prov['peak_rss_kb']} "
          + " ".join(f"{n}:n_differing={o['gate']['n_differing']},floor={o['conversion_floor_nmse']:.6g}"
                     for n, o in outputs.items()), flush=True)
    return 0 if n_diff == 0 else 1


def verify(odir) -> int:
    """Recheck each output's sha256, `starts = pr1 + pr2` on every chromosome, and the gate."""
    odir = Path(odir)
    probs = []
    try:
        prov = json.loads((odir / "provenance.json").read_text())
    except (OSError, ValueError) as e:
        print(f"[verify] {odir}: no readable provenance.json ({e})")
        return 1
    outs = prov.get("outputs", {})
    for name in FILES:
        o = outs.get(name)
        if o is None or not (odir / name).is_file():
            probs.append(f"{name}: missing")
            continue
        if sha256_file(odir / name) != o.get("sha256"):
            probs.append(f"{name}: sha256 differs from provenance")
        g = o.get("gate", {})
        if g.get("n_differing") != 0 or g.get("result") != "pass":
            probs.append(f"{name}: recorded gate {g.get('result')} n_differing={g.get('n_differing')}")
    if all((odir / n).is_file() for n in FILES):
        with h5py.File(odir / "starts.h5", "r") as fc, h5py.File(odir / "starts_pr1.h5", "r") as f1, \
                h5py.File(odir / "starts_pr2.h5", "r") as f2:
            if set(fc) != set(f1) or set(fc) != set(f2):
                probs.append("chromosome sets differ between the three files")
            for c in fc:
                if c in f1 and c in f2:
                    s = np.asarray(fc[c], np.int64)
                    h = np.asarray(f1[c], np.int64) + np.asarray(f2[c], np.int64)
                    if s.shape != h.shape or not np.array_equal(s, h):
                        probs.append(f"{c}: starts != pr1 + pr2")
    for p in probs:
        print(f"[verify] {odir.name}: {p}")
    print(f"[verify] {odir.name}: {'OK' if not probs else 'FAIL'}")
    return 0 if not probs else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--measurement", required=True)
    r.add_argument("--measurements", required=True, help="eic_v1_2 measurement dirs")
    r.add_argument("--work-root", default=None,
                   help=f"replaces the prefix {WORK_PREFIX} of the read files; others unchanged")
    r.add_argument("--testbed-root", default=None,
                   help=f"replaces the prefix {TESTBED_PREFIX} of the read files; others unchanged")
    r.add_argument("--out", required=True, help="store root")
    r.add_argument("--reach", type=int, default=REACH)
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True, help="<store root>/<measurement id>")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.measurement, a.measurements, a.work_root, a.out, reach=a.reach,
                   testbed_root=a.testbed_root)
    return verify(a.out)


if __name__ == "__main__":
    sys.exit(main())
