"""t142 — whole-read thinning of a tagAlign into store-rule 25 bp counts (PI ruling 2026-10-30).

Why. The t137 floor and the t141 depth null thinned the 25 bp counts binomially, bin by bin. A read
adds 1 to every 25 bp bin it overlaps (4 to 5 bins for a 76-101 bp read), so per-bin thinning drops
the pieces of one read independently, and every multi-bin sum (1 kb shape, peak-height means) comes
out about sqrt(k) too smooth. Per-bin marginals are exact either way. Here the unit is the READ: a
tagAlign line is kept or dropped whole, and the kept lines are counted with the t117 rule.

The counting rule is the one t117 used to make every `pr{1,2}.counts25.npz`:
`tools/t112/bin25.py::counts_from_tagalign` (called by `tools/t117/pseudoreps.py::run_readset` on
`origin/data-acquisition/t117-pseudoreplicates`). Each line adds 1 to bins `start // 25 ..
end // 25` INCLUSIVE on a grid of `len // 25 + 1` bins; a line whose start bin is past the grid is
dropped and its end bin clipped to the grid; the grid is then truncated to `len // 25`; lines on a
chromosome outside the size table are read (and drawn for) but not counted; uint32 out, keys in
MAIN_CHROMS order (chr1..chr22, chrX). No fragment extension, no shift: the tagAlign coordinates
are used as they are. `count` reproduces bin25 bit for bit (tested, and checked on every product
in `product` mode against the t117 npz before anything is written).

The draw. One uniform per unit, `u < ratio` keeps it, in file order, from
`np.random.default_rng([seed, pid_index, depth_index])` (t137 floor.py's key). The unit is a line,
or with `--pairs` two consecutive lines (a paired-end tagAlign from ENCODE bam2ta / spr keeps the
two mates of a pair on consecutive lines; the `pe` arms). A uniform is drawn for every unit,
whatever its chromosome, so the draw depends only on line order; chunking does not change it.
Several tagAligns given to one rng are read one after the other: `--union` (the depth null: the base
read set's pr1 then pr2, which together are the base product's reads), and the floor ladder (pr1
then pr2 from the same rng object, floor.py's order).

    python readthin.py count <tagAlign> <out.npz> [--chrsz F | --like NPZ]
    python readthin.py thin  <tagAlign> <ratio> <seed> <pid_index> <depth_index> <out.npz>
                             [--union TA ...] [--pairs] [--chrsz F | --like NPZ]
    python readthin.py product <pseudoreps_dir> <products_dir> <out_dir> <pid_index>

`product` (what slurm/t142/readthin.sh runs) for the product at `pid_index` (its position in the
sorted pid list of <pseudoreps_dir>/MANIFEST.tsv, as t137 floor.py), into <out_dir>/<pid>/:
  floor_<r>_s<seed>_<half>.npz  the floor ladder, ratios 0.5 / 0.25 / 0.125 x seeds 0, 1 x pr1, pr2,
        key [seed, pid_index, depth_index] with depth_index = 1, 2, 3; chr19 + chr21 only, except
        r0.25 s0 (the t137 bins1k draw), which has every chromosome.
  null_s<seed>.npz  depth arms only: the track base's read set (pr1 then pr2) thinned to
        depth(arm) / depth(base), key [seed, arm_pid_index, 0] with arm_pid_index the arm's position
        in the sorted pid list of <products_dir>/MANIFEST.tsv (t141 arms.py's pid_index; depth_index
        0 is never drawn with by the floor ladder); every chromosome.
  <name>.json beside each npz (ratio, rng key, tagAligns, lines read and kept), product.json, DONE.
Every half's full count is checked bit-exact against <pseudoreps_dir>/<pid>/<half>/counts25.npz,
and the base read set's full count against <products_dir>/<base>/counts25.npz, before any output is
written (exit 3 otherwise). Writes are atomic (tmp, then rename); DONE is written last.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "t112"))
import bin25  # noqa: E402  (the t117 counting rule; numpy + pandas only)

RES = 25
CHUNK = 4_000_000                                  # lines per pandas chunk; even, for --pairs
#: the t137 floor ladder (floor.RATIOS[1:], their depth_index 1..3; floor.SEEDS; the bins1k draw)
FLOOR_RATIOS = ((1, 0.5), (2, 0.25), (3, 0.125))
SEEDS = (0, 1)
BINS1K = (0.25, 0)
FLOOR_CHROMS = ("chr19", "chr21")                  # knob_effects.CHROMS: what floor.py scores
NULL_DEPTH_INDEX = 0
DROP = ("chrM", "chrY")
CHRSZ = ("/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/"
         "GRCh38_EBV.chrom.sizes.tsv")             # bam_arm.CHRSZ, the size table t117 used
RATIO_LABEL = {0.5: "r0.5", 0.25: "r0.25", 0.125: "r0.125"}   # floor.RATIO_LABEL


def rng_for(seed: int, pid_index: int, depth_index: int):
    return np.random.default_rng([seed, pid_index, depth_index])


def floor_name(ratio: float, seed: int, half: str) -> str:
    return f"floor_{RATIO_LABEL[ratio]}_s{seed}_{half}.npz"


def sizes_like(npz_path) -> dict:
    """Chromosome sizes that give the same grid as an existing counts25 npz: len = n_bins * 25.

    The rule only uses `len // 25` (grid `len // 25 + 1`, truncated to `len // 25`), so this
    reproduces the original size table's output exactly."""
    with np.load(npz_path, allow_pickle=False) as z:
        return {c: int(z[c].size) * RES for c in z.files}


def sizes_from(chrsz=None, like=None) -> dict:
    if like is not None:
        return sizes_like(like)
    return bin25.load_chrsz(chrsz or CHRSZ)


class Acc:
    """One difference array over `chroms` (bin25's per-chromosome arrays, laid end to end), fed the
    lines a keep mask selects."""

    def __init__(self, sizes: dict, chroms, ratio=None, rng=None):
        self.sizes = sizes
        self.chroms = [c for c in sizes if c in set(chroms)]
        self.ratio, self.rng = ratio, rng
        self.off, o = {}, 0
        for c in self.chroms:
            self.off[c] = o
            o += sizes[c] // RES + 2
        self.d = np.zeros(o, dtype=np.int64)
        self.kept_lines = 0

    def keep(self, n: int, pairs: bool):
        if self.ratio is None:
            return None
        if pairs:
            if n % 2:
                raise ValueError("odd number of lines in a paired-end chunk")
            return np.repeat(self.rng.random(n // 2) < self.ratio, 2)
        return self.rng.random(n) < self.ratio

    def finish(self) -> dict:
        out = {}
        for c in self.chroms:
            nb = self.sizes[c] // RES
            v = np.cumsum(self.d[self.off[c]: self.off[c] + nb + 1])[:nb]
            if v.size and int(v.max()) > np.iinfo(np.uint32).max:
                raise SystemExit(f"{c}: bin count {int(v.max())} overflows uint32")
            out[c] = v.astype(np.uint32)
        return out


def feed(ta_path, accs, pairs: bool = False, chunksize: int = CHUNK) -> dict:
    """Stream one tagAlign into every Acc; returns {n_lines, n_lines_main}. Each Acc draws its own
    keep mask (one uniform per unit, every unit) from its own rng, so the draw is in line order."""
    if pairs and chunksize % 2:
        raise ValueError("--pairs needs an even chunksize")
    sizes = accs[0].sizes
    names = list(sizes)
    code_of = {c: i for i, c in enumerate(names)}
    nbins = np.array([sizes[c] // RES for c in names] + [0], dtype=np.int64)
    offs = []
    for a in accs:   # per Acc: chromosome code -> offset, -1 where the Acc does not hold it
        offs.append(np.array([a.off.get(c, -1) for c in names] + [-1], dtype=np.int64))
    n_lines = n_main = 0
    reader = pd.read_csv(ta_path, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"], chunksize=chunksize,
                         dtype={"chrom": str, "start": np.int64, "end": np.int64})
    for chunk in reader:
        n = len(chunk)
        n_lines += n
        code = chunk["chrom"].map(code_of).fillna(-1).to_numpy(np.int64)   # -1 -> the last slot
        main = code >= 0
        n_main += int(main.sum())
        b0 = chunk["start"].to_numpy() // RES
        b1 = chunk["end"].to_numpy() // RES
        n_grid = nbins[code] + 1
        ok = main & (b0 < n_grid)
        b1 = np.minimum(b1, n_grid - 1)
        for a, off in zip(accs, offs):
            k = a.keep(n, pairs)          # drawn for every unit, counted or not
            o = off[code]
            sel = ok & (o >= 0) if k is None else ok & (o >= 0) & k
            if k is not None:
                a.kept_lines += int(k.sum())
            g0 = o[sel] + b0[sel]
            g1 = o[sel] + b1[sel] + 1
            a.d += np.bincount(g0, minlength=a.d.size)
            a.d -= np.bincount(g1, minlength=a.d.size)
    if pairs and n_lines % 2:
        raise ValueError(f"{ta_path}: odd line count {n_lines} for a paired-end tagAlign")
    return {"n_lines": n_lines, "n_lines_main": n_main}


def count(ta_path, sizes: dict, chunksize: int = CHUNK) -> dict:
    a = Acc(sizes, list(sizes))
    feed(ta_path, [a], chunksize=chunksize)
    return a.finish()


def thin(ta_paths, ratio: float, rng, sizes: dict, pairs: bool = False, chroms=None,
         chunksize: int = CHUNK) -> tuple[dict, dict]:
    """Whole-read thinning of one or more tagAligns (read in order from one rng) into one count."""
    a = Acc(sizes, chroms or list(sizes), ratio, rng)
    stats = [feed(p, [a], pairs, chunksize) for p in ta_paths]
    return a.finish(), {"n_lines": sum(s["n_lines"] for s in stats), "kept_lines": a.kept_lines}


# ---------------------------------------------------------------------------------------------
# atomic writes


def write_npz(path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[: -len(".npz")] + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def same(a: dict, ref: dict) -> list:
    """Chromosomes where two count dicts differ (keys of `ref` but chrM / chrY)."""
    bad = [c for c in ref if c not in DROP and (c not in a or a[c].shape != ref[c].shape
                                                 or not np.array_equal(a[c], ref[c]))]
    return bad + [c for c in a if c not in ref]


# ---------------------------------------------------------------------------------------------
# product mode


def _tsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _load(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {c: z[c] for c in z.files}


def ladder(tas: dict, pid_index: int, sizes: dict, pairs: bool, chunksize: int = CHUNK):
    """The floor ladder of one product: pr1 then pr2 from one rng per (ratio, seed), as floor.py.

    Returns ({half: full count}, {(ratio, seed, half): counts}, {(ratio, seed, half): stats})."""
    rngs = {(r, s): rng_for(s, pid_index, di) for di, r in FLOOR_RATIOS for s in SEEDS}
    full, out, stats = {}, {}, {}
    for h in ("pr1", "pr2"):
        f = Acc(sizes, list(sizes))
        accs = {(r, s): Acc(sizes, list(sizes) if (r, s) == BINS1K else FLOOR_CHROMS, r,
                            rngs[(r, s)]) for (r, s) in rngs}
        st = feed(tas[h], [f, *accs.values()], pairs, chunksize)
        full[h] = f.finish()
        for k, a in accs.items():
            out[(*k, h)] = a.finish()
            stats[(*k, h)] = {"n_lines": st["n_lines"], "kept_lines": a.kept_lines}
    return full, out, stats


def null(tas: list, ratio: float, pid_index: int, sizes: dict, pairs: bool,
         chunksize: int = CHUNK):
    """The depth null: the base read set's tagAligns, in order, thinned per seed; plus the full
    count of the union (to check against the base product)."""
    f = Acc(sizes, list(sizes))
    accs = {s: Acc(sizes, list(sizes), ratio, rng_for(s, pid_index, NULL_DEPTH_INDEX))
            for s in SEEDS}
    n = 0
    for p in tas:
        n += feed(p, [f, *accs.values()], pairs, chunksize)["n_lines"]
    return f.finish(), {s: a.finish() for s, a in accs.items()}, \
        {s: {"n_lines": n, "kept_lines": a.kept_lines} for s, a in accs.items()}


def product(pseudoreps_dir, products_dir, out_dir, pid_index: int,
            chunksize: int = CHUNK) -> int:
    pr, prod, out = Path(pseudoreps_dir), Path(products_dir), Path(out_dir)
    man = _tsv(pr / "MANIFEST.tsv")
    pids = sorted({r["pid"] for r in man})
    if not 0 <= pid_index < len(pids):
        print(f"pid_index {pid_index} outside 0..{len(pids) - 1}", file=sys.stderr)
        return 2
    pid = pids[pid_index]
    od = out / pid
    if (od / "DONE").exists():
        print(f"{pid}: {od}/DONE exists; nothing to do", flush=True)
        return 0
    halves = {r["half"]: r for r in man if r["pid"] == pid}
    if set(halves) != {"pr1", "pr2"}:
        print(f"{pid}: halves {sorted(halves)} in MANIFEST.tsv", file=sys.stderr)
        return 2
    pairs = halves["pr1"]["paired_end"] == "True"
    tas = {h: halves[h]["tagalign"] for h in halves}
    sizes = sizes_like(pr / pid / "pr1" / "counts25.npz")
    meta = {"pid": pid, "pid_index": pid_index, "paired_end": pairs, "tagalign": tas,
            "count_rule": "tools/t112/bin25.py counts_from_tagalign (the t117 rule)",
            "unit": "line pair (consecutive mates)" if pairs else "line",
            "sizes": f"len = n_bins x 25 of {pr / pid / 'pr1' / 'counts25.npz'}",
            "chunksize": chunksize, "files": {}}

    full, fl, fst = ladder(tas, pid_index, sizes, pairs, chunksize)
    for h in ("pr1", "pr2"):
        ref = _load(pr / pid / h / "counts25.npz")
        bad = same(full[h], ref)
        if bad:
            print(f"{pid} {h}: count of {tas[h]} != {pr / pid / h / 'counts25.npz'} on {bad}",
                  file=sys.stderr)
            return 3
        n_want = int(halves[h]["n_lines"])
        if fst[(0.5, 0, h)]["n_lines"] != n_want:
            print(f"{pid} {h}: {fst[(0.5, 0, h)]['n_lines']} lines, MANIFEST n_lines {n_want}",
                  file=sys.stderr)
            return 3
    meta["bit_exact"] = {h: "PASS" for h in ("pr1", "pr2")}
    print(f"{pid}: pr1, pr2 counts reproduce the t117 counts25 bit-exactly", flush=True)

    nl = None
    rows = _tsv(prod / "MANIFEST.tsv")
    me = next((r for r in rows if r["pid"] == pid), None)
    if me is not None and me["arm"] == "depth":
        base = next((r for r in rows if r["track"] == me["track"] and r["arm"] == "base"), None)
        if base is None:
            print(f"{pid}: no base product for track {me['track']}", file=sys.stderr)
            return 2
        ratio = float(me["depth"]) / float(base["depth"])
        if not 0 < ratio < 1:
            print(f"{pid}: thin ratio {ratio} not in (0, 1)", file=sys.stderr)
            return 3
        arm_index = sorted({r["pid"] for r in rows}).index(pid)
        bh = {r["half"]: r for r in man if r["pid"] == base["pid"]}
        btas = [bh["pr1"]["tagalign"], bh["pr2"]["tagalign"]]
        bpairs = bh["pr1"]["paired_end"] == "True"
        bsizes = sizes_like(prod / base["pid"] / "counts25.npz")
        bfull, nl, nst = null(btas, ratio, arm_index, bsizes, bpairs, chunksize)
        bad = same(bfull, _load(prod / base["pid"] / "counts25.npz"))
        if bad:
            print(f"{pid}: base {base['pid']} pr1 + pr2 tagAligns != its counts25 on {bad}",
                  file=sys.stderr)
            return 3
        meta["null"] = {"base_pid": base["pid"], "tagalign": btas, "paired_end": bpairs,
                        "ratio": ratio, "depth": me["depth"], "base_depth": base["depth"],
                        "pid_index": arm_index, "depth_index": NULL_DEPTH_INDEX,
                        "base_lines": nst[0]["n_lines"], "bit_exact_base": "PASS"}
        print(f"{pid}: base {base['pid']} union reproduces its counts25; null ratio {ratio:.4f}",
              flush=True)

    od.mkdir(parents=True, exist_ok=True)
    for (r, s, h), arr in fl.items():
        name = floor_name(r, s, h)
        write_npz(od / name, arr)
        di = next(d for d, rr in FLOOR_RATIOS if rr == r)
        side = {"pid": pid, "half": h, "ratio": r, "seed": s, "rng": [s, pid_index, di],
                "tagalign": tas[h], "after": tas["pr1"] if h == "pr2" else None,
                "chroms": list(arr), **fst[(r, s, h)]}
        write_json(od / (name[:-4] + ".json"), side)
        meta["files"][name] = side
    if nl is not None:
        for s, arr in nl.items():
            name = f"null_s{s}.npz"
            write_npz(od / name, arr)
            side = {"pid": pid, "base_pid": meta["null"]["base_pid"], "ratio": meta["null"]["ratio"],
                    "seed": s, "rng": [s, meta["null"]["pid_index"], NULL_DEPTH_INDEX],
                    "tagalign": meta["null"]["tagalign"], "chroms": list(arr), **nst[s]}
            write_json(od / f"null_s{s}.json", side)
            meta["files"][name] = side
    write_json(od / "product.json", meta)
    (od / "DONE").write_text(f"{len(meta['files'])} files\n")
    print(f"done {pid}: {len(meta['files'])} files under {od}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count", help="store-rule counts of a tagAlign (no thinning)")
    c.add_argument("tagalign")
    c.add_argument("out")
    t = sub.add_parser("thin", help="whole-read thinning of a tagAlign, then counts")
    t.add_argument("tagalign")
    t.add_argument("ratio", type=float)
    t.add_argument("seed", type=int)
    t.add_argument("pid_index", type=int)
    t.add_argument("depth_index", type=int)
    t.add_argument("out")
    t.add_argument("--union", nargs="+", default=[], help="tagAligns read after the first, same rng")
    t.add_argument("--pairs", action="store_true", help="the unit is two consecutive lines")
    for p in (c, t):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--chrsz", help=f"chrom.sizes (default {CHRSZ})")
        g.add_argument("--like", help="take the grid from an existing counts25 npz")
    q = sub.add_parser("product", help="the floor ladder (+ depth null) of one product")
    for name in ("pseudoreps_dir", "products_dir", "out_dir"):
        q.add_argument(name)
    q.add_argument("pid_index", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "product":
        return product(a.pseudoreps_dir, a.products_dir, a.out_dir, a.pid_index)
    sizes = sizes_from(a.chrsz, a.like)
    if a.cmd == "count":
        write_npz(a.out, count(a.tagalign, sizes))
        print(f"count {a.tagalign} -> {a.out}", flush=True)
        return 0
    if not 0 < a.ratio <= 1:
        print(f"ratio {a.ratio} not in (0, 1]", file=sys.stderr)
        return 2
    arr, st = thin([a.tagalign, *a.union], a.ratio, rng_for(a.seed, a.pid_index, a.depth_index),
                   sizes, a.pairs)
    write_npz(a.out, arr)
    write_json(Path(a.out).with_suffix(".json"),
               {"tagalign": [a.tagalign, *a.union], "ratio": a.ratio, "pairs": a.pairs,
                "rng": [a.seed, a.pid_index, a.depth_index], **st})
    print(f"thin {a.ratio} -> {a.out}: kept {st['kept_lines']} of {st['n_lines']} lines", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
