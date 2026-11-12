"""t152 Amendment 3 chunk RCt -- recount tagAligns at 1 bp with the point rule into the region store.

    python recount.py regions --evalset <json> --mask <mask.h5> --n-train 512 --seed 0 --out regions.tsv
    python recount.py run --work <dir> --experiment <acc> --regions <tsv> \\
        (--frag-len <int|0> | --frag-len-source xcor [--measurements <dir>] [--tables <eic_v1>]) \\
        --halves {0|1} --out <store dir> [--unit pooled] [--layer uniform] [--run-arm base] \\
        [--roles all|eval] [--chrom-sizes <json>]
    python recount.py verify --out <measurement dir> --counts <eic_v1 counts.h5>
    python recount.py verify --all --tables <eic_res_v1> [--eic-v1 <tables dir>] [--workers N]
    python recount.py tables --eic-v1 <tables dir> --store <dir> --out <eic_res_v1>

POINT RULE (plan, Amendment 3, the fact paragraph). One position per tagAlign line (one read): its
5' end, i.e. `start` on strand `+` and `end - 1` on strand `-` (column 6; BED is half-open), moved
by `shift = frag_len // 2` toward the read's 3' end (`+`: up, `-`: down). ChIP-seq uses the
experiment's fragment length; DNase-seq (the 5' end is the cut site) and ATAC-seq (the pipeline's
tagAlign already carries the Tn5 shift, `align_views.TN5_AWK`) use `--frag-len 0`. Any bin size is
then an exact block sum of the 1 bp array (`pool`). The stored `eic_v1` counts use the overlap rule
(`bin25.counts_from_tagalign`), which does not add across bin sizes.

REGIONS. `regions.tsv` (`region, chrom, start_bp, role`; index = row order): the evalset's eval
regions first (role eval, in evalset order), then `--n-train` seeded training regions (role train,
sorted by chrom and start). Every region is 409,600 bp (16,384 bins of 25 bp) and aligned to a
multiple of its length, as `evalset.candidate_blocks` draws them. A training region has a mask valid
fraction >= 0.9 and is neither an eval region nor a fit region of the evalset.

READ SETS. A measurement `<exp>.<unit>.<layer>.base` reads the base read set's two half tagAligns,
`<work>/<exp>/<unit>.<layer>/base/pr{1,2}.tagAlign.gz`; a pooled unit of a multi-replicate
experiment has no read set of its own and reads the halves of every `br<i>.<layer>/base`, as
`align_views.build` pools them. `verify` proves the read set equals the stored one.

STORE (pinned interface E). `run` writes `<out>/<measurement id>/`: `point1bp.h5` (and with
`--halves 1` `point1bp_pr1.h5`, `point1bp_pr2.h5`): one dataset `/r<region index>` [409600] per
region, uint8, or uint16 (uint32) for a view with a value above 255 (65535); attrs `rule`,
`frag_len_shift`, `frag_len_source`. `counts.h5` (and `pr1.h5`, `pr2.h5`): the same reads pooled to
25 bp in the layout of pinned interface B (one uint32 dataset per chromosome of `--chrom-sizes`,
`floor(len / 25)` bins, 0 outside the regions, attrs `resolution = 25`, `counting_rule = point`).
`provenance.json` records the tagAligns, the regions file and its sha256, the fragment length and
every output's sha256. `<out>/regions.tsv` is a copy of `--regions` (it must match if present).
`--roles eval` (the `br<i>` views of the validation experiments) restricts the files to the eval
regions.

FRAGMENT LENGTH (`--frag-len-source xcor`, ChIP-seq). Per replicate from
`<work>/<exp>/<run arm>/rep<i>.xcor.qc` (column 3, first value; the file `build_tables.py` reads),
else from `fraglen_per_rep` of `<measurements>/<exp>.pooled.<layer>.base/provenance.json`; the
rounded mean over replicates as `align_views.chip_fraglen` (chip.wdl) takes it. With neither, the
median over the assay's experiments of the `frag_len` column of `<tables>/measurements.tsv`, and
`frag_len_source = assay_median`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS, TOOLS / "t112", TOOLS / "t117"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import align_views as AV  # noqa: E402

RES = 25
REGION_BINS = 16384
REGION_BP = REGION_BINS * RES            # 409,600
POOL_SIZES = (5, 25, 50, 125, 250)
MIN_VALID = 0.9
EXCLUDE_CHROMS = ("chrY", "chrM")
HALVES = ("pr1", "pr2")
REGION_COLUMNS = ("region", "chrom", "start_bp", "role")
POINT_RULE = ("point rule: one position per tagAlign line, the 5' end (start on +, end - 1 on -) "
              "moved frag_len // 2 toward the 3' end")
COUNTING_RULE = "point"
TABLE_RUNGS = ("pseudoreplicate", "isogenic", "anisogenic")
CHUNKSIZE = 5_000_000


# --------------------------------------------------------------------------------------------
# small pieces


def pool(x_1bp, b: int) -> np.ndarray:
    """Exact block sum of a 1 bp array (last axis) into blocks of b bp; a trailing partial block is
    dropped. Integer input sums in int64, float input in float64."""
    b = int(b)
    if b < 1:
        raise ValueError(f"block size {b} < 1")
    x = np.asarray(x_1bp)
    n = x.shape[-1] // b
    acc = np.int64 if np.issubdtype(x.dtype, np.integer) or x.dtype == bool else np.float64
    return x[..., : n * b].reshape(x.shape[:-1] + (n, b)).sum(axis=-1, dtype=acc)


def point_positions(start, end, strand, shift: int) -> np.ndarray:
    """0-based point-rule positions of reads given as BED (half-open) start, end and strand."""
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    strand = np.asarray(strand).astype(str)
    plus, minus = strand == "+", strand == "-"
    if not np.all(plus | minus):
        bad = sorted(set(strand[~(plus | minus)].tolist()))[:3]
        raise ValueError(f"tagAlign strand (column 6) not + or -: {bad}")
    return np.where(plus, start + int(shift), end - 1 - int(shift))


def sha256_file(path) -> str:
    return AV.sha256_file(path)


def read_regions(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        if tuple(rd.fieldnames or ()) != REGION_COLUMNS:
            raise ValueError(f"{path}: columns {rd.fieldnames}, want {list(REGION_COLUMNS)}")
        rows = [dict(r) for r in rd]
    out = []
    for i, r in enumerate(rows):
        if int(r["region"]) != i:
            raise ValueError(f"{path}: row {i} has region {r['region']} (index = row order)")
        if r["role"] not in ("eval", "train"):
            raise ValueError(f"{path}: region {i} role {r['role']!r}")
        out.append({"region": i, "chrom": r["chrom"], "start_bp": int(r["start_bp"]),
                    "role": r["role"]})
    return out


def write_tsv(path, columns, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    tmp = Path(str(path) + ".part")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# regions


def make_regions(evalset: dict, mask_path, n_train: int, seed: int) -> list[dict]:
    """Eval regions of the evalset, then n_train seeded training regions disjoint from the eval and
    fit regions, each a mask-valid (>= 0.9) block of REGION_BINS bins aligned to REGION_BINS."""
    import h5py
    rb = int(evalset.get("region_bins", REGION_BINS))
    if rb != REGION_BINS:
        raise ValueError(f"evalset region_bins {rb}, the region store needs {REGION_BINS}")
    taken = {(c, int(s)) for c, s in evalset["regions"]}
    taken |= {(c, int(s)) for c, s in evalset.get("fit_regions", [])}
    cands = []
    with h5py.File(mask_path, "r") as mf:
        for chrom in sorted(mf.keys()):
            if chrom in EXCLUDE_CHROMS:
                continue
            m = np.asarray(mf[chrom][:], bool)
            for s in range(0, m.shape[0] - REGION_BINS + 1, REGION_BINS):
                if (chrom, s) not in taken and m[s:s + REGION_BINS].mean() >= MIN_VALID:
                    cands.append((chrom, s))
    if len(cands) < n_train:
        raise ValueError(f"{len(cands)} candidate training regions < --n-train {n_train}")
    rng = np.random.default_rng(seed)
    train = sorted(cands[i] for i in rng.permutation(len(cands))[:n_train])
    rows = [(c, int(s), "eval") for c, s in evalset["regions"]] + [(c, s, "train") for c, s in train]
    return [{"region": i, "chrom": c, "start_bp": s * RES, "role": role}
            for i, (c, s, role) in enumerate(rows)]


# --------------------------------------------------------------------------------------------
# counting


class Layout:
    """Slots of the selected regions, indexed per chromosome for a vectorised lookup."""

    def __init__(self, regions: list[dict]):
        self.regions = regions
        self.by_chrom = {}
        for slot, r in enumerate(regions):
            self.by_chrom.setdefault(r["chrom"], []).append((r["start_bp"], slot))
        for c, v in self.by_chrom.items():
            v.sort()
            self.by_chrom[c] = (np.array([s for s, _ in v], np.int64),
                                np.array([k for _, k in v], np.int64))


def scan_tagaligns(paths, layout: Layout, shift: int, *, points=True, overlap=False) -> dict:
    """One pass over tagAligns. Returns `n_lines`, `n_in` (reads whose point lies in a region),
    `points` (uint32 [n_slots * REGION_BP], flat) and `overlap` (int64 [n_slots, REGION_BINS],
    the overlap rule of `bin25.counts_from_tagalign` at 25 bp, restricted to the regions)."""
    import pandas as pd
    n_slots = len(layout.regions)
    pts = np.zeros(n_slots * REGION_BP, np.uint32) if points else None
    ovd = np.zeros(n_slots * (REGION_BINS + 1), np.int64) if overlap else None
    n_lines = n_in = 0
    for path in paths:
        reader = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 5],
                             names=["chrom", "start", "end", "strand"], chunksize=CHUNKSIZE,
                             dtype={"chrom": str, "start": np.int64, "end": np.int64,
                                    "strand": str})
        for chunk in reader:
            n_lines += len(chunk)
            for chrom, g in chunk.groupby("chrom", sort=False):
                if chrom not in layout.by_chrom:
                    continue
                starts, slots = layout.by_chrom[chrom]
                st, en = g["start"].to_numpy(), g["end"].to_numpy()
                pos = point_positions(st, en, g["strand"].to_numpy(), shift)
                k = np.searchsorted(starts, pos, side="right") - 1
                ok = (k >= 0) & (pos < starts[np.maximum(k, 0)] + REGION_BP)
                n_in += int(ok.sum())
                if points and ok.any():
                    flat = slots[k[ok]] * REGION_BP + (pos[ok] - starts[k[ok]])
                    u, c = np.unique(flat, return_counts=True)
                    pts[u] += c.astype(np.uint32)
                if overlap:
                    _overlap_add(ovd, st // RES, en // RES, starts // RES, slots)
    out = {"n_lines": n_lines, "n_in": n_in}
    if points:
        out["points"] = pts
    if overlap:
        out["overlap"] = np.cumsum(ovd.reshape(n_slots, REGION_BINS + 1), axis=1)[:, :REGION_BINS]
    return out


def _overlap_add(ovd, b0, b1, sbins, slots) -> None:
    """Difference-array update: each read adds 1 to bins b0..b1 inclusive inside each region it
    meets (a read meets at most the region of b0 and the region of b1)."""
    W = REGION_BINS + 1
    k0 = np.searchsorted(sbins, b0, side="right") - 1
    ok0 = (k0 >= 0) & (b0 < sbins[np.maximum(k0, 0)] + REGION_BINS)
    k1 = np.searchsorted(sbins, b1, side="right") - 1
    ok1 = (k1 >= 0) & (b1 < sbins[np.maximum(k1, 0)] + REGION_BINS) & ~(ok0 & (k1 == k0))
    for k, ok in ((k0, ok0), (k1, ok1)):
        if not ok.any():
            continue
        s = sbins[k[ok]]
        lo = np.maximum(b0[ok], s) - s
        hi = np.minimum(b1[ok], s + REGION_BINS - 1) - s
        base = slots[k[ok]] * W
        np.add.at(ovd, base + lo, 1)
        np.add.at(ovd, base + hi + 1, -1)


# --------------------------------------------------------------------------------------------
# read sets and the fragment length


def readset_halves(work, exp: str, unit: str, layer: str) -> dict:
    """{pr1: [tagAlign], pr2: [tagAlign]} of a measurement's base read set (module doc)."""
    root = Path(work) / exp
    d = root / f"{unit}.{layer}" / "base"
    if all((d / f"{h}.tagAlign.gz").is_file() for h in HALVES):
        return {h: [str(d / f"{h}.tagAlign.gz")] for h in HALVES}
    if unit == "pooled":
        pat = re.compile(rf"br(\d+)\.{re.escape(layer)}")
        brs = sorted((int(m.group(1)), p) for p in root.glob(f"br*.{layer}")
                     if (m := pat.fullmatch(p.name)))
        dirs = [p / "base" for _, p in brs]
        if dirs and all((x / f"{h}.tagAlign.gz").is_file() for x in dirs for h in HALVES):
            return {h: [str(x / f"{h}.tagAlign.gz") for x in dirs] for h in HALVES}
    raise FileNotFoundError(f"no base read set for {exp} unit {unit} layer {layer} under {root}")


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def frag_len_from_sources(work, exp: str, *, run_arm="base", measurements=None, layer="uniform",
                          tables=None) -> dict:
    """{frag_len, frag_len_source, frag_len_per_rep} by the order of the module doc."""
    run = Path(work) / exp / run_arm
    reps = sorted((int(m.group(1)), p) for p in run.glob("rep*.xcor.qc")
                  if (m := re.fullmatch(r"rep(\d+)\.xcor\.qc", p.name)))
    vals = []
    for _, p in reps:
        f = p.read_text().split("\n")[0].split("\t")
        if len(f) >= 3 and f[2].strip():
            vals.append(int(float(f[2].split(",")[0])))
    if vals:
        return {"frag_len": int(round(sum(vals) / len(vals))), "frag_len_source": "xcor",
                "frag_len_per_rep": vals}
    if measurements:
        pj = Path(measurements) / AV.measurement_id(exp, "pooled", layer, "base") / "provenance.json"
        if pj.is_file():
            v = _find_key(json.loads(pj.read_text()), "fraglen_per_rep")
            if v:
                v = [int(x) for x in v]
                return {"frag_len": int(round(sum(v) / len(v))), "frag_len_source": "provenance",
                        "frag_len_per_rep": v}
    if tables:
        rows = list(csv.DictReader(open(Path(tables) / "measurements.tsv", encoding="utf-8"),
                                   delimiter="\t"))
        assays = {r["assay"] for r in rows if r["experiment"] == exp}
        if len(assays) != 1:
            raise SystemExit(f"{exp}: {len(assays)} assays in {tables}/measurements.tsv")
        assay = assays.pop()
        per_exp = {}
        for r in rows:
            if r["assay"] == assay and r.get("frag_len", "").strip():
                per_exp.setdefault(r["experiment"], float(r["frag_len"]))
        if per_exp:
            return {"frag_len": int(round(float(np.median(list(per_exp.values()))))),
                    "frag_len_source": "assay_median", "frag_len_per_rep": []}
    raise SystemExit(f"{exp}: no xcor.qc under {run}, no fraglen_per_rep provenance, and no assay "
                     f"median (pass --measurements / --tables)")


# --------------------------------------------------------------------------------------------
# run


def _dtype_for(vmax: int):
    return np.uint8 if vmax <= 255 else np.uint16 if vmax <= 65535 else np.uint32


def write_region_h5(path, flat: np.ndarray, regions: list[dict], attrs: dict) -> None:
    import h5py
    vmax = int(flat.max()) if flat.size else 0
    dt = _dtype_for(vmax)
    tmp = Path(str(path) + ".part")
    with h5py.File(tmp, "w") as f:
        for slot, r in enumerate(regions):
            f.create_dataset(f"r{r['region']}", data=flat[slot * REGION_BP:(slot + 1) * REGION_BP]
                             .astype(dt), chunks=(16384,), compression="gzip",
                             compression_opts=4, shuffle=True)
        f.attrs.update({**attrs, "dtype": np.dtype(dt).name, "region_bp": REGION_BP})
    os.replace(tmp, path)


def write_pooled_h5(path, flat: np.ndarray, regions: list[dict], sizes: dict, attrs: dict) -> None:
    """The 25 bp pooled file in the layout of pinned interface B (0 outside the regions)."""
    arrays = {c: np.zeros(n // RES, np.uint32) for c, n in sizes.items()}
    for slot, r in enumerate(regions):
        if r["chrom"] not in arrays:
            raise SystemExit(f"region {r['region']} on {r['chrom']}, not in the chromosome sizes")
        s = r["start_bp"] // RES
        arrays[r["chrom"]][s:s + REGION_BINS] = pool(flat[slot * REGION_BP:(slot + 1) * REGION_BP],
                                                     RES)
    tmp = Path(str(path) + ".part")
    AV.write_h5(tmp, arrays, attrs, np.uint32)
    os.replace(tmp, path)


def run(a) -> int:
    t0 = time.time()
    regions_all = read_regions(a.regions)
    regions = [r for r in regions_all if a.roles == "all" or r["role"] == "eval"]
    mid = AV.measurement_id(a.experiment, a.unit, a.layer, "base")
    if a.frag_len is not None:
        fl = {"frag_len": int(a.frag_len), "frag_len_source": "given" if a.frag_len else "unshifted",
              "frag_len_per_rep": []}
    else:
        fl = frag_len_from_sources(a.work, a.experiment, run_arm=a.run_arm,
                                   measurements=a.measurements, layer=a.layer, tables=a.tables)
    shift = fl["frag_len"] // 2
    tas = readset_halves(a.work, a.experiment, a.unit, a.layer)
    sizes = AV.load_sizes(a.chrom_sizes)
    store = Path(a.out)
    store.mkdir(parents=True, exist_ok=True)
    reg_copy = store / "regions.tsv"
    reg_text = Path(a.regions).read_bytes()
    if reg_copy.is_file():
        if reg_copy.read_bytes() != reg_text:
            raise SystemExit(f"{reg_copy} differs from {a.regions}: one store holds one region set")
    else:
        tmp = Path(str(reg_copy) + ".part")
        tmp.write_bytes(reg_text)
        os.replace(tmp, reg_copy)
    d = store / mid
    d.mkdir(parents=True, exist_ok=True)
    layout = Layout(regions)
    half = {h: scan_tagaligns(tas[h], layout, shift) for h in HALVES}
    base = half["pr1"]["points"].astype(np.uint64) + half["pr2"]["points"]
    if base.size and int(base.max()) > np.iinfo(np.uint32).max:
        raise SystemExit(f"{mid}: a 1 bp count overflows uint32")
    base = base.astype(np.uint32)
    attrs = {"rule": POINT_RULE, "frag_len_shift": int(shift),
             "frag_len_source": fl["frag_len_source"], "frag_len": int(fl["frag_len"]),
             "measurement_id": mid, "regions_sha256": hashlib.sha256(reg_text).hexdigest()}
    views = {"": (base, half["pr1"]["n_lines"] + half["pr2"]["n_lines"],
                  half["pr1"]["n_in"] + half["pr2"]["n_in"])}
    if a.halves:
        views.update({f"_{h}": (half[h]["points"], half[h]["n_lines"], half[h]["n_in"])
                      for h in HALVES})
    outputs = {}
    for suf, (flat, n_lines, n_in) in views.items():
        pname = f"point1bp{suf}.h5"
        cname = "counts.h5" if not suf else f"{suf[1:]}.h5"
        write_region_h5(d / pname, flat, regions, {**attrs, "n_lines": n_lines,
                                                   "n_reads_in_regions": n_in})
        write_pooled_h5(d / cname, flat, regions, sizes, {
            "schema": 1, "measurement_id": mid, "depth_reads": int(n_lines),
            "reads_in_regions": int(n_in), "resolution": RES, "counting_rule": COUNTING_RULE,
            "frag_len_shift": int(shift)})
        outputs[pname] = sha256_file(d / pname)
        outputs[cname] = sha256_file(d / cname)
    prov = {"schema": 1, "measurement_id": mid, "experiment": a.experiment, "unit": a.unit,
            "layer": a.layer, "rule": POINT_RULE, "halves": tas, "with_halves": bool(a.halves),
            "regions_tsv": str(Path(a.regions).resolve()), "regions_sha256": attrs["regions_sha256"],
            "regions_role": a.roles, "region_indices": [r["region"] for r in regions],
            "n_lines": views[""][1], "n_reads_in_regions": views[""][2],
            "frag_len": int(fl["frag_len"]), "frag_len_shift": int(shift),
            "frag_len_source": fl["frag_len_source"], "frag_len_per_rep": fl["frag_len_per_rep"],
            "chrom_sizes": str(a.chrom_sizes), "outputs": outputs, "code": {"git_sha": AV.git_sha()},
            "wall_s": round(time.time() - t0, 1), "created_utc": AV.utc()}
    write_json(d / "provenance.json", prov)
    print(f"[recount] {mid}: {views[''][1]} reads, {views[''][2]} in {len(regions)} regions, "
          f"shift {shift} ({fl['frag_len_source']}), wall {prov['wall_s']} s", flush=True)
    return 0


# --------------------------------------------------------------------------------------------
# verify


def _read_region(f, r) -> np.ndarray:
    return np.asarray(f[f"r{r['region']}"][()], np.int64)


def verify_one(mdir, counts_v1) -> list[str]:
    """Problems of one store measurement against its eic_v1 counts.h5 (module doc)."""
    import h5py
    mdir = Path(mdir)
    probs = []
    try:
        prov = json.loads((mdir / "provenance.json").read_text())
    except (OSError, ValueError) as e:
        return [f"provenance.json unreadable: {e}"]
    regions = [r for r in read_regions(prov["regions_tsv"]) if r["region"] in
               set(prov["region_indices"])]
    if hashlib.sha256(Path(prov["regions_tsv"]).read_bytes()).hexdigest() != prov["regions_sha256"]:
        probs.append("regions.tsv changed since the run")
    for name, sha in prov["outputs"].items():
        if not (mdir / name).is_file() or sha256_file(mdir / name) != sha:
            probs.append(f"{name}: missing or sha256 differs from provenance")
    if probs:
        return probs
    shift = int(prov["frag_len_shift"])
    ev = [r for r in regions if r["role"] == "eval"]
    # (1) the read set is the stored one: overlap rule at 25 bp on the eval regions
    layout = Layout(regions)
    tas = [p for h in HALVES for p in prov["halves"][h]]
    scan = scan_tagaligns(tas, layout, shift, points=False, overlap=True)
    with h5py.File(counts_v1, "r") as f:
        for slot, r in enumerate(regions):
            if r["role"] != "eval":
                continue
            s = r["start_bp"] // RES
            ref = np.asarray(f[r["chrom"]][s:s + REGION_BINS], np.int64)
            if ref.shape[0] != REGION_BINS or not np.array_equal(ref, scan["overlap"][slot]):
                n = int((ref != scan["overlap"][slot][:ref.shape[0]]).sum()) + \
                    REGION_BINS - ref.shape[0]
                probs.append(f"overlap recount differs from {counts_v1} on region {r['region']} "
                             f"({r['chrom']}:{r['start_bp']}): {n} bins")
    if not ev:
        probs.append("no eval region: the read set cannot be checked against eic_v1")
    if scan["n_lines"] != prov["n_lines"]:
        probs.append(f"tagAlign lines {scan['n_lines']} != provenance {prov['n_lines']}")
    # (2) pooled = pool(point1bp, 25), sum = reads inside the regions
    views = [("point1bp.h5", "counts.h5", scan["n_in"])]
    if prov.get("with_halves"):
        for h in HALVES:
            sh = scan_tagaligns(prov["halves"][h], layout, shift, points=False)
            views.append((f"point1bp_{h}.h5", f"{h}.h5", sh["n_in"]))
    pts = {}
    for pname, cname, n_in in views:
        with h5py.File(mdir / pname, "r") as fp, h5py.File(mdir / cname, "r") as fc:
            tot_p = 0
            arrs = []
            for r in regions:
                x = _read_region(fp, r)
                arrs.append(x)
                tot_p += int(x.sum())
                s = r["start_bp"] // RES
                if not np.array_equal(np.asarray(fc[r["chrom"]][s:s + REGION_BINS], np.int64),
                                      pool(x, RES)):
                    probs.append(f"{cname} != pool({pname}, 25) on region {r['region']}")
            tot_c = sum(int(np.asarray(fc[c][()], np.int64).sum()) for c in fc.keys())
            pts[pname] = arrs
        if tot_p != n_in:
            probs.append(f"{pname} sums to {tot_p}, reads inside the regions {n_in}")
        if tot_c != tot_p:
            probs.append(f"{cname} sums to {tot_c} != {pname} {tot_p} (counts outside regions?)")
    if prov.get("with_halves"):
        for i, r in enumerate(regions):
            if not np.array_equal(pts["point1bp.h5"][i],
                                  pts["point1bp_pr1.h5"][i] + pts["point1bp_pr2.h5"][i]):
                probs.append(f"point1bp != pr1 + pr2 on region {r['region']}")
    return probs


def _verify_task(args):
    mid, mdir, cv1 = args
    try:
        return mid, verify_one(mdir, cv1)
    except Exception as e:  # noqa: BLE001  (a crash is a failure of that measurement)
        return mid, [f"verify crashed: {type(e).__name__}: {e}"]


def verify_all(tables, eic_v1=None, workers=1) -> int:
    tables = Path(tables)
    if eic_v1 is None:
        eic_v1 = json.loads((tables / "build_info.json").read_text())["eic_v1"]
    v1 = {r["measurement_id"]: r for r in csv.DictReader(
        open(Path(eic_v1) / "measurements.tsv", encoding="utf-8"), delimiter="\t")}
    tasks, fails = [], []
    for r in csv.DictReader(open(tables / "measurements.tsv", encoding="utf-8"), delimiter="\t"):
        mid = r["measurement_id"]
        if mid not in v1 or not v1[mid]["counts_path"].startswith("h5:"):
            fails.append((mid, [f"no h5 counts_path in {eic_v1}/measurements.tsv"]))
            continue
        tasks.append((mid, str(Path(r["counts_path"][3:]).parent), v1[mid]["counts_path"][3:]))
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(_verify_task, tasks, chunksize=1)
    else:
        res = [_verify_task(t) for t in tasks]
    fails += [(m, pr) for m, pr in res if pr]
    for m, pr in fails:
        for x in pr:
            print(f"FAIL {m}: {x}")
    print(f"measurements checked: {len(tasks)}")
    print(f"verify failures: {len(fails)}")
    return 1 if fails else 0


# --------------------------------------------------------------------------------------------
# tables


def build_tables(eic_v1, store, out) -> int:
    from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, WITHHELD_COLUMNS, read_tsv
    eic_v1, store, out = Path(eic_v1), Path(store).resolve(), Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: frozen tables are never overwritten")
    v1 = {r["measurement_id"]: r for r in read_tsv(eic_v1 / "measurements.tsv",
                                                   MEASUREMENT_COLUMNS)}
    reg = store / "regions.tsv"
    reg_sha = sha256_file(reg)
    rows, blind = [], []
    for pj in sorted(store.glob("*/provenance.json")):
        prov = json.loads(pj.read_text())
        mid = prov["measurement_id"]
        if mid not in v1:
            raise SystemExit(f"{mid}: not a row of {eic_v1}/measurements.tsv")
        if prov["regions_sha256"] != reg_sha:
            raise SystemExit(f"{mid}: regions differ from {reg}")
        r = dict(v1[mid])
        if r["in_eic"] == "blind":
            blind.append(mid)
            continue
        d = pj.parent
        r["counts_path"] = f"h5:{d / 'counts.h5'}"
        r["sha256"] = sha256_file(d / "counts.h5")
        r["has_signal"] = "0"
        r["has_pseudoreps"] = "1" if all((d / f"{h}.h5").is_file() for h in HALVES) else "0"
        if int(prov["frag_len_shift"]) > 0:
            r["frag_len"] = str(int(prov["frag_len"]))
        rows.append(r)
    kept = {r["measurement_id"]: r for r in rows}
    pairs = []
    for p in read_tsv(eic_v1 / "pairs.tsv", PAIR_COLUMNS):
        if p["rung"] not in TABLE_RUNGS or p["split"] != "val":
            continue
        if p["a"] not in kept or p["b"] not in kept:
            continue
        if p["rung"] == "pseudoreplicate" and kept[p["a"]]["has_pseudoreps"] != "1":
            continue
        pairs.append(p)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regions.tsv").write_bytes(reg.read_bytes())
    write_tsv(out / "measurements.tsv", MEASUREMENT_COLUMNS, rows)
    write_tsv(out / "pairs.tsv", PAIR_COLUMNS, pairs)
    write_tsv(out / "withheld.tsv", WITHHELD_COLUMNS, [])
    (out / "covariates.json").write_bytes((eic_v1 / "covariates.json").read_bytes())
    write_json(out / "build_info.json", {
        "eic_v1": str(eic_v1.resolve()), "store": str(store), "regions_sha256": reg_sha,
        "n_measurements": len(rows), "n_pairs": len(pairs), "blind_skipped": blind,
        "code": {"git_sha": AV.git_sha()}, "created_utc": AV.utc()})
    print(f"[recount tables] {len(rows)} measurements, {len(pairs)} pairs, "
          f"{len(blind)} blind rows skipped -> {out}")
    return 0


# --------------------------------------------------------------------------------------------
# cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("regions")
    g.add_argument("--evalset", required=True)
    g.add_argument("--mask", required=True)
    g.add_argument("--n-train", type=int, default=512)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--experiment", required=True)
    r.add_argument("--regions", required=True)
    fl = r.add_mutually_exclusive_group(required=True)
    fl.add_argument("--frag-len", type=int)
    fl.add_argument("--frag-len-source", choices=("xcor",))
    r.add_argument("--measurements", help="eic_v1 measurement dirs (fraglen_per_rep fallback)")
    r.add_argument("--tables", help="eic_v1 tables dir (assay-median fallback)")
    r.add_argument("--run-arm", default="base")
    r.add_argument("--halves", type=int, choices=(0, 1), required=True)
    r.add_argument("--out", required=True, help="store root; writes <out>/<measurement id>/")
    r.add_argument("--unit", default="pooled")
    r.add_argument("--layer", default="uniform")
    r.add_argument("--roles", choices=("all", "eval"), default="all")
    r.add_argument("--chrom-sizes", default=AV.GENOME_SIZES)
    v = sub.add_parser("verify")
    v.add_argument("--out", help="one store measurement dir")
    v.add_argument("--counts", help="its eic_v1 counts.h5")
    v.add_argument("--all", action="store_true")
    v.add_argument("--tables", help="eic_res_v1 tables dir (with --all)")
    v.add_argument("--eic-v1", help="eic_v1 tables dir (default: from build_info.json)")
    v.add_argument("--workers", type=int, default=1)
    t = sub.add_parser("tables")
    t.add_argument("--eic-v1", required=True)
    t.add_argument("--store", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "regions":
        rows = make_regions(json.loads(Path(a.evalset).read_text()), a.mask, a.n_train, a.seed)
        write_tsv(a.out, REGION_COLUMNS, rows)
        print(f"[recount regions] {sum(x['role'] == 'eval' for x in rows)} eval + "
              f"{sum(x['role'] == 'train' for x in rows)} train -> {a.out}")
        return 0
    if a.cmd == "run":
        if a.frag_len is not None and a.frag_len < 0:
            ap.error("--frag-len must be >= 0")
        return run(a)
    if a.cmd == "verify":
        if a.all:
            if not a.tables:
                ap.error("verify --all needs --tables")
            return verify_all(a.tables, a.eic_v1, a.workers)
        if not (a.out and a.counts):
            ap.error("verify needs --out and --counts, or --all --tables")
        probs = verify_one(a.out, a.counts)
        for x in probs:
            print(f"FAIL {Path(a.out).name}: {x}")
        print(f"verify failures: {1 if probs else 0}")
        return 1 if probs else 0
    return build_tables(a.eic_v1, a.store, a.out)


if __name__ == "__main__":
    sys.exit(main())
