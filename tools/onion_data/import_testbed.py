"""t171 (chunk DT) -- import the t112 counterfactual testbed as measurements of layer `testbed`.

The testbed holds, for 7 EIC tracks, one product per (track, arm, level) under
`<testbed>/products/<pid>/{counts25.npz,pval25.npz,covariates.json,provenance.json}` and its two
pseudoreplicate halves under `<testbed>/pseudoreps/<pid>/pr{1,2}/{counts25.npz,pval25.npz,
provenance.json}` (t117: `encode_task_spr.py`, seed 0, on the product's own treatment tagAlign).
Product ids are `<track>__<arm>__<level>` (`tools/t112/arms.py`).

This writes one pinned-interface-B measurement directory per product whose counts differ from its
track's base, in the layout `align_views.py` writes:

    <out>/<experiment>.pooled.testbed.<arm>/{counts.h5,pr1.h5,pr2.h5,signal.h5,pr1_signal.h5,
                                             pr2_signal.h5,meta.json,provenance.json}

`counts.h5` is `pr1 + pr2`, and it must equal the product's `counts25.npz` bin for bin (the halves
partition the product's reads). The signal files are the testbed's MACS2 `-log10 p` tracks
(`pval25.npz`) of the product and of its halves.

DROP RULE. A product whose `counts25.npz` is byte-identical to its track's base `counts25.npz` adds
no count view; it is dropped and listed. On the testbed these are the signal-only arms (`ratio`,
`ctlid`, `ctldepth`, `extsize`: they change only the MACS2 step) and the two DNase-seq `mapq`
products (the ATAC pipeline in DNase mode runs the filter with `--multimapping 4`, which ignores
`--mapq-thresh`). A signal-only arm whose counts differ from the base is an error, not a view.

BLIND. Every row carries the track's EIC role from the accession ledger in `in_eic`. The rows
table `testbed_rows.tsv` is the `measurements.tsv` columns plus one trailing column `use`; on a
blind row it reads `blind: evaluation use only, never training`. The extra column makes a plain
`read_tsv(path, MEASUREMENT_COLUMNS)` refuse the file, so no table builder takes these rows without
a deliberate step (`rows_for_tables`), and `OnionDataset` refuses `in_eic = blind` rows in
training. Each measurement's `meta.json` holds exactly the `measurements.tsv` columns.

READ OVERLAP. All arms of a track come from the same sequencing reads, and each arm's halves were
split independently. So the pr1 half of one arm and the pr2 half of another arm of the same track
can share reads. A correction pair between two testbed arms is not read-disjoint.

INPUT SAFETY. The half tagAligns exist only under Nibi `/scratch/mforooz/t112_cf/pseudoreps/
_readsets/` (purge 2026-11-20). `run` copies the ones the kept measurements use to
`--halves-copy` on /project, checks each md5 against the testbed record, and every provenance
points at the copy (the chr21 recount of `verify` reads it).

    python import_testbed.py run    [--testbed T] [--ledger L] [--out DIR] [--rows TSV]
                                    [--halves-copy DIR] [--workers N]
    python import_testbed.py verify --rows TSV [--recount sample|all|none] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS, TOOLS / "t112"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arms as T112  # noqa: E402  (tools/t112/arms.py)
import bin25  # noqa: E402
import records  # noqa: E402

TESTBED = "/project/def-maxwl/mforooz/t112_cf"
LEDGER = "/project/def-maxwl/mforooz/ONION/eic/accession_ledger.tsv"
OUT = "/scratch/mforooz/onion/measurements"
ROWS = "/project/def-maxwl/mforooz/ONION/eic/testbed_rows.tsv"
HALVES_COPY = "/project/def-maxwl/mforooz/ONION/testbed/readsets"
GENOME_SIZES = "/project/def-maxwl/mforooz/ONION/genome/chrom_sizes.json"

LAYER = "testbed"
UNIT = "pooled"
RES = 25
CHUNK = 4096
HALVES = ("pr1", "pr2")
USE_BLIND = "blind: evaluation use only, never training"
SIGNAL_ONLY_ARMS = ("ratio", "ctlid", "ctldepth", "extsize")
PIPELINE_NAME = {"ENCODE-DCC/chip-seq-pipeline2": "chip-seq-pipeline2",
                 "ENCODE-DCC/atac-seq-pipeline": "atac-seq-pipeline"}
SIGNAL_KIND = "-log10 p, MACS2 signal step, 25 bp mean (t112 testbed pval25.npz)"


def measurement_columns() -> tuple:
    from minicandi.dataset import MEASUREMENT_COLUMNS
    return tuple(MEASUREMENT_COLUMNS)


def row_columns() -> tuple:
    return measurement_columns() + ("use",)


def dropped_path(rows) -> Path:
    p = Path(rows)
    return p.with_name(p.stem.replace("_rows", "") + "_dropped.tsv")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    stamp = TOOLS.parent / ".kit_stamp"
    if stamp.is_file():
        return stamp.read_text().strip().splitlines()[0]
    import subprocess
    out = subprocess.run(["git", "-C", str(TOOLS.parent), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def job_ids() -> list:
    return [os.environ[k] for k in ("SLURM_JOB_ID",) if os.environ.get(k)]


def write_tsv(path, rows, columns) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    os.replace(tmp, path)


def read_tsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def rows_for_tables(path, allow_blind: bool = False) -> list:
    """The rows of `testbed_rows.tsv` projected onto the `measurements.tsv` columns, for a table
    builder. Blind rows are refused unless `allow_blind` (they are evaluation-only)."""
    cols = measurement_columns()
    out = []
    for r in read_tsv(path):
        if r["in_eic"] == "blind" and not allow_blind:
            raise ValueError(f"{r['measurement_id']}: {r['use'] or 'blind row'}; pass allow_blind")
        out.append({k: r[k] for k in cols})
    return out


# --------------------------------------------------------------------------------------------
# arm mapping (pure)


def parse_pid(pid: str) -> tuple:
    parts = pid.split("__")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"product id {pid!r} is not <track>__<arm>__<level>")
    assert T112.pid(*parts) == pid
    return tuple(parts)


def fmt_fraction(q: float) -> str:
    """0.5 -> '0.50', 0.25 -> '0.25', 0.125 -> '0.125', 0.3 -> '0.30' (align_views style)."""
    s = f"{q:.6f}".rstrip("0")
    whole, _, frac = s.partition(".")
    return f"{whole}.{frac.ljust(2, '0')}"


def arm_spec(arm: str, level: str, cov: dict, base_depth: int) -> dict:
    """Measurement arm name and the covariate columns it sets, for one counted testbed arm.

    `depth`: the product's reads are a seeded subsample of the base tagAlign
    (`encode_task_subsample_ctl.py --subsample <n>`), so the value is the fraction n / base depth
    and `reads_subset_of` is the base. `pe`: the base is single-ended (mate 1), the arm is the
    paired-end run of the same FASTQs; knob `se` with value `pe`."""
    if arm == "base":
        return {"arm": "base", "arm_knob": "", "arm_value": "", "subset": False}
    if arm == "depth":
        q = int(cov["depth"]) / int(base_depth)
        v = fmt_fraction(q)
        return {"arm": f"depth{v}", "arm_knob": "depth", "arm_value": v, "subset": True}
    if arm == "pe":
        return {"arm": "pe", "arm_knob": "se", "arm_value": "pe", "subset": False}
    if arm == "dedup":
        if level != "off":
            raise ValueError(f"dedup level {level!r}")
        return {"arm": "nodedup", "arm_knob": "nodedup", "arm_value": "", "subset": False}
    if arm == "crop":
        return {"arm": f"crop{int(level)}", "arm_knob": "crop", "arm_value": str(int(level)),
                "subset": False}
    if arm == "mapq":
        return {"arm": f"mapq{int(level)}", "arm_knob": "mapq", "arm_value": str(int(level)),
                "subset": False}
    if arm == "abproxy":
        if not level.startswith("f"):
            raise ValueError(f"abproxy level {level!r}")
        v = f"{float(level[1:]):g}"
        return {"arm": f"abproxy{v}", "arm_knob": "abproxy", "arm_value": v, "subset": False}
    raise ValueError(f"arm {arm!r} has no count-level mapping")


# --------------------------------------------------------------------------------------------
# plan: which products, which rows (reads only small json and md5s)


def load_product(testbed: Path, pid: str) -> dict:
    pdir = testbed / "products" / pid
    cov = json.loads((pdir / "covariates.json").read_text())
    if cov.get("pid") != pid:
        raise SystemExit(f"{pdir}/covariates.json: pid {cov.get('pid')!r} != {pid!r}")
    halves = {}
    for h in HALVES:
        hd = testbed / "pseudoreps" / pid / h
        prov = json.loads((hd / "provenance.json").read_text())
        if prov.get("pid") != pid or prov.get("half") != h:
            raise SystemExit(f"{hd}/provenance.json: pid/half mismatch")
        sp = prov["split"]
        halves[h] = {"dir": str(hd), "n_lines": int(sp["n_lines"]),
                     "tagalign": sp["tagalign"], "tagalign_md5": sp["tagalign_md5"],
                     "paired_end": bool(sp["seed"]["paired_end"]),
                     "pval_n_nan": int(prov.get("pval_n_nan", 0) or 0),
                     "pipeline": prov["pipeline"], "readset": prov.get("readset", "")}
    return {"pid": pid, "dir": str(pdir), "cov": cov, "halves": halves,
            "counts_md5": records.md5_file(pdir / "counts25.npz")}


def list_pids(testbed: Path) -> list:
    return sorted(p.name for p in (testbed / "products").iterdir()
                  if p.is_dir() and (p / "covariates.json").is_file())


def ledger_by_track(path) -> dict:
    out = {}
    for r in read_tsv(path):
        if r.get("eic_track"):
            out.setdefault(r["eic_track"], []).append(r)
    return out


def plan(testbed, ledger) -> tuple:
    """(kept items, dropped rows). Each kept item carries its measurements row (no counts_path or
    sha256 yet) and what `build_one` needs."""
    from minicandi import contract as C
    testbed = Path(testbed)
    led = ledger_by_track(ledger)
    prods = {pid: load_product(testbed, pid) for pid in list_pids(testbed)}
    by_track = {}
    for pid in prods:
        track, arm, level = parse_pid(pid)
        by_track.setdefault(track, {})[(arm, level)] = pid
    kept, dropped = [], []
    for track in sorted(by_track):
        if ("base", "base") not in by_track[track]:
            raise SystemExit(f"{track}: no base product")
        lrows = [r for r in led.get(track, []) if r.get("match") == "1"]
        if len(lrows) != 1:
            raise SystemExit(f"{track}: {len(lrows)} ledger rows with match = 1, expected 1")
        L = lrows[0]
        if L["role"] not in ("train", "val", "blind"):
            raise SystemExit(f"{track}: ledger role {L['role']!r}")
        base = prods[by_track[track][("base", "base")]]
        base_mid = f"{L['experiment']}.{UNIT}.{LAYER}.base"
        for (arm, level), pid in sorted(by_track[track].items(),
                                        key=lambda kv: (kv[0][0] != "base", kv[1])):
            p = prods[pid]
            identical = arm != "base" and p["counts_md5"] == base["counts_md5"]
            if arm in SIGNAL_ONLY_ARMS and not identical:
                raise SystemExit(f"{pid}: signal-only arm but counts differ from the base")
            if identical:
                why = ("signal-only arm: counts25.npz byte-identical to the base"
                       if arm in SIGNAL_ONLY_ARMS else
                       "counts25.npz byte-identical to the base (ATAC pipeline in DNase mode: "
                       "--multimapping 4 ignores --mapq-thresh)" if arm == "mapq" else
                       "counts25.npz byte-identical to the base")
                dropped.append({"pid": pid, "track": track, "arm": arm, "level": level,
                                "reason": why, "counts25_md5": p["counts_md5"],
                                "base_counts25_md5": base["counts_md5"]})
                continue
            kept.append(make_item(p, L, track, arm, level, base, base_mid, C))
    return kept, dropped


def make_item(p, L, track, arm, level, base, base_mid, C) -> dict:
    cov = p["cov"]
    if cov["assay"] != L["assay"]:
        raise SystemExit(f"{p['pid']}: assay {cov['assay']!r} != ledger {L['assay']!r}")
    spec = arm_spec(arm, level, cov, base["cov"]["depth"])
    pe = {h: p["halves"][h]["paired_end"] for h in HALVES}
    if pe["pr1"] != pe["pr2"] or pe["pr1"] != (cov["run_type"] == "paired-ended"):
        raise SystemExit(f"{p['pid']}: paired_end of the halves {pe} != run_type {cov['run_type']}")
    lpu = 2 if pe["pr1"] else 1
    lines = {h: p["halves"][h]["n_lines"] for h in HALVES}
    depth_reads = lines["pr1"] + lines["pr2"]
    if depth_reads != int(cov["depth"]):
        raise SystemExit(f"{p['pid']}: halves hold {depth_reads} lines, product depth {cov['depth']}")
    pip = p["halves"]["pr1"]["pipeline"]
    is_chip = pip["repo"].endswith("chip-seq-pipeline2")
    mid = f"{L['experiment']}.{UNIT}.{LAYER}.{spec['arm']}"
    if spec["arm_knob"] == "mapq":
        mapq = spec["arm_value"]
    else:
        # ChIP default filter: --mapq-thresh 30. The DNase-seq base ran the ATAC filter with
        # --multimapping 4, which ignores --mapq-thresh: no effective threshold, left empty.
        mapq = "30" if is_chip else ""
    row = {k: "" for k in measurement_columns()}
    row.update({
        "measurement_id": mid, "experiment": L["experiment"], "assay": cov["assay"],
        "assay_class": C.assay_class(cov["assay"]), "cell": cov["cell"], "eic_track": track,
        "in_eic": L["role"], "unit": UNIT, "layer": LAYER,
        "arm_knob": spec["arm_knob"], "arm_value": spec["arm_value"],
        "parent_measurement_id": "" if arm == "base" else base_mid,
        "reads_subset_of": base_mid if spec["subset"] else "",
        "depth_reads": str(depth_reads), "depth_fragments": str(depth_reads // lpu),
        "run_type": cov["run_type"], "read_length": str(cov["read_length"]),
        "platform": L.get("platform", ""), "lab": L.get("lab", ""),
        "pipeline": PIPELINE_NAME[pip["repo"]], "pipeline_version": pip["release"],
        "image_digest": "sha256:" + pip["sif_sha256"],
        "dedup": "0" if spec["arm_knob"] == "nodedup" else "1", "mapq_threshold": mapq,
        "has_pseudoreps": "1",
        "frag_len": str(cov["fraglen"]) if is_chip else "",
    })
    if row["assay_class"] != L.get("assay_class", row["assay_class"]):
        raise SystemExit(f"{p['pid']}: assay_class {row['assay_class']} != ledger "
                         f"{L['assay_class']}")
    return {"pid": p["pid"], "track": track, "arm": arm, "level": level, "mid": mid,
            "lpu": lpu, "lines": lines, "row": row, "use": USE_BLIND if L["role"] == "blind" else "",
            "product_dir": p["dir"], "halves": p["halves"], "counts_md5": p["counts_md5"],
            "covariates": cov, "base_mid": base_mid}


# --------------------------------------------------------------------------------------------
# build one measurement


def _h5_ds(f, c, v, dtype):
    f.create_dataset(c, data=v.astype(dtype), chunks=(max(1, min(CHUNK, v.size)),),
                     compression="gzip", compression_opts=4, shuffle=True)


def copy_half(src: str, md5: str, dst_root: str, readset: str) -> str:
    """Copy one half tagAlign to `<dst_root>/<readset>/<name>` once; md5 checked both ways."""
    dst = Path(dst_root) / readset / Path(src).name
    if dst.is_file() and records.md5_file(dst) == md5:
        return str(dst)
    if not Path(src).is_file():
        raise SystemExit(f"half tagAlign {src} missing and no verified copy at {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = Path(f"{dst}.part")
    shutil.copyfile(src, part)
    got = records.md5_file(part)
    if got != md5:
        part.unlink()
        raise SystemExit(f"copy of {src}: md5 {got} != recorded {md5}")
    os.replace(part, dst)
    return str(dst)


def build_one(item: dict, cfg: dict) -> dict:
    """Write one measurement dir (staged, then renamed). Skip if `provenance.json` exists."""
    import h5py
    mid = item["mid"]
    final = Path(cfg["out"]) / mid
    if (final / "provenance.json").is_file():
        meta = json.loads((final / "meta.json").read_text())
        return {"mid": mid, "skipped": True, "meta": meta}
    stage = Path(cfg["out"]) / f".{mid}.partial"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    pdir = Path(item["product_dir"])
    hd = {h: Path(item["halves"][h]["dir"]) for h in HALVES}
    tas = {h: copy_half(item["halves"][h]["tagalign"], item["halves"][h]["tagalign_md5"],
                        cfg["halves_copy"], item["halves"][h]["readset"] or item["pid"])
           for h in HALVES}
    want_signal = all((p / "pval25.npz").is_file() for p in (pdir, hd["pr1"], hd["pr2"]))
    zc = np.load(pdir / "counts25.npz")
    zh = {h: np.load(hd[h] / "counts25.npz") for h in HALVES}
    zs = ({"signal.h5": np.load(pdir / "pval25.npz"),
           "pr1_signal.h5": np.load(hd["pr1"] / "pval25.npz"),
           "pr2_signal.h5": np.load(hd["pr2"] / "pval25.npz")} if want_signal else {})
    chroms = [c for c in bin25.MAIN_CHROMS if c in zc.files]
    if set(zc.files) != set(chroms) or any(set(z.files) != set(chroms) for z in zh.values()):
        raise SystemExit(f"{mid}: product and halves hold different chromosomes")
    sizes = {c: cfg["genome_sizes"][c] for c in chroms}
    lpu, lines = item["lpu"], item["lines"]
    attrs = {"counts.h5": lines["pr1"] + lines["pr2"], "pr1.h5": lines["pr1"],
             "pr2.h5": lines["pr2"]}
    files = {n: h5py.File(stage / n, "w") for n in (*attrs, *zs)}
    nonfinite = 0
    try:
        for c in chroms:
            n = sizes[c] // RES
            a = {h: zh[h][c] for h in HALVES}
            ref = zc[c]
            for x in (ref, *a.values()):
                if x.shape != (n,):
                    raise SystemExit(f"{mid} {c}: shape {x.shape} != genome grid ({n},)")
            tot = a["pr1"].astype(np.uint64) + a["pr2"]
            if not np.array_equal(tot, ref):
                raise SystemExit(f"{mid} {c}: pr1 + pr2 != product counts25")
            _h5_ds(files["counts.h5"], c, tot, np.uint32)
            for h in HALVES:
                _h5_ds(files[f"{h}.h5"], c, a[h], np.uint32)
            for name, z in zs.items():
                v = z[c].astype(np.float32)
                if v.shape != (n,):
                    raise SystemExit(f"{mid} {c}: {name} shape {v.shape} != ({n},)")
                nonfinite += int(np.size(v) - np.isfinite(v).sum())
                _h5_ds(files[name], c, v, np.float32)
        for name, depth in attrs.items():
            files[name].attrs.update({"schema": 1, "measurement_id": mid,
                                      "depth_reads": int(depth),
                                      "depth_fragments": int(depth // lpu), "resolution": RES,
                                      "counting_rule": records.COUNT_RULE})
        for name in zs:
            files[name].attrs.update({"schema": 1, "measurement_id": mid, "resolution": RES,
                                      "kind": SIGNAL_KIND})
    finally:
        for f in files.values():
            f.close()
    if nonfinite:
        for name in zs:
            (stage / name).unlink()
        zs = {}
    row = dict(item["row"])
    row["has_signal"] = "1" if zs else "0"
    row["counts_path"] = f"h5:{final.resolve() / 'counts.h5'}"
    row["sha256"] = sha256_file(stage / "counts.h5")
    meta = {k: str(row[k]) for k in measurement_columns()}
    (stage / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    refs = {"counts25": str(pdir / "counts25.npz"),
            "pr1_counts25": str(hd["pr1"] / "counts25.npz"),
            "pr2_counts25": str(hd["pr2"] / "counts25.npz")}
    if zs:
        refs.update({"pval25": str(pdir / "pval25.npz"),
                     "pr1_pval25": str(hd["pr1"] / "pval25.npz"),
                     "pr2_pval25": str(hd["pr2"] / "pval25.npz")})
    for k in list(refs):
        refs[k + "_md5"] = records.md5_file(refs[k])
    prov = {
        "schema": 1, "measurement_id": mid, "experiment": meta["experiment"], "unit": UNIT,
        "arm": mid.rsplit(".", 1)[1], "layer": LAYER, "lines_per_fragment": lpu,
        "halves": {h: {"tagalign": [tas[h]], "n_lines": lines[h],
                       "tagalign_md5": item["halves"][h]["tagalign_md5"],
                       "scratch_source": item["halves"][h]["tagalign"]} for h in HALVES},
        "chrom_sizes": {"path": cfg["genome_sizes_path"], "sizes": sizes,
                        "note": "the testbed counted chr1-chr22 and chrX only"},
        "pipeline": {"name": meta["pipeline"], "version": meta["pipeline_version"],
                     "sif": item["halves"]["pr1"]["pipeline"]},
        "testbed": {"pid": item["pid"], "track": item["track"], "arm": item["arm"],
                    "level": item["level"], "product_dir": str(pdir),
                    "covariates": item["covariates"],
                    "pseudoreps": {h: str(hd[h]) for h in HALVES},
                    "signal_dropped_nonfinite": nonfinite if nonfinite else 0},
        "use": item["use"],
        "references": refs,
        "outputs": {n: records.md5_file(stage / n) for n in sorted(os.listdir(stage))
                    if n.endswith(".h5") or n == "meta.json"},
        "slurm_job_ids": job_ids(), "code": {"git_sha": git_sha(), "tool":
                                              "tools/onion_data/import_testbed.py"},
        "created_utc": utc(),
    }
    (stage / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(stage, final)
    return {"mid": mid, "skipped": False, "meta": meta}


def _build_star(args):
    return build_one(*args)


def load_genome_sizes(path) -> dict:
    d = json.loads(Path(path).read_text())
    return {c: int(v) for c, v in d.get("chrom_sizes", d).items()}


def run(a) -> int:
    t0 = time.time()
    kept, dropped = plan(a.testbed, a.ledger)
    cfg = {"out": a.out, "halves_copy": a.halves_copy, "genome_sizes_path": a.chrom_sizes,
           "genome_sizes": load_genome_sizes(a.chrom_sizes)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"[import_testbed] plan: {len(kept)} measurements, {len(dropped)} dropped", flush=True)
    jobs = [(it, cfg) for it in kept]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            res = []
            for r in pool.imap_unordered(_build_star, jobs):
                res.append(r)
                print(f"[import_testbed] {len(res)}/{len(jobs)} {r['mid']}"
                      f"{' (existing)' if r['skipped'] else ''}", flush=True)
    else:
        res = [build_one(*j) for j in jobs]
    metas = {r["mid"]: r["meta"] for r in res}
    rows = []
    for it in kept:
        r = dict(metas[it["mid"]])
        r["use"] = it["use"]
        rows.append(r)
    write_tsv(a.rows, rows, row_columns())
    write_tsv(dropped_path(a.rows), dropped,
              ("pid", "track", "arm", "level", "reason", "counts25_md5", "base_counts25_md5"))
    print(f"[import_testbed] wrote {a.rows} ({len(rows)} rows) and {dropped_path(a.rows)} "
          f"({len(dropped)} rows); wall {time.time() - t0:.0f} s", flush=True)
    return 0


# --------------------------------------------------------------------------------------------
# verify


def check_measurement(row: dict, recount: bool, base_row: dict | None) -> dict:
    """Problems of one measurement directory, and whether its halves sum to its counts."""
    import h5py
    probs, halves_ok = [], True
    cols = measurement_columns()
    counts_path = row["counts_path"]
    if not counts_path.startswith("h5:"):
        return {"mid": row["measurement_id"], "probs": ["counts_path is not h5:"],
                "halves_ok": False}
    d = Path(counts_path[3:]).parent
    mid = row["measurement_id"]
    need = ("counts.h5", "pr1.h5", "pr2.h5", "meta.json", "provenance.json")
    miss = [n for n in need if not (d / n).is_file()]
    if miss:
        return {"mid": mid, "probs": [f"missing {miss}"], "halves_ok": False}
    meta = json.loads((d / "meta.json").read_text())
    prov = json.loads((d / "provenance.json").read_text())
    if list(meta) != list(cols):
        probs.append("meta.json columns differ from MEASUREMENT_COLUMNS")
    if {k: row[k] for k in cols} != meta:
        probs.append("rows table row != meta.json")
    if d.name != mid or meta.get("measurement_id") != mid or prov.get("measurement_id") != mid:
        probs.append("measurement_id != directory name")
    if meta.get("layer") != LAYER or meta.get("unit") != UNIT:
        probs.append(f"layer/unit {meta.get('layer')}/{meta.get('unit')}")
    if meta.get("in_eic") == "blind" and row.get("use") != USE_BLIND:
        probs.append("blind row without the evaluation-only flag")
    sizes = prov["chrom_sizes"]["sizes"]
    names = ["counts.h5", "pr1.h5", "pr2.h5"]
    if meta["has_signal"] == "1":
        names += ["signal.h5", "pr1_signal.h5", "pr2_signal.h5"]
    fs = {}
    try:
        for n in names:
            if not (d / n).is_file():
                probs.append(f"missing {n}")
                continue
            f = fs[n] = h5py.File(d / n, "r")
            a = dict(f.attrs)
            if set(f.keys()) != set(sizes):
                probs.append(f"{n}: chromosomes != chrom sizes")
            for c in f.keys():
                ds = f[c]
                if ds.shape != (sizes[c] // RES,):
                    probs.append(f"{n} {c}: shape {ds.shape}")
                want = (np.float32,) if "signal" in n else (np.uint16, np.uint32)
                if ds.dtype not in want:
                    probs.append(f"{n} {c}: dtype {ds.dtype}")
                if ds.chunks is None or ds.chunks[0] > CHUNK:
                    probs.append(f"{n} {c}: chunks {ds.chunks}")
            if (a.get("schema"), a.get("measurement_id"), a.get("resolution")) != (1, mid, RES):
                probs.append(f"{n}: root attrs")
        if probs:
            return {"mid": mid, "probs": probs, "halves_ok": False}
        dep = {n: int(fs[n].attrs["depth_reads"]) for n in ("counts.h5", "pr1.h5", "pr2.h5")}
        if dep["pr1.h5"] + dep["pr2.h5"] != dep["counts.h5"]:
            probs.append("depth_reads: pr1 + pr2 != counts")
        if str(dep["counts.h5"]) != meta["depth_reads"]:
            probs.append("meta depth_reads != counts.h5 attrs")
        refs = prov.get("references", {})
        z = {k: np.load(refs[k]) for k in ("counts25", "pr1_counts25", "pr2_counts25", "pval25")
             if refs.get(k)}
        bf = None
        if meta["arm_knob"] == "depth":
            q = float(meta["arm_value"])
            if base_row is None:
                probs.append(f"base {meta['reads_subset_of']} not in the rows table")
            else:
                br = int(base_row["depth_reads"])
                if abs(int(meta["depth_reads"]) - q * br) > 1e-3 * q * br:
                    probs.append(f"depth {meta['depth_reads']} not within 0.1% of {q} x {br}")
                bf = h5py.File(Path(base_row["counts_path"][3:]), "r")
        try:
            for c in sizes:
                cnt = fs["counts.h5"][c][()]
                p1, p2 = fs["pr1.h5"][c][()], fs["pr2.h5"][c][()]
                if not np.array_equal(p1.astype(np.uint64) + p2, cnt):
                    if halves_ok:
                        probs.append(f"halves do not sum to counts on {c}")
                    halves_ok = False
                for k, arr in (("counts25", cnt), ("pr1_counts25", p1), ("pr2_counts25", p2)):
                    if k in z and not np.array_equal(z[k][c], arr):
                        probs.append(f"{k} != testbed reference on {c}")
                if "pval25" in z and "signal.h5" in fs:
                    s = fs["signal.h5"][c][()]
                    if not np.isfinite(s).all() or \
                            float(np.max(np.abs(s - z["pval25"][c]), initial=0)) > 1e-4:
                        probs.append(f"signal.h5 != testbed pval25 on {c}")
                for n in ("pr1_signal.h5", "pr2_signal.h5"):
                    if n in fs and not np.isfinite(fs[n][c][()]).all():
                        probs.append(f"{n}: non-finite on {c}")
                if bf is not None and np.any(cnt > bf[c][()]):
                    probs.append(f"depth arm counts exceed the base on {c}")
        finally:
            if bf is not None:
                bf.close()
    finally:
        for f in fs.values():
            f.close()
    if sha256_file(d / "counts.h5") != meta["sha256"]:
        probs.append("counts.h5 sha256 != meta.json")
    for n, m in prov.get("outputs", {}).items():
        if records.md5_file(d / n) != m:
            probs.append(f"{n}: md5 != provenance")
    if recount:
        rc = "chr21" if "chr21" in sizes else sorted(sizes)[0]
        for h in HALVES:
            ta = prov["halves"][h]["tagalign"]
            got = sum(bin25.counts_from_tagalign(t, {rc: sizes[rc]})[rc].astype(np.uint64)
                      for t in ta)
            with h5py.File(d / f"{h}.h5", "r") as f:
                if not np.array_equal(got, f[rc][()]):
                    probs.append(f"{h}: {rc} != recount of {ta}")
    return {"mid": mid, "probs": probs, "halves_ok": halves_ok}


def _check_star(args):
    return check_measurement(*args)


def verify(rows_path, recount="sample", workers=1) -> int:
    rows = read_tsv(rows_path)
    want = row_columns()
    with open(rows_path, encoding="utf-8") as fh:
        header = tuple(fh.readline().rstrip("\n").split("\t"))
    bad_header = header != want
    by_mid = {r["measurement_id"]: r for r in rows}
    if len(by_mid) != len(rows):
        print("FAIL duplicate measurement ids in the rows table")
        return 1
    sample = set()
    if recount == "sample":
        for r in rows:  # the base of each track
            if r["arm_knob"] == "":
                sample.add(r["measurement_id"])
    jobs = [(r, recount == "all" or r["measurement_id"] in sample,
             by_mid.get(r["reads_subset_of"]) if r["arm_knob"] == "depth" else None)
            for r in rows]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_check_star, jobs, chunksize=1)
    else:
        res = [check_measurement(*j) for j in jobs]
    per_track = {}
    for r in rows:
        per_track[r["eic_track"]] = per_track.get(r["eic_track"], 0) + 1
    print(f"measurements written: {len(rows)}")
    print("per track: " + ", ".join(f"{t} {n}" for t, n in sorted(per_track.items())))
    splits = sorted({(r["eic_track"], r["in_eic"]) for r in rows})
    print("EIC split: " + ", ".join(f"{t} {s}" for t, s in splits))
    n_flag = sum(1 for r in rows if r["in_eic"] == "blind" and r.get("use") == USE_BLIND)
    n_blind = sum(1 for r in rows if r["in_eic"] == "blind")
    print(f"blind rows flagged '{USE_BLIND}': {n_flag} of {n_blind}")
    n_ok = sum(1 for x in res if x["halves_ok"])
    print(f"halves sum to base: {'all' if n_ok == len(res) else f'{n_ok} of {len(res)}'}")
    n_rc = sum(1 for j in jobs if j[1])
    print(f"recount chr21 from the half tagAligns: {n_rc} measurements ({recount})")
    dp = dropped_path(rows_path)
    dropped = read_tsv(dp) if dp.is_file() else []
    print(f"dropped products: {len(dropped)}" + ("" if dp.is_file() else f" ({dp} missing)"))
    for x in dropped:
        print(f"  {x['pid']}: {x['reason']}")
    nbad = 0
    if bad_header:
        print(f"FAIL rows header {list(header)} != {list(want)}")
        nbad += 1
    for x in res:
        if x["probs"]:
            nbad += 1
            for p in x["probs"][:5]:
                print(f"  FAIL {x['mid']}: {p}")
    if nbad or not dp.is_file():
        print(f"{nbad} measurements fail")
        return 1
    print("all checks pass")
    return 0


# --------------------------------------------------------------------------------------------
# CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    r = sub.add_parser("run", help="write the measurement dirs and the rows table")
    r.add_argument("--testbed", default=TESTBED)
    r.add_argument("--ledger", default=LEDGER)
    r.add_argument("--out", default=OUT)
    r.add_argument("--rows", default=ROWS)
    r.add_argument("--halves-copy", default=HALVES_COPY)
    r.add_argument("--chrom-sizes", default=GENOME_SIZES)
    r.add_argument("--workers", type=int, default=workers)
    v = sub.add_parser("verify", help="check every measurement of a rows table")
    v.add_argument("--rows", default=ROWS)
    v.add_argument("--recount", choices=("sample", "all", "none"), default="sample",
                   help="recount chr21 from the half tagAligns: the bases, every row, or none")
    v.add_argument("--workers", type=int, default=workers)
    p = sub.add_parser("plan", help="print the kept and dropped products; writes nothing")
    p.add_argument("--testbed", default=TESTBED)
    p.add_argument("--ledger", default=LEDGER)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "verify":
        return verify(a.rows, a.recount, a.workers)
    kept, dropped = plan(a.testbed, a.ledger)
    for it in kept:
        print(f"keep {it['pid']} -> {it['mid']} depth_reads={it['row']['depth_reads']}")
    for x in dropped:
        print(f"drop {x['pid']}: {x['reason']}")
    print(f"{len(kept)} kept, {len(dropped)} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
