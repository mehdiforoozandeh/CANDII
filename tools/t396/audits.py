#!/usr/bin/env python3
"""t396 C5: table audits for iteration 1 (data only, no model).

    audits.py pe-depth      --tables <dir> --rs-store <dir> --out <dir>
    audits.py tag-fragments --tables <dir> --rs-store <dir> --tars <dir> --members <dir> --out <dir>
    audits.py pair-census   --tables <dir> --evalset <json> --rs-store <dir> --out <dir>

pe-depth (h61): one row per measurement of `measurements.tsv` with the table depth columns, the
read-start store's line counts (`provenance.json` of `<rs-store>/<id>/`), the per-strand read
counts, the total overlap count of chr1 from the read-count store (`counts_path`), and
`log2_depth` as `minicandi.covariates.CovariateEncoder.meta` computes it. Rows with `in_eic` =
`blind` keep their table columns; no store file of theirs is read. Writes `rows.tsv`,
`summary.tsv` (per run type) and `listed_rows.tsv` (the rows named in the h61 audit).

tag-fragments (h61): counts lines and distinct read names (a trailing /1 or /2 removed) in the
pr1 tagAlign of a few PE, mixed and SE rows, streamed out of the tagAlign tars. Both mates of a
fragment carry one name, so distinct names = fragments. Writes `tag_fragments.tsv`.

pair-census (R39, decision D4): every pair of `pairs.tsv` and every correction pair of the
evaluation set, with the covariates that differ, split, never_trained, subset_pair, store and
halves presence, and the headline flag (never_trained = 1 and subset_pair = 0). Writes
`census.tsv` (only pairs whose differing knobs include nodedup or se, to stay under 1 MB),
`by_covariate.tsv` (counts over every pair) and `candidates.tsv` (the dedup and SE pairs that the
widened correction headline can use: knob nodedup or se, subset_pair = 0, split = val, both
members with counts and both halves in the read-count store).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

COVARIATES = ("nodedup", "se", "crop", "mapq", "depth")


def log2_depth(row: dict) -> float:
    """`log2_depth` exactly as the model's covariate encoder computes it (column 0 of `meta`)."""
    from minicandi.covariates import CovariateEncoder
    return float(CovariateEncoder({}).meta(row)[0])


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})


def _fmt(v) -> str:
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return str(v)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def counts_file(row: dict) -> Path | None:
    p = (row.get("counts_path") or "").strip()
    if p.startswith("h5:"):
        p = p[3:]
    return Path(p) if p else None


# --------------------------------------------------------------------------- pe-depth

def chr1_overlap_total(path: Path, chrom: str = "chr1") -> int:
    import h5py
    with h5py.File(path, "r") as f:
        return int(np.asarray(f[chrom][:]).sum(dtype=np.int64))


def pe_depth_rows(tables: Path, rs_store: Path, chrom: str = "chr1") -> list[dict]:
    out = []
    for r in read_tsv(tables / "measurements.tsv"):
        mid = r["measurement_id"]
        dr, df = _f(r.get("depth_reads")), _f(r.get("depth_fragments"))
        o = {"measurement_id": mid, "experiment": r.get("experiment", ""),
             "assay": r.get("assay", ""), "pipeline": r.get("pipeline", ""),
             "layer": r.get("layer", ""), "arm_knob": r.get("arm_knob", ""),
             "run_type": r.get("run_type", ""), "in_eic": r.get("in_eic", ""),
             "read_length": r.get("read_length", ""), "depth_reads": r.get("depth_reads", ""),
             "depth_fragments": r.get("depth_fragments", ""),
             "reads_per_fragment_table": dr / df if df > 0 else float("nan"),
             "log2_depth": log2_depth(r),
             "log2_fragments": math.log2(df) if df > 0 else float("nan")}
        if r.get("in_eic") != "blind":
            prov = rs_store / mid / "provenance.json"
            o["in_start_store"] = int(prov.is_file())
            if prov.is_file():
                p = json.loads(prov.read_text())
                s = p["outputs"]["starts.h5"]
                plus, minus = (int(x) for x in s["n_reads_strand"])
                o.update(n_lines_store=int(s["n_lines"]), n_main_store=int(s["n_main"]),
                         lines_per_fragment_store=int(p["lines_per_fragment"]),
                         n_reads_strand_plus=plus, n_reads_strand_minus=minus,
                         strand_balance=abs(plus - minus) / max(plus + minus, 1),
                         n_lines_over_depth_reads=int(s["n_lines"]) / dr if dr > 0 else float("nan"))
            cf = counts_file(r)
            o["in_count_store"] = int(bool(cf) and cf.is_file())
            if o["in_count_store"]:
                tot = chr1_overlap_total(cf, chrom)
                o["chr1_overlap_total"] = tot
                if o.get("n_main_store"):
                    o["chr1_overlap_over_main_lines"] = tot / o["n_main_store"]
        out.append(o)
    return out


ROW_COLS = ["measurement_id", "experiment", "assay", "pipeline", "layer", "arm_knob", "run_type",
            "in_eic", "read_length", "depth_reads", "depth_fragments", "reads_per_fragment_table",
            "in_start_store", "n_lines_store", "n_main_store", "lines_per_fragment_store",
            "n_reads_strand_plus", "n_reads_strand_minus", "strand_balance",
            "n_lines_over_depth_reads", "in_count_store", "chr1_overlap_total",
            "chr1_overlap_over_main_lines", "log2_depth", "log2_fragments"]


def _q(xs: list[float]) -> tuple[float, float, float]:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return (float("nan"),) * 3
    a = np.asarray(xs, dtype=float)
    return float(a.min()), float(np.median(a)), float(a.max())


def pe_depth_summary(rows: list[dict]) -> list[dict]:
    """Per run type. Table columns count every row; store columns count rows read from the store."""
    by = defaultdict(list)
    for r in rows:
        by[r["run_type"]].append(r)
    out = []
    for rt in sorted(by):
        rs = by[rt]
        ratio = [_f(r["reads_per_fragment_table"]) for r in rs]
        st = [r for r in rs if r.get("n_lines_store") not in (None, "")]
        lo, med, hi = _q(ratio)
        nl = _q([_f(r["n_lines_over_depth_reads"]) for r in st])
        sb = _q([_f(r["strand_balance"]) for r in st])
        ov = _q([_f(r.get("chr1_overlap_over_main_lines")) for r in st])
        out.append({
            "run_type": rt, "n_rows": len(rs),
            "n_rows_not_blind": sum(r["in_eic"] != "blind" for r in rs),
            "n_ratio_1": sum(abs(x - 1.0) < 1e-9 for x in ratio),
            "n_ratio_2": sum(abs(x - 2.0) < 1e-9 for x in ratio),
            "n_ratio_other": sum(not (abs(x - 1.0) < 1e-9 or abs(x - 2.0) < 1e-9) for x in ratio),
            "ratio_min": lo, "ratio_median": med, "ratio_max": hi,
            "n_in_start_store": len(st),
            "n_store_lines_eq_depth_reads": sum(int(r["n_lines_store"]) == int(_f(r["depth_reads"]))
                                                for r in st),
            "n_store_lpf_2": sum(int(r["lines_per_fragment_store"]) == 2 for r in st),
            "n_store_lpf_eq_table_ratio": sum(abs(int(r["lines_per_fragment_store"])
                                                  - _f(r["reads_per_fragment_table"])) < 1e-9
                                              for r in st),
            "lines_over_depth_reads_min": nl[0], "lines_over_depth_reads_max": nl[2],
            "n_strand_balance_0": sum(_f(r["strand_balance"]) == 0.0 for r in st),
            "strand_balance_median": sb[1],
            "chr1_overlap_over_main_lines_median": ov[1]})
    return out


SUMMARY_COLS = ["run_type", "n_rows", "n_rows_not_blind", "n_ratio_1", "n_ratio_2",
                "n_ratio_other", "ratio_min", "ratio_median", "ratio_max", "n_in_start_store",
                "n_store_lines_eq_depth_reads", "n_store_lpf_2", "n_store_lpf_eq_table_ratio",
                "lines_over_depth_reads_min", "lines_over_depth_reads_max", "n_strand_balance_0",
                "strand_balance_median", "chr1_overlap_over_main_lines_median"]


def listed(rows: list[dict]) -> list[dict]:
    """The rows h61 names: PE at table ratio 1, mixed at 2, SE not at 1, and every row where the
    store disagrees with the table (lines_per_fragment != ratio, or n_lines != depth_reads)."""
    out = []
    for r in rows:
        x = _f(r["reads_per_fragment_table"])
        why = []
        if r["run_type"] == "paired-ended" and abs(x - 1) < 1e-9:
            why.append("pe_ratio_1")
        if r["run_type"] == "paired-ended" and abs(x - 2) < 1e-9:
            why.append("pe_ratio_2")
        if r["run_type"] == "mixed" and abs(x - 2) < 1e-9:
            why.append("mixed_ratio_2")
        if r["run_type"] == "mixed" and abs(x - 1) < 1e-9:
            why.append("mixed_ratio_1")
        if r["run_type"] == "single-ended" and not abs(x - 1) < 1e-9:
            why.append("se_ratio_not_1")
        if r.get("n_lines_store") not in (None, ""):
            if abs(int(r["lines_per_fragment_store"]) - x) > 1e-9:
                why.append("store_lpf_ne_table_ratio")
            if int(r["n_lines_store"]) != int(_f(r["depth_reads"])):
                why.append("store_lines_ne_depth_reads")
        if why:
            out.append(dict(r, reason="+".join(why)))
    return out


def cmd_pe_depth(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pe_depth_rows(Path(a.tables), Path(a.rs_store), a.chrom)
    write_tsv(out / "rows.tsv", rows, ROW_COLS)
    write_tsv(out / "summary.tsv", pe_depth_summary(rows), SUMMARY_COLS)
    write_tsv(out / "listed_rows.tsv", listed(rows), ["reason"] + ROW_COLS)
    print(f"pe-depth: {len(rows)} rows -> {out}")


# --------------------------------------------------------------------------- tag-fragments

def strip_mate(name: str) -> str:
    return name[:-2] if name.endswith(("/1", "/2")) else name


def count_names(lines) -> tuple[int, int]:
    """(n_lines, n_distinct_names) of tagAlign lines (name = column 4, mate suffix removed)."""
    n, names = 0, set()
    for ln in lines:
        n += 1
        names.add(strip_mate(ln.split("\t")[3]))
    return n, len(names)


def _stream_counts(tar: Path, member: str, tmp: str) -> tuple[int, int]:
    """Same as `count_names`, for a member of a tar, through tar | gzip | sort -u (bounded memory)."""
    sh = (f"set -o pipefail; OCC=-q; tar --version 2>/dev/null | grep -q GNU && OCC=--occurrence=1; "
          f"tar -xOf '{tar}' $OCC '{member}' | gzip -dc | cut -f4 "
          f"| tee >(wc -l > '{tmp}/lines') | sed -E 's#/[12]$##' | LC_ALL=C sort -u -S 1500M "
          f"-T '{tmp}' | wc -l")
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    distinct = int(r.stdout.split()[0])
    import time
    for _ in range(50):
        t = Path(tmp, "lines").read_text().strip()
        if t:
            break
        time.sleep(0.2)
    return int(t), distinct


def member_index(members: Path) -> dict[str, str]:
    """member path -> tar file name, from `members_<tar stem>.txt` lists."""
    idx = {}
    for f in sorted(members.glob("members_*.txt")):
        tar = f.stem[len("members_"):] + ".tar"
        for ln in f.read_text().split():
            idx[ln] = tar
    return idx


def pick_samples(rows: list[dict], rs_store: Path, idx: dict, per_class: int) -> list[dict]:
    """per class (run_type, table ratio): the rows with one pr1 tagAlign found in a tar, smallest
    pr1 line count first."""
    cand = defaultdict(list)
    for r in rows:
        if r.get("in_eic") == "blind":
            continue
        prov = rs_store / r["measurement_id"] / "provenance.json"
        if not prov.is_file():
            continue
        p = json.loads(prov.read_text())
        files = p.get("read_files", {}).get("pr1", [])
        if len(files) != 1:
            continue
        mem = files[0].split("/rs392/work/", 1)[-1]
        if mem not in idx:
            continue
        x = _f(r["reads_per_fragment_table"])
        cls = (r["run_type"], round(x) if not math.isnan(x) else -1)
        cand[cls].append(dict(r, member=mem, tar=idx[mem],
                              n_lines_store_pr1=int(p["outputs"]["starts_pr1.h5"]["n_lines"]),
                              lines_per_fragment_store=int(p["lines_per_fragment"])))
    out = []
    for cls in sorted(cand):
        k = 1 if cls[0] == "single-ended" else per_class
        out += sorted(cand[cls], key=lambda r: r["n_lines_store_pr1"])[:k]
    return out


def cmd_tag_fragments(a) -> None:
    import tempfile
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(r, reads_per_fragment_table=_f(r["depth_reads"]) / _f(r["depth_fragments"])
                 if _f(r["depth_fragments"]) > 0 else float("nan"))
            for r in read_tsv(Path(a.tables) / "measurements.tsv")]
    samples = pick_samples(rows, Path(a.rs_store), member_index(Path(a.members)), a.per_class)
    res = []
    for s in samples:
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            try:
                n, d = _stream_counts(Path(a.tars) / s["tar"], s["member"], tmp)
                err = ""
            except Exception as e:  # noqa: BLE001 -- recorded in the table, the loop goes on
                n, d, err = -1, -1, str(e).replace("\t", " ").replace("\n", " ")[:200]
        res.append(dict(s, tag_lines_pr1=n, tag_distinct_names_pr1=d,
                        tag_lines_per_name=n / d if d > 0 else float("nan"),
                        lines_match_store=int(n == s["n_lines_store_pr1"]), error=err))
        print(f"tag-fragments: {s['measurement_id']} lines={n} names={d}", flush=True)
    write_tsv(out / "tag_fragments.tsv", res,
              ["measurement_id", "run_type", "pipeline", "reads_per_fragment_table",
               "lines_per_fragment_store", "member", "n_lines_store_pr1", "tag_lines_pr1",
               "tag_distinct_names_pr1", "tag_lines_per_name", "lines_match_store", "error"])


# --------------------------------------------------------------------------- pair-census

def knob_flags(knobs: str) -> dict:
    ks = set(k for k in (knobs or "").split("+") if k)
    return {c: int(c in ks) for c in COVARIATES}


def store_presence(meas: dict, rs_store: Path, mid: str) -> dict:
    r = meas.get(mid)
    cf = counts_file(r) if r else None
    d = cf.parent if cf else None
    return {"count": int(bool(cf) and cf.is_file()),
            "halves": int(bool(d) and (d / "pr1.h5").is_file() and (d / "pr2.h5").is_file()),
            "start": int((rs_store / mid / "starts.h5").is_file())}


def census_rows(tables: Path, evalset: dict, rs_store: Path) -> list[dict]:
    meas = {r["measurement_id"]: r for r in read_tsv(tables / "measurements.tsv")}
    ev_ids = {p["pair_id"] for p in evalset.get("correction_pairs", [])}
    pres = {}

    def pr(mid):
        if mid not in pres:
            pres[mid] = store_presence(meas, rs_store, mid)
        return pres[mid]

    def parent(mid):
        r = meas.get(mid) or {}
        return r.get("parent_measurement_id") or mid

    def row(src, pid, a, b, rung, assay, cell, knobs, split, nt, sp, in_ev):
        pa, pb = pr(a), pr(b)
        par = [pr(parent(a)), pr(parent(b))]
        o = {"source_file": src, "pair_id": pid, "a": a, "b": b, "rung": rung, "assay": assay,
             "cell": cell, "knobs": knobs, "split": split, "never_trained": int(nt or 0),
             "subset_pair": int(sp or 0), "in_evalset": int(in_ev),
             "experiment": a.split(".")[0],
             "both_in_count_store": int(pa["count"] and pb["count"]),
             "halves_present": int(pa["halves"] and pb["halves"]),
             "both_in_start_store": int(pa["start"] and pb["start"]),
             "parents_in_both_stores": int(all(x["count"] and x["start"] for x in par))}
        o.update(knob_flags(knobs))
        o["headline_now"] = int(o["never_trained"] == 1 and o["subset_pair"] == 0)
        o["widened_candidate"] = int((o["nodedup"] or o["se"]) and o["subset_pair"] == 0
                                     and split == "val" and o["both_in_count_store"]
                                     and o["halves_present"])
        return o

    out = []
    for p in read_tsv(tables / "pairs.tsv"):
        out.append(row("pairs.tsv", p["pair_id"], p["a"], p["b"], p.get("rung", ""),
                       p.get("assay", ""), p.get("cell", ""), p.get("knobs_differing", ""),
                       p.get("split", ""), p.get("never_trained"), p.get("subset_pair"),
                       p["pair_id"] in ev_ids))
    for p in evalset.get("correction_pairs", []):
        out.append(row("evalset", p["pair_id"], p["source"], p["target"], "", p.get("assay", ""),
                       p.get("cell", ""), p.get("knob", ""), p.get("split", ""),
                       p.get("never_trained"), p.get("subset_pair"), True))
    return out


CENSUS_COLS = ["source_file", "pair_id", "a", "b", "rung", "assay", "cell", "knobs", *COVARIATES,
               "split", "never_trained", "subset_pair", "in_evalset", "both_in_count_store",
               "halves_present", "both_in_start_store", "parents_in_both_stores", "headline_now",
               "widened_candidate"]


def by_covariate(rows: list[dict]) -> list[dict]:
    """Counts per (source file, covariate, split, never_trained, subset_pair): pairs, experiments,
    pairs with halves, headline pairs now, widened candidates."""
    groups = defaultdict(list)
    for r in rows:
        for c in COVARIATES:
            if r[c]:
                groups[(r["source_file"], c, r["split"], r["never_trained"],
                        r["subset_pair"])].append(r)
    out = []
    for k in sorted(groups):
        rs = groups[k]
        out.append({"source_file": k[0], "covariate": k[1], "split": k[2], "never_trained": k[3],
                    "subset_pair": k[4], "n_pairs": len(rs),
                    "n_experiments": len({r["experiment"] for r in rs}),
                    "n_in_evalset": sum(r["in_evalset"] for r in rs),
                    "n_halves_present": sum(r["halves_present"] for r in rs),
                    "n_parents_in_both_stores": sum(r["parents_in_both_stores"] for r in rs),
                    "n_headline_now": sum(r["headline_now"] for r in rs),
                    "n_widened_candidate": sum(r["widened_candidate"] for r in rs)})
    return out


BYCOV_COLS = ["source_file", "covariate", "split", "never_trained", "subset_pair", "n_pairs",
              "n_experiments", "n_in_evalset", "n_halves_present", "n_parents_in_both_stores",
              "n_headline_now", "n_widened_candidate"]


def cmd_pair_census(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = census_rows(Path(a.tables), json.loads(Path(a.evalset).read_text()), Path(a.rs_store))
    write_tsv(out / "census.tsv", [r for r in rows if r["nodedup"] or r["se"]], CENSUS_COLS)
    write_tsv(out / "by_covariate.tsv", by_covariate(rows), BYCOV_COLS)
    write_tsv(out / "candidates.tsv", [r for r in rows if r["widened_candidate"]], CENSUS_COLS)
    c = Counter((r["source_file"], r["widened_candidate"]) for r in rows)
    print(f"pair-census: {len(rows)} pairs, candidates {dict(c)} -> {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pe-depth")
    p.add_argument("--tables", required=True)
    p.add_argument("--rs-store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chrom", default="chr1")
    p.set_defaults(fn=cmd_pe_depth)
    p = sub.add_parser("tag-fragments")
    p.add_argument("--tables", required=True)
    p.add_argument("--rs-store", required=True)
    p.add_argument("--tars", required=True)
    p.add_argument("--members", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--tmp", default=None)
    p.set_defaults(fn=cmd_tag_fragments)
    p = sub.add_parser("pair-census")
    p.add_argument("--tables", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--rs-store", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pair_census)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
