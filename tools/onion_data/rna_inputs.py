#!/usr/bin/env python
"""XRd (Amendment 3): list, fetch, bin and check the bulk RNA-seq inputs of the EIC cells.

The two RNA-seq arms (`rna_track`, `rna_vector`) are ablation only (PI ruling 27). This tool makes
their inputs, in the layout of pinned interface E:

    rna_inputs.py list   --census eic_bulk_rna_census.tsv --out rna_cells.tsv
    rna_inputs.py fetch  --cells rna_cells.tsv [--files DIR] [--max-gb 100]
    rna_inputs.py tracks --cells rna_cells.tsv --chrsz SIZES --out-dir TRACKS [--files DIR]
    rna_inputs.py vector --cells rna_cells.tsv --gene-types GTF|TSV --out-dir ROOT [--files DIR]
    rna_inputs.py check  --root ROOT --tracks TRACKS [--cells ROOT/rna_cells.tsv] [--files DIR]

list. One row per census cell with a `best_exp`, from the ENCODE REST API. The experiment must be
released and unperturbed. Signal: the plus and minus strand "signal of unique reads" bigWigs
(GRCh38) of one biological replicate set (stranded = 1); without them the one unstranded "signal of
unique reads" bigWig serves both channels (stranded = 0, `plus_bigwig = minus_bigwig`, half the
signal to each channel in `tracks`). Gene quantifications: GRCh38 "gene quantifications" tsv. The
GENCODE version that the most cells have a gene quantification of is preferred for every cell (a
tie goes to the higher version); a cell without it takes its highest version. Column `gencode`
records each cell's version. Within a version: the signal's replicate set first, then the portal's
`preferred_default`, then the lowest replicate, then the accession. `rna_match` is `same` for a
census row whose `same_donor_or_biosample` starts with "yes", else `other_donor`. A cell whose
experiment fails these rules is named on stderr and left out.

fetch. md5-checked downloads with `fastq_uniform.fetch_many` into `<files>/<experiment>/`; refuses
when the summed bytes of the listed files pass `--max-gb` (100 GB, the chunk's cap).

tracks. `<out-dir>/<cell>.h5` with `/plus/chrN` and `/minus/chrN` float16, the 25 bp mean of the
bigWig over `floor(len / 25)` bins (`bin25.pval_from_bigwig`'s bigWig route: exact mean, 0 where the
bigWig has no interval). Values are stored as absolute values (a minus-strand bigWig may carry
negative values; the attribute `<strand>_negative_sum` records the sign before that) and clipped to
the float16 maximum (attribute `<strand>_n_clipped`). The reader applies log1p.

vector. `expression.tsv`: one row per cell, column `cell`, then log1p TPM of every protein-coding
gene present in all cells' annotations (Ensembl id without the version suffix; `_PAR_Y` rows
dropped; duplicate ids summed), minus the mitochondrial genes (chrM in the GTF, or a `chrom` of
chrM or a `gene_name` starting "MT-" in a TSV; foreman decision of 2026-11-12: their TPM tracks the
library protocol, not the cell state; `--keep-mito` keeps them). `genes.tsv`: the 2,000 of those genes with the highest variance of
log1p TPM across cells (population variance; a tie goes to the lower gene id). That choice is an
agent call; it is logged on stdout and in `vector_log.json`.

check. Prints `cells with tracks: N`, `cells with vector: N`, `md5 mismatches: N`, and names each
listed cell that lacks a track or a vector. Exits 1 when a listed cell lacks one, a file is missing
or an md5 mismatches.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS, TOOLS / "t112"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import fastq_uniform as FU  # noqa: E402

PORTAL = FU.PORTAL
RES = 25
N_GENES = 2000
MAX_GB = 100.0
FILES_DIR = "/scratch/mforooz/onion/aux/rna/files"
RNA_ASSAYS = ("total RNA-seq", "polyA plus RNA-seq")
OT_PLUS = "plus strand signal of unique reads"
OT_MINUS = "minus strand signal of unique reads"
OT_UNSTRANDED = "signal of unique reads"
OT_GENE_QUANT = "gene quantifications"
F16_MAX = float(np.finfo(np.float16).max)
COLUMNS = ("cell", "cell_id", "experiment", "rna_match", "stranded", "gencode",
           "plus_bigwig", "minus_bigwig", "gene_quant", "bytes", "md5",
           "plus_url", "minus_url", "gene_quant_url",
           "plus_bytes", "minus_bytes", "gene_quant_bytes")


def log(msg):
    print(f"[rna_inputs] {msg}", file=sys.stderr, flush=True)


def get_json(url):
    """The portal GET (patched in tests)."""
    return FU._get_json(url)


# --- list ----------------------------------------------------------------------------------------

def read_tsv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, columns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, columns, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def rna_match(census_row) -> str:
    v = census_row.get("same_donor_or_biosample", "").strip().lower()
    return "same" if v.startswith("yes") else "other_donor"


def _ver(annotation) -> int:
    m = re.match(r"^V(\d+)", annotation or "")
    return int(m.group(1)) if m else -1


def _reps(f) -> tuple:
    return tuple(sorted(f.get("biological_replicates") or ()))


def _url(f) -> str:
    ext = f.get("file_format", "")
    return ((f.get("cloud_metadata") or {}).get("url")
            or f"{PORTAL}/files/{f['accession']}/@@download/{f['accession']}.{ext}")


def _rank(f, want_reps=None) -> tuple:
    """Smaller is better: matching replicate set, preferred_default, lowest replicate, accession."""
    reps = _reps(f)
    return (0 if want_reps is not None and reps == want_reps else 1,
            0 if f.get("preferred_default") else 1, reps or (1 << 30,), f["accession"])


def experiment_files(experiment) -> list[dict]:
    url = (f"{PORTAL}/search/?type=File&dataset=/experiments/{experiment}/&status=released"
           "&assembly=GRCh38&frame=object&limit=all&format=json")
    return [f for f in get_json(url)["@graph"]
            if f.get("status") == "released" and f.get("assembly") == "GRCh38"]


def experiment_ok(experiment) -> str:
    """'' when usable, else the reason."""
    e = get_json(f"{PORTAL}/experiments/{experiment}/?format=json&frame=object")
    if e.get("status") != "released":
        return f"status {e.get('status')}"
    if e.get("perturbed"):
        return "perturbed"
    if e.get("assay_title") not in RNA_ASSAYS:
        return f"assay {e.get('assay_title')}"
    return ""


def choose_signal(files) -> dict | None:
    """Stranded plus/minus pair of one replicate set (the best-ranked pair), else unstranded."""
    bw = [f for f in files if f.get("file_format") == "bigWig"]
    plus = sorted((f for f in bw if f.get("output_type") == OT_PLUS),
                  key=lambda f: (-_ver(f.get("genome_annotation")),) + _rank(f))
    minus = [f for f in bw if f.get("output_type") == OT_MINUS]
    for p in plus:
        m = [f for f in minus if _reps(f) == _reps(p)
             and f.get("genome_annotation") == p.get("genome_annotation")]
        if m:
            return {"stranded": 1, "plus": p, "minus": min(m, key=_rank),
                    "gencode": p.get("genome_annotation", "")}
    un = sorted((f for f in bw if f.get("output_type") == OT_UNSTRANDED),
                key=lambda f: (-_ver(f.get("genome_annotation")),) + _rank(f))
    if un:
        return {"stranded": 0, "plus": un[0], "minus": un[0],
                "gencode": un[0].get("genome_annotation", "")}
    return None


def gene_quants(files) -> list[dict]:
    return [f for f in files if f.get("output_type") == OT_GENE_QUANT
            and f.get("file_format") == "tsv"]


def majority_gencode(per_cell_versions: list[set]) -> str:
    """The version the most cells have; a tie goes to the higher version."""
    c = Counter(v for vs in per_cell_versions for v in vs)
    if not c:
        return ""
    return max(c, key=lambda v: (c[v], _ver(v)))


def choose_gene_quant(gqs, want_version, want_reps):
    same = [f for f in gqs if f.get("genome_annotation") == want_version]
    pool = same or [f for f in gqs if _ver(f.get("genome_annotation")) ==
                    max(_ver(g.get("genome_annotation")) for g in gqs)]
    return min(pool, key=lambda f: _rank(f, want_reps))


def list_cells(census_rows) -> tuple[list[dict], list[tuple[str, str]]]:
    """Rows of rna_cells.tsv and (cell, reason) for every cell left out."""
    cand, missing = [], []
    for r in census_rows:
        cell, exp = r["eic_name"], (r.get("best_exp") or "").strip()
        if not exp:
            missing.append((cell, "no RNA-seq experiment in the census"))
            continue
        why = experiment_ok(exp)
        files = experiment_files(exp) if not why else []
        sig = choose_signal(files) if not why else None
        gqs = gene_quants(files)
        if not why and sig is None:
            why = "no GRCh38 unique-read signal bigWig"
        if not why and not gqs:
            why = "no GRCh38 gene quantifications"
        if why:
            missing.append((cell, f"{exp}: {why}"))
            continue
        cand.append((r, exp, sig, gqs))
    want = majority_gencode([{g.get("genome_annotation") for g in gqs} for _, _, _, gqs in cand])
    log(f"GENCODE version preferred for gene quantifications: {want}")
    rows = []
    for r, exp, sig, gqs in cand:
        gq = choose_gene_quant(gqs, want, _reps(sig["plus"]))
        files = [sig["plus"]] + ([sig["minus"]] if sig["stranded"] else []) + [gq]
        rows.append({
            "cell": r["eic_name"], "cell_id": r.get("cell", ""), "experiment": exp,
            "rna_match": rna_match(r), "stranded": sig["stranded"],
            "gencode": gq.get("genome_annotation", ""),
            "plus_bigwig": sig["plus"]["accession"], "minus_bigwig": sig["minus"]["accession"],
            "gene_quant": gq["accession"],
            "bytes": sum(int(f["file_size"]) for f in files),
            "md5": ";".join(f"{f['accession']}:{f['md5sum']}" for f in files),
            "plus_url": _url(sig["plus"]), "minus_url": _url(sig["minus"]),
            "gene_quant_url": _url(gq),
            "plus_bytes": sig["plus"]["file_size"], "minus_bytes": sig["minus"]["file_size"],
            "gene_quant_bytes": gq["file_size"],
        })
    return rows, missing


# --- fetch ---------------------------------------------------------------------------------------

def _ext(url) -> str:
    return ".tsv" if url.endswith(".tsv") else ".bigWig"


def file_items(cells, files_dir) -> list[dict]:
    """Unique downloads of the listed cells: {url, dest, md5, bytes, accession}."""
    items, seen = [], set()
    for c in cells:
        md5 = dict(kv.split(":", 1) for kv in c["md5"].split(";"))
        for k in ("plus", "minus", "gene_quant"):
            acc = c["gene_quant" if k == "gene_quant" else f"{k}_bigwig"]
            if acc in seen:
                continue
            seen.add(acc)
            url = c[f"{k}_url"]
            items.append({"accession": acc, "url": url, "md5": md5[acc],
                          "bytes": int(c[f"{k}_bytes"]),
                          "dest": str(Path(files_dir) / c["experiment"] / (acc + _ext(url)))})
    return items


def local_path(cell, kind, files_dir) -> Path:
    acc = cell["gene_quant" if kind == "gene_quant" else f"{kind}_bigwig"]
    return Path(files_dir) / cell["experiment"] / (acc + _ext(cell[f"{kind}_url"]))


def fetch(cells, files_dir, max_gb=MAX_GB, fetcher=None) -> dict:
    items = file_items(cells, files_dir)
    total = sum(it["bytes"] for it in items)
    print(f"files: {len(items)}  bytes: {total}  ({total / 1e9:.3f} GB)", flush=True)
    if total > max_gb * 1e9:
        raise SystemExit(f"fetch refused: {total / 1e9:.3f} GB > {max_gb} GB")
    return (fetcher or FU.fetch_many)(items)


# --- tracks --------------------------------------------------------------------------------------

def load_sizes(path) -> dict:
    """`align_views.load_sizes`: the genome layer's `chrom_sizes.json`, or a 2-column TSV."""
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return {c: int(v) for c, v in d.get("chrom_sizes", d).items()}
    import bin25
    return bin25.load_chrsz(p)


def read_bigwig(path, sizes, tmpdir):
    """25 bp exact mean of a bigWig, `{chrom: float32[len // 25]}` (patched in tests)."""
    import bin25
    return bin25.pval_from_bigwig(path, sizes, tmpdir, res=RES)


def strand_arrays(binned: dict, scale: float) -> tuple[dict, dict]:
    out, st = {}, {"negative_sum": 0.0, "n_clipped": 0}
    for c, a in binned.items():
        a = np.asarray(a, dtype=np.float64) * scale
        neg = a[a < 0].sum()
        st["negative_sum"] += float(neg)
        a = np.abs(a)
        st["n_clipped"] += int((a > F16_MAX).sum())
        out[c] = np.minimum(a, F16_MAX).astype(np.float16)
    return out, st


def write_track(path, cell, plus, minus, attrs):
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with h5py.File(part, "w") as h:
        for name, arrs in (("plus", plus), ("minus", minus)):
            g = h.create_group(name)
            for c, a in arrs.items():
                g.create_dataset(c, data=a, compression="gzip", chunks=True)
        h.attrs["cell"] = cell
        h.attrs["resolution"] = RES
        for k, v in attrs.items():
            h.attrs[k] = v
    part.replace(path)


def build_tracks(cells, sizes, out_dir, files_dir, tmpdir, reader=None) -> int:
    reader = reader or read_bigwig
    n = 0
    for c in cells:
        stranded = int(c["stranded"])
        plus_raw = reader(local_path(c, "plus", files_dir), sizes, tmpdir)
        scale = 1.0 if stranded else 0.5
        plus, sp = strand_arrays(plus_raw, scale)
        if stranded:
            minus, sm = strand_arrays(reader(local_path(c, "minus", files_dir), sizes, tmpdir), 1.0)
        else:
            minus, sm = plus, sp
        write_track(Path(out_dir) / f"{c['cell']}.h5", c["cell"], plus, minus, {
            "experiment": c["experiment"], "stranded": stranded, "rna_match": c["rna_match"],
            "plus_bigwig": c["plus_bigwig"], "minus_bigwig": c["minus_bigwig"],
            "value": "mean unique-read signal per 25 bp bin, absolute value"
                     + ("" if stranded else ", half of the unstranded signal per channel"),
            "plus_negative_sum": sp["negative_sum"], "minus_negative_sum": sm["negative_sum"],
            "plus_n_clipped": sp["n_clipped"], "minus_n_clipped": sm["n_clipped"]})
        log(f"{c['cell']}: track written (stranded={stranded})")
        n += 1
    return n


# --- vector --------------------------------------------------------------------------------------

def gene_id(raw) -> str | None:
    """Ensembl gene id without the version suffix; None for spike-ins and `_PAR_Y` rows."""
    if not raw.startswith("ENSG") or raw.endswith("_PAR_Y"):
        return None
    return raw.split(".")[0]


def read_gene_types(path) -> dict:
    """`{gene id: gene_type}` from a GENCODE GTF (optionally gzipped) or a gene_id/gene_type TSV."""
    p = str(path)
    op = gzip.open if p.endswith(".gz") else open
    out = {}
    with op(p, "rt") as f:
        if ".gtf" in p:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                gid = re.search(r'gene_id "([^"]+)"', parts[8])
                gt = re.search(r'gene_(?:bio)?type "([^"]+)"', parts[8])
                g = gene_id(gid.group(1)) if gid else None
                if g and gt:
                    out[g] = gt.group(1)
        else:
            for r in csv.DictReader(f, delimiter="\t"):
                g = gene_id(r["gene_id"])
                if g:
                    out[g] = r["gene_type"]
    return out


def read_mito_genes(path) -> set:
    """Ensembl ids (no version) of the mitochondrial genes: chrM rows of a GTF, or TSV rows whose
    `chrom` is chrM/MT or whose `gene_name` starts with "MT-"."""
    p = str(path)
    op = gzip.open if p.endswith(".gz") else open
    out = set()
    with op(p, "rt") as f:
        if ".gtf" in p:
            for line in f:
                if not line.startswith(("chrM\t", "MT\t")):
                    continue
                parts = line.split("\t")
                if len(parts) >= 9 and parts[2] == "gene":
                    gid = re.search(r'gene_id "([^"]+)"', parts[8])
                    g = gene_id(gid.group(1)) if gid else None
                    if g:
                        out.add(g)
        else:
            for r in csv.DictReader(f, delimiter="\t"):
                g = gene_id(r["gene_id"])
                if g and (r.get("chrom") in ("chrM", "MT")
                          or (r.get("gene_name") or "").startswith("MT-")):
                    out.add(g)
    return out


def read_tpm(path) -> dict:
    """`{gene id: TPM}` from an ENCODE RSEM gene quantification tsv; duplicate ids summed."""
    out: dict[str, float] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            g = gene_id(r["gene_id"])
            if g:
                out[g] = out.get(g, 0.0) + float(r["TPM"])
    return out


def choose_genes(tpm_by_cell: dict, gene_types: dict, n=N_GENES, exclude=frozenset()):
    """Genes in every cell, protein-coding and not in `exclude`; top `n` by population variance of log1p TPM,
    a tie to the lower id. Returns (all kept genes sorted, matrix [cells, genes], chosen rows)."""
    cells = sorted(tpm_by_cell)
    common = set.intersection(*(set(tpm_by_cell[c]) for c in cells))
    pc = sorted(g for g in common if gene_types.get(g) == "protein_coding")
    excluded = [g for g in pc if g in exclude]
    keep = [g for g in pc if g not in exclude]
    x = np.log1p(np.array([[tpm_by_cell[c][g] for g in keep] for c in cells], dtype=np.float64))
    var = x.var(axis=0) if len(cells) else np.zeros(len(keep))
    order = sorted(range(len(keep)), key=lambda j: (-var[j], keep[j]))[:n]
    chosen = [{"gene_id": keep[j], "rank": i + 1, "variance": float(var[j]),
               "mean": float(x[:, j].mean())} for i, j in enumerate(order)]
    return cells, keep, x, chosen, {"cells": len(cells), "genes_in_all_cells": len(common),
                                    "protein_coding_in_all_cells": len(pc),
                                    "excluded_mito": len(excluded), "excluded_mito_ids": excluded,
                                    "genes_kept": len(keep)}


def build_vector(cells, gene_types_path, out_dir, files_dir, n=N_GENES, exclude_mito=True) -> dict:
    tpm = {c["cell"]: read_tpm(local_path(c, "gene_quant", files_dir)) for c in cells}
    mito = read_mito_genes(gene_types_path) if exclude_mito else set()
    names, keep, x, chosen, stats = choose_genes(tpm, read_gene_types(gene_types_path), n, mito)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "expression.tsv", "w") as f:
        f.write("cell\t" + "\t".join(keep) + "\n")
        for i, c in enumerate(names):
            f.write(c + "\t" + "\t".join(f"{v:.6g}" for v in x[i]) + "\n")
    write_tsv(out / "genes.tsv", chosen, ("gene_id", "rank", "variance", "mean"))
    rec = dict(stats, chosen=len(chosen), gencode=sorted({c["gencode"] for c in cells}),
               exclude_mito=exclude_mito,
               rule=f"top {n} protein-coding genes by population variance of log1p TPM across "
                    "cells; genes absent from any cell's annotation removed first"
                    + ("; mitochondrial genes removed (foreman decision 2026-11-12)"
                       if exclude_mito else "")
                    + "; tie to the lower Ensembl id (agent call, XRd)")
    (out / "vector_log.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("vector: " + json.dumps(rec), flush=True)
    return rec


# --- check ---------------------------------------------------------------------------------------

def track_ok(path, sizes=None) -> str:
    """'' when the h5 has a plus and a minus group with the same chromosomes, else the reason."""
    import h5py
    if not Path(path).exists():
        return "no track file"
    try:
        with h5py.File(path, "r") as h:
            if "plus" not in h or "minus" not in h:
                return "plus or minus group missing"
            if set(h["plus"]) != set(h["minus"]) or not len(h["plus"]):
                return "plus and minus chromosomes differ or are empty"
    except OSError as e:
        return f"unreadable: {e}"
    return ""


def check(root, tracks, cells_tsv=None, files_dir=FILES_DIR, md5=True) -> int:
    cells = read_tsv(cells_tsv or Path(root) / "rna_cells.tsv")
    bad, n_tr, n_vec, n_mis, n_missing = [], 0, 0, 0, 0
    vec = set()
    ex = Path(root) / "expression.tsv"
    if ex.exists():
        with open(ex) as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > 1 and all(np.isfinite(float(v)) for v in parts[1:]):
                    vec.add(parts[0])
    for c in cells:
        why = track_ok(Path(tracks) / f"{c['cell']}.h5")
        n_tr += not why
        if why:
            bad.append(f"{c['cell']}: track: {why}")
        if c["cell"] in vec:
            n_vec += 1
        else:
            bad.append(f"{c['cell']}: vector: no finite row in expression.tsv")
    if md5:
        for it in file_items(cells, files_dir):
            p = Path(it["dest"])
            if not p.exists():
                n_missing += 1
                bad.append(f"{it['accession']}: file missing ({p})")
            elif FU.md5_file(p) != it["md5"]:
                n_mis += 1
                bad.append(f"{it['accession']}: md5 mismatch")
    print(f"cells listed: {len(cells)}")
    print(f"cells with tracks: {n_tr}")
    print(f"cells with vector: {n_vec}")
    print(f"md5 mismatches: {n_mis}")
    print(f"files missing: {n_missing}")
    for b in bad:
        print(f"  {b}")
    return 1 if bad else 0


# --- CLI -----------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("--census", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("fetch")
    a.add_argument("--cells", required=True)
    a.add_argument("--files", default=FILES_DIR)
    a.add_argument("--max-gb", type=float, default=MAX_GB)
    a.add_argument("--ranges", type=int, default=FU.RANGES)
    a.add_argument("--parallel-files", type=int, default=FU.PARALLEL_FILES)
    a = sub.add_parser("tracks")
    a.add_argument("--cells", required=True)
    a.add_argument("--chrsz", required=True, help="chrom_sizes.json or a 2-column sizes TSV")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--files", default=FILES_DIR)
    a.add_argument("--tmp", default="/tmp")
    a.add_argument("--cell", action="append", help="only these cells (repeatable)")
    a = sub.add_parser("vector")
    a.add_argument("--cells", required=True)
    a.add_argument("--gene-types", required=True, help="GENCODE GTF[.gz] or gene_id/gene_type tsv")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--files", default=FILES_DIR)
    a.add_argument("--n-genes", type=int, default=N_GENES)
    a.add_argument("--keep-mito", action="store_true", help="keep the mitochondrial genes")
    a = sub.add_parser("check")
    a.add_argument("--root", required=True)
    a.add_argument("--tracks", required=True)
    a.add_argument("--cells")
    a.add_argument("--files", default=FILES_DIR)
    a.add_argument("--no-md5", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        rows, missing = list_cells(read_tsv(args.census))
        write_tsv(args.out, rows, COLUMNS)
        print(f"cells listed: {len(rows)}  same: "
              f"{sum(r['rna_match'] == 'same' for r in rows)}  other_donor: "
              f"{sum(r['rna_match'] == 'other_donor' for r in rows)}  unstranded: "
              f"{sum(r['stranded'] == 0 for r in rows)}  bytes: {sum(r['bytes'] for r in rows)}")
        print(f"gencode: {dict(Counter(r['gencode'] for r in rows))}")
        print(f"cells without RNA-seq input: {len(missing)}")
        for c, why in missing:
            print(f"  {c}: {why}")
        return 0
    if args.cmd == "fetch":
        cells = read_tsv(args.cells)
        fetch(cells, args.files, args.max_gb,
              fetcher=lambda items: FU.fetch_many(items, args.ranges, args.parallel_files))
        return 0
    if args.cmd == "tracks":
        cells = [c for c in read_tsv(args.cells) if not args.cell or c["cell"] in args.cell]
        n = build_tracks(cells, load_sizes(args.chrsz), args.out_dir, args.files, args.tmp)
        print(f"tracks written: {n}")
        return 0
    if args.cmd == "vector":
        build_vector(read_tsv(args.cells), args.gene_types, args.out_dir, args.files, args.n_genes,
                     exclude_mito=not args.keep_mito)
        return 0
    return check(args.root, args.tracks, args.cells, args.files, md5=not args.no_md5)


if __name__ == "__main__":
    sys.exit(main())
