"""t112 — export the counterfactual product tree into a CANDI_STORE source tree for corpus `cf`.

`python -m candi.store build-biosample / build-manifest / verify` then build corpus `cf` from it
unchanged. Nothing here writes an h5: this only puts every product where `store/writer.py` looks.

One store biosample per (cell, arm, level) — `arms.biosample()` — holding every track of that cell
for that arm level. Written under `<source_root>`:

```
<BIOSAMPLE>/<ASSAY>/signal_DSF1_res25/{chr*.npz, metadata.json}   <- products/<pid>/counts25.npz
               /signal_BW_res25/chr*.npz                          <- products/<pid>/pval25.npz
               /file_metadata.json
<BIOSAMPLE>/chipseq-control/signal_DSF1_res25/{chr*.npz, metadata.json}
                                                                 <- control_counts25.npz of the
                                                                    biosample's first track (C1 order)
cf_metadata.csv                 one row per (biosample, assay) incl. chipseq-control; the columns
                                are `candi.store.manifest.CSV_COLUMNS`, read from the module
signal_provenance.cf.json       the shape of configs/signal_provenance.eic.json; every pval layer
                                is MACS2 -log10 p, so every track is `signal p-value`
```

Every track of a biosample must name the same control accession, or this raises: one biosample has
one `chipseq-control` column. Whether two tracks' control ARRAYS agree is not checked here (C10).
Fields the products do not carry (`bios_accession`, `exp_accession`, `lab`, `sequencing_platform`)
are left empty, so the manifest records them as gaps instead of inventing them (D19).

The per-chrom npz key is the chromosome name; the store reads `files[0]` whatever its key.
Needs numpy and the `candi` package (for CSV_COLUMNS), so on Nibi it runs inside the store job.

    python tools/t112/export_store.py --products DIR --manifest MANIFEST.tsv --source-root DIR \
        --chrsz GRCh38_EBV.chrom.sizes.tsv [--genome-json CANDI_STORE/genome/chrom_sizes.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms                                                              # noqa: E402
from bin25 import MAIN_CHROMS, load_chrsz                                # noqa: E402

from candi.store.layout import CONTROL_TRACK                             # noqa: E402
from candi.store.manifest import CSV_COLUMNS, SIGNAL_OUTPUT_TYPE         # noqa: E402

RES = 25
ASSEMBLY = "GRCh38"
CSV_NAME = "cf_metadata.csv"
PROVENANCE_NAME = "signal_provenance.cf.json"


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_chroms(out_dir: Path, npz_path: Path, dtype, n_bins: dict) -> None:
    """One `<chrom>.npz` per main chromosome, each exactly `len // 25` bins of `dtype`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as z:
        for chrom, nb in n_bins.items():
            if chrom not in z.files:
                raise ValueError(f"{npz_path}: no array for {chrom}")
            arr = z[chrom]
            if arr.shape != (nb,):
                raise ValueError(f"{npz_path}:{chrom}: shape {arr.shape} != ({nb},) = len // {RES}")
            if np.dtype(dtype).kind == "u" and arr.dtype.kind not in "ui":
                raise ValueError(f"{npz_path}:{chrom}: counts must be integer, got {arr.dtype}")
            np.savez_compressed(out_dir / f"{chrom}.npz", **{chrom: arr.astype(dtype)})


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def _file_metadata(assay: str, accession: str, read_length: int, run_type: str) -> dict:
    return {"assay": {"1": assay}, "accession": {"1": accession},
            "read_length": {"1": read_length}, "run_type": {"1": run_type}}


def export(products_dir, manifest_tsv, source_root, chrsz, genome_json=None) -> dict:
    """Write the source tree, `cf_metadata.csv` and `signal_provenance.cf.json`; return a summary."""
    products_dir, source_root = Path(products_dir), Path(source_root)
    if source_root.is_dir() and any(source_root.iterdir()):
        raise ValueError(f"{source_root} is not empty; a stale track dir would be built into the "
                         f"corpus. Remove it or pick a fresh --source-root.")
    sizes = load_chrsz(chrsz)
    n_bins = {c: n // RES for c, n in sizes.items()}
    if genome_json is not None:
        _write_json(Path(genome_json), {c: sizes[c] for c in MAIN_CHROMS if c in sizes})

    track_order = list(arms.TRACKS)
    by_bios: dict = {}
    for row in read_manifest(manifest_tsv):
        by_bios.setdefault(row["biosample"], []).append(row)

    csv_rows, tracks_prov = [], {}
    for bios in sorted(by_bios):
        rows = sorted(by_bios[bios], key=lambda r: track_order.index(r["track"]))
        assays = [r["assay"] for r in rows]
        if len(set(assays)) != len(assays) or CONTROL_TRACK in assays:
            raise ValueError(f"{bios}: assays {assays} are not distinct store tracks")
        controls = {r["control_accession"] for r in rows}
        if len(controls) != 1:
            raise ValueError(f"{bios}: tracks name different controls "
                             f"{ {r['pid']: r['control_accession'] for r in rows} }")

        for r in rows:
            pdir, tdir = products_dir / r["pid"], source_root / bios / r["assay"]
            depth, read_length = int(r["depth"]), int(r["read_length"])
            accession = arms.TRACKS[r["track"]]["treat_acc"]
            _write_chroms(tdir / f"signal_DSF1_res{RES}", pdir / "counts25.npz", np.uint32, n_bins)
            _write_json(tdir / f"signal_DSF1_res{RES}" / "metadata.json", {"depth": depth, "dsf": 1})
            _write_chroms(tdir / f"signal_BW_res{RES}", pdir / "pval25.npz", np.float32, n_bins)
            _write_json(tdir / "file_metadata.json",
                        _file_metadata(r["assay"], accession, read_length, r["run_type"]))
            csv_rows.append({"biosample_name": bios, "assay_name": r["assay"],
                             "file_accession": accession, "assembly": ASSEMBLY,
                             "read_length": read_length, "run_type": r["run_type"], "depth": depth})
            tracks_prov.setdefault(bios, {})[r["assay"]] = {
                "signal_bigwig_accession": None,
                "output_type": SIGNAL_OUTPUT_TYPE,
                "assay_term_name": "DNase-seq" if r["assay"] == "DNase-seq" else "ChIP-seq",
                "file_format": "npz",
                "assembly": ASSEMBLY,
                "derivation": {k: r[k] for k in ("pid", "route", "knob", "knob_value",
                                                 "pipeline_repo", "pipeline_release", "sif_md5",
                                                 "pval25_md5")},
            }

        ctl_acc = controls.pop()
        if ctl_acc:
            first = rows[0]
            cov = json.loads((products_dir / first["pid"] / "covariates.json").read_text("utf-8"))
            reads = int(cov["control"]["reads"])
            cdir = source_root / bios / CONTROL_TRACK
            _write_chroms(cdir / f"signal_DSF1_res{RES}",
                          products_dir / first["pid"] / "control_counts25.npz", np.uint32, n_bins)
            _write_json(cdir / f"signal_DSF1_res{RES}" / "metadata.json", {"depth": reads, "dsf": 1})
            _write_json(cdir / "file_metadata.json",
                        _file_metadata(CONTROL_TRACK, ctl_acc, int(first["read_length"]),
                                       first["run_type"]))
            csv_rows.append({"biosample_name": bios, "assay_name": CONTROL_TRACK,
                             "file_accession": ctl_acc, "assembly": ASSEMBLY,
                             "read_length": int(first["read_length"]),
                             "run_type": first["run_type"], "depth": reads})

    csv_path = source_root / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), restval="")  # extra key -> ValueError
        w.writeheader()
        w.writerows(csv_rows)
    prov_path = source_root / PROVENANCE_NAME
    _write_json(prov_path, {
        "corpus": "cf",
        "declared_output_type": SIGNAL_OUTPUT_TYPE,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"products": str(products_dir), "manifest_tsv": str(manifest_tsv),
                    "note": "t112 products: MACS2 -log10 p from the ENCODE pipeline's own signal "
                            "step, binned by tools/t112/bin25.py; no portal bigWig accession"},
        "tracks": tracks_prov,
    })
    return {"biosamples": sorted(by_bios), "n_csv_rows": len(csv_rows),
            "metadata_csv": str(csv_path), "signal_provenance": str(prov_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--chrsz", required=True, type=Path)
    ap.add_argument("--genome-json", type=Path, default=None,
                    help="also write the bare {chrom: len} chrom_sizes.json the store build reads")
    args = ap.parse_args(argv)
    s = export(args.products, args.manifest, args.source_root, args.chrsz, args.genome_json)
    print(f"{len(s['biosamples'])} biosamples, {s['n_csv_rows']} tracks -> {args.source_root}")
    print(f"metadata csv: {s['metadata_csv']}")
    print(f"signal provenance: {s['signal_provenance']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
