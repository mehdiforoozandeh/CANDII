"""collect — ChromImpute `Apply` output -> the `RIVALS_PLAN.md` §4.1 prediction root.

The return leg of `prepare.py`. `Apply` writes one gzipped fixed-step wig per chromosome:

    track type=wiggle_0 name=<sample>_<mark>_imputed     <- absent under -noprintbrowserheader
    fixedStep  chrom=chr21 start=1 step=25 span=25
    0.42
    ...

one value per 25 bp bin, rounded to two decimals by ChromImpute's own `NumberFormat`. We read those
values into the store's absolute bin grid and write

    <pred_root>/manifest.json
    <pred_root>/<input_bios>__<target_bios>__<assay>/<chrom>.npz     # signal_mu, float32

`signal_mu` is the only array. ChromImpute predicts a point in `-log10 p` and nothing else: no
spread, so no `signal_sigma` (§6.1 fits one later and hands it to the scorer as a table); no count
prediction, because B1b forbids inventing a read depth; no peak score, so the bench falls back to
coverage ranking and records `has_peak_head=False`.

The length check is not defensive coding. A wig one bin longer than the grid is the signature of a
`chrominfo.txt` that declared the true chromosome length instead of `n_bins*25` — see
`prepare.write_chrominfo` — and it is the one mistake that would otherwise slide through as an
off-by-one on every score in the panel.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from prepare import load_json, read_targets

METHOD = "ChromImpute"
SEP = "__"


def track_dirname(input_bios: str, target_bios: str, assay: str) -> str:
    """§4.1 — the bench `track_key` with `|` swapped for the filesystem-safe `__`."""
    return f"{input_bios}{SEP}{target_bios}{SEP}{assay}"


def apply_output_name(sample: str, mark: str) -> str:
    """The `-o` we hand `Apply`. Ours, not ChromImpute's default.

    The default is `impute_<sample>_<mark>.wig`, which cannot be split back into sample and mark
    when both contain `_` — and ours do. Passing `-o` keeps the parse on our side of the boundary.
    """
    return f"impute.{sample}.{mark}.wig"


def read_wig(path: Path, expect: int) -> np.ndarray:
    """One gzipped fixed-step wig -> `(expect,)` float32. Header lines are skipped, not counted.

    `Apply` writes one or two header lines depending on `-noprintbrowserheader`, so the reader
    keys on the content ("track"/"browser"/"fixedStep"/"variableStep"/"#") rather than on a line
    count. A `variableStep` file is refused: `Apply` never writes one, and silently treating a
    sparse file as dense is exactly the class of error this module exists to make loud.
    """
    vals: List[float] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            head = line[:12].lower()
            if head.startswith(("track", "browser", "#", "fixedstep")):
                continue
            if head.startswith("variablestep"):
                raise ValueError(
                    f"{path}: variableStep wig. Apply writes fixedStep; a sparse file read as "
                    f"dense would shift every bin.")
            line = line.strip()
            if line:
                vals.append(float(line))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size != expect:
        raise ValueError(
            f"{path}: {arr.size} bins, the store's grid wants {expect}. A one-bin excess means "
            f"chrominfo.txt declared the true chromosome length instead of n_bins*25 "
            f"(prepare.write_chrominfo).")
    return arr


def write_track(pred_root: Path, input_bios: str, target_bios: str, assay: str,
                per_chrom: Dict[str, np.ndarray]) -> Path:
    d = pred_root / track_dirname(input_bios, target_bios, assay)
    d.mkdir(parents=True, exist_ok=True)
    for chrom, arr in per_chrom.items():
        np.savez_compressed(d / f"{chrom}.npz", signal_mu=arr.astype(np.float32))
    return d


def write_manifest(pred_root: Path, *, version: str, notes: str) -> Path:
    path = pred_root / "manifest.json"
    path.write_text(json.dumps({
        "method": METHOD,
        "version": version,
        "generated_by": "competitors/chromimpute/collect.py",
        "date": _dt.date.today().isoformat(),
        "arms": ["pval"],
        "notes": notes,
    }, indent=2) + "\n", encoding="utf-8")
    return path


def jar_version(jar: Path) -> str:
    try:
        out = subprocess.run(["java", "-jar", str(jar), "Version"], capture_output=True,
                             text=True, timeout=120).stdout
        for tok in out.split():
            if tok[:1].isdigit() and "." in tok:
                return tok
    except Exception:  # pragma: no cover - version is provenance, never control flow
        pass
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect.py",
        description="ChromImpute Apply wigs -> RIVALS_PLAN.md §4.1 prediction root.")
    p.add_argument("--store", required=True, help="corpus root, for the manifest's bin table")
    p.add_argument("--targets", required=True, help="targets.tsv written by prepare.py")
    p.add_argument("--impute-dir", required=True, help="Apply's OUTPUTIMPUTEDIR")
    p.add_argument("--pred-root", required=True, help="§4.1 prediction root to write")
    p.add_argument("--chroms", default="chr21", help="comma list of chromosomes to collect")
    p.add_argument("--jar", default=None, help="ChromImpute.jar, for the manifest's version field")
    p.add_argument("--notes", default="", help="free text copied into manifest.json")
    p.add_argument("--allow-missing", action="store_true",
                   help="skip targets whose wigs are absent instead of failing")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(Path(args.store) / "manifest.json")
    n_bins = dict(manifest["genome"]["n_bins"])
    chroms = [c.strip() for c in args.chroms.split(",")]
    impute = Path(args.impute_dir)
    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    done: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for input_bios, target_bios, assay in read_targets(Path(args.targets)):
        name = apply_output_name(input_bios, assay)
        per_chrom: Dict[str, np.ndarray] = {}
        absent = [c for c in chroms if not (impute / f"{c}_{name}.gz").exists()]
        if absent:
            missing.append(f"{track_dirname(input_bios, target_bios, assay)} [{','.join(absent)}]")
            continue
        for chrom in chroms:
            per_chrom[chrom] = read_wig(impute / f"{chrom}_{name}.gz", int(n_bins[chrom]))
        write_track(pred_root, input_bios, target_bios, assay, per_chrom)
        done.append((input_bios, target_bios, assay))
        print(f"[collect] {track_dirname(input_bios, target_bios, assay)}: "
              f"{len(per_chrom)} chromosome(s)")

    if missing and not args.allow_missing:
        raise SystemExit(
            "[collect] no Apply output for:\n  " + "\n  ".join(missing) +
            "\n--allow-missing writes the partial root anyway; bench.external will still refuse "
            "it without --allow-missing of its own.")
    version = jar_version(Path(args.jar)) if args.jar else "unknown"
    notes = args.notes or (f"Apply output, {len(done)} target(s), chroms={','.join(chroms)}. "
                           f"Values are ChromImpute's own 2-decimal rounding of -log10 p.")
    write_manifest(pred_root, version=version, notes=notes)
    print(f"[collect] {len(done)} track(s), {len(missing)} missing -> {pred_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
