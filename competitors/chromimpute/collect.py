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

A D32 REGION GRID, AND WHY IT NEEDS `--regime` (2026-09-01)
------------------------------------------------------------
`Apply` predicts on whatever `chrominfo` it is handed, and under a `regions` regime the TRAINING
grid `prepare.write_chrominfo` writes is one declared pseudo-chromosome per Pilot Region — that is
`prepare.region_scope`, and it is what the §7 σ pass applies its predictors on. Those names are not
chromosomes of the store, so looking each one up in `genome.n_bins` failed outright and the
`eic_pilot` σ table could not be collected at all.

`--regime` is the map back: `region_scope` re-derives `(name, source chromosome, first bin, end
bin)` from the same BED, through the same hash gate, so a wig on a pseudo-chromosome is written at
its true offset on the store's absolute grid. **The bins no region covers are `NaN`, not zero.**
`bench.external.read_track_arrays` demands a full-length array, and a zero there is a confident
`-log10 p` of 0 at a locus nothing predicted; `competitors.sigma_pass` cuts the residual to
`scored_bins`, which under the same BED is a subset of the contained bins, so a correct run never
reads one — and a run whose scope slipped gets a NaN σ rather than a plausible one.
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

from prepare import load_json, read_targets, region_scope

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


def resolve_grid(names: Sequence[str], n_bins: Dict[str, int], regime_path: str | None
                 ) -> List[Tuple[str, str, int, int]]:
    """`[(wig chromosome, store chromosome, first bin, end bin), …]` for the names `Apply` wrote.

    A real chromosome is the whole of itself. Anything else must be a D32 region pseudo-chromosome,
    and `--regime` is the only thing that can say which chromosome it sits on and where — the name
    and the bin count in `chrominfo.train.txt` do not carry an offset, so the map is re-derived from
    the BED by `prepare.region_scope` rather than guessed from what is on disk.

    IT IS THE **SOURCE** REGIME, not a derived σ regime. `region_scope` reads the BED against
    `train_chroms`, and `tools/sigma_training_regime.py` empties that list (its training slice moves
    to `eval_chroms`), so the derived file cannot describe the grid its own run was applied on. The
    grid came from `prepare.py` under the source regime; so does the map back.

    A store chromosome named BOTH whole and by region is refused: the two would write the same
    `<chrom>.npz` from two different grids and the last one would win silently.
    """
    unknown = [c for c in names if c not in n_bins]
    if not unknown:
        return [(c, c, 0, int(n_bins[c])) for c in names]
    if not regime_path:
        raise SystemExit(
            f"[collect] {unknown[:3]} is not a chromosome of the store's `genome.n_bins` "
            f"({sorted(n_bins)[:5]}…). Under a D32 `regions` regime the training grid is one "
            f"declared pseudo-chromosome per region (prepare.region_scope), and only the regime can "
            f"say which chromosome each sits on and at which bin. Pass --regime.")
    regime = load_json(Path(regime_path))
    try:
        spans = {name: (chrom, first, stop)
                 for name, chrom, first, stop in region_scope(regime, regime_path)}
    except ValueError as exc:
        # `region_scope` cuts the BED to `train_chroms`, so an empty list here is almost always a
        # DERIVED regime being handed in: the sigma tool moves the training slice to `eval_chroms`
        # and leaves `train_chroms` empty. Say that rather than repeating its message.
        raise SystemExit(
            f"[collect] {Path(regime_path).name} cannot describe the grid {unknown[:3]}: {exc} "
            f"Hand this the SOURCE regime the training grid was written from — a derived sigma "
            f"regime declares no `train_chroms`, so its BED cuts to nothing.")
    if not spans:
        raise SystemExit(
            f"[collect] {Path(regime_path).name} declares no `regions`, so it cannot explain the "
            f"grid {unknown[:3]}. Hand this pass the regime the training grid was written from.")
    still = [c for c in unknown if c not in spans]
    if still:
        raise SystemExit(
            f"[collect] {still[:3]} is neither a chromosome of the store nor a region of "
            f"{Path(regime_path).name} ({len(spans)} region(s)). The wigs were applied on a grid "
            f"this regime does not describe.")
    out: List[Tuple[str, str, int, int]] = [
        (c, c, 0, int(n_bins[c])) if c in n_bins else (c, *spans[c]) for c in names]
    both = sorted({chrom for wig, chrom, _, _ in out if wig == chrom}
                  & {chrom for wig, chrom, _, _ in out if wig != chrom})
    if both:
        raise SystemExit(
            f"[collect] {both} is named both as a whole chromosome and by one of its regions. Both "
            f"write {both[0]}.npz, from two different grids; collect one grid at a time.")
    return out


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
    p.add_argument("--chroms", default="chr21",
                   help="comma list of the grid Apply wrote — store chromosomes, or the D32 region "
                        "pseudo-chromosomes of chrominfo.train.txt (then --regime is required)")
    p.add_argument("--regime", default=None,
                   help="the SOURCE regime whose `regions` BED names the pseudo-chromosomes in "
                        "--chroms — the one prepare.py wrote the training grid from, never a "
                        "derived sigma regime; each name is written back at its true offset on the "
                        "store's absolute grid, with NaN outside the regions")
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
    grid = resolve_grid(chroms, n_bins, args.regime)
    impute = Path(args.impute_dir)
    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    done: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for input_bios, target_bios, assay in read_targets(Path(args.targets)):
        name = apply_output_name(input_bios, assay)
        per_chrom: Dict[str, np.ndarray] = {}
        absent = [wig for wig, _, _, _ in grid if not (impute / f"{wig}_{name}.gz").exists()]
        if absent:
            missing.append(f"{track_dirname(input_bios, target_bios, assay)} [{','.join(absent)}]")
            continue
        for wig, chrom, first, stop in grid:
            arr = read_wig(impute / f"{wig}_{name}.gz", stop - first)
            if wig == chrom:
                per_chrom[chrom] = arr
                continue
            # NaN, not zero, outside the declared regions — see the module docstring. Allocated
            # once per store chromosome, then filled region by region.
            buf = per_chrom.get(chrom)
            if buf is None:
                buf = np.full(int(n_bins[chrom]), np.nan, dtype=np.float32)
                per_chrom[chrom] = buf
            buf[first:stop] = arr
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
    written = sorted({chrom for _, chrom, _, _ in grid})
    # The APPLY grid and the STORE chromosomes it landed on are two different lists under a region
    # grid, and a reader of this root has to be able to tell that the arrays are region-sparse.
    scope = (f"chroms={','.join(written)}" if written == chroms else
             f"chroms={','.join(written)} from the {len(grid)}-region grid {','.join(chroms[:3])}…, "
             f"NaN outside the regions")
    notes = args.notes or (f"Apply output, {len(done)} target(s), {scope}. "
                           f"Values are ChromImpute's own 2-decimal rounding of -log10 p.")
    write_manifest(pred_root, version=version, notes=notes)
    print(f"[collect] {len(done)} track(s), {len(missing)} missing -> {pred_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
