#!/usr/bin/env python3
"""Convert 2019 ENCODE Imputation Challenge bigwigs onto our grid, in the §4.1 on-disk layout.

    # the blind truth, as a TRUTH root for `candi.bench.external --truth-root`
    python tools/challenge_bigwigs.py truth-root \
        --bigwig-dir /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/blind_truth \
        --bridge .../output/bridge/eic_bridge.csv \
        --regime configs/regime.eic_19.B_.json \
        --chrom-sizes .../CANDI_STORE/eic/genome/chrom_sizes.json \
        --chroms chr20,chr21,chr22 --out /project/.../t81_truth_challenge/B_

    # one 2019 entrant's submission, as a PREDICTION root for the same scorer
    python tools/challenge_bigwigs.py pred-root \
        --bigwig-dir .../t54_submissions_round2/Guacamole \
        --bridge ... --regime ... --chrom-sizes ... --chroms chr20,chr21,chr22 \
        --method Guacamole --out /project/.../t81_pred_B/anchor/Guacamole/B_

`plan/BENCHMARK_DESIGN.md` §4.1 says the anchor block is 25 entrants placed beside our methods on
the truth the 2019 leaderboard actually used. Every piece of that existed except the path from a
bigwig to something `candi.bench.external` can open. This is that path, and it is deliberately the
whole of it: nothing here scores anything.

**Three decisions worth stating, because each one could be made differently and be wrong.**

*The grid is OURS, not the challenge's.* `competitors/entrants/vendor/fir_tracks.py` bins the same
files at `ceil(chrom_len / 25)` with a special case for the partial last window, because that is
what produced the published numbers. This tool bins at `floor(chrom_len / 25)` (D13,
`store.layout.n_bins_for`) with no special case — under a floor every bin is a full 25 bp window,
so the special case has nothing to fix. A converted track is therefore one bin shorter than the
2019 array and is on exactly the grid `external.read_track_arrays` asserts. The two are not
interchangeable and a number from one must never be quoted beside a number from the other.

*The ID join is the bridge CSV, never string surgery.* Our biosample names are opaque ids (D16) and
the challenge's are `C##M##`. `eic_bridge.csv` carries both, joined on ENCODE experiment accession,
and this tool reads `biosample_dir` + `assay_name` -> `filename` out of it. A missing row is an
error naming the track, not a guess.

*The declared track list is the REGIME's.* Which tracks a root must contain is decided by the same
`_expected` the scorer uses, so a root this tool wrote and the run that scores it cannot disagree
about what a complete panel is.

pyBigWig is imported LAZILY, inside the reader factory. It is present in the Fir venv and absent on
the laptop, and the binning and the bridge join are the parts worth testing — so both take an
injected `reader(chrom, start, end) -> np.ndarray` of per-base values and this module imports
cleanly with no pyBigWig anywhere.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candi.bench.external import track_dirname                        # noqa: E402
from candi.bench.harness import open_source                           # noqa: E402
from candi.store import layout as L                                   # noqa: E402

#: The bin rule, verbatim into every manifest this tool writes. A converted array is worth nothing
#: without it: the same bigwig under `ceil` and under `floor` gives two different vectors.
BIN_RULE = ("mean of each {res} bp bin over [i*{res}, (i+1)*{res}), NaN->0, "
            "length floor(chrom_len/{res}), anchored at position 0")

#: The bridge columns this tool reads. Named so a bridge that moves is an error here rather than a
#: silent mis-join downstream. Read off the file on Fir, 2026-09-01.
BRIDGE_BIOSAMPLE, BRIDGE_ASSAY, BRIDGE_FILE = "biosample_dir", "assay_name", "filename"

#: Suffixes a challenge track may carry. Synapse ships `.bigwig`; the other two are what the same
#: files are called elsewhere, and trying them costs one `stat` each.
SUFFIXES: Tuple[str, ...] = (".bigwig", ".bigWig", ".bw")

#: `pred-root` writes this as the manifest's `version`. Every anchor row is one 2019 submission,
#: and a version string that named this tool's own release would be about the converter rather than
#: about the thing being scored.
ENTRANT_VERSION = "2019-submission"

Reader = Callable[[str, int, int], np.ndarray]


class ConvertError(ValueError):
    """A conversion that cannot be made honestly. Always names the offending track or column."""


# ---------------------------------------------------------------------------
# the ID join
# ---------------------------------------------------------------------------

def read_bridge(path: Path | str) -> Tuple[Dict[Tuple[str, str], str], str]:
    """`({(biosample, assay): challenge filename}, sha256 of the CSV)`.

    `eic_bridge.csv` joins our corpus to the challenge's `C##M##` ids on ENCODE experiment
    accession, 1:1 on all 363 experiments. It is the authority and this reads it whole: a duplicate
    `(biosample, assay)` is refused rather than resolved, because whichever row won would be a
    silent choice between two experiments.
    """
    p = Path(path)
    raw = p.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if not rows:
        raise ConvertError(f"{p} has no rows")
    for col in (BRIDGE_BIOSAMPLE, BRIDGE_ASSAY, BRIDGE_FILE):
        if col not in rows[0]:
            raise ConvertError(
                f"{p} has no `{col}` column (it has {sorted(rows[0])}). This tool joins on "
                f"`{BRIDGE_BIOSAMPLE}` + `{BRIDGE_ASSAY}` -> `{BRIDGE_FILE}` and does no name "
                f"munging, so a renamed column is a stop rather than a fallback.")
    out: Dict[Tuple[str, str], str] = {}
    for row in rows:
        key = (str(row[BRIDGE_BIOSAMPLE]), str(row[BRIDGE_ASSAY]))
        name = str(row[BRIDGE_FILE])
        if key in out and out[key] != name:
            raise ConvertError(
                f"{p}: {key} appears twice, as {out[key]} and {name}. One (biosample, assay) is "
                f"one experiment here; picking either would be a silent choice.")
        out[key] = name
    return out, hashlib.sha256(raw).hexdigest()


def bigwig_path(bigwig_dir: Path, stem: str) -> Optional[Path]:
    """`<dir>/<stem><suffix>` for the first suffix that exists, or `None`."""
    for suf in SUFFIXES:
        p = bigwig_dir / f"{stem}{suf}"
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# the bin rule
# ---------------------------------------------------------------------------

def bin_track(reader: Reader, chrom: str, chrom_len: int, *, resolution: int = 25,
              chunk_bins: int = 400_000) -> np.ndarray:
    """One chromosome, binned onto our grid: `float32`, length `floor(chrom_len / resolution)`.

    Bin `i` is the MEAN of the per-base values on `[i*resolution, (i+1)*resolution)` with NaN read
    as 0. Anchored at position 0 and never at the first covered base, because index `i` has to be
    the bin at `i * resolution` bp for anything downstream to line up (§4.1's length assertion is
    the other half of the same rule).

    NaN -> 0 rather than nanmean: a bigwig has no value where the signal is zero, so a bin that is
    half uncovered is half zero, and a nanmean would report the covered half's height as the whole
    bin's. That is what the 2019 scorer did and what these tracks must be compared under.

    Read in chunks so a whole chromosome never lands in memory at once — chr1 is 248 Mbp, which is
    2 GB as float64.
    """
    n = L.n_bins_for(int(chrom_len), resolution)
    out = np.empty(n, dtype=np.float32)
    step = max(1, int(chunk_bins))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        raw = np.asarray(reader(chrom, lo * resolution, hi * resolution), dtype=np.float64)
        want = (hi - lo) * resolution
        if raw.ndim != 1 or raw.shape[0] != want:
            raise ConvertError(
                f"{chrom}[{lo * resolution}:{hi * resolution}]: the reader returned {raw.shape} "
                f"and {want} per-base values were asked for. The bin rule is a mean over a FIXED "
                f"window, so a short read would shift every later bin.")
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out[lo:hi] = raw.reshape(hi - lo, resolution).mean(axis=1).astype(np.float32)
    return out


def pybigwig_reader(path: Path | str) -> Reader:
    """A `reader(chrom, start, end)` over one bigwig. **The only place pyBigWig is imported.**

    It is in the Fir venv and not on the laptop, and it is deliberately not in any requirements
    file here: a bigwig never reaches this machine, so a laptop that cannot open one is correct.
    """
    import pyBigWig                                       # noqa: PLC0415 — lazy on purpose

    bw = pyBigWig.open(str(path))

    def reader(chrom: str, start: int, end: int) -> np.ndarray:
        return bw.values(chrom, int(start), int(end), numpy=True)

    reader.close = bw.close                                            # type: ignore[attr-defined]
    return reader


# ---------------------------------------------------------------------------
# the roots
# ---------------------------------------------------------------------------

def declared_tracks(regime: Path | str, chroms: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """`{track dirname: (target biosample, assay)}` — the regime's declared tracks, and only those.

    Derived through `open_source`, so the list this tool writes to disk and the list the scorer
    demands are the same list, produced by the same code. The TARGET cell is what a challenge
    track names: the input cell is the prompt and has no bigwig of its own here.
    """
    src = open_source(store=regime, chroms=tuple(chroms))
    try:
        out: Dict[str, Tuple[str, str]] = {}
        for pair in src.pairs("impute"):
            for a in src.targets(pair, "impute"):
                assay = src.assays[a]
                out[track_dirname(pair, assay)] = (pair.target_biosample, assay)
        return out
    finally:
        src.close()


def build_root(out_root: Path | str, *, bigwig_dir: Path | str, bridge: Path | str,
               regime: Path | str, chrom_sizes: Path | str, chroms: Sequence[str],
               kind: str, method: Optional[str] = None, allow_missing: bool = False,
               resolution: int = 25, open_reader: Callable[[Path], Reader] = pybigwig_reader,
               notes: str = "", progress: bool = True) -> Dict[str, object]:
    """Write `<out>/manifest.json` + one `<dirname>/<chrom>.npz` per declared track. Returns it.

    `kind` is `"truth"` or `"pred"` and decides only the manifest — the arrays are the same bytes
    under the same rule either way, which is the point: a truth root and a prediction root scored
    against each other are two reads of one converter.
    """
    if kind not in ("truth", "pred"):
        raise ConvertError(f"kind must be 'truth' or 'pred', got {kind!r}")
    if kind == "pred" and not method:
        raise ConvertError("--method names the entrant and is what `provenance.method` becomes")
    bwdir = Path(bigwig_dir)
    if not bwdir.is_dir():
        raise ConvertError(f"--bigwig-dir {bwdir} is not a directory")
    sizes = L.load_chrom_sizes(chrom_sizes)
    chroms = [str(c) for c in chroms]
    unknown = [c for c in chroms if c not in sizes]
    if unknown:
        raise ConvertError(f"{unknown} are not in {chrom_sizes}; the grid comes from that file")

    table, bridge_sha = read_bridge(bridge)
    want = declared_tracks(regime, chroms)

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    tracks: Dict[str, str] = {}
    skipped: List[Dict[str, str]] = []
    for i, (dirname, (biosample, assay)) in enumerate(sorted(want.items())):
        stem = table.get((biosample, assay))
        src = None if stem is None else bigwig_path(bwdir, stem)
        if stem is None or src is None:
            why = (f"no row in the bridge for ({biosample}, {assay})" if stem is None else
                   f"{stem} is not in {bwdir} under any of {list(SUFFIXES)}")
            if not allow_missing:
                raise ConvertError(
                    f"{dirname}: {why}. A converted root with a hole in it reads as a whole panel "
                    f"downstream (D2). Pass --allow-missing to record it in the manifest's "
                    f"`skipped_tracks` and convert the rest — the score run then needs "
                    f"--allow-missing too.")
            skipped.append({"track": dirname, "reason": why})
            continue
        d = root / dirname
        d.mkdir(exist_ok=True)
        reader = open_reader(src)
        try:
            for c in chroms:
                np.savez(d / f"{c}.npz",
                         signal_mu=bin_track(reader, c, int(sizes[c]), resolution=resolution))
        finally:
            close = getattr(reader, "close", None)
            if close is not None:
                close()
        tracks[dirname] = src.name
        if progress:
            print(f"[challenge_bigwigs] {i + 1}/{len(want)} {dirname} <- {src.name}", flush=True)

    common = {
        "source_dir": str(bwdir),
        "bridge_sha256": bridge_sha,
        "declared_tracks": len(want),
        "chroms": list(chroms),
        "bin_rule": BIN_RULE.format(res=resolution),
        "tracks": tracks,
        "skipped_tracks": skipped,
        "generated_by": "tools/challenge_bigwigs.py",
        "date": _dt.date.today().isoformat(),
    }
    manifest: Dict[str, object] = (
        {"kind": "truth", "truth": "challenge", **common} if kind == "truth" else
        {"method": str(method), "version": ENTRANT_VERSION, "arms": ["pval"],
         "lineage": "entrant", "notes": notes, **common})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bigwig-dir", required=True,
                   help="directory of `C##M##.bigwig` — the blind-truth folder, or one entrant's")
    p.add_argument("--bridge", required=True, help="eic_bridge.csv (the ID join, D16)")
    p.add_argument("--regime", required=True,
                   help="the regime whose DECLARED eval_pairs say which tracks this root must hold")
    p.add_argument("--chrom-sizes", required=True,
                   help="the STORE's genome/chrom_sizes.json — the grid is ours, not the 2019 one")
    p.add_argument("--chroms", required=True, help="comma-separated, e.g. chr20,chr21,chr22")
    p.add_argument("--out", required=True, help="root to write")
    p.add_argument("--resolution", type=int, default=25)
    p.add_argument("--allow-missing", action="store_true",
                   help="convert a panel with a hole in it and record it in `skipped_tracks`. "
                        "UIOWA_Michaelson never submitted C38M18, so its root needs this — and so "
                        "does the run that scores it.")
    p.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/challenge_bigwigs.py",
        description="Bin 2019 challenge bigwigs onto the store's 25 bp grid, in the §4.1 layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("truth-root", help="the challenge's blind truth, as a --truth-root")
    _common(t)
    d = sub.add_parser("pred-root", help="one entrant's submission, as a --pred root")
    _common(d)
    d.add_argument("--method", required=True, help="the entrant slug; becomes provenance.method")
    d.add_argument("--notes", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    manifest = build_root(
        a.out, bigwig_dir=a.bigwig_dir, bridge=a.bridge, regime=a.regime,
        chrom_sizes=a.chrom_sizes, chroms=[c.strip() for c in a.chroms.split(",")],
        kind="truth" if a.verb == "truth-root" else "pred",
        method=getattr(a, "method", None), notes=getattr(a, "notes", ""),
        allow_missing=a.allow_missing, resolution=a.resolution, progress=not a.quiet)
    if not a.quiet:
        print(f"[challenge_bigwigs] wrote {a.out}: {len(manifest['tracks'])} of "
              f"{manifest['declared_tracks']} declared tracks, "
              f"{len(manifest['skipped_tracks'])} skipped", flush=True)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
