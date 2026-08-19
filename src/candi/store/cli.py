"""`python -m candi.store build-biosample | build-genome | build-manifest | verify`.

Thin argparse over `writer.py` and `manifest.py`. Every subcommand is idempotent and writes
through a temp file, so a SLURM array task that dies mid-write leaves no half-file behind.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from candi.store import layout as L
from candi.store.manifest import build_manifest, verify_store, write_manifest
from candi.store.writer import build_biosample, discover_biosamples

__all__ = ["main", "build_parser"]


def _csv_list(s: str) -> list:
    return [x for x in (part.strip() for part in s.split(",")) if x]


def _chrom_sizes_arg(args) -> dict:
    """`--chrom-sizes`, or `genome/chrom_sizes.json` next to the corpus store."""
    if args.chrom_sizes:
        return L.load_chrom_sizes(args.chrom_sizes)
    guess = L.corpus_genome_dir(args.corpus_root) / "chrom_sizes.json"
    if guess.is_file():
        return L.load_chrom_sizes(guess)
    raise SystemExit(
        f"--chrom-sizes is required (and {guess} does not exist). n_bins = floor(len/resolution) "
        f"cannot be derived without it."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m candi.store", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-biosample", help="npz tree -> counts/peaks/pval h5 for one biosample")
    b.add_argument("--source-root", required=True, type=Path, help="the ENCODE-style npz tree")
    b.add_argument("--corpus-root", required=True, type=Path,
                   help="CANDI_STORE/<corpus>, e.g. .../CANDI_STORE/eic")
    b.add_argument("--biosample", action="append", default=None,
                   help="repeatable; omit to build every biosample under --source-root")
    b.add_argument("--kinds", type=_csv_list, default=["counts", "peaks"],
                   help=f"subset of {','.join(L.KINDS)} (default counts,peaks)")
    b.add_argument("--chroms", type=_csv_list, default=None,
                   help="explicit chromosome list; omit to take every one the source has")
    b.add_argument("--chrom-sizes", type=Path, default=None)
    b.add_argument("--resolution", type=int, default=L.DEFAULT_RESOLUTION)
    b.add_argument("--dsf", type=int, default=1, help="source signal_DSF{d} level; D6 says 1")
    b.add_argument("--counts-dtype", default=None, choices=[L.dtype_name(d) for d in L.COUNTS_DTYPES],
                   help="skip the D7 max scan and force this dtype")
    b.add_argument("--pval-nan", default="error", choices=["error", "zero"],
                   help="what to do with a non-finite -log10 p (default: refuse)")
    b.add_argument("--overwrite", action="store_true")

    g = sub.add_parser("build-genome", help="FASTA -> dna.h5 ; FASTA + blacklist -> mask.h5 (t7)")
    g.add_argument("--store-root", type=Path, default=None)

    m = sub.add_parser("build-manifest", help="metadata CSVs + h5 attrs -> manifest.json")
    m.add_argument("--corpus-root", required=True, type=Path)
    m.add_argument("--corpus", required=True, help="corpus id written into the manifest, e.g. eic")
    m.add_argument("--metadata-csv", action="append", required=True, type=Path,
                   help="repeatable; add t5's control_metadata.csv as one more of these")
    m.add_argument("--source-root", type=Path, default=None,
                   help="npz tree; enables the file_metadata.json cross-check (D20)")
    m.add_argument("--genome", type=Path, default=None,
                   help="genome dir (default: the sibling genome/ of --corpus-root)")
    m.add_argument("--no-strict", action="store_true",
                   help="downgrade a CSV/json conflict to a warning — for triage, not for a build")

    v = sub.add_parser("verify", help="structural check of a built store against its manifest")
    v.add_argument("--corpus-root", required=True, type=Path)
    return p


def _cmd_build_biosample(args) -> int:
    chrom_sizes = _chrom_sizes_arg(args)
    names = args.biosample or discover_biosamples(args.source_root)
    for name in names:
        summary = build_biosample(
            args.source_root,
            args.corpus_root,
            name,
            chrom_sizes=chrom_sizes,
            kinds=args.kinds,
            chroms=args.chroms,
            resolution=args.resolution,
            dsf=args.dsf,
            counts_dtype=args.counts_dtype,
            nan_policy=args.pval_nan,
            overwrite=args.overwrite,
        )
        for kind, info in summary["kinds"].items():
            print(
                f"{name}/{kind}: {len(info['tracks'])} tracks x {len(info['chroms'])} chroms, "
                f"{info['dtype']}, control_col={info['control_col']}, "
                f"{info['bytes'] / 1e6:.1f} MB -> {info['path']}",
                flush=True,
            )
    return 0


def _cmd_build_genome(args) -> int:
    raise NotImplementedError(
        "build-genome is task t7 (`src/candi/store/genome.py`): FASTA -> dna.h5 (D10) and "
        "FASTA + ENCODE hg38 blacklist v2 -> mask.h5 (D11, D12). It needs the real blacklist "
        "from t4 — the in-repo data/hg38_blacklist_v2.bed is 8 bytes containing '# Empty'. "
        "See STORE_PLAN.md §6/t7."
    )


def _cmd_build_manifest(args) -> int:
    manifest = build_manifest(
        args.corpus_root,
        args.corpus,
        args.metadata_csv,
        source_root=args.source_root,
        genome=args.genome,
        strict=not args.no_strict,
    )
    out = write_manifest(args.corpus_root, manifest)
    n_tracks = sum(len(b["tracks"]) for b in manifest["biosamples"].values())
    print(
        f"{out}: {len(manifest['biosamples'])} biosamples, {n_tracks} tracks, "
        f"{len(manifest['assay_vocabulary'])} assays, {len(manifest['metadata_gaps'])} metadata gaps",
        flush=True,
    )
    return 0


def _cmd_verify(args) -> int:
    problems = verify_store(args.corpus_root)
    if not problems:
        print(f"{args.corpus_root}: OK", flush=True)
        return 0
    print(f"{args.corpus_root}: {len(problems)} problem(s)", flush=True)
    for p in problems:
        print(f"  {p}", flush=True)
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "build-biosample": _cmd_build_biosample,
        "build-genome": _cmd_build_genome,
        "build-manifest": _cmd_build_manifest,
        "verify": _cmd_verify,
    }[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
