"""`python -m candi.store build-biosample | build-genome | build-manifest | verify`.

Thin argparse over `writer.py` and `manifest.py`. Every subcommand is idempotent and writes
through a temp file, so a SLURM array task that dies mid-write leaves no half-file behind.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from candi.store import genome as G
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
    # D25/D29 — flags, not an edit, so a rebuild at a different codec is a submit line. They default
    # to the shipped codec; t25's job script passes both EXPLICITLY anyway, so the choice lands in
    # all 455 sbatch logs instead of being a default someone has to go and measure afterwards.
    b.add_argument("--pval-scale", type=int, default=L.PVAL_SCALE,
                   help=f"integer codes per unit of the transformed -log10 p (default "
                        f"{L.PVAL_SCALE}). With --pval-transform arcsinh this is a FLAT relative "
                        f"precision of 1/(2*scale) = {1 / (2 * L.PVAL_SCALE):.1e}.")
    b.add_argument("--pval-transform", default=L.PVAL_TRANSFORM, choices=list(L.PVAL_TRANSFORMS),
                   help=f"pval storage codec (default {L.PVAL_TRANSFORM}). 'linear' is the pre-D25 "
                        f"codec whose ceiling is 65535/scale — at scale 100 that is 655.35, which "
                        f"truncated 62 of the EIC corpus's 363 tracks. Recorded in the file's "
                        f"'{L.ATTR_TRANSFORM}' root attr; the reader inverts whatever it finds.")
    b.add_argument("--overwrite", action="store_true")

    g = sub.add_parser("build-genome", help="FASTA -> dna.h5 ; dna.h5 + blacklist -> mask.h5 (t7)")
    g.add_argument("--store-root", required=True, type=Path,
                   help="CANDI_STORE root; the layer is written to <store-root>/genome/")
    g.add_argument("--fasta", type=Path, default=None, help="uncompressed hg38 FASTA (D10)")
    g.add_argument("--blacklist", type=Path, default=None,
                   help="ENCODE hg38 blacklist v2 BED from t4 (D11)")
    g.add_argument("--chrom-sizes", type=Path, default=None,
                   help="default: <store-root>/genome/chrom_sizes.json")
    g.add_argument("--chroms", type=_csv_list, default=None,
                   help="explicit chromosome list; omit to take every one in chrom_sizes")
    g.add_argument("--resolution", type=int, default=L.DEFAULT_RESOLUTION)
    g.add_argument("--build", default=G.GENOME_BUILD)
    g.add_argument("--fasta-sha256", default=None,
                   help="verify the FASTA against this before building (D10 build-mismatch check)")
    g.add_argument("--only", choices=["both", "dna", "mask"], default="both")
    g.add_argument("--min-valid-frac", type=float, default=G.DEFAULT_MIN_VALID_FRAC,
                   help="D12 eligibility threshold used for the report (not baked into mask.h5)")
    g.add_argument("--context-bins", type=_csv_list, default=["768", "6144"],
                   help="window sizes to count eligible windows for in the report")
    g.add_argument("--report", type=Path, default=None,
                   help="write the per-chromosome coverage + eligible-window JSON here")
    g.add_argument("--overwrite", action="store_true")

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
            pval_scale=args.pval_scale,
            pval_transform=args.pval_transform,
        )
        for kind, info in summary["kinds"].items():
            codec = (f", codec={info['pval_transform']}x{info['pval_scale']}"
                     f" (ceiling {L.pval_max_encodable(info['pval_scale'], info['pval_transform']):.3g}"
                     f", clipped {max(info['pval_clip_frac'] or [0.0]):.3g})"
                     if kind == "pval" else "")
            print(
                f"{name}/{kind}: {len(info['tracks'])} tracks x {len(info['chroms'])} chroms, "
                f"{info['dtype']}, control_col={info['control_col']}{codec}, "
                f"{info['bytes'] / 1e6:.1f} MB -> {info['path']}",
                flush=True,
            )
    return 0


def _cmd_build_genome(args) -> int:
    sizes_path = args.chrom_sizes or (L.genome_dir(args.store_root) / "chrom_sizes.json")
    if not Path(sizes_path).is_file():
        raise SystemExit(
            f"--chrom-sizes is required (and {sizes_path} does not exist). n_bins = "
            f"floor(len/resolution) cannot be derived without it."
        )
    chrom_sizes = G.load_genome_chrom_sizes(sizes_path)
    if args.only in ("both", "dna") and args.fasta is None:
        raise SystemExit("--fasta is required to build dna.h5")
    if args.only in ("both", "mask") and args.blacklist is None:
        raise SystemExit("--blacklist is required to build mask.h5 (t4 writes it)")

    def say(msg: str) -> None:
        print(f"[build-genome] {msg}", flush=True)

    result = G.build_genome(
        args.store_root,
        args.fasta,
        args.blacklist,
        chrom_sizes=chrom_sizes,
        chroms=args.chroms,
        resolution=args.resolution,
        build=args.build,
        fasta_sha256=args.fasta_sha256,
        what=args.only,
        overwrite=args.overwrite,
        progress=say,
    )
    if "dna" in result:
        d = result["dna"]
        say(f"dna.h5: {d['bytes'] / 1e9:.3f} GB, {len(d['chroms'])} chroms, "
            f"fasta_sha256={d['fasta_sha256']}, iupac_folded={d['iupac_counts']}")
    if "mask" in result:
        m = result["mask"]
        say(f"mask.h5: {m['bytes'] / 1e6:.1f} MB, {m['blacklist_intervals']} blacklist intervals "
            f"({m['blacklist_bp']} bp), genome valid_frac={m['valid_frac_genome']:.6f}")

    problems = G.verify_genome(L.genome_dir(args.store_root), chrom_sizes=chrom_sizes)
    for p in problems:
        say(f"PROBLEM: {p}")

    report = G.genome_report(
        L.genome_dir(args.store_root),
        context_bins=[int(x) for x in args.context_bins],
        min_valid_frac=args.min_valid_frac,
    )
    report["build_summary"] = result
    for chrom, rec in report["per_chrom"].items():
        elig = " ".join(
            f"L={cb}:tiled={v['tiled']},every={v['every_start']}"
            for cb, v in rec["eligible"].items()
        )
        say(f"{chrom}: n_bins={rec['n_bins']} valid={rec['n_valid']} "
            f"({rec['valid_frac']:.6f}) {elig}")
    gtot = report["genome"]
    say(f"GENOME: n_bins={gtot['n_bins']} valid={gtot['n_valid']} "
        f"({gtot['valid_frac']:.6f})")
    for cb, v in gtot["eligible"].items():
        say(f"GENOME eligible L={cb}: tiled={v['tiled']} every_start={v['every_start']}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        say(f"report -> {args.report}")
    return 1 if problems else 0


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
