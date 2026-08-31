"""dump — write a CANDI checkpoint's predictions to the §4.1 external contract.

    python -m candi.bench.dump --store regime.json --ckpt model.pt --arch-from run.json \
        --out pred_root --method CANDI

`candi.bench` scores a checkpoint in memory. `candi.bench.external` scores a prediction root
somebody else wrote to disk. This module is the missing producer: the same load and the same
forward as `candi.bench` (`cli._build_model`, `harness.stream_tracks`), written out as the
directory a rival's exporter would have written, so the frozen instrument can score CANDI the
way it scores everyone else.

Nothing here computes a metric. Lengths are the store's absolute 25 bp grid, because that is
what `external.read_track_arrays` will refuse to pad or trim. `peak_score` is emitted only when
the checkpoint actually has a peak head — `stream_tracks` falls back to the NB mean otherwise,
and fabricating a ranking from `mu` would label the row as a real peak head.

`signal_mu` / `signal_sigma` are written already in `-log10 p` (`RIVALS_PLAN.md` §4.1). The
head may have been trained in a transformed space; this module inverts with the same
`signal_target_transform` `candi.bench` uses, so the external path does not have to know about
CANDI's training space.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from candi.bench.distributional import invert_signal_prediction
from candi.bench.external import _expected, track_dirname
from candi.bench.harness import EvalSource, TrackRecord, cross_cell, open_source, stream_tracks
from candi.precision import no_autocast

__all__ = ["DumpError", "write_manifest", "arrays_for_chrom", "dump_predictions",
           "build_parser", "main"]


class DumpError(ValueError):
    """A dump that cannot meet the §4.1 contract. Always names the offending track or regime."""


# ---------------------------------------------------------------------------
# the on-disk contract (§4.1)
# ---------------------------------------------------------------------------

def write_manifest(root: Path, *, method: str, declared_tracks: Sequence[str],
                   arms: Sequence[str], version: str = "", notes: str = "") -> None:
    """`<pred_root>/manifest.json` — copied verbatim into every score file's provenance."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "method": method,
        "version": version,
        "generated_by": "candi.bench.dump",
        "date": datetime.date.today().isoformat(),
        "arms": list(arms),
        "declared_tracks": list(declared_tracks),
        "notes": notes,
    }, indent=2) + "\n", encoding="utf-8")


def arrays_for_chrom(rec: TrackRecord, chrom: str, n_bins: int, *,
                     signal_target_transform: str = "none") -> Dict[str, np.ndarray]:
    """The §4.1 arrays for one chromosome of one track, length-checked against the grid.

    Count parameters always; the Gaussian head when the checkpoint has one; `peak_score` only
    when `rec.has_peak_head` is true. The NB-mean fallback `stream_tracks` writes into
    `peak_score` is NOT emitted — that would make `bench.external` treat a coverage ranking as a
    real peak head and emit a `bernoulli_nll` of an unbounded count.
    """
    def _vec(name: str, arr) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if out.ndim != 1 or out.shape[0] != n_bins:
            raise DumpError(
                f"{track_dirname(rec.pair, rec.assay)}/{chrom}.npz: `{name}` is {out.shape} and "
                f"{chrom} is {n_bins} bins on the absolute 25 bp grid. The dump does not "
                f"truncate or pad — bin i must be the bin starting at i*25 bp.")
        return out

    payload: Dict[str, np.ndarray] = {
        "mu": _vec("mu", rec.mu[chrom]),
        "n": _vec("n", rec.n[chrom]),
    }
    if rec.has_pval:
        mu, sigma = invert_signal_prediction(
            rec.signal_mu[chrom], rec.signal_sigma[chrom], signal_target_transform)
        payload["signal_mu"] = _vec("signal_mu", mu)
        payload["signal_sigma"] = _vec("signal_sigma", sigma)
    if rec.has_peak_head:
        payload["peak_score"] = _vec("peak_score", rec.peak_score[chrom])
    return payload


def _write_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def dump_predictions(model, source: EvalSource, root: Path | str, *, method: str,
                     signal_target_transform: str = "none", batch_windows: int = 4,
                     version: str = "", notes: str = "", progress: bool = False,
                     device: Any = "cpu") -> Path:
    """Stream the checkpoint over the regime's declared eval pairs and write a §4.1 root.

    Reuses `harness.stream_tracks` — the same windows, the same prompt splice, the same overlap
    rule — so a round-trip through `candi.bench.external` is an equality against `run_bench`,
    not a correlation.
    """
    if not method:
        raise DumpError("--method is required: provenance.method has nothing to name.")
    # t77 moved `cross_cell` off the source and onto the pair: a regime may declare some pairs that
    # cross cells and some that do not, so the source-level question is "does any pair cross".
    if not any(cross_cell(p, "impute") for p in source.pairs("impute")):
        raise DumpError(
            f"{getattr(source, 'regime_path', source)} declares no `eval_pairs`. The external "
            f"format is `<input>__<target>__<assay>` — without declared pairs there is no input "
            f"cell to name. Score that regime with a checkpoint instead.")

    out = Path(root)
    expected = _expected(source)
    declared = sorted(expected)
    chroms = list(source.eval_chroms)
    n_bins = {c: source.n_bins(c) for c in chroms}

    written: Dict[str, TrackRecord] = {}
    arms: list[str] = ["count"]
    with no_autocast(device):
        for rec in stream_tracks(model, source, device, kind="impute",
                                 batch_windows=batch_windows, progress=progress):
            dirname = track_dirname(rec.pair, rec.assay)
            if dirname not in expected:
                raise DumpError(
                    f"{dirname} is not a declared pair x target assay. The regime declares "
                    f"{len(expected)} tracks; a dump that writes anything else cannot be scored.")
            track_dir = out / dirname
            for c in rec.chroms:
                _write_npz(track_dir / f"{c}.npz",
                           arrays_for_chrom(rec, c, n_bins[c],
                                            signal_target_transform=signal_target_transform))
            written[dirname] = rec
            if rec.has_pval and "pval" not in arms:
                arms.append("pval")
            if progress:
                print(f"[bench.dump] wrote {dirname}", flush=True)

    missing = [d for d in declared if d not in written]
    if missing:
        raise DumpError(
            f"{out} is missing {len(missing)} of the {len(declared)} declared tracks — "
            f"{missing[:5]}. A panel scored with holes in it reads as a whole panel (D2).")

    write_manifest(out, method=method, declared_tracks=declared, arms=arms,
                   version=version, notes=notes)
    return out


# ---------------------------------------------------------------------------
# CLI — checkpoint flags mirror `candi.bench` so `_build_model` can consume this namespace
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m candi.bench.dump",
        description="Write a CANDI checkpoint's predictions to the RIVALS_PLAN.md §4.1 "
                    "prediction-root contract, so candi.bench.external can score it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("data")
    src.add_argument("--store", required=True,
                     help="a CANDI_STORE REGIME FILE (json) declaring `eval_pairs`. Grid, pairs "
                          "and the declared track list all come from it.")
    src.add_argument("--chroms", default=None,
                     help="comma-separated subset of the eval chromosomes (default: all of them)")
    src.add_argument("--biosamples", default=None,
                     help="comma-separated eval biosamples (default: the regime's)")

    mdl = p.add_argument_group("checkpoint")
    mdl.add_argument("--ckpt", required=True)
    mdl.add_argument("--arch-from", default=None,
                     help="a run's own JSON. Reads config.arch and rebuilds the EXACT model that "
                          "wrote the checkpoint. Prefer this over retyping the flags below.")
    mdl.add_argument("--offset", "--arm", dest="offset", default="on",
                     choices=["on", "off", "offset_on", "offset_off"])
    mdl.add_argument("--depth-center", type=float, default=None)
    mdl.add_argument("--d-model", type=int, default=0)
    mdl.add_argument("--nhead", type=int, default=4)
    mdl.add_argument("--embed-dim", type=int, default=32)
    mdl.add_argument("--dropout", type=float, default=0.1)
    mdl.add_argument("--n-transformer-layers", type=int, default=2)
    mdl.add_argument("--decoder-lane", type=int, default=8)
    mdl.add_argument("--deconv-norm", default="lane", choices=["lane", "group"])
    mdl.add_argument("--heads", default="count",
                     help="comma-separated. Must match the checkpoint. `count,signal` is "
                          "required for the pval arm.")
    mdl.add_argument("--meta-embed-layernorm", default="on", choices=["on", "off"])
    mdl.add_argument("--signal-target-transform", dest="signal_target_transform", default=None,
                     choices=["none", "arcsinh", "log1p"],
                     help="D30 — the space the Gaussian head was TRAINED in. Unset reads the "
                          "resolved value out of --arch-from; dumped signal_mu is always in "
                          "-log10 p, so this flag decides the inversion, not the on-disk units.")
    mdl.add_argument("--device", default=None, help="default: cuda when available, else cpu")

    run = p.add_argument_group("what to write")
    run.add_argument("--out", required=True, help="prediction root: manifest.json + one dir per track")
    run.add_argument("--method", required=True,
                     help="manifest `method` — the name candi.bench.external copies into provenance")
    run.add_argument("--version", default="", help="manifest `version`")
    run.add_argument("--notes", default="", help="manifest `notes`")
    run.add_argument("--batch-windows", type=int, default=4,
                     help="windows per forward pass. Throughput only — every bin is written either way.")
    run.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from candi.bench.cli import _build_model, resolve_signal_target_transform

    a = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    chroms = tuple(c.strip() for c in a.chroms.split(",")) if a.chroms else None
    bios = tuple(b.strip() for b in a.biosamples.split(",")) if a.biosamples else None
    stt, stt_src = resolve_signal_target_transform(a)
    source = open_source(store=a.store, chroms=chroms, biosamples=bios)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        model = _build_model(a, source, device)
        if not a.quiet:
            n = sum(p.numel() for p in model.parameters())
            print(f"[bench.dump] {n:,} params on {device}; {len(source.assays)} assays; "
                  f"eval chroms {list(source.eval_chroms)}", flush=True)
            print(f"[bench.dump] signal_target_transform={stt} (from {stt_src}) — dumped "
                  f"signal_mu is in -log10 p", flush=True)
        dump_predictions(
            model, source, a.out, method=a.method, signal_target_transform=stt,
            batch_windows=a.batch_windows, version=a.version, notes=a.notes,
            progress=not a.quiet, device=device)
    finally:
        source.close()
    if not a.quiet:
        print(f"[bench.dump] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
