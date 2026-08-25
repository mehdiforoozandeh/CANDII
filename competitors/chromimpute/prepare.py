"""prepare — CANDI_STORE `pval` tracks -> the three files ChromImpute's `Convert` wants.

`RIVALS_PLAN.md` §7.2 runs ChromImpute as published: their jar, their seven commands. Everything
on this side of the boundary is I/O. This module writes

    <out>/chrominfo.txt            chrom <TAB> n_bins*25          (see `write_chrominfo`)
    <out>/inputinfofile.txt        sample <TAB> mark <TAB> file   TRAINING CELLS ONLY (§6.2)
    <out>/targets.tsv              input_bios <TAB> target_bios <TAB> assay
    <out>/signal/<chrom>_<sample>.<mark>.bedgraph.gz

and nothing else. `collect.py` is the return leg.

Three decisions are worth reading before the code.

**The chromosome length we declare is a multiple of 25.** `Convert` writes
`(chromsize - 1) / resolution + 1` bins (`ChromImpute.java`, the Convert writer) — i.e.
`ceil(len/25)`, one more than our grid whenever the real length is not a multiple of 25. The store's
grid is `floor(len/25)` and `bench.external` refuses a prediction whose length is not exactly that.
So `chrominfo.txt` declares `n_bins * 25` rather than the true chromosome length, and the partial
tail bin never exists on either side. This is the only place the two grids could have disagreed.

**Fairness is enforced here, in one function.** `training_tracks` filters to the regime's
`biosamples.train` list and to the regime's assay vocabulary. No `V_` or `B_` biosample can enter
`inputinfofile.txt`, and `inputinfofile.txt` is the whole of what ChromImpute is allowed to see —
`Convert`, `ComputeGlobalDist`, `GenerateTrainData`, `Train` and `Apply` all take it and none of
them reads anything outside it. `chipseq-control` is dropped with the same filter: it is a control
column in the manifest, not an assay the regime declares.

**Bedgraph, run-length encoded, at four decimals.** ChromImpute's `Convert` averages the input
signal over the bases of each bin; our values are already constant across each 25 bp bin, so the
average is the bin value and the re-binning is an identity. `Convert` then writes its output through
a `NumberFormat` with `setMaximumFractionDigits(2)`, so anything below the second decimal is
invisible downstream no matter what we hand it. Four decimals is the safety margin, and rounding
before the run-length pass is what makes the runs long.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

RESOLUTION = 25

#: Manifest entries that are not assays. The regime's own `assays` list is the authority; this is
#: only here to make the reason for the drop greppable.
NOT_AN_ASSAY = ("chipseq-control",)


# ---------------------------------------------------------------------------------------------
# the grid, the compendium, the targets
# ---------------------------------------------------------------------------------------------


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_assays(manifest: dict, biosample: str, kind: str = "pval") -> List[str]:
    """The assays `biosample` actually carries in `kind`, straight off the store manifest."""
    rec = manifest["biosamples"][biosample]
    if kind not in rec.get("kinds", []):
        return []
    return sorted({t["assay"] for t in rec["tracks"]})


def training_tracks(manifest: dict, regime: dict, kind: str = "pval") -> List[Tuple[str, str]]:
    """§6.2, enforced: `(sample, mark)` for the regime's TRAINING biosamples and no others.

    This is the fairness gate for the whole method. What comes out of here becomes
    `inputinfofile.txt`, and `inputinfofile.txt` is the entire compendium ChromImpute sees.
    """
    allowed = [a for a in regime["assays"] if a not in NOT_AN_ASSAY]
    out: List[Tuple[str, str]] = []
    for sample in regime["biosamples"]["train"]:
        if not sample.startswith("T_"):
            raise ValueError(
                f"{sample!r} is in regime biosamples.train but does not carry the T_ prefix. "
                f"RIVALS_PLAN.md §6.2 lets rivals see training-split cells only; refusing to "
                f"guess which split this is.")
        have = set(manifest_assays(manifest, sample, kind))
        out.extend((sample, mark) for mark in allowed if mark in have)
    return out


def impute_targets(manifest: dict, regime: dict, kind: str = "pval") -> List[Tuple[str, str, str]]:
    """`(input_bios, target_bios, assay)` — one per declared pair × assay the pair holds out.

    A declared pair `(T_X, V_X)` is one cell type twice: `T_X` carries the assays the method may
    condition on, `V_X` carries the assays it must predict. The target set is therefore
    `assays(V_X) - assays(T_X)`; an assay present in both is not held out and is not a target.
    """
    allowed = [a for a in regime["assays"] if a not in NOT_AN_ASSAY]
    out: List[Tuple[str, str, str]] = []
    for inp, tgt in regime["eval_pairs"]:
        have = set(manifest_assays(manifest, inp, kind))
        want = set(manifest_assays(manifest, tgt, kind))
        out.extend((inp, tgt, a) for a in allowed if a in want and a not in have)
    return out


def pilot_subset(targets: Sequence[Tuple[str, str, str]], n: int) -> List[Tuple[str, str, str]]:
    """§7.2's "20 representative (cell, mark) pairs" — one target per mark, widest marks first.

    The pilot exists to price the grid, and the two stages whose cost scales with something other
    than the target count are `GenerateTrainData` (per mark) and `ComputeGlobalDist` (per mark). So
    the subset spreads over marks rather than over cells: `n` marks, ordered by how many targets
    each carries, one target each. Deterministic, so a re-run prices the same work.
    """
    from collections import Counter
    per_mark = Counter(a for _, _, a in targets)
    marks = sorted(per_mark, key=lambda m: (-per_mark[m], m))[:n]
    return sorted(min(t for t in targets if t[2] == m) for m in marks)


def signal_filename(sample: str, mark: str) -> str:
    """The `inputinfofile` third column. `Convert -c chrom` reads `<chrom>_` + this from INPUTDIR.

    Sample and mark are joined with `.` rather than `_` because both of ours contain `_`
    (`T_K562`, `H3K4me3` is fine but `T_upper_lobe_of_left_lung` is not) and a name we cannot split
    again is a name we cannot debug. ChromImpute never parses this string — it only concatenates it
    — so the separator is ours to choose.
    """
    return f"{sample}.{mark}.bedgraph.gz"


# ---------------------------------------------------------------------------------------------
# the three files
# ---------------------------------------------------------------------------------------------


def write_chrominfo(path: Path, n_bins: Dict[str, int], chroms: Sequence[str]) -> Path:
    """`chrom <TAB> n_bins*25` — our grid, declared as a length so `Convert` reproduces it exactly.

    `Convert` emits `(len - 1) // 25 + 1` bins. At `len = n_bins * 25` that is exactly `n_bins`,
    which is the length `bench.external` asserts. Declaring the true chromosome length would emit
    one extra, partially-covered bin (the manual's own `wigToBigWig -clip` note) and every track
    would fail the length check.
    """
    lines = []
    for c in chroms:
        if c not in n_bins:
            raise ValueError(f"{c!r} is not in the store's bin table: {sorted(n_bins)}")
        lines.append(f"{c}\t{int(n_bins[c]) * RESOLUTION}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_inputinfo(path: Path, tracks: Iterable[Tuple[str, str]]) -> Path:
    lines = [f"{s}\t{m}\t{signal_filename(s, m)}" for s, m in tracks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_targets(path: Path, targets: Iterable[Tuple[str, str, str]]) -> Path:
    lines = [f"{i}\t{t}\t{a}" for i, t, a in targets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_targets(path: Path) -> List[Tuple[str, str, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}: expected 3 tab-separated fields, got {line!r}")
        out.append((parts[0], parts[1], parts[2]))
    return out


# ---------------------------------------------------------------------------------------------
# the signal itself
# ---------------------------------------------------------------------------------------------


def bedgraph_lines(values: np.ndarray, chrom: str, *, decimals: int = 4,
                   resolution: int = RESOLUTION) -> Iterable[str]:
    """Run-length bedgraph over the absolute bin grid: bin `i` is `[i*25, (i+1)*25)`.

    Equal *rounded* neighbours collapse into one interval. `Convert` averages per base, so a wide
    interval and 25-bp intervals of the same value give the identical converted bin — the collapse
    is free. Rounding first is what makes the runs long: `-log10 p` is dense in the low tail and
    two bins that differ in the sixth decimal are the same number once ChromImpute has written it.
    """
    if values.ndim != 1:
        raise ValueError(f"{chrom}: expected a 1-D bin vector, got shape {values.shape}")
    if values.size == 0:
        return []
    v = np.round(np.asarray(values, dtype=np.float64), decimals)
    if not np.isfinite(v).all():
        bad = int((~np.isfinite(v)).sum())
        raise ValueError(
            f"{chrom}: {bad} non-finite bins in the source signal. ChromImpute parses every line "
            f"with Float.parseFloat and would take NaN as a value; refusing to write it.")
    change = np.empty(v.size, dtype=bool)
    change[0] = True
    np.not_equal(v[1:], v[:-1], out=change[1:])
    idx = np.flatnonzero(change)
    starts = idx * resolution
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = v.size * resolution
    fmt = f"{{}}\t{{}}\t{{}}\t{{:.{decimals}f}}\n"
    vals = v[idx]
    return [fmt.format(chrom, s, e, x)
            for s, e, x in zip(starts.tolist(), ends.tolist(), vals.tolist())]


def write_bedgraph(path: Path, values: np.ndarray, chrom: str, *, decimals: int = 4) -> int:
    """Write one `<chrom>_<sample>.<mark>.bedgraph.gz`. Returns the interval count."""
    lines = list(bedgraph_lines(values, chrom, decimals=decimals))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wt", compresslevel=6) as fh:
        fh.writelines(lines)
    tmp.replace(path)
    return len(lines)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare.py",
        description="CANDI_STORE pval -> ChromImpute Convert inputs (RIVALS_PLAN.md §7.2).")
    p.add_argument("--store", required=True, help="corpus root, e.g. .../CANDI_STORE/eic")
    p.add_argument("--regime", required=True, help="configs/regime.eic_val.json")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--chroms", default="chr21",
                   help="comma list, or 'all' for every chromosome the store carries")
    p.add_argument("--decimals", type=int, default=4,
                   help="decimals kept in the bedgraph; Convert itself writes 2 (default: 4)")
    p.add_argument("--only-sample", default=None, help="write signal for this sample only")
    p.add_argument("--only-mark", default=None, help="write signal for this mark only")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="write only track i where i %% N == I, for a SLURM array")
    p.add_argument("--pilot", type=int, default=None, metavar="N",
                   help="also write targets_pilot.tsv, N representative targets (§7.2 pilot gate)")
    p.add_argument("--no-signal", action="store_true",
                   help="write chrominfo/inputinfo/targets and stop")
    p.add_argument("--force", action="store_true", help="rewrite bedgraphs that already exist")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from candi.store.reader import CorpusStore  # local: competitors are never imported by candi

    root = Path(args.store)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regime = load_json(args.regime)
    manifest = load_json(root / "manifest.json")

    store = CorpusStore(root, kinds=["pval"])
    n_bins = dict(manifest["genome"]["n_bins"])
    chroms = sorted(n_bins) if args.chroms == "all" else [c.strip() for c in args.chroms.split(",")]

    tracks = training_tracks(manifest, regime)
    targets = impute_targets(manifest, regime)
    write_chrominfo(out / "chrominfo.txt", n_bins, chroms)
    write_inputinfo(out / "inputinfofile.txt", tracks)
    write_targets(out / "targets.tsv", targets)
    print(f"[prepare] {len(tracks)} training tracks, {len(targets)} imputation targets, "
          f"{len(chroms)} chromosome(s) -> {out}")
    if args.pilot:
        sub = pilot_subset(targets, args.pilot)
        write_targets(out / "targets_pilot.tsv", sub)
        print(f"[prepare] pilot subset: {len(sub)} targets over "
              f"{len({a for _, _, a in sub})} marks -> targets_pilot.tsv")
    if args.no_signal:
        return 0

    todo = [(s, m) for s, m in tracks
            if (args.only_sample in (None, s) and args.only_mark in (None, m))]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = [t for k, t in enumerate(todo) if k % n == i]
    if not todo:
        raise SystemExit("[prepare] the sample/mark/shard filters selected no track")

    sig = out / "signal"
    for sample, mark in todo:
        view = store[sample][mark]
        for chrom in chroms:
            path = sig / f"{chrom}_{signal_filename(sample, mark)}"
            if path.exists() and not args.force:
                print(f"[prepare] skip {path.name} (exists)")
                continue
            values = view.pval(chrom)
            want = int(n_bins[chrom])
            if values.size != want:
                raise SystemExit(
                    f"[prepare] {sample}/{mark}/{chrom}: store returned {values.size} bins, "
                    f"manifest says {want}")
            n = write_bedgraph(path, values, chrom, decimals=args.decimals)
            print(f"[prepare] {path.name}: {n} intervals over {want} bins "
                  f"({n / want:.3f} of the grid)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
