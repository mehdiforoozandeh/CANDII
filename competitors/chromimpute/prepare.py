"""prepare — CANDI_STORE `pval` tracks -> the three files ChromImpute's `Convert` wants.

`RIVALS_PLAN.md` §7.2 runs ChromImpute as published: their jar, their seven commands. Everything
on this side of the boundary is I/O. This module writes

    <out>/chrominfo.txt            chrom <TAB> n_bins*25          the grid `Apply` predicts on
    <out>/chrominfo.train.txt      chrom <TAB> n_bins*25          the grid the predictors are fit on
    <out>/inputinfofile.txt        sample <TAB> mark <TAB> file   TRAINING CELLS ONLY (§6.2)
    <out>/targets.tsv              input_bios <TAB> target_bios <TAB> assay
    <out>/signal/<chrom>_<sample>.<mark>.bedgraph.gz

and nothing else. `collect.py` is the return leg.

Four decisions are worth reading before the code.

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

**The training grid is always declared separately from the apply grid.** `GenerateTrainData`
spreads its 100,000 locations over everything the `chrominfo` it is handed declares, and the
predictors `Train` fits from them are reused at every position `Apply` predicts — transferable
parameters, so Rule 2 (`BENCHMARK_DESIGN.md` §2) puts them on the regime's training loci. The jar
has no flag for a locus scope: the smallest thing any command understands is a chromosome out of
`chrominfo`. So `training_scope` declares that scope as a second grid, `chrominfo.train.txt`, and
`chrominfo.txt` stays the real chromosomes `Apply` predicts on. Two files rather than one because
one file cannot say two things: put the eval chromosomes in the training grid and the sample lands
on them, which is the whole of what this prevents.

The scope is the regime's `train_chroms`, or — under a `regions` regime (`eic.pilot`, D32) — each
training region declared as its own chromosome. Departure from published practice, recorded in
`README.md`: ChromImpute as published samples inside the chromosomes it later predicts.
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
# the training grid — Rule 2's locus scope, declared as chromosomes because the jar has nothing else
# ---------------------------------------------------------------------------------------------


def region_scope(regime: dict, regime_path: str | Path) -> List[Tuple[str, str, int, int]]:
    """`(name, source_chrom, first_bin, end_bin)` per training region, or `[]` without `regions`.

    D32 is written for CANDI's 768-bin windows: a locus counts only if it lies WHOLLY inside one
    region, on the chromosome's own bin grid. ChromImpute samples single 25 bp locations rather
    than windows, so the same rule at a window of one bin is `[i*25, (i+1)*25)` inside one region
    — i.e. bins `ceil(start/25)` up to `end//25`. That is a containment count and not
    `bp // 25`: the hg38 Pilot Regions do not begin or end on the grid, so a region's first and
    last partial bins belong to no region and are dropped, exactly as they are for CANDI. The
    indices are chromosome bin indices, so the grid is anchored at chromosome bin 0 and never
    re-anchored at a region start (§3.1 ruled against per-region anchoring).

    Regions off `train_chroms` are absent: the Rule 2 cut is the regime's chromosome list, not the
    BED, so the same BED stays correct for any other split (§3.1).

    Each region becomes one declared chromosome downstream, which is what makes containment a
    property of the grid rather than of the sampler: the only positions the jar can reach are
    contained ones. It also keeps every feature window inside one region — concatenating the 40
    into one pseudo-chromosome would let the 400-bin wide window span two loci megabases apart.
    """
    from candi.store.regime import RegionSet  # local: competitors are never imported by candi

    if not regime.get("regions"):
        return []
    rs = RegionSet.from_obj(regime["regions"], base=Path(regime_path).resolve().parent)
    train = set(regime["train_chroms"])
    out: List[Tuple[str, str, int, int]] = []
    for chrom, start, end, name in rs.intervals:
        if chrom not in train:
            continue
        first, stop = -(-int(start) // RESOLUTION), int(end) // RESOLUTION
        if stop > first:
            out.append((name, chrom, first, stop))
    if not out:
        raise ValueError(
            f"{regime['regions']['bed']} has no region on train_chroms, so the regime declares a "
            f"training scope with nothing in it.")
    names = [n for n, _, _, _ in out]
    clash = sorted(set(names) & train)
    if len(set(names)) != len(names) or clash:
        raise ValueError(
            f"the BED's names must be unique and must not collide with a chromosome name — each "
            f"becomes a declared chromosome of the training grid. duplicates="
            f"{sorted({n for n in names if names.count(n) > 1})} collisions={clash}")
    # By source position, so the regions of one chromosome are adjacent: the signal writer holds
    # one decompressed chromosome at a time and this is what keeps it to one read each.
    return sorted(out, key=lambda r: (r[1], r[2]))


def training_scope(regime: dict, regime_path: str | Path, n_bins: Dict[str, int]
                   ) -> List[Tuple[str, str, int, int]]:
    """`(name, source_chrom, first_bin, end_bin)` per declared training locus. Never empty.

    Rule 2 (§2) names the loci a method's transferable parameters may be fit on, and ChromImpute's
    predictors are transferable — `Train` turns the sampled instances into one predictor per
    (sample, mark) that `Apply` then reuses at every position. So the sampler goes on the regime's
    training loci, and this is the list that becomes `chrominfo.train.txt`.

    A `regions` regime narrows those loci further, to the BED (D32); without one the scope is the
    regime's `train_chroms` whole. Rule 1 is untouched either way — neither branch decides which
    *tracks* the method sees, and no scored track is written into the run at all (`training_tracks`).
    """
    regions = region_scope(regime, regime_path)
    if regions:
        return regions
    if not regime.get("train_chroms"):
        raise ValueError(
            "the regime declares no `train_chroms` and no `regions`, so there is no locus scope to "
            "fit predictors on. Rule 2 needs one named; refusing to fall back to the eval "
            "chromosomes, which is the bug this function exists to prevent.")
    missing = [c for c in regime["train_chroms"] if c not in n_bins]
    if missing:
        raise ValueError(
            f"the regime's train_chroms name {missing}, which the store's bin table does not "
            f"carry: {sorted(n_bins)}")
    return [(c, c, 0, int(n_bins[c])) for c in regime["train_chroms"]]


def signal_slices(chroms: Sequence[str], n_bins: Dict[str, int],
                  training: Sequence[Tuple[str, str, int, int]]
                  ) -> List[Tuple[str, str, int, int]]:
    """Every grid the signal writer emits, as `(declared_name, source_chrom, first_bin, end_bin)`.

    A real chromosome is the whole of itself; a D32 region is a slice of the chromosome it sits
    on. One list so the apply grid and the training grid cannot drift apart in the writer.

    Deduplicated on the declared name, because the two grids may name the same chromosome: a regime
    whose `train_chroms` and eval chromosomes overlap declares it in both files, and one bedgraph
    serves both. Writing it twice would be the same bytes at twice the cost.
    """
    out: List[Tuple[str, str, int, int]] = []
    seen: set = set()
    for sl in [(c, c, 0, int(n_bins[c])) for c in chroms] + list(training):
        if sl[0] not in seen:
            seen.add(sl[0])
            out.append(sl)
    return out


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

    # All three kinds, though only `pval` is read. `BiosampleStore.control_col` is taken from
    # `kinds[0]`, and the manifest's `control_col` describes `counts`; opening `kinds=["pval"]`
    # makes the reader's own manifest cross-check compare pval's -1 against the manifest's 11 and
    # refuse the biosample. Opening everything keeps the check comparing like with like.
    store = CorpusStore(root)
    n_bins = dict(manifest["genome"]["n_bins"])
    chroms = sorted(n_bins) if args.chroms == "all" else [c.strip() for c in args.chroms.split(",")]

    tracks = training_tracks(manifest, regime)
    targets = impute_targets(manifest, regime)
    write_chrominfo(out / "chrominfo.txt", n_bins, chroms)
    write_inputinfo(out / "inputinfofile.txt", tracks)
    write_targets(out / "targets.tsv", targets)
    print(f"[prepare] {len(tracks)} training tracks, {len(targets)} imputation targets, "
          f"{len(chroms)} chromosome(s) -> {out}")
    training = training_scope(regime, args.regime, n_bins)
    write_chrominfo(out / "chrominfo.train.txt",
                    {name: stop - first for name, _, first, stop in training},
                    [name for name, _, _, _ in training])
    overlap = sorted({c for _, c, _, _ in training} & set(chroms))
    print(f"[prepare] training grid: {len(training)} declared locus/loci over "
          f"{len({c for _, c, _, _ in training})} chromosome(s), "
          f"{sum(stop - first for _, _, first, stop in training):,} bins "
          f"-> chrominfo.train.txt")
    # Not fatal — a smoke regime may legitimately train and score on one chromosome — but it is the
    # condition Rule 2 is about, so it is said out loud rather than left to be read off two files.
    if overlap:
        print(f"[prepare] WARNING: the training grid sits on {', '.join(overlap)}, which is also "
              f"being predicted. Under BENCHMARK_DESIGN.md Rule 2 that makes the regime's scope "
              f"name wrong for this run.")
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

    slices = signal_slices(chroms, n_bins, training)
    sig = out / "signal"
    for sample, mark in todo:
        view = store[sample][mark]
        held, whole = None, None
        for name, chrom, first, stop in slices:
            path = sig / f"{name}_{signal_filename(sample, mark)}"
            if path.exists() and not args.force:
                print(f"[prepare] skip {path.name} (exists)")
                continue
            if held != chrom:
                held, whole = chrom, view.pval(chrom, 0)
                want = int(n_bins[chrom])
                if whole.size != want:
                    raise SystemExit(
                        f"[prepare] {sample}/{mark}/{chrom}: store returned {whole.size} bins, "
                        f"manifest says {want}")
            values = whole[first:stop]
            n = write_bedgraph(path, values, name, decimals=args.decimals)
            print(f"[prepare] {path.name}: {n} intervals over {values.size} bins "
                  f"({n / values.size:.3f} of the grid)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
