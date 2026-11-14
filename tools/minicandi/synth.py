"""A small synthetic tables directory for miniCANDI tests (pinned interface B layout).

`make_synthetic_tables(root, ...)` writes

  root/genome/{dna.h5, mask.h5, chrom_sizes.json}
  root/measurements/<measurement_id>/{counts.h5, pr1.h5, pr2.h5}
  root/tables/synth/{measurements.tsv, pairs.tsv, withheld.tsv, covariates.json}

and returns the tables directory. Planted effects:

  context      each cell has a latent activity z(bin) = 0.7 (a field shared by all cells) +
               0.7 (the cell's own field) + 2 (GC(bin) - 0.5); both fields are smoothed
               Gaussians (width `smooth` bins); z is shared by all the cell's tracks
  cross-assay  every track's log rate is loading_a * z + offset_a, with loadings of both signs
               (H3K27me3 is anti-correlated), so the given tracks predict a masked one
  covariate    views of each track: base; depth0.50 (binomial thinning of base, subset of it);
               mapq0 (rate x (1 + 1.5 lowmap(bin)), where lowmap marks AT-rich bins);
               se (rate x 0.8, single-end, 36 bp)
  noise        counts ~ NB(mean s * rate * effect, size 5), s = depth / 2**24;
               pr1 ~ Binomial(counts, 1/2), pr2 = counts - pr1

Split per TRACK, as EIC: cells s0..s<n-1> each hold training tracks, and `val_tracks` /
`blind_tracks` (cell index, assay) mark the validation and blind tracks (defaults: s0 H3K4me3,
s1 H3K27ac, s2 DNase-seq val; s3 H3K4me3, s4 H3K27ac blind); every other track, the control
included, is training. Assays default to DNase-seq, H3K27ac, H3K4me3, H3K27me3 and the control.
Datasets are chunked at <= 4096 bins. `withheld.tsv` holds two combinations:
(mapq 0 -> se) for narrow_histone, and (depth 0.50 -> mapq 0) for every class (`*`).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, WITHHELD_COLUMNS  # noqa: E402

DEFAULT_ASSAYS = ("DNase-seq", "H3K27ac", "H3K4me3", "H3K27me3", C.CONTROL)
LOADINGS = {"DNase-seq": 1.0, "H3K27ac": 1.2, "H3K4me3": 1.5, "H3K27me3": -0.8,
            C.CONTROL: 0.05}
ARMS = (("base", ""), ("depth", "0.50"), ("mapq", "0"), ("se", ""))
WITHHELD = (("mapq", "0", "se", "", "narrow_histone"), ("depth", "0.50", "mapq", "0", "*"))
NB_SIZE = 5.0
LABS = ("labA", "labB")


def _arm_name(knob: str, value: str) -> str:
    return "base" if knob == "base" else f"{knob}{value}"


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    k /= k.sum()
    y = np.convolve(x, k, mode="same")
    return y / (y.std() + 1e-9)


def _write_counts(path: Path, arrays: dict, mid: str, depth: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for chrom, v in arrays.items():
            f.create_dataset(chrom, data=v.astype(np.uint32), chunks=(min(4096, v.size),))
        f.attrs.update({"schema": 1, "measurement_id": mid, "depth_reads": depth,
                        "depth_fragments": depth, "resolution": C.RES,
                        "counting_rule": "synthetic"})


def _nb(rng, mu: np.ndarray) -> np.ndarray:
    p = NB_SIZE / (NB_SIZE + mu)
    return rng.negative_binomial(NB_SIZE, p)


VAL_TRACKS = ((0, "H3K4me3"), (1, "H3K27ac"), (2, "DNase-seq"))
BLIND_TRACKS = ((3, "H3K4me3"), (4, "H3K27ac"))


def make_synthetic_tables(root, *, n_cells: int = 6, val_tracks=VAL_TRACKS,
                          blind_tracks=BLIND_TRACKS, assays=DEFAULT_ASSAYS, chrom_bins=None, smooth: int = 6,
                          seed: int = 0, name: str = "synth") -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    chrom_bins = dict(chrom_bins or {"chr1": 2048, "chr2": 1024})

    # genome: dna codes, GC per bin, an AT-rich "low mappability" flag, a validity mask
    gdir = root / "genome"
    gdir.mkdir(parents=True, exist_ok=True)
    gc, lowmap = {}, {}
    with h5py.File(gdir / "dna.h5", "w") as fd, h5py.File(gdir / "mask.h5", "w") as fm:
        for chrom, nb in chrom_bins.items():
            p_gc = np.clip(0.5 + 0.25 * _smooth(rng.standard_normal(nb), 10), 0.1, 0.9)
            bp_gc = np.repeat(p_gc, C.RES)
            is_gc = rng.random(nb * C.RES) < bp_gc
            codes = np.where(is_gc, rng.integers(1, 3, nb * C.RES),
                             np.where(rng.random(nb * C.RES) < 0.5, 0, 3)).astype(np.uint8)
            codes[: 10 * C.RES] = 4                                  # an N run at the start
            fd.create_dataset(chrom, data=codes)
            g = (codes.reshape(nb, C.RES) == 1) | (codes.reshape(nb, C.RES) == 2)
            gc[chrom] = g.mean(1)
            lowmap[chrom] = (gc[chrom] < 0.35).astype(np.float64)
            m = np.ones(nb, np.uint8)
            m[:10] = 0
            fm.create_dataset(chrom, data=m)
    (gdir / "chrom_sizes.json").write_text(
        json.dumps({c: nb * C.RES for c, nb in chrom_bins.items()}), encoding="utf-8")

    roles = {**{(i, a): "val" for i, a in val_tracks},
             **{(i, a): "blind" for i, a in blind_tracks}}
    cells = [f"s{i}" for i in range(n_cells)]
    shared = {c: _smooth(rng.standard_normal(nb), smooth) for c, nb in chrom_bins.items()}
    rows, pairs = [], []
    offsets = {a: (0.0 if a == C.CONTROL else 0.5) for a in assays}
    track_no = 0
    for ci, cell in enumerate(cells):
        z = {c: 0.7 * shared[c] + 0.7 * _smooth(rng.standard_normal(nb), smooth)
             + 2.0 * (gc[c] - 0.5) for c, nb in chrom_bins.items()}
        lab = LABS[ci % len(LABS)]
        donor = f"donor{ci}"
        for a in assays:
            role = roles.get((ci, a), "train")
            track_no += 1
            exp = f"ENCSR{track_no:06d}"
            eic = f"C{ci + 1:02d}M{C.TRACK_INDEX[a] + 1:02d}"
            rate = {c: np.exp(LOADINGS.get(a, 1.0) * z[c] + offsets[a]) for c in chrom_bins}
            depth = float(2 ** rng.uniform(23.0, 25.0))
            s = depth / 2 ** 24
            base_counts = {c: _nb(rng, s * rate[c]) for c in chrom_bins}
            mids = {}
            for knob, value in ARMS:
                arm = _arm_name(knob, value)
                mid = f"{exp}.pooled.uniform.{arm}"
                mids[(knob, value)] = mid
                d = depth
                run_type, read_length, mapq = "paired-ended", 100, 30
                if knob == "base":
                    counts = base_counts
                elif knob == "depth":
                    q = float(value)
                    counts = {c: rng.binomial(base_counts[c], q) for c in chrom_bins}
                    d = depth * q
                elif knob == "mapq":
                    counts = {c: _nb(rng, s * rate[c] * (1 + 1.5 * lowmap[c]))
                              for c in chrom_bins}
                    mapq = 0
                else:                                                # se
                    counts = {c: _nb(rng, 0.8 * s * rate[c]) for c in chrom_bins}
                    run_type, read_length, d = "single-ended", 36, 0.8 * depth
                mdir = root / "measurements" / mid
                _write_counts(mdir / "counts.h5", counts, mid, d)
                pr1 = {c: rng.binomial(v, 0.5) for c, v in counts.items()}
                _write_counts(mdir / "pr1.h5", pr1, mid, d / 2)
                _write_counts(mdir / "pr2.h5", {c: counts[c] - pr1[c] for c in counts}, mid,
                              d / 2)
                row = {k: "" for k in MEASUREMENT_COLUMNS}
                row.update({
                    "measurement_id": mid, "experiment": exp, "assay": a,
                    "assay_class": C.assay_class(a), "cell": cell, "eic_track": eic,
                    "in_eic": role, "unit": "pooled", "layer": "uniform",
                    "arm_knob": "" if knob == "base" else knob, "arm_value": value,
                    "parent_measurement_id": "" if knob == "base" else mids[("base", "")],
                    "reads_subset_of": mids[("base", "")] if knob == "depth" else "",
                    "depth_reads": f"{d:.0f}", "depth_fragments": f"{d / 2:.0f}",
                    "run_type": run_type, "read_length": str(read_length),
                    "platform": "Illumina", "lab": lab,
                    "pipeline": {"DNase-seq": "dnase-seq-pipeline",
                                 "ATAC-seq": "atac-seq-pipeline"}.get(a, "chip-seq-pipeline2"),
                    "pipeline_version": "v2.2.2", "dedup": "1", "mapq_threshold": str(mapq),
                    "antibody": "" if a in ("DNase-seq", C.CONTROL) else f"ab-{a}",
                    "donor": donor, "has_pseudoreps": "1", "has_signal": "0",
                    "nsc": f"{1.05 + 0.1 * rng.random():.3f}",
                    "rsc": f"{0.8 + 0.6 * rng.random():.3f}",
                    "nrf": f"{0.7 + 0.3 * rng.random():.3f}",
                    "pbc1": f"{0.7 + 0.3 * rng.random():.3f}",
                    "pbc2": f"{2 + 8 * rng.random():.3f}",
                    "frip": f"{0.05 + 0.3 * rng.random():.3f}", "frag_len": "200",
                    "counts_path": f"h5:{(mdir / 'counts.h5').resolve()}", "sha256": "",
                })
                rows.append(row)
            for (ka, va), ma in mids.items():
                for (kb, vb), mb in mids.items():
                    if ma == mb:
                        continue
                    nt = role != "train" and any(
                        w[0] == ka and w[1] == va and w[2] == kb and w[3] == vb
                        and w[4] in ("*", C.assay_class(a)) for w in WITHHELD)
                    subset = int({ka, kb} == {"base", "depth"})
                    pairs.append({"pair_id": f"p{len(pairs):05d}", "a": ma, "b": mb,
                                  "rung": "designed", "assay": a, "cell": cell,
                                  "knobs_differing": kb if ka == "base" else ka,
                                  "subset_pair": str(subset), "split": role,
                                  "never_trained": str(int(nt))})

    tdir = root / "tables" / name
    tdir.mkdir(parents=True, exist_ok=True)
    _write_tsv(tdir / "measurements.tsv", MEASUREMENT_COLUMNS, rows)
    _write_tsv(tdir / "pairs.tsv", PAIR_COLUMNS, pairs)
    _write_tsv(tdir / "withheld.tsv", WITHHELD_COLUMNS,
               [dict(zip(WITHHELD_COLUMNS, w)) for w in WITHHELD])
    vocab = {f: sorted({r[f] for r in rows if r[f]})
             for f in ("pipeline", "pipeline_version", "lab", "platform", "antibody", "donor")}
    (tdir / "covariates.json").write_text(json.dumps(
        {"schema": 1, "meta_columns": list(C.META_COLUMNS), "qc_columns": list(C.QC_COLUMNS),
         "vocab": vocab}, indent=1), encoding="utf-8")
    return tdir


def _write_tsv(path: Path, columns: tuple, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_evalset(tables_dir, *, n_regions: int = 2, region_bins: int = 256, seed: int = 0,
                 out=None) -> Path:
    """A tiny evalset.json over the synthetic tables (pinned interface D keys used by the
    dataset): one leave-one-assay-out target per val track (inputs = the same cell's training
    base views), one correction pair per val track (se arm -> base)."""
    tables_dir = Path(tables_dir)
    rows = list(csv.DictReader(open(tables_dir / "measurements.tsv", encoding="utf-8"),
                               delimiter="\t"))
    sizes = json.loads((tables_dir.parent.parent / "genome" / "chrom_sizes.json").read_text())
    rng = np.random.default_rng(seed)
    chroms = sorted(sizes)
    regions = []
    for _ in range(n_regions):
        c = chroms[int(rng.integers(len(chroms)))]
        nb = sizes[c] // C.RES
        regions.append([c, int(rng.integers(0, nb - region_bins + 1))])
    base = {(r["cell"], r["assay"]): r["measurement_id"] for r in rows
            if not r["arm_knob"] and r["in_eic"] == "train"}      # inputs: TRAINING tracks
    imp, cor = [], []
    for r in rows:
        if r["in_eic"] != "val" or r["arm_knob"]:
            continue
        others = [m for (c, a), m in base.items() if c == r["cell"] and a != r["assay"]]
        imp.append({"target": r["measurement_id"], "inputs": sorted(others),
                    "regime": "leave_one_assay_out", "split": "val"})
        se = r["measurement_id"].replace(".base", ".se")
        cor.append({"source": se, "target": r["measurement_id"], "never_trained": 0,
                    "subset_pair": 0, "split": "val"})
    ev = {"tables": tables_dir.name, "regions": regions, "imputation_targets": imp,
          "correction_pairs": cor, "replicate_pairs": [], "scorable": {}, "strata": []}
    out = Path(out) if out else (tables_dir.parent.parent / "eval" / tables_dir.name
                                 / "evalset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return out
