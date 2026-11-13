#!/usr/bin/env python3
"""STt -- training-region lists and strata tables, from training data only (plan Amendment 7, B.2-B.4).

    python tools/onion_data/train_strata.py universe --evalset <json> --genome <dir> --seeds 0,1,2
        --out <dir> [--sizes 64,256,1024,4096]
    python tools/onion_data/train_strata.py tiles --tables <dir> --peaks <dir> --peak-files <tsv>
        --average <h5> --dna-embedding <NT v2 PCA h5 or dir> --genome <dir> --evalset <json>
        --out <dir> [--k 8192]
    python tools/onion_data/train_strata.py region-classes --tile-table <tsv> --evalset <json>
        --tables <dir> --peaks <dir> --peak-files <tsv> --out <tsv>

Every list is a `train_regions` file of pinned interface I: header comment lines
`# roles_read=train`, `# universe=<text>`, `# sha256_inputs=<sha>`, then the columns
`chrom, start_bin, n_bins, stratum, sample_weight, loss_weight`. In every file the sample weights
sum to 1 and the sum over rows of sample_weight x loss_weight is 1, so the expected weighted loss
equals the uniform loss (B.4). Inside one stratum the rows share the stratum's mass in proportion
to their mask-valid bins (agent decision: so a stratum is drawn uniformly over its valid territory,
as today's sampler draws the genome); `kcenter` gives each chosen tile the same mass.

Names. Block: an aligned unit of `region_bins` (16,384) bins with a mask-valid fraction >= 0.9
(`evalset.candidate_blocks`). Universe U: the blocks that are neither evaluation nor fit regions
of the eval set. Tile: an aligned unit of 2,048 bins inside a block of U, kept only with a
mask-valid fraction >= 0.9.

`universe` writes `blocks_minus_eval.tsv` (all of U, w = 1), `eval_heavy.tsv` (half the mass on the
evaluation regions, half on U, w = 1; the dose arm of B.3, so unweighted) and, per seed s and size R,
`terr_r<R>_s<s>.tsv`: the first R blocks of a seeded permutation of U (generator
`default_rng([s, TERR_SALT])`), so the sets are nested within a seed and differ between seeds.

`tiles` writes `tile_table.tsv` (chrom, start_bin, valid_bins, peak_fraction, ccre_like_class,
variance) and the five strata files of B.4, the control `tile_uniform`, plus `STRATA.md`:
  tile_uniform the whole tile table, q in proportion to valid bins, w = 1: the control that shares
               the tile arms' window-in-tile edge effect (a window of 1,024 bins inside a tile of
               2,048 covers the tile's edge bins far less often than its middle bins); its config
               must name `$O/region/lists/tile_uniform.tsv`
  peakfreq_w   5 strata of peak_fraction: 0, (0, .05], (.05, .2], (.2, .5], > .5; q_h ~ sqrt(g_h);
               w = g_h / q_h
  peakfreq_u   the same q_h, w = 1
  ccre_like_w  per tile, by priority: promoter_like (one training cell has an accessibility peak
               and an H3K4me3 peak in the tile), enhancer_like (one training cell that has an
               H3K4me3 track has an accessibility peak, an H3K27ac or H3K4me1 peak, and no H3K4me3
               peak), accessible_only (any accessibility peak), other_marked (any other training
               peak), none; q_h ~ sqrt(g_h); w = g_h / q_h. STRATA.md states the cell counts.
  variance_w   deciles (by tile rank; ties share a decile) of `variance`: per assay, the sample
               variance across training cells of log1p(the tile's TOTAL signal in reads per million,
               summed over its valid bins), rank-normalised to [0, 1] over tiles; then the mean over
               the assays with >= 2 training cells (the ChIP-seq control is left out: it is never a
               target); the top 3 deciles get twice the mass per genome share; w = g_h / q_h
  kcenter      k (8,192) tiles by greedy k-center (first centre seeded by KCENTER_SEED) on a per-tile
               vector: log1p of the tile mean of the training-average rate per assay (sum / n_cells of
               the average h5) joined to the tile mean of the Nucleotide Transformer v2 embedding; each
               column z-scored, then each part divided by sqrt(its column count), so the two parts
               weigh the same; w = cluster size / mean cluster size (every tile of the table is in the
               cluster of its nearest centre). A biased coreset by design: most tiles get q = 0.
`peak_fraction` is the fraction of the peak-frequency denominator tracks with a peak overlapping
the tile by >= 1 bp. The denominator: the training tracks of the tables (every row of the
eic_track has in_eic = train) that have a peak file (`peak_files.tsv` role train, non-empty
`file`, `<peaks>/<eic_track>.narrowPeak.gz` on disk).

`region-classes` writes, per evaluation region, `chrom, start_bin, class, peak_fraction`: the mean
over the region's tiles of the per-tile peak fraction from the same training tracks, classed with
the peakfreq edges. It reads no validation track; it checks that the tile table was built from the
same denominator tracks (the `# peak_tracks_sha256` header) and raises otherwise.

Leak guards (ruling 40). Second experiments (`evalset.second_experiment_ids`) are removed first, as
`ag2_baselines` does. Only measurement rows with in_eic = train are read; a peak file is read only
when its `peak_files.tsv` role is train and every tables row of its eic_track has in_eic = train
(else LeakError); a role outside train, val, blind raises; the average h5 must list
`measurement_ids` in every group, each a tables row with in_eic = train (else LeakError). The
embedding is read only when its `meta.json` (written by `dna_embed.precompute`; for a single h5
file, its root attributes) has `source == "ntv2"` and `trained_on_encode_tracks` false, and every
chromosome file's `source` attribute, where present, is ntv2 (else LeakError). No cCRE and no
ChromHMM file is taken.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval.ag2_peaks import read_peaks  # noqa: E402
from onion_eval.evalset import (CONTROL, MIN_VALID, REGION_BINS,  # noqa: E402
                                candidate_blocks, read_tsv, second_experiment_ids)

RES = 25
TILE_BINS = 2048
TERR_SIZES = (64, 256, 1024, 4096)
TERR_SALT = 283
KCENTER_SEED = 0
KCENTER_K = 8192
WINDOW_BINS = 1024          # one training window (context_bins of the ladder configs)
PEAKFREQ_EDGES = (0.05, 0.20, 0.50)
PEAKFREQ_NAMES = ("pf_0", "pf_0-0.05", "pf_0.05-0.20", "pf_0.20-0.50", "pf_gt0.50")
CCRE_NAMES = ("promoter_like", "enhancer_like", "accessible_only", "other_marked", "none")
ENHANCER_MARKS = ("H3K27ac", "H3K4me1")
PROMOTER_MARK = "H3K4me3"
VARIANCE_TOP_DECILES = 3
VARIANCE_TOP_FACTOR = 2.0
LIST_COLUMNS = ("chrom", "start_bin", "n_bins", "stratum", "sample_weight", "loss_weight")
TILE_COLUMNS = ("chrom", "start_bin", "valid_bins", "peak_fraction", "ccre_like_class", "variance")
ROLES = ("train", "val", "blind")
EMBEDDING_SOURCE = "ntv2"
FULL_HASH_BYTES = 64 << 20


class LeakError(PermissionError):
    """An input that is not training data reached a strata table."""


# ------------------------------------------------------------------------------------ hashing

def file_digest(path: Path) -> str:
    """sha256 of a file; a file above 64 MB gets a labelled partial digest (size, first and last
    4 MB), since the average and embedding files are tens of GB."""
    path = Path(path)
    n = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if n <= FULL_HASH_BYTES:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
            return h.hexdigest()
        h.update(str(n).encode())
        h.update(fh.read(4 << 20))
        fh.seek(n - (4 << 20))
        h.update(fh.read(4 << 20))
    return "partial:" + h.hexdigest()


def inputs_manifest(paths: dict) -> tuple[str, list]:
    """(sha256 over the manifest lines, the lines `name<TAB>path<TAB>digest`); a directory is
    expanded to its sorted files."""
    lines = []
    for name in sorted(paths):
        p = Path(paths[name])
        files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
        for q in files:
            lines.append(f"{name}\t{q}\t{file_digest(q)}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


# ------------------------------------------------------------------------------------ lists

def _fmt(x: float) -> str:
    return format(float(x), ".17g")


def write_list(path: Path, rows: list, universe: str, sha: str) -> None:
    """rows: dicts with LIST_COLUMNS. Refuses weights that break the two sums."""
    sw = np.array([r["sample_weight"] for r in rows], np.float64)
    lw = np.array([r["loss_weight"] for r in rows], np.float64)
    if not rows or abs(sw.sum() - 1.0) > 1e-9 or abs((sw * lw).sum() - 1.0) > 1e-9:
        raise ValueError(f"{path}: sum q = {sw.sum()}, sum q w = {(sw * lw).sum()}; both must be 1")
    if any(int(r["n_bins"]) <= 0 for r in rows):
        raise ValueError(f"{path}: a row with no bins")
    tmp = Path(str(path) + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("# roles_read=train\n")
        fh.write(f"# universe={universe}\n")
        fh.write(f"# sha256_inputs={sha}\n")
        fh.write("\t".join(LIST_COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join([r["chrom"], str(int(r["start_bin"])), str(int(r["n_bins"])),
                                r["stratum"], _fmt(r["sample_weight"]),
                                _fmt(r["loss_weight"])]) + "\n")
    os.replace(tmp, path)


def read_list(path) -> tuple[dict, list]:
    """(header dict, rows) of a list written by `write_list`."""
    head, rows, cols = {}, [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            k, _, v = line[2:].partition("=")
            head[k] = v
        elif cols is None:
            cols = line.split("\t")
        elif line:
            r = dict(zip(cols, line.split("\t")))
            r["start_bin"], r["n_bins"] = int(r["start_bin"]), int(r["n_bins"])
            r["sample_weight"], r["loss_weight"] = float(r["sample_weight"]), float(r["loss_weight"])
            rows.append(r)
    return head, rows


def strata_rows(units: list, valid: np.ndarray, strata: list, q: dict, w: dict,
                n_bins: int) -> list:
    """List rows: unit i of stratum h gets q_h x valid_i / (valid bins of h) and loss weight w_h."""
    tot = defaultdict(float)
    for v, h in zip(valid, strata):
        tot[h] += float(v)
    return [{"chrom": c, "start_bin": int(s), "n_bins": n_bins, "stratum": h,
             "sample_weight": q[h] * float(v) / tot[h], "loss_weight": w[h]}
            for (c, s), v, h in zip(units, valid, strata)]


def genome_shares(valid: np.ndarray, strata: list) -> dict:
    g = defaultdict(float)
    for v, h in zip(valid, strata):
        g[h] += float(v)
    tot = sum(g.values())
    return {h: x / tot for h, x in g.items()}


def sqrt_design(g: dict) -> tuple[dict, dict]:
    """q_h ~ sqrt(g_h), w_h = g_h / q_h."""
    s = sum(math.sqrt(x) for x in g.values())
    q = {h: math.sqrt(x) / s for h, x in g.items()}
    return q, {h: g[h] / q[h] for h in g}


# ------------------------------------------------------------------------------------ universe

def load_evalset(path) -> dict:
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("regions", "fit_regions"):
        ev[k] = [(str(c), int(s)) for c, s in ev.get(k, [])]
    ev["region_bins"] = int(ev.get("region_bins", REGION_BINS))
    return ev


def universe_blocks(ev: dict, genome: Path) -> list:
    """Blocks of the genome mask minus the evaluation and fit regions; every evaluation and fit
    region must be a block of this mask (else the genome is not the eval set's)."""
    rb = ev["region_bins"]
    if rb % TILE_BINS:
        raise ValueError(f"region_bins {rb} is not a multiple of the tile ({TILE_BINS})")
    cands = [(c, int(s)) for c, s in candidate_blocks(Path(genome), rb)]
    taken = set(ev["regions"]) | set(ev["fit_regions"])
    missing = sorted(taken - set(cands))
    if missing:
        raise ValueError(f"{len(missing)} evaluation or fit regions are not blocks of {genome} "
                         f"(first {missing[0]}): another genome mask")
    return [b for b in cands if b not in taken]


def mask_valid(genome: Path, units: list, n: int) -> np.ndarray:
    """Valid bins of each unit [chrom, start_bin] of n bins."""
    import h5py
    out = np.zeros(len(units), np.int64)
    by = defaultdict(list)
    for i, (c, s) in enumerate(units):
        by[c].append((i, s))
    with h5py.File(Path(genome) / "mask.h5", "r") as mf:
        for c, v in by.items():
            m = np.asarray(mf[c][:], bool)
            for i, s in v:
                out[i] = int(m[s:s + n].sum())
    return out


def terr_lists(U: list, seed: int, sizes) -> dict:
    """R -> the first R blocks of the seeded permutation of U (nested in R), sorted."""
    if max(sizes) > len(U):
        raise ValueError(f"territory {max(sizes)} blocks > |U| = {len(U)}")
    order = np.random.default_rng([int(seed), TERR_SALT]).permutation(len(U))
    return {R: sorted(U[i] for i in order[:R]) for R in sizes}


def run_universe(evalset, genome, seeds, out, sizes=TERR_SIZES) -> dict:
    ev = load_evalset(evalset)
    genome, out = Path(genome), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rb = ev["region_bins"]
    U = universe_blocks(ev, genome)
    sha, _ = inputs_manifest({"evalset": evalset, "mask": genome / "mask.h5",
                              "chrom_sizes": genome / "chrom_sizes.json"})
    vu = mask_valid(genome, U, rb)
    uni = (f"blocks_minus_eval: {len(U)} blocks of {rb} bins (mask valid >= {MIN_VALID}) minus "
           f"{len(ev['regions'])} evaluation and {len(ev['fit_regions'])} fit regions")
    write_list(out / "blocks_minus_eval.tsv",
               strata_rows(U, vu, ["U"] * len(U), {"U": 1.0}, {"U": 1.0}, rb), uni, sha)
    E = list(ev["regions"])
    ve = mask_valid(genome, E, rb)
    write_list(out / "eval_heavy.tsv",
               strata_rows(E + U, np.concatenate([ve, vu]), ["eval"] * len(E) + ["U"] * len(U),
                           {"eval": 0.5, "U": 0.5}, {"eval": 1.0, "U": 1.0}, rb),
               f"eval_heavy: half the mass on {len(E)} evaluation regions, half on {uni}", sha)
    written = {"U": len(U)}
    for s in seeds:
        for R, blocks in terr_lists(U, s, sizes).items():
            v = mask_valid(genome, blocks, rb)
            write_list(out / f"terr_r{R}_s{s}.tsv",
                       strata_rows(blocks, v, ["terr"] * R, {"terr": 1.0}, {"terr": 1.0}, rb),
                       f"terr_r{R}_s{s}: {R} blocks of U by seed {s} (salt {TERR_SALT}); {uni}", sha)
            written[f"terr_r{R}_s{s}"] = R
    return written


# ------------------------------------------------------------------------------------ guards

def train_tables(tables) -> tuple[list, dict]:
    """(rows with in_eic = train, eic_track -> set of in_eic over all rows), after every second
    experiment (`evalset.second_experiment_ids`: evaluation-only, never training data) is removed,
    as `ag2_baselines` does."""
    second = second_experiment_ids(tables)
    rows = [r for r in read_tsv(Path(tables) / "measurements.tsv")
            if r["measurement_id"] not in second]
    roles = defaultdict(set)
    for r in rows:
        roles[r.get("eic_track", "")].add(r["in_eic"])
    return [r for r in rows if r["in_eic"] == "train"], roles


def require_train(row: dict) -> dict:
    if row.get("in_eic") != "train":
        raise LeakError(f"{row.get('measurement_id')}: in_eic {row.get('in_eic')!r}; only "
                        f"training rows may be read")
    return row


def peak_tracks(tables, peaks, peak_files) -> list:
    """[(eic_track, cell, assay, path)] of the denominator: training tracks with a peak file."""
    trows, roles = train_tables(tables)
    meta = {}
    for r in trows:
        meta.setdefault(r["eic_track"], (r["cell"], r["assay"]))
    out = []
    for p in read_tsv(peak_files):
        role = p.get("role", "")
        if role not in ROLES:
            raise LeakError(f"{peak_files}: {p.get('eic_track')}: role {role!r} is not one of {ROLES}")
        if role != "train":
            continue                                   # never opened
        t = p["eic_track"]
        if roles.get(t, set()) != {"train"}:
            raise LeakError(f"{peak_files}: {t} is listed as train, but its tables rows have "
                            f"in_eic {sorted(roles.get(t, {'absent'}))}")
        if not (p.get("file") or "").strip():
            continue                                   # no peak file for this track
        path = Path(peaks) / f"{t}.narrowPeak.gz"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: listed in {peak_files} but not on disk")
        c, a = meta[t]
        if a == CONTROL:
            continue
        out.append((t, c, a, path))
    if not out:
        raise ValueError("no training track with a peak file")
    return sorted(out)


def tracks_sha(tracks: list) -> str:
    return hashlib.sha256("\n".join(t for t, *_ in tracks).encode()).hexdigest()


def check_average(path, tables) -> list:
    """Groups (assays) of a training-average h5; every summed id must be a train row."""
    import h5py
    trows, _ = train_tables(tables)
    rows = {r["measurement_id"]: r for r in trows}
    with h5py.File(path, "r") as f:
        groups = sorted(f.keys())
        for a in groups:
            if "measurement_ids" not in f[a].attrs:
                raise LeakError(f"{path}: group {a} has no measurement_ids; its sources are unknown")
            for m in np.atleast_1d(f[a].attrs["measurement_ids"]):
                m = m.decode() if isinstance(m, bytes) else str(m)
                if m not in rows or rows[m]["in_eic"] != "train":
                    raise LeakError(f"{path}: group {a} sums {m!r}, which is not a training row "
                                    f"(or is a second experiment) of {tables}")
    return groups


def embedding_meta(path) -> dict:
    """The identity record of an embedding: `<dir>/meta.json` (the chunk A5 layout, written by
    `dna_embed.precompute`), or the root attributes of a single h5 file."""
    import h5py
    p = Path(path)
    if p.is_dir():
        mp = p / "meta.json"
        if not mp.is_file():
            raise LeakError(f"{p}: no meta.json; the embedding's model cannot be checked")
        return json.loads(mp.read_text(encoding="utf-8"))
    with h5py.File(p, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v.item() if hasattr(v, "item") else v)
                for k, v in f.attrs.items()}


def check_embedding(path) -> dict:
    """Accept only the Nucleotide Transformer v2 set: `source == "ntv2"` and
    `trained_on_encode_tracks` false in its meta (ruling 40). Every chromosome file that carries a
    `source` attribute must also say ntv2."""
    import h5py
    p = Path(path)
    meta = embedding_meta(p)
    src, enc = meta.get("source"), meta.get("trained_on_encode_tracks")
    if src != EMBEDDING_SOURCE or enc is not False:
        raise LeakError(f"{p}: source {src!r}, trained_on_encode_tracks {enc!r}; only source "
                        f"{EMBEDDING_SOURCE!r} with trained_on_encode_tracks false is read (a model "
                        f"trained on ENCODE tracks may hold EIC validation or blind experiments)")
    files = sorted(p.glob("*.h5")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"{p}: no embedding file")
    for q in files:
        with h5py.File(q, "r") as f:
            for o in [f] + [f[k] for k in f.keys()]:
                v = o.attrs.get("source")
                v = v.decode() if isinstance(v, bytes) else v
                if v is not None and v != EMBEDDING_SOURCE:
                    raise LeakError(f"{q}: {o.name} has source {v!r}, not {EMBEDDING_SOURCE!r}")
    return meta


# ------------------------------------------------------------------------------------ tiles

def tiles_of(U: list, rb: int) -> list:
    return [(c, s + k * TILE_BINS) for c, s in U for k in range(rb // TILE_BINS)]


def tile_hits(path: Path, tiles: list, index: dict) -> np.ndarray:
    """bool [n_tiles]: a peak overlaps the tile by >= 1 bp."""
    out = np.zeros(len(tiles), bool)
    for c, (s, e) in read_peaks(path).items():
        if c not in index:
            continue
        starts, rows = index[c]
        keep = e > s
        t0 = (s[keep] // RES) // TILE_BINS
        t1 = ((e[keep] - 1) // RES) // TILE_BINS
        n = int(max(t1.max(initial=0), starts.max(initial=0) // TILE_BINS)) + 2
        diff = np.zeros(n + 1, np.int64)
        np.add.at(diff, t0, 1)
        np.add.at(diff, t1 + 1, -1)
        hit = np.cumsum(diff)[:n] > 0
        out[rows] = hit[starts // TILE_BINS]
    return out


def tile_index(tiles: list) -> dict:
    by = defaultdict(list)
    for i, (c, s) in enumerate(tiles):
        by[c].append((s, i))
    return {c: (np.array([s for s, _ in v], np.int64), np.array([i for _, i in v], np.int64))
            for c, v in by.items()}


def peak_matrix(tracks: list, tiles: list) -> np.ndarray:
    idx = tile_index(tiles)
    return np.stack([tile_hits(p, tiles, idx) for _, _, _, p in tracks]) if tracks \
        else np.zeros((0, len(tiles)), bool)


def peakfreq_class(frac: np.ndarray) -> list:
    k = np.where(frac <= 0, 0, 1 + np.searchsorted(PEAKFREQ_EDGES, frac, side="left"))
    return [PEAKFREQ_NAMES[int(i)] for i in k]


def ccre_like(tracks: list, hits: np.ndarray, counts: dict | None = None) -> list:
    """Per tile, the cCRE-recipe class from training peaks of one cell at a time. A cell counts
    for promoter_like only when it has an accessibility and an H3K4me3 track, and for
    enhancer_like only when it has an accessibility, an H3K4me3 and an H3K27ac or H3K4me1 track
    (so "no H3K4me3 peak" is never met by a cell without the track). `counts`, when given, gets
    the number of cells of each rule (the denominators)."""
    n = hits.shape[1]
    by_cell = defaultdict(dict)
    for (t, c, a, _), h in zip(tracks, hits):
        by_cell[c][a] = by_cell[c].get(a, np.zeros(n, bool)) | h
    prom, enh, acc = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    n_acc = n_prom = n_enh = 0
    for marks in by_cell.values():
        has_acc = "DNase-seq" in marks or "ATAC-seq" in marks
        if not has_acc:
            continue
        n_acc += 1
        z = np.zeros(n, bool)
        a = marks.get("DNase-seq", z) | marks.get("ATAC-seq", z)
        acc |= a
        if PROMOTER_MARK not in marks:
            continue
        n_prom += 1
        k4 = marks[PROMOTER_MARK]
        prom |= a & k4
        if not any(m in marks for m in ENHANCER_MARKS):
            continue
        n_enh += 1
        en = marks.get(ENHANCER_MARKS[0], z) | marks.get(ENHANCER_MARKS[1], z)
        enh |= a & en & ~k4
    if counts is not None:
        counts.update(cells=len(by_cell), accessibility=n_acc, promoter_rule=n_prom,
                      enhancer_rule=n_enh)
    anyp = hits.any(axis=0)
    out = np.full(n, CCRE_NAMES[4], object)
    out[anyp] = CCRE_NAMES[3]
    out[acc] = CCRE_NAMES[2]
    out[enh] = CCRE_NAMES[1]
    out[prom] = CCRE_NAMES[0]
    return list(out)


def _tile_means(arr: np.ndarray, tiles_c: list, m: np.ndarray) -> np.ndarray:
    """Mean over the valid bins of each tile start in tiles_c of a per-bin array (1-D or 2-D)."""
    out = []
    for s in tiles_c:
        v = m[s:s + TILE_BINS]
        x = arr[s:s + TILE_BINS]
        out.append(x[v[: x.shape[0]]].mean(axis=0) if v.any() else np.zeros(x.shape[1:]))
    return np.asarray(out, np.float64)


def _by_chrom(tiles: list) -> dict:
    by = defaultdict(list)
    for i, (c, s) in enumerate(tiles):
        by[c].append((i, s))
    return by


def tile_totals(tables, tiles: list, genome: Path) -> dict:
    """assay -> float64 [cells, tiles]: log1p of each training cell's TOTAL depth-normalised signal
    (counts x 1e6 / depth_reads, summed over the tile's valid bins), for the assays with >= 2
    training cells (the ChIP-seq control is left out: it is never a target). Only training base
    views are opened."""
    import h5py
    from onion_eval.evalset import base_views
    trows, _ = train_tables(tables)
    base = base_views(trows, ("train",))
    by_assay = defaultdict(list)
    for (c, a), r in sorted(base.items()):
        require_train(r)
        if a == CONTROL or not r.get("depth_reads") or float(r["depth_reads"]) <= 0:
            continue
        by_assay[a].append(r)
    by_assay = {a: v for a, v in by_assay.items() if len(v) >= 2}
    if not by_assay:
        raise ValueError("no assay has two training cells; the variance strata are undefined")
    bc = _by_chrom(tiles)
    with h5py.File(Path(genome) / "mask.h5", "r") as mf:
        masks = {c: np.asarray(mf[c][:], bool) for c in bc}
    out = {}
    for a, rs in sorted(by_assay.items()):
        vals = np.zeros((len(rs), len(tiles)), np.float64)
        for j, r in enumerate(rs):
            path = require_train(r)["counts_path"]
            if not path.startswith("h5:"):
                raise ValueError(f"{r['measurement_id']}: counts_path {path!r}: only h5: is read")
            scale = 1e6 / float(r["depth_reads"])
            with h5py.File(path[3:], "r") as f:
                for c, v in bc.items():
                    x = np.asarray(f[c][:], np.float64) * scale
                    m = masks[c]
                    for i, s in v:
                        xs = x[s:s + TILE_BINS]
                        vals[j, i] = xs[m[s:s + xs.shape[0]]].sum()
        out[a] = np.log1p(vals)
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    """Rank of each value in [0, 1] (ties get their lowest rank)."""
    srt = np.sort(x)
    return np.searchsorted(srt, x, side="left") / max(len(x) - 1, 1)


def variance_score(totals: dict) -> np.ndarray:
    """Per tile: the mean over assays of the rank-normalised (in [0, 1] over tiles) sample variance
    across training cells of log1p(total signal). A log scale makes the variance relative, so two
    tiles with the same relative spread score alike whatever their mean."""
    return np.mean([rank01(v.var(axis=0, ddof=1)) for _, v in sorted(totals.items())], axis=0)


def tile_variance(tables, tiles: list, genome: Path) -> np.ndarray:
    return variance_score(tile_totals(tables, tiles, genome))


def decile_strata(x: np.ndarray) -> list:
    """Deciles by rank (ties share their lowest rank, so equal values share a decile)."""
    srt = np.sort(x)
    rank = np.searchsorted(srt, x, side="left")
    d = (rank * 10) // len(x)
    return [f"d{int(i) + 1:02d}" for i in d]


def kcenter_features(tiles: list, average, embedding, genome: Path) -> np.ndarray:
    import h5py
    bc = _by_chrom(tiles)
    with h5py.File(Path(genome) / "mask.h5", "r") as mf:
        masks = {c: np.asarray(mf[c][:], bool) for c in bc}
    parts = []
    with h5py.File(average, "r") as f:
        cols = []
        for a in sorted(f.keys()):
            g = f[a]
            n = max(int(g.attrs["n_cells"]), 1)
            col = np.zeros(len(tiles), np.float64)
            for c, v in bc.items():
                if c not in g:
                    continue
                x = np.asarray(g[c][:], np.float64) / n
                col[[i for i, _ in v]] = _tile_means(x, [s for _, s in v], masks[c])
            cols.append(np.log1p(np.maximum(col, 0.0)))
        parts.append(np.stack(cols, axis=1))
    p = Path(embedding)
    emb = None
    for c, v in bc.items():
        if p.is_dir():
            fh = h5py.File(p / f"{c}.h5", "r")
            ds = fh["emb"]
        else:
            fh = h5py.File(p, "r")
            obj = fh[c]
            ds = obj["emb"] if isinstance(obj, h5py.Group) else obj
        with fh:
            if emb is None:
                emb = np.zeros((len(tiles), int(ds.shape[1])), np.float64)
            for i, s in v:
                x = np.asarray(ds[s:s + TILE_BINS], np.float64)
                m = masks[c][s:s + x.shape[0]]
                emb[i] = x[m].mean(axis=0) if m.any() else 0.0
    parts.append(emb)
    out = []
    for X in parts:
        sd = X.std(axis=0)
        Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        out.append(Z / math.sqrt(X.shape[1]))
    return np.concatenate(out, axis=1)


def kcenter(X: np.ndarray, k: int, seed: int = KCENTER_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Greedy k-center (Gonzalez): (centre indices in pick order, cluster sizes of each centre;
    every point joins its nearest centre)."""
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k = {k} > {n} tiles")
    first = int(np.random.default_rng(seed).integers(n))
    centres = [first]
    d = ((X - X[first]) ** 2).sum(axis=1)
    owner = np.zeros(n, np.int64)
    for j in range(1, k):
        nxt = int(np.argmax(d))
        centres.append(nxt)
        dn = ((X - X[nxt]) ** 2).sum(axis=1)
        closer = dn < d
        owner[closer] = j
        d = np.minimum(d, dn)
    return np.array(centres), np.bincount(owner, minlength=k)


def arm_summary(name: str, rows: list, tile_peak: dict, tile_valid: dict, note: str = "") -> list:
    """STRATA.md lines of one arm; g_h from the valid bins of the tile table."""
    q, w, g = defaultdict(float), {}, defaultdict(int)
    gv = defaultdict(float)
    for t, v in tile_valid.items():
        gv[t] = float(v)
    vb = defaultdict(float)
    nopeak = 0.0
    for r in rows:
        q[r["stratum"]] += r["sample_weight"]
        w[r["stratum"]] = r["loss_weight"]
        g[r["stratum"]] += 1
        vb[r["stratum"]] += gv[(r["chrom"], r["start_bin"])]
        if tile_peak.get((r["chrom"], r["start_bin"]), 1.0) == 0:
            nopeak += r["sample_weight"]
    ess = 1.0 / sum(r["sample_weight"] * r["loss_weight"] ** 2 for r in rows)
    lines = [f"## {name}", "", note, "",
             f"Effective sample size per draw, 1 / sum of q x w^2: {ess:.4f}. "
             f"Share of the sampling mass on tiles with no training peak: {nopeak:.4f}.", ""]
    if len(q) <= 12:
        lines += ["| stratum | tiles | g_h | q_h | w_h |", "|---|---|---|---|---|"]
        tot = sum(vb.values())
        for h in sorted(q):
            gh = vb[h] / tot
            lines.append(f"| {h} | {g[h]} | {gh:.6f} | {q[h]:.6f} | {w[h]:.6f} |")
        lines.append("")
    return lines


def edge_coverage(n: int, L: int = WINDOW_BINS) -> tuple[float, float]:
    """A window of L bins drawn with a uniform start inside a row of n bins: (coverage of the
    row's first bin relative to its middle bin, share of the row's bins covered at less than half
    the middle bin's rate)."""
    j = np.arange(n)
    cov = np.minimum(j, n - L) - np.maximum(0, j - L + 1) + 1
    return float(cov[0] / cov.max()), float((cov < cov.max() / 2).mean())


def run_tiles(tables, peaks, peak_files, average, embedding, genome, evalset, out,
              k: int = KCENTER_K) -> dict:
    genome, out = Path(genome), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    check_embedding(embedding)
    groups = check_average(average, tables)
    tracks = peak_tracks(tables, peaks, peak_files)
    ev = load_evalset(evalset)
    rb = ev["region_bins"]
    U = universe_blocks(ev, genome)
    all_tiles = tiles_of(U, rb)
    tv = mask_valid(genome, all_tiles, TILE_BINS)
    keep = tv >= MIN_VALID * TILE_BINS
    tiles = [t for t, kk in zip(all_tiles, keep) if kk]
    valid = tv[keep]
    taken = set(ev["regions"]) | set(ev["fit_regions"])
    for c, s in tiles:                       # by construction; a guard against a later edit
        if (c, s - s % rb) in taken:
            raise AssertionError(f"tile {c}:{s} lies in an evaluation or fit region")
    sha, manifest = inputs_manifest({"tables": Path(tables) / "measurements.tsv",
                                     "peak_files": peak_files,
                                     **{f"peak:{t}": p for t, _, _, p in tracks},
                                     "average": average, "embedding": embedding,
                                     "evalset": evalset, "mask": genome / "mask.h5",
                                     "chrom_sizes": genome / "chrom_sizes.json"})
    hits = peak_matrix(tracks, tiles)
    frac = hits.mean(axis=0)
    ccre_n: dict = {}
    ccre = ccre_like(tracks, hits, ccre_n)
    var = tile_variance(tables, tiles, genome)
    with open(out / "tile_table.tsv", "w", encoding="utf-8") as fh:
        fh.write("# roles_read=train\n")
        fh.write(f"# peak_denominator={len(tracks)}\n")
        fh.write(f"# peak_tracks_sha256={tracks_sha(tracks)}\n")
        fh.write(f"# sha256_inputs={sha}\n")
        fh.write("\t".join(TILE_COLUMNS) + "\n")
        for (c, s), v, f_, cl, va in zip(tiles, valid, frac, ccre, var):
            fh.write(f"{c}\t{s}\t{int(v)}\t{_fmt(f_)}\t{cl}\t{_fmt(va)}\n")
    uni = (f"tiles of U: {len(tiles)} tiles of {TILE_BINS} bins (mask valid >= {MIN_VALID}) in "
           f"{len(U)} blocks of {rb} bins minus {len(ev['regions'])} evaluation and "
           f"{len(ev['fit_regions'])} fit regions")
    tile_peak = {t: float(f_) for t, f_ in zip(tiles, frac)}
    tile_valid = dict(zip(tiles, valid))
    arms = {}
    arms["tile_uniform"] = (strata_rows(tiles, valid, ["tile"] * len(tiles), {"tile": 1.0},
                                        {"tile": 1.0}, TILE_BINS),
                            "The tile-uniform control: the tile table, q in proportion to valid "
                            "bins, w = 1. It shares the window-in-tile edge effect of the tile arms "
                            "(see above), which `blocks_minus_eval` does not.")
    pf = peakfreq_class(frac)
    g = genome_shares(valid, pf)
    q, w = sqrt_design(g)
    arms["peakfreq_w"] = (strata_rows(tiles, valid, pf, q, w, TILE_BINS),
                          "Peak-frequency strata; q_h proportional to sqrt(g_h); w = g_h / q_h.")
    arms["peakfreq_u"] = (strata_rows(tiles, valid, pf, q, {h: 1.0 for h in q}, TILE_BINS),
                          "The same draw as peakfreq_w with w = 1 (the unweighted control). "
                          "Here the sum of q x w is 1 because w = 1.")
    g = genome_shares(valid, ccre)
    q, w = sqrt_design(g)
    arms["ccre_like_w"] = (strata_rows(tiles, valid, ccre, q, w, TILE_BINS),
                           "cCRE-recipe classes from training peaks only; q_h proportional to "
                           "sqrt(g_h); w = g_h / q_h.")
    dec = decile_strata(var)
    g = genome_shares(valid, dec)
    top = {f"d{i:02d}" for i in range(11 - VARIANCE_TOP_DECILES, 11)}
    m = {h: g[h] * (VARIANCE_TOP_FACTOR if h in top else 1.0) for h in g}
    q = {h: x / sum(m.values()) for h, x in m.items()}
    arms["variance_w"] = (strata_rows(tiles, valid, dec, q, {h: g[h] / q[h] for h in g}, TILE_BINS),
                          f"Deciles of the variance score (per assay, the sample variance across "
                          f"training cells of log1p of the tile's total signal in reads per million "
                          f"over its valid bins, rank-normalised to [0, 1] over tiles; then the mean "
                          f"over the assays with >= 2 training cells); the top {VARIANCE_TOP_DECILES} "
                          f"deciles at {VARIANCE_TOP_FACTOR:g} times the mass per genome share; "
                          f"w = g_h / q_h.")
    X = kcenter_features(tiles, average, embedding, genome)
    centres, sizes = kcenter(X, k)
    order = np.argsort(centres)
    rows = [{"chrom": tiles[centres[j]][0], "start_bin": tiles[centres[j]][1], "n_bins": TILE_BINS,
             "stratum": f"kc{j:05d}", "sample_weight": 1.0 / k,
             "loss_weight": float(sizes[j]) / float(sizes.mean())} for j in order]
    qs = np.quantile(sizes, [0, 0.25, 0.5, 0.75, 1.0])
    arms["kcenter"] = (rows, f"Greedy k-center, k = {k}, first centre seeded by {KCENTER_SEED}; "
                       f"{X.shape[1]} features ({len(groups)} average assays + embedding); "
                       f"each tile q = 1/k, w = cluster size / mean cluster size. Cluster sizes "
                       f"(tiles of the table): min {qs[0]:g}, quartiles {qs[1]:g}, {qs[2]:g}, "
                       f"{qs[3]:g}, max {qs[4]:g}; sum {int(sizes.sum())}. "
                       f"kcenter is a biased coreset by design: {len(tiles) - k} of {len(tiles)} "
                       f"tiles ({(len(tiles) - k) / len(tiles):.1%}) get q = 0, and the cluster "
                       f"weights restore the mass of each cluster, not the uniform loss over U.")
    n_low = len(all_tiles) - len(tiles)
    zero = {a: 1.0 - len({(r["chrom"], r["start_bin"]) for r in rows if r["sample_weight"] > 0})
            / len(tiles) for a, (rows, _) in arms.items()}
    e_tile, e_blk = edge_coverage(TILE_BINS), edge_coverage(rb)
    md = ["# Strata tables (STt, plan Amendment 7, B.4)", "",
          f"Universe: {uni}.", "",
          f"Peak-frequency denominator: {len(tracks)} training tracks of the tables that have a "
          f"peak file (role train in `peak_files.tsv`; ChIP-seq controls left out).", "",
          "g_h is the share of the valid bins of the tile table in stratum h; q_h is its sampling "
          "share; w_h is its loss weight. In every file the sum of q_h x w_h is 1.", "",
          f"Tiles below {MIN_VALID} valid get q = 0 in every tile arm: {n_low} of {len(all_tiles)} "
          f"tiles of U ({n_low / len(all_tiles):.2%}) are not in the tile table. Share of the tile "
          f"table's tiles with q = 0, per arm: " + ", ".join(
              f"{a} {zero[a]:.1%}" for a in arms) + ".", "",
          f"Window-in-tile edge effect. A window of {WINDOW_BINS} bins is drawn with a uniform start "
          f"inside its row. In a tile of {TILE_BINS} bins the first bin is covered "
          f"{e_tile[0]:.4f} times as often as the middle bin, and {e_tile[1]:.1%} of the tile's "
          f"bins are covered at less than half the middle rate; in a block of {rb} bins these are "
          f"{e_blk[0]:.6f} and {e_blk[1]:.1%}. So every tile arm under-samples tile edges far more "
          f"than the block lists do; `tile_uniform` is the control that shares this effect.", "",
          f"cCRE-like denominators: {ccre_n['cells']} training cells with a peak file; "
          f"{ccre_n['accessibility']} have an accessibility track; {ccre_n['promoter_rule']} also "
          f"have an H3K4me3 track (the cells of the promoter_like rule); {ccre_n['enhancer_rule']} "
          f"also have an H3K27ac or H3K4me1 track (the cells of the enhancer_like rule). A cell "
          f"without an H3K4me3 track never meets \"no H3K4me3 peak\".", ""]
    for name, (rows, note) in arms.items():
        write_list(out / f"{name}.tsv", rows, f"{name}: {uni}", sha)
        md += arm_summary(name, rows, tile_peak, tile_valid, note)
    md += ["## Inputs", "", "| name | path | sha256 |", "|---|---|---|"]
    md += ["| " + " | ".join(line.split("\t")) + " |" for line in manifest]
    (out / "STRATA.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"tiles": len(tiles), "tracks": len(tracks), **{a: len(r) for a, (r, _) in arms.items()}}


# ------------------------------------------------------------------------------------ classes

def run_region_classes(tile_table, evalset, tables, peaks, peak_files, out) -> list:
    head = {}
    for line in Path(tile_table).read_text(encoding="utf-8").splitlines():
        if not line.startswith("# "):
            break
        k, _, v = line[2:].partition("=")
        head[k] = v
    tracks = peak_tracks(tables, peaks, peak_files)
    if head.get("peak_tracks_sha256") != tracks_sha(tracks):
        raise ValueError(f"{tile_table} was built from other peak tracks than {peak_files}")
    ev = load_evalset(evalset)
    rb = ev["region_bins"]
    E = list(ev["regions"])
    tiles = tiles_of(E, rb)
    frac = peak_matrix(tracks, tiles).mean(axis=0).reshape(len(E), rb // TILE_BINS).mean(axis=1)
    cls = peakfreq_class(frac)
    rows = [{"chrom": c, "start_bin": s, "class": k, "peak_fraction": f}
            for (c, s), k, f in zip(E, cls, frac)]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("chrom\tstart_bin\tclass\tpeak_fraction\n")
        for r in rows:
            fh.write(f"{r['chrom']}\t{r['start_bin']}\t{r['class']}\t{_fmt(r['peak_fraction'])}\n")
    return rows


# ------------------------------------------------------------------------------------ cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("universe")
    u.add_argument("--evalset", required=True)
    u.add_argument("--genome", required=True)
    u.add_argument("--seeds", default="0,1,2")
    u.add_argument("--sizes", default=",".join(map(str, TERR_SIZES)))
    u.add_argument("--out", required=True)
    t = sub.add_parser("tiles")
    for k in ("tables", "peaks", "peak-files", "average", "dna-embedding", "genome", "evalset", "out"):
        t.add_argument(f"--{k}", required=True)
    t.add_argument("--k", type=int, default=KCENTER_K)
    r = sub.add_parser("region-classes")
    for k in ("tile-table", "evalset", "tables", "peaks", "peak-files", "out"):
        r.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "universe":
        res = run_universe(a.evalset, a.genome, [int(s) for s in a.seeds.split(",") if s],
                           a.out, tuple(int(x) for x in a.sizes.split(",") if x))
    elif a.cmd == "tiles":
        res = run_tiles(a.tables, a.peaks, a.peak_files, a.average, a.dna_embedding, a.genome,
                        a.evalset, a.out, k=a.k)
    else:
        res = {"regions": len(run_region_classes(a.tile_table, a.evalset, a.tables, a.peaks,
                                                 a.peak_files, a.out))}
    print(json.dumps(res, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
