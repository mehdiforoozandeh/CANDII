"""Resolution study: which variance source lives at which Haar scale (plan chunk RS).

An analysis, not a change of bin size. Per assay class and per Haar level j = 1..14 plus the coarse
remainder (level j compares the two halves of blocks of 2^j bins: 25 bp x 2^j), it reports the
variance in log1p counts, at ONE matched depth per assay class, of these sources:

  pseudoreplicate     counting noise of one track: half the variance of pr1 - pr2 (the two halves of
                      each base track), i.e. the noise a single track carries at the matched depth
  <rung>              each further replicate rung of `layers.RUNGS` that the pairs populate: half its
                      layer increment over the nearest populated lower rung (variance one replicate
                      adds), from `layers.layer_table` with single levels as bands
  knob:<knob>         each designed knob, base against arm, non-subset pairs only: the variance of the
                      view difference beyond counting noise,
                        V(base.pr1, arm.pr2) - (V(base.pr1, base.pr2) + V(arm.pr1, arm.pr2)) / 2,
                      where V is the pair variance of `layers.pair_variance`. The halves share no reads
                      (an arm reuses its parent's read split), so the noise terms cancel in expectation
  depth_subset        the same estimate for depth-subset pairs, reported apart and computed at the
                      pair's own matched depth (at one matched depth a pure subset has no effect: a
                      sanity row, expected 0)
  shared              the part of a base track's variance common to the cells of its assay: the mean
                      Haar Gram entry with tracks of other cells (`error_budget._gram`,
                      `errdecomp.shared_specific`, groups = cell)
  cell_specific       the rest of the track's variance minus its own counting noise:
                      Gram diagonal - shared - pseudoreplicate. Replicate-rung variance lives inside it
  total               the Gram diagonal: a base track's variance at that level. By construction
                      total = shared + cell_specific + pseudoreplicate, track by track

`share_of_level_total` divides each source by `total` at the same level (knob and rung rows can exceed
it). Per-level variances are Haar energy over the number of bins, so the 15 levels of a source add up
exactly to its plain variance (tested by a second, block-sum method). Intervals are 95% intervals from
the cluster bootstrap over cells of `layers.py` (t on n_cells - 1 df, small-sample corrected); with one
cell the interval is (-inf, inf). For shared and cell_specific each resample recomputes the shared part
from the Gram entries between distinct cells of the resample (per-track shared values are correlated
across cells, so resampling them as fixed values under-covered). `n_pairs` counts pairs
(pseudoreplicate, rung, knob rows) or tracks (shared, cell_specific, total rows).

Inputs: base views (one per cell and assay, `evalset.base_views`) of the splits read (train and val;
blind only with --include-blind) with pseudoreplicate halves (`pr1.h5` / `pr2.h5` siblings of an
`h5:` counts path); the evalset's `replicate_pairs` (else every replicate-rung pair); `pairs.tsv` rows
of rung `designed` whose one side is a base view. `pairs.tsv` rows of rung `pseudoreplicate` are not
read: the halves give that rung. Regions: the evalset's `regions` (2^14 bins each).

CLI:
  python tools/onion_eval/resolution.py run --tables <dir> --evalset <json> --out <resolution.tsv>
  python tools/onion_eval/resolution.py figure --in <resolution.tsv> --out <png>   (needs matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from eic_decomp import errdecomp  # noqa: E402
from onion_eval import error_budget as EB  # noqa: E402
from onion_eval import layers as LY  # noqa: E402

J = errdecomp.J
L = 2 ** J
BIN_BP = errdecomp.BIN_BP
LEVELS = [str(j) for j in range(1, J + 1)] + ["remainder"]
LEVEL_BANDS = {lv: [i] for i, lv in enumerate(LEVELS)}          # Haar scale index i = j - 1; J = remainder
COLUMNS = ["assay_class", "level", "bin_bp", "source", "variance", "share_of_level_total", "n_pairs",
           "n_cells", "ci_low", "ci_high"]
PARTS = ("shared", "cell_specific", "total")


def bin_bp(level: str) -> int:
    """Bin width of a level: 25 bp x 2^j; the coarse remainder has the width of the 2^J block."""
    return BIN_BP * 2 ** (J if level == "remainder" else int(level))


# ---------------------------------------------------------------------------------------------------
# Per-pair and per-track pieces

def level_variance(a, b, depth_a, depth_b, depth, seed=0):
    """Pair variance of `layers.pair_variance` at each single Haar level: array (15,), adding up to the
    plain variance of the depth-matched, centred log1p difference."""
    pv = LY.pair_variance(a, b, depth_a, depth_b, None, LEVEL_BANDS, depth=depth, seed=seed)
    return np.array([pv["variance"]["all"][lv] for lv in LEVELS])


def track_gram(tracks):
    """Haar Gram per level of centred tracks over bins (error_budget's Gram code): (15, T, T).
    tracks: (T, M), M a multiple of 2^J (regions end to end; the transform never mixes two regions)."""
    X = np.asarray(tracks, np.float64)
    X = X - X.mean(1, keepdims=True)
    return EB._gram(X, X) / X.shape[1]


def track_parts(tracks, groups):
    """Shared, specific and total variance per level of each track: (shared, specific, total), each
    (T, 15); shared is NaN for a track with no track of another cell (`errdecomp.shared_specific`)."""
    G = track_gram(tracks)
    sh, sp = errdecomp.shared_specific(G, np.asarray(groups))
    return sh.T, sp.T, np.einsum("kbb->bk", G)


def _cell_weights(n_cells, n_boot, seed):
    """Cluster-bootstrap weights (layers.py scheme): (n_boot, n_cells) copies of each cell."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_cells, size=(n_boot, n_cells))
    return np.stack([np.bincount(d, minlength=n_cells) for d in draws]).astype(float)


def parts_boot(grams, wc, cell_index):
    """Track means of shared, cell_specific and total per level under cell weights wc (n, C).

    grams: list of (G (K, T, T), cells (T,), noise (T, K)), one per assay. A track's shared part is the
    weighted mean of its Gram entries with tracks of OTHER cells; with wc all ones this is
    `track_parts` averaged over tracks. Returns {source: (n, K)}."""
    wc = np.atleast_2d(np.asarray(wc, float))
    num = {s: 0.0 for s in PARTS}
    den = {s: 0.0 for s in PARTS}
    for G, cells, noise in grams:
        cells = np.asarray(cells)
        wt = wc[:, [cell_index[c] for c in cells]]                      # (n, T)
        other = (cells[:, None] != cells[None, :]).astype(float)
        diag = np.einsum("kbb->bk", G)                                   # (T, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.einsum("kbc,bc,nc->nbk", G, other, wt) / (wt @ other.T)[..., None]
        vals = {"shared": sh, "cell_specific": diag[None] - sh - np.asarray(noise)[None],
                "total": np.broadcast_to(diag[None], sh.shape)}
        for s, v in vals.items():
            ok = np.isfinite(v)
            w = wt[..., None] * ok
            num[s] = num[s] + (w * np.where(ok, v, 0.0)).sum(1)
            den[s] = den[s] + w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {s: num[s] / den[s] for s in PARTS}


def mean_ci(vals, cells, n_boot=1000, seed=0):
    """Mean over records and its 95% interval from the cluster bootstrap over cells (layers.py scheme).
    vals: (n, K) with NaN for a missing value; cells: (n,). Returns (mean, lo, hi, n_used, n_cells),
    each (K,)."""
    vals = np.asarray(vals, float)
    cells = np.asarray(cells)
    uc = sorted(set(cells.tolist()))
    w = _cell_weights(len(uc), n_boot, seed)[:, [uc.index(c) for c in cells]]
    K = vals.shape[1]
    mean, lo, hi = np.full(K, np.nan), np.full(K, np.nan), np.full(K, np.nan)
    n_used, n_cells = np.zeros(K, int), np.zeros(K, int)
    for k in range(K):
        ok = np.isfinite(vals[:, k])
        if not ok.any():
            continue
        v, ww = vals[ok, k], w[:, ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (ww * v).sum(1) / ww.sum(1)
        mean[k] = float(v.mean())
        n_used[k], n_cells[k] = int(ok.sum()), len(set(cells[ok].tolist()))
        lo[k], hi[k] = LY._interval(mean[k], boot, n_cells[k])
    return mean, lo, hi, n_used, n_cells


# ---------------------------------------------------------------------------------------------------
# One assay class

def class_rows(cls, units, n_boot=1000, seed=0):
    """resolution.tsv rows of one assay class from its per-unit records.

    units: dict with
      "pr":     list of {"cell", "v": (15,)}  pseudoreplicate pair variance of each base track
      "rungs":  list of {"rung", "cell", "v": (15,)}  replicate-rung pair variances
      "knobs":  list of {"source", "cell", "v": (15,)}  knob estimate per base-arm pair
      "grams":  list of (G (15, T, T), cells (T,), noise (T, 15)) per assay: the Haar Gram of its base
                tracks and each track's counting noise (its pseudoreplicate pair variance / 2)
    """
    rows = []

    def emit(source, mean, lo, hi, n_used, n_cells):
        for i, lv in enumerate(LEVELS):
            rows.append(dict(assay_class=cls, level=lv, bin_bp=bin_bp(lv), source=source,
                             variance=float(mean[i]), n_pairs=int(n_used[i]), n_cells=int(n_cells[i]),
                             ci_low=float(lo[i]), ci_high=float(hi[i])))

    # replicate ladder: halves give the pseudoreplicate rung, then layers' increments, all per track (/2)
    recs = [dict(rung="pseudoreplicate", cell=r["cell"], v={("all", lv): x for lv, x in zip(LEVELS, r["v"])})
            for r in units["pr"]]
    recs += [dict(rung=r["rung"], cell=r["cell"], v={("all", lv): x for lv, x in zip(LEVELS, r["v"])})
             for r in units["rungs"]]
    if recs:
        lt = LY.layer_table(recs, n_boot=n_boot, seed=seed)
        for rung in LY.RUNGS:
            got = {r["band"]: r for r in lt if r["rung"] == rung}
            if not got or got[LEVELS[0]]["n_pairs"] == 0:
                continue
            key = "variance" if rung == "pseudoreplicate" else "layer_increment"
            g = [got[lv] for lv in LEVELS]
            emit(rung, np.array([x[key] for x in g]) / 2, np.array([x["ci_low"] for x in g]) / 2,
                 np.array([x["ci_high"] for x in g]) / 2, [x["n_pairs"] for x in g], [x["n_cells"] for x in g])

    for source in sorted({k["source"] for k in units["knobs"]}):
        ks = [k for k in units["knobs"] if k["source"] == source]
        emit(source, *mean_ci([k["v"] for k in ks], [k["cell"] for k in ks], n_boot, seed))

    grams = units["grams"]
    if grams:
        uc = sorted({c for _, cells, _ in grams for c in cells})
        cidx = {c: i for i, c in enumerate(uc)}
        est = parts_boot(grams, np.ones(len(uc)), cidx)
        boot = parts_boot(grams, _cell_weights(len(uc), n_boot, seed), cidx)
        n_tracks = sum(len(cells) for _, cells, _ in grams)
        for source in PARTS:
            e = est[source][0]
            lo, hi = np.full(len(LEVELS), np.nan), np.full(len(LEVELS), np.nan)
            for i in range(len(LEVELS)):
                if np.isfinite(e[i]):
                    lo[i], hi[i] = LY._interval(float(e[i]), boot[source][:, i], len(uc))
            emit(source, e, lo, hi, [n_tracks] * len(LEVELS), [len(uc)] * len(LEVELS))

    tot = {r["level"]: r["variance"] for r in rows if r["source"] == "total"}
    for r in rows:
        t = tot.get(r["level"], np.nan)
        r["share_of_level_total"] = r["variance"] / t if np.isfinite(t) and t > 0 else np.nan
    return rows


# ---------------------------------------------------------------------------------------------------
# Reading the tables

def _half_path(counts_path, half):
    kind, _, rest = counts_path.partition(":")
    if kind != "h5":
        return None
    p = Path(rest).with_name(f"{half}.h5")
    return f"h5:{p}" if p.is_file() else None


def _depth(m):
    return float(m["depth_reads"] or m["depth_fragments"])


def run(tables, evalset, out, n_boot=1000, seed=0, include_blind=False):
    from onion_eval.evalset import base_views

    tables = Path(tables)
    ev = json.loads(Path(evalset).read_text())
    rows = LY._read_tsv(tables / "measurements.tsv")
    meas = {r["measurement_id"]: r for r in rows}
    pairs = LY._read_tsv(tables / "pairs.tsv") if (tables / "pairs.tsv").exists() else []
    roles = ("train", "val", "blind") if include_blind else ("train", "val")
    regions = [(c, int(s)) for c, s in ev["regions"]]
    reader = LY.CountsReader()

    def read(path):
        return np.stack([reader.read(path, c, s, s + L) for c, s in regions])

    def halves(m):
        p1, p2 = _half_path(m["counts_path"], "pr1"), _half_path(m["counts_path"], "pr2")
        return (p1, p2) if p1 and p2 else None

    all_base = base_views(rows, roles)
    base = {k: m for k, m in all_base.items() if halves(m)}
    base_ids = {m["measurement_id"] for m in base.values()}

    want = ev.get("replicate_pairs")
    rep = [p for p in pairs if p["rung"] in LY.RUNGS[1:] and (want is None or p["pair_id"] in want)
           and p["split"] in roles and p["a"] in meas and p["b"] in meas]
    knob_pairs, seen = [], set()
    for p in pairs:
        if p["rung"] != "designed" or p["split"] not in roles or p["a"] not in meas or p["b"] not in meas:
            continue
        a, b = (p["a"], p["b"]) if p["a"] in base_ids else (p["b"], p["a"])
        if a not in base_ids or b in base_ids or (a, b) in seen or not halves(meas[b]):
            continue
        seen.add((a, b))
        knob_pairs.append((p, a, b))

    # one matched depth per assay class: the shallowest half or replicate track used outside depth subsets
    common = {}

    def lower(cls, d):
        common[cls] = min(common.get(cls, np.inf), d)

    for m in base.values():
        lower(m["assay_class"], _depth(m) / 2)
    for p, a, b in knob_pairs:
        if p["subset_pair"] != "1":
            lower(meas[b]["assay_class"], _depth(meas[b]) / 2)
    for p in rep:
        lower(meas[p["a"]]["assay_class"], min(_depth(meas[p["a"]]), _depth(meas[p["b"]])))

    units = {c: {"pr": [], "rungs": [], "knobs": [], "grams": []} for c in common}
    pr_cache = {}

    def pr_var(m, depth):
        key = (m["measurement_id"], depth)
        if key not in pr_cache:
            h1, h2 = halves(m)
            d2 = _depth(m) / 2
            pr_cache[key] = level_variance(read(h1), read(h2), d2, d2, depth,
                                           seed=zlib.crc32(f"pr:{m['measurement_id']}".encode()))
        return pr_cache[key]

    by_assay = {}
    for (cell, assay), m in sorted(base.items()):
        v = pr_var(m, common[m["assay_class"]])
        units[m["assay_class"]]["pr"].append(dict(cell=cell, v=v))
        by_assay.setdefault(assay, []).append((m, v))
    for assay, ms in sorted(by_assay.items()):
        cls = ms[0][0]["assay_class"]
        D = common[cls]
        rng = np.random.default_rng(zlib.crc32(f"tracks:{assay}".encode()))
        X = np.stack([np.log1p(LY.thin(read(m["counts_path"]), D / _depth(m), rng)).reshape(-1)
                      for m, _ in ms])
        units[cls]["grams"].append((track_gram(X), [m["cell"] for m, _ in ms],
                                    np.stack([v / 2 for _, v in ms])))
    for p in rep:
        ma, mb = meas[p["a"]], meas[p["b"]]
        v = level_variance(read(ma["counts_path"]), read(mb["counts_path"]), _depth(ma), _depth(mb),
                           common[ma["assay_class"]], seed=zlib.crc32(p["pair_id"].encode()))
        units[ma["assay_class"]]["rungs"].append(dict(rung=p["rung"], cell=p["cell"], v=v))
    for p, a, b in knob_pairs:
        ma, mb = meas[a], meas[b]
        subset = p["subset_pair"] == "1"
        cls = ma["assay_class"]
        D = min(_depth(ma), _depth(mb)) / 2 if subset else common[cls]
        cross = level_variance(read(halves(ma)[0]), read(halves(mb)[1]), _depth(ma) / 2, _depth(mb) / 2, D,
                               seed=zlib.crc32(f"x:{p['pair_id']}".encode()))
        v = cross - (pr_var(ma, D) + pr_var(mb, D)) / 2
        units[cls]["knobs"].append(dict(source="depth_subset" if subset else f"knob:{p['knobs_differing']}",
                                        cell=p["cell"], v=v))

    out_rows = []
    for cls in sorted(units):
        out_rows += class_rows(cls, units[cls], n_boot=n_boot, seed=seed)
    write_tsv(out_rows, out)
    if len(all_base) > len(base):
        print(f"resolution: {len(all_base) - len(base)} base views without pseudoreplicate halves skipped",
              file=sys.stderr)
    return out_rows


def write_tsv(rows, path):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else repr(v)
        return str(v)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([fmt(r[c]) for c in COLUMNS])


# ---------------------------------------------------------------------------------------------------
# Figure (cluster only: matplotlib is not in the laptop env)

def figure(tsv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = LY._read_tsv(tsv)
    classes = sorted({r["assay_class"] for r in rows})
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4), squeeze=False)
    for ax, cls in zip(axes[0], classes):
        rs = [r for r in rows if r["assay_class"] == cls]
        for src in sorted({r["source"] for r in rs if r["source"] != "total"}):
            pts = [(LEVELS.index(r["level"]), float(r["variance"])) for r in rs
                   if r["source"] == src and r["variance"] != ""]
            if pts:
                x, y = zip(*sorted(pts))
                ax.plot(x, y, marker="o", ms=3, label=src)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([f"{bin_bp(lv) / 1000:g}k" if bin_bp(lv) >= 1000 else str(bin_bp(lv))
                            for lv in LEVELS], rotation=90, fontsize=7)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_title(cls)
        ax.set_xlabel("Haar level (bin width, bp)")
        ax.set_ylabel("variance in log1p counts")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tables", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--include-blind", action="store_true")
    f = sub.add_parser("figure")
    f.add_argument("--in", dest="inp", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "figure":
        figure(args.inp, args.out)
        return 0
    rows = run(args.tables, args.evalset, args.out, args.n_boot, args.seed, args.include_blind)
    print(f"resolution: {len(rows)} rows, {len({r['assay_class'] for r in rows})} assay classes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
