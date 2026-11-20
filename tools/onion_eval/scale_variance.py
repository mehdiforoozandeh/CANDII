"""Variance by scale on the 1 bp region store (plan Amendment 3, section C; chunk RVt).

Question: how much variance does a 25 bp bin lose? For a block size b with block index Z, the law of
total variance over positions gives `Var(Y_1bp) = E[Var(Y|Z)] + Var(E[Y|Z])`. The first term is the
variance below b ("within"), the second the variance between blocks of b ("between"). Blocks never
cross a region boundary; the mean and the variances run over every position of every region.

Chains. A chain is a nested list of block sizes, each dividing the next (1,5,25,125 or 1,5,25,50).
Span rule (section C): each region is cut to its first floor(L / b_max) * b_max bp, i.e. 409,500 bp
for the chain that ends at 125 bp and the whole 409,600 bp for the chain that ends at 50 bp. One row
per step (from_bp, to_bp) of a chain.

Two scales.
  counts  exact and additive. The point-rule 1 bp counts are summed into blocks, so `within` at a step
          is the variance below to_bp, cumulative from 1 bp: within(to_bp) + between(to_bp) = total.
          share_raw = within(to_bp) / total, on the full track.
  log1p   of the block sums at from_bp, u = log1p(sum of x over a from_bp block). The same identity
          holds for u over the to_bp blocks, but u is not additive across steps. So on log1p a row
          is the share of the from_bp log1p variance that lies below to_bp (per step, not cumulative).

Replicate-reproducible share. Two measurements a and b of one pair (the pseudoreplicate halves pr1
and pr2, two biological replicates, or two cell types) are independent given what they share, so
their cross-covariance carries no sampling term. The cross-covariance splits over the same blocks
exactly (the block means are nested orthogonal projections): cross_within(b) + cross_between(b) =
cross_total. Its within part is the variance below b that the pair reproduces.
  counts  share_reproducible = [cross_within(to_bp) / (s_a s_b)] / [total_full / S^2], with s the
          sums of each track over the span (the depth normalization). The denominator is the full
          track's total variance (default full = pr1 + pr2), the same as share_raw's. So
          share_raw - share_reproducible is the share of the sampling noise below to_bp. Without a
          full track (two cell types) the denominator is the mean of the two normalized totals.
  log1p   share_reproducible = cross_within / (0.5 (total_a + total_b)), all of u at from_bp. The
          halves are at half depth, so on log1p this share is at half depth, and the log1p variance
          depends on depth: compare log1p rows only with log1p rows of the same pair kind.

Layer of a row (`layer_or_all`). `all`: every selected pooled track, share_raw only. A rung of
`layers.RUNGS`: share_reproducible from the pairs of that rung: `pseudoreplicate` (pr1, pr2 of each
pooled track; full = the pooled track), `isogenic` (br<i> pairs of pairs.tsv; full = the pooled track
of the experiment), `other_cell_type` (pairs.tsv rows, else every pair of pooled tracks of one assay
in two cells; no full track). The reproducible share at a rung is the variance below b shared beyond
that rung; a layer is the difference of two rungs. Rungs with no pair get a row with n_tracks = 0 and
empty shares (`not populated`): the store holds no technical or sequencing views.

Aggregation: the mean over a row's records (tracks or pairs) of the per-record shares. Interval: 95%
cluster bootstrap over cells, as `layers.py` (`layers._interval`: t quantile on n_cells - 1 df and
the sqrt(n / (n - 1)) correction), on share_reproducible, or on share_raw for `all` rows.

CLI:
  python tools/onion_eval/scale_variance.py run --store <res dir> --tables <eic_res_v1> \
      --chains 1,5,25,125 1,5,25,50 --out scale_variance.tsv
  python tools/onion_eval/scale_variance.py figure --in scale_variance.tsv --out scale_variance.png
      (needs matplotlib: cluster only)
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import layers as LY  # noqa: E402

SCALES = ("counts", "log1p")
COLUMNS = ["assay", "layer_or_all", "chain", "from_bp", "to_bp", "scale", "share_raw",
           "share_reproducible", "n_tracks", "n_cells", "ci_low", "ci_high"]
POPULATED_HERE = ("pseudoreplicate", "isogenic", "other_cell_type")


# ---------------------------------------------------------------------------------------------------
# Blocks and moments

def parse_chain(chain):
    """'1,5,25,125' or a sequence -> tuple of ints; each size must divide the next."""
    if isinstance(chain, str):
        chain = [int(c) for c in chain.split(",") if c]
    chain = tuple(int(c) for c in chain)
    if len(chain) < 2 or chain[0] < 1:
        raise ValueError(f"chain {chain} needs at least two sizes, the first >= 1")
    for a, b in zip(chain, chain[1:]):
        if b <= a or b % a:
            raise ValueError(f"chain {chain} is not nested: {a} does not divide {b}")
    return chain


def span(L, chain):
    """Span rule of section C: the first floor(L / b_max) * b_max positions of a region."""
    return (int(L) // chain[-1]) * chain[-1]


def pool(x, b):
    """Exact block sums of size b along the last axis; a trailing partial block is dropped."""
    x = np.asarray(x)
    n = (x.shape[-1] // b) * b
    return x[..., :n].reshape(*x.shape[:-1], n // b, b).sum(-1)


class _Moments:
    """Running sums for the block identity of x against y (y = x for a variance).

    Over n positions: total = mean(xy) - mean(x) mean(y); between(k) = sum over blocks of size k of
    X_k Y_k / k / n - mean(x) mean(y); within(k) = total - between(k)."""

    def __init__(self, sizes):
        self.sizes = tuple(sizes)
        self.n, self.sx, self.sy, self.sxy = 0, 0.0, 0.0, 0.0
        self.q = {k: 0.0 for k in self.sizes}

    def add(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxy += float(np.sum(x * y))
        for k in self.sizes:
            self.q[k] += float(np.sum(pool(x, k) * pool(y, k))) / k

    def parts(self, k):
        n = self.n
        mxy, mm, qk = self.sxy / n, (self.sx / n) * (self.sy / n), self.q[k] / n
        return {"within": mxy - qk, "between": qk - mm, "total": mxy - mm}


class _ChainMoments:
    """The block identity of one chain on one scale, accumulated region by region."""

    def __init__(self, chain, scale):
        if scale not in SCALES:
            raise ValueError(f"scale {scale!r} not in {SCALES}")
        self.chain, self.scale = parse_chain(chain), scale
        self.steps = list(zip(self.chain, self.chain[1:]))
        if scale == "counts":
            base = self.chain[0]
            self.acc = _Moments([t // base for t in self.chain[1:]])
        else:
            self.acc = [_Moments([t // f]) for f, t in self.steps]

    def add(self, x, y):
        """x, y: one region (L,) or regions (R, L) at 1 bp; cut by the span rule here."""
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_2d(np.asarray(y))
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
        s = span(x.shape[-1], self.chain)
        if s == 0:
            raise ValueError(f"region length {x.shape[-1]} is shorter than the block {self.chain[-1]}")
        x, y = x[:, :s], y[:, :s]
        if self.scale == "counts":
            base = self.chain[0]
            self.acc.add(pool(x, base) if base > 1 else x, pool(y, base) if base > 1 else y)
        else:
            for (f, _t), acc in zip(self.steps, self.acc):
                acc.add(np.log1p(pool(x, f).astype(np.float64)),
                        np.log1p(pool(y, f).astype(np.float64)))

    def rows(self):
        out = []
        for i, (f, t) in enumerate(self.steps):
            if self.scale == "counts":
                p, s = self.acc.parts(t // self.chain[0]), (self.acc.sx, self.acc.sy)
            else:
                p, s = self.acc[i].parts(t // f), (self.acc[i].sx, self.acc[i].sy)
            out.append(dict(from_bp=f, to_bp=t, scale=self.scale, sum_x=s[0], sum_y=s[1], **p))
        return out


def block_identity(x_1bp, chain, scale="counts"):
    """Within and between per step of `chain` for one track.

    x_1bp: 1 bp values of one region (L,) or of R regions (R, L). Returns one dict per step with
    from_bp, to_bp, scale, within, between, total (within + between = total to rounding) and
    share_raw = within / total. On counts `within` is the variance below to_bp from 1 bp; on log1p it
    is the variance of log1p(block sums at from_bp) below to_bp."""
    m = _ChainMoments(chain, scale)
    m.add(x_1bp, x_1bp)
    rows = m.rows()
    for r in rows:
        del r["sum_x"], r["sum_y"]
        r["share_raw"] = r["within"] / r["total"] if r["total"] > 0 else np.nan
    return rows


class _PairShares:
    """Shares of one record: a pair (a, b) with an optional full track (full=None: no full track)."""

    def __init__(self, chain, scale, has_full=True):
        self.scale, self.has_full = scale, has_full
        self.cross = _ChainMoments(chain, scale)
        self.aa = _ChainMoments(chain, scale)
        self.bb = _ChainMoments(chain, scale)
        self.ff = _ChainMoments(chain, scale) if has_full else None

    def add(self, a, b, full=None):
        self.cross.add(a, b)
        self.aa.add(a, a)
        self.bb.add(b, b)
        if self.has_full:
            self.ff.add(full, full)

    def rows(self):
        out = []
        for c, ra, rb, rf in zip(self.cross.rows(), self.aa.rows(), self.bb.rows(),
                                 self.ff.rows() if self.has_full else [None] * len(self.aa.rows())):
            sa, sb = c["sum_x"], c["sum_y"]

            def share(r):
                return r["within"] / r["total"] if r["total"] > 0 else np.nan
            if rf is not None:
                raw = share(rf)
            else:
                raw = 0.5 * (share(ra) + share(rb))
            if self.scale == "counts":
                if rf is not None:
                    den = rf["total"] / rf["sum_x"] ** 2 if rf["sum_x"] > 0 else np.nan
                else:
                    den = 0.5 * (ra["total"] / sa ** 2 + rb["total"] / sb ** 2) if sa > 0 and sb > 0 else np.nan
                num = c["within"] / (sa * sb) if sa > 0 and sb > 0 else np.nan
            else:
                den = 0.5 * (ra["total"] + rb["total"])
                num = c["within"]
            rep = num / den if den and np.isfinite(den) and den > 0 else np.nan
            out.append(dict(from_bp=c["from_bp"], to_bp=c["to_bp"], scale=self.scale,
                            cross_within=c["within"], cross_between=c["between"],
                            cross_total=c["total"], share_raw=raw, share_reproducible=rep))
        return out


def reproducible_share(pr1, pr2, chain, scale="counts", full="sum"):
    """Replicate-reproducible share per step of `chain` from the cross-covariance of two halves.

    pr1, pr2: 1 bp values on the same regions, (L,) or (R, L). full: "sum" (default) takes the full
    track as pr1 + pr2; an array gives the full track; None means no full track (a pair of two cell
    types: shares are taken against the mean of the two tracks). Returns one dict per step with
    cross_within, cross_between, cross_total (they add up), share_raw (of the full track) and
    share_reproducible (module docstring)."""
    if isinstance(full, str):
        if full != "sum":
            raise ValueError(f"full={full!r}")
        full = np.asarray(pr1, np.int64) + np.asarray(pr2, np.int64)
    ps = _PairShares(chain, scale, has_full=full is not None)
    ps.add(pr1, pr2, full)
    return ps.rows()


# ---------------------------------------------------------------------------------------------------
# Aggregation over records

def aggregate(records, n_boot=1000, seed=0):
    """records: dicts {assay, layer, chain, scale, from_bp, to_bp, cell, share_raw, share_reproducible}.
    Returns the rows of scale_variance.tsv (without the not-populated rows)."""
    groups = {}
    for r in records:
        key = (r["assay"], r["layer"], r["chain"], r["from_bp"], r["to_bp"], r["scale"])
        groups.setdefault(key, []).append(r)
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[5], k[3])):
        recs = groups[key]
        cells = sorted({r["cell"] for r in recs})
        cidx = {c: i for i, c in enumerate(cells)}
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(cells), size=(n_boot, len(cells)))
        weights = np.stack([np.bincount(d, minlength=len(cells)) for d in draws]).astype(float)
        on = "share_raw" if key[1] == "all" else "share_reproducible"
        ok = [r for r in recs if np.isfinite(r[on])]

        def mean(field):
            v = np.array([r[field] for r in ok], float)
            return float(np.nanmean(v)) if len(v) else np.nan
        est = mean(on)
        if ok:
            vals = np.array([r[on] for r in ok], float)
            w = weights[:, [cidx[r["cell"]] for r in ok]]
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (w * vals).sum(1) / w.sum(1)
            lo, hi = LY._interval(est, boot, len({r["cell"] for r in ok}))
        else:
            lo, hi = np.nan, np.nan
        rows.append(dict(assay=key[0], layer_or_all=key[1], chain=key[2], from_bp=key[3],
                         to_bp=key[4], scale=key[5], share_raw=mean("share_raw"),
                         share_reproducible=mean("share_reproducible") if key[1] != "all" else np.nan,
                         n_tracks=len(recs), n_cells=len(cells), ci_low=lo, ci_high=hi))
    return rows


def write_tsv(rows, path):
    def fmt(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else repr(v)
        return str(v)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([fmt(r[c]) for c in COLUMNS])


# ---------------------------------------------------------------------------------------------------
# The region store

class _Store:
    """Reads /r<i> datasets of <store>/<measurement_id>/<name>.h5 (pinned interface E)."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, mid, name):
        return self.root / mid / f"{name}.h5"

    def has(self, mid, name):
        return self.path(mid, name).exists()

    def region(self, mid, name, idx):
        import h5py
        with h5py.File(self.path(mid, name), "r") as f:
            return np.asarray(f[f"r{idx}"][:])


def _eval_regions(tables):
    rows = LY._read_tsv(Path(tables) / "regions.tsv")
    return [i for i, r in enumerate(rows) if r["role"] == "eval"]


def _feed(acc_list, store, specs, regions, max_regions=None):
    """specs: per accumulator (mid_a, name_a, mid_b, name_b, mid_full or None)."""
    for idx in regions[:max_regions] if max_regions else regions:
        cache = {}

        def get(mid, name):
            k = (mid, name)
            if k not in cache:
                cache[k] = store.region(mid, name, idx)
            return cache[k]
        for acc, (ma, na, mb, nb, mf) in zip(acc_list, specs):
            a, b = get(ma, na), get(mb, nb)
            for ps in acc:
                ps.add(a, b, get(mf, "point1bp") if mf else None)


def run(store, tables, chains, out, split="val", n_boot=1000, seed=0, max_regions=None):
    store, tables = _Store(store), Path(tables)
    chains = [parse_chain(c) for c in chains]
    chain_str = {c: ",".join(map(str, c)) for c in chains}
    meas = LY._read_tsv(tables / "measurements.tsv")
    pooled = [m for m in meas if m["unit"] == "pooled" and m["in_eic"] == split
              and (m.get("arm_knob") or "") in ("", "base") and store.has(m["measurement_id"], "point1bp")]
    by_mid = {m["measurement_id"]: m for m in meas}
    pooled_of = {m["experiment"]: m["measurement_id"] for m in pooled}
    regions = _eval_regions(tables)

    # records: (assay, layer, cell, spec, has_full)
    items = []
    for m in pooled:
        mid = m["measurement_id"]
        if store.has(mid, "point1bp_pr1") and store.has(mid, "point1bp_pr2"):
            # its full-track share_raw also gives the track's `all` record (below)
            items.append((m["assay"], "pseudoreplicate", m["cell"],
                          (mid, "point1bp_pr1", mid, "point1bp_pr2", mid), True))
        else:
            items.append((m["assay"], "all", m["cell"], (mid, "point1bp", mid, "point1bp", mid), True))
    pairs = LY._read_tsv(tables / "pairs.tsv") if (tables / "pairs.tsv").exists() else []
    have_oct = False
    for p in pairs:
        a, b = p["a"], p["b"]
        if p["rung"] not in POPULATED_HERE or p["rung"] == "pseudoreplicate":
            continue
        if not (store.has(a, "point1bp") and store.has(b, "point1bp")):
            continue
        if a not in by_mid or by_mid[a]["in_eic"] != split:
            continue
        if p["rung"] == "isogenic":
            full = pooled_of.get(by_mid[a]["experiment"])
            items.append((p["assay"], "isogenic", p["cell"], (a, "point1bp", b, "point1bp", full),
                          full is not None))
        else:
            have_oct = True
            items.append((p["assay"], "other_cell_type", p["cell"],
                          (a, "point1bp", b, "point1bp", None), False))
    if not have_oct:
        by_assay = {}
        for m in pooled:
            by_assay.setdefault(m["assay"], []).append(m)
        for assay, ms in by_assay.items():
            for x, y in combinations(sorted(ms, key=lambda m: m["measurement_id"]), 2):
                if x["cell"] != y["cell"]:
                    items.append((assay, "other_cell_type", x["cell"],
                                  (x["measurement_id"], "point1bp", y["measurement_id"], "point1bp",
                                   None), False))

    accs = [[_PairShares(c, s, has_full=hf) for c in chains for s in SCALES]
            for (_a, _l, _c, _sp, hf) in items]
    _feed(accs, store, [it[3] for it in items], regions, max_regions)

    records = []
    for (assay, layer, cell, _sp, _hf), acc in zip(items, accs):
        for ps, (c, s) in zip(acc, [(c, s) for c in chains for s in SCALES]):
            for r in ps.rows():
                base = dict(assay=assay, cell=cell, chain=chain_str[c], scale=s,
                            from_bp=r["from_bp"], to_bp=r["to_bp"], share_raw=r["share_raw"])
                records.append(dict(base, layer=layer, share_reproducible=r["share_reproducible"]))
                if layer == "pseudoreplicate":
                    records.append(dict(base, layer="all", share_reproducible=np.nan))
    rows = aggregate(records, n_boot=n_boot, seed=seed)

    # every rung of layers.RUNGS is printed; the ones without pairs as not populated
    assays = sorted({m["assay"] for m in pooled})
    have = {(r["assay"], r["layer_or_all"]) for r in rows}
    empty = []
    for a in assays:
        for rung in LY.RUNGS:
            if (a, rung) in have:
                continue
            empty.append((a, rung))
            for c in chains:
                for s in SCALES:
                    for f, t in zip(c, c[1:]):
                        rows.append(dict(assay=a, layer_or_all=rung, chain=chain_str[c], from_bp=f,
                                         to_bp=t, scale=s, share_raw=np.nan,
                                         share_reproducible=np.nan, n_tracks=0, n_cells=0,
                                         ci_low=np.nan, ci_high=np.nan))
    write_tsv(rows, out)
    return rows, empty


# ---------------------------------------------------------------------------------------------------
# Figure (cluster only: matplotlib is not in the laptop env)

def figure(tsv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in LY._read_tsv(tsv) if r["scale"] == "counts"]
    chains = sorted({r["chain"] for r in rows})
    assays = sorted({r["assay"] for r in rows})
    fig, axes = plt.subplots(1, len(chains), figsize=(6 * len(chains), 4), squeeze=False)
    for ax, ch in zip(axes[0], chains):
        for a in assays:
            for layer, field, ls in (("all", "share_raw", "-"),
                                     ("pseudoreplicate", "share_reproducible", "--")):
                pts = [(int(r["to_bp"]), float(r[field]), r["ci_low"], r["ci_high"]) for r in rows
                       if r["chain"] == ch and r["assay"] == a and r["layer_or_all"] == layer
                       and r[field] != ""]
                if not pts:
                    continue
                pts.sort()
                x, y = [p[0] for p in pts], [p[1] for p in pts]
                line, = ax.plot(x, y, ls, marker="o", ms=3,
                                label=f"{a} {'raw' if layer == 'all' else 'reproducible'}")
                lo = [float(p[2]) if p[2] != "" else np.nan for p in pts]
                hi = [float(p[3]) if p[3] != "" else np.nan for p in pts]
                ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(f"chain {ch} bp (counts)")
        ax.set_xlabel("block size (bp)")
        ax.set_ylabel("share of variance below the block size")
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--store", required=True, help="region store root (/scratch/mforooz/onion/res)")
    r.add_argument("--tables", required=True, help="eic_res_v1 tables dir")
    r.add_argument("--chains", nargs="+", default=["1,5,25,125", "1,5,25,50"])
    r.add_argument("--out", required=True)
    r.add_argument("--split", default="val")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-regions", type=int, default=None, help="smoke runs only")
    f = sub.add_parser("figure")
    f.add_argument("--in", dest="inp", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "figure":
        figure(args.inp, args.out)
        return 0
    rows, empty = run(args.store, args.tables, args.chains, args.out, args.split, args.n_boot,
                      args.seed, args.max_regions)
    for a, rung in empty:
        print(f"scale_variance: {a} {rung}: not populated")
    print(f"scale_variance: {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
