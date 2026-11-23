#!/usr/bin/env python3
"""Observed-data ladder probe (crux task t390): what each step from reads to the EIC board costs.

No model is involved. For one blind experiment, every rung is an "entry" track on the EIC's 25 bp
grid, written as the challenge's own .npy format (a pickled {chrom: float32} dict on the raw
ceil(len / 25) grid, blacklist not removed) or linked as a bigWig, then scored by
`tools/onion_eval/eic_score.py score` (the exact EIC scorer) against both truths.

  L1  Jacob Schreiber's single-end (SE) track, raw                    -> link to the bigWig
  L2  L1 quantile-normalized by Max's rule, Jacob's own narrowPeak     -> `qnorm`
  L3  our pipeline's -log10 p bigWig, raw                              -> link to the bigWig
  L4  L3 quantile-normalized by the same rule, our pipeline narrowPeak -> `qnorm`
  L5  MACS2's formula from 1 bp read 5' ends (treatment + control)     -> `formula`
  L6  the count route: 25 bp overlap counts -> signal_from_counts.py   -> `counts_route`
      (plug_in, box_d, own observed control)
  L7  L6 quantile-normalized, peaks called on L6                       -> `callpeaks`, `qnorm`

Subcommands (all CPU, numpy/scipy/pandas only; no torch: the `candi` package __init__ is
bypassed by module stubs so that `candi.bench.annotations` and `candi.store.pval_from_counts`
load alone):

  starts        one pass over a tagAlign: 5' positions by strand per chromosome, the store-rule
                25 bp overlap counts (tools/t112/bin25.py::counts_from_tagalign), line count and
                mean read length -> <out>.npz + <out>.json
  formula       L5 from `starts` files (credit: Max Libbrecht, sprint 10, see FORMULA below)
  counts_route  L6 from `starts` files (tools/onion_eval/signal_from_counts.py::signal_track)
  qnorm         Max's per-base peak/background quantile normalization of a bigWig or 25 bp .npy
  callpeaks     threshold/merge/min-length peak calling on a 25 bp track (approximation of
                MACS2 callpeak -p 0.01; see cmd_callpeaks)
  peakcheck     base-level Jaccard of two BED files (validates `callpeaks` against MACS2)
  compare       normalized MSE between two tracks on the scorer's grid (blacklist deleted)
  entrants      score the 25 published entries on given experiments against one truth
  placement     rank each rung among the 25 entrants per measure (all23 chromosome set)
  eic_score ... passes the rest to tools/onion_eval/eic_score.py's CLI (needs the stubs)

FORMULA (L5) is ported from Max Libbrecht's sprint 10 testbed thread:
GitHub mlibbrecht/01_2026-10-19_research, projects/03_2026-10-20_epi_imputation/sprints/
10_2026-11-11_covariate_correction/scripts/testbed/phys1.py (functions msum, pileup, lam1, ptrack
with conv "gtr64") and notes/testbed.md section 5 ("2026-11-12 03:25 MST, formula rerun") and
"For the wiki". The quantile normalization (qnorm) is ported from the same repository,
sprints/04-2026-10-28_eic/scripts/qnorm/apply_bp.py. Changes from the originals: whole genome
(chr1-22, chrX) instead of four chromosomes; positions kept as sorted int32 lists, not 1 bp count
arrays; memory-bounded moving sums; the Poisson tail from candi.store.pval_from_counts.
log10_upper_tail (-log10 P(X > k), the same strict tail Max found); bigWigs read with
eic_score.BigWig instead of pyBigWig.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _name, _sub in (("candi", "src/candi"), ("candi.store", "src/candi/store"),
                    ("candi.bench", "src/candi/bench")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [str(ROOT / _sub)]
        sys.modules[_name] = _m
sys.path.insert(0, str(ROOT / "tools"))

from onion_eval import eic_score as E  # noqa: E402

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]     # natural order (QN rank order)
G_HS = 2.7e9                                              # MACS2 -g hs
WIN = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chrom_sizes(bigwig):
    with E.BigWig(bigwig) as bw:
        s = bw.chroms()
    return {c: int(s[c]) for c in CHROMS}


def ngrid(clen):
    return (clen - 1) // WIN + 1


def bin25_full(y, clen):
    """apply_bp.py::bin25: mean over each 25 bp bin of a fully covered per-base array; the last
    (partial) bin is the mean over its real bases."""
    n = ngrid(clen)
    buf = np.zeros(n * WIN, np.float64)
    buf[:clen] = y
    out = buf.reshape(-1, WIN).mean(1)
    last = (clen - 1) // WIN
    out[last] = float(np.mean(y[last * WIN: clen], dtype=np.float64))
    return out.astype(np.float32)


def save_track(path, d):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    arr = np.empty((), dtype=object)
    arr[()] = {c: np.asarray(d[c], np.float32) for c in d}
    np.save(tmp, arr, allow_pickle=True)
    os.replace(tmp, path)


def load_track_raw(path, chroms=CHROMS):
    """{chrom: float32} on the raw ceil grid (no blacklist deletion)."""
    path = str(path)
    if path.endswith(".npy"):
        d = np.load(path, allow_pickle=True)[()]
        return {c: np.asarray(d[c], np.float32) for c in chroms}
    return E.bin_bigwig(path, chroms)


def per_base_bigwig(bw, c, clen):
    st, en, va = bw.items(c)
    return np.nan_to_num(E._per_base(st, en, va, clen)).astype(np.float32)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# starts
# --------------------------------------------------------------------------------------------
def cmd_starts(a):
    import pandas as pd
    from t112 import bin25

    t0 = time.time()
    sizes = chrom_sizes(a.sizes_from)
    pos = {c: ([], []) for c in CHROMS}
    n_lines = n_main = 0
    len_sum = 0
    reader = pd.read_csv(a.ta, sep="\t", header=None, usecols=[0, 1, 2, 5],
                         names=["chrom", "start", "end", "strand"], chunksize=5_000_000,
                         dtype={"chrom": str, "start": np.int64, "end": np.int64, "strand": str})
    for chunk in reader:
        n_lines += len(chunk)
        chunk = chunk[chunk["chrom"].isin(sizes)]
        for c, g in chunk.groupby("chrom", sort=False):
            s = g["start"].to_numpy(np.int64)
            e = g["end"].to_numpy(np.int64)
            plus = g["strand"].to_numpy() == "+"
            pos[c][0].append(s[plus].astype(np.int32))
            pos[c][1].append(e[~plus].astype(np.int32))      # MACS2's - tag position: BED end
            n_main += len(g)
            len_sum += int((e - s).sum())
        log(f"{n_lines} lines")
    arrs = {}
    for c in CHROMS:
        p = np.sort(np.concatenate(pos[c][0])) if pos[c][0] else np.zeros(0, np.int32)
        m = np.sort(np.concatenate(pos[c][1])) if pos[c][1] else np.zeros(0, np.int32)
        arrs[f"{c}_p"], arrs[f"{c}_m"] = p, m
    del pos
    st = {}
    cnt = bin25.counts_from_tagalign(a.ta, sizes, stats=st)
    for c in CHROMS:
        arrs[f"{c}_cnt"] = cnt[c]
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, a.out + ".npz")
    info = dict(tagalign=a.ta, n_lines=n_lines, n_lines_main=n_main,
                mean_read_length=len_sum / max(n_main, 1), bin25_stats=st,
                seconds=time.time() - t0)
    write_json(a.out + ".json", info)
    log(json.dumps(info))
    return 0


def load_starts(prefix):
    z = np.load(prefix + ".npz")
    info = json.load(open(prefix + ".json"))
    return z, info


# --------------------------------------------------------------------------------------------
# L5: MACS2's formula at 1 bp (Max's phys1.py, conv "gtr64")
# --------------------------------------------------------------------------------------------
def msum_from_positions(pos, L, w):
    """Number of tags whose 5' position lies in [i - w//2, i + w - w//2), for i in [0, L)
    (phys1.py::msum, non-directional, centred as MACS2's pileup of the control)."""
    cnt = np.bincount(np.clip(pos, 0, L), minlength=L + 1)[: L + 1]
    cs = np.zeros(L + 2, np.int64)
    np.cumsum(cnt, out=cs[1:])
    del cnt
    A, B = w // 2, w - w // 2
    # cs index of hi_i = min(i + B, L + 1) ... positions up to L inclusive are counted
    hi = np.minimum(np.arange(B, L + B, dtype=np.int64), L + 1)
    out = cs[hi].astype(np.float64)
    del hi
    lo = np.maximum(np.arange(-A, L - A, dtype=np.int64), 0)
    out -= cs[lo]
    return out


def pileup_from_positions(pp, pm, L, d, dnase):
    """Fragment pileup per base from 5' positions (phys1.py::pileup). ChIP: + tags [s, s+d),
    - tags [e-d, e). DNase (--shift -d/2 --extsize d): [x - d//2, x - d//2 + d) for both."""
    if dnase:
        a = d // 2
        x = np.concatenate([pp, pm]).astype(np.int64)
        starts, ends = x - a, x - a + d
    else:
        pp = pp.astype(np.int64)
        pm = pm.astype(np.int64)
        starts = np.concatenate([pp, pm - d])
        ends = np.concatenate([pp + d, pm])
    df = np.bincount(np.clip(starts, 0, L), minlength=L + 1)[: L + 1].astype(np.int64)
    df -= np.bincount(np.clip(ends, 0, L), minlength=L + 1)[: L + 1]
    return np.cumsum(df[:L]).astype(np.int32)


def ptrack25(pile, lam, T, clen):
    """Mean over 25 bp bins of the pipeline's per-base score, through constant runs.

    gtr64 (phys1.py): callpeak writes pileup and lambda per million treatment reads with 5
    decimals; bdgcmp -S T/1e6 multiplies back and truncates the pileup to an integer; the score
    is -log10 P(Poisson(lambda) > k)."""
    from candi.store.pval_from_counts import log10_upper_tail

    n = len(pile)
    chg = np.flatnonzero((pile[1:] != pile[:-1]) | (lam[1:] != lam[:-1])) + 1
    st = np.concatenate([[0], chg])
    sv = T / 1e6
    k = np.floor(np.round(pile[st].astype(np.float64) / sv, 5) * sv).astype(np.int64)
    lv = np.round(lam[st] / sv, 5) * sv
    v = log10_upper_tail(k, lv)
    v = np.maximum(v, 0.0)
    per = np.repeat(v, np.diff(np.concatenate([st, [n]])))
    return bin25_full(per, clen)


def cmd_formula(a):
    t0 = time.time()
    sizes = chrom_sizes(a.sizes_from)
    zt, it = load_starts(a.trt)
    T = it["n_lines"]
    dnase = a.ctl == "none"
    if not dnase:
        zc, ic = load_starts(a.ctl)
        C = ic["n_lines"]
        if T > C:
            raise SystemExit(f"treatment {T} > control {C}: MACS2 would scale the treatment; "
                             "not implemented (untested in Max's testbed too)")
        s = T / C
    d = int(a.d)
    out = {}
    for c in CHROMS:
        L = sizes[c]
        pp, pm = zt[f"{c}_p"], zt[f"{c}_m"]
        pile = pileup_from_positions(pp, pm, L, d, dnase)
        lam = np.full(L, d * T / G_HS)
        if dnase:
            lam = np.maximum(lam, d / 10000.0 * msum_from_positions(np.concatenate([pp, pm]), L, 10000))
        else:
            cc = np.concatenate([zc[f"{c}_p"], zc[f"{c}_m"]])
            for w, f in ((d, 1.0), (1000, d / 1000.0), (10000, d / 10000.0)):
                lam = np.maximum(lam, s * f * msum_from_positions(cc, L, w))
            del cc
        out[c] = ptrack25(pile, lam, T, L)
        del pile, lam
        log(f"{c} done ({time.time() - t0:.0f}s)")
    save_track(a.out, out)
    write_json(a.out + ".json", dict(trt=a.trt, ctl=a.ctl, d=d, T=T, C=None if dnase else C,
                                     seconds=time.time() - t0))
    return 0


# --------------------------------------------------------------------------------------------
# L6: the count route (signal_from_counts.py, plug_in, box_d)
# --------------------------------------------------------------------------------------------
def cmd_counts_route(a):
    from onion_eval import signal_from_counts as SFC

    t0 = time.time()
    sizes = chrom_sizes(a.sizes_from)
    zt, it = load_starts(a.trt)
    cov = dict(depth_reads=it["n_lines"], read_length=it["mean_read_length"])
    dnase = a.ctl == "none"
    if dnase:
        assay_class = "DNase-seq"
    else:
        zc, ic = load_starts(a.ctl)
        assay_class = "narrow_histone"          # the ChIP route is one route for both classes
        cov.update(frag_len=float(a.d), control_depth_reads=ic["n_lines"],
                   control_read_length=ic["mean_read_length"])
    out = {}
    for c in CHROMS:
        mu = zt[f"{c}_cnt"].astype(np.float64) + a.eps   # plug_in asks mu > 0; floor() drops eps
        ctl = None if dnase else zc[f"{c}_cnt"].astype(np.float64)
        sig, src = SFC.signal_track(mu, None, assay_class, ctl, cov, mode="plug_in",
                                    pileup="box_d", dnase_route="macs2_no_control")
        n = ngrid(sizes[c])
        y = np.zeros(n, np.float32)
        y[: sig.shape[0]] = sig[:n]
        out[c] = y
        log(f"{c} done ({time.time() - t0:.0f}s), lambda source {src}")
    save_track(a.out, out)
    write_json(a.out + ".json", dict(trt=a.trt, ctl=a.ctl, covariates=cov, assay_class=assay_class,
                                     mode="plug_in", pileup="box_d", eps=a.eps,
                                     seconds=time.time() - t0))
    return 0


# --------------------------------------------------------------------------------------------
# quantile normalization (Max's apply_bp.py)
# --------------------------------------------------------------------------------------------
def load_mask_intervals(path):
    iv = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 3 or f[0] not in CHROMS:
                continue
            iv.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    return iv


def mask_for(iv, c, L):
    m = np.zeros(L, bool)
    for s, e in iv.get(c, []):
        m[s:e] = True
    return m


class _Source:
    """Per-base values of a bigWig, or of a 25 bp .npy track (each base = its bin's value)."""

    def __init__(self, path, sizes):
        self.path, self.sizes = str(path), sizes
        self.bw = None
        if self.path.endswith(".npy"):
            self.d = np.load(self.path, allow_pickle=True)[()]
        else:
            self.bw = E.BigWig(self.path)

    def get(self, c):
        L = self.sizes[c]
        if self.bw is not None:
            return per_base_bigwig(self.bw, c, L)
        return np.repeat(np.asarray(self.d[c], np.float32), WIN)[:L]


def cmd_qnorm(a):
    t0 = time.time()
    sizes = chrom_sizes(a.sizes_from)
    R = np.load(a.ref)
    Rf = {1: R["R_p"].astype(np.float64), 0: R["R_b"].astype(np.float64)}
    Mf = {f: len(Rf[f]) for f in Rf}
    src = _Source(a.x, sizes)
    iv = load_mask_intervals(a.mask)
    # pass A: value counts per family
    parts = {0: ([], []), 1: ([], [])}
    for c in CHROMS:
        x = src.get(c)
        m = mask_for(iv, c, len(x))
        for f, sel in ((1, m), (0, ~m)):
            u, n = np.unique(x[sel], return_counts=True)
            parts[f][0].append(u)
            parts[f][1].append(n)
    uniq, start, N = {}, {}, {}
    for f in (0, 1):
        u, inv = np.unique(np.concatenate(parts[f][0]), return_inverse=True)
        cnt = np.bincount(inv, weights=np.concatenate(parts[f][1]).astype(np.float64)).astype(np.int64)
        uniq[f] = u.astype(np.float32)
        start[f] = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        N[f] = int(cnt.sum())
    del parts
    log(f"pass A done ({time.time() - t0:.0f}s): N_peak {N[1]}, N_bg {N[0]}")
    seen = {f: np.zeros(len(uniq[f]), np.int64) for f in (0, 1)}
    ver = E.BigWig(a.verify) if a.verify else None
    mism = tot = 0
    maxabs = 0.0
    out = {}
    for c in CHROMS:
        x = src.get(c)
        m = mask_for(iv, c, len(x))
        y = np.empty(len(x), np.float64)
        for f, sel in ((1, m), (0, ~m)):
            idx = np.flatnonzero(sel)
            ids = np.searchsorted(uniq[f], x[idx])
            o = np.argsort(ids, kind="stable")
            ids_s = ids[o]
            first = np.r_[0, np.flatnonzero(np.diff(ids_s)) + 1]
            grp_start = np.repeat(first, np.diff(np.r_[first, len(ids_s)]))
            occ = np.arange(len(ids_s)) - grp_start
            r = np.empty(len(idx), np.int64)
            r[o] = start[f][ids_s] + seen[f][ids_s] + occ
            seen[f] += np.bincount(ids, minlength=len(uniq[f]))
            k = ((r + 1) * (Mf[f] - 1)) // N[f]
            y[idx] = Rf[f][k]
            del idx, ids, o, ids_s, first, grp_start, occ, r, k
        if ver is not None:
            yy = per_base_bigwig(ver, c, len(x)).astype(np.float64)
            dd = np.abs(yy - y.astype(np.float32).astype(np.float64))
            mism += int((dd > 1e-6 * np.maximum(np.abs(yy), 1e-3)).sum())
            tot += len(y)
            maxabs = max(maxabs, float(dd.max()))
        out[c] = bin25_full(y, len(x))
        log(f"{c} done ({time.time() - t0:.0f}s); per-base mismatches so far {mism}")
    save_track(a.out, out)
    rep = dict(x=a.x, mask=a.mask, ref=a.ref, N_peak=N[1], N_bg=N[0], seconds=time.time() - t0)
    if ver is not None:
        ref25 = E.bin_bigwig(a.verify, CHROMS)
        d25 = max(float(np.max(np.abs(ref25[c].astype(np.float64) - out[c]))) for c in CHROMS)
        rep.update(verify=a.verify, bases=tot, mismatched_bases=mism, frac_mismatched=mism / max(tot, 1),
                   max_abs_diff_per_base=maxabs, max_abs_diff_25bp_vs_bin_bigwig=d25)
    write_json(a.out + ".json", rep)
    log(json.dumps(rep))
    return 0


# --------------------------------------------------------------------------------------------
# peaks
# --------------------------------------------------------------------------------------------
def cmd_callpeaks(a):
    """Peaks on a 25 bp -log10 p track, an approximation of MACS2 callpeak -p 0.01 as the
    pipeline runs it: bins with value >= thresh, runs merged across gaps of <= max_gap bp (MACS2
    --max-gap defaults to the read length), runs shorter than min_len bp dropped (MACS2
    --min-length defaults to d), the top `cap` runs by their maximum kept (the pipeline's 500K
    ChIP / 300K ATAC cap). Writes BED: chrom, start, end, name, score=max*10, ., max."""
    d = load_track_raw(a.x)
    gap = int(np.ceil(a.max_gap / WIN))
    minb = int(np.ceil(a.min_len / WIN))
    peaks = []
    for c in CHROMS:
        y = d[c]
        on = y >= a.thresh
        if not on.any():
            continue
        e = np.diff(np.r_[0, on.astype(np.int8), 0])
        s0, e0 = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        # merge runs whose gap (in bins) is <= gap
        keep = np.r_[True, (s0[1:] - e0[:-1]) > gap]
        gs = s0[keep]
        ge = np.r_[e0[np.flatnonzero(keep)[1:] - 1], e0[-1]]
        ok = (ge - gs) >= minb
        gs, ge = gs[ok], ge[ok]
        cm = [float(y[s:e].max()) for s, e in zip(gs, ge)]
        peaks += [(c, int(s) * WIN, int(e) * WIN, float(v)) for s, e, v in zip(gs, ge, cm)]
    peaks.sort(key=lambda p: -p[3])
    peaks = peaks[: a.cap]
    peaks.sort(key=lambda p: (CHROMS.index(p[0]), p[1]))
    with open(a.out, "w") as fh:
        for i, (c, s, e, v) in enumerate(peaks):
            fh.write(f"{c}\t{s}\t{e}\tpeak{i}\t{int(min(v * 10, 1000))}\t.\t{v:.5f}\n")
    rep = dict(x=a.x, thresh=a.thresh, max_gap=a.max_gap, min_len=a.min_len, cap=a.cap,
               n_peaks=len(peaks), bases=int(sum(p[2] - p[1] for p in peaks)))
    write_json(a.out + ".json", rep)
    log(json.dumps(rep))
    return 0


def cmd_peakcheck(a):
    sizes = chrom_sizes(a.sizes_from)
    ia, ib = load_mask_intervals(a.a), load_mask_intervals(a.b)
    inter = union = na = nb = 0
    for c in CHROMS:
        ma, mb = mask_for(ia, c, sizes[c]), mask_for(ib, c, sizes[c])
        inter += int((ma & mb).sum())
        union += int((ma | mb).sum())
        na += int(ma.sum())
        nb += int(mb.sum())
    rep = dict(a=a.a, b=a.b, bases_a=na, bases_b=nb, bases_both=inter,
               jaccard=inter / max(union, 1), frac_a_in_b=inter / max(na, 1), frac_b_in_a=inter / max(nb, 1))
    write_json(a.out, rep)
    log(json.dumps(rep))
    return 0


# --------------------------------------------------------------------------------------------
# probe 2: read halves, the pipeline's naive-overlap rule, the 25 bp fragment-pileup route
# --------------------------------------------------------------------------------------------
def cmd_split(a):
    """Two random halves of a tagAlign's lines (seeded numpy permutation). Like the pipeline's
    `encode_task_spr.py::spr_se` (chip-seq-pipeline2 v2.2.2, src/encode_task_spr.py:40-81), the
    first half gets (N + 1) // 2 lines; the pipeline's own shuffle (GNU shuf seeded by the file's
    byte size) is not reproduced. Writes <out>.pr1.tagAlign, <out>.pr2.tagAlign (uncompressed)."""
    t0 = time.time()
    op = gzip.open if a.ta.endswith(".gz") else open
    with op(a.ta, "rb") as fh:
        lines = fh.read().splitlines()
    n = len(lines)
    rng = np.random.default_rng(a.seed)
    first = np.zeros(n, bool)
    first[rng.permutation(n)[: (n + 1) // 2]] = True
    for tag, sel in (("pr1", first), ("pr2", ~first)):
        tmp = f"{a.out}.{tag}.tagAlign.tmp"
        with open(tmp, "wb") as fh:
            idx = np.flatnonzero(sel)
            for i0 in range(0, len(idx), 2_000_000):
                fh.write(b"\n".join(lines[i] for i in idx[i0: i0 + 2_000_000]) + b"\n")
        os.replace(tmp, f"{a.out}.{tag}.tagAlign")
    rep = dict(ta=a.ta, seed=a.seed, n_lines=n, n_pr1=int(first.sum()), n_pr2=int(n - first.sum()),
               seconds=time.time() - t0)
    write_json(a.out + ".split.json", rep)
    log(json.dumps(rep))
    return 0


def _read_peaks(path):
    """Rows of a BED / narrowPeak file (plain or .gz) as lists of fields (main chromosomes only)."""
    op = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[0] in CHROMS:
                rows.append(f)
    return rows


def _overlap_keep(A, B):
    """Port of the naive-overlap filter of the ENCODE pipelines (chip-seq-pipeline2 v2.2.2 and
    atac-seq-pipeline v2.2.3, identical files, src/encode_task_overlap.py:89-91 and 109-117):

        intersectBed -wo -a A -b B | awk '{s1=$3-$2; s2=$13-$12;
            if (($21/s1 >= 0.5) || ($21/s2 >= 0.5)) print}' | cut -f 1-10 | sort | uniq

    i.e. keep each row of A that overlaps some row of B by at least half of the A row's length or
    half of the B row's length (overlap in bp, half-open, > 0). Returns the kept rows of A (each
    once; `uniq` on identical full rows)."""
    out, seen = [], set()
    byc = {}
    for f in B:
        byc.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    idx = {}
    for c, iv in byc.items():
        iv.sort()
        s = np.array([x[0] for x in iv], np.int64)
        e = np.array([x[1] for x in iv], np.int64)
        idx[c] = (s, e, int((e - s).max()))
    for f in A:
        if f[0] not in idx:
            continue
        s, e, lmax = idx[f[0]]
        a0, a1 = int(f[1]), int(f[2])
        lo = np.searchsorted(s, a0 - lmax, "left")
        hi = np.searchsorted(s, a1, "left")
        bs, be = s[lo:hi], e[lo:hi]
        ov = np.minimum(be, a1) - np.maximum(bs, a0)
        ok = ov > 0
        if not ok.any():
            continue
        ov, lb = ov[ok], (be - bs)[ok]
        if np.any((2 * ov >= (a1 - a0)) | (2 * ov >= lb)):
            key = "\t".join(f)
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


def _blacklist_drop(rows, blacklist):
    """Port of encode_lib_blacklist_filter.py::blacklist_filter (same source tree, lines 62-66):
    `bedtools intersect -v -a peaks -b blacklist` (drop any peak overlapping a blacklist interval by
    >= 1 bp) and the score column capped at 1000. The chromosome regex `chr[\\dXY]+` keeps every
    main chromosome, so it is a no-op here."""
    bl = {}
    for f in _read_peaks(blacklist):
        bl.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    ix = {}
    for c, iv in bl.items():
        iv.sort()
        ix[c] = (np.array([x[0] for x in iv]), np.array([x[1] for x in iv]), max(x[1] - x[0] for x in iv))
    out = []
    for f in rows:
        if f[0] in ix:
            s, e, lmax = ix[f[0]]
            a0, a1 = int(f[1]), int(f[2])
            lo, hi = np.searchsorted(s, a0 - lmax), np.searchsorted(s, a1)
            if np.any(e[lo:hi] > a0):
                continue
        g = list(f)
        if len(g) > 4 and g[4].lstrip("-").isdigit() and int(g[4]) > 1000:
            g[4] = "1000"
        out.append(g)
    return out


def cmd_overlap(a):
    """The pipeline's pseudoreplicate overlap set: pooled peaks reproducible in both halves, then
    blacklist-filtered (encode_task_overlap.py::naive_overlap, main lines 140-147)."""
    p1, p2, pooled = _read_peaks(a.peak1), _read_peaks(a.peak2), _read_peaks(a.pooled)
    keep = _overlap_keep(_overlap_keep(pooled, p1), p2)
    keep = _blacklist_drop(keep, a.blacklist) if a.blacklist else keep
    keep.sort(key=lambda f: (CHROMS.index(f[0]), int(f[1]), int(f[2])))
    with open(a.out, "w") as fh:
        for f in keep:
            fh.write("\t".join(f) + "\n")
    rep = dict(peak1=a.peak1, peak2=a.peak2, pooled=a.pooled, blacklist=a.blacklist,
               n_pr1=len(p1), n_pr2=len(p2), n_pooled=len(pooled), n_overlap=len(keep),
               bases=int(sum(int(f[2]) - int(f[1]) for f in keep)))
    write_json(a.out + ".json", rep)
    log(json.dumps(rep))
    return 0


def mlog10_sf(k, lam):
    """-log10 P(N >= k), N ~ Poisson(lam), real k >= 0. Verbatim port of Max Libbrecht's sprint 10
    cac_lib.py::mlog10_sf (sprints/10_2026-11-11_covariate_correction/scripts/testbed/cac_lib.py:183-199)."""
    from scipy.special import gammainc, gammaln

    k = np.asarray(k, np.float64)
    lam = np.asarray(lam, np.float64)
    out = np.zeros(np.broadcast(k, lam).shape)
    pos = k > 0
    kk, ll = np.broadcast_to(k, out.shape)[pos], np.broadcast_to(lam, out.shape)[pos]
    g = gammainc(kk, ll)
    v = np.empty_like(g)
    ok = g > 1e-280
    v[ok] = -np.log10(g[ok])
    bad = ~ok
    if bad.any():
        aa, x = kk[bad], ll[bad]
        lp = aa * np.log(x) - x - gammaln(aa + 1) - np.log1p(-np.minimum(x / (aa + 1), 0.999))
        v[bad] = -lp / np.log(10.0)
    out[pos] = v
    return np.maximum(out, 0.0)


def box(v, nb):
    """Centred moving sum over nb bins, zeros outside (cac_lib.py::box, lines 150-156)."""
    nb = max(1, int(nb))
    cs = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    i = np.arange(len(v))
    lo = np.clip(i - nb // 2, 0, len(v))
    hi = np.clip(i - nb // 2 + nb, 0, len(v))
    return cs[hi] - cs[lo]


def lam_hat25(d, T, c25, s, t25):
    """MACS2's local lambda from 25 bp tracks (cac_lib.py::lam_hat25, lines 170-180): c25 / t25 =
    control / treatment 25 bp overlap counts divided by kappa (read 5' ends per bin)."""
    lam = np.full(len(t25), d * T / G_HS)
    if c25 is not None:
        for w, f in ((d, 1.0), (1000.0, d / 1000.0), (10000.0, d / 10000.0)):
            lam = np.maximum(lam, s * f * box(c25, max(1, round(w / 25.0))))
    else:
        lam = np.maximum(lam, d / 10000.0 * box(t25, 400))
    return lam


KAPPA_CHROMS = ["chr20", "chr13", "chr21", "chr22"]


def kappa_of(z):
    """Mean number of 25 bp bins a read overlaps under the store rule (cac_lib.py Data.kappa, lines
    110-118): sum of the 25 bp counts / reads, over chr13, chr20, chr21, chr22."""
    n = sum(len(z[f"{c}_p"]) + len(z[f"{c}_m"]) for c in KAPPA_CHROMS)
    return sum(float(z[f"{c}_cnt"].sum()) for c in KAPPA_CHROMS) / n


def _on_grid(v, n):
    y = np.zeros(n, np.float64)
    m = min(n, len(v))
    y[:m] = v[:m]
    return y


def cmd_fragpile_route(a):
    """L6f: MACS2's formula on a 25 bp fragment pileup, as Max's sprint 10 testbed computed its "pf"
    feature (cac_lib.py::pc_features, lines 277-292, and cac_train.py:408-411 "formula on the 25 bp
    fragment pileup"): pile25 = each read extended to d along its strand (DNase-seq: centred on the
    5' end), the mean coverage over each 25 bp bin; lambda = lam_hat25 from the control's (no
    control: the treatment's) 25 bp overlap counts / kappa; score = mlog10_sf(pile25 + 1, lambda) =
    -log10 P(N > pile25) ("gt", MACS2's convention). Raw -log10 p is written (Max scored asinh of it).
    Deviation: the pileup is built from 1 bp read 5' ends (Max: 5 bp), i.e. the exact 25 bp mean."""
    t0 = time.time()
    sizes = chrom_sizes(a.sizes_from)
    zt, it = load_starts(a.trt)
    T = float(it["n_lines"])
    d = float(a.d)
    dnase = a.ctl == "none"
    kt = kappa_of(zt)
    if not dnase:
        zc, ic = load_starts(a.ctl)
        kc = kappa_of(zc)
        s = T / float(ic["n_lines"])
    out = {}
    for c in CHROMS:
        L = sizes[c]
        n = ngrid(L)
        pile = pileup_from_positions(zt[f"{c}_p"], zt[f"{c}_m"], L, int(d), dnase)
        p25 = bin25_full(pile, L).astype(np.float64)
        del pile
        t25 = _on_grid(zt[f"{c}_cnt"].astype(np.float64), n) / kt
        c25 = None if dnase else _on_grid(zc[f"{c}_cnt"].astype(np.float64), n) / kc
        lam = lam_hat25(d, T, c25, 0.0 if dnase else s, t25)
        out[c] = mlog10_sf(p25 + 1.0, lam).astype(np.float32)
        log(f"{c} done ({time.time() - t0:.0f}s)")
    save_track(a.out, out)
    write_json(a.out + ".json", dict(trt=a.trt, ctl=a.ctl, d=d, T=T, kappa_trt=kt,
                                     kappa_ctl=None if dnase else kc, s=None if dnase else s,
                                     seconds=time.time() - t0))
    return 0


def cmd_arrdiff(a):
    """Max |A - B| and the count of differing bins of two challenge .npy tracks, raw grid, all bins."""
    A, B = load_track_raw(a.a), load_track_raw(a.b)
    mx, nd, tot = 0.0, 0, 0
    for c in CHROMS:
        x, y = A[c].astype(np.float64), B[c].astype(np.float64)
        if len(x) != len(y):
            raise SystemExit(f"{c}: length {len(x)} != {len(y)}")
        dd = np.abs(x - y)
        mx = max(mx, float(dd.max()))
        nd += int((dd > 0).sum())
        tot += len(x)
    rep = dict(a=a.a, b=a.b, bins=tot, bins_differing=nd, max_abs_diff=mx)
    write_json(a.out, rep)
    log(json.dumps(rep))
    return 0


# --------------------------------------------------------------------------------------------
# compare, entrants, placement
# --------------------------------------------------------------------------------------------
def cmd_compare(a):
    """Normalized MSE of track A against reference B: mean((A - B)^2) / var(B), on the scorer's
    grid (25 bp, blacklist bins deleted, chr1-22 + chrX), plus max |A - B| and Pearson r."""
    bl = E.load_blacklist_bins(E.asset_dir())
    A = E.load_prediction(a.a, E.CHROMS, bl)
    B = E.load_prediction(a.b, E.CHROMS, bl)
    P, T = E.align(A, B, E.CHROMS)
    x = np.concatenate([P[c] for c in E.CHROMS]).astype(np.float64)
    y = np.concatenate([T[c] for c in E.CHROMS]).astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    rep = dict(pair=a.name, a=a.a, b=a.b, n_bins=int(x.size), mse=mse, var_b=float(y.var()),
               nmse=mse / float(y.var()), max_abs_diff=float(np.max(np.abs(x - y))),
               pearson=float(np.corrcoef(x, y)[0, 1]),
               nmse_asinh=float(np.mean((np.arcsinh(x) - np.arcsinh(y)) ** 2) / np.arcsinh(y).var()))
    write_json(a.out, rep)
    log(json.dumps(rep))
    return 0


def cmd_entrants(a):
    exps = a.exps.split(",")
    E.prepare_truth(a.truth, a.cache, a.workers)
    by_exp = {e: [(f"team_{tid}", name, tid, f"{a.submissions}/{tid}/{e}.bigwig")
                  for tid, name, _ in E.TEAMS] for e in exps}
    E._run_scoring(a.truth, a.cache, by_exp, a.workers, (0, 1))
    return 0


def cmd_placement(a):
    rows = []
    for spec in a.rung:                                   # name=path-to-score-tsv
        name, path = spec.split("=", 1)
        for r in E.read_score_rows(path):
            if r["chrom_set"] == "all23":
                rows.append((name, r))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["truth", "experiment", "rung", "measure", "value", "rank_among_entrants",
                    "n_entrants", "best_entrant", "median_entrant", "worst_entrant"])
        for name, r in rows:
            exp = r["experiment"]
            ent = []
            for tid, _, _ in E.TEAMS:
                p = Path(a.cache) / "scores" / f"team_{tid}" / f"{exp}.tsv"
                er = [x for x in E.read_score_rows(p) if x["chrom_set"] == "all23"]
                ent.append(er[0])
            for m in E.MEASURES9:
                v = r[m]
                ev = np.array([x[m] for x in ent])
                hib = m in E.HIGHER_IS_BETTER
                better = int(np.sum(ev > v)) if hib else int(np.sum(ev < v))
                best = float(ev.max() if hib else ev.min())
                worst = float(ev.min() if hib else ev.max())
                w.writerow([a.truth_name, exp, name, m, repr(v), better + 1, len(ev), repr(best),
                            repr(float(np.median(ev))), repr(worst)])
    log(f"wrote {a.out}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "eic_score":           # eic_score.py's CLI, with the candi stubs in place
        return E.main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("starts"); p.add_argument("--ta", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sizes-from", required=True, help="any GRCh38 bigWig (chromosome sizes)")
    p.set_defaults(fn=cmd_starts)
    for nm, fn in (("formula", cmd_formula), ("counts_route", cmd_counts_route)):
        p = sub.add_parser(nm)
        p.add_argument("--trt", required=True, help="starts prefix")
        p.add_argument("--ctl", required=True, help="starts prefix, or none (DNase-seq)")
        p.add_argument("--d", type=float, required=True, help="fragment length / smooth window, bp")
        p.add_argument("--sizes-from", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--eps", type=float, default=1e-6)
        p.set_defaults(fn=fn)
    p = sub.add_parser("fragpile_route")
    p.add_argument("--trt", required=True, help="starts prefix")
    p.add_argument("--ctl", required=True, help="starts prefix, or none (DNase-seq)")
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--sizes-from", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fragpile_route)
    p = sub.add_parser("split")
    p.add_argument("--ta", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261122)
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("overlap")
    p.add_argument("--peak1", required=True); p.add_argument("--peak2", required=True)
    p.add_argument("--pooled", required=True); p.add_argument("--blacklist", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_overlap)
    p = sub.add_parser("arrdiff")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_arrdiff)
    p = sub.add_parser("qnorm")
    p.add_argument("--x", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--verify", default=""); p.add_argument("--sizes-from", required=True)
    p.set_defaults(fn=cmd_qnorm)
    p = sub.add_parser("callpeaks")
    p.add_argument("--x", required=True); p.add_argument("--out", required=True)
    p.add_argument("--thresh", type=float, default=2.0); p.add_argument("--max-gap", type=float, required=True)
    p.add_argument("--min-len", type=float, required=True); p.add_argument("--cap", type=int, required=True)
    p.set_defaults(fn=cmd_callpeaks)
    p = sub.add_parser("peakcheck")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--out", required=True); p.add_argument("--sizes-from", required=True)
    p.set_defaults(fn=cmd_peakcheck)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--name", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("entrants")
    p.add_argument("--truth", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--exps", required=True); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--submissions", default=E.DEFAULT_SUBMISSIONS)
    p.set_defaults(fn=cmd_entrants)
    p = sub.add_parser("placement")
    p.add_argument("--cache", required=True); p.add_argument("--truth-name", required=True)
    p.add_argument("--rung", nargs="+", required=True, help="name=score.tsv")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_placement)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
