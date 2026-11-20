#!/usr/bin/env python3
"""The exact ENCODE Imputation Challenge (EIC) scorer, against the organizers' corrected truth.

Chunk X1 of t152 (crux task t166). Decisions D9 and D16: the EIC headline is scored with the
challenge's own nine measures against the corrected truth (Zenodo record 14736610, top level of
the tarball, never `old/`), and external methods are compared through their published submissions.

PORTED, NOT REWRITTEN. Every scoring line comes from Max Libbrecht's sprint 04, which reproduced
the published corrected board (Additional File 4, "MOESM4", of Schreiber et al., Genome Biology
2023, doi:10.1186/s13059-023-02915-y) value for value. Source: GitHub
`mlibbrecht/01_2026-10-19_research` at commit 3e7478a101807a1d5414852c3fc703717e7098b2, folder
`projects/03_2026-10-20_epi_imputation/sprints/04-2026-10-28_eic/scripts/`:

  inputs/vendor_001/eic_metrics.py   the nine measures, bootstrap groups, annotations, blacklist
                                     -> CHROMS, BOOTSTRAP_CHROM, load_annotations,
                                        load_blacklist_bins, apply_blacklist, score_reference
  vendor031/metrics9.py              per-chromosome partial sums, _flat_idx -> ExpScorer
  score/fastglobal.py                one global sort per track -> ExpScorer.global_measures
  vendor031/tracks022.py             the challenge's 25 bp binning -> bin_bigwig
  vendor031/tracks_io.py             blacklist deletion, then truncation -> align
  vendor031/varutil.py, emit_var.py  msevar weights -> two_pass_var, msevar members
  vendor031/rank_eic.py              the organizers' ranking statistic -> combined_ranks,
                                     global_ranks
  score/score_multi.py, compare.py   the driver and the published-table comparison

The one new piece is `BigWig`, a pure-numpy bigWig reader. The ONION kit venv has no pyBigWig and
the brief allows no new dependency; Max's loader read the files through pyBigWig. The reader is
checked against Max's pyBigWig-binned arrays by the `check-bins` subcommand.

Normalization (the open question the brief asked to settle). The corrected truth is quantile-
normalized; the submissions are scored as submitted. The challenge's `score.py` calls
`score_metrics.normalize_dict` on prediction and truth, and that function returns its input
unchanged (its body is commented out; ENCODE-DCC/imputation_challenge at 181b8023, lines
208-218). So a prediction is only (1) binned to 25 bp by the mean over each bin, missing = 0, and
(2) its blacklisted bins deleted. Nothing else is applied. `verify-published` reproduces MOESM4
on exactly that rule.

Subcommands:
  fetch-truth       stream Zenodo 14736610 (-> qnorm/) and 14736608 (-> se_unnormalized/), md5 checked
  link-submissions  symlink the 25 published entries' bigWig folders under one directory
  check-bins        bitwise check of the bigWig reader against reference .npy arrays
  score             nine measures for a directory of predictions
  verify-published  score the 25 published entries and compare with MOESM4, value and rank
  board             add scored entries, each alone, to the published field and rank them
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import struct
import sys
import tarfile
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------
CHROMS = sorted(["chr" + str(i) for i in range(1, 23)] + ["chrX"])

# The ten "bootstrap" chromosome groups, verbatim from the --bootstrap-chrom argument of the
# round-2 scoring command in the challenge repository's README (eic_metrics.py, ported).
BOOTSTRAP_CHROM = [g.split(",") for g in [
    "chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr16,chr17,chr18,chr19,chr2,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chrX",
    "chr1,chr10,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr19,chr2,chr20,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
]]
CHROM_SETS = [("all23", CHROMS)] + [(f"boot{i}", sorted(g)) for i, g in enumerate(BOOTSTRAP_CHROM)]

MEASURES9 = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh",
             "msevar", "mse1obs", "mse1imp"]
HIGHER_IS_BETTER = {"gwcorr", "gwspear"}
WINDOW = 25
PROM_LOC = 80
# msevar weights are the per-assay cross-cell-type variance of the blind truth with this DNase-seq
# track left out (Max's experiment 022; it reproduces MOESM4's msevar to 1e-9).
MSEVAR_LEFT_OUT = {"C38M02"}

BLIND_EXPS = ("C05M17 C05M18 C05M20 C05M29 C06M16 C06M17 C06M18 C07M20 C07M29 C12M01 C12M02 "
              "C19M16 C19M17 C19M18 C19M20 C19M22 C19M29 C22M16 C22M17 C28M17 C28M18 C28M22 "
              "C28M29 C31M01 C31M02 C31M16 C31M29 C38M01 C38M02 C38M17 C38M18 C38M20 C38M22 "
              "C38M29 C39M16 C39M17 C39M18 C39M20 C39M22 C39M29 C40M16 C40M17 C40M18 C40M20 "
              "C40M22 C40M29 C51M16 C51M17 C51M18 C51M20 C51M29").split()

# The 25 published entries: team_id (as in MOESM4), name (organizers' team_name.tsv), and the
# Nibi folder of the submitted bigWigs (Max's sprint 04 team_map.tsv). Synapse downloads live in
# Max's tree; Average and Avocado are the organizers' baselines, copied by Mehdi in August 2026.
_SYN = "/project/def-maxwl/maxwl/epi_imputation/031/submissions"
_REPRO = "/project/def-maxwl/mforooz/EIC_REPRO/inputs"
TEAMS = [
    (3393457, "imp", f"{_SYN}/3393457"),
    (3393574, "Lavawizard", f"{_SYN}/3393574"),
    (3393756, "imp1", f"{_SYN}/3393756"),
    (3393417, "Hongyang Li and Yuanfang Guan v1", f"{_SYN}/3393417"),
    (3393847, "Guacamole", f"{_SYN}/3393847"),
    (3393817, "BrokenNodes_v3", f"{_SYN}/3393817"),
    (100, "Average", f"{_REPRO}/average_test"),
    (3393458, "CUWA", f"{_SYN}/3393458"),
    (3330254, "Hongyang Li and Yuanfang Guan", f"{_SYN}/3330254"),
    (3388185, "LiPingChun", f"{_SYN}/3388185"),
    (3393860, "CUImpute1", f"{_SYN}/3393860"),
    (3393861, "ICU", f"{_SYN}/3393861"),
    (3379072, "BrokenNodes", f"{_SYN}/3379072"),
    (3393580, "Song Lab 3", f"{_SYN}/3393580"),
    (3389318, "Song Lab", f"{_SYN}/3389318"),
    (3393606, "BrokenNodes_v2", f"{_SYN}/3393606"),
    (3379312, "CostaLab", f"{_SYN}/3379312"),
    (3393418, "Hongyang Li and Yuanfang Guan v2", f"{_SYN}/3393418"),
    (3393851, "NittanyLions2", f"{_SYN}/3393851"),
    (3391272, "UIOWA Michaelson Lab", f"{_SYN}/3391272"),
    (3386902, "KKT-ENCODE-Impute", f"{_SYN}/3386902"),
    (3344979, "NittanyLions", f"{_SYN}/3344979"),
    (3393128, "Aug2019Imputation", f"{_SYN}/3393128"),
    (3393579, "Song Lab 2", f"{_SYN}/3393579"),
    (0, "Avocado", f"{_REPRO}/avocado_test"),
]
TEAM_NAME = {t: n for t, n, _ in TEAMS}

ZENODO = {"qnorm": 14736610, "se_unnormalized": 14736608}
MOESM4_NAME = "13059_2023_2915_MOESM4_ESM.csv"
MOESM4_URL = ("https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-023-02915-y/"
              "MediaObjects/13059_2023_2915_MOESM4_ESM.csv")
MOESM4_SHA256 = "96e4164af6ecb11de72228af82bd6b2f7fa0f94b95e6d4f2d2948e02ce5a944d"

ONION_EIC = "/project/def-maxwl/mforooz/ONION/eic"
DEFAULT_PUBLISHED = f"{ONION_EIC}/published/{MOESM4_NAME}"
DEFAULT_SUBMISSIONS = f"{ONION_EIC}/submissions"
SCRATCH_EIC = "/scratch/mforooz/onion/eic"


def log(msg):
    print(msg, flush=True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# annotations (eic_metrics.py, ported). Assets: the organizers' own files, pinned in candi.bench.
# --------------------------------------------------------------------------------------------
def asset_dir():
    from candi.bench import annotations as A
    A.verify_assets(["gencode.v29.genes.gtf.bed.gz", "F5.hg38.enhancers.bed.gz",
                     "hg38.blacklist.bed.gz"])
    return Path(A.ASSET_DIR)


def _open_bed(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield line.split()


def load_annotations(adir):
    """Per-chromosome bin intervals [s, e) for promoters, gene bodies and enhancers.

    Bin arithmetic of score_metrics.py: start//25, end//25 + 1. A + strand promoter is
    y[s-80:s]; a negative start gives an empty numpy slice, so such genes contribute nothing.
    """
    prom, body, enh = {}, {}, {}
    for f in _open_bed(Path(adir) / "gencode.v29.genes.gtf.bed.gz"):
        chrom, start, end, _, _, strand = f[:6]
        s, e = int(start) // WINDOW, int(end) // WINDOW + 1
        body.setdefault(chrom, []).append((s, e))
        if strand == "+":
            if s - PROM_LOC >= 0:
                prom.setdefault(chrom, []).append((s - PROM_LOC, s))
        else:
            prom.setdefault(chrom, []).append((e, e + PROM_LOC))
    for f in _open_bed(Path(adir) / "F5.hg38.enhancers.bed.gz"):
        enh.setdefault(f[0], []).append((int(f[1]) // WINDOW, int(f[2]) // WINDOW + 1))
    to_arr = lambda d: {c: np.array(v, dtype=np.int64) for c, v in d.items()}
    return to_arr(prom), to_arr(body), to_arr(enh)


def load_blacklist_bins(adir):
    """chrom -> sorted bin ids overlapping the ENCODE blacklist."""
    bins = {}
    for f in _open_bed(Path(adir) / "hg38.blacklist.bed.gz"):
        s, e = int(f[1]) // WINDOW, int(f[2]) // WINDOW + 1
        bins.setdefault(f[0], set()).update(range(s, e))
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in bins.items()}


def apply_blacklist(y, ids):
    """Delete (not mask) blacklisted bins, as bw_to_npy.py does; later indices shift."""
    if ids is None or len(ids) == 0:
        return y
    return np.delete(y, ids[ids < len(y)])


# --------------------------------------------------------------------------------------------
# bigWig reader (pure numpy; new in this file -- see the module docstring)
# --------------------------------------------------------------------------------------------
class BigWig:
    """Read the full-resolution data of a bigWig file (UCSC format, version 4).

    Only what binning needs: chromosome sizes and the (start, end, value) items of one
    chromosome, in file order. Zoom levels are ignored.
    """
    BIGWIG_MAGIC = 0x888FFC26
    BPT_MAGIC = 0x78CA8C91
    CIR_MAGIC = 0x2468ACE0

    def __init__(self, path):
        self.path = str(path)
        self.fh = open(self.path, "rb")
        head = self.fh.read(64)
        if struct.unpack("<I", head[:4])[0] == self.BIGWIG_MAGIC:
            self.e = "<"
        elif struct.unpack(">I", head[:4])[0] == self.BIGWIG_MAGIC:
            self.e = ">"
        else:
            raise ValueError(f"{path}: not a bigWig file")
        (_, _, _, ctree, _, findex, _, _, _, _, ubuf, _) = struct.unpack(
            self.e + "IHHQQQHHQQIQ", head)
        self.compressed = ubuf > 0
        self._chroms = self._read_chrom_tree(ctree)          # name -> (id, size)
        self._blocks = self._read_index(findex)              # (sci, eci, offset, size)

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _read(self, off, n):
        self.fh.seek(off)
        b = self.fh.read(n)
        if len(b) != n:
            raise ValueError(f"{self.path}: truncated at offset {off}")
        return b

    def chroms(self):
        return {c: v[1] for c, v in self._chroms.items()}

    def _read_chrom_tree(self, off):
        magic, _, key_size, _, _, _ = struct.unpack(self.e + "IIIIQQ", self._read(off, 32))
        if magic != self.BPT_MAGIC:
            raise ValueError(f"{self.path}: bad chromosome tree")
        out = {}

        def node(o):
            is_leaf, _, count = struct.unpack(self.e + "BBH", self._read(o, 4))
            item = key_size + 8
            raw = self._read(o + 4, count * item)
            for i in range(count):
                rec = raw[i * item:(i + 1) * item]
                key = rec[:key_size].split(b"\x00", 1)[0].decode()
                if is_leaf:
                    cid, size = struct.unpack(self.e + "II", rec[key_size:])
                    out[key] = (cid, size)
                else:
                    node(struct.unpack(self.e + "Q", rec[key_size:])[0])

        node(off + 32)
        return out

    def _read_index(self, off):
        if struct.unpack(self.e + "I", self._read(off, 4))[0] != self.CIR_MAGIC:
            raise ValueError(f"{self.path}: bad R-tree index")
        out = []

        def node(o):
            is_leaf, _, count = struct.unpack(self.e + "BBH", self._read(o, 4))
            if is_leaf:
                raw = self._read(o + 4, count * 32)
                for i in range(count):
                    sci, _, eci, _, doff, dsize = struct.unpack(
                        self.e + "IIIIQQ", raw[i * 32:(i + 1) * 32])
                    out.append((sci, eci, doff, dsize))
            else:
                raw = self._read(o + 4, count * 24)
                for i in range(count):
                    node(struct.unpack(self.e + "IIIIQ", raw[i * 24:(i + 1) * 24])[4])

        node(off + 48)
        out.sort(key=lambda b: b[2])                         # file order
        return out

    def items(self, chrom):
        """(starts, ends, values) of every data item on `chrom`, in file order."""
        cid = self._chroms[chrom][0]
        e = self.e
        sec = np.dtype([("cid", e + "u4"), ("start", e + "u4"), ("end", e + "u4"),
                        ("step", e + "u4"), ("span", e + "u4"), ("type", "u1"),
                        ("res", "u1"), ("n", e + "u2")])
        bg = np.dtype([("s", e + "u4"), ("e", e + "u4"), ("v", e + "f4")])
        vs = np.dtype([("s", e + "u4"), ("v", e + "f4")])
        fs = np.dtype(e + "f4")
        S, E, V = [], [], []
        for sci, eci, doff, dsize in self._blocks:
            if not (sci <= cid <= eci):
                continue
            buf = self._read(doff, dsize)
            if self.compressed:
                buf = zlib.decompress(buf)
            p = 0
            while p < len(buf):
                h = np.frombuffer(buf, sec, 1, p)[0]
                p += 24
                n, typ = int(h["n"]), int(h["type"])
                if typ == 1:
                    a = np.frombuffer(buf, bg, n, p)
                    p += 12 * n
                    s, en, v = a["s"], a["e"], a["v"]
                elif typ == 2:
                    a = np.frombuffer(buf, vs, n, p)
                    p += 8 * n
                    s, v = a["s"], a["v"]
                    en = s + np.uint32(h["span"])
                elif typ == 3:
                    v = np.frombuffer(buf, fs, n, p)
                    p += 4 * n
                    s = np.uint32(h["start"]) + np.arange(n, dtype=np.uint32) * np.uint32(h["step"])
                    en = s + np.uint32(h["span"])
                else:
                    raise ValueError(f"{self.path}: unknown section type {typ}")
                if int(h["cid"]) == cid:
                    S.append(s.astype(np.uint32))         # uint32 as on disk: bounds memory
                    E.append(en.astype(np.uint32))
                    V.append(v.astype(np.float32))
        if not S:
            z = np.zeros(0, np.uint32)
            return z, z, np.zeros(0, np.float32)
        return np.concatenate(S), np.concatenate(E), np.concatenate(V)


def _per_base(starts, ends, vals, clen):
    """Per-base float32 values of [0, clen), NaN where no item covers the base (pyBigWig.values)."""
    s = np.clip(starts, 0, clen)
    e = np.clip(ends, 0, clen)
    keep = e > s
    s, e, v = s[keep], e[keep], vals[keep]
    if s.size and np.all(s[1:] >= e[:-1]):
        # sorted and disjoint: one np.repeat over alternating gap / item runs, no index array
        runs_v = np.empty(2 * s.size + 1, np.float32)
        runs_n = np.empty(2 * s.size + 1, np.int64)
        runs_v[0::2] = np.nan
        runs_v[1::2] = v
        runs_n[0] = s[0]
        runs_n[2:-1:2] = s[1:] - e[:-1]
        runs_n[1::2] = e - s
        runs_n[-1] = clen - e[-1]
        return np.repeat(runs_v, runs_n)
    out = np.full(clen, np.nan, np.float32)
    for a, b, x in zip(s.tolist(), e.tolist(), v.tolist()):   # overlaps: file order, last wins
        out[a:b] = x
    return out


def bin_bigwig(path, chroms, window=WINDOW):
    """Bin a bigWig to the challenge grid (tracks022.load_official_bigwig, ported).

    n = ceil(chrom length / 25) bins; each bin is the mean of its 25 bases with missing = 0
    (the final partial bin is padded with zeros), then the bin holding the end of the last item
    is replaced by the exact mean over the bases that have data (bw_to_npy.py's fix, which
    pyBigWig's `stats(exact=True)` computes; 0 when no base has data). float32 out.
    """
    out = {}
    with BigWig(path) as bw:
        sizes = bw.chroms()
        for c in chroms:
            clen = sizes[c]
            n = (clen - 1) // window + 1
            st, en, va = bw.items(c)
            raw = _per_base(st, en, va, clen)
            last_end = int(en[-1]) if en.size else None
            if last_end is not None:                          # keep only items near the last bin
                lo = (last_end // window) * window
                near = np.flatnonzero((en > lo) & (st < lo + window))
                st, en, va = st[near].astype(np.int64), en[near].astype(np.int64), va[near]
            y = np.empty(n, np.float64)
            step = 1 << 22                                    # bins per chunk, bounds memory
            for a in range(0, n, step):
                b = min(n, a + step)
                buf = np.zeros((b - a) * window, np.float64)
                seg = raw[a * window: min(b * window, clen)]
                buf[:seg.shape[0]] = np.nan_to_num(seg)
                y[a:b] = buf.reshape(-1, window).mean(axis=1)
            if last_end is not None:
                last = last_end // window
                if 0 <= last < n:
                    lo, hi = last * window, min((last + 1) * window, clen)
                    tot, cov = 0.0, 0
                    for s0, e0, v0 in zip(st.tolist(), en.tolist(), va.tolist()):
                        o = min(e0, hi) - max(s0, lo)
                        if o > 0:
                            tot += o * float(v0)
                            cov += o
                    y[last] = tot / cov if cov else 0.0
            out[c] = y.astype(np.float32)
            del raw
    return out


def load_prediction(path, chroms, blacklist):
    """{chrom: float32} on the blacklist-deleted grid, from a .bigwig or a challenge .npy.

    A .npy is the challenge's own format (bw_to_npy.write_dict_to_npy): a pickled dict
    {chrom: array} on the raw 25 bp grid, blacklist not yet removed.
    """
    path = str(path)
    if path.endswith(".npy"):
        d = np.load(path, allow_pickle=True)[()]
        raw = {c: np.asarray(d[c], np.float32) for c in chroms}
    else:
        raw = bin_bigwig(path, chroms)
    return {c: apply_blacklist(raw[c], blacklist.get(c)) for c in chroms}


# --------------------------------------------------------------------------------------------
# the nine measures
# --------------------------------------------------------------------------------------------
def rank_average(x):
    """Tie-aware average ranks, 1-based (scipy.stats.rankdata 'average'), vectorised."""
    n = x.shape[0]
    order = np.argsort(x, kind="stable")
    xs = x[order]
    isnew = np.empty(n, bool)
    isnew[0] = True
    np.not_equal(xs[1:], xs[:-1], out=isnew[1:])
    starts = np.flatnonzero(isnew)
    ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) * 0.5 + 1.0
    out = np.empty(n, np.float64)
    out[order] = avg[np.cumsum(isnew) - 1]
    return out


def _interval_sse(yt_d, yp_d, chroms, intervals):
    sse, n = 0.0, 0.0
    for c in chroms:
        iv = intervals.get(c)
        if iv is None:
            continue
        yt, yp = yt_d[c], yp_d[c]
        L = len(yt)
        for a, b in zip(np.clip(iv[:, 0], 0, L), np.clip(iv[:, 1], 0, L)):
            if b <= a:
                continue
            d = yt[a:b] - yp[a:b]
            sse += float(np.dot(d, d))
            n += b - a
    return sse / n if n else float("nan")


def _top1_threshold(sorted_vals):
    """sort(y)[-n1] with n1 = int(0.01 N); n1 = 0 gives sort(y)[0], as the challenge's code."""
    n = sorted_vals.shape[0]
    n1 = int(n * 0.01)
    return sorted_vals[n - n1] if n1 else sorted_vals[0]


def score_reference(yt_d, yp_d, chroms, prom, body, enh, var_d=None):
    """The nine measures, written as the challenge computes them (eic_metrics.score, ported).

    Slow; it is the reference that ExpScorer is checked against.
    """
    from scipy.stats import rankdata
    yt = np.concatenate([yt_d[c] for c in chroms]).astype(np.float64)
    yp = np.concatenate([yp_d[c] for c in chroms]).astype(np.float64)
    yt_d = {c: yt_d[c].astype(np.float64) for c in chroms}
    yp_d = {c: yp_d[c].astype(np.float64) for c in chroms}
    d = yt - yp
    out = {"mse": float(np.mean(d * d)),
           "gwcorr": float(np.corrcoef(yt, yp)[0, 1]),
           "gwspear": float(np.corrcoef(rankdata(yt, "average"), rankdata(yp, "average"))[0, 1]),
           "mseprom": _interval_sse(yt_d, yp_d, chroms, prom),
           "msegene": _interval_sse(yt_d, yp_d, chroms, body),
           "mseenh": _interval_sse(yt_d, yp_d, chroms, enh)}
    if var_d is None:
        out["msevar"] = float("nan")
    else:
        v = np.concatenate([var_d[c] for c in chroms]).astype(np.float64)
        out["msevar"] = float(np.dot(d * d, v) / v.sum())
    sel = yt >= _top1_threshold(np.sort(yt))
    out["mse1obs"] = float(np.mean(d[sel] ** 2))
    sel = yp >= _top1_threshold(np.sort(yp))
    out["mse1imp"] = float(np.mean(d[sel] ** 2))
    return out


def _flat_idx(intervals, L):
    """Concatenated bin indices of [start, end) intervals clipped to L. Duplicates KEPT: a bin in
    two overlapping genes counts twice, as _interval_sse does."""
    if intervals is None or len(intervals) == 0:
        return np.zeros(0, np.int64)
    s = np.clip(intervals[:, 0], 0, L)
    e = np.clip(intervals[:, 1], 0, L)
    keep = e > s
    s, e = s[keep], e[keep]
    if s.size == 0:
        return np.zeros(0, np.int64)
    lens = e - s
    starts_at = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = np.ones(int(lens.sum()), np.int64)
    out[0] = s[0]
    if s.size > 1:
        out[starts_at[1:]] = s[1:] - e[:-1] + 1
    return np.cumsum(out)


def _subset_ranks(y, order, m, loc):
    """Average ranks (1-based) of y[m], laid out like y[m], and y[m] sorted (fastglobal.py)."""
    o = order[m[order]]
    v = y[o]
    n = v.shape[0]
    brk = np.empty(n, bool)
    brk[0] = True
    np.not_equal(v[1:], v[:-1], out=brk[1:])
    starts = np.flatnonzero(brk)
    avg = (starts + np.r_[starts[1:], n] + 1) / 2.0
    r = np.empty(n)
    r[loc[o]] = avg[np.cumsum(brk) - 1]
    return r, v


class ExpScorer:
    """Truth-side precomputation for one experiment, shared by every entry scored on it.

    metrics9.Scorer + fastglobal.FastScorer, ported: the five sum-type measures recombine from
    per-chromosome partials; gwspear, mse1obs and mse1imp come from one stable sort per track.
    """

    def __init__(self, truth, ann, var=None, chroms=CHROMS):
        self.chroms = list(chroms)
        prom, body, enh = ann
        self.truth = {c: truth[c].astype(np.float64) for c in self.chroms}
        self.n = {c: self.truth[c].shape[0] for c in self.chroms}
        self.idx = {name: {c: _flat_idx(a.get(c), self.n[c]) for c in self.chroms}
                    for name, a in (("prom", prom), ("gene", body), ("enh", enh))}
        self.tp = {c: (float(t.sum()), float(t @ t)) for c, t in self.truth.items()}
        self.var = None
        if var is not None:
            self.var = {c: np.asarray(var[c], np.float64)[: self.n[c]] for c in self.chroms}
            for c in self.chroms:
                if self.var[c].shape[0] != self.n[c]:
                    raise ValueError(f"msevar weights for {c}: {self.var[c].shape[0]} bins, "
                                     f"truth {self.n[c]}: different grids")
            self.vs = {c: float(self.var[c].sum()) for c in self.chroms}
        self.cs = sorted(self.chroms)
        self.cid = np.concatenate([np.full(self.n[c], i, np.int8) for i, c in enumerate(self.cs)])
        self.yt = np.concatenate([self.truth[c] for c in self.cs])
        self.ot = np.argsort(self.yt, kind="stable")

    def partials(self, pred):
        out = {}
        for c in self.chroms:
            t, p = self.truth[c], pred[c].astype(np.float64)
            d = t - p
            row = dict(sse=float(d @ d), sp=float(p.sum()), sp2=float(p @ p), stp=float(t @ p))
            if self.var is not None:
                row["wsse"] = float((d * d) @ self.var[c])
            for name in ("prom", "gene", "enh"):
                ix = self.idx[name][c]
                dd = d[ix]
                row[f"sse_{name}"] = float(dd @ dd) if ix.size else 0.0
                row[f"n_{name}"] = int(ix.size)
            out[c] = row
        return out

    def combine(self, part, chroms):
        cs = sorted(chroms)
        n = sum(self.n[c] for c in cs)
        st = sum(self.tp[c][0] for c in cs)
        st2 = sum(self.tp[c][1] for c in cs)
        sp = sum(part[c]["sp"] for c in cs)
        sp2 = sum(part[c]["sp2"] for c in cs)
        stp = sum(part[c]["stp"] for c in cs)
        cov = stp / n - (st / n) * (sp / n)
        vt = st2 / n - (st / n) ** 2
        vp = sp2 / n - (sp / n) ** 2
        den = np.sqrt(max(vt, 0.0) * max(vp, 0.0))
        out = {"mse": sum(part[c]["sse"] for c in cs) / n,
               "gwcorr": float(cov / den) if den > 0 else float("nan")}
        for name, key in (("prom", "mseprom"), ("gene", "msegene"), ("enh", "mseenh")):
            nn = sum(part[c][f"n_{name}"] for c in cs)
            out[key] = sum(part[c][f"sse_{name}"] for c in cs) / nn if nn else float("nan")
        if self.var is not None:
            vs = sum(self.vs[c] for c in cs)
            out["msevar"] = sum(part[c]["wsse"] for c in cs) / vs if vs > 0 else float("nan")
        else:
            out["msevar"] = float("nan")
        return out

    def global_measures(self, yp_all, op, chroms):
        m = np.isin(self.cid, [self.cs.index(c) for c in chroms])
        loc = np.cumsum(m) - 1
        rt, vt = _subset_ranks(self.yt, self.ot, m, loc)
        rp, vp = _subset_ranks(yp_all, op, m, loc)
        gwspear = float(np.corrcoef(rt, rp)[0, 1])
        yt, yp = self.yt[m], yp_all[m]
        sel = yt >= _top1_threshold(vt)
        d = yt[sel] - yp[sel]
        mse1obs = float(np.mean(d * d))
        sel = yp >= _top1_threshold(vp)
        d = yt[sel] - yp[sel]
        return {"gwspear": gwspear, "mse1obs": mse1obs, "mse1imp": float(np.mean(d * d))}

    def score_sets(self, pred, sets=CHROM_SETS):
        """{set name: {measure: value}} for every chromosome set."""
        part = self.partials(pred)
        yp_all = np.concatenate([pred[c].astype(np.float64) for c in self.cs])
        op = np.argsort(yp_all, kind="stable")
        out = {}
        for name, cs in sets:
            r = self.combine(part, cs)
            r.update(self.global_measures(yp_all, op, cs))
            out[name] = r
        return out


def two_pass_var(arrays):
    """Per-bin population variance across tracks (numpy.std(stack, axis=0)**2, one track at a
    time; varutil.py, ported)."""
    if len(arrays) < 2:
        raise ValueError("a variance over fewer than two tracks is identically zero")
    mean = np.zeros(arrays[0].shape[0], np.float64)
    for a in arrays:
        mean += a
    mean /= len(arrays)
    out = np.zeros_like(mean)
    tmp = np.empty_like(mean)
    for a in arrays:
        np.subtract(a, mean, out=tmp)
        tmp *= tmp
        out += tmp
    out /= len(arrays)
    return out


def align(pred, truth, chroms):
    """Truncate prediction and truth (both already blacklist-deleted) to their common length per
    chromosome; a difference of more than one bin means different grids (score_multi.py)."""
    P, T = {}, {}
    for c in chroms:
        if abs(pred[c].shape[0] - truth[c].shape[0]) > 1:
            raise ValueError(f"{c}: prediction {pred[c].shape[0]} vs truth "
                             f"{truth[c].shape[0]} bins: different grids")
        L = min(pred[c].shape[0], truth[c].shape[0])
        P[c], T[c] = pred[c][:L], truth[c][:L]
    return P, T


# --------------------------------------------------------------------------------------------
# ranking (rank_eic.py, ported): the organizers' statistic
# --------------------------------------------------------------------------------------------
def combined_ranks(rows, measures=MEASURES9):
    """Rank of each entry within one (experiment, bootstrap): sum over measures of
    log(rank/(n+1)), re-ranked. `rows` is a list of dicts."""
    from scipy.stats import rankdata
    n = len(rows)
    tot = np.zeros(n)
    for m in measures:
        v = np.array([r[m] for r in rows], np.float64)
        rk = rankdata(-v, "average") if m in HIGHER_IS_BETTER else rankdata(v, "average")
        tot += np.log(rk / float(n + 1))
    return rankdata(tot, "average")


def global_ranks(rows, measures=MEASURES9):
    """Leaderboard from bootstrap rows (team_id, cell, assay, bootstrap_id, measures...).

    Per (experiment, bootstrap) the combined rank r is squashed to min(0.5, r/(n+1)); a missing
    experiment counts 0.5. Per bootstrap the entries are ranked on the sum; `rank_stat` is the
    second-best of the ten bootstrap ranks and `score_mean` (mean over bootstraps of the mean
    squashed rank) breaks ties. Lower is better. Returns a list of dicts, best first.
    """
    from scipy.stats import rankdata
    teams = sorted({r["team_id"] for r in rows})
    groups = {}
    for r in rows:
        groups.setdefault((r["cell"], r["assay"]), {}).setdefault(int(r["bootstrap_id"]), []).append(r)
    boots = sorted({int(r["bootstrap_id"]) for r in rows})
    per_boot = {b: {t: [] for t in teams} for b in boots}
    for key in groups:                                        # experiment order does not matter
        for b, sub in groups[key].items():
            n = len(sub)
            ranks = combined_ranks(sub, measures)
            seen = set()
            for r, rk in zip(sub, ranks):
                per_boot[b][r["team_id"]].append(min(0.5, rk / (n + 1)))
                seen.add(r["team_id"])
            for t in set(teams) - seen:
                per_boot[b][t].append(0.5)
    score = {t: [] for t in teams}
    rank = {t: [] for t in teams}
    for b in boots:
        items = sorted(per_boot[b].items(), key=lambda x: sum(x[1]))
        rr = rankdata([sum(x[1]) for x in items])
        for (t, sc), r in zip(items, rr):
            score[t].append(sum(sc) / len(sc))
            rank[t].append(r)
    out = [dict(team_id=t, team=TEAM_NAME.get(t, str(t)), rank_stat=float(sorted(rank[t])[1]),
                score_mean=float(np.mean(score[t]))) for t in teams]
    out.sort(key=lambda d: (d["rank_stat"], d["score_mean"]))
    for i, d in enumerate(out):
        d["position"] = i + 1
    return out


def read_published(path):
    """MOESM4 rows (bootstrap rows of the 25 entries), sha256-pinned."""
    got = sha256_file(path)
    if got != MOESM4_SHA256:
        raise SystemExit(f"{path}: sha256 {got} is not the published MOESM4 ({MOESM4_SHA256})")
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = dict(team_id=int(r["team_id"]), cell=r["cell"], assay=r["assay"],
                     bootstrap_id=int(r["bootstrap_id"]))
            d.update({m: float(r[m]) for m in MEASURES9})
            rows.append(d)
    return rows


# --------------------------------------------------------------------------------------------
# truth cache: binned truth, msevar weights
# --------------------------------------------------------------------------------------------
def _atomic_save_npy(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _truth_exps(truth_dir):
    return sorted(p.stem for p in Path(truth_dir).glob("C*M*.bigwig"))


def _bin_truth_one(args):
    truth_dir, cache, exp = args
    out = Path(cache) / "truth" / exp
    if (out / "chrX.npy").exists():
        return exp, 0.0
    t0 = time.time()
    bl = load_blacklist_bins(asset_dir())
    y = bin_bigwig(Path(truth_dir) / f"{exp}.bigwig", CHROMS)
    for c in [c for c in CHROMS if c != "chrX"] + ["chrX"]:   # chrX last: it marks completion
        _atomic_save_npy(out / f"{c}.npy", apply_blacklist(y[c], bl.get(c)).astype(np.float32))
    return exp, time.time() - t0


def load_truth(cache, exp):
    return {c: np.load(Path(cache) / "truth" / exp / f"{c}.npy") for c in CHROMS}


def _var_one(args):
    cache, assay, members = args
    out = Path(cache) / "var" / f"var_{assay}.npz"
    if out.exists():
        return assay
    var = {}
    for c in CHROMS:
        arrs = [np.load(Path(cache) / "truth" / e / f"{c}.npy").astype(np.float64) for e in members]
        L = min(a.shape[0] for a in arrs)
        var[c] = two_pass_var([a[:L] for a in arrs])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **var)
    os.replace(tmp, out)
    (out.with_suffix(".json")).write_text(json.dumps({"assay": assay, "members": members}))
    return assay


def msevar_members(exps):
    by = {}
    for e in exps:
        if e not in MSEVAR_LEFT_OUT:
            by.setdefault(e[3:], []).append(e)
    return by


def load_var(cache, assay):
    p = Path(cache) / "var" / f"var_{assay}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {c: z[c] for c in CHROMS}


def prepare_truth(truth_dir, cache, workers):
    exps = _truth_exps(truth_dir)
    if not exps:
        raise SystemExit(f"no C##M##.bigwig under {truth_dir}")
    log(f"[truth] {len(exps)} truth tracks in {truth_dir}; cache {cache}")
    _pool_map(_bin_truth_one, [(str(truth_dir), str(cache), e) for e in exps], workers,
              lambda r: log(f"[truth] binned {r[0]} in {r[1]:.0f}s") if r[1] else None)
    mem = msevar_members(exps)
    _pool_map(_var_one, [(str(cache), a, m) for a, m in sorted(mem.items()) if len(m) >= 2],
              workers, lambda a: log(f"[truth] msevar weights {a}"))
    return exps


def _pool_map(fn, tasks, workers, on_result):
    if workers <= 1:
        for t in tasks:
            on_result(fn(t))
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fn, tasks):
            on_result(r)


# --------------------------------------------------------------------------------------------
# scoring one experiment for several entries
# --------------------------------------------------------------------------------------------
ROW_KEYS = ["entry", "team_id", "experiment", "cell", "assay", "chrom_set"] + MEASURES9


def _score_exp(args):
    """Score every (entry, prediction file) of one experiment; one row per chromosome set.

    Writes <cache>/scores/<entry key>/<exp>.tsv per entry, skipping existing files. On the first
    entry it also checks ExpScorer against score_reference on one chromosome set (1e-9)."""
    cache, exp, entries, validate = args
    todo = [(k, name, tid, p) for k, name, tid, p in entries
            if not (Path(cache) / "scores" / k / f"{exp}.tsv").exists()]
    if not todo:
        return exp, 0, 0.0, None
    t0 = time.time()
    adir = asset_dir()
    ann = load_annotations(adir)
    bl = load_blacklist_bins(adir)
    truth = load_truth(cache, exp)
    var = load_var(cache, exp[3:])
    scorers = {}
    worst = None
    for i, (key, name, tid, ppath) in enumerate(todo):
        pred = load_prediction(ppath, CHROMS, bl)
        P, T = align(pred, truth, CHROMS)
        lk = tuple(T[c].shape[0] for c in CHROMS)
        if lk not in scorers:
            scorers.clear()
            scorers[lk] = ExpScorer(T, ann, var)
        S = scorers[lk]
        res = S.score_sets(P)
        if validate and i == 0:
            ref = score_reference(S.truth, P, sorted(BOOTSTRAP_CHROM[0]), *ann, S.var)
            got = res["boot0"]
            worst = max(abs(ref[m] - got[m]) / max(abs(ref[m]), 1e-12)
                        for m in MEASURES9 if not np.isnan(ref[m]))
            if worst > 1e-9:
                raise SystemExit(f"{exp}: fast scorer differs from the reference by {worst:.2e}")
        out = Path(cache) / "scores" / key / f"{exp}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + f".tmp{os.getpid()}")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(ROW_KEYS)
            for sname, _ in CHROM_SETS:
                w.writerow([name, tid, exp, exp[:3], exp[3:], sname]
                           + [repr(float(res[sname][m])) for m in MEASURES9])
        os.replace(tmp, out)
    return exp, len(todo), time.time() - t0, worst


def read_score_rows(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            d = {k: r[k] for k in ("entry", "experiment", "cell", "assay", "chrom_set")}
            d["team_id"] = int(r["team_id"]) if r["team_id"] not in ("", "None") else r["entry"]
            d.update({m: float(r[m]) for m in MEASURES9})
            rows.append(d)
    return rows


def _default_cache(truth_dir):
    return f"{SCRATCH_EIC}/score_cache/{Path(os.path.realpath(truth_dir)).name}"


def _workers(arg):
    if arg:
        return arg
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) // 4)


def _run_scoring(truth_dir, cache, entries_by_exp, workers, shard):
    k, n = shard
    exps = [e for i, e in enumerate(sorted(entries_by_exp)) if i % n == k]
    tasks = [(str(cache), e, entries_by_exp[e], True) for e in exps]
    done = [0]

    def on(r):
        exp, nd, dt, worst = r
        done[0] += 1
        if nd:
            wtxt = f", fast vs reference {worst:.1e}" if worst is not None else ""
            log(f"[score] {exp}: {nd} entries in {dt:.0f}s{wtxt} ({done[0]}/{len(tasks)})")

    _pool_map(_score_exp, tasks, workers, on)


# --------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------
class _HashReader:
    """File-like wrapper that md5-hashes every byte read through it."""

    def __init__(self, fh):
        self.fh, self.md5, self.n = fh, hashlib.md5(), 0

    def read(self, size=-1):
        b = self.fh.read(size)
        self.md5.update(b)
        self.n += len(b)
        return b


def _fetch_record(record, out_dir):
    meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=120))
    files = meta["files"]
    if len(files) != 1:
        raise SystemExit(f"record {record}: expected one tarball, found {len(files)} files")
    f = files[0]
    want = f["checksum"].split(":", 1)[1]
    url = f["links"]["self"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[fetch] record {record}: {f['key']} {f['size']} bytes md5 {want} -> {out_dir}")
    resp = urllib.request.urlopen(url, timeout=600)
    hr = _HashReader(resp)
    kept, skipped = {}, []
    with tarfile.open(fileobj=hr, mode="r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts
            if not m.isfile():
                continue
            if len(parts) != 2:                               # not top level (e.g. old/...)
                skipped.append(m.name)
                continue
            src = tar.extractfile(m)
            dst = out_dir / parts[1]
            tmp = dst.with_name(dst.name + ".part")
            h = hashlib.md5()
            with open(tmp, "wb") as fh:
                for b in iter(lambda: src.read(1 << 22), b""):
                    h.update(b)
                    fh.write(b)
            os.replace(tmp, dst)
            kept[parts[1]] = h.hexdigest()
    while hr.read(1 << 22):                                   # hash the tarball to its end
        pass
    got = hr.md5.hexdigest()
    ok = got == want
    (out_dir / "MD5SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(kept.items())))
    (out_dir / "FETCH.json").write_text(json.dumps(dict(
        record=record, doi=meta.get("doi"), title=meta["metadata"]["title"], tarball=f["key"],
        tarball_bytes=hr.n, tarball_md5=got, zenodo_md5=want, md5_ok=ok,
        extracted=len(kept), skipped_not_top_level=len(skipped),
        skipped_example=skipped[:3]), indent=1))
    log(f"[fetch] record {record}: tarball md5 {got} {'==' if ok else '!='} zenodo {want}; "
        f"{len(kept)} top-level files kept, {len(skipped)} below top level skipped")
    if not ok:
        raise SystemExit(f"record {record}: md5 mismatch; do not use {out_dir}")
    return kept


def cmd_fetch_truth(a):
    out = Path(a.out)
    for sub in a.records.split(","):
        kept = _fetch_record(ZENODO[sub], out / sub)
        if a.compare_to and sub == "qnorm":
            ref = Path(a.compare_to)
            same = sum(1 for k, v in kept.items() if (ref / k).exists() and md5_file(ref / k) == v)
            log(f"[fetch] {sub}: {same} of {len(kept)} files byte-identical to {ref}")
    pub = Path(a.published_out)
    pub.parent.mkdir(parents=True, exist_ok=True)
    tmp = pub.with_name(pub.name + ".part")
    urllib.request.urlretrieve(MOESM4_URL, tmp)
    if sha256_file(tmp) != MOESM4_SHA256:
        raise SystemExit(f"{MOESM4_URL}: sha256 differs from the pinned MOESM4")
    os.replace(tmp, pub)
    log(f"[fetch] MOESM4 -> {pub} (sha256 ok)")
    return 0


def cmd_link_submissions(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for tid, name, src in TEAMS:
        dst = out / str(tid)
        if not dst.exists():
            os.symlink(src, dst)
        n = len(list(Path(dst).glob("C*M*.bigwig")))
        log(f"[link] {tid}\t{name}\t{n} bigWigs\t-> {src}")
    return 0


def cmd_check_bins(a):
    """Bitwise check of bin_bigwig (+ blacklist deletion) against reference .npy arrays."""
    bl = load_blacklist_bins(asset_dir())
    bad = 0
    for exp in a.exps.split(","):
        y = bin_bigwig(Path(a.bigwigs) / f"{exp}.bigwig", CHROMS)
        for c in CHROMS:
            ours = apply_blacklist(y[c], bl.get(c)).astype(np.float32)
            ref = np.load(Path(a.reference) / exp / f"{c}.npy")
            same = ours.shape == ref.shape and np.array_equal(ours.view(np.uint32), ref.view(np.uint32))
            if not same:
                bad += 1
                n = min(ours.shape[0], ref.shape[0])
                nd = int(np.sum(ours[:n] != ref[:n]))
                log(f"[check-bins] {exp} {c}: DIFFERS (shape {ours.shape} vs {ref.shape}, "
                    f"{nd} bins differ)")
        log(f"[check-bins] {exp}: done")
    log(f"[check-bins] chromosome arrays differing: {bad}")
    return 1 if bad else 0


def _entries_from_dir(pred_dir, exps, key, name):
    out = {}
    for e in exps:
        for ext in (".bigwig", ".bw", ".npy"):
            p = Path(pred_dir) / f"{e}{ext}"
            if p.exists():
                out.setdefault(e, []).append((key, name, "", str(p)))
                break
    return out


def cmd_score(a):
    cache = a.cache or _default_cache(a.truth)
    workers = _workers(a.workers)
    exps = prepare_truth(a.truth, cache, workers)
    name = a.entry or Path(os.path.realpath(a.pred)).name
    key = "pred_" + hashlib.sha256(os.path.realpath(a.pred).encode()).hexdigest()[:12]
    by_exp = _entries_from_dir(a.pred, exps, key, name)
    if not by_exp:
        raise SystemExit(f"no C##M##.bigwig/.npy under {a.pred} matches the truth")
    log(f"[score] {name}: {len(by_exp)} experiments")
    _run_scoring(a.truth, cache, by_exp, workers, (0, 1))
    rows = []
    for e in sorted(by_exp):
        rows += read_score_rows(Path(cache) / "scores" / key / f"{e}.tsv")
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ROW_KEYS)
        for r in rows:
            w.writerow([name, "", r["experiment"], r["cell"], r["assay"], r["chrom_set"]]
                       + [repr(r[m]) for m in MEASURES9])
    log(f"[score] wrote {len(rows)} rows to {a.out}")
    return 0


def _board_rows(score_rows):
    out = []
    for r in score_rows:
        if r["chrom_set"].startswith("boot"):
            d = dict(r)
            d["bootstrap_id"] = int(r["chrom_set"][4:])
            out.append(d)
    return out


def cmd_verify_published(a):
    cache = a.cache or _default_cache(a.truth)
    workers = _workers(a.workers)
    k, n = (int(x) for x in a.shard.split("/"))
    exps = prepare_truth(a.truth, cache, workers)
    missing_truth = sorted(set(BLIND_EXPS) - set(exps))
    if missing_truth:
        raise SystemExit(f"truth lacks {len(missing_truth)} blind experiments: {missing_truth[:5]}")
    by_exp = {}
    for tid, name, _ in TEAMS:
        for e, ents in _entries_from_dir(Path(a.submissions) / str(tid), BLIND_EXPS,
                                         f"team_{tid}", name).items():
            by_exp.setdefault(e, []).extend([(kk, nn, tid, p) for kk, nn, _, p in ents])
    n_files = sum(len(v) for v in by_exp.values())
    log(f"[verify] {n_files} submitted tracks of {len(TEAMS)} entries on {len(by_exp)} experiments")
    _run_scoring(a.truth, cache, by_exp, workers, (k, n))
    if n > 1:
        log(f"[verify] shard {k}/{n} scored; run without --shard to compare")
        return 0

    pub = read_published(a.published)
    ours = []
    for tid, name, _ in TEAMS:
        for e in BLIND_EXPS:
            p = Path(cache) / "scores" / f"team_{tid}" / f"{e}.tsv"
            if p.exists():
                ours += _board_rows(read_score_rows(p))
    key = lambda r: (r["team_id"], r["cell"], r["assay"], r["bootstrap_id"])
    pmap = {key(r): r for r in pub}
    joined = [(r, pmap[key(r)]) for r in ours if key(r) in pmap]
    log(f"[verify] rows: ours {len(ours)}, published {len(pub)}, joined {len(joined)}")
    measures_out = []
    worst_rel = 0.0
    for m in MEASURES9:
        absd = np.array([abs(o[m] - p[m]) for o, p in joined])
        rel = np.array([abs(o[m] - p[m]) / max(abs(p[m]), 1e-12) for o, p in joined])
        i = int(np.argmax(rel))
        o, p = joined[i]
        measures_out.append(dict(measure=m, n=len(joined), max_abs_diff=float(absd.max()),
                                 max_rel_diff=float(rel.max()), median_rel_diff=float(np.median(rel)),
                                 worst_row=f"{TEAM_NAME[o['team_id']]}|{o['cell']}{o['assay']}|boot{o['bootstrap_id']}",
                                 worst_ours=o[m], worst_published=p[m]))
        worst_rel = max(worst_rel, float(rel.max()))
    bo = {d["team_id"]: d for d in global_ranks(ours)}
    bp = global_ranks(pub)
    board = []
    for d in bp:
        o = bo.get(d["team_id"], {})
        board.append(dict(team_id=d["team_id"], team=d["team"],
                          position_published=d["position"], position_ours=o.get("position"),
                          rank_stat_published=d["rank_stat"], rank_stat_ours=o.get("rank_stat"),
                          score_mean_published=d["score_mean"], score_mean_ours=o.get("score_mean")))
    diffs = [b for b in board if b["position_ours"] != b["position_published"]]

    rep = Path(a.report_dir or cache)
    rep.mkdir(parents=True, exist_ok=True)
    for fname, recs in (("verify_published.tsv", board),
                        ("verify_published_measures.tsv", measures_out)):
        with open(rep / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(recs)
    log("measure\tn\tmax_abs_diff\tmax_rel_diff\tmedian_rel_diff\tworst_row")
    for r in measures_out:
        log(f"{r['measure']}\t{r['n']}\t{r['max_abs_diff']:.3e}\t{r['max_rel_diff']:.3e}\t"
            f"{r['median_rel_diff']:.3e}\t{r['worst_row']}")
    log("pos_pub\tpos_ours\tscore_mean_pub\tscore_mean_ours\tteam")
    for b in board:
        log(f"{b['position_published']}\t{b['position_ours']}\t{b['score_mean_published']:.6f}\t"
            f"{(b['score_mean_ours'] if b['score_mean_ours'] is not None else float('nan')):.6f}\t"
            f"{b['team']}")
    log(f"reports: {rep / 'verify_published.tsv'} {rep / 'verify_published_measures.tsv'}")
    log(f"max relative difference: {worst_rel:.3e}")
    log(f"max relative difference < 1e-6: {worst_rel < 1e-6}")
    if diffs:
        for b in diffs:
            log(f"rank difference: {b['team']} published {b['position_published']} "
                f"ours {b['position_ours']}")
    complete = len(joined) == len(pub)
    log(f"rows complete: {complete}")
    log(f"ranks identical: {not diffs and complete}")
    return 0 if (not diffs and complete) else 1


def cmd_board(a):
    """Each scored entry is added alone, as a 26th entry, to the 25 published ones and ranked
    (Max's protocol, never two of ours in one field)."""
    pub = read_published(a.published)
    out = []
    base = {d["team_id"]: d for d in global_ranks(pub)}
    for d in base.values():
        out.append(dict(entry=d["team"], source="published MOESM4", added_alone="",
                        position=d["position"], of=len(base), rank_stat=d["rank_stat"],
                        score_mean=d["score_mean"]))
    for path in a.scores:
        rows = _board_rows(read_score_rows(path))
        name = rows[0]["entry"]
        for r in rows:
            r["team_id"] = -1
        res = {d["team_id"]: d for d in global_ranks(pub + rows)}[-1]
        out.append(dict(entry=name, source=str(path), added_alone="yes", position=res["position"],
                        of=len(base) + 1, rank_stat=res["rank_stat"], score_mean=res["score_mean"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sorted(out, key=lambda d: (d["added_alone"] == "yes", d["position"])))
    log(f"[board] wrote {len(out)} rows to {a.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("fetch-truth", help="download and md5-check the Zenodo truth records")
    p.add_argument("--out", required=True, help="writes <out>/qnorm and <out>/se_unnormalized")
    p.add_argument("--records", default="qnorm,se_unnormalized")
    p.add_argument("--compare-to", default="", help="another copy of the qnorm files to md5-compare")
    p.add_argument("--published-out", default=DEFAULT_PUBLISHED)
    p.set_defaults(fn=cmd_fetch_truth)

    p = sp.add_parser("link-submissions", help="symlink the 25 published entries' folders")
    p.add_argument("--out", default=DEFAULT_SUBMISSIONS)
    p.set_defaults(fn=cmd_link_submissions)

    p = sp.add_parser("check-bins", help="bitwise check of the bigWig reader vs reference npy")
    p.add_argument("--bigwigs", required=True)
    p.add_argument("--reference", required=True, help="<ref>/<exp>/<chrom>.npy")
    p.add_argument("--exps", required=True, help="comma-separated C##M##")
    p.set_defaults(fn=cmd_check_bins)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--truth", required=True, help="folder of C##M##.bigwig truth tracks")
    common.add_argument("--cache", default="", help="default /scratch/mforooz/onion/eic/score_cache/<truth folder name>")
    common.add_argument("--workers", type=int, default=0, help="default SLURM_CPUS_PER_TASK // 4")

    p = sp.add_parser("score", parents=[common], help="nine measures for a folder of predictions")
    p.add_argument("--pred", required=True, help="folder of C##M##.bigwig or challenge .npy")
    p.add_argument("--entry", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_score)

    p = sp.add_parser("verify-published", parents=[common],
                      help="score the 25 published entries; compare with MOESM4")
    p.add_argument("--submissions", default=DEFAULT_SUBMISSIONS, help="<dir>/<team_id>/C##M##.bigwig")
    p.add_argument("--published", default=DEFAULT_PUBLISHED)
    p.add_argument("--shard", default="0/1", help="k/n: score experiments i with i %% n == k, no compare")
    p.add_argument("--report-dir", default="")
    p.set_defaults(fn=cmd_verify_published)

    p = sp.add_parser("board", help="add scored entries alone to the published field")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--published", default=DEFAULT_PUBLISHED)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_board)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
