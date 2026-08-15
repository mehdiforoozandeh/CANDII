"""Independent verification that a baked h5 faithfully mirrors the source npz tree.

    python -m candi.verify_bake --h5 /scratch/$USER/candi/eic_full.h5 \
        --root /project/6014832/mforooz/DATA_CANDI_EIC \
        --fasta /project/6014832/mforooz/EpiDenoise/data/hg38.fa --n 300

WHY THIS EXISTS AND WHY IT DOES NOT IMPORT THE BAKE
`candi.prep.bake --verify` runs INTERNAL-CONSISTENCY gates (F3 no zero-filled meta row, F4 the
DSF depth ladder, F7 raw integer counts, F15 control presence). Those catch a corrupt h5, but they
read only the h5: a bake that faithfully wrote the WRONG SOURCE would pass all of them. This module
answers the different question -- does the h5 equal the data on disk -- and so it re-reads the npz
tree with plain numpy rather than going through `handler.py`. Sharing the reader would mean a bug in
the reader cancels itself out on both sides of the comparison.

THE CHECK THAT MATTERS MOST IS V2. Assay columns are positional in this model: assay `a` must be
column `a` for EVERY biosample, with a MISSING sentinel where the assay is absent. The source tree
has no such convention -- each biosample directory simply holds the assays it has. A permuted or
shifted column mapping produces an h5 that passes every structural gate, trains without complaint,
and yields nonsense. V2 pins the mapping by CONTENT: the counts in column `fi` must equal the npz of
the assay that `h5.attrs["assays"][fi]` names.

SOURCE LAYOUT, as re-derived here
  <root>/<biosample>/<assay>/signal_DSF{k}_res{res}/{chrom}.npz   raw integer counts
  <root>/<biosample>/<assay>/signal_BW_res{res}/{chrom}.npz       -log10 p-value signal
  <root>/<biosample>/<assay>/peaks_res{res}/{chrom}.npz           peak calls
  <root>/<biosample>/chipseq-control/signal_DSF{k}_res{res}/...   input control
  <root>/<biosample>/<assay>/file_metadata.json                   read length, run type
  <root>/<biosample>/<assay>/signal_DSF{k}_res{res}/metadata.json depth
Each npz holds ONE array under its first key, spanning a whole chromosome at `res` bp. A window
starting at `start` bp maps to `arr[start // res : start // res + context_bins]`.

TRANSFORMS THE BAKE APPLIES, which a naive comparison would flag as corruption
  counts  raw integers, untransformed        -> compared EXACTLY
  peaks   raw, untransformed                 -> compared exactly
  control raw, untransformed                 -> compared exactly
  pval    ARCSINH-TRANSFORMED at bake time (`load_bios_BW(..., arcsinh=True)`) and stored float16,
          so it is compared against arcsinh(source) at float16 tolerance. The check reports which of
          {raw, arcsinh} the stored values actually match, so the convention is verified rather than
          assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

CONTROL_DIR = "chipseq-control"


class Result:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, cid: str, ok: Optional[bool], msg: str) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((cid, status, msg))
        print(f"[{status:4s}] {cid:4s} {msg}", flush=True)

    @property
    def failed(self):
        return [c for c, s, _ in self.rows if s == "FAIL"]

    @property
    def skipped(self):
        return [c for c, s, _ in self.rows if s == "SKIP"]


# ---------------------------------------------------------------------------
# source readers — plain numpy, deliberately NOT handler.py
# ---------------------------------------------------------------------------

def _npz(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z[z.files[0]])


def _src_counts(root, bios, assay, dsf, res, chrom):
    return _npz(os.path.join(root, bios, assay, f"signal_DSF{dsf}_res{res}", f"{chrom}.npz"))


def _src_pval(root, bios, assay, res, chrom):
    return _npz(os.path.join(root, bios, assay, f"signal_BW_res{res}", f"{chrom}.npz"))


def _src_peaks(root, bios, assay, res, chrom):
    return _npz(os.path.join(root, bios, assay, f"peaks_res{res}", f"{chrom}.npz"))


def _src_control(root, bios, dsf, res, chrom):
    return _npz(os.path.join(root, bios, CONTROL_DIR, f"signal_DSF{dsf}_res{res}", f"{chrom}.npz"))


def _src_json(root, bios, assay, name, sub=None) -> Optional[dict]:
    p = os.path.join(root, bios, assay, sub, name) if sub else os.path.join(root, bios, assay, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _assays_on_disk(root, bios) -> List[str]:
    d = os.path.join(root, bios)
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d)
                  if os.path.isdir(os.path.join(d, x)) and x != CONTROL_DIR)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def _sample_units(h5, root, res, rng, n) -> List[Tuple[str, str, int, int]]:
    """(biosample, assay, assay_col, window_index) triples that are AVAILABLE in the h5."""
    order = json.loads(h5["biosamples"].attrs["order"])
    assays = json.loads(h5.attrs["assays"])
    n_win = h5["windows/start"].shape[0]
    pool = []
    for b in order:
        g = h5["biosamples"][b.replace("/", "_")]
        depth = np.asarray(g["meta_dsf1"][0])
        for fi in np.where(depth != -1.0)[0]:
            pool.append((b, assays[int(fi)], int(fi)))
    if not pool:
        return []
    return [(*rng.choice(pool), rng.randrange(n_win)) for _ in range(n)]


def check_V1_coverage(h5, root, res, r: Result) -> None:
    """Every (biosample, assay) on disk must be marked available in the h5, and vice versa."""
    order = json.loads(h5["biosamples"].attrs["order"])
    assays = json.loads(h5.attrs["assays"])
    missing_in_h5, extra_in_h5, n_pairs = [], [], 0
    for b in order:
        g = h5["biosamples"][b.replace("/", "_")]
        depth = np.asarray(g["meta_dsf1"][0])
        disk = set(_assays_on_disk(root, b))
        for fi, a in enumerate(assays):
            avail_h5 = bool(depth[fi] != -1.0)
            # "on disk" means the assay dir exists AND carries the DSF1 counts the bake reads
            avail_disk = a in disk and _src_counts(root, b, a, 1, res, "chr19") is not None
            n_pairs += 1
            if avail_disk and not avail_h5:
                missing_in_h5.append(f"{b}/{a}")
            if avail_h5 and not avail_disk:
                extra_in_h5.append(f"{b}/{a}")
    ok = not missing_in_h5 and not extra_in_h5
    r.add("V1", ok,
          f"availability bijection over {n_pairs} (biosample, assay) slots: "
          f"{len(missing_in_h5)} on disk but absent from h5, {len(extra_in_h5)} in h5 but not on disk"
          + ("" if ok else f"; missing={missing_in_h5[:5]} extra={extra_in_h5[:5]}"))


def check_V2_column_identity(h5, root, res, ctx, units, r: Result) -> None:
    """THE critical one: column fi must hold the assay h5.attrs['assays'][fi] names."""
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    assays = json.loads(h5.attrs["assays"])
    checked = 0
    wrong: List[str] = []
    ambiguous = 0
    for b, a, fi, wi in units:
        src = _src_counts(root, b, a, 1, res, chroms[wi])
        if src is None:
            continue
        s = int(starts[wi]) // res
        want = np.asarray(src[s:s + ctx]).ravel()
        if want.size != ctx:
            continue
        g = h5["biosamples"][b.replace("/", "_")]
        got = np.asarray(g["counts_dsf1"][wi, :, fi])
        checked += 1
        if not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
            wrong.append(f"{b}/{a}@col{fi}/win{wi}")
            continue
        # a window that is all-zero in every column proves nothing about the mapping
        allcols = np.asarray(g["counts_dsf1"][wi])
        if int(allcols.max()) == 0:
            ambiguous += 1
    if checked == 0:
        r.add("V2", None, "no comparable (biosample, assay, window) unit sampled")
        return
    informative = checked - ambiguous
    r.add("V2", not wrong and informative > 0,
          f"assay->column mapping verified BY CONTENT on {checked} units "
          f"({informative} informative, {ambiguous} all-zero and therefore uninformative); "
          f"{len(wrong)} mismatched" + ("" if not wrong else f": {wrong[:5]}")
          + (" — WARNING: no informative unit" if informative == 0 else ""))


def check_V3_counts(h5, root, res, ctx, units, r: Result) -> None:
    """Raw counts must match the source EXACTLY, at every DSF level."""
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    dsfs = json.loads(h5.attrs["dsf_list"])
    per_dsf = {d: [0, 0] for d in dsfs}          # [checked, mismatched]
    examples: List[str] = []
    for b, a, fi, wi in units:
        g = h5["biosamples"][b.replace("/", "_")]
        s = int(starts[wi]) // res
        for d in dsfs:
            src = _src_counts(root, b, a, d, res, chroms[wi])
            if src is None:
                continue
            want = np.asarray(src[s:s + ctx]).ravel()
            if want.size != ctx:
                continue
            got = np.asarray(g[f"counts_dsf{d}"][wi, :, fi])
            per_dsf[d][0] += 1
            if not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
                per_dsf[d][1] += 1
                if len(examples) < 5:
                    diff = int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max())
                    examples.append(f"{b}/{a}/dsf{d}/win{wi} maxdiff={diff}")
    tot = sum(v[0] for v in per_dsf.values())
    bad = sum(v[1] for v in per_dsf.values())
    detail = ", ".join(f"dsf{d}: {v[0]-v[1]}/{v[0]}" for d, v in sorted(per_dsf.items()))
    if tot == 0:
        r.add("V3", None, "no counts comparison was possible")
        return
    r.add("V3", bad == 0, f"counts exact on {tot-bad}/{tot} (unit x dsf) slices — {detail}"
          + ("" if not bad else f"; e.g. {examples}"))


def check_V4_windows(h5, res, ctx, r: Result) -> None:
    """Window coordinates must be self-consistent and aligned to the resolution grid."""
    starts = np.asarray(h5["windows/start"])
    ends = np.asarray(h5["windows/end"])
    span = ctx * res
    bad_span = int((ends - starts != span).sum())
    bad_align = int((starts % res != 0).sum())
    tr = set(json.loads(h5.attrs["train_chroms"]))
    ev = set(json.loads(h5.attrs["eval_chroms"]))
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    n_tr = sum(1 for c in chroms if c in tr)
    n_ev = sum(1 for c in chroms if c in ev)
    n_other = len(chroms) - n_tr - n_ev
    ok = bad_span == 0 and bad_align == 0 and not (tr & ev)
    r.add("V4", ok,
          f"{len(starts)} windows: {bad_span} wrong span (expect {span} bp), {bad_align} off the "
          f"{res} bp grid; train {n_tr} / eval {n_ev} / type2 {n_other}; train∩eval={sorted(tr & ev)}")


def check_V5_pval(h5, root, res, ctx, units, r: Result) -> None:
    """pval is arcsinh-transformed at bake time — verify that, do not assume it."""
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    n_arc = n_raw = n_neither = 0
    worst = 0.0
    for b, a, fi, wi in units:
        src = _src_pval(root, b, a, res, chroms[wi])
        if src is None:
            continue
        s = int(starts[wi]) // res
        want = np.asarray(src[s:s + ctx], dtype=np.float64).ravel()
        if want.size != ctx:
            continue
        got = np.asarray(h5["biosamples"][b.replace("/", "_")]["pval"][wi, :, fi], dtype=np.float64)
        d_arc = float(np.abs(got - np.arcsinh(want)).max())
        d_raw = float(np.abs(got - want).max())
        tol = 1e-2 + 1e-2 * float(np.abs(np.arcsinh(want)).max())   # float16 storage
        if d_arc <= tol:
            n_arc += 1
            worst = max(worst, d_arc)
        elif d_raw <= tol:
            n_raw += 1
        else:
            n_neither += 1
    tot = n_arc + n_raw + n_neither
    if tot == 0:
        r.add("V5", None, "no pval comparison was possible")
        return
    r.add("V5", n_arc == tot,
          f"pval matches arcsinh(source) on {n_arc}/{tot} units (max |delta| {worst:.4g}, float16 "
          f"storage); raw-matching {n_raw}, matching neither {n_neither}")


def check_V6_peaks(h5, root, res, ctx, units, r: Result) -> None:
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    tot = bad = 0
    for b, a, fi, wi in units:
        src = _src_peaks(root, b, a, res, chroms[wi])
        if src is None:
            continue
        s = int(starts[wi]) // res
        want = np.asarray(src[s:s + ctx]).ravel()
        if want.size != ctx:
            continue
        got = np.asarray(h5["biosamples"][b.replace("/", "_")]["peaks"][wi, :, fi])
        tot += 1
        if not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
            bad += 1
    if tot == 0:
        r.add("V6", None, "no peaks comparison was possible")
        return
    r.add("V6", bad == 0, f"peaks exact on {tot-bad}/{tot} units")


def check_V7_control(h5, root, res, ctx, units, r: Result) -> None:
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    tot = bad = absent = 0
    seen = set()
    for b, a, fi, wi in units:
        if (b, wi) in seen:
            continue
        seen.add((b, wi))
        src = _src_control(root, b, 1, res, chroms[wi])
        got = np.asarray(h5["biosamples"][b.replace("/", "_")]["control"][wi, :, 0], dtype=np.float64)
        if src is None:
            # legitimately absent (e.g. B_DND-41 is accessibility-only) -> sentinel, not zeros
            absent += 1
            continue
        s = int(starts[wi]) // res
        want = np.asarray(src[s:s + ctx], dtype=np.float64).ravel()
        if want.size != ctx:
            continue
        tot += 1
        if not np.allclose(got, want, rtol=0, atol=1e-6):
            bad += 1
    if tot == 0 and absent == 0:
        r.add("V7", None, "no control comparison was possible")
        return
    r.add("V7", bad == 0, f"control exact on {tot-bad}/{tot} (biosample, window) pairs; "
                          f"{absent} had no chipseq-control on disk (expected for ATAC/DNase-only cells)")


def check_V8_dna(h5, fasta_path, res, ctx, units, r: Result) -> None:
    """One-hot DNA must equal the reference at that locus, and be a valid one-hot."""
    if not fasta_path:
        r.add("V8", None, "no --fasta given; DNA not verified against the reference")
        return
    try:
        import pysam
    except ImportError:
        r.add("V8", None, "pysam unavailable; DNA not verified")
        return
    fa = pysam.FastaFile(fasta_path)
    starts = np.asarray(h5["windows/start"])
    chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["windows/chrom"][:]]
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    tot = bad = badonehot = 0
    seen = set()
    for b, a, fi, wi in units:
        if (b, wi) in seen or len(seen) >= 40:
            continue
        seen.add((b, wi))
        s, c = int(starts[wi]), chroms[wi]
        try:
            seq = fa.fetch(c, s, s + ctx * res).upper()
        except (KeyError, ValueError):
            continue
        if len(seq) != ctx * res:
            continue
        got = np.asarray(h5["biosamples"][b.replace("/", "_")]["dna"][wi], dtype=np.int8)
        rs = got.sum(axis=1)
        if not np.all((rs == 0) | (rs == 1)):
            badonehot += 1
        want = np.zeros_like(got)
        for j, ch in enumerate(seq):
            k = idx.get(ch)
            if k is not None:
                want[j, k] = 1
        tot += 1
        if not np.array_equal(got, want):
            bad += 1
    if tot == 0:
        r.add("V8", None, "no DNA window could be compared against the reference")
        return
    r.add("V8", bad == 0 and badonehot == 0,
          f"one-hot DNA matches {fasta_path} on {tot-bad}/{tot} windows; "
          f"{badonehot} had a row sum outside {{0,1}} (0 is correct for N)")


def check_V9_metadata(h5, root, res, units, r: Result) -> None:
    """meta row 0 must be log2(depth) from the source json, and row 1 the assay's own column index."""
    dsfs = json.loads(h5.attrs["dsf_list"])
    assays = json.loads(h5.attrs["assays"])
    tot = bad_depth = bad_assay = 0
    examples: List[str] = []
    seen = set()
    for b, a, fi, wi in units:
        if (b, a) in seen:
            continue
        seen.add((b, a))
        js = _src_json(root, b, a, "metadata.json", sub=f"signal_DSF1_res{res}")
        g = h5["biosamples"][b.replace("/", "_")]
        md = np.asarray(g["meta_dsf1"])
        tot += 1
        if float(md[1, fi]) != float(fi):
            bad_assay += 1
        depth = None
        if isinstance(js, dict):
            for k in ("depth", "sequencing_depth", "total_reads", "n_reads"):
                if k in js:
                    depth = float(js[k])
                    break
        if depth and depth > 0:
            if abs(float(md[0, fi]) - math.log2(depth)) > 1e-3:
                bad_depth += 1
                if len(examples) < 4:
                    examples.append(f"{b}/{a}: h5 {float(md[0, fi]):.4f} vs log2({depth:.0f})="
                                    f"{math.log2(depth):.4f}")
    if tot == 0:
        r.add("V9", None, "no metadata comparison was possible")
        return
    r.add("V9", bad_assay == 0 and bad_depth == 0,
          f"metadata on {tot} (biosample, assay) pairs: {bad_assay} wrong assay_id row, "
          f"{bad_depth} wrong log2 depth" + ("" if not examples else f"; e.g. {examples}"))


# Same bar as the bake's own F4 gate (prep/bake.py). It is NOT float slop -- it is physical.
# Downsampling by k takes a RANDOM SUBSAMPLE of the reads, so the level-k depth is only
# approximately depth_1 / k. Measured on the probe bake the worst deviation over all 18
# (biosample, assay, level) triples was 0.00105 bits: 6,757,467 reads against an exact 6,762,392,
# i.e. 0.07%. An earlier draft of this check used 1e-3 and reported that single triple as a bake
# defect. Do not tighten this without re-reading that sentence.
DSF_DEPTH_TOL = 0.05


def check_V10_dsf_ladder(h5, units, r: Result) -> None:
    """DSF levels must be a REAL downsample: depth drops by ~log2(k) and the counts actually differ."""
    dsfs = sorted(json.loads(h5.attrs["dsf_list"]))
    tot = bad_depth = identical = 0
    worst = 0.0
    worst_at = ""
    seen = set()
    for b, a, fi, wi in units:
        if (b, a, wi) in seen:
            continue
        seen.add((b, a, wi))
        g = h5["biosamples"][b.replace("/", "_")]
        d1 = float(np.asarray(g["meta_dsf1"])[0, fi])
        if d1 == -1.0:
            continue
        c1 = np.asarray(g["counts_dsf1"][wi, :, fi])
        for k in dsfs[1:]:
            dk = float(np.asarray(g[f"meta_dsf{k}"])[0, fi])
            if dk == -1.0:
                continue
            tot += 1
            err = abs(dk - (d1 - math.log2(k)))
            if err > worst:
                worst, worst_at = err, f"{b}/{a}/dsf{k}"
            if err >= DSF_DEPTH_TOL:
                bad_depth += 1
            ck = np.asarray(g[f"counts_dsf{k}"][wi, :, fi])
            if int(c1.sum()) > 0 and np.array_equal(c1, ck):
                identical += 1
    if tot == 0:
        r.add("V10", None, "no DSF ladder comparison was possible")
        return
    r.add("V10", bad_depth == 0 and identical == 0,
          f"DSF ladder on {tot} (unit x level) pairs: {bad_depth} exceeded the {DSF_DEPTH_TOL} bit "
          f"tolerance on depth = dsf1 - log2(k) (worst {worst:.5f} at {worst_at}); "
          f"{identical} had counts BYTE-IDENTICAL to dsf1 on a non-empty window (a copy, not a downsample)")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(h5_path: str, root: str, *, fasta: Optional[str] = None, n: int = 300,
        seed: int = 0) -> Result:
    r = Result()
    rng = random.Random(seed)
    with h5py.File(h5_path, "r") as h5:
        res = int(h5.attrs["resolution"])
        ctx = int(h5.attrs["context_bins"])
        assays = json.loads(h5.attrs["assays"])
        order = json.loads(h5["biosamples"].attrs["order"])
        print(f"[vb] h5={h5_path}\n[vb] root={root}\n[vb] {len(assays)} assays x {len(order)} "
              f"biosamples, {h5['windows/start'].shape[0]} windows, res={res} ctx={ctx}, "
              f"dsf={json.loads(h5.attrs['dsf_list'])}", flush=True)
        units = _sample_units(h5, root, res, rng, n)
        print(f"[vb] sampled {len(units)} (biosample, assay, window) units at seed {seed}", flush=True)

        check_V4_windows(h5, res, ctx, r)
        check_V1_coverage(h5, root, res, r)
        check_V2_column_identity(h5, root, res, ctx, units, r)
        check_V3_counts(h5, root, res, ctx, units, r)
        check_V5_pval(h5, root, res, ctx, units, r)
        check_V6_peaks(h5, root, res, ctx, units, r)
        check_V7_control(h5, root, res, ctx, units, r)
        check_V8_dna(h5, fasta, res, ctx, units, r)
        check_V9_metadata(h5, root, res, units, r)
        check_V10_dsf_ladder(h5, units, r)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--root", required=True, help="the ENCODE-style source tree the bake read")
    ap.add_argument("--fasta", default=None, help="reference fasta, to verify the one-hot DNA")
    ap.add_argument("--n", type=int, default=300, help="sampled (biosample, assay, window) units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    r = run(a.h5, a.root, fasta=a.fasta, n=a.n, seed=a.seed)
    print("\n" + "=" * 78)
    for cid, status, msg in r.rows:
        print(f"{status:4s}  {cid:4s}  {msg}")
    print("=" * 78)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            [dict(id=c, status=s, message=m) for c, s, m in r.rows], indent=2))
    if r.failed:
        print(f"[vb] FAILED: {r.failed}")
        return 1
    if r.skipped:
        print(f"[vb] passed, but SKIPPED: {r.skipped} — a skipped V2 or V3 is not a verified bake")
        return 2
    print("[vb] BAKE VERIFIED AGAINST SOURCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
