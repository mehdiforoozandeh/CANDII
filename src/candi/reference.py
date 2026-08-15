"""Per-assay, per-position average reference over `T_` biosamples — the h74 deviation target.

h27 says the average epigenome across cell types is a brutally strong predictor and that real skill is
the cell-type DEVIATION on top of it. h73's arms clear the per-assay marginal by only ~0.02 CRPS, so
that deviation is largely unlearned. This module builds the reference so the model can be asked to
predict the deviation directly, as a GLM offset.

THREE THINGS THIS FILE EXISTS TO GET RIGHT
------------------------------------------
1. **"Residual" is not subtraction.** The trained loss is NB NLL over counts; you cannot subtract a
   reference from a count and keep a valid likelihood. The reference enters as an OFFSET ON THE LOG
   MEAN, exactly like the depth size-factor the decoder already carries:

       log2_mu = (d - depth_center) + log2(R + c) + eta

   so `eta` — the thing the network computes — is `log2(mu / R)`, the log-deviation from the average
   cell. Standard GLM offset; h27's mechanism in the natural parameterization for counts.

2. **The reference must be DEPTH-FREE or it fights the depth offset.** `R` averages over cells
   sequenced at different depths, so a raw-count average silently carries a depth mixture that the
   existing offset then re-applies. Every contribution is therefore rescaled to `depth_center` before
   summing (`scale = 2**(depth_center - d)`), which makes `R` a pure shape and lets the two offsets
   COMPOSE. Asserted by `tests/test_reference.py::test_L3_reference_is_depth_free`.

3. **Leave-one-out is mandatory or the reference leaks the answer.** A mean over ALL `T_` cells
   contains cell *i*'s own value for the assay cell *i* is being asked to predict. At 51 cells that
   is a 1/51 leak; for an assay held by 2 cells it is 50% of the answer handed straight to the model
   through the offset. So `sum` and `count` are cached rather than the mean, and the reference is
   formed per batch as

       R_-i = (sum - x_i) / (count - present_i)

   `present_i` is 0 for an assay cell *i* does not have, which makes the same expression correct for
   the imputation columns without a branch. At eval the imputation target is a `V_`/`B_` cell that
   never contributed, so the subtraction is a no-op there — but the DENOISING columns of the same
   forward pass are the `T_` cell's own assays, where it is live and necessary.

**HARD RULE: no `V_` or `B_` biosample contributes to any reference value, ever.** This is the one
defect that would invalidate the whole run silently, because a leaked reference makes imputation look
brilliant. Enforced at build time (`_contributors`), stamped into the file's attrs, and re-checked on
load by `ReferenceTable.__init__`.

FILE LAYOUT (`*.reference.h5`)
    /sum      float32 [W, L, A]  depth-normalized counts summed over contributors
    /count    int32   [A]        number of contributors holding each assay
    /present  uint8   [n_bios, A]
    /scale    float32 [n_bios, A]  2**(depth_center - log2_depth), 0 where absent
  attrs: src_fingerprint, depth_center, assays, bios_order, contributor_prefix, n_contributors

CLI:
    python -m candi.reference build  --h5 <baked.h5> --out <baked.reference.h5>
    python -m candi.reference verify --h5 <baked.h5> --ref <baked.reference.h5>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

__all__ = [
    "REFERENCE_PSEUDOCOUNT",
    "ReferenceTable",
    "build_reference",
    "h5_fingerprint",
    "reference_path_for",
]

# Pseudocount added inside the log so an all-zero reference position is a finite floor rather than
# -inf. 0.25 of a depth-centered count: small enough that a real signal position is unaffected
# (log2(8.25) vs log2(8) = 0.04), large enough that the floor sits at log2(0.25) = -2 instead of the
# decoder's -15 clamp, which would demand an implausible +13 from `eta` to recover an ordinary bin.
REFERENCE_PSEUDOCOUNT = 0.25

CONTRIBUTOR_PREFIX = "T_"
_EXCLUDED_PREFIXES = ("V_", "B_")


def reference_path_for(h5_path) -> Path:
    """`/x/eic_full.h5` -> `/x/eic_full.reference.h5`. One convention, so no run can pair the wrong two."""
    p = Path(h5_path)
    return p.with_suffix("").with_suffix("") if p.suffixes[-2:] == [".reference", ".h5"] else \
        p.with_name(p.stem + ".reference.h5")


def h5_fingerprint(h5_path) -> str:
    """Cheap content fingerprint: file size, every top-level attr, and ALL of `meta_dsf1`.

    A sha1 of the 24 GB payload takes minutes and would be run on every train launch, so this hashes
    the parts that decide what the reference MEANS. The attrs carry the panel, the assay order and the
    chromosome split; `meta_dsf1` carries every biosample's per-assay depth, availability, read length
    and run type — 89 x 4 x 35 floats, read in milliseconds.

    Attrs alone are not enough, and that is not hypothetical: a re-bake that swapped one biosample's
    sequencing depth leaves every attr and the file size untouched while changing every reference
    value that biosample contributes to. `meta_dsf1` is exactly the array the builder reads to decide
    who contributes and at what scale, so hashing it makes the check cover its own inputs.
    """
    p = Path(h5_path)
    h = hashlib.sha1()
    h.update(str(p.stat().st_size).encode())
    with h5py.File(str(p), "r") as f:
        for k in sorted(f.attrs):
            v = f.attrs[k]
            h.update(k.encode())
            h.update((v.decode() if isinstance(v, bytes) else str(v)).encode())
        order = json.loads(f["biosamples"].attrs["order"])
        h.update(json.dumps(order).encode())
        for b in order:
            h.update(np.ascontiguousarray(f["biosamples"][b.replace("/", "_")]["meta_dsf1"][:],
                                          dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _contributors(bios_order: Sequence[str]) -> List[str]:
    """The `T_` biosamples, and nothing else. The exclusion is asserted, not assumed."""
    out = [b for b in bios_order if b.startswith(CONTRIBUTOR_PREFIX)]
    leaked = [b for b in out if b.startswith(_EXCLUDED_PREFIXES)]
    if leaked:                                          # unreachable unless the prefix scheme changes
        raise AssertionError(f"held-out biosamples reached the contributor set: {leaked}")
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_reference(h5_path, out_path, *, depth_center: Optional[float] = None,
                    window_chunk: int = 512, verbose: bool = True) -> Dict:
    """Stream the baked h5 once and write the sum/count reference. Returns a summary dict.

    Progress goes to stdout with flush; the bake convention of writing progress to stderr is not
    followed here because this runs as its own SLURM job and the log is read as one stream.
    """
    from candi.dataset import h5_depth_center

    h5_path, out_path = Path(h5_path), Path(out_path)
    if depth_center is None:
        depth_center = h5_depth_center(h5_path)
    depth_center = float(depth_center)
    fp = h5_fingerprint(h5_path)
    t0 = time.time()

    with h5py.File(str(h5_path), "r") as h:
        assays: List[str] = json.loads(h.attrs["assays"])
        bios_order: List[str] = json.loads(h["biosamples"].attrs["order"])
        A = len(assays)
        W = int(h["windows/chrom"].shape[0])
        L = int(h.attrs["context_bins"])
        contribs = _contributors(bios_order)

        present = np.zeros((len(bios_order), A), dtype=np.uint8)
        scale = np.zeros((len(bios_order), A), dtype=np.float32)
        for bi, b in enumerate(bios_order):
            d = np.asarray(h["biosamples"][b.replace("/", "_")]["meta_dsf1"][0], dtype=np.float64)
            ok = np.isfinite(d) & (d != -1.0)
            present[bi] = ok.astype(np.uint8)
            # 2**(depth_center - d) rescales this cell's counts to the common depth_center exposure,
            # which is the same unit `eta` already lives in under the depth offset.
            scale[bi, ok] = np.power(2.0, depth_center - d[ok]).astype(np.float32)

        c_idx = [bios_order.index(b) for b in contribs]
        count = present[c_idx].sum(axis=0).astype(np.int32)
        if verbose:
            print(f"[ref] {len(contribs)} contributors ({CONTRIBUTOR_PREFIX}*), "
                  f"{len(bios_order) - len(contribs)} excluded; W={W} L={L} A={A}; "
                  f"depth_center={depth_center:.4f}", flush=True)
            thin = [(assays[a], int(count[a])) for a in range(A) if count[a] <= 2]
            if thin:
                print(f"[ref] THIN assays (<=2 contributors) — leave-one-out matters most here: "
                      f"{thin}", flush=True)

        acc = np.zeros((W, L, A), dtype=np.float64)                    # ~1.6 GB
        for k, b in enumerate(contribs):
            g = h["biosamples"][b.replace("/", "_")]
            s = scale[bios_order.index(b)]                              # [A], 0 where absent
            for w0 in range(0, W, window_chunk):
                w1 = min(w0 + window_chunk, W)
                blk = np.asarray(g["counts_dsf1"][w0:w1], dtype=np.float64)   # [w, L, A]
                acc[w0:w1] += blk * s                                   # absent assays contribute 0
            if verbose:
                print(f"[ref] {k + 1}/{len(contribs)} {b}  ({time.time() - t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as o:
        # CONTIGUOUS and uncompressed, deliberately. Training reads `sum[wi]` for ~8 SCATTERED
        # windows every step; a chunked layout would force h5py to fault in whole chunks (a 64-window
        # chunk is 6.9 MB) to serve 107 KB of it. Contiguous makes each window a plain hyperslab read
        # of exactly the bytes needed, and 0.8 GB is cheap enough that compression is not worth the
        # per-step decompression.
        o.create_dataset("sum", data=acc.astype(np.float32), dtype="f4")
        o.create_dataset("count", data=count, dtype="i4")
        o.create_dataset("present", data=present, dtype="u1")
        o.create_dataset("scale", data=scale, dtype="f4")
        o.attrs["src_fingerprint"] = fp
        o.attrs["src_h5"] = str(h5_path)
        o.attrs["depth_center"] = depth_center
        o.attrs["assays"] = json.dumps(assays)
        o.attrs["bios_order"] = json.dumps(bios_order)
        o.attrs["contributor_prefix"] = CONTRIBUTOR_PREFIX
        o.attrs["contributors"] = json.dumps(contribs)
        o.attrs["n_contributors"] = len(contribs)
        o.attrs["pseudocount_default"] = REFERENCE_PSEUDOCOUNT
        o.attrs["version"] = 1

    nz = float((acc > 0).mean())
    summary = dict(out=str(out_path), n_contributors=len(contribs), W=W, L=L, A=A,
                   depth_center=depth_center, src_fingerprint=fp,
                   count_per_assay={assays[a]: int(count[a]) for a in range(A)},
                   n_assays_zero_contributors=int((count == 0).sum()),
                   frac_positions_nonzero=nz, wall_s=round(time.time() - t0, 1))
    if verbose:
        print(f"[ref] wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) in "
              f"{summary['wall_s']}s; {nz:.1%} of positions have nonzero reference", flush=True)
    return summary


# ---------------------------------------------------------------------------
# read side
# ---------------------------------------------------------------------------

class ReferenceTable:
    """Read-side handle. Turns (window indices, biosample) into `log_ref` [B, L, A].

    The h5 is opened lazily per process and kept open; `sum` is read as slabs, never loaded whole.
    """

    def __init__(self, ref_path, *, src_h5=None, pseudocount: float = REFERENCE_PSEUDOCOUNT):
        self.path = Path(ref_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"reference not found: {self.path} — build it with "
                f"`python -m candi.reference build --h5 <baked.h5>`")
        self.pseudocount = float(pseudocount)
        with h5py.File(str(self.path), "r") as f:
            self.assays: List[str] = json.loads(f.attrs["assays"])
            self.bios_order: List[str] = json.loads(f.attrs["bios_order"])
            self.contributors: List[str] = json.loads(f.attrs["contributors"])
            self.depth_center = float(f.attrs["depth_center"])
            self.src_fingerprint = str(f.attrs["src_fingerprint"])
            self.count = np.asarray(f["count"][:], dtype=np.int64)          # [A]
            self.present = np.asarray(f["present"][:], dtype=np.int64)      # [n_bios, A]
            self.scale = np.asarray(f["scale"][:], dtype=np.float32)        # [n_bios, A]
            self.shape = tuple(int(x) for x in f["sum"].shape)
        self._bios_idx = {b: i for i, b in enumerate(self.bios_order)}
        # L2 re-check on load. Cheap, and it fires on a reference built by an older/edited builder.
        bad = [b for b in self.contributors if b.startswith(_EXCLUDED_PREFIXES)]
        if bad:
            raise ValueError(f"{self.path} was built with held-out biosamples contributing: {bad}. "
                             "Rebuild — every imputation number computed against it is invalid.")
        if src_h5 is not None:
            got = h5_fingerprint(src_h5)
            if got != self.src_fingerprint:
                raise ValueError(
                    f"reference/h5 mismatch: {self.path} was built from a source with fingerprint "
                    f"{self.src_fingerprint}, but {src_h5} fingerprints {got}. The h5 was re-baked; "
                    "rebuild the reference.")
        self._f: Optional[h5py.File] = None

    # -- lazy handle (survives fork; each worker opens its own) -------------------------------
    @property
    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(str(self.path), "r")
        return self._f

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"] = None
        return d

    def contributes(self, biosample: str) -> bool:
        return biosample in self.contributors

    def cell_scale(self, biosample: str) -> np.ndarray:
        """Per-assay depth-normalization factor for this cell, 0 where the assay is absent. [A]"""
        return self.scale[self._bios_idx[biosample]]

    def log_ref(self, window_idx: Sequence[int], biosample: Optional[str] = None,
                counts_dsf1: Optional[np.ndarray] = None) -> np.ndarray:
        """`log2(R + pseudocount)` for the given windows. [B, L, A] float32.

        `biosample`/`counts_dsf1` are the LEAVE-ONE-OUT pair: when the biosample is a contributor, its
        own depth-normalized dsf1 counts for exactly these windows are removed from the sum and its
        presence from the denominator. Pass neither for a plain (non-contributor) target.

        Requesting LOO for a contributor WITHOUT its counts is refused rather than silently downgraded
        to the leaky full mean — that failure would be invisible in every downstream number.
        """
        wi = np.asarray(window_idx, dtype=np.int64)
        order = np.argsort(wi, kind="stable")                 # h5py fancy-index needs increasing order
        s = np.asarray(self._file["sum"][wi[order]], dtype=np.float64)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        s = s[inv]                                            # [B, L, A], back in caller order
        den = self.count.astype(np.float64).copy()            # [A]

        if biosample is not None and self.contributes(biosample):
            if counts_dsf1 is None:
                raise ValueError(
                    f"{biosample} contributes to the reference, so leave-one-out is required, but no "
                    "counts_dsf1 slab was supplied. Refusing to return the leaky full mean.")
            bi = self._bios_idx[biosample]
            s = s - np.asarray(counts_dsf1, dtype=np.float64) * self.scale[bi].astype(np.float64)
            den = den - self.present[bi].astype(np.float64)

        ok = den > 0
        R = np.zeros_like(s)
        R[:, :, ok] = s[:, :, ok] / den[ok]
        # Float cancellation in (sum - x_i) can leave a value a hair below zero on an all-zero column.
        np.clip(R, 0.0, None, out=R)
        return np.log2(R + self.pseudocount).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _verify(h5_path, ref_path, *, n_windows: int = 8, seed: int = 0) -> int:
    """Gates L1/L2/L3 re-run against the REAL artifacts, not a synthetic panel."""
    import random
    rt = ReferenceTable(ref_path, src_h5=h5_path)
    rc = 0
    print(f"[verify] L2 contributor set: {len(rt.contributors)} biosamples, all "
          f"{CONTRIBUTOR_PREFIX}*: OK ({len(rt.bios_order) - len(rt.contributors)} excluded)")

    rng = random.Random(seed)
    with h5py.File(str(h5_path), "r") as h:
        W = int(h["windows/chrom"].shape[0])
        wi = sorted(rng.sample(range(W), n_windows))
        # L1: for a contributor, LOO must exactly equal rebuilding the sum from the other cells.
        b = rt.contributors[0]
        g = h["biosamples"][b.replace("/", "_")]
        own = np.asarray(g["counts_dsf1"][wi], dtype=np.float64)
        lr_loo = rt.log_ref(wi, b, own)
        others = np.zeros_like(own)
        den = np.zeros(own.shape[-1])
        for ob in rt.contributors:
            if ob == b:
                continue
            og = h["biosamples"][ob.replace("/", "_")]
            others += np.asarray(og["counts_dsf1"][wi], dtype=np.float64) * rt.cell_scale(ob)
            den += rt.present[rt._bios_idx[ob]]
        R = np.zeros_like(others)
        R[:, :, den > 0] = others[:, :, den > 0] / den[den > 0]
        expect = np.log2(np.clip(R, 0, None) + rt.pseudocount).astype(np.float32)
        err = float(np.max(np.abs(expect - lr_loo)))
        ok = err < 1e-4
        rc |= 0 if ok else 1
        print(f"[verify] L1 leave-one-out exact on {b}: max|Δlog2| = {err:.3e} "
              f"({'OK' if ok else 'FAIL'})")

        # L3: the reference is depth-free — two cells at different depths for the same assay
        # contribute on a common scale, so `scale * counts` must be the depth-invariant quantity.
        d = np.asarray(g["meta_dsf1"][0], dtype=np.float64)
        live = np.where(rt.present[rt._bios_idx[b]] > 0)[0]
        if live.size:
            a = int(live[0])
            implied = rt.cell_scale(b)[a] * (2.0 ** d[a])
            ref_exposure = 2.0 ** rt.depth_center
            ok3 = abs(implied / ref_exposure - 1.0) < 1e-4
            rc |= 0 if ok3 else 1
            print(f"[verify] L3 depth-free: assay {rt.assays[a]} at log2 depth {d[a]:.3f} rescales to "
                  f"exposure 2^{np.log2(implied):.4f} vs depth_center 2^{rt.depth_center:.4f} "
                  f"({'OK' if ok3 else 'FAIL'})")

    zero = [rt.assays[a] for a in range(len(rt.assays)) if rt.count[a] == 0]
    thin = {rt.assays[a]: int(rt.count[a]) for a in range(len(rt.assays)) if 0 < rt.count[a] <= 2}
    print(f"[verify] contributors per assay: min={int(rt.count.min())} max={int(rt.count.max())}; "
          f"zero-contributor assays={zero or 'none'}; thin(<=2)={thin or 'none'}")
    rt.close()
    print(f"[verify] rc={rc}")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--h5", required=True)
    b.add_argument("--out", default=None, help="default: <h5 stem>.reference.h5 beside the h5")
    b.add_argument("--depth-center", type=float, default=None)
    b.add_argument("--window-chunk", type=int, default=512)
    b.add_argument("--summary-json", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--h5", required=True)
    v.add_argument("--ref", default=None)
    v.add_argument("--n-windows", type=int, default=8)
    a = ap.parse_args()

    if a.cmd == "build":
        out = a.out or reference_path_for(a.h5)
        s = build_reference(a.h5, out, depth_center=a.depth_center, window_chunk=a.window_chunk)
        if a.summary_json:
            Path(a.summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.summary_json).write_text(json.dumps(s, indent=2))
        return
    raise SystemExit(_verify(a.h5, a.ref or reference_path_for(a.h5), n_windows=a.n_windows))


if __name__ == "__main__":
    main()
