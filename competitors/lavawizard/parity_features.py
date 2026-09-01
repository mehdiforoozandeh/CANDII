"""Prove `dataset3` + `features` reproduce their preprocessing, on the real challenge bigwigs.

`parity_keras.py` settles the *model*: our torch module against Keras on their weights.
This settles the *inputs*: our binning and our cross-cell moments against `00_data_generation.py`,
run on the same Dataset-3 files. Both halves matter, because a perfect model fed a subtly
different average is still a different method — and the average is added straight to the output,
so an error in it passes through undamped.

Their three functions are **vendored verbatim** below from upstream commit `d638b204` (md5s in
`README.md`), reindented only where the original sat inside a script. They are the reference; our
code is the thing under test. Do not tidy them.

```bash
cd ~/scratch/t53_lavawizard
MAMBA_ROOT_PREFIX=$PWD/mamba ./bin/micromamba run -n lw_period python \
    port/lavawizard/parity_features.py \
    --data /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/training_data \
    --meta repo/data/Encode_meta.tsv --chrom chr21 --mark M17
```

Both halves run in the same environment on purpose: the question is whether two pieces of code
agree on one input, not whether two runtimes do.

**What the three reported numbers mean, and which one is the bug detector.**

Measured on chr21 against `training_data`, three marks with 4 / 16 / 22 contributors:

| quantity | worst absolute disagreement | reading |
|---|---|---|
| single track, per bin | **0.000e+00**, every mark | `bin_arcsinh` is bit-identical to `get_binvals` |
| average | 4.8e-07 … 9.5e-07 | float32 vs float64 accumulation, nothing else |
| variance | 3.1e-06 … 1.06e-05 | *their* float32 cancellation, growing with signal and contributor count |

The **single-track row is the bug detector**: any error in the binning — the ceil grid, the
zero-pad, `arcsinh` before rather than after the mean — moves it off zero immediately. It is
exactly zero, so the transcription is right.

The variance row is not our error, it is theirs. `E[x^2] - E[x]^2` in float32 loses precision by
cancellation, and it loses more of it the larger the signal and the more contributors are summed;
at H3K27ac's peak variance of ~6 the float32 granularity of `E[x^2]` is already ~4e-6 before any
cancellation. We accumulate in float64 and are the more accurate of the two. Reproducing their
rounding exactly would mean deliberately accumulating in float32, and is only worth doing if a
parity check against their released weights ever demands it.

So the gate is **absolute**, not relative, and set at **1e-4**. Absolute because the variance
enters the model as a raw scalar through `concat_last`, so absolute error is what propagates; a
relative gate is meaningless in the near-zero-variance bins, where M01 shows 1.7e-3 relative on an
absolute error of 9e-6. 1e-4 sits an order of magnitude above their float32 noise and three orders
below anything a transcription bug would produce.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["upstream_get_binvals", "upstream_avg_var", "main"]


# ===================== VERBATIM from 00_data_generation.py (d638b204) =====================
def upstream_get_binvals(filepath, chrom):
    """Get 25 bin values of given chrom. Take arcsinh first then bin (mean)."""
    import pyBigWig

    bw = pyBigWig.open(filepath)

    end = bw.chroms(chrom)
    nbins = (-(-end // 25))  ## round-up

    vals = bw.values(chrom, 0, end)
    vals = np.array(vals)
    vals[np.isnan(vals)] = 0

    # calc arcsinh
    vals = np.arcsinh(vals)

    # pad 0 in order to match nbin*25
    n_extra = nbins * 25 - end
    vals = np.append(vals, np.zeros(n_extra))

    # calc mean
    vals = vals.reshape(nbins, -1).mean(axis=1)
    vals = vals.astype('float32')

    bw.close()

    return (vals)


def upstream_avg_var(tracks):
    """`get_avg` / `get_var` bodies, for the contributor set already selected.

    Upstream selects `indices[1,:] == a_id` and then sums; here the caller has done the selection,
    so this is that loop with the same float32 accumulation and the same `E[x^2] - E[x]^2`.
    """
    track_avg = np.zeros(len(tracks[0]), dtype=np.float32)
    track_var = np.zeros(len(tracks[0]), dtype=np.float32)
    n_other_cells = 0
    for t in tracks:
        track_avg += t
        track_var += t ** 2
        n_other_cells += 1
    avg = track_avg / n_other_cells
    var = track_var / n_other_cells - (track_avg / n_other_cells) ** 2
    return avg, var
# =========================================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True, help="challenge training_data directory")
    p.add_argument("--meta", type=Path, required=True, help="Encode_meta.tsv")
    p.add_argument("--chrom", default="chr21")
    p.add_argument("--mark", default="M17")
    p.add_argument("--max-contributors", type=int, default=0, help="0 = all")
    p.add_argument("--max-abs", type=float, default=None,
                   help="absolute gate; 1e-4 is the calibrated value — see the module docstring")
    ns = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lavawizard import dataset3, features

    meta = dataset3.read_meta(ns.meta)
    pool = dataset3.contributor_pool(meta, ns.mark, splits=("T",))
    pool = [c for c in pool if dataset3.track_path(ns.data, c, ns.mark).exists()]
    if ns.max_contributors:
        pool = pool[:ns.max_contributors]
    if len(pool) < 2:
        print(f"only {len(pool)} contributor(s) for {ns.mark} in {ns.data} — pick another mark")
        return 2
    print(f"{ns.chrom} / {ns.mark}: {len(pool)} contributors  {pool[:8]}"
          f"{' …' if len(pool) > 8 else ''}")

    # --- reference: their code -------------------------------------------------------------
    t0 = time.time()
    ref_tracks = [upstream_get_binvals(str(dataset3.track_path(ns.data, c, ns.mark)), ns.chrom)
                  for c in pool]
    ref_avg, ref_var = upstream_avg_var(ref_tracks)
    t_ref = time.time() - t0
    print(f"upstream: {len(ref_avg)} bins in {t_ref:.1f} s")

    # --- under test: ours ------------------------------------------------------------------
    t0 = time.time()
    reader = dataset3.Dataset3Reader(ns.data, ns.chrom, max_cached=len(pool) + 1)
    got_avg, got_var, k = features.cross_cell_moments(
        reader.read, pool, ns.chrom, 0, reader.n_bins, assay=ns.mark, transform="none")
    t_ours = time.time() - t0
    print(f"ours:     {len(got_avg)} bins in {t_ours:.1f} s  (k={k})")

    if len(got_avg) != len(ref_avg):
        print(f"GRID MISMATCH: ours {len(got_avg)} vs theirs {len(ref_avg)} — stop here")
        return 1

    worst = 0.0
    for name, ref, got in (("average", ref_avg, got_avg), ("variance", ref_var, got_var)):
        abs_err = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-9)
        print(f"  {name:<9} theirs[mean {ref.mean():.6f} max {ref.max():.6f}]  "
              f"abs err max {abs_err.max():.3e}  mean {abs_err.mean():.3e}  "
              f"rel max {(abs_err / denom).max():.3e}")
        worst = max(worst, float(abs_err.max()))

    # The bug detector. A single track has no cross-cell accumulation, so float32 cancellation
    # cannot hide behind it: any binning error moves this off zero at once.
    one = reader.read(pool[0], ns.mark, 0, reader.n_bins)
    solo = np.abs(one.astype(np.float64) - ref_tracks[0].astype(np.float64)).max()
    print(f"  single track {pool[0]}{ns.mark}: abs err max {solo:.3e}"
          f"   {'<- bit-identical' if solo == 0.0 else '<- NOT bit-identical, investigate'}")
    if solo != 0.0:
        print("  the binning must be bit-identical; a non-zero here is a transcription bug, "
              "not a precision difference. Do not raise the gate to cover it.")
    worst = max(worst, float(solo))

    print(f"\nworst absolute disagreement: {worst:.3e}")
    if ns.max_abs is None:
        print("no --max-abs given: reporting only.")
        return 0
    ok = worst <= ns.max_abs
    print(f"GATE worst <= {ns.max_abs:g}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
