"""project_gtd — price the genome-wide `GenerateTrainData` stage from the tasks that finished.

    python project_gtd.py <rundir> [--budget-h 500]

The pilot memo's one soft number. `GenerateTrainData` was modelled as linear in **bins scanned** —
it streams the converted data to place its 100 000 sampled locations — and that assumption carries
59 % of the full-grid estimate. Genome-wide the stage runs one task per (mark, chromosome), so a
handful of finished tasks give seconds-per-bin directly, and the stage total is that times the whole
genome times the number of target marks.

Exit status is the point: **1 if the projection is over budget**, so it can gate a chain. The
default budget is 500 CPU-h — 3× the 166 CPU-h estimate, the abort threshold agreed for this run.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

RESOLUTION = 25


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="project_gtd.py")
    ap.add_argument("rundir")
    ap.add_argument("--budget-h", type=float, default=500.0)
    ap.add_argument("--estimate-h", type=float, default=166.0)
    args = ap.parse_args(argv)
    r = args.rundir

    n_bins = {}
    with open(os.path.join(r, "input", "chrominfo.txt"), encoding="utf-8") as fh:
        for line in fh:
            chrom, length = line.split()
            n_bins[chrom] = int(length) // RESOLUTION
    genome_bins = sum(n_bins.values())

    with open(os.path.join(r, "lists", "gtd.txt"), encoding="utf-8") as fh:
        items = [ln.rstrip("\n") for ln in fh if ln.strip()]
    n_marks = len({ln.split("\t")[0] for ln in items})

    secs = 0.0
    bins = 0
    done = 0
    for path in glob.glob(os.path.join(r, "timing", "gtd.*.tsv")):
        with open(path, encoding="utf-8") as fh:
            parts = fh.read().rstrip("\n").split("\t")
        # stage, mark, chrom, seconds, jobid  — the item itself carries a tab
        if len(parts) < 5:
            continue
        chrom, elapsed = parts[2], float(parts[3])
        if chrom not in n_bins:
            continue
        secs += elapsed
        bins += n_bins[chrom]
        done += 1

    if not done:
        print("project_gtd: no completed GenerateTrainData task has written a timing file yet")
        return 0

    per_bin = secs / bins
    projected_h = per_bin * genome_bins * n_marks / 3600.0
    over = projected_h > args.budget_h
    print(f"project_gtd: {done}/{len(items)} tasks — {secs:,.0f} s over {bins:,} bins")
    print(f"  {per_bin * 1e6:.2f} us/bin  ->  stage total {projected_h:,.0f} CPU-h "
          f"({n_marks} marks x {genome_bins:,} bins)")
    print(f"  memo estimate {args.estimate_h:,.0f} CPU-h · abort budget {args.budget_h:,.0f} CPU-h")
    print(f"  VERDICT: {'OVER BUDGET — pause the chain' if over else 'within budget'}")
    return 1 if over else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
