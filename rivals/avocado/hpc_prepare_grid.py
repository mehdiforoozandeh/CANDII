#!/usr/bin/env python3
"""Write the canonical genomic grid every later stage shares.  Runs on Fir.

Taken from a challenge bigWig header rather than from hg38.chrom.sizes, so that
the bin count is by construction the one 001's scoring code will use:
n_bins = (chrom_len - 1) // 25 + 1, i.e. ceil(chrom_len / 25).

Emits, into <out>:
  chroms.txt         the 23 scored chromosomes, in 001's sorted order
  chrom_lengths.json chrom -> length in bp
  chrom_sizes.json   chrom -> n_bins on the official 25 bp grid
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "inputs", "vendor_001"))
import eic_metrics as EM     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigwig", required=True, help="any challenge bigWig")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyBigWig
    bw = pyBigWig.open(args.bigwig)
    lengths = {c: int(bw.chroms()[c]) for c in EM.CHROMS}
    bw.close()
    sizes = {c: (l - 1) // EM.WINDOW + 1 for c, l in lengths.items()}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "chroms.txt"), "w") as fh:
        fh.write("\n".join(EM.CHROMS) + "\n")
    json.dump(lengths, open(os.path.join(args.out, "chrom_lengths.json"), "w"), indent=1)
    json.dump(sizes, open(os.path.join(args.out, "chrom_sizes.json"), "w"), indent=1)
    print(f"[grid] {len(EM.CHROMS)} chromosomes, {sum(sizes.values())} bins total "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
