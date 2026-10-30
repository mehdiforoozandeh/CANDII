"""Regenerate the tiny t138 build fixtures (deterministic). Run from anywhere:

    python tests/fixtures/t138/make_fixtures.py

chr1 30 500 bp (30 x 1 kb bins), chr2 20 700 bp (20 bins), plus chrUn_x and chrM that the bins1k
layout does not carry. Bin i of chr1/chr2 repeats a 10-mer with k = i % 11 G/C letters, so its GC
is k / 10 (odd bins lower-case). chr1 bin 3 is all N (NaN), bin 4 has 400 N (GC 0.4), bin 5 has
600 N (NaN), bin 6 has exactly 500 N (GC 0.6).
"""
import gzip
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIZES = {"chr1": 30_500, "chr2": 20_700, "chrUn_x": 500, "chrM": 100}


def unit(k):
    return ("GC" * 5)[:k] + ("AT" * 5)[: 10 - k]


def bin_seq(c, i):
    k = i % 11
    s = unit(k) * 100
    if c == "chr1" and i == 3:
        s = "N" * 1000
    if c == "chr1" and i == 4:
        s = "N" * 400 + unit(k) * 60
    if c == "chr1" and i == 5:
        s = "N" * 600 + unit(k) * 40
    if c == "chr1" and i == 6:
        s = "N" * 500 + unit(k) * 50
    return s.lower() if i % 2 else s


def chrom_seq(c):
    n = SIZES[c] // 1000 if c in ("chr1", "chr2") else 0
    s = "".join(bin_seq(c, i) for i in range(n))
    return s + ("ACGT" * 1000)[: SIZES[c] - len(s)]


def gz(path, text):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(text.encode())


def main():
    ann = HERE / "annot"
    ann.mkdir(exist_ok=True)
    fa = []
    for c in SIZES:
        s = chrom_seq(c)
        fa.append(f">{c} synthetic")
        fa += [s[i:i + 60] for i in range(0, len(s), 60)]
    gz(ann / "hg38.fa.gz", "\n".join(fa) + "\n")
    (ann / "hg38.chrom.sizes").write_text("".join(f"{c}\t{n}\n" for c, n in SIZES.items()))
    rm = [("chr1", 2000, 2500), ("chr1", 2300, 2800), ("chr1", 5500, 7250), ("chr2", 0, 100),
          ("chrUn_x", 0, 100)]
    gz(ann / "rmsk.txt.gz", "".join(
        f"585\t100\t0\t0\t0\t{c}\t{s}\t{e}\t-1\t+\tL1\tLINE\tL1\t1\t2\t3\t{i}\n"
        for i, (c, s, e) in enumerate(rm)))
    gz(ann / "k36.umap.bedgraph.gz", "track type=bedGraph name=k36\n" + "".join(
        f"{c}\t{s}\t{e}\t{v}\n" for c, s, e, v in [
            ("chr1", 0, 1000, 1.0), ("chr1", 1000, 1500, 0.5), ("chr1", 10000, 10250, 0.8),
            ("chr2", 19000, 20700, 0.6)]))

    npd = HERE / "narrowpeak"
    for t, name, rows in [
            ("C19M16", "call_peak_shard0_x.bfilt.narrowPeak.gz",
             [("chr2", 5200, 5300), ("chr1", 26000, 26100)]),
            ("C12M02", "call_peak_pr1_x.no_chrM_MT.bfilt.narrowPeak.gz", [("chr1", 1999, 2001)]),
            ("C12M02", "call_peak_pr2_x.no_chrM_MT.bfilt.narrowPeak.gz", [("chr2", 0, 20000)])]:
        (npd / t).mkdir(parents=True, exist_ok=True)
        gz(npd / t / name, "".join(f"{c}\t{s}\t{e}\tp\t1000\t.\t5\t5\t5\t50\n" for c, s, e in rows))

    b1 = HERE / "bins1k"
    b1.mkdir(exist_ok=True)
    for t, plateau in [("C19M16", True), ("C12M02", False)]:
        arr = {}
        for c in ("chr1", "chr2"):
            n = SIZES[c] // 1000
            x = np.full(n, 5, dtype=np.uint32)
            if plateau and c == "chr1":
                x[22:] = 200
            for k in ("a15", "b15", "a375", "b375"):
                arr[f"{k}_{c}"] = x.copy()
        np.savez_compressed(b1 / f"{t}__base__base.npz", **arr)


if __name__ == "__main__":
    main()
