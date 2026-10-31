"""t112 C2 — `tools/t112/bin25.py`: store-rule counts of a tagAlign, mean-binned p of a bigwig.

Counts are checked against the store's literal per-read Python loop (`dnase_bam_rebuild.py`
docstring). The pval path is checked by a fake `pyBigWig` whose intervals are a synthetic bedGraph:
the result must equal `dnase_macs2_pval.bin_bedgraph` run on that bedGraph directly.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dnase_macs2_pval import bin_bedgraph                               # noqa: E402

_spec = importlib.util.spec_from_file_location("t112_bin25", ROOT / "tools" / "t112" / "bin25.py")
bin25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bin25)

RES = 25
# chr1 is an exact multiple of 25, chr2 has a remainder, chrX is tiny.
SIZES = {"chr1": 1000, "chr2": 1013, "chrX": 60}


def literal_counts(reads, sizes, res=RES):
    """The store's loop: grid `len // res + 1`, end bin inclusive, then truncate to `len // res`."""
    out = {}
    for c, n in sizes.items():
        bins = [0] * (n // res + 1)
        for chrom, s, e in reads:
            if chrom != c:
                continue
            for i in range(s // res, e // res + 1):
                bins[i] += 1
        out[c] = np.asarray(bins[: n // res], dtype=np.int64)
    return out


def write_chrsz(path, extra=()):
    lines = [f"{c}\t{n}" for c, n in SIZES.items()] + list(extra)
    path.write_text("\n".join(lines) + "\n")
    return path


def make_reads(seed=0, n=400):
    rng = np.random.default_rng(seed)
    reads = [
        ("chr1", 0, 25), ("chr1", 25, 50), ("chr1", 24, 25), ("chr1", 25, 26),   # bin edges
        ("chr1", 0, 101), ("chr1", 50, 76), ("chr1", 49, 50),                     # ends on bounds
        ("chr1", 924, 1000), ("chr1", 975, 1000), ("chr1", 999, 1000),            # chrom end, exact
        ("chr2", 912, 1013), ("chr2", 1000, 1013), ("chr2", 999, 1000),           # chrom end, rem
        ("chr2", 975, 1000), ("chrX", 0, 60), ("chrX", 50, 60), ("chrX", 25, 50),
    ]
    for _ in range(n):
        c = rng.choice(list(SIZES))
        ln = int(rng.integers(1, min(102, SIZES[c] + 1)))
        s = int(rng.integers(0, SIZES[c] - ln + 1))
        reads.append((str(c), s, s + ln))
    return reads


def write_tagalign(path, reads, junk=()):
    with gzip.open(path, "wt") as fo:
        for i, (c, s, e) in enumerate(list(reads) + list(junk)):
            fo.write(f"{c}\t{s}\t{e}\tN\t1000\t{'+-'[i % 2]}\n")
    return path


def test_load_chrsz_main_only_in_order(tmp_path):
    p = tmp_path / "sz.tsv"
    p.write_text("chrX\t60\nchrM\t16569\nchr2\t1013\nchrEBV\t171823\nchr1\t1000\n\n")
    sz = bin25.load_chrsz(p)
    assert list(sz) == ["chr1", "chr2", "chrX"]
    assert sz == SIZES


@pytest.mark.parametrize("chunksize", [7, 1_000_000])
def test_counts_match_literal_loop(tmp_path, chunksize):
    reads = make_reads()
    junk = [("chrM", 0, 100), ("chrEBV", 10, 50), ("chr1_KI270706v1_random", 0, 30)]
    ta = write_tagalign(tmp_path / "x.tagAlign.gz", reads, junk)
    stats = {}
    got = bin25.counts_from_tagalign(ta, SIZES, RES, chunksize=chunksize, stats=stats)
    want = literal_counts(reads, SIZES)
    assert list(got) == list(SIZES)
    for c in SIZES:
        assert got[c].dtype == np.uint32
        assert got[c].shape == (SIZES[c] // RES,)
        np.testing.assert_array_equal(got[c].astype(np.int64), want[c])
    assert stats == {"n_lines": len(reads) + len(junk), "n_lines_main": len(reads)}


def test_counts_end_on_boundary_credits_next_bin(tmp_path):
    ta = write_tagalign(tmp_path / "x.tagAlign.gz", [("chr1", 25, 50)])
    got = bin25.counts_from_tagalign(ta, SIZES, RES)
    assert got["chr1"][:4].tolist() == [0, 1, 1, 0]


def test_counts_overhang_past_chrom_end_is_dropped(tmp_path):
    # the store loop would index past its grid; dnase_bam_rebuild.slow_reference drops those bins
    ta = write_tagalign(tmp_path / "x.tagAlign.gz", [("chrX", 40, 90), ("chrX", 80, 120)])
    got = bin25.counts_from_tagalign(ta, SIZES, RES)
    assert got["chrX"].tolist() == [0, 1]


class _FakeBW:
    def __init__(self, table):
        self.table = table
        self.closed = False

    def chroms(self):
        return {c: 10 ** 6 for c in self.table}

    def intervals(self, chrom):
        return self.table[chrom] or None

    def close(self):
        self.closed = True


def make_bedgraph(seed=1):
    """Contiguous intervals covering each chrom, lengths 1..80, float32 values like a bigwig."""
    rng = np.random.default_rng(seed)
    table = {}
    for c, n in list(SIZES.items()) + [("chrM", 70), ("chrEBV", 40)]:
        ivs, s = [], 0
        while s < n:
            e = min(n, s + int(rng.integers(1, 81)))
            ivs.append((s, e, float(np.float32(rng.exponential(3.0)))))
            s = e
        table[c] = ivs
    table["chr21"] = []     # a chrom in the bigwig header with no data
    return table


def install_fake_pybigwig(monkeypatch, table, opened):
    mod = types.ModuleType("pyBigWig")

    def _open(path):
        bw = _FakeBW(table)
        opened.append((path, bw))
        return bw

    mod.open = _open
    monkeypatch.setitem(sys.modules, "pyBigWig", mod)


def test_pval_equals_bin_bedgraph(tmp_path, monkeypatch):
    table = make_bedgraph()
    bdg = tmp_path / "ref.bdg"
    with open(bdg, "w") as fo:
        for c, ivs in table.items():
            for s, e, v in ivs:
                fo.write(f"{c}\t{s}\t{e}\t{v!r}\n")
    want = bin_bedgraph(bdg, SIZES, RES)[0]

    opened = []
    install_fake_pybigwig(monkeypatch, table, opened)
    tmp = tmp_path / "tmp"
    stats = {}
    got = bin25.pval_from_bigwig(tmp_path / "x.pval.signal.bigwig", SIZES, tmp, RES, stats=stats)
    assert opened and opened[0][1].closed
    assert list(got) == list(SIZES)
    for c in SIZES:
        assert got[c].dtype == np.float32
        assert got[c].shape == (SIZES[c] // RES,)
        np.testing.assert_array_equal(got[c], want[c])
    assert stats["n_lines"] == sum(len(v) for v in table.values())
    assert stats["n_lines_main"] == sum(len(table[c]) for c in SIZES)
    assert list(tmp.iterdir()) == []                        # bedGraph removed after binning


def test_npz_roundtrip_keys_in_main_order(tmp_path):
    arrays = {"chrX": np.arange(2, dtype=np.uint32), "chr2": np.arange(3, dtype=np.uint32),
              "chr1": np.arange(4, dtype=np.uint32)}
    p = tmp_path / "a" / "counts25.npz"
    bin25.write_npz(p, arrays)
    back = bin25.read_npz(p)
    assert list(back) == ["chr1", "chr2", "chrX"]
    for c in arrays:
        np.testing.assert_array_equal(back[c], arrays[c])
        assert back[c].dtype == arrays[c].dtype


def test_cli_counts_and_pval(tmp_path, monkeypatch):
    sz = write_chrsz(tmp_path / "sz.tsv", extra=["chrM\t16569"])
    reads = make_reads(seed=3, n=50)
    ta = write_tagalign(tmp_path / "x.tagAlign.gz", reads, [("chrM", 0, 10)])
    out, js = tmp_path / "counts25.npz", tmp_path / "counts.json"
    assert bin25.main(["counts", "--ta", str(ta), "--chrsz", str(sz),
                       "--out", str(out), "--json", str(js)]) == 0
    rec = json.loads(js.read_text())
    want = literal_counts(reads, SIZES)
    back = bin25.read_npz(out)
    assert rec["n_lines"] == len(reads) + 1 and rec["n_lines_main"] == len(reads)
    for c in SIZES:
        np.testing.assert_array_equal(back[c].astype(np.int64), want[c])
        assert rec["per_chrom"][c] == {"n_bins": SIZES[c] // RES, "sum": int(want[c].sum()),
                                       "max": int(want[c].max())}

    table = make_bedgraph(seed=4)
    install_fake_pybigwig(monkeypatch, table, [])
    out, js = tmp_path / "pval25.npz", tmp_path / "pval.json"
    assert bin25.main(["pval", "--bigwig", str(tmp_path / "x.bw"), "--chrsz", str(sz),
                       "--tmpdir", str(tmp_path / "t"), "--out", str(out),
                       "--json", str(js)]) == 0
    rec = json.loads(js.read_text())
    back = bin25.read_npz(out)
    assert list(back) == list(SIZES)
    for c in SIZES:
        pc = rec["per_chrom"][c]
        assert pc["n_bins"] == SIZES[c] // RES and pc["n_nan"] == 0
        assert pc["max"] == pytest.approx(float(back[c].max()))
        assert pc["mean"] == pytest.approx(float(back[c].mean()))
