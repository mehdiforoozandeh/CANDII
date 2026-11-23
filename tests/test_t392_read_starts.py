"""t392 chunk C1, case (1): the stranded read-start recount and its read-level gate.

Written against `cruxvault/INSTRUCTIONS/t392_read_starts/INSTRUCTIONS.md`, decision D9 (the gate
is exact at read level) and the Interfaces lines of plan chunk C3, before
`tools/onion_data/read_starts.py` existed. The reference for the overlap rule is
`tools/t112/bin25.py::counts_from_tagalign`; the 5' base rule is `recount.point_positions`.

Every module under test is imported inside a test or a fixture, so collection passes without it.
"""
from __future__ import annotations

import gzip
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tools" / "t112"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RES = 25
REACH = 8
#: bp lengths; chr1 ends in a partial bin of 10 bp (1010 // 25 = 40 bins kept, bin 40 dropped)
SIZES = {"chr1": 1010, "chr2": 500}


# --------------------------------------------------------------------------- helpers

def _rs():
    from onion_data import read_starts
    return read_starts


def _write_tagalign(path: Path, reads) -> Path:
    """reads: (chrom, start, end, strand); a gzipped 6-column tagAlign as bedtools writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for chrom, s, e, strand in reads:
            fh.write(f"{chrom}\t{s}\t{e}\tN\t1000\t{strand}\n")
    return path


def _per_chrom(res: dict, key: str, chrom: str) -> np.ndarray:
    """`scan` returns per-chromosome arrays; accept res[key][chrom] or res[chrom][key]."""
    if key in res and isinstance(res[key], dict):
        return np.asarray(res[key][chrom])
    return np.asarray(res[chrom][key])


def _cli(module, argv) -> int:
    try:
        rc = module.main(argv)
    except SystemExit as e:
        rc = e.code
    return int(rc or 0)


def _expected_starts(reads, sizes):
    """Independent rule: 5' base is start on +, end - 1 on -; bin = base // 25; grid len // 25."""
    out = {c: np.zeros((sizes[c] // RES, 2), np.int64) for c in sizes}
    off = 0
    for chrom, s, e, strand in reads:
        if chrom not in sizes:
            continue
        base = s if strand == "+" else e - 1
        b = base // RES
        if b < sizes[chrom] // RES:
            out[chrom][b, 0 if strand == "+" else 1] += 1
        else:
            off += 1
    return out, off


# Reads on chr1 with known placements (bins of 25 bp):
#   + 10..60    start bin 0, overlap bins 0..2   (offsets 0, 1, 2)
#   + 30..55    start bin 1, overlap bins 1..2   (offsets 0, 1)
#   - 30..80    5' base 79 -> bin 3, overlap bins 1..3 (offsets -2..0)
#   - 30..75    5' base 74 -> bin 2, overlap bins 1..3 (75 // 25 = 3: offsets -1..+1)
#   - 100..140  5' base 139 -> bin 5, overlap bins 4..5
#   + 600..650  start bin 24, overlap bins 24..26
#   - 990..1005 5' base 1004 -> bin 40, the partial bin: OFF the grid; overlap bins 39..40
#   + 1002..1009 start bin 40: OFF the grid
# chr2: three reads; chrUn: one read on a non-main chromosome.
READS = [
    ("chr1", 10, 60, "+"), ("chr1", 30, 55, "+"), ("chr1", 30, 80, "-"),
    ("chr1", 30, 75, "-"), ("chr1", 100, 140, "-"), ("chr1", 600, 650, "+"),
    ("chr1", 990, 1005, "-"), ("chr1", 1002, 1009, "+"),
    ("chr2", 0, 36, "+"), ("chr2", 200, 250, "-"), ("chr2", 460, 500, "-"),
    ("chrUn_KI270302v1", 5, 40, "+"),
]


@pytest.fixture
def ta(tmp_path):
    return _write_tagalign(tmp_path / "reads.tagAlign.gz", READS)


@pytest.fixture
def scanned(ta):
    return _rs().scan([str(ta)], SIZES, reach=REACH)


# --------------------------------------------------------------------------- the read-start rule

def test_plus_and_minus_start_counts_land_in_the_hand_computed_bins(scanned):
    exp, _ = _expected_starts(READS, SIZES)
    for c in SIZES:
        got = _per_chrom(scanned, "starts", c)
        assert got.shape == (SIZES[c] // RES, 2)
        np.testing.assert_array_equal(got.astype(np.int64), exp[c], err_msg=c)
    s1 = _per_chrom(scanned, "starts", "chr1")
    assert s1[0, 0] == 1 and s1[1, 0] == 1 and s1[24, 0] == 1         # plus at the start bin
    assert s1[3, 1] == 1 and s1[2, 1] == 1 and s1[5, 1] == 1          # minus at bin (end-1)//25


def test_a_minus_read_counts_at_the_bin_of_end_minus_one(tmp_path):
    # end = 75 is a bin boundary: end // 25 = 3 but the 5' base 74 is in bin 2
    p = _write_tagalign(tmp_path / "m.tagAlign.gz", [("chr2", 30, 75, "-")])
    r = _rs().scan([str(p)], {"chr2": 500}, reach=REACH)
    s = _per_chrom(r, "starts", "chr2")
    assert s[:, 1].sum() == 1 and s[2, 1] == 1 and s[3, 1] == 0
    assert s[:, 0].sum() == 0


def test_start_bin_agrees_with_recount_point_positions(scanned):
    from onion_data.recount import point_positions
    rows = [r for r in READS if r[0] in SIZES]
    pos = point_positions([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], 0)
    exp = {c: np.zeros((SIZES[c] // RES, 2), np.int64) for c in SIZES}
    for (c, _, _, strand), p in zip(rows, pos):
        b = int(p) // RES
        if b < SIZES[c] // RES:
            exp[c][b, 0 if strand == "+" else 1] += 1
    for c in SIZES:
        np.testing.assert_array_equal(_per_chrom(scanned, "starts", c).astype(np.int64), exp[c])


def test_start_total_equals_reads_kept_on_the_grid(scanned):
    total = sum(int(_per_chrom(scanned, "starts", c).sum()) for c in SIZES)
    n_main = sum(1 for r in READS if r[0] in SIZES)
    assert int(scanned["n_off_grid"]) == 2
    assert total == n_main - 2
    nrs = np.asarray(scanned["n_reads_strand"]).astype(np.int64)
    assert nrs.shape == (2,)
    plus = sum(int(_per_chrom(scanned, "starts", c)[:, 0].sum()) for c in SIZES)
    minus = sum(int(_per_chrom(scanned, "starts", c)[:, 1].sum()) for c in SIZES)
    assert (int(nrs[0]), int(nrs[1])) == (plus, minus)
    assert plus + minus == total


def test_line_counters_separate_main_and_other_chromosomes(scanned):
    assert int(scanned["n_lines"]) == len(READS)
    assert int(scanned["n_main"]) == len(READS) - 1


# --------------------------------------------------------------------------- the overlap rule

def test_overlap_of_the_same_scan_equals_bin25_bin_for_bin(ta, scanned):
    import bin25
    ref = bin25.counts_from_tagalign(str(ta), SIZES)
    for c in SIZES:
        got = _per_chrom(scanned, "overlap", c)
        assert got.shape == ref[c].shape
        np.testing.assert_array_equal(got.astype(np.int64), ref[c].astype(np.int64), err_msg=c)


def test_off_grid_minus_read_still_adds_its_overlap_to_kept_bins(tmp_path):
    import bin25
    # 5' base 1004 is in bin 40 (dropped); the span covers bins 39 (kept) and 40 (dropped)
    p = _write_tagalign(tmp_path / "edge.tagAlign.gz", [("chr1", 990, 1005, "-")])
    r = _rs().scan([str(p)], {"chr1": 1010}, reach=REACH)
    assert int(r["n_off_grid"]) == 1
    assert int(_per_chrom(r, "starts", "chr1").sum()) == 0
    ov = _per_chrom(r, "overlap", "chr1").astype(np.int64)
    assert ov.shape == (40,)
    assert ov[39] == 1 and ov.sum() == 1
    np.testing.assert_array_equal(ov, bin25.counts_from_tagalign(str(p), {"chr1": 1010})["chr1"])
    assert np.asarray(r["n_reads_strand"]).sum() == 0
    assert int(np.asarray(r["kernel_counts"]).sum()) == 0      # only reads kept on the grid


def test_overlap_matches_bin25_on_random_reads(tmp_path):
    import bin25
    rng = np.random.default_rng(3)
    sizes = {"chr1": 5013, "chr2": 2500}
    reads = []
    for c, n in sizes.items():
        for _ in range(400):
            ln = int(rng.integers(20, 151))
            s = int(rng.integers(0, n - 15))
            e = min(s + ln, n)
            reads.append((c, s, e, "+" if rng.random() < 0.5 else "-"))
    p = _write_tagalign(tmp_path / "rand.tagAlign.gz", reads)
    r = _rs().scan([str(p)], sizes, reach=REACH)
    ref = bin25.counts_from_tagalign(str(p), sizes)
    exp, off = _expected_starts(reads, sizes)
    assert int(r["n_off_grid"]) == off
    for c in sizes:
        np.testing.assert_array_equal(_per_chrom(r, "overlap", c).astype(np.int64),
                                      ref[c].astype(np.int64))
        np.testing.assert_array_equal(_per_chrom(r, "starts", c).astype(np.int64), exp[c])


def test_several_read_files_add_up(tmp_path):
    a = _write_tagalign(tmp_path / "a.tagAlign.gz", READS[:5])
    b = _write_tagalign(tmp_path / "b.tagAlign.gz", READS[5:])
    rs = _rs()
    ra = rs.scan([str(a)], SIZES, reach=REACH)
    rb = rs.scan([str(b)], SIZES, reach=REACH)
    rab = rs.scan([str(a), str(b)], SIZES, reach=REACH)
    for key in ("starts", "overlap"):
        for c in SIZES:
            np.testing.assert_array_equal(
                _per_chrom(rab, key, c).astype(np.int64),
                _per_chrom(ra, key, c).astype(np.int64) + _per_chrom(rb, key, c).astype(np.int64))
    np.testing.assert_array_equal(np.asarray(rab["kernel_counts"]),
                                  np.asarray(ra["kernel_counts"]) + np.asarray(rb["kernel_counts"]))


# --------------------------------------------------------------------------- the span kernel

def test_kernel_counts_are_the_covered_offsets_per_strand(tmp_path):
    reads = [("chr2", 10, 60, "+"),     # start bin 0, covers offsets 0, 1, 2
             ("chr2", 30, 80, "-"),     # start bin 3, covers bins 1..3: offsets -2, -1, 0
             ("chr2", 30, 75, "-"),     # start bin 2, covers bins 1..3: offsets -1, 0, +1
             ("chr2", 460, 500, "-")]   # 5' base 499 -> bin 19 (last kept bin of 20); covers
                                        # bins 18, 19, 20; bin 20 is dropped: offsets -1, 0 only
    p = _write_tagalign(tmp_path / "k.tagAlign.gz", reads)
    r = _rs().scan([str(p)], {"chr2": 500}, reach=REACH)
    k = np.asarray(r["kernel_counts"]).astype(np.int64)
    assert k.shape == (2, 2 * REACH + 1)
    exp = np.zeros((2, 17), np.int64)
    exp[0, REACH + 0] += 1
    exp[0, REACH + 1] += 1
    exp[0, REACH + 2] += 1
    for off in (-2, -1, 0):
        exp[1, REACH + off] += 1
    for off in (-1, 0, 1):
        exp[1, REACH + off] += 1
    for off in (-1, 0):
        exp[1, REACH + off] += 1
    np.testing.assert_array_equal(k, exp)
    assert [int(x) for x in np.asarray(r["n_reads_strand"])] == [1, 3]


def test_kernel_times_reads_rebuilds_the_overlap_sum_of_kept_reads(tmp_path):
    rng = np.random.default_rng(11)
    n = 4000
    reads = []
    for _ in range(300):
        s = int(rng.integers(0, n - 200))
        reads.append(("chr1", s, s + int(rng.integers(20, 120)), "+" if rng.random() < .5 else "-"))
    p = _write_tagalign(tmp_path / "s.tagAlign.gz", reads)
    r = _rs().scan([str(p)], {"chr1": n}, reach=REACH)
    assert int(r["n_off_grid"]) == 0
    assert int(np.asarray(r["kernel_counts"]).sum()) == int(_per_chrom(r, "overlap", "chr1").sum())


def test_a_read_whose_span_leaves_the_reach_raises(tmp_path):
    p = _write_tagalign(tmp_path / "long.tagAlign.gz", [("chr1", 0, 400, "+")])   # 17 bins
    rs = _rs()                                  # outside the raises: a missing module must fail
    with pytest.raises(ValueError, match="reach"):
        rs.scan([str(p)], {"chr1": 1010}, reach=REACH)


# --------------------------------------------------------------------------- the gate

def _write_reference(path: Path, arrays: dict) -> Path:
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint32))
    return path


def _overlap_dict(r):
    return {c: _per_chrom(r, "overlap", c) for c in SIZES}


def test_gate_passes_with_zero_differing_bins_on_the_exact_reference(ta, scanned, tmp_path):
    import bin25
    ref = _write_reference(tmp_path / "counts.h5", bin25.counts_from_tagalign(str(ta), SIZES))
    g = _rs().gate(_overlap_dict(scanned), str(ref))
    for key in ("reference", "reference_sha256", "n_bins", "n_differing", "result"):
        assert key in g, key
    assert int(g["n_differing"]) == 0
    assert int(g["n_bins"]) == sum(SIZES[c] // RES for c in SIZES)
    assert g["reference_sha256"] == hashlib.sha256(ref.read_bytes()).hexdigest()


def test_one_changed_reference_bin_gives_one_differing_bin_and_a_failed_result(ta, scanned,
                                                                               tmp_path):
    import bin25
    good = bin25.counts_from_tagalign(str(ta), SIZES)
    bad = {c: v.copy() for c, v in good.items()}
    bad["chr2"][7] += 1
    rs = _rs()
    g_ok = rs.gate(_overlap_dict(scanned), str(_write_reference(tmp_path / "ok.h5", good)))
    g_bad = rs.gate(_overlap_dict(scanned), str(_write_reference(tmp_path / "bad.h5", bad)))
    assert int(g_bad["n_differing"]) == 1
    assert g_bad["result"] != g_ok["result"]


def test_gate_counts_a_difference_on_the_last_kept_bin(ta, scanned, tmp_path):
    import bin25
    bad = {c: v.copy() for c, v in bin25.counts_from_tagalign(str(ta), SIZES).items()}
    bad["chr1"][-1] += 3                       # bin 39, fed only by the off-grid minus read
    g = _rs().gate(_overlap_dict(scanned), str(_write_reference(tmp_path / "bad.h5", bad)))
    assert int(g["n_differing"]) == 1


# --------------------------------------------------------------------------- the conversion floor

def test_conversion_floor_is_zero_when_every_read_has_the_same_span(tmp_path):
    # every plus read starts on a bin boundary and spans 25 bp: bins b and b + 1 (end inclusive)
    rng = np.random.default_rng(5)
    reads = [("chr1", 25 * int(b), 25 * int(b) + 25, "+") for b in rng.integers(4, 150, 200)]
    p = _write_tagalign(tmp_path / "f.tagAlign.gz", reads)
    rs = _rs()
    r = rs.scan([str(p)], {"chr1": 4000}, reach=REACH)
    kc = np.asarray(r["kernel_counts"], np.float64)
    nrs = np.asarray(r["n_reads_strand"], np.float64)
    kernel = np.divide(kc, nrs[:, None], out=np.zeros_like(kc), where=nrs[:, None] > 0)
    starts = {"chr1": _per_chrom(r, "starts", "chr1")}
    overlap = {"chr1": _per_chrom(r, "overlap", "chr1")}
    assert rs.conversion_floor(starts, kernel, overlap) == pytest.approx(0.0, abs=1e-12)


def test_conversion_floor_is_the_nmse_of_kernel_convolved_starts(tmp_path):
    rng = np.random.default_rng(9)
    reads = []
    for _ in range(500):
        s = int(rng.integers(400, 3400))      # away from both ends: no padding question
        reads.append(("chr1", s, s + int(rng.integers(20, 150)), "+" if rng.random() < .5 else "-"))
    p = _write_tagalign(tmp_path / "g.tagAlign.gz", reads)
    rs = _rs()
    r = rs.scan([str(p)], {"chr1": 4000}, reach=REACH)
    kc = np.asarray(r["kernel_counts"], np.float64)
    nrs = np.asarray(r["n_reads_strand"], np.float64)
    kernel = kc / nrs[:, None]
    st = _per_chrom(r, "starts", "chr1").astype(np.float64)
    b = _per_chrom(r, "overlap", "chr1").astype(np.float64)
    n = st.shape[0]
    a = np.zeros(n)
    for s in range(2):
        for k in range(-REACH, REACH + 1):
            w = kernel[s, REACH + k]
            if k >= 0:
                a[k:] += w * st[:n - k, s]
            else:
                a[:n + k] += w * st[-k:, s]
    want = np.mean((a - b) ** 2) / np.var(b)
    got = rs.conversion_floor({"chr1": st}, kernel, {"chr1": b})
    assert want > 0
    assert got == pytest.approx(want, rel=1e-3)
