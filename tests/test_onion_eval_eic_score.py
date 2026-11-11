"""Tests for tools/onion_eval/eic_score.py: the nine EIC measures on hand-computed arrays, the
fast scorer against the reference, the challenge's quirks, the bigWig reader and the ranking."""
from __future__ import annotations

import gzip
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import eic_score as ES  # noqa: E402

# truth and prediction of one 4-bin chromosome, and annotations built to exercise the quirks
T4 = {"chrA": np.array([0, 1, 2, 3], np.float32)}
P4 = {"chrA": np.array([0, 0, 2, 4], np.float32)}
PROM = {"chrA": np.array([[0, 2]], np.int64)}
BODY = {"chrA": np.array([[1, 2], [1, 3]], np.int64)}       # overlapping: bin 1 counts twice
ENH = {"chrA": np.array([[3, 10]], np.int64)}               # runs past the end: clipped to 4
VAR4 = {"chrA": np.array([1, 0, 0, 3], np.float64)}


def _hand():
    # d = t - p = [0, 1, 0, -1]
    return {"mse": 0.5,
            "gwcorr": 7 / math.sqrt(5 * 11),
            "gwspear": 4.5 / math.sqrt(5 * 4.5),            # pred ranks [1.5, 1.5, 3, 4]
            "mseprom": 0.5,                                 # [0, 2): sse 1 over 2 bins
            "msegene": 2 / 3,                               # [1,2) + [1,3): sse 2 over 3 bins
            "mseenh": 1.0,                                  # [3, 4)
            "msevar": 0.75,                                 # (3 * 1) / (1 + 3)
            "mse1obs": 0.5,                                 # N < 100: n1 = 0 selects every bin
            "mse1imp": 0.5}


def test_nine_measures_reference_hand_computed():
    got = ES.score_reference(T4, P4, ["chrA"], PROM, BODY, ENH, VAR4)
    for m, v in _hand().items():
        assert got[m] == pytest.approx(v, rel=1e-12), m


def test_nine_measures_fast_scorer_hand_computed():
    S = ES.ExpScorer(T4, (PROM, BODY, ENH), VAR4, chroms=["chrA"])
    got = S.score_sets(P4, sets=[("one", ["chrA"])])["one"]
    for m, v in _hand().items():
        assert got[m] == pytest.approx(v, rel=1e-12), m


def test_top1_measures_hand_computed():
    t = np.arange(200, dtype=np.float32)
    p = t.copy()
    p[199] = 190.0
    p[0] = 500.0
    T, P = {"chrA": t}, {"chrA": p}
    ann = ({}, {}, {})
    ref = ES.score_reference(T, P, ["chrA"], *ann)
    fast = ES.ExpScorer(T, ann, chroms=["chrA"]).score_sets(P, sets=[("one", ["chrA"])])["one"]
    # n1 = 2. Top 2 of truth: bins 198, 199 -> d = [0, 9]. Top 2 of prediction: 500 (bin 0) and
    # 198 (bin 198) -> d = [-500, 0].
    for got in (ref, fast):
        assert got["mse1obs"] == pytest.approx(81 / 2)
        assert got["mse1imp"] == pytest.approx(250000 / 2)
        assert math.isnan(got["msevar"])


def test_fast_scorer_equals_reference_on_chromosome_subsets():
    rng = np.random.default_rng(0)
    chroms = ["c1", "c2", "c3"]
    T = {c: np.round(rng.gamma(0.3, 2.0, 400 + 37 * i), 1).astype(np.float32)
         for i, c in enumerate(chroms)}                      # rounded: many ties, as real tracks
    P = {c: np.round(T[c] * 0.6 + rng.gamma(0.3, 1.0, T[c].shape[0]), 1).astype(np.float32)
         for c in chroms}
    V = {c: rng.random(T[c].shape[0]) for c in chroms}
    prom = {c: np.array([[5, 85], [100, 180]]) for c in chroms}
    body = {c: np.array([[10, 60], [40, 90], [380, 500]]) for c in chroms}
    enh = {"c2": np.array([[0, 30]])}
    sets = [("all", chroms), ("a", ["c1", "c3"]), ("b", ["c2"])]
    S = ES.ExpScorer(T, (prom, body, enh), V, chroms=chroms)
    got = S.score_sets(P, sets=sets)
    for name, cs in sets:
        ref = ES.score_reference(T, P, sorted(cs), prom, body, enh, V)
        for m in ES.MEASURES9:
            assert got[name][m] == pytest.approx(ref[m], rel=1e-9, nan_ok=True), (name, m)


def test_rank_average_matches_scipy():
    from scipy.stats import rankdata
    x = np.array([3, 1, 2, 2, 0, 3, 3, 1.5])
    assert np.array_equal(ES.rank_average(x), rankdata(x, "average"))


def test_blacklist_deletes_bins_and_shifts_indices():
    y = np.arange(10, dtype=np.float32)
    out = ES.apply_blacklist(y, np.array([2, 3, 20]))
    assert out.tolist() == [0, 1, 4, 5, 6, 7, 8, 9]


def test_align_truncates_and_refuses_other_grids():
    P, T = ES.align({"c": np.zeros(5)}, {"c": np.zeros(4)}, ["c"])
    assert P["c"].shape == T["c"].shape == (4,)
    with pytest.raises(ValueError):
        ES.align({"c": np.zeros(6)}, {"c": np.zeros(4)}, ["c"])


def test_two_pass_var_is_population_variance():
    a = [np.array([0.0, 1.0]), np.array([2.0, 1.0]), np.array([4.0, 4.0])]
    assert np.allclose(ES.two_pass_var(a), np.std(np.array(a), axis=0) ** 2)
    with pytest.raises(ValueError):
        ES.two_pass_var(a[:1])


def test_msevar_members_leave_out_c38m02():
    m = ES.msevar_members(["C38M02", "C12M02", "C31M02", "C05M17"])
    assert m == {"M02": ["C12M02", "C31M02"], "M17": ["C05M17"]}


def test_load_annotations_follows_the_challenge_bin_arithmetic(tmp_path):
    def wgz(name, lines):
        with gzip.open(tmp_path / name, "wt") as fh:
            fh.write("".join("\t".join(map(str, r)) + "\n" for r in lines))
    wgz("gencode.v29.genes.gtf.bed.gz", [
        ("chr1", 1000, 1600, "g1", 0, "+"),     # s = 40 < 80: promoter dropped
        ("chr1", 5000, 5100, "g2", 0, "+"),     # s = 200, e = 205: promoter [120, 200)
        ("chr1", 2000, 2500, "g3", 0, "-"),     # s = 80, e = 101: promoter [101, 181)
    ])
    wgz("F5.hg38.enhancers.bed.gz", [("chr1", 50, 99) + (0,) * 9])
    prom, body, enh = ES.load_annotations(tmp_path)
    assert prom["chr1"].tolist() == [[120, 200], [101, 181]]
    assert body["chr1"].tolist() == [[40, 65], [200, 205], [80, 101]]
    assert enh["chr1"].tolist() == [[2, 4]]


# ---- bigWig reader -------------------------------------------------------------------------
def _write_bigwig(path, chroms, blocks):
    """Minimal bigWig v4 writer: chroms [(name, size)], blocks [(chrom id, section bytes)];
    zlib-compressed blocks, one-leaf chromosome B+ tree and one-leaf R-tree."""
    key = max(len(n) for n, _ in chroms)
    ctree = struct.pack("<IIIIQQ", 0x78CA8C91, len(chroms), key, 8, len(chroms), 0)
    ctree += struct.pack("<BBH", 1, 0, len(chroms))
    for i, (n, s) in enumerate(chroms):
        ctree += n.encode().ljust(key, b"\x00") + struct.pack("<II", i, s)
    ctree_off = 64
    data_off = ctree_off + len(ctree)
    data = struct.pack("<Q", len(blocks))
    leaves = []
    for cid, sec in blocks:
        comp = zlib.compress(sec)
        leaves.append((cid, data_off + len(data), len(comp)))
        data += comp
    index_off = data_off + len(data)
    index = struct.pack("<IIQIIIIQII", 0x2468ACE0, len(blocks), len(blocks), 0, 0, 0, 0, 0, 1, 0)
    index += struct.pack("<BBH", 1, 0, len(leaves))
    for cid, off, size in leaves:
        index += struct.pack("<IIIIQQ", cid, 0, cid, 1 << 30, off, size)
    head = struct.pack("<IHHQQQHHQQIQ", 0x888FFC26, 4, 0, ctree_off, data_off, index_off,
                       0, 0, 0, 0, 32768, 0)
    Path(path).write_bytes(head + ctree + data + index)


def _section(cid, start, end, step, span, typ, items):
    if typ == 1:
        body = b"".join(struct.pack("<IIf", s, e, v) for s, e, v in items)
    elif typ == 2:
        body = b"".join(struct.pack("<If", s, v) for s, v in items)
    else:
        body = b"".join(struct.pack("<f", v) for v in items)
    return struct.pack("<IIIIIBBH", cid, start, end, step, span, typ, 0, len(items)) + body


def test_bigwig_reader_bins_like_the_challenge(tmp_path):
    bw = tmp_path / "t.bigwig"
    _write_bigwig(bw, [("chr1", 60), ("chr2", 27)], [
        (0, _section(0, 0, 30, 0, 0, 1, [(0, 10, 1.0), (10, 30, 2.0)])),     # bedGraph
        (0, _section(0, 40, 45, 0, 5, 2, [(40, 4.0)])),                     # varStep
        (1, _section(1, 0, 26, 10, 6, 3, [1.0, 2.0, 3.0])),                 # fixedStep
    ])
    with ES.BigWig(bw) as r:
        assert r.chroms() == {"chr1": 60, "chr2": 27}
        s, e, v = r.items("chr2")
        assert s.tolist() == [0, 10, 20] and e.tolist() == [6, 16, 26]
    y = ES.bin_bigwig(bw, ["chr1", "chr2"])
    # chr1: bin 0 = (10*1 + 15*2)/25; bin 1 holds the end of the last item (45), so it is the
    # mean over covered bases: (5*2 + 5*4)/10; bin 2 has no data.
    assert y["chr1"].tolist() == pytest.approx([1.6, 3.0, 0.0])
    # chr2: bin 0 = (6*1 + 6*2 + 5*3)/25; bin 1 = [25, 27) covered by one base of value 3.
    assert y["chr2"].tolist() == pytest.approx([1.32, 3.0])
    assert y["chr1"].dtype == np.float32


# ---- ranking -------------------------------------------------------------------------------
def _rows(team, cell, values):
    return [dict(team_id=team, cell=cell, assay="M17", bootstrap_id=b,
                 **{m: values[m] for m in ES.MEASURES9}) for b in range(10)]


def test_global_ranks_hand_computed():
    good = {m: (0.9 if m in ES.HIGHER_IS_BETTER else 1.0) for m in ES.MEASURES9}
    bad = {m: (0.1 if m in ES.HIGHER_IS_BETTER else 2.0) for m in ES.MEASURES9}
    rows = _rows(1, "C01", good) + _rows(2, "C01", bad) + _rows(1, "C02", good)
    board = ES.global_ranks(rows)
    assert [d["team_id"] for d in board] == [1, 2]
    a, b = board
    # C01 (n = 2): team 1 rank 1 -> 1/3, team 2 rank 2 -> min(0.5, 2/3) = 0.5.
    # C02: team 1 alone -> min(0.5, 1/2) = 0.5; team 2 missing -> 0.5.
    assert a["score_mean"] == pytest.approx((1 / 3 + 0.5) / 2)
    assert b["score_mean"] == pytest.approx(0.5)
    assert a["rank_stat"] == 1.0 and b["rank_stat"] == 2.0
