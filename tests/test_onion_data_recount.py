"""t152 Amendment 3 chunk RCt: the 1 bp point-rule recount on synthetic tagAligns (CPU, seconds).

Two chromosomes of 4 region blocks each; a pooled base read set split in halves; the eic_v1
counts.h5 is made with the store's own overlap rule (`bin25.counts_from_tagalign`), so `verify`
checks the real rule. Every synthetic read's point lies inside a region.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tools" / "t112", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import recount as RC  # noqa: E402
import bin25  # noqa: E402

RB = RC.REGION_BP
SIZES = {"chr1": 4 * RB + 37, "chr2": 4 * RB + 11}
EXP = "ENCSRTST001"
MID = f"{EXP}.pooled.uniform.base"
SHIFT_FL = 150


def write_ta(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for c, s, e, st in rows:
            fh.write(f"{c}\t{s}\t{e}\tN\t1000\t{st}\n")


def make_reads(rng, regions, n, fl):
    """Reads whose point (fl // 2 shift) lies inside a region; returns (rows, points)."""
    sh = fl // 2
    rows, pts = [], []
    for _ in range(n):
        r = regions[rng.integers(len(regions))]
        p = r["start_bp"] + int(rng.integers(100, RB - 100))
        ln = int(rng.integers(20, 80))
        if rng.random() < 0.5:
            s = p - sh
            rows.append((r["chrom"], s, s + ln, "+"))
        else:
            e = p + sh + 1
            rows.append((r["chrom"], e - ln, e, "-"))
        pts.append((r["chrom"], p))
    return rows, pts


@pytest.fixture()
def tree(tmp_path):
    gdir = tmp_path / "genome"
    gdir.mkdir()
    (gdir / "chrom_sizes.json").write_text(json.dumps({"chrom_sizes": SIZES}))
    with h5py.File(gdir / "mask.h5", "w") as f:
        for c, n in SIZES.items():
            m = np.ones(n // 25, bool)
            if c == "chr2":
                m[3 * RC.REGION_BINS:3 * RC.REGION_BINS + 4000] = False   # block chr2:3 invalid
            f.create_dataset(c, data=m)
    ev = {"regions": [["chr1", RC.REGION_BINS], ["chr2", 0]],
          "fit_regions": [["chr1", 2 * RC.REGION_BINS]], "region_bins": RC.REGION_BINS}
    (tmp_path / "evalset.json").write_text(json.dumps(ev))
    assert RC.main(["regions", "--evalset", str(tmp_path / "evalset.json"), "--mask",
                    str(gdir / "mask.h5"), "--n-train", "3", "--seed", "0", "--out",
                    str(tmp_path / "regions.tsv")]) == 0
    regions = RC.read_regions(tmp_path / "regions.tsv")
    rng = np.random.default_rng(1)
    rows, pts = make_reads(rng, regions, 6000, SHIFT_FL)
    # a pile-up above 255 at one position forces uint16
    rows += [("chr1", RC.REGION_BINS * 25 + 1000 - SHIFT_FL // 2,
              RC.REGION_BINS * 25 + 1050 - SHIFT_FL // 2, "+")] * 300
    rs = tmp_path / "work" / EXP / "pooled.uniform" / "base"
    half = rng.random(len(rows)) < 0.5
    write_ta(rs / "pr1.tagAlign.gz", [r for r, h in zip(rows, half) if h])
    write_ta(rs / "pr2.tagAlign.gz", [r for r, h in zip(rows, half) if not h])
    (tmp_path / "work" / EXP / "base").mkdir(parents=True)
    (tmp_path / "work" / EXP / "base" / "rep1.xcor.qc").write_text(
        "x\t1\t140,200\t0\t0\t0\t0\t0\t1.1\t1.2\t0\n")
    (tmp_path / "work" / EXP / "base" / "rep2.xcor.qc").write_text(
        "x\t1\t161\t0\t0\t0\t0\t0\t1.1\t1.2\t0\n")
    # eic_v1: counts.h5 by the store's overlap rule, from both halves
    v1m = tmp_path / "meas_v1" / MID
    v1m.mkdir(parents=True)
    hc = [bin25.counts_from_tagalign(rs / f"{h}.tagAlign.gz", SIZES) for h in ("pr1", "pr2")]
    with h5py.File(v1m / "counts.h5", "w") as f:
        for c in SIZES:
            f.create_dataset(c, data=hc[0][c] + hc[1][c])
    pts += [("chr1", RC.REGION_BINS * 25 + 1000)] * 300
    return {"tmp": tmp_path, "regions": regions, "rows": rows, "pts": pts, "rs": rs, "v1m": v1m}


def run_store(t, extra=()):
    store = t["tmp"] / "res"
    args = ["run", "--work", str(t["tmp"] / "work"), "--experiment", EXP, "--regions",
            str(t["tmp"] / "regions.tsv"), "--frag-len", str(SHIFT_FL), "--halves", "1", "--out",
            str(store), "--chrom-sizes", str(t["tmp"] / "genome" / "chrom_sizes.json"), *extra]
    assert RC.main(args) == 0
    return store / MID


def test_regions(tree):
    regs = tree["regions"]
    assert [r["role"] for r in regs] == ["eval", "eval", "train", "train", "train"]
    assert [(r["chrom"], r["start_bp"]) for r in regs[:2]] == [("chr1", RB), ("chr2", 0)]
    keys = {(r["chrom"], r["start_bp"]) for r in regs}
    assert len(keys) == 5
    tr = {(r["chrom"], r["start_bp"]) for r in regs if r["role"] == "train"}
    assert ("chr1", 2 * RB) not in tr and ("chr2", 3 * RB) not in tr     # fit region, mask
    assert all(r["start_bp"] % RB == 0 for r in regs)


def test_pool_exact():
    x = np.random.default_rng(0).integers(0, 4, RB).astype(np.uint8)
    for b in RC.POOL_SIZES:
        y = RC.pool(x, b)
        assert y.shape[0] == RB // b
        assert y.sum() == x[: (RB // b) * b].sum()
    assert RC.pool(x, 125).shape[0] == 3276 and RC.pool(x, 250).shape[0] == 1638
    assert RC.pool(x, 25).sum() == x.sum() and RC.pool(x, 5).sum() == x.sum()
    assert np.array_equal(RC.pool(RC.pool(x, 5), 5), RC.pool(x, 25))


def test_point_rule():
    p = RC.point_positions([100, 100], [150, 150], ["+", "-"], 10)
    assert p.tolist() == [110, 139]
    assert RC.point_positions([100], [150], ["-"], 0).tolist() == [149]
    with pytest.raises(ValueError):
        RC.point_positions([1], [2], ["."], 0)


def test_run_counts_every_read_once(tree):
    d = run_store(tree)
    n = len(tree["rows"])
    with h5py.File(d / "point1bp.h5", "r") as f:
        assert f.attrs["frag_len_shift"] == SHIFT_FL // 2
        assert f.attrs["frag_len_source"] == "given"
        assert f.attrs["rule"] == RC.POINT_RULE
        assert f["r0"].dtype == np.uint16                                # the pile-up
        assert sorted(f.keys()) == [f"r{i}" for i in range(5)]
        tot = sum(int(f[k][()].sum()) for k in f.keys())
        x0 = f["r0"][()].astype(np.int64)
    assert tot == n
    want = np.zeros(RB, np.int64)                       # brute force from the planted points
    for c, p in tree["pts"]:
        if c == "chr1" and RB <= p < 2 * RB:
            want[p - RB] += 1
    assert np.array_equal(x0, want)
    assert x0[1000] >= 300
    for b in RC.POOL_SIZES:
        assert RC.pool(x0, b).sum() == x0[: (RB // b) * b].sum()
    with h5py.File(d / "counts.h5", "r") as f:
        assert f.attrs["resolution"] == 25 and f.attrs["counting_rule"] == "point"
        assert {c: f[c].shape[0] for c in f} == {c: n_ // 25 for c, n_ in SIZES.items()}
        assert sum(int(f[c][()].sum()) for c in f) == n
    tot_h = 0
    for h in ("pr1", "pr2"):
        with h5py.File(d / f"point1bp_{h}.h5", "r") as f:
            tot_h += sum(int(f[k][()].sum()) for k in f.keys())
    assert tot_h == n
    assert (tree["tmp"] / "res" / "regions.tsv").read_bytes() == \
        (tree["tmp"] / "regions.tsv").read_bytes()


def test_verify_pass_and_dropped_read(tree, capsys):
    d = run_store(tree)
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5")]) == 0
    assert "verify failures: 0" in capsys.readouterr().out
    # drop one read that lies in an eval region from the read set, and recount
    pr1 = tree["rs"] / "pr1.tagAlign.gz"
    lines = gzip.open(pr1, "rt").read().splitlines(keepends=True)
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("chr2\t") and int(ln.split("\t")[1]) < RB - 1000)   # eval chr2:0
    with gzip.open(pr1, "wt") as fh:
        fh.writelines(lines[:i] + lines[i + 1:])
    import shutil
    shutil.rmtree(tree["tmp"] / "res")
    d = run_store(tree)
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5")]) == 1
    out = capsys.readouterr().out
    assert "overlap recount differs" in out and "verify failures: 1" in out


def test_verify_catches_tampered_pool(tree, capsys):
    d = run_store(tree)
    with h5py.File(d / "counts.h5", "a") as f:
        f["chr1"][RC.REGION_BINS + 5] += 1
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5")]) == 1
    assert "sha256 differs" in capsys.readouterr().out


def test_frag_len_sources(tree, tmp_path):
    work = tree["tmp"] / "work"
    fl = RC.frag_len_from_sources(work, EXP)
    assert fl == {"frag_len": 150, "frag_len_source": "xcor", "frag_len_per_rep": [140, 161]}
    pm = tmp_path / "m" / f"OTHER.pooled.uniform.base"
    pm.mkdir(parents=True)
    (pm / "provenance.json").write_text(json.dumps({"signal": {"fraglen_per_rep": [200, 211]}}))
    fl = RC.frag_len_from_sources(work, "OTHER", measurements=tmp_path / "m")
    assert fl["frag_len"] == 206 and fl["frag_len_source"] == "provenance"
    tb = tmp_path / "t"
    tb.mkdir()
    with open(tb / "measurements.tsv", "w") as fh:
        fh.write("experiment\tassay\tfrag_len\n")
        fh.write("A\tH3K4me3\t100\nA\tH3K4me3\t100\nB\tH3K4me3\t180\nC\tH3K4me3\t220\n"
                 "NONE\tH3K4me3\t\nD\tCTCF\t90\n")
    fl = RC.frag_len_from_sources(work, "NONE", tables=tb)
    assert fl["frag_len"] == 180 and fl["frag_len_source"] == "assay_median"


def test_tables_and_verify_all(tree, capsys):
    from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, WITHHELD_COLUMNS, read_tsv
    run_store(tree)
    v1 = tree["tmp"] / "eic_v1"
    v1.mkdir()
    row = {k: "" for k in MEASUREMENT_COLUMNS}
    row.update({"measurement_id": MID, "experiment": EXP, "assay": "H3K27ac", "in_eic": "val",
                "unit": "pooled", "layer": "uniform", "has_signal": "1", "has_pseudoreps": "1",
                "counts_path": f"h5:{tree['v1m'] / 'counts.h5'}", "sha256": "x"})
    RC.write_tsv(v1 / "measurements.tsv", MEASUREMENT_COLUMNS, [row])
    pair = {k: "" for k in PAIR_COLUMNS}
    pair.update({"pair_id": "p1", "a": MID, "b": MID, "rung": "pseudoreplicate", "split": "val"})
    other = dict(pair, pair_id="p2", rung="designed")
    RC.write_tsv(v1 / "pairs.tsv", PAIR_COLUMNS, [pair, other])
    (v1 / "covariates.json").write_text('{"x": 1}\n')
    out = tree["tmp"] / "eic_res_v1"
    assert RC.main(["tables", "--eic-v1", str(v1), "--store", str(tree["tmp"] / "res"),
                    "--out", str(out)]) == 0
    rows = read_tsv(out / "measurements.tsv", MEASUREMENT_COLUMNS)
    assert len(rows) == 1 and rows[0]["has_signal"] == "0" and rows[0]["frag_len"] == "150"
    assert rows[0]["counts_path"].endswith(f"res/{MID}/counts.h5")
    assert [p["pair_id"] for p in read_tsv(out / "pairs.tsv", PAIR_COLUMNS)] == ["p1"]
    assert read_tsv(out / "withheld.tsv", WITHHELD_COLUMNS) == []
    assert (out / "covariates.json").read_text() == '{"x": 1}\n'
    assert RC.read_regions(out / "regions.tsv") == tree["regions"]
    capsys.readouterr()
    assert RC.main(["verify", "--all", "--tables", str(out)]) == 0
    assert "verify failures: 0" in capsys.readouterr().out
    with pytest.raises(SystemExit):     # frozen tables are never overwritten
        RC.main(["tables", "--eic-v1", str(v1), "--store", str(tree["tmp"] / "res"),
                 "--out", str(out)])
