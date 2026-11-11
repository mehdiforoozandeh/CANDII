"""t167 (chunk D5): the alignment-level view kit on tiny synthetic tagAligns (CPU, seconds).

The pipeline SIF is not on the laptop, so the split here is the seeded python split; the read-level
contract (halves partition the reads, every arm is built from one half alone, nested exact
thinning, covariates per arm) is what is tested. Each synthetic read carries a unique id in
tagAlign column 4 so that read identity can be followed through the halves and the arms.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import align_views as AV  # noqa: E402

SIZES = {"chr1": 2000, "chr21": 1000}


def _write_ta(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for c, s, e, name, strand in rows:
            f.write(f"{c}\t{s}\t{e}\t{name}\t1000\t{strand}\n")
    return path


def _read_ta(path) -> list:
    with gzip.open(path, "rt") as f:
        return [ln.rstrip("\n").split("\t") for ln in f]


def _random_rows(n, seed, prefix="r"):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = "chr1" if rng.random() < 0.6 else "chr21"
        s = int(rng.integers(0, SIZES[c] - 50))
        out.append((c, s, s + 36, f"{prefix}{i}", "+-"[i % 2]))
    return out


def _build(tmp_path, sources, **kw):
    sz = tmp_path / "chrom_sizes.json"
    sz.write_text(json.dumps({"chrom_sizes": SIZES}))
    tag = kw.pop("tag", "a")
    (tmp_path / f"tmp_{tag}").mkdir(exist_ok=True)
    argv = ["build", "--experiment", "ENCSRTEST", "--assay", "H3K27ac", "--cell", "C01",
            "--eic-track", "C01M16", "--role", "val",
            "--arms", kw.pop("arms", "base,depth0.50,depth0.25,depth0.125"),
            "--splitter", "python", "--seed", str(kw.pop("seed", 0)),
            "--work", str(tmp_path / f"work_{tag}"), "--out", str(tmp_path / f"out_{tag}"),
            "--chrom-sizes", str(sz), "--tmp", str(tmp_path / f"tmp_{tag}"),
            "--read-length", "36", "--meta", "lab=TestLab", "--meta", "platform=Illumina"]
    if kw.pop("paired_end", False):
        argv.append("--paired-end")
    for s in sources:
        argv += ["--source", str(s)]
    assert not kw
    assert AV.main(argv) == 0
    return tmp_path / f"out_{tag}", tmp_path / f"work_{tag}"


def _mdir(out, unit, arm):
    return out / f"ENCSRTEST.{unit}.uniform.{arm}"


def test_counts_match_hand_count(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", [
        ("chr1", 0, 25, "a", "+"),      # end bin inclusive: bins 0 and 1
        ("chr1", 30, 60, "b", "-"),     # bins 1, 2
        ("chr21", 990, 1000, "c", "+"),  # bin 39; bin 40 is past the 40-bin grid
        ("chrM", 5, 40, "d", "+"),      # not a genome-layer chromosome: ignored
    ])
    out, _ = _build(tmp_path, [ta], arms="base")
    c = AV.read_h5(_mdir(out, "pooled", "base") / "counts.h5")
    want1 = np.zeros(80, np.uint32)
    np.add.at(want1, [0, 1, 1, 2], 1)
    want21 = np.zeros(40, np.uint32)
    want21[39] = 1
    assert np.array_equal(c["chr1"], want1)
    assert np.array_equal(c["chr21"], want21)


def test_halves_partition_reads_and_sum_to_pooled(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(1001, 1))
    out, work = _build(tmp_path, [ta])
    d = _mdir(out, "pooled", "base")
    c, p1, p2 = (AV.read_h5(d / n) for n in ("counts.h5", "pr1.h5", "pr2.h5"))
    for k in c:
        assert np.array_equal(p1[k].astype(np.uint64) + p2[k], c[k])
    rs = work / "ENCSRTEST" / "pooled.uniform" / "base"
    ids = {h: [r[3] for r in _read_ta(rs / f"{h}.tagAlign.gz")] for h in AV.HALVES}
    assert len(ids["pr1"]) == 501 and len(ids["pr2"]) == 500          # ceil / floor, as spr
    assert set(ids["pr1"]).isdisjoint(ids["pr2"])
    assert sorted(ids["pr1"] + ids["pr2"]) == sorted(r[3] for r in _read_ta(ta))
    meta = json.loads((d / "meta.json").read_text())
    assert meta["depth_reads"] == "1001"


def test_thinning_depth_and_seed_reproducible(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(4000, 2))
    out, work = _build(tmp_path, [ta], tag="a", seed=0)
    out_b, work_b = _build(tmp_path, [ta], tag="b", seed=0)
    out_c, work_c = _build(tmp_path, [ta], tag="c", seed=7)
    for lab, q in AV.DEPTH_LEVELS:
        m = json.loads((_mdir(out, "pooled", f"depth{lab}") / "meta.json").read_text())
        n = int(m["depth_reads"])
        sd = np.sqrt(4000 * q * (1 - q))
        assert abs(n - q * 4000) <= 3 * sd                     # within binomial error
        assert abs(n - q * 4000) <= 2                          # exact per half, up to rounding
        assert m["reads_subset_of"] == "ENCSRTEST.pooled.uniform.base"
        rel = Path("ENCSRTEST") / "pooled.uniform" / f"depth{lab}" / "pr1.tagAlign.gz"
        md5 = [hashlib.md5((w / rel).read_bytes()).hexdigest() for w in (work, work_b, work_c)]
        assert md5[0] == md5[1] and md5[0] != md5[2]
    # nested: each level's reads are a subset of the next larger level's, half by half
    rs = work / "ENCSRTEST" / "pooled.uniform"
    for h in AV.HALVES:
        sets = [{r[3] for r in _read_ta(rs / a / f"{h}.tagAlign.gz")}
                for a in ("base", "depth0.50", "depth0.25", "depth0.125")]
        assert sets[3] <= sets[2] <= sets[1] <= sets[0]


def test_arm_from_pr1_shares_no_read_with_pr2(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(3000, 3))
    _, work = _build(tmp_path, [ta])
    rs = work / "ENCSRTEST" / "pooled.uniform"
    pr2_any = set()
    for a in ("base", "depth0.50", "depth0.25", "depth0.125"):
        pr2_any |= {r[3] for r in _read_ta(rs / a / "pr2.tagAlign.gz")}
    for a in ("base", "depth0.50", "depth0.25", "depth0.125"):
        assert {r[3] for r in _read_ta(rs / a / "pr1.tagAlign.gz")}.isdisjoint(pr2_any)


def test_name_split_follows_reads_across_arms(tmp_path):
    """BAM route: a read in base and in nodedup lands in the same half; names become N."""
    # unique positions, so a position identifies a read; a duplicate repeats its read's position
    base = [("chr1" if i < 1900 else "chr21", i % 1900, i % 1900 + 36, f"r{i}", "+")
            for i in range(2000)]
    dups = [(c, s, e, f"dup{i}", st) for i, (c, s, e, _, st) in enumerate(base[:500])]
    nb = _write_ta(tmp_path / "base.named.gz", base)
    nn = _write_ta(tmp_path / "nodedup.named.gz", base + dups)
    salt = b"0|ENCSRTEST|pooled|"
    AV.split_named(nb, tmp_path / "b1.gz", tmp_path / "b2.gz", salt)
    AV.split_named(nn, tmp_path / "n1.gz", tmp_path / "n2.gz", salt)
    assert all(r[3] == "N" for r in _read_ta(tmp_path / "b1.gz"))
    side = {}
    for name, half in [(r[3], AV.name_half(r[3].encode(), salt)) for r in base + dups]:
        side[name] = half
    b1 = [r for r in base if side[r[3]] == 0]
    assert len(_read_ta(tmp_path / "b1.gz")) == len(b1)
    pos = lambda rows: {tuple(r[:3]) + (r[5],) for r in rows}  # noqa: E731
    n2 = pos(_read_ta(tmp_path / "n2.gz"))
    b1_unique = pos([[c, str(s), str(e), n, "1000", st] for c, s, e, n, st in b1])
    dup_pos_pr2 = pos([[c, str(s), str(e), n, "1000", st] for c, s, e, n, st in dups
                       if side[n] == 1])
    # every pr2 line of nodedup is a pr2 read of base or a pr2 duplicate: never a pr1 base read
    assert (n2 & b1_unique) <= dup_pos_pr2
    assert 0.4 < len(b1) / len(base) < 0.6


def test_paired_end_fragments_stay_together(tmp_path):
    rows = []
    for i, (c, s, e, _, _) in enumerate(_random_rows(600, 5)):
        rows += [(c, s, e, f"f{i}", "+"), (c, s + 100, e + 100, f"f{i}", "-")]
    ta = _write_ta(tmp_path / "pe.tagAlign.gz", rows)
    out, work = _build(tmp_path, [ta], paired_end=True)
    for a in ("base", "depth0.25"):
        for h in AV.HALVES:
            r = _read_ta(work / "ENCSRTEST" / "pooled.uniform" / a / f"{h}.tagAlign.gz")
            assert len(r) % 2 == 0
            assert all(r[i][3] == r[i + 1][3] for i in range(0, len(r), 2))
    m = json.loads((_mdir(out, "pooled", "base") / "meta.json").read_text())
    assert m["run_type"] == "paired-ended" and m["depth_fragments"] == "600"


def test_replicates_pool_as_union(tmp_path):
    a = _write_ta(tmp_path / "r1.tagAlign.gz", _random_rows(800, 6, "a"))
    b = _write_ta(tmp_path / "r2.tagAlign.gz", _random_rows(700, 7, "b"))
    out, _ = _build(tmp_path, [a, b], arms="base,depth0.50")
    for arm in ("base", "depth0.50"):
        for n in ("counts.h5", "pr1.h5", "pr2.h5"):
            p = AV.read_h5(_mdir(out, "pooled", arm) / n)
            r1 = AV.read_h5(_mdir(out, "br1", arm) / n)
            r2 = AV.read_h5(_mdir(out, "br2", arm) / n)
            for k in p:
                assert np.array_equal(p[k], r1[k].astype(np.uint64) + r2[k])
    m = json.loads((_mdir(out, "br2", "base") / "meta.json").read_text())
    assert m["bio_rep"] == "2" and m["reads_subset_of"] == "ENCSRTEST.pooled.uniform.base"
    assert json.loads((_mdir(out, "pooled", "base") / "meta.json").read_text())["depth_reads"] \
        == "1500"


def test_meta_covariates_per_arm(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(400, 8))
    out, _ = _build(tmp_path, [ta])
    from minicandi.dataset import MEASUREMENT_COLUMNS
    base = json.loads((_mdir(out, "pooled", "base") / "meta.json").read_text())
    assert list(base) == list(MEASUREMENT_COLUMNS)
    assert (base["arm_knob"], base["arm_value"], base["dedup"], base["mapq_threshold"]) == \
        ("", "", "1", "30")
    assert (base["assay_class"], base["in_eic"], base["unit"], base["layer"]) == \
        ("narrow_histone", "val", "pooled", "uniform")
    assert base["lab"] == "TestLab" and base["has_pseudoreps"] == "1"
    assert base["pipeline"] == "chip-seq-pipeline2" and base["pipeline_version"] == "v2.2.2"
    assert base["counts_path"].startswith("h5:") and base["counts_path"].endswith("counts.h5")
    d = json.loads((_mdir(out, "pooled", "depth0.125") / "meta.json").read_text())
    assert (d["arm_knob"], d["arm_value"], d["parent_measurement_id"]) == \
        ("depth", "0.125", "ENCSRTEST.pooled.uniform.base")
    f = {"dedup": "1", "mapq_threshold": "30"}
    assert AV.arm_fields("nodedup", f) == {"arm_knob": "nodedup", "arm_value": "", "dedup": "0",
                                           "mapq_threshold": "30"}
    assert AV.arm_fields("mapq0", f) == {"arm_knob": "mapq", "arm_value": "0", "dedup": "1",
                                         "mapq_threshold": "0"}


def test_filter_command_changes_one_setting_and_refuses_the_atac_trap():
    kw = dict(paired_end=False, chrsz="/c/GRCh38_EBV.chrom.sizes.tsv", nth=4, mem_gb=8)
    nd = AV.filter_command("/x.bam", "/o", pipeline="chip", arm="nodedup", **kw)
    mq = AV.filter_command("/x.bam", "/o", pipeline="chip", arm="mapq0", **kw)
    assert "--no-dup-removal" in nd and "--mapq-thresh 30" in nd
    assert "--no-dup-removal" not in mq and "--mapq-thresh 0" in mq and "--multimapping 0" in mq
    with pytest.raises(SystemExit, match="mapq0 refused"):
        AV.filter_command("/x.bam", "/o", pipeline="atac", arm="mapq0", **kw)


def test_verify_passes_then_catches_tampering(tmp_path, capsys):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(2000, 9))
    out, _ = _build(tmp_path, [ta])
    assert AV.verify(out) == 0
    assert "1 experiments, all checks pass" in capsys.readouterr().out
    import h5py
    with h5py.File(_mdir(out, "pooled", "depth0.50") / "pr1.h5", "r+") as f:
        v = f["chr1"][()]
        v[3] += 1
        f["chr1"][...] = v
    assert AV.verify(out) == 1
    assert "halves do not sum" in capsys.readouterr().out


def test_skip_if_exists(tmp_path):
    ta = _write_ta(tmp_path / "src.tagAlign.gz", _random_rows(300, 10))
    out, _ = _build(tmp_path, [ta], arms="base")
    p = _mdir(out, "pooled", "base") / "provenance.json"
    before = p.read_text()
    _build(tmp_path, [ta], arms="base")
    assert p.read_text() == before
