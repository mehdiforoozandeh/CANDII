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


# --- KF (t180): one driver from a build-list row; BAM route; DNase; native; signal switch -------

import shutil  # noqa: E402
import subprocess  # noqa: E402


def test_name_key_drops_mate_suffix_and_salt_is_per_experiment():
    s = AV.split_salt(0, "ENCSRX")
    assert s == b"0|ENCSRX|"
    for n in (b"read7", b"HWI-1:2:3:4"):
        assert AV.name_half(n + b"/1", s) == AV.name_half(n + b"/2", s) == AV.name_half(n, s)
        assert AV.name_token(n + b"/2") == AV.name_token(n)
    # the same read lands in the same half whatever the unit or arm (salt has neither)
    assert AV.split_salt(0, "ENCSRX") == AV.split_salt(0, "ENCSRX")
    assert AV.split_salt(0, "ENCSRX") != AV.split_salt(0, "ENCSRY")


def test_skip_fe_runs_every_command_but_the_fold_enrichment_branch(tmp_path):
    """SKIP_FE_PY on a stand-in signal script: the p-value branch runs as written, FE does not."""
    lib = tmp_path / "encode_lib_common.py"
    lib.write_text("import subprocess\n"
                   "def run_shell_cmd(c):\n"
                   "    open(LOG, 'a').write(c + '\\n')\n"
                   f"LOG = {str(tmp_path / 'ran.txt')!r}\n")
    script = tmp_path / "encode_task_macs2_signal_track_chip.py"
    script.write_text(
        "import sys\nfrom encode_lib_common import run_shell_cmd\n"
        "def main():\n"
        "    p = sys.argv[1]\n"
        "    run_shell_cmd(f' macs2 callpeak -t {p} -B --SPMR')\n"
        "    run_shell_cmd(f'macs2 bdgcmp -t \"{p}_treat_pileup.bdg\" -m FE ')\n"
        "    run_shell_cmd(f'bedtools slop -i \"{p}_FE.bdg\" | bedClip stdin x {p}.fc.signal.bedgraph')\n"
        "    run_shell_cmd(f'sort {p}.fc.signal.bedgraph > {p}.fc.signal.srt.bedgraph')\n"
        "    run_shell_cmd(f'bedGraphToBigWig {p}.fc.signal.srt.bedgraph c {p}.fc.signal.bigwig')\n"
        "    run_shell_cmd(f'macs2 bdgcmp -t \"{p}_treat_pileup.bdg\" -m ppois -S 1.0')\n"
        "    run_shell_cmd(f'bedGraphToBigWig {p}.pval.signal.srt.bedgraph c {p}.pval.signal.bigwig')\n"
        "if __name__ == '__main__':\n    raise SystemExit('must not run as __main__')\n")
    subprocess.run([sys.executable, "-c", AV.SKIP_FE_PY, str(script), "PFX"], check=True,
                   cwd=tmp_path)
    ran = (tmp_path / "ran.txt").read_text().splitlines()
    assert ran == [" macs2 callpeak -t PFX -B --SPMR",
                   'macs2 bdgcmp -t "PFX_treat_pileup.bdg" -m ppois -S 1.0',
                   "bedGraphToBigWig PFX.pval.signal.srt.bedgraph c PFX.pval.signal.bigwig"]


def test_signal_commands_per_pipeline():
    chip = AV.signal_inner("chip", "/t/a.tagAlign.gz", "/o", chrsz="/c/sz.tsv",
                           ctl="/t/c.tagAlign.gz", fraglen=212, ctl_subsample=0)
    assert "encode_task_macs2_signal_track_chip.py" in chip and "--fraglen 212" in chip
    assert "/t/a.tagAlign.gz /t/c.tagAlign.gz" in chip and "--ctl-subsample" not in chip
    assert "--pval-thresh 0.01" in chip and chip.startswith("python3 -c ")
    sub = AV.signal_inner("chip", "/t/a.tagAlign.gz", "/o", chrsz="/c", ctl="/t/c.gz",
                          fraglen=200, ctl_subsample=5000, ctl_paired_end=True)
    assert "--ctl-subsample 5000 --ctl-paired-end" in sub
    for p in ("atac", "dnase"):   # ruling 19: DNase uses the ATAC no-control recipe
        c = AV.signal_inner(p, "/t/a.tagAlign.gz", "/o", chrsz="/c")
        assert "encode_task_macs2_signal_track_atac.py" in c and "--smooth-win 150" in c
        assert "--fraglen" not in c
    with pytest.raises(SystemExit, match="control"):
        AV.signal_inner("chip", "/t/a.tagAlign.gz", "/o", chrsz="/c")


def test_dnase_filter_reruns_the_release_steps_with_one_setting():
    c0 = AV.dnase_filter_command("/w/rep1.unfiltered.bam", "/f", min_mapq=0, nth=8)
    c10 = AV.dnase_filter_command("/w/rep1.unfiltered.bam", "/f", min_mapq=10, nth=8)
    assert "--min_mapq 0" in c0 and "--min_mapq 10" in c10
    assert c0.replace("--min_mapq 0", "--min_mapq 10") == c10          # one setting differs
    order = ["filter_reads.py", "samtools sort", "RevertOriginalBaseQualitiesAndAddMateCigar",
             "MarkDuplicatesWithMateCigar", "view -F 512", "view -F 4096"]
    pos = [c0.index(x) for x in order]
    assert pos == sorted(pos) and "MINIMUM_DISTANCE=300" in c0
    assert "--multimapping" in AV.filter_command("/x.bam", "/o", pipeline="chip", arm="base",
                                                 paired_end=False, chrsz="/c", nth=2, mem_gb=4)


def test_arm_fields_for_fastq_level_arms():
    f = {"dedup": "1", "mapq_threshold": "30"}
    assert AV.arm_fields("se", f)["arm_knob"] == "se"
    assert (AV.arm_fields("crop36", f)["arm_knob"], AV.arm_fields("crop36", f)["arm_value"]) == \
        ("crop", "36")


# a tiny "aligned" experiment: names r<i>; DUP<i> duplicates; LOW<i> reads below MAPQ 30
def _named(rows):
    return [(c, s, e, n, st) for c, s, e, n, st in rows]


def _exp_reads(seed, n, prefix):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = "chr1" if rng.random() < 0.6 else "chr21"
        s = int(rng.integers(0, SIZES[c] - 60))
        out.append((c, s, s + 36, f"{prefix}r{i}", "+-"[i % 2]))
    dups = [(c, s, e, f"{prefix}DUP{i}", st) for i, (c, s, e, _, st) in enumerate(out[:n // 5])]
    low = [(c, s + 3, e + 3, f"{prefix}LOW{i}", st) for i, (c, s, e, _, st) in enumerate(out[:n // 7])]
    return out, dups, low


def _write_rep(d: Path, k: int, seed: int, n: int, prefix: str, dnase=False):
    good, dups, low = _exp_reads(seed, n, prefix)
    # the filtered BAM: DNase keeps duplicates flagged; chip has removed them
    _write_ta(d / f"rep{k}.filtered.bam", good + dups if dnase else good)
    _write_ta(d / f"rep{k}.unfiltered.bam", good + dups + low)
    _write_ta(d / f"rep{k}.tagAlign.gz", [(c, s, e, "N", st) for c, s, e, _, st in good])
    return good


BL_COLS = ("row priority experiment eic_track role assay assay_class unit arm fastq_files "
           "fastq_gb control_experiment pipeline keep_bam expected_measurements").split()


def _bl(path: Path, rows):
    lines = ["\t".join(BL_COLS)] + ["\t".join(str(r.get(k, "")) for k in BL_COLS) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _exp(mids):
    return ";".join(f"{m}:counts+pr1+pr2" for m in mids)


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """Patch the SIF steps: a "BAM" here is a named tagAlign, so bamtobed is a copy; the filter
    re-run drops the reads the setting would drop (DUP = duplicate, LOW = MAPQ < 30)."""
    def named_ta(bam, out, *, paired_end, tmp, nth, pipeline="chip", drop_dups=False):
        if drop_dups:
            return f"gzip -dc {bam} | grep -v DUP | gzip -nc > {out}"
        return f"cp {bam} {out}"

    def filtered(cfg, cost, unit, arm, unfiltered):
        fdir = Path(tempfile.mkdtemp(dir=cfg["tmp"]))
        drop = {"nodedup": ["LOW"], "mapq0": ["DUP"], "base": ["DUP", "LOW"]}[arm]
        if cfg["pipeline"] == "dnase":
            drop = [x for x in drop if x != "DUP"]          # DNase flags duplicates, keeps them
        rows = [r for r in _read_ta(unfiltered) if not any(x in r[3] for x in drop)]
        out = fdir / "x.bam"
        _write_ta(out, [(r[0], int(r[1]), int(r[2]), r[3], r[5]) for r in rows])
        return out, f"filter {arm}", fdir

    import tempfile
    monkeypatch.setattr(AV, "named_ta_command", named_ta)
    monkeypatch.setattr(AV, "_filtered_bam", filtered)
    monkeypatch.setattr(AV.bam_arm, "apptainer", lambda inner, sif, tmp, wd: inner)
    monkeypatch.setattr(AV, "sif_info", lambda sif, p: {"path": sif, "sha256": "0" * 64})
    sz = tmp_path / "chrom_sizes.json"
    sz.write_text(json.dumps({"chrom_sizes": SIZES}))
    work, out, fq = tmp_path / "work", tmp_path / "meas", tmp_path / "fq"

    def arm_json(d, reps, run_type, **kw):
        d.mkdir(parents=True, exist_ok=True)
        (d / "arm.json").write_text(json.dumps({
            "reps": [{"rep": i, "bio_rep": b} for i, b in enumerate(reps, 1)],
            "paired_end_per_rep": [False] * len(reps), "run_type": run_type,
            "read_length": kw.pop("read_length", 76), "platforms": ["HiSeq2000"],
            "pipeline": kw.pop("pipeline", "chip-seq-pipeline2"),
            "pipeline_version": kw.pop("version", "v2.2.2"), "image_digest": "sha256:abc",
            "mapq_threshold": kw.pop("mapq", 30), **kw}))
        (d / "ALIGN_DONE").write_text("{}")

    # control (1 rep), ChIP validation experiment (2 reps) + its se arm + one lane, DNase (1 rep)
    d = work / "ENCSRCTL" / "base"
    arm_json(d, [1], "single-ended")
    _write_rep(d, 1, 1, 300, "c")
    d = work / "ENCSRCHP" / "base"
    arm_json(d, [1, 3], "single-ended")
    g1 = _write_rep(d, 1, 2, 400, "a")
    g2 = _write_rep(d, 2, 3, 300, "b")
    d = work / "ENCSRCHP" / "se"
    arm_json(d, [1, 3], "single-ended", arm_knob="se", arm_value="1")
    for k, g in ((1, g1), (2, g2)):     # mate 1 of the same fragments: the same names
        _write_ta(d / f"rep{k}.filtered.bam", [(c, s + 1, e + 1, n, st) for c, s, e, n, st in g])
        _write_ta(d / f"rep{k}.unfiltered.bam", [(c, s + 1, e + 1, n, st) for c, s, e, n, st in g])
    d = work / "ENCSRCHP" / "br1tr1ln1.base"
    arm_json(d, [1], "single-ended")
    _write_ta(d / "rep1.filtered.bam", g1[:150])
    _write_ta(d / "rep1.unfiltered.bam", g1[:150])
    d = work / "ENCSRDNS" / "base"
    arm_json(d, [1], "single-ended", read_length=36, pipeline="dnase-seq-pipeline",
             version="v3.0.0", mapq=10)
    _write_rep(d, 1, 4, 500, "d", dnase=True)
    d = work / "ENCSRMIX" / "se"
    d.mkdir(parents=True)
    (d / "NOT_APPLICABLE").write_text(json.dumps({"reason": "mixed run types"}))
    # native row: a staged redacted BAM
    (fq / "ENCSRNAT").mkdir(parents=True)
    good, dups, low = _exp_reads(5, 350, "n")
    _write_ta(fq / "ENCSRNAT" / "ENCFF989MMP.bam", good + dups + low)
    (fq / "ENCSRNAT" / "bams.json").write_text(json.dumps({"bams": [{
        "accession": "ENCFF989MMP", "bio_rep": 1, "md5": "x",
        "bytes": (fq / "ENCSRNAT" / "ENCFF989MMP.bam").stat().st_size,
        "mapped_read_length": 36, "analysis_step_version": "/v/bwa-alignment-step-v-1-0/"}]}))

    chp = ["ENCSRCHP.pooled.uniform.base", "ENCSRCHP.br1.uniform.base", "ENCSRCHP.br3.uniform.base",
           "ENCSRCHP.pooled.uniform.depth0.50", "ENCSRCHP.pooled.uniform.nodedup",
           "ENCSRCHP.pooled.uniform.mapq0"]
    rows = [
        dict(row=1, experiment="ENCSRCTL", eic_track="C01CTL", role="val",
             assay="chipseq-control", assay_class="control", unit="pooled", arm="base",
             pipeline="chip-seq-pipeline2 v2.2.2", keep_bam=0,
             expected_measurements=_exp(["ENCSRCTL.pooled.uniform.base"])),
        dict(row=2, experiment="ENCSRCHP", eic_track="C01M16", role="val", assay="H3K27ac",
             assay_class="narrow_histone", unit="pooled", arm="base", control_experiment="ENCSRCTL",
             pipeline="chip-seq-pipeline2 v2.2.2", keep_bam=1,
             expected_measurements=_exp(chp).replace(
                 "ENCSRCHP.pooled.uniform.base:counts+pr1+pr2",
                 "ENCSRCHP.pooled.uniform.base:counts+pr1+pr2+signal+pr1_signal+pr2_signal")),
        dict(row=3, experiment="ENCSRCHP", eic_track="C01M16", role="val", assay="H3K27ac",
             assay_class="narrow_histone", unit="pooled", arm="se", control_experiment="ENCSRCTL",
             pipeline="chip-seq-pipeline2 v2.2.2", keep_bam=0,
             expected_measurements=_exp(["ENCSRCHP.pooled.uniform.se"])),
        dict(row=4, experiment="ENCSRCHP", eic_track="C01M16", role="val", assay="H3K27ac",
             assay_class="narrow_histone", unit="br1tr1ln1", arm="base",
             pipeline="chip-seq-pipeline2 v2.2.2", keep_bam=0,
             expected_measurements=_exp(["ENCSRCHP.br1tr1ln1.uniform.base"])),
        dict(row=5, experiment="ENCSRDNS", eic_track="C02M02", role="train", assay="DNase-seq",
             assay_class="accessibility_dnase", unit="pooled", arm="base",
             pipeline="dnase-seq-pipeline v3.0.0", keep_bam=0,
             expected_measurements=_exp([f"ENCSRDNS.pooled.uniform.{a}" for a in
                                         ("base", "depth0.50", "nodedup", "mapq0")])),
        dict(row=6, experiment="ENCSRNAT", eic_track="C04M22", role="train", assay="H3K4me3",
             assay_class="narrow_histone", unit="pooled", arm="base", fastq_files="ENCFF989MMP",
             pipeline="native-redacted-bam", keep_bam=0,
             expected_measurements=_exp([f"ENCSRNAT.pooled.native.{a}" for a in
                                         ("base", "nodedup", "mapq0")])),
        dict(row=7, experiment="ENCSRMIX", eic_track="C31M25", role="val", assay="H3K79me2",
             assay_class="broad_histone", unit="pooled", arm="se",
             pipeline="chip-seq-pipeline2 v2.2.2", keep_bam=0,
             expected_measurements=_exp(["ENCSRMIX.pooled.uniform.se"])),
    ]
    bl = _bl(tmp_path / "build_list.tsv", rows)

    def run(row, *extra):     # --tmp need not exist (build_row.sh passes $SLURM_TMPDIR/views)
        return AV.main(["run", "--build-list", str(bl), "--row", str(row), "--work", str(work),
                        "--out", str(out), "--fastq-root", str(fq), "--chrom-sizes", str(sz),
                        "--tmp", str(tmp_path / f"tmp{row}" / "views"), "--name-tokens", "--nth", "1",
                        *extra])
    return {"run": run, "out": out, "work": work, "chp": chp, "bl": bl}


def _meta_of(out, mid):
    return json.loads((out / mid / "meta.json").read_text())


def test_driver_builds_exactly_the_rows_measurements(driver, capsys, monkeypatch):
    run, out = driver["run"], driver["out"]
    # the ChIP validation row asks for signal; its control is not built yet -> exit 75, every
    # other measurement written, the pooled base (the signal carrier) held back
    assert run(2) == 75
    got = {p.name for p in out.iterdir() if not p.name.startswith(".")}
    assert got == set(driver["chp"]) - {"ENCSRCHP.pooled.uniform.base"}
    assert run(1) == 0
    fake = {"signal.h5": None}

    def fake_signal(cfg, half_tas, cost, sources=None):
        assert cfg["signal_spec"]["controls"] == ["ENCSRCTL"]
        assert AV.control_readsets(cfg["work"], "ENCSRCTL", cfg["out"])   # now built
        z = {c: np.zeros(n // 25, np.float32) for c, n in SIZES.items()}
        fake["signal.h5"] = half_tas
        return {"signal.h5": z, "pr1_signal.h5": z, "pr2_signal.h5": z, "_prov": {"fake": 1}}
    monkeypatch.setattr(AV, "build_signal", fake_signal)
    assert run(2) == 0
    assert len(fake["signal.h5"]["pr1"]) == 2              # both replicates' halves
    assert _meta_of(out, "ENCSRCHP.pooled.uniform.base")["has_signal"] == "1"
    assert AV.control_readsets(driver["work"], "ENCSRCTL", out)
    for r in (3, 4, 5, 6, 7):
        assert run(r) == 0
    names = {p.name for p in out.iterdir() if not p.name.startswith(".")}
    assert names == set(driver["chp"]) | {
        "ENCSRCTL.pooled.uniform.base", "ENCSRCHP.pooled.uniform.se",
        "ENCSRCHP.br1tr1ln1.uniform.base"} | {f"ENCSRDNS.pooled.uniform.{a}" for a in
                                              ("base", "depth0.50", "nodedup", "mapq0")} | {
        f"ENCSRNAT.pooled.native.{a}" for a in ("base", "nodedup", "mapq0")}
    assert not any(n.startswith("ENCSRMIX") for n in names)      # NOT_APPLICABLE row: nothing
    capsys.readouterr()
    # verify: halves sum, pooled = br1 + br3, nodedup >= base, covariates, and with name tokens:
    # no read of any pr1 (any unit, any arm, the se arm included) is in any pr2
    assert AV.verify(out) == 0
    txt = capsys.readouterr().out
    assert "all checks pass" in txt and txt.count("0 shared") == 4


def test_driver_covariates_per_pipeline(driver):
    run, out = driver["run"], driver["out"]
    for r in (1, 3, 4, 5, 6):
        assert run(r) == 0
    se = _meta_of(out, "ENCSRCHP.pooled.uniform.se")
    assert (se["arm_knob"], se["arm_value"], se["run_type"]) == ("se", "1", "single-ended")
    assert se["parent_measurement_id"] == "ENCSRCHP.pooled.uniform.base"
    assert se["control_measurement_id"] == "ENCSRCTL.pooled.uniform.base"
    assert se["platform"] == "HiSeq2000" and se["cell"] == "C01" and se["in_eic"] == "val"
    lane = _meta_of(out, "ENCSRCHP.br1tr1ln1.uniform.base")
    assert lane["unit"] == "br1tr1ln1" and lane["bio_rep"] == "1"
    dn = {a: _meta_of(out, f"ENCSRDNS.pooled.uniform.{a}") for a in ("base", "nodedup", "mapq0")}
    assert dn["base"]["mapq_threshold"] == "10" and dn["base"]["pipeline"] == "dnase-seq-pipeline"
    assert (dn["nodedup"]["dedup"], dn["mapq0"]["mapq_threshold"]) == ("0", "0")
    # DNase nodedup keeps the flagged duplicates: more reads than the base
    assert int(dn["nodedup"]["depth_reads"]) > int(dn["base"]["depth_reads"])
    assert dn["base"]["has_signal"] == "0"                 # a training row asks for no signal
    nat = _meta_of(out, "ENCSRNAT.pooled.native.base")
    assert (nat["layer"], nat["pipeline"], nat["pipeline_version"], nat["run_type"]) == \
        ("native", "native-redacted-bam", "bwa-alignment-step-v-1-0", "single-ended")
    assert int(_meta_of(out, "ENCSRNAT.pooled.native.mapq0")["depth_reads"]) > \
        int(nat["depth_reads"])


def test_driver_refuses_a_named_tagalign_that_differs_from_the_pipelines(driver):
    d = driver["work"] / "ENCSRCTL" / "base"
    rows = _read_ta(d / "rep1.tagAlign.gz")
    _write_ta(d / "rep1.tagAlign.gz", [(r[0], int(r[1]), int(r[2]), r[3], r[5])
                                       for r in rows[:-1]])
    with pytest.raises(SystemExit, match="pipeline's"):
        driver["run"](1)


def test_driver_signal_switch(driver, monkeypatch):
    calls = []
    monkeypatch.setattr(AV, "build_signal", lambda cfg, h, cost, sources=None: calls.append(1) or {
        n: {c: np.zeros(k // 25, np.float32) for c, k in SIZES.items()}
        for n in ("signal.h5", "pr1_signal.h5", "pr2_signal.h5")} | {"_prov": {}})
    assert driver["run"](5, "--signal", "off") == 0 and not calls     # auto: no signal asked
    shutil.rmtree(driver["out"])
    assert driver["run"](5, "--signal", "on") == 0 and calls == [1]


def test_verify_catches_a_wrong_arm_covariate(driver, capsys):
    assert driver["run"](5) == 0
    m = driver["out"] / "ENCSRDNS.pooled.uniform.nodedup" / "meta.json"
    d = json.loads(m.read_text())
    d["dedup"] = "1"
    m.write_text(json.dumps(d))
    assert AV.verify(driver["out"]) == 1
    assert "dedup" in capsys.readouterr().out


def test_drop_bams_after_views(driver):
    d = driver["work"] / "ENCSRCTL" / "base"
    lines = ["name\tsrc\tbytes\tmd5"] + [
        f"{n}\ts\t{(d / n).stat().st_size}\tm" for n in
        ("rep1.unfiltered.bam", "rep1.filtered.bam", "rep1.tagAlign.gz")]
    (d / "HARVEST.tsv").write_text("\n".join(lines) + "\n")
    assert driver["run"](1, "--drop-bams") == 0           # keep_bam = 0
    assert not (d / "rep1.filtered.bam").exists() and (d / "rep1.tagAlign.gz").exists()
    assert "rep1.unfiltered.bam" in (d / "DROPPED.tsv").read_text()


def _readset(work, exp, unit, layer="uniform"):
    d = work / exp / f"{unit}.{layer}" / "base"
    d.mkdir(parents=True, exist_ok=True)
    for h in ("pr1", "pr2"):
        (d / f"{h}.tagAlign.gz").write_bytes(b"")
    (d / "READSET.json").write_text("{}")


def test_partly_built_control_is_pending(tmp_path):
    """KF2 B1: a 2-replicate control with only br1's read set is not built; the ChIP row's
    signal must wait (ControlPending, exit 75), never run on br1 alone."""
    work, out = tmp_path / "work", tmp_path / "meas"
    (work / "CTL2" / "base").mkdir(parents=True)
    (work / "CTL2" / "base" / "arm.json").write_text(json.dumps(
        {"reps": [{"rep": 1, "bio_rep": 1}, {"rep": 2, "bio_rep": 2}]}))
    _readset(work, "CTL2", "br1")
    assert AV.control_readsets(work, "CTL2", out, "uniform") == {}
    cfg = {"work": str(work), "out": str(out), "experiment": "CHP", "tmp": str(tmp_path / "t"),
           "control_layers": {"CTL2": "uniform"}}
    with pytest.raises(AV.ControlPending):
        AV.chip_signal_inputs(cfg, AV.Cost(), "x", {"controls": ["CTL2"]})
    _readset(work, "CTL2", "br2")                  # every replicate of arm.json: built
    assert sorted(AV.control_readsets(work, "CTL2", out, "uniform")) == ["br1", "br2"]
    # no run record and no pooled measurement: pending; the pooled base measurement settles it
    (work / "CTL2" / "base" / "arm.json").unlink()
    assert AV.control_readsets(work, "CTL2", out, "uniform") == {}
    (out / "CTL2.pooled.uniform.base").mkdir(parents=True)
    (out / "CTL2.pooled.uniform.base" / "provenance.json").write_text("{}")
    assert sorted(AV.control_readsets(work, "CTL2", out, "uniform")) == ["br1", "br2"]
    # a native control: bams.json in the staging dir lists the replicates; layers never mix
    fq = tmp_path / "fq"
    (fq / "NAT").mkdir(parents=True)
    (fq / "NAT" / "bams.json").write_text(json.dumps({"bams": [{"bio_rep": 1}, {"bio_rep": 2}]}))
    _readset(work, "NAT", "br1", "native")
    assert AV.control_readsets(work, "NAT", out, "native", fq) == {}
    _readset(work, "NAT", "br2", "native")
    assert sorted(AV.control_readsets(work, "NAT", out, "native", fq)) == ["br1", "br2"]
    assert AV.control_readsets(work, "NAT", out, "uniform", fq) == {}


def test_bam_drop_waits_for_every_row_reading_the_run_dir(driver, capsys):
    """KF2 M3: BAMs of a keep_bam = 0 row go only when no row that reads its run dir lacks a
    measurement. Arm rows (se) and lane rows read their own run dirs, so they never block."""
    import csv as _csv
    d = driver["work"] / "ENCSRCTL" / "base"
    lines = ["name\tsrc\tbytes\tmd5"] + [
        f"{n}\ts\t{(d / n).stat().st_size}\tm" for n in
        ("rep1.unfiltered.bam", "rep1.filtered.bam", "rep1.tagAlign.gz")]
    (d / "HARVEST.tsv").write_text("\n".join(lines) + "\n")
    bl = driver["bl"]
    with open(bl) as fh:
        rows = list(_csv.DictReader(fh, delimiter="\t"))
    # a later-priority row that reads the same run dir (same experiment, unit, arm) and is unbuilt
    extra = dict(rows[0], row="99", priority="4",
                 expected_measurements="ENCSRCTL.pooled.uniform.depth0.50:counts+pr1+pr2")
    _bl(bl, [{k: r[k] for k in BL_COLS} for r in rows] + [extra])
    assert driver["run"](1, "--drop-bams") == 0
    assert (d / "rep1.filtered.bam").exists() and not (d / "DROPPED.tsv").exists()
    assert "rows ['99'] still read them" in capsys.readouterr().out
    _bl(bl, [{k: r[k] for k in BL_COLS} for r in rows])     # only arm/lane rows elsewhere
    assert driver["run"](1, "--drop-bams") == 0
    assert not (d / "rep1.filtered.bam").exists() and (d / "rep1.tagAlign.gz").exists()
    # the ChIP experiment's se arm row and lane row have their own run dirs
    FU = AV._fu()
    chp = [r for r in rows if r["experiment"] == "ENCSRCHP"]
    assert len({FU.arm_dir(driver["work"], r) for r in chp}) == len(chp)


@pytest.mark.skipif(not (shutil.which("samtools") and shutil.which("bedtools")),
                    reason="needs samtools and bedtools on PATH")
def test_named_tagalign_commands_on_real_bams(tmp_path):
    """SE, PE (mates together, Tn5 shift for ATAC) and DNase (flag 0x400 dropped) on real BAMs."""
    hdr = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:2000\n"
    se = hdr + "".join(f"q{i}\t{16 if i % 2 else 0}\tchr1\t{100 + 10 * i}\t60\t36M\t*\t0\t0\t"
                       f"{'A' * 36}\t{'I' * 36}\n" for i in range(6))
    se += f"qd\t1024\tchr1\t500\t60\t36M\t*\t0\t0\t{'A' * 36}\t{'I' * 36}\n"
    pe = hdr + "".join(
        f"p{i}\t99\tchr1\t{100 + 20 * i}\t60\t36M\t=\t{300 + 20 * i}\t236\t{'A' * 36}\t{'I' * 36}\n"
        f"p{i}\t147\tchr1\t{300 + 20 * i}\t60\t36M\t=\t{100 + 20 * i}\t-236\t{'A' * 36}\t{'I' * 36}\n"
        for i in range(4))
    bams = {}
    for name, sam in (("se", se), ("pe", pe)):
        (tmp_path / f"{name}.sam").write_text(sam)
        subprocess.run(f"samtools view -b -o {tmp_path / name}.bam {tmp_path / name}.sam",
                       shell=True, check=True)
        bams[name] = tmp_path / f"{name}.bam"

    def run(cmd):
        subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)

    out = tmp_path / "o.gz"
    run(AV.named_ta_command(bams["se"], out, paired_end=False, tmp=tmp_path, nth=1))
    r = _read_ta(out)
    assert len(r) == 7 and r[0][3] == "q0" and r[0][4] == "1000"
    run(AV.named_ta_command(bams["se"], out, paired_end=False, tmp=tmp_path, nth=1,
                            pipeline="dnase", drop_dups=True))
    assert [x[3] for x in _read_ta(out)] == [f"q{i}" for i in range(6)]
    run(AV.named_ta_command(bams["pe"], out, paired_end=True, tmp=tmp_path, nth=1))
    r = _read_ta(out)
    assert len(r) == 8 and all(r[i][3] == r[i + 1][3] for i in range(0, 8, 2))
    plain = {(x[3], x[5]): (int(x[1]), int(x[2])) for x in r}
    run(AV.named_ta_command(bams["pe"], out, paired_end=True, tmp=tmp_path, nth=1,
                            pipeline="atac"))
    for x in _read_ta(out):
        s, e = plain[(x[3], x[5])]
        assert (int(x[1]), int(x[2])) == ((s + 4, e) if x[5] == "+" else (s, e - 5))
    # bamtobed without -bedpe names mates q/1, q/2: the split key is the same fragment
    salt = AV.split_salt(0, "E")
    assert AV.name_half(b"p1/1", salt) == AV.name_half(b"p1/2", salt)


def test_filter_output_name_follows_filter_chrs(tmp_path):
    for n in ("rep1.unfiltered.dupmark.bam", "rep1.unfiltered.nodup.no_chrM_MT.bam",
              "rep1.unfiltered.filt.no_chrM_MT.bam", "x.nodup.bam"):
        (tmp_path / n).write_bytes(b"")
    assert AV.filter_output(tmp_path, "base", ["chrM", "MT"]).name == \
        "rep1.unfiltered.nodup.no_chrM_MT.bam"            # atac
    assert AV.filter_output(tmp_path, "nodedup", ["chrM", "MT"]).name == \
        "rep1.unfiltered.filt.no_chrM_MT.bam"
    assert AV.filter_output(tmp_path, "mapq0", []).name == "x.nodup.bam"   # chip
