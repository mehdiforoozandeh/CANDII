"""t171 (chunk DT): importing the counterfactual testbed as `testbed` measurements, on a tiny
synthetic testbed tree (CPU, seconds). The tree mimics Nibi `t112_cf/{products,pseudoreps}` and
the half tagAligns under a fake scratch; each half's counts come from its own tagAlign, so the
chr21 recount and the "halves sum to base" checks are real."""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tools" / "t112"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import import_testbed as IT  # noqa: E402
import bin25  # noqa: E402

SIZES = {"chr1": 4000, "chr21": 2000, "chrX": 1000}  # bp; grid floor(len / 25)
GENOME = {**SIZES, "chrY": 1000}                      # the genome layer also holds chrY


def _ta(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for c, s, e in rows:
            f.write(f"{c}\t{s}\t{e}\tN\t1000\t+\n")
    return str(path)


def _reads(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = ["chr1", "chr21", "chrX"][int(rng.integers(0, 3))]
        s = int(rng.integers(0, SIZES[c] - 60))
        out.append((c, s, s + 36))
    return out


def _md5(p):
    import records
    return records.md5_file(p)


def _product(root: Path, scratch: Path, pid: str, cov: dict, halves_reads: dict,
             pipeline: dict, paired: bool):
    pdir = root / "products" / pid
    pdir.mkdir(parents=True)
    counts, half_counts = None, {}
    for h, reads in halves_reads.items():
        ta = _ta(scratch / "_readsets" / pid / f"x.{h}.tagAlign.gz", reads)
        half_counts[h] = bin25.counts_from_tagalign(ta, SIZES)
        hd = root / "pseudoreps" / pid / h
        hd.mkdir(parents=True)
        bin25.write_npz(hd / "counts25.npz", half_counts[h])
        bin25.write_npz(hd / "pval25.npz", {c: v.astype(np.float32) * 0.5
                                            for c, v in half_counts[h].items()})
        (hd / "provenance.json").write_text(json.dumps({
            "schema": 1, "pid": pid, "half": h, "readset": pid, "pval_n_nan": 0,
            "pipeline": pipeline,
            "split": {"tagalign": ta, "tagalign_md5": _md5(ta), "n_lines": len(reads),
                      "seed": {"arg": 0, "paired_end": paired}}}))
    counts = {c: (half_counts["pr1"][c].astype(np.uint64) + half_counts["pr2"][c])
              .astype(np.uint32) for c in SIZES}
    bin25.write_npz(pdir / "counts25.npz", counts)
    bin25.write_npz(pdir / "pval25.npz", {c: v.astype(np.float32) for c, v in counts.items()})
    (pdir / "covariates.json").write_text(json.dumps(cov))
    (pdir / "provenance.json").write_text(json.dumps({"pid": pid}))


CHIP = {"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.2.2", "sif": "/x.sif",
        "sif_md5": "0" * 32, "sif_sha256": "a" * 64}
ATAC = {"repo": "ENCODE-DCC/atac-seq-pipeline", "release": "v2.2.3", "sif": "/y.sif",
        "sif_md5": "1" * 32, "sif_sha256": "b" * 64}


def _cov(pid, cell, assay, depth, run_type="single-ended", read_length=101, fraglen=180):
    track, arm, level = pid.split("__")
    return {"schema": 1, "pid": pid, "track": track, "cell": cell, "assay": assay, "arm": arm,
            "level": level, "depth": depth, "read_length": read_length, "run_type": run_type,
            "fraglen": fraglen}


@pytest.fixture()
def tb(tmp_path):
    """A 2-track testbed: a ChIP track (base, depth, pe, dedup, crop, mapq, abproxy, two
    signal-only arms) and a DNase track (base, depth, a mapq arm identical to the base)."""
    root, scratch = tmp_path / "t112_cf", tmp_path / "scratch"
    base = _reads(400, 1)
    b1, b2 = base[:200], base[200:]
    T = "C19M16"
    _product(root, scratch, f"{T}__base__base", _cov(f"{T}__base__base", "C19", "H3K27ac", 400),
             {"pr1": b1, "pr2": b2}, CHIP, False)
    sub = base[::2]  # a subsample of the base reads, split independently
    _product(root, scratch, f"{T}__depth__15M", _cov(f"{T}__depth__15M", "C19", "H3K27ac", 200),
             {"pr1": sub[1::2], "pr2": sub[0::2]}, CHIP, False)
    pe = _reads(200, 2)
    pe_lines = [x for r in pe for x in (r, (r[0], r[1] + 100, r[2] + 100))]
    _product(root, scratch, f"{T}__pe__pe",
             _cov(f"{T}__pe__pe", "C19", "H3K27ac", 400, run_type="paired-ended"),
             {"pr1": pe_lines[:200], "pr2": pe_lines[200:]}, CHIP, True)
    for pid, seed, rl in ((f"{T}__dedup__off", 3, 101), (f"{T}__crop__36", 4, 36),
                          (f"{T}__mapq__10", 5, 101), (f"{T}__abproxy__f0.5", 6, 101)):
        r = _reads(400, seed)
        _product(root, scratch, pid, _cov(pid, "C19", "H3K27ac", 400, read_length=rl),
                 {"pr1": r[:200], "pr2": r[200:]}, CHIP, False)
    for pid in (f"{T}__ratio__k2", f"{T}__extsize__k0.5"):
        _product(root, scratch, pid, _cov(pid, "C19", "H3K27ac", 400), {"pr1": b1, "pr2": b2},
                 CHIP, False)
    D = "C12M02"
    dn = _reads(500, 7)
    _product(root, scratch, f"{D}__base__base",
             _cov(f"{D}__base__base", "C12", "DNase-seq", 500, read_length=76, fraglen=150),
             {"pr1": dn[:250], "pr2": dn[250:]}, ATAC, False)
    _product(root, scratch, f"{D}__mapq__0",
             _cov(f"{D}__mapq__0", "C12", "DNase-seq", 500, read_length=76, fraglen=150),
             {"pr1": dn[:250], "pr2": dn[250:]}, ATAC, False)
    _product(root, scratch, f"{D}__depth__15M",
             _cov(f"{D}__depth__15M", "C12", "DNase-seq", 150, read_length=76, fraglen=150),
             {"pr1": dn[:150:2], "pr2": dn[1:150:2]}, ATAC, False)
    (root / "products" / "MANIFEST.tsv").write_text("pid\n")
    ledger = tmp_path / "ledger.tsv"
    with open(ledger, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["eic_track", "role", "cell", "assay", "assay_class", "experiment", "match",
                    "platform", "lab"])
        w.writerow([T, "blind", "HAP-1", "H3K27ac", "narrow_histone", "ENCSR131DVD", "1",
                    "Illumina HiSeq 4000", "Michael Snyder, Stanford"])
        w.writerow([D, "blind", "DND-41", "DNase-seq", "accessibility_dnase", "ENCSR903SKE", "1",
                    "Illumina HiSeq 4000", "John Stamatoyannopoulos, UW"])
    gs = tmp_path / "chrom_sizes.json"
    gs.write_text(json.dumps({"chrom_sizes": GENOME}))
    return {"root": root, "scratch": scratch, "ledger": ledger, "sizes": gs,
            "out": tmp_path / "measurements", "rows": tmp_path / "eic" / "testbed_rows.tsv",
            "copy": tmp_path / "project_copy"}


def _run(tb, workers=1):
    return IT.main(["run", "--testbed", str(tb["root"]), "--ledger", str(tb["ledger"]),
                    "--out", str(tb["out"]), "--rows", str(tb["rows"]),
                    "--halves-copy", str(tb["copy"]), "--chrom-sizes", str(tb["sizes"]),
                    "--workers", str(workers)])


def _rows(tb):
    return {r["measurement_id"]: r for r in IT.read_tsv(tb["rows"])}


def test_arm_spec_names():
    cov = {"depth": 15000000}
    assert IT.arm_spec("depth", "15M", cov, 30000000)["arm"] == "depth0.50"
    assert IT.arm_spec("depth", "3.75M", {"depth": 3750000}, 30000000)["arm"] == "depth0.125"
    assert IT.arm_spec("depth", "15M", cov, 50000000)["arm"] == "depth0.30"
    assert IT.arm_spec("dedup", "off", cov, 1)["arm"] == "nodedup"
    assert IT.arm_spec("crop", "36", cov, 1)["arm"] == "crop36"
    assert IT.arm_spec("mapq", "0", cov, 1)["arm"] == "mapq0"
    assert IT.arm_spec("abproxy", "f0.9", cov, 1)["arm"] == "abproxy0.9"
    assert IT.arm_spec("pe", "pe", cov, 1)["arm_knob"] == "se"
    with pytest.raises(ValueError):
        IT.arm_spec("ratio", "k2", cov, 1)
    with pytest.raises(ValueError):
        IT.parse_pid("C19M16__base")


def test_plan_drops_identical_counts(tb):
    kept, dropped = IT.plan(tb["root"], tb["ledger"])
    assert {x["pid"] for x in dropped} == {"C19M16__ratio__k2", "C19M16__extsize__k0.5",
                                           "C12M02__mapq__0"}
    assert len(kept) == 9
    assert all("byte-identical" in x["reason"] for x in dropped)


def test_signal_only_arm_with_new_counts_is_an_error(tb):
    r = _reads(400, 99)
    d = tb["root"] / "products" / "C19M16__ctlid__other"
    _product(tb["root"], tb["scratch"], "C19M16__ctlid__other",
             _cov("C19M16__ctlid__other", "C19", "H3K27ac", 400), {"pr1": r[:200], "pr2": r[200:]},
             CHIP, False)
    assert d.is_dir()
    with pytest.raises(SystemExit, match="signal-only"):
        IT.plan(tb["root"], tb["ledger"])


def test_run_then_verify(tb, capsys):
    assert _run(tb) == 0
    rows = _rows(tb)
    assert len(rows) == 9
    with open(tb["rows"]) as fh:
        assert tuple(fh.readline().rstrip("\n").split("\t")) == IT.row_columns()
    b = rows["ENCSR131DVD.pooled.testbed.base"]
    assert (b["layer"], b["unit"], b["in_eic"], b["use"]) == ("testbed", "pooled", "blind",
                                                             IT.USE_BLIND)
    assert b["mapq_threshold"] == "30" and b["dedup"] == "1" and b["frag_len"] == "180"
    dep = rows["ENCSR131DVD.pooled.testbed.depth0.50"]
    assert dep["reads_subset_of"] == b["measurement_id"] and dep["arm_knob"] == "depth"
    pe = rows["ENCSR131DVD.pooled.testbed.pe"]
    assert pe["depth_reads"] == "400" and pe["depth_fragments"] == "200"
    assert rows["ENCSR131DVD.pooled.testbed.nodedup"]["dedup"] == "0"
    assert rows["ENCSR131DVD.pooled.testbed.mapq10"]["mapq_threshold"] == "10"
    assert rows["ENCSR131DVD.pooled.testbed.crop36"]["read_length"] == "36"
    assert rows["ENCSR131DVD.pooled.testbed.abproxy0.5"]["reads_subset_of"] == ""
    dn = rows["ENCSR903SKE.pooled.testbed.depth0.30"]
    assert dn["assay_class"] == "accessibility_dnase" and dn["mapq_threshold"] == ""
    assert dn["pipeline"] == "atac-seq-pipeline" and dn["has_signal"] == "1"
    # meta.json holds exactly the measurements.tsv columns; provenance points at the /project copy
    d = tb["out"] / "ENCSR131DVD.pooled.testbed.base"
    meta = json.loads((d / "meta.json").read_text())
    assert tuple(meta) == IT.measurement_columns()
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["halves"]["pr1"]["tagalign"][0].startswith(str(tb["copy"]))
    with h5py.File(d / "counts.h5") as f:
        assert set(f) == set(SIZES) and f["chr1"].dtype == np.uint32
        assert f.attrs["depth_reads"] == 400
    # a plain strict read of the rows table refuses it; the projector refuses blind rows
    from minicandi.dataset import read_tsv
    with pytest.raises(ValueError):
        read_tsv(tb["rows"], IT.measurement_columns())
    with pytest.raises(ValueError, match="evaluation use only"):
        IT.rows_for_tables(tb["rows"])
    assert len(IT.rows_for_tables(tb["rows"], allow_blind=True)) == 9
    capsys.readouterr()
    assert IT.main(["verify", "--rows", str(tb["rows"]), "--recount", "all",
                    "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "measurements written: 9" in out
    assert "halves sum to base: all" in out
    assert "dropped products: 3" in out and "C12M02__mapq__0" in out
    assert "all checks pass" in out
    # rerun skips existing measurements and rewrites identical tables
    before = tb["rows"].read_text()
    assert _run(tb) == 0
    assert tb["rows"].read_text() == before


def test_verify_catches_a_corrupt_half(tb, capsys):
    assert _run(tb) == 0
    d = tb["out"] / "ENCSR131DVD.pooled.testbed.mapq10"
    with h5py.File(d / "pr1.h5", "a") as f:
        v = f["chr21"][()]
        v[3] += 1
        f["chr21"][...] = v
    capsys.readouterr()
    assert IT.main(["verify", "--rows", str(tb["rows"]), "--recount", "none",
                    "--workers", "1"]) == 1
    out = capsys.readouterr().out
    assert "halves sum to base: 8 of 9" in out
    assert "FAIL ENCSR131DVD.pooled.testbed.mapq10" in out


def test_halves_copy_survives_scratch_purge(tb, capsys):
    assert _run(tb) == 0
    import shutil
    shutil.rmtree(tb["scratch"])
    capsys.readouterr()
    assert IT.main(["verify", "--rows", str(tb["rows"]), "--recount", "all",
                    "--workers", "1"]) == 0
