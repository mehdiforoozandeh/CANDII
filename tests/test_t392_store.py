"""t392 chunk C1, case (2): the read-start store that `read_starts.py run` writes.

Written against `cruxvault/INSTRUCTIONS/t392_read_starts/INSTRUCTIONS.md`, decisions D9 and D11,
and the Interfaces lines of plan chunk C3, before `tools/onion_data/read_starts.py` existed.

The synthetic measurement imitates one `eic_v1_2` measurement folder: `counts.h5`, `pr1.h5`,
`pr2.h5` (overlap counts made by `tools/t112/bin25.py::counts_from_tagalign`) and a
`provenance.json` whose `halves` lists the read files of each half. Assumed (not pinned by the
plan): `halves` is `{"pr1": [tagAlign paths], "pr2": [...]}` with absolute paths, the grid of a
chromosome is the shape of the reference dataset, and `depth_reads` is an attribute of
`counts.h5`. Chromosome lengths here are multiples of 25, so the grid is the same either way.
"""
from __future__ import annotations

import gzip
import hashlib
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

RES = 25
SIZES = {"chr1": 3000, "chr2": 1500}          # bp, multiples of 25: 120 and 60 bins
FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
ATTRS = ("schema", "counting_rule", "resolution", "strand_columns", "measurement_id",
         "depth_reads", "n_reads_strand", "n_off_grid", "kernel_counts", "kernel",
         "conversion_floor_nmse")


def _rs():
    from onion_data import read_starts
    return read_starts


def _cli(argv) -> int:
    try:
        rc = _rs().main(argv)
    except SystemExit as e:
        rc = e.code
    return int(rc or 0)


def _random_reads(rng, n):
    reads = []
    for _ in range(n):
        c = "chr1" if rng.random() < 0.6 else "chr2"
        s = int(rng.integers(0, SIZES[c] - 160))
        reads.append((c, s, s + int(rng.integers(20, 150)), "+" if rng.random() < .5 else "-"))
    return reads


def _write_tagalign(path: Path, reads) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for chrom, s, e, strand in reads:
            fh.write(f"{chrom}\t{s}\t{e}\tN\t1000\t{strand}\n")
    return path


def _write_counts(path: Path, arrays: dict, mid: str, depth: float) -> None:
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint32))
        f.attrs.update({"schema": 1, "measurement_id": mid, "depth_reads": depth,
                        "resolution": RES, "counting_rule": "overlap"})


def make_measurement(root: Path, mid: str, halves: dict) -> dict:
    """halves: {"pr1": reads, "pr2": reads}. Writes the eic_v1_2-like folder and the tagAligns."""
    import bin25
    work = root / "work" / mid
    mdir = root / "measurements" / mid
    mdir.mkdir(parents=True, exist_ok=True)
    tas, ov = {}, {}
    for h, reads in halves.items():
        tas[h] = _write_tagalign(work / f"{h}.tagAlign.gz", reads)
        ov[h] = bin25.counts_from_tagalign(str(tas[h]), SIZES)
    depth = float(sum(len(r) for r in halves.values()))
    _write_counts(mdir / "counts.h5", {c: ov["pr1"][c] + ov["pr2"][c] for c in SIZES}, mid, depth)
    _write_counts(mdir / "pr1.h5", ov["pr1"], mid, float(len(halves["pr1"])))
    _write_counts(mdir / "pr2.h5", ov["pr2"], mid, float(len(halves["pr2"])))
    (mdir / "provenance.json").write_text(json.dumps(
        {"measurement_id": mid, "halves": {h: [str(p)] for h, p in tas.items()}}))
    return {"mdir": mdir, "work": root / "work", "measurements": root / "measurements",
            "halves": halves, "overlap": ov}


def _expected_starts(reads):
    out = {c: np.zeros((SIZES[c] // RES, 2), np.int64) for c in SIZES}
    for c, s, e, strand in reads:
        base = s if strand == "+" else e - 1
        out[c][base // RES, 0 if strand == "+" else 1] += 1
    return out


def _run(m, mid, out) -> int:
    return _cli(["run", "--measurement", mid, "--measurements", str(m["measurements"]),
                 "--work-root", str(m["work"]), "--out", str(out)])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("rs_store")
    rng = np.random.default_rng(1)
    mid = "ENCSR000AAA.pooled.uniform"
    m = make_measurement(root, mid, {"pr1": _random_reads(rng, 600),
                                     "pr2": _random_reads(rng, 500)})
    out = root / "store"
    rc = _run(m, mid, out)
    return {**m, "mid": mid, "out": out, "dir": out / mid, "rc": rc}


def _read(path, chrom):
    with h5py.File(path, "r") as f:
        return np.asarray(f[chrom])


# --------------------------------------------------------------------------- layout

def test_run_exits_zero_and_writes_the_three_files(built):
    assert built["rc"] == 0
    for name in FILES.values():
        assert (built["dir"] / name).is_file(), name
    assert (built["dir"] / "provenance.json").is_file()


def test_one_dataset_per_chromosome_of_shape_bins_by_two(built):
    for name in FILES.values():
        with h5py.File(built["dir"] / name, "r") as f:
            assert set(f.keys()) == set(SIZES), name
            for c in SIZES:
                assert f[c].shape == (SIZES[c] // RES, 2), (name, c)
                assert f[c].dtype in (np.uint16, np.uint32)
                assert f[c].compression == "gzip" and f[c].shuffle


def test_column_zero_is_plus_and_column_one_is_minus(built):
    h = built["halves"]
    want = {"pr1": _expected_starts(h["pr1"]), "pr2": _expected_starts(h["pr2"])}
    want["counts"] = {c: want["pr1"][c] + want["pr2"][c] for c in SIZES}
    for which, name in FILES.items():
        for c in SIZES:
            np.testing.assert_array_equal(_read(built["dir"] / name, c).astype(np.int64),
                                          want[which][c], err_msg=f"{name} {c}")


def test_starts_equals_pr1_plus_pr2(built):
    d = built["dir"]
    for c in SIZES:
        np.testing.assert_array_equal(
            _read(d / FILES["counts"], c).astype(np.int64),
            _read(d / FILES["pr1"], c).astype(np.int64) + _read(d / FILES["pr2"], c).astype(np.int64))


def test_file_attributes(built):
    for which, name in FILES.items():
        with h5py.File(built["dir"] / name, "r") as f:
            a = dict(f.attrs)
        for k in ATTRS:
            assert k in a, (name, k)
        assert int(a["schema"]) == 1
        assert str(a["counting_rule"]) == "read_start"
        assert int(a["resolution"]) == 25
        assert str(a["strand_columns"]) == "plus,minus"
        assert str(a["measurement_id"]) == built["mid"]
        kc = np.asarray(a["kernel_counts"], np.float64)
        k = np.asarray(a["kernel"])
        assert kc.shape == (2, 17) and k.shape == (2, 17)
        assert k.dtype == np.float64
        nrs = np.asarray(a["n_reads_strand"], np.float64)
        np.testing.assert_allclose(k, kc / nrs[:, None], rtol=1e-12)
        assert int(a["n_off_grid"]) == 0
        assert np.isfinite(float(a["conversion_floor_nmse"]))


def test_n_reads_strand_equals_the_stored_start_totals(built):
    for name in FILES.values():
        with h5py.File(built["dir"] / name, "r") as f:
            nrs = np.asarray(f.attrs["n_reads_strand"], np.int64)
            tot = sum(np.asarray(f[c], np.int64).sum(axis=0) for c in SIZES)
        np.testing.assert_array_equal(nrs, tot)


def test_kernel_mass_times_reads_equals_the_overlap_sum(built):
    ref = {"counts": "counts.h5", "pr1": "pr1.h5", "pr2": "pr2.h5"}
    for which, name in FILES.items():
        with h5py.File(built["dir"] / name, "r") as f:
            k = np.asarray(f.attrs["kernel"], np.float64)
            nrs = np.asarray(f.attrs["n_reads_strand"], np.float64)
        overlap = sum(int(_read(built["mdir"] / ref[which], c).astype(np.int64).sum())
                      for c in SIZES)
        lhs = float((k.sum(axis=1) * nrs).sum())
        assert lhs == pytest.approx(overlap, rel=1e-9), name


def test_dnase_is_stored_with_both_strands(tmp_path):
    # D11: every assay keeps both strands in the store; the one-channel rule is a model rule
    rng = np.random.default_rng(4)
    mid = "ENCSR000DNS.pooled.uniform"
    reads = {"pr1": _random_reads(rng, 200), "pr2": _random_reads(rng, 200)}
    m = make_measurement(tmp_path, mid, reads)
    assert _run(m, mid, tmp_path / "store") == 0
    s = _read(tmp_path / "store" / mid / FILES["counts"], "chr1")
    assert s.shape[1] == 2 and s[:, 0].sum() > 0 and s[:, 1].sum() > 0


def test_a_bin_above_65535_is_stored_as_uint32(tmp_path):
    mid = "ENCSR000BIG.pooled.uniform"
    pile = [("chr1", 250, 280, "+")] * 66_000
    m = make_measurement(tmp_path, mid, {"pr1": pile, "pr2": [("chr2", 50, 90, "-")]})
    assert _run(m, mid, tmp_path / "store") == 0
    with h5py.File(tmp_path / "store" / mid / FILES["counts"], "r") as f:
        assert f["chr1"].dtype == np.uint32
        assert int(f["chr1"][10, 0]) == 66_000


# --------------------------------------------------------------------------- provenance and gate

def test_provenance_records_hashes_gates_and_cost(built):
    prov = json.loads((built["dir"] / "provenance.json").read_text())
    text = json.dumps(prov)
    for name in FILES.values():
        digest = hashlib.sha256((built["dir"] / name).read_bytes()).hexdigest()
        assert digest in text, name
    for ref in ("counts.h5", "pr1.h5", "pr2.h5"):
        assert ref in text
    assert text.count('"n_differing": 0') >= 3
    for key in ("conversion_floor_nmse", "wall_s", "peak_rss_kb"):
        assert key in text, key
    for p in built["halves"]:
        assert f"{p}.tagAlign.gz" in text


def test_one_changed_reference_bin_fails_the_run_after_writing(tmp_path):
    rng = np.random.default_rng(2)
    mid = "ENCSR000BAD.pooled.uniform"
    m = make_measurement(tmp_path, mid, {"pr1": _random_reads(rng, 300),
                                         "pr2": _random_reads(rng, 300)})
    with h5py.File(m["mdir"] / "pr1.h5", "r+") as f:
        v = np.asarray(f["chr2"])
        v[17] += 1
        f["chr2"][...] = v
    out = tmp_path / "store"
    assert _run(m, mid, out) != 0
    for name in FILES.values():
        assert (out / mid / name).is_file(), name
    prov = json.loads((out / mid / "provenance.json").read_text())
    assert '"n_differing": 1' in json.dumps(prov)


# --------------------------------------------------------------------------- verify, read_strands

def test_verify_passes_on_a_fresh_store(built):
    assert _cli(["verify", "--out", str(built["dir"])]) == 0


def test_verify_fails_when_starts_is_not_pr1_plus_pr2(tmp_path):
    rng = np.random.default_rng(6)
    mid = "ENCSR000VER.pooled.uniform"
    m = make_measurement(tmp_path, mid, {"pr1": _random_reads(rng, 200),
                                         "pr2": _random_reads(rng, 200)})
    out = tmp_path / "store"
    assert _run(m, mid, out) == 0
    with h5py.File(out / mid / FILES["counts"], "r+") as f:
        v = np.asarray(f["chr1"])
        v[3, 0] += 1
        f["chr1"][...] = v
    assert _cli(["verify", "--out", str(out / mid)]) != 0


def test_read_strands_returns_the_stored_slice_as_int32(built):
    p = built["dir"] / FILES["counts"]
    got = _rs().read_strands(str(p), "chr1", 10, 30)
    assert got.dtype == np.int32 and got.shape == (30, 2)
    np.testing.assert_array_equal(got, _read(p, "chr1")[10:40].astype(np.int32))
