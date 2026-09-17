"""t112 C11 — `tools/t112/export_store.py`: the product tree becomes a corpus the real store CLI builds.

End to end on tiny synthetic chrom sizes: two biosamples (a C19 ChIP pair with one control, a C12
DNase track with none) are exported, then `python -m candi.store build-biosample / build-manifest
(strict) / verify` run as subprocesses, exactly as `slurm/t112/build_store.sh` runs them.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from candi.store import layout as L
from candi.store.manifest import CSV_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("t112_export_store",
                                               ROOT / "tools" / "t112" / "export_store.py")
export_store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_store)

# chr21 has a remainder, chrX is small; chrEBV is not a main chromosome and must be ignored.
CHRSZ_TEXT = "chr21\t1013\nchrX\t260\nchrEBV\t500\n"
N_BINS = {"chr21": 1013 // 25, "chrX": 260 // 25}

MANIFEST_COLUMNS = ("pid biosample track cell assay arm level route knob knob_value depth read_length "
                    "run_type fraglen control_accession control_source pipeline_repo pipeline_release "
                    "sif_md5 counts25_md5 pval25_md5 control_counts25_md5 slurm_job_ids "
                    "product_dir").split()

# (pid, biosample, track, cell, assay, arm, level, depth, fraglen, control, control reads)
PRODUCTS = (
    ("C19M16__base__base", "CF_C19__base__base", "C19M16", "C19", "H3K27ac", "base", "base",
     30000000, 180, "ENCFF433TZR", 58073570),
    ("C19M22__base__base", "CF_C19__base__base", "C19M22", "C19", "H3K4me3", "base", "base",
     29999990, 205, "ENCFF433TZR", 58073570),
    ("C12M02__depth__15M", "CF_C12__depth__15M", "C12M02", "C12", "DNase-seq", "depth", "15M",
     15000000, 150, "", None),
)


def _manifest_row(p, products):
    pid, bios, track, cell, assay, arm, level, depth, fraglen, ctl, _ = p
    chip = assay != "DNase-seq"
    return {
        "pid": pid, "biosample": bios, "track": track, "cell": cell, "assay": assay, "arm": arm,
        "level": level, "route": "bam", "knob": "none" if arm == "base" else "treatment_reads",
        "knob_value": "null" if arm == "base" else "15000000", "depth": str(depth),
        "read_length": "101" if chip else "76", "run_type": "single-ended", "fraglen": str(fraglen),
        "control_accession": ctl, "control_source": "matched" if ctl else "",
        "pipeline_repo": "ENCODE-DCC/chip-seq-pipeline2" if chip else "ENCODE-DCC/atac-seq-pipeline",
        "pipeline_release": "v2.2.2" if chip else "v2.2.3",
        "sif_md5": "f6e408f7e77bafde4556883f33552191" if chip else "04d9fa482d67cf633845653750f58cef",
        "counts25_md5": "0" * 32, "pval25_md5": "1" * 32, "control_counts25_md5": "2" * 32 if ctl else "",
        "slurm_job_ids": "1,2", "product_dir": str(products / pid),
    }


def _make_products(tmp_path, rng):
    products = tmp_path / "products"
    arrays, ctl_arrays = {}, {}
    for p in PRODUCTS:
        pid, ctl, reads = p[0], p[9], p[10]
        d = products / pid
        d.mkdir(parents=True)
        counts = {c: rng.integers(0, 40, nb).astype(np.uint32) for c, nb in N_BINS.items()}
        pval = {c: (rng.random(nb) * 12).astype(np.float32) for c, nb in N_BINS.items()}
        np.savez_compressed(d / "counts25.npz", **counts)
        np.savez_compressed(d / "pval25.npz", **pval)
        arrays[pid] = {"counts": counts, "pval": pval}
        control = None
        if ctl:
            # one control per accession: both C19 tracks carry the same control, as in the real tree
            ctl_counts = ctl_arrays.setdefault(
                ctl, {c: rng.integers(0, 9, nb).astype(np.uint32) for c, nb in N_BINS.items()})
            np.savez_compressed(d / "control_counts25.npz", **ctl_counts)
            arrays[pid]["control"] = ctl_counts
            control = {"accession": ctl, "source": "matched", "reads": reads}
        (d / "covariates.json").write_text(json.dumps({"pid": pid, "control": control}))
    rows = [_manifest_row(p, products) for p in PRODUCTS]
    manifest = products / "MANIFEST.tsv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return products, manifest, arrays, rows


@pytest.fixture()
def setup(tmp_path):
    products, manifest, arrays, rows = _make_products(tmp_path, np.random.default_rng(112))
    chrsz = tmp_path / "chrom.sizes.tsv"
    chrsz.write_text(CHRSZ_TEXT)
    return {"tmp": tmp_path, "products": products, "manifest": manifest, "arrays": arrays,
            "rows": rows, "chrsz": chrsz}


def _store(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "candi.store", *map(str, args)],
                          capture_output=True, text=True, env=env)


def test_export_then_real_store_cli_builds_and_verifies(setup):
    t = setup
    src = t["tmp"] / "store_src"
    store = t["tmp"] / "CANDI_STORE"
    corpus = store / "cf"
    genome_json = store / "genome" / "chrom_sizes.json"
    assert export_store.main([str(x) for x in ["--products", t["products"], "--manifest", t["manifest"],
                              "--source-root", src, "--chrsz", t["chrsz"],
                              "--genome-json", genome_json]]) == 0
    assert json.loads(genome_json.read_text()) == {"chr21": 1013, "chrX": 260}

    with (src / "cf_metadata.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == tuple(CSV_COLUMNS)
        csv_rows = list(reader)
    assert sorted((r["biosample_name"], r["assay_name"]) for r in csv_rows) == [
        ("CF_C12__depth__15M", "DNase-seq"),
        ("CF_C19__base__base", "H3K27ac"), ("CF_C19__base__base", "H3K4me3"),
        ("CF_C19__base__base", "chipseq-control")]

    b = _store("build-biosample", "--source-root", src, "--corpus-root", corpus,
               "--chrom-sizes", genome_json, "--kinds", "counts,pval",
               "--biosample", "CF_C19__base__base", "--biosample", "CF_C12__depth__15M")
    assert b.returncode == 0, b.stderr
    m = _store("build-manifest", "--corpus-root", corpus, "--corpus", "cf",
               "--metadata-csv", src / "cf_metadata.csv", "--source-root", src,
               "--signal-provenance", src / "signal_provenance.cf.json")
    assert m.returncode == 0, m.stderr
    v = _store("verify", "--corpus-root", corpus)
    assert v.returncode == 0, v.stdout + v.stderr
    assert v.stdout.splitlines()[0] == f"{corpus}: OK"

    manifest = json.loads(L.manifest_path(corpus).read_text())
    assert manifest["corpus"] == "cf" and len(manifest["biosamples"]) == 2
    c19 = manifest["biosamples"]["CF_C19__base__base"]
    assert [tr["assay"] for tr in c19["tracks"]] == ["H3K27ac", "H3K4me3", "chipseq-control"]
    assert c19["control_col"] == 2
    by = {tr["assay"]: tr for tr in c19["tracks"]}
    assert by["H3K4me3"]["depth"] == by["H3K4me3"]["npz_depth"] == 29999990
    assert by["chipseq-control"]["depth"] == 58073570
    assert by["chipseq-control"]["file_accession"] == "ENCFF433TZR"
    assert by["H3K27ac"]["file_accession"] == "ENCFF254LWX"
    assert by["H3K27ac"]["signal_output_type"] == "signal p-value"
    assert by["H3K27ac"]["signal_derivation"]["pid"] == "C19M16__base__base"
    c12 = manifest["biosamples"]["CF_C12__depth__15M"]
    assert [tr["assay"] for tr in c12["tracks"]] == ["DNase-seq"] and c12["control_col"] == -1
    assert c12["tracks"][0]["read_length"] == 76
    assert not [g for g in manifest["metadata_gaps"] if g["reason"] != "absent"]

    # the arrays in the built h5 are the product arrays
    arr = t["arrays"]
    with h5py.File(L.kind_path(corpus, "CF_C19__base__base", "counts"), "r") as f:
        for chrom in N_BINS:
            np.testing.assert_array_equal(f[chrom][:, 1], arr["C19M22__base__base"]["counts"][chrom])
            np.testing.assert_array_equal(f[chrom][:, 2], arr["C19M16__base__base"]["control"][chrom])
    with h5py.File(L.kind_path(corpus, "CF_C12__depth__15M", "pval"), "r") as f:
        a = L.read_root_attrs(f)
        got = L.decode_pval(f["chrX"][:, 0], a[L.ATTR_SCALE], a[L.ATTR_TRANSFORM])
        # the arcsinh codec: one code is 1/scale in asinh space, i.e. at most ~sqrt(1+x^2)/scale in x
        tol = 1.0 / a[L.ATTR_SCALE]
        np.testing.assert_allclose(got, arr["C12M02__depth__15M"]["pval"]["chrX"], rtol=tol, atol=tol)


def _export(t):
    return export_store.export(t["products"], t["manifest"], t["tmp"] / "src", t["chrsz"])


def _set_control(t, pid, **fields):
    path = t["products"] / pid / "covariates.json"
    cov = json.loads(path.read_text())
    cov["control"].update(fields)
    path.write_text(json.dumps(cov))


def test_tracks_naming_different_controls_raise(setup):
    t = setup
    rows = t["rows"]
    rows[1]["control_accession"] = "ENCFF337JNL"
    with t["manifest"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    _set_control(t, "C19M22__base__base", accession="ENCFF337JNL")
    with pytest.raises(ValueError, match="different controls"):
        _export(t)


def test_manifest_accession_disagreeing_with_covariates_raises(setup):
    t = setup
    _set_control(t, "C19M22__base__base", accession="ENCFF337JNL")
    with pytest.raises(ValueError, match="MANIFEST control_accession"):
        _export(t)


def test_same_accession_different_reads_raises(setup):
    """The ctldepth case: both C19 tracks name ENCFF433TZR at every level; only `reads` differs."""
    t = setup
    _set_control(t, "C19M22__base__base", reads=29036785)
    with pytest.raises(ValueError, match="different controls"):
        _export(t)


def test_same_control_object_different_control_array_raises(setup):
    t = setup
    d = t["products"] / "C19M22__base__base"
    ctl = dict(np.load(d / "control_counts25.npz"))
    ctl["chrX"] = ctl["chrX"].copy()
    ctl["chrX"][3] += 1
    np.savez_compressed(d / "control_counts25.npz", **ctl)
    with pytest.raises(ValueError, match="differs"):
        _export(t)


def test_identical_controls_pass_and_carry_the_shared_depth(setup):
    t = setup
    # rewritten with the same arrays: a different npz file (zip timestamps), the same control
    d = t["products"] / "C19M22__base__base"
    np.savez_compressed(d / "control_counts25.npz", **dict(np.load(d / "control_counts25.npz")))
    _export(t)
    meta = t["tmp"] / "src" / "CF_C19__base__base" / "chipseq-control" / "signal_DSF1_res25" / "metadata.json"
    assert json.loads(meta.read_text()) == {"depth": 58073570, "dsf": 1}


def test_counts_outside_uint32_raise(setup):
    t = setup
    d = t["products"] / "C12M02__depth__15M"
    np.savez_compressed(d / "counts25.npz", chr21=np.full(N_BINS["chr21"], -1, np.int64),
                        chrX=np.zeros(N_BINS["chrX"], np.int64))
    with pytest.raises(ValueError, match="do not fit uint32"):
        _export(t)


def test_wrong_length_array_and_non_empty_source_root_raise(setup):
    t = setup
    d = t["products"] / "C12M02__depth__15M"
    np.savez_compressed(d / "counts25.npz", chr21=np.zeros(N_BINS["chr21"] + 1, np.uint32),
                        chrX=np.zeros(N_BINS["chrX"], np.uint32))
    with pytest.raises(ValueError, match="shape"):
        export_store.export(t["products"], t["manifest"], t["tmp"] / "src", t["chrsz"])
    with pytest.raises(ValueError, match="not empty"):
        export_store.export(t["products"], t["manifest"], t["tmp"] / "src", t["chrsz"])


def test_build_store_script_parses():
    r = subprocess.run(["bash", "-n", str(ROOT / "slurm" / "t112" / "build_store.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
