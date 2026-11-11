"""DNase-seq FASTQ kit (t152, chunk D7d): units, arms and pipeline input per unit and arm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_data import fastq_dnase as D  # noqa: E402

AD1 = {"sequence_R1": "AGATCGGAAGAGCACAC", "sequence_R2": "AGATCGGAAGAGCGTCG"}


def _fq(acc, rep, rl, pe=None, mate=None, status="released"):
    f = {"accession": acc, "status": status, "replicate": rep, "read_length": rl,
         "md5sum": "0" * 32, "href": f"/files/{acc}/@@download/{acc}.fastq.gz",
         "run_type": "paired-ended" if pe else "single-ended"}
    if pe:
        f["paired_end"] = pe
        f["paired_with"] = f"/files/{mate}/"
    return f


def _lib(acc, adapters):
    ads = []
    if adapters:
        ads = [{"type": "read1 3' adapter", "sequence": adapters["sequence_R1"]},
               {"type": "read2 3' adapter", "sequence": adapters["sequence_R2"]}]
    return {"accession": acc, "adapters": ads}


@pytest.fixture
def meta():
    """br1: one library, two 76 bp read pairs. br2: one library, one 36 bp single-end FASTQ.
    One replaced FASTQ that must not enter."""
    exp = {"accession": "ENCSRTEST", "replicates": [
        {"@id": "/replicates/r1/", "biological_replicate_number": 1,
         "technical_replicate_number": 1, "library": _lib("ENCLB001", AD1)},
        {"@id": "/replicates/r2/", "biological_replicate_number": 2,
         "technical_replicate_number": 1, "library": _lib("ENCLB002", None)},
    ]}
    fastqs = [
        _fq("ENCFF00A1", "/replicates/r1/", 76, "1", "ENCFF00A2"),
        _fq("ENCFF00A2", "/replicates/r1/", 76, "2", "ENCFF00A1"),
        _fq("ENCFF00B1", "/replicates/r1/", 76, "1", "ENCFF00B2"),
        _fq("ENCFF00B2", "/replicates/r1/", 76, "2", "ENCFF00B1"),
        _fq("ENCFF00C1", "/replicates/r2/", 36),
        _fq("ENCFF00X9", "/replicates/r2/", 36, status="replaced"),
    ]
    return {"experiment": exp, "fastqs": fastqs}


def path(f):
    return f"/tmp/fq/{f['accession']}.fastq.gz"


def test_units_and_arms(meta):
    got = {u["unit"]: D.arms_for(u) for u in D.units(meta)}
    assert got == {
        "experiment": ["base", "se"],                     # shortest read is 36 bp: no crop
        "br1tr1": ["base", "se", "crop36", "crop50"],
        "br1tr1ln1": ["base", "se", "crop36", "crop50"],
        "br1tr1ln2": ["base", "se", "crop36", "crop50"],
        "br2tr1": ["base"],                               # one file: no lane units
    }


def test_replaced_fastq_never_enters(meta):
    accs = {r["R1"]["accession"] for u in D.units(meta) for r in D.unit_runs(u)}
    assert "ENCFF00X9" not in accs


def test_base_experiment_input(meta):
    u = D.select_unit(meta, "experiment")
    inp = D.input_json("ENCSRTEST", u, "base", path, "/refs")
    reps = inp["dnase_align_only.replicates"]
    assert [r["number"] for r in reps] == [1, 2]
    assert reps[0]["read_length"] == 76 and reps[0]["adapters"] == AD1
    assert reps[0]["pe_fastqs"] == [
        {"R1": "/tmp/fq/ENCFF00A1.fastq.gz", "R2": "/tmp/fq/ENCFF00A2.fastq.gz"},
        {"R1": "/tmp/fq/ENCFF00B1.fastq.gz", "R2": "/tmp/fq/ENCFF00B2.fastq.gz"}]
    assert "se_fastqs" not in reps[0]
    assert reps[1] == {"accession": "ENCSRTEST", "number": 2, "read_length": 36,
                       "se_fastqs": ["/tmp/fq/ENCFF00C1.fastq.gz"]}
    assert inp["dnase_align_only.machine_sizes"]["align"] == "large2x"
    refs = inp["dnase_align_only.references"]
    assert refs["bwa_index"]["sa"] == "/refs/GRCh38.fa.sa"
    assert refs["indexed_fasta"]["fai"] == "/refs/GRCh38.fa.fai"
    assert refs["nuclear_chroms"] == "/refs/nuclear_chroms.txt"
    assert not any(k.startswith("hotspot") for k in refs)
    assert D.product_names("experiment", [1, 2]) == ["br1", "br2"]
    assert D.out_dir_name("experiment", "base") == "base"


def test_se_arm_takes_mate1_only(meta):
    u = D.select_unit(meta, "br1tr1")
    (rep,) = D.input_json("ENCSRTEST", u, "se", path, "/refs")["dnase_align_only.replicates"]
    assert rep["se_fastqs"] == ["/tmp/fq/ENCFF00A1.fastq.gz", "/tmp/fq/ENCFF00B1.fastq.gz"]
    assert "pe_fastqs" not in rep and "adapters" not in rep
    assert rep["read_length"] == 76


@pytest.mark.parametrize("arm,n", [("crop36", 36), ("crop50", 50)])
def test_crop_arm_sets_trim_length(meta, arm, n):
    u = D.select_unit(meta, "br1tr1ln2")
    (rep,) = D.input_json("ENCSRTEST", u, arm, path, "/refs")["dnase_align_only.replicates"]
    assert rep["read_length"] == n
    assert rep["pe_fastqs"] == [{"R1": "/tmp/fq/ENCFF00B1.fastq.gz",
                                 "R2": "/tmp/fq/ENCFF00B2.fastq.gz"}]
    assert D.out_dir_name("br1tr1ln2", arm) == f"br1tr1ln2.{arm}"
    assert D.product_names("br1tr1ln2", [1]) == ["br1tr1ln2"]


def test_arm_that_does_not_apply_is_refused(meta):
    with pytest.raises(SystemExit, match="does not apply"):
        D.input_json("ENCSRTEST", D.select_unit(meta, "experiment"), "crop36", path, "/refs")
    with pytest.raises(SystemExit, match="does not apply"):
        D.input_json("ENCSRTEST", D.select_unit(meta, "br2tr1"), "se", path, "/refs")
    with pytest.raises(SystemExit, match="unknown arm"):
        D.input_json("ENCSRTEST", D.select_unit(meta, "br2tr1"), "mapq0", path, "/refs")


def test_paired_end_without_adapters_is_refused(meta):
    meta["experiment"]["replicates"][0]["library"] = _lib("ENCLB001", None)
    with pytest.raises(SystemExit, match="adapters"):
        D.input_json("ENCSRTEST", D.select_unit(meta, "br1tr1"), "base", path, "/refs")


def test_wrapper_stops_after_filtered_bam():
    w = D.wrapper_wdl()
    assert 'import "wdl/workflows/concatenate_trim_and_align_fastqs.wdl"' in w
    assert 'import "wdl/workflows/merge_mark_and_filter_bams.wdl"' in w
    assert w.count("import ") == 2 and w.count("call ") == 2
    for later in ("hotspot", "footprint", "normalize", "qc"):
        assert later not in w.lower().replace("trimstats", "")


def test_dry_run_from_build_list(meta, tmp_path):
    md = tmp_path / "meta.json"
    md.write_text(json.dumps(meta))
    bl = tmp_path / "build.tsv"
    bl.write_text("experiment\tunit\tarm\nENCSRTEST\texperiment\tbase\nENCSRTEST\tbr1tr1\tcrop50\n")
    D.main(["run", "--build-list", str(bl), "--row", "1", "--work", str(tmp_path / "work"),
            "--metadata", str(md), "--tmp", str(tmp_path / "tmp"), "--dry-run",
            "--scratch-refs", "/refs"])
    inp = json.loads((tmp_path / "work/ENCSRTEST/br1tr1.crop50/input.json").read_text())
    (rep,) = inp["dnase_align_only.replicates"]
    assert rep["read_length"] == 50 and len(rep["pe_fastqs"]) == 2
    assert rep["pe_fastqs"][0]["R1"].endswith("/tmp/fastq/ENCFF00A1.fastq.gz")


def test_build_list_needs_columns(tmp_path):
    bl = tmp_path / "b.tsv"
    bl.write_text("experiment\tarm\nE\tbase\n")
    with pytest.raises(SystemExit, match="needs columns"):
        D.read_build_list(bl, 0)


def test_reference_bundle_record_is_complete():
    assert {D.REF_INDEXED_FASTA, D.REF_BWA_INDEX, D.REF_NUCLEAR_CHROMS} <= set(D.REFERENCE_FILES)
    assert all(len(v[2]) == 32 for v in D.REFERENCE_FILES.values())
    assert D.IMAGE_DIGEST.startswith("sha256:") and len(D.IMAGE_DIGEST) == 71
