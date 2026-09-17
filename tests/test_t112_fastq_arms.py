"""`tools/t112/fastq_arms.py` — the FASTQ-arm Caper driver.

Nothing here reaches Nibi. The input-JSON fixtures are copies of the shapes of
`$EIC/inputs_bwa/C19M16.bwa.{se,pe}.json` (read on Nibi 2026-09-17); the rows come from the real
`arms.py rows --route fastq`; caper and squeue are replaced by a fake `sh`; the Cromwell tree is a
fake with the layout of a real ChIP run (`execution/` + `execution/glob-<hash>/` hard links).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t112" / "fastq_arms.py"
ARMS = REPO / "tools" / "t112" / "arms.py"

URL = "https://www.encodeproject.org/files/{0}/@@download/{0}.fastq.gz"
SIF = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/sif/chip-seq-pipeline_v2.2.2.sif"


def _load():
    spec = importlib.util.spec_from_file_location("t112_fastq_arms", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t112_fastq_arms"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fa():
    return _load()


def _common(run_type):
    return {
        "chip.pipeline_type": "histone",
        "chip.align_only": False,
        "chip.true_rep_only": False,
        "chip.genome_tsv": "https://storage.googleapis.com/encode-pipeline-genome-data/genome_tsv/v3/hg38.tsv",
        "chip.genome_name": "hg38",
        "chip.exp_ctl_depth_ratio_limit": 0.0,
        "chip.ctl_depth_limit": 0,
        "chip.subsample_reads": 30000000,
        "chip.filter_cpu": 4,
        "chip.xcor_cpu": 8,
        "chip.bam2ta_cpu": 8,
        "chip.bam2ta_mem_factor": 6,
        "chip.title": "ENCODE Imputation Challenge C19M16",
        "chip.description": f"Running ENCSR131DVD from the challenge with {run_type} settings.",
        "chip.aligner": "bwa",
    }


SE_JSON = {
    **_common("single-end"),
    "chip.paired_end": False,
    "chip.ctl_paired_end": False,
    "chip.fastqs_rep1_R1": [URL.format(a) for a in
                            ("ENCFF254LWX", "ENCFF641LAG", "ENCFF490XNW", "ENCFF737LVW")],
    "chip.ctl_fastqs_rep1_R1": [URL.format(a) for a in ("ENCFF433TZR", "ENCFF044UIT")],
    "chip.singularity": SIF,
}

PE_JSON = {
    **_common("paired-end"),
    "chip.paired_end": True,
    "chip.ctl_paired_end": True,
    "chip.fastqs_rep1_R1": [URL.format(a) for a in ("ENCFF254LWX", "ENCFF641LAG")],
    "chip.fastqs_rep1_R2": [URL.format(a) for a in ("ENCFF490XNW", "ENCFF737LVW")],
    "chip.ctl_fastqs_rep1_R1": [URL.format("ENCFF433TZR")],
    "chip.ctl_fastqs_rep1_R2": [URL.format("ENCFF044UIT")],
    "chip.singularity": SIF,
}


@pytest.fixture()
def env(tmp_path):
    """A fake $EIC, $CF (with a 0444 FASTQ cache) and refcache, plus the C19M16 fastq rows."""
    eic, cf, ref = tmp_path / "eic", tmp_path / "cf", tmp_path / "refcache"
    (eic / "inputs_bwa").mkdir(parents=True)
    (eic / "inputs_bwa" / "C19M16.bwa.se.json").write_text(json.dumps(SE_JSON, indent=2) + "\n")
    (eic / "inputs_bwa" / "C19M16.bwa.pe.json").write_text(json.dumps(PE_JSON, indent=2) + "\n")
    (eic / "chip-seq-pipeline2").mkdir()
    (eic / "chip-seq-pipeline2" / "chip.wdl").write_text("version 1.0\n")
    (eic / "sif").mkdir()
    (eic / "sif" / "chip-seq-pipeline_v2.2.2.sif").write_text("sif")
    (eic / "runner").mkdir()
    (eic / "runner" / "env.sh").write_text("")

    for acc in ("ENCFF254LWX", "ENCFF433TZR"):
        url = URL.format(acc)
        f = cf / "fastq_cache" / hashlib.md5(url.encode()).hexdigest() / f"{acc}.fastq.gz"
        f.parent.mkdir(parents=True)
        f.write_text(acc)
        f.chmod(0o444)
    (cf / "fastq_cache" / "VERIFIED.tsv").write_text("acc\n")
    (cf / "fastq_cache" / "VERIFIED.tsv").chmod(0o444)
    g = ref / "3ff4ac4c3f59d096b1a3842a182072ae" / "ENCFF110MCL.tar.gz"
    g.parent.mkdir(parents=True)
    g.write_text("genome")
    g.chmod(0o444)

    tsv = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "fastq", "--dnase", "none",
                          "--ratio", "yes", "--tracks", "C19M16"],
                         capture_output=True, text=True, check=True).stdout
    rows_path = tmp_path / "rows.tsv"
    rows_path.write_text(tsv)
    return {"eic": eic, "cf": cf, "ref": ref, "rows": rows_path, "tmp": tmp_path}


def _row(fa, env, pid):
    return next(r for r in fa.read_rows(env["rows"]) if r["pid"] == pid)


# --- rows and input JSON -----------------------------------------------------------------------

def test_rows_are_arms_py_fastq_rows(fa, env):
    rows = fa.read_rows(env["rows"])
    assert [r["pid"] for r in rows] == ["C19M16__pe__pe", "C19M16__dedup__off", "C19M16__crop__36",
                                        "C19M16__crop__50", "C19M16__mapq__0", "C19M16__mapq__10"]
    assert [r["knob_value"] for r in rows] == [True, True, 36, 50, 0, 10]
    assert {r["pipeline"] for r in rows} == {"chip"}


def test_all_36_chip_rows_read(fa, tmp_path):
    tsv = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "fastq", "--dnase", "none",
                          "--ratio", "yes"], capture_output=True, text=True, check=True).stdout
    (tmp_path / "r.tsv").write_text(tsv)
    rows = fa.select_rows(fa.read_rows(tmp_path / "r.tsv"))
    assert len(rows) == 36 and len({r["pid"] for r in rows}) == 36


def test_bam_rows_refused(fa, tmp_path):
    tsv = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "all", "--dnase", "none",
                          "--ratio", "yes", "--tracks", "C19M16"],
                         capture_output=True, text=True, check=True).stdout
    (tmp_path / "r.tsv").write_text(tsv)
    with pytest.raises(SystemExit, match="not FASTQ-route"):
        fa.select_rows(fa.read_rows(tmp_path / "r.tsv"))


@pytest.mark.parametrize("pid,knob,value", [
    ("C19M16__dedup__off", "chip.no_dup_removal", True),
    ("C19M16__crop__36", "chip.crop_length", 36),
    ("C19M16__crop__50", "chip.crop_length", 50),
    ("C19M16__mapq__0", "chip.mapq_thresh", 0),
    ("C19M16__mapq__10", "chip.mapq_thresh", 10),
])
def test_build_json_one_knob(fa, env, pid, knob, value):
    new = fa.build_json(_row(fa, env, pid), env["eic"])
    assert fa.diff_keys(SE_JSON, new) == {knob, "chip.title", "chip.description"}
    assert new[knob] == value and type(new[knob]) is type(value)
    assert new["chip.title"] == f"CF {pid}"
    assert new["chip.description"] == f"t112 counterfactual arm {pid}"
    for k, v in SE_JSON.items():
        if k not in (knob, "chip.title", "chip.description"):
            assert json.dumps(new[k]) == json.dumps(v), k
    # key order kept; the new knob is appended
    assert list(new)[:len(SE_JSON)] == list(SE_JSON)


def test_build_json_pe(fa, env):
    new = fa.build_json(_row(fa, env, "C19M16__pe__pe"), env["eic"])
    assert fa.diff_keys(SE_JSON, new) == {
        "chip.paired_end", "chip.ctl_paired_end", "chip.fastqs_rep1_R1", "chip.fastqs_rep1_R2",
        "chip.ctl_fastqs_rep1_R1", "chip.ctl_fastqs_rep1_R2", "chip.title", "chip.description"}
    assert fa.diff_keys(PE_JSON, new) == {"chip.title", "chip.description"}
    assert new["chip.paired_end"] is True and new["chip.ctl_paired_end"] is True


def test_build_json_pe_keeps_se_resources(fa, env):
    # C07M29 on Nibi: se carries align_cpu 12 / filter_cpu 8, pe has no align_cpu and filter_cpu 4
    se = dict(SE_JSON, **{"chip.align_cpu": 12, "chip.filter_cpu": 8})
    pe = {k: v for k, v in PE_JSON.items()}
    (env["eic"] / "inputs_bwa" / "C19M16.bwa.se.json").write_text(json.dumps(se))
    (env["eic"] / "inputs_bwa" / "C19M16.bwa.pe.json").write_text(json.dumps(pe))
    new = fa.build_json(_row(fa, env, "C19M16__pe__pe"), env["eic"])
    assert new["chip.align_cpu"] == 12 and new["chip.filter_cpu"] == 8
    assert fa.diff_keys(se, new) == {
        "chip.paired_end", "chip.ctl_paired_end", "chip.fastqs_rep1_R1", "chip.fastqs_rep1_R2",
        "chip.ctl_fastqs_rep1_R1", "chip.ctl_fastqs_rep1_R2", "chip.title", "chip.description"}


def test_titles_unique(fa, env):
    titles = [fa.build_json(r, env["eic"])["chip.title"] for r in fa.read_rows(env["rows"])]
    assert len(set(titles)) == len(titles)


def test_build_json_refuses_a_knob_that_does_not_move(fa, env):
    se = dict(SE_JSON, **{"chip.mapq_thresh": 10})
    (env["eic"] / "inputs_bwa" / "C19M16.bwa.se.json").write_text(json.dumps(se))
    with pytest.raises(SystemExit, match="moves"):
        fa.build_json(_row(fa, env, "C19M16__mapq__10"), env["eic"])


def test_atac_row_refused_until_c13(fa, env):
    row = dict(_row(fa, env, "C19M16__mapq__0"), pipeline="atac", knob="atac.mapq_thresh")
    with pytest.raises(SystemExit, match="C13"):
        fa.build_json(row, env["eic"])


# --- the local genome TSV (the encode-pipeline-genome-data bucket is gone) -----------------------

@pytest.fixture()
def gtsv(tmp_path):
    """A stand-in for C7's `$CF/fastqarms/genome/hg38.local.tsv`."""
    p = tmp_path / "genome" / "hg38.local.tsv"
    p.parent.mkdir(parents=True)
    p.write_text("hg38\t/scratch/mforooz/EIC_REPRO/003/refcache/abc/GRCh38.fa\n")
    return p


@pytest.mark.parametrize("pid,knob", [
    ("C19M16__dedup__off", "chip.no_dup_removal"),
    ("C19M16__crop__36", "chip.crop_length"),
    ("C19M16__mapq__10", "chip.mapq_thresh"),
])
def test_build_json_genome_tsv_widens_the_allowed_set_by_one_key(fa, env, gtsv, pid, knob):
    new = fa.build_json(_row(fa, env, pid), env["eic"], gtsv)
    assert fa.diff_keys(SE_JSON, new) == {knob, "chip.title", "chip.description", "chip.genome_tsv"}
    assert new["chip.genome_tsv"] == str(gtsv)
    assert SE_JSON["chip.genome_tsv"].startswith("https://storage.googleapis.com/")
    for k, v in SE_JSON.items():
        if k not in (knob, "chip.title", "chip.description", "chip.genome_tsv"):
            assert json.dumps(new[k]) == json.dumps(v), k


def test_build_json_genome_tsv_pe(fa, env, gtsv):
    new = fa.build_json(_row(fa, env, "C19M16__pe__pe"), env["eic"], gtsv)
    assert fa.diff_keys(SE_JSON, new) == {
        "chip.paired_end", "chip.ctl_paired_end", "chip.fastqs_rep1_R1", "chip.fastqs_rep1_R2",
        "chip.ctl_fastqs_rep1_R1", "chip.ctl_fastqs_rep1_R2", "chip.title", "chip.description",
        "chip.genome_tsv"}
    assert fa.diff_keys(PE_JSON, new) == {"chip.title", "chip.description", "chip.genome_tsv"}


def test_build_json_without_the_flag_is_unchanged(fa, env):
    for pid in ("C19M16__crop__50", "C19M16__pe__pe"):
        row = _row(fa, env, pid)
        assert fa.build_json(row, env["eic"]) == fa.build_json(row, env["eic"], None)
        assert fa.build_json(row, env["eic"])["chip.genome_tsv"] == SE_JSON["chip.genome_tsv"]


def test_build_json_genome_tsv_uses_the_rows_pipeline_prefix(fa, env, gtsv, tmp_path, monkeypatch):
    """The override key is `<pipeline>.genome_tsv`. C13 owns the real atac branch; this only pins
    that the DNase JSON gets `atac.genome_tsv`, not `chip.genome_tsv`."""
    atac_base = {"atac.pipeline_type": "dnase", "atac.paired_end": False,
                 "atac.genome_tsv": SE_JSON["chip.genome_tsv"], "atac.mapq_thresh": 30,
                 "atac.title": "ENCODE Imputation Challenge C12M02", "atac.description": "base"}
    base = tmp_path / "inputs_dnase" / "C12M02.dnase.se.json"
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps(atac_base, indent=2) + "\n")
    monkeypatch.setitem(fa.PIPELINES, "atac",
                        {"prefix": "atac", "wdl": "atac-seq-pipeline/atac.wdl", "sif": "sif/atac.sif"})
    monkeypatch.setattr(fa, "base_json_path", lambda row, eic_dir, tag: base)
    row = {"pid": "C12M02__mapq__0", "arm": "mapq", "knob": "atac.mapq_thresh", "knob_value": 0,
           "pipeline": "atac", "track": "C12M02", "route": "fastq"}
    new = fa.build_json(row, env["eic"], gtsv)
    assert fa.diff_keys(atac_base, new) == {"atac.mapq_thresh", "atac.title", "atac.description",
                                            "atac.genome_tsv"}
    assert new["atac.genome_tsv"] == str(gtsv) and "chip.genome_tsv" not in new


# --- stage -------------------------------------------------------------------------------------

def test_stage_hardlinks_and_state(fa, env):
    row = _row(fa, env, "C19M16__mapq__10")
    st = fa.stage(row, env["cf"], env["eic"], env["ref"])
    p = fa.paths(env["cf"], row["pid"])
    assert st["status"] == "staged" and st["attempts"] == 0 and st["leader_job_id"] is None
    assert json.loads(p["input"].read_text()) == fa.build_json(row, env["eic"])
    for src_root in (env["cf"] / "fastq_cache", env["ref"]):
        for f in src_root.rglob("*"):
            if f.is_file():
                d = p["loc"] / f.relative_to(src_root)
                assert d.exists() and os.path.samefile(f, d), d
    url = URL.format("ENCFF254LWX")
    assert (p["loc"] / hashlib.md5(url.encode()).hexdigest() / "ENCFF254LWX.fastq.gz").exists()
    assert set(json.loads(p["state"].read_text())) >= {
        "pid", "status", "leader_job_id", "attempts", "workflow_id", "metadata_json", "updated_utc"}
    # idempotent
    assert fa.stage(row, env["cf"], env["eic"], env["ref"])["status"] == "staged"


def test_stage_refuses_writable_source(fa, env):
    f = next(env["ref"].rglob("*.tar.gz"))
    f.chmod(0o644)
    with pytest.raises(SystemExit, match="not mode 0444"):
        fa.stage(_row(fa, env, "C19M16__crop__36"), env["cf"], env["eic"], env["ref"])
    assert not fa.paths(env["cf"], "C19M16__crop__36")["state"].exists()


def test_stage_refuses_foreign_file_in_loc(fa, env):
    row = _row(fa, env, "C19M16__crop__36")
    loc = fa.paths(env["cf"], row["pid"])["loc"]
    impostor = loc / "3ff4ac4c3f59d096b1a3842a182072ae" / "ENCFF110MCL.tar.gz"
    impostor.parent.mkdir(parents=True)
    impostor.write_text("genome")
    with pytest.raises(SystemExit, match="not a hard link"):
        fa.stage(row, env["cf"], env["eic"], env["ref"])


def test_stage_records_the_genome_tsv_and_its_md5(fa, env, gtsv):
    row = _row(fa, env, "C19M16__mapq__10")
    st = fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    assert st["genome_tsv"] == str(gtsv)
    assert st["genome_tsv_md5"] == hashlib.md5(gtsv.read_bytes()).hexdigest()
    written = json.loads(fa.paths(env["cf"], row["pid"])["input"].read_text())
    assert written["chip.genome_tsv"] == str(gtsv)
    # same flag again: idempotent
    again = fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    assert (again["genome_tsv"], again["genome_tsv_md5"]) == (st["genome_tsv"], st["genome_tsv_md5"])
    assert json.loads(fa.paths(env["cf"], row["pid"])["input"].read_text()) == written


def test_stage_without_the_flag_records_none(fa, env):
    st = fa.stage(_row(fa, env, "C19M16__mapq__10"), env["cf"], env["eic"], env["ref"])
    assert st["genome_tsv"] is None and st["genome_tsv_md5"] is None
    written = json.loads(fa.paths(env["cf"], "C19M16__mapq__10")["input"].read_text())
    assert written["chip.genome_tsv"] == SE_JSON["chip.genome_tsv"]


def test_restage_with_a_different_genome_tsv_overwrites_json_and_state(fa, env, gtsv, tmp_path):
    row = _row(fa, env, "C19M16__crop__36")
    fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    other = tmp_path / "genome2" / "hg38.local.tsv"
    other.parent.mkdir()
    other.write_text("hg38\t/scratch/mforooz/EIC_REPRO/003/refcache/def/GRCh38.fa\n")
    st = fa.stage(row, env["cf"], env["eic"], env["ref"], other)
    assert st["genome_tsv"] == str(other)
    assert st["genome_tsv_md5"] == hashlib.md5(other.read_bytes()).hexdigest()
    written = json.loads(fa.paths(env["cf"], row["pid"])["input"].read_text())
    assert written["chip.genome_tsv"] == str(other)
    assert written["chip.crop_length"] == 36  # nothing else moved


def test_restage_refuses_a_difference_that_is_not_the_genome_tsv(fa, env, gtsv):
    row = _row(fa, env, "C19M16__crop__36")
    fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    (env["eic"] / "inputs_bwa" / "C19M16.bwa.se.json").write_text(
        json.dumps(dict(SE_JSON, **{"chip.xcor_cpu": 16}), indent=2) + "\n")
    with pytest.raises(SystemExit, match="not in genome_tsv alone"):
        fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)


def test_stage_refuses_a_missing_genome_tsv(fa, env, tmp_path):
    with pytest.raises(SystemExit, match="is not a file"):
        fa.stage(_row(fa, env, "C19M16__crop__36"), env["cf"], env["eic"], env["ref"],
                 tmp_path / "nope.tsv")


def test_restage_of_a_failed_pid_rewrites_and_returns_to_staged(fa, env, gtsv):
    """C19M16__mapq__10: leader 22148552 died in 21 s on the dead genome_tsv URL. Nothing live
    holds that JSON, so the local TSV goes in and the pid goes back to `staged`."""
    row = _row(fa, env, "C19M16__mapq__10")
    fa.stage(row, env["cf"], env["eic"], env["ref"])
    fa.write_state(env["cf"], row["pid"], status="failed", leader_job_id="22148552", attempts=1)
    st = fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    assert st["status"] == "staged"
    assert st["leader_job_id"] == "22148552" and st["attempts"] == 1  # history kept
    assert st["genome_tsv"] == str(gtsv)
    assert st["genome_tsv_md5"] == hashlib.md5(gtsv.read_bytes()).hexdigest()
    written = json.loads(fa.paths(env["cf"], row["pid"])["input"].read_text())
    assert written["chip.genome_tsv"] == str(gtsv) and written["chip.mapq_thresh"] == 10


def test_restage_of_a_failed_pid_still_refuses_a_non_genome_difference(fa, env, gtsv):
    row = _row(fa, env, "C19M16__crop__36")
    fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    fa.write_state(env["cf"], row["pid"], status="failed", attempts=1)
    (env["eic"] / "inputs_bwa" / "C19M16.bwa.se.json").write_text(
        json.dumps(dict(SE_JSON, **{"chip.xcor_cpu": 16}), indent=2) + "\n")
    with pytest.raises(SystemExit, match="not in genome_tsv alone"):
        fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    assert fa.read_state(env["cf"], row["pid"])["status"] == "failed"


@pytest.mark.parametrize("status", ["submitted", "succeeded", "harvested"])
def test_a_launched_pid_is_never_restaged_with_a_new_genome_tsv(fa, env, gtsv, tmp_path, status):
    row = _row(fa, env, "C19M16__mapq__0")
    fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)
    before = fa.paths(env["cf"], row["pid"])["input"].read_text()
    fa.write_state(env["cf"], row["pid"], status=status)
    other = tmp_path / "genome3" / "hg38.local.tsv"
    other.parent.mkdir()
    other.write_text("a different TSV\n")
    with pytest.raises(SystemExit, match="REFUSING to re-stage"):
        fa.stage(row, env["cf"], env["eic"], env["ref"], other)
    assert fa.paths(env["cf"], row["pid"])["input"].read_text() == before
    assert fa.read_state(env["cf"], row["pid"])["genome_tsv"] == str(gtsv)
    # the same TSV is the no-op it always was
    assert fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)["status"] == status


# --- submit ------------------------------------------------------------------------------------

class FakeSh:
    def __init__(self, out="", rc=0, squeue=(0, "")):
        self.calls, self.out, self.rc, self.squeue = [], out, rc, squeue

    def __call__(self, argv, timeout=900):
        self.calls.append(list(argv))
        if argv[0] == "squeue":
            return self.squeue
        return self.rc, self.out


def _staged(fa, env, pid):
    row = _row(fa, env, pid)
    fa.stage(row, env["cf"], env["eic"], env["ref"])
    return row


def test_submit_single_token_command(fa, env, monkeypatch):
    row = _staged(fa, env, "C19M16__mapq__10")
    fake = FakeSh(out="some caper chatter\nSubmitted batch job 4242\n")
    monkeypatch.setattr(fa, "sh", fake)
    st = fa.submit(row, env["cf"], env["eic"])
    p = fa.paths(env["cf"], row["pid"])
    assert st["status"] == "submitted" and st["leader_job_id"] == "4242" and st["attempts"] == 1
    assert st["cromwell_dir"] == str(p["cromwell"]) and p["cromwell"].is_dir()

    argv = fake.calls[0]
    assert argv[:2] == ["bash", "-lc"] and len(argv) == 3
    eic = env["eic"]
    assert shlex.split(argv[2]) == [
        "source", f"{eic}/runner/env.sh", ">/dev/null", "2>&1", "&&", "cd", str(p["logs"]), "&&",
        "caper", "hpc", "submit", f"{eic}/chip-seq-pipeline2/chip.wdl",
        "-i", str(p["input"]),
        "--leader-job-name", "t112_C19M16__mapq__10",
        "--local-loc-dir", str(p["loc"]),
        "--local-out-dir", str(p["cromwell"]),
        "--singularity", f"{eic}/sif/chip-seq-pipeline_v2.2.2.sif"]
    assert (p["logs"] / "C19M16__mapq__10.submit1.log").exists()
    # a submitted pid is not submitted twice
    fa.submit(row, env["cf"], env["eic"])
    assert len(fake.calls) == 1


def test_submit_failure_keeps_staged(fa, env, monkeypatch):
    row = _staged(fa, env, "C19M16__crop__50")
    monkeypatch.setattr(fa, "sh", FakeSh(out="caper: error: something", rc=1))
    with pytest.raises(SystemExit, match="submit FAILED"):
        fa.submit(row, env["cf"], env["eic"])
    assert fa.read_state(env["cf"], row["pid"])["status"] == "staged"


def test_submit_refuses_lock_files(fa, env, monkeypatch):
    row = _staged(fa, env, "C19M16__crop__50")
    lock = fa.paths(env["cf"], row["pid"])["loc"] / "3ff4ac4c3f59d096b1a3842a182072ae" / "x.lock"
    lock.write_text("")
    fake = FakeSh(out="Submitted batch job 1")
    monkeypatch.setattr(fa, "sh", fake)
    with pytest.raises(SystemExit, match="lock"):
        fa.submit(row, env["cf"], env["eic"])
    assert fake.calls == [] and lock.exists()


def test_retry_goes_to_a_fresh_out_dir(fa, env, monkeypatch):
    row = _staged(fa, env, "C19M16__dedup__off")
    monkeypatch.setattr(fa, "sh", FakeSh(out="Submitted batch job 10"))
    fa.submit(row, env["cf"], env["eic"])
    fa.write_state(env["cf"], row["pid"], status="failed")
    fake = FakeSh(out="Submitted batch job 11")
    monkeypatch.setattr(fa, "sh", fake)
    assert fa.submit(row, env["cf"], env["eic"])["status"] == "failed"  # no --retry: skipped
    assert fake.calls == []
    st = fa.submit(row, env["cf"], env["eic"], retry=True)
    assert st["attempts"] == 2 and st["leader_job_id"] == "11"
    assert st["cromwell_dir"].endswith("cromwell/C19M16__dedup__off__attempt2")
    fa.write_state(env["cf"], row["pid"], status="failed")
    with pytest.raises(SystemExit, match="max-attempts"):
        fa.submit(row, env["cf"], env["eic"], retry=True)


def test_submit_after_a_failed_pid_is_restaged(fa, env, gtsv, monkeypatch):
    """A failed pid re-staged with the local TSV needs no --retry, and still gets a fresh out dir."""
    row = _staged(fa, env, "C19M16__mapq__10")
    monkeypatch.setattr(fa, "sh", FakeSh(out="Submitted batch job 22148552"))
    fa.submit(row, env["cf"], env["eic"])
    fa.write_state(env["cf"], row["pid"], status="failed")
    assert fa.stage(row, env["cf"], env["eic"], env["ref"], gtsv)["status"] == "staged"
    monkeypatch.setattr(fa, "sh", FakeSh(out="Submitted batch job 22150000"))
    st = fa.submit(row, env["cf"], env["eic"])
    assert st["attempts"] == 2 and st["leader_job_id"] == "22150000"
    assert st["cromwell_dir"].endswith("cromwell/C19M16__mapq__10__attempt2")
    assert st["genome_tsv"] == str(gtsv)


# --- fake Cromwell tree --------------------------------------------------------------------------

def _fake_run(fa, env, pid, status="Succeeded", attempt=1):
    """A ChIP workflow tree under the pid's cromwell dir, and its metadata.json."""
    row = _row(fa, env, pid)
    cdir = fa.paths(env["cf"], pid, attempt)["cromwell"]
    wfid = "f4e3dc0f-be72-4154-848c-f2d0ad614e7d"
    root = cdir / "chip" / wfid
    t, c = "ENCFF254LWX", "ENCFF433TZR"
    nodup = "filt" if row["arm"] == "dedup" else "nodup"
    spec = {  # call: {output key: file name}, plus distractors under "_"
        "filter": {"nodup_bam": f"{t}.merged.srt.{nodup}.bam", "nodup_bai": f"{t}.merged.srt.{nodup}.bam.bai",
                   "_": [f"{t}.merged.srt.{nodup}.samstats.qc", f"{t}.merged.srt.dup.qc"]},
        "filter_ctl": {"nodup_bam": f"{c}.merged.srt.{nodup}.bam", "nodup_bai": f"{c}.merged.srt.{nodup}.bam.bai"},
        "bam2ta": {"ta": f"{t}.merged.srt.{nodup}.30M.tagAlign.gz"},
        "bam2ta_ctl": {"ta": f"{c}.merged.srt.{nodup}.tagAlign.gz"},
        "macs2_signal_track": {
            "pval_bw": f"{t}.merged.srt.nodup.30M_x_{c}.merged.srt.nodup.pval.signal.bigwig",
            "fc_bw": f"{t}.merged.srt.nodup.30M_x_{c}.merged.srt.nodup.fc.signal.bigwig"},
        "xcor": {"score": f"{t}.merged.trim_50bp.srt.filt.no_chrM.15M.cc.qc",
                 "fraglen_log": f"{t}.merged.trim_50bp.srt.filt.no_chrM.15M.cc.fraglen.txt"},
        "qc_report": {"qc_json": "qc.json", "report": "qc.html"},
        "align": {"bam": f"{t}.merged.srt.bam", "bai": f"{t}.merged.srt.bam.bai"},
        "align_ctl": {"bam": f"{c}.merged.srt.bam", "bai": f"{c}.merged.srt.bam.bai"},
    }
    calls = {}
    for call, outs in spec.items():
        shard = -1 if call == "qc_report" else 0
        call_root = root / f"call-{call}" / ("" if shard < 0 else "shard-0")
        if attempt > 1:
            call_root = call_root / f"attempt-{attempt}"
        ex = call_root / "execution"
        recorded = {}
        for i, (key, name) in enumerate([(k, v) for k, v in outs.items() if k != "_"]
                                        + [(None, v) for v in outs.get("_", [])]):
            g = ex / f"glob-{i:032x}" / name
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_text(f"{call}:{name}")
            os.link(g, ex / name)  # Cromwell's execution/<name> is the same inode
            if key:
                recorded[key] = str(g)
        # a downstream call's localized copy of an upstream output must never be harvested
        inp = call_root / "inputs" / "-123" / f"{t}.merged.srt.nodup.bam"
        inp.parent.mkdir(parents=True, exist_ok=True)
        inp.write_text("localized copy")
        calls[f"chip.{call}"] = [{"shardIndex": shard, "attempt": attempt, "executionStatus": "Done",
                                  "callRoot": str(call_root), "outputs": recorded}]
    meta = {"id": wfid, "status": status, "workflowRoot": str(root), "calls": calls}
    (root / "metadata.json").write_text(json.dumps(meta))
    return root / "metadata.json"


def _submitted(fa, env, pid, monkeypatch):
    row = _staged(fa, env, pid)
    monkeypatch.setattr(fa, "sh", FakeSh(out="Submitted batch job 777"))
    fa.submit(row, env["cf"], env["eic"])
    return row


# --- poll --------------------------------------------------------------------------------------

def test_poll_alive_stays_submitted(fa, env, monkeypatch):
    row = _submitted(fa, env, "C19M16__mapq__0", monkeypatch)
    fake = FakeSh(squeue=(0, "RUNNING\n"))
    monkeypatch.setattr(fa, "sh", fake)
    assert fa.poll(row, env["cf"])["status"] == "submitted"
    assert fake.calls[0][:4] == ["squeue", "-h", "-j", "777"]


def test_poll_unreachable_scheduler_is_not_death(fa, env, monkeypatch):
    row = _submitted(fa, env, "C19M16__mapq__0", monkeypatch)
    _fake_run(fa, env, row["pid"], status="Failed")
    monkeypatch.setattr(fa, "sh", FakeSh(squeue=(1, "slurm_load_jobs error: Unable to contact slurm controller")))
    assert fa.poll(row, env["cf"])["status"] == "submitted"


@pytest.mark.parametrize("wf_status,expect", [("Succeeded", "succeeded"), ("Failed", "failed")])
def test_poll_dead_reads_metadata(fa, env, monkeypatch, wf_status, expect):
    row = _submitted(fa, env, "C19M16__mapq__0", monkeypatch)
    meta = _fake_run(fa, env, row["pid"], status=wf_status)
    monkeypatch.setattr(fa, "sh", FakeSh(squeue=(1, "slurm_load_jobs error: Invalid job id specified")))
    st = fa.poll(row, env["cf"])
    assert st["status"] == expect
    assert st["metadata_json"] == str(meta) and st["workflow_id"] == "f4e3dc0f-be72-4154-848c-f2d0ad614e7d"


def test_poll_dead_without_metadata_is_failed(fa, env, monkeypatch):
    row = _submitted(fa, env, "C19M16__mapq__0", monkeypatch)
    monkeypatch.setattr(fa, "sh", FakeSh(squeue=(0, "")))
    assert fa.poll(row, env["cf"])["status"] == "failed"


# --- harvest -----------------------------------------------------------------------------------

def _succeeded(fa, env, pid, monkeypatch):
    row = _submitted(fa, env, pid, monkeypatch)
    _fake_run(fa, env, pid)
    monkeypatch.setattr(fa, "sh", FakeSh(squeue=(1, "Invalid job id specified")))
    assert fa.poll(row, env["cf"])["status"] == "succeeded"
    return row


def _harvest_tsv(fa, env, pid):
    lines = (fa.paths(env["cf"], pid)["harvest"] / "HARVEST.tsv").read_text().rstrip("\n").split("\n")
    assert lines[0] == "role\tsrc\tdest\tbytes\tmd5"
    return [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]]


BASE_ROLES = ["treat_bam", "treat_bai", "ctl_bam", "ctl_bai", "treat_ta", "ctl_ta", "pval_bigwig",
              "xcor_qc", "qc_json", "metadata"]


def test_harvest_copies_and_verifies(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__crop__36", monkeypatch)
    st = fa.harvest(row, env["cf"])
    assert st["status"] == "harvested"
    recs = _harvest_tsv(fa, env, row["pid"])
    assert [r["role"] for r in recs] == BASE_ROLES
    by = {r["role"]: r for r in recs}
    assert by["treat_ta"]["dest"].endswith("ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz")
    assert by["ctl_bam"]["dest"].endswith("ENCFF433TZR.merged.srt.nodup.bam")
    assert by["pval_bigwig"]["dest"].endswith(".pval.signal.bigwig")
    assert by["xcor_qc"]["dest"].endswith(".cc.qc")
    for r in recs:
        src, dest = Path(r["src"]), Path(r["dest"])
        assert "/inputs/" not in r["src"]
        assert dest.parent == fa.paths(env["cf"], row["pid"])["harvest"]
        assert not os.path.samefile(src, dest)  # a copy, not a link or a move
        assert src.exists()
        assert fa.md5_file(src) == fa.md5_file(dest) == r["md5"]
        assert int(r["bytes"]) == dest.stat().st_size
    # a harvested pid is not harvested again
    assert fa.harvest(row, env["cf"])["status"] == "harvested"


def test_harvest_mapq_keeps_unfiltered_bams(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__mapq__10", monkeypatch)
    fa.harvest(row, env["cf"])
    recs = {r["role"]: r for r in _harvest_tsv(fa, env, row["pid"])}
    assert recs["unfiltered_bam"]["dest"].endswith("ENCFF254LWX.merged.srt.bam")
    assert recs["unfiltered_ctl_bam"]["dest"].endswith("ENCFF433TZR.merged.srt.bam")


def test_harvest_dedup_off_takes_filt_bam(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__dedup__off", monkeypatch)
    fa.harvest(row, env["cf"])
    recs = {r["role"]: r for r in _harvest_tsv(fa, env, row["pid"])}
    assert recs["treat_bam"]["dest"].endswith("ENCFF254LWX.merged.srt.filt.bam")
    assert "unfiltered_bam" not in recs


def test_harvest_refuses_two_distinct_matches(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__crop__50", monkeypatch)
    meta = json.loads(Path(fa.read_state(env["cf"], row["pid"])["metadata_json"]).read_text())
    ex = Path(meta["calls"]["chip.bam2ta"][0]["callRoot"]) / "execution"
    (ex / "stray.tagAlign.gz").write_text("another file")
    with pytest.raises(SystemExit, match="2 distinct files"):
        fa.harvest(row, env["cf"])
    assert fa.read_state(env["cf"], row["pid"])["status"] == "succeeded"


def test_harvest_refuses_basename_not_in_metadata(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__crop__50", monkeypatch)
    mpath = Path(fa.read_state(env["cf"], row["pid"])["metadata_json"])
    meta = json.loads(mpath.read_text())
    meta["calls"]["chip.macs2_signal_track"][0]["outputs"]["pval_bw"] = "/x/other.pval.signal.bigwig"
    mpath.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="output pval_bw"):
        fa.harvest(row, env["cf"])


def test_harvest_takes_final_attempt(fa, env, monkeypatch):
    row = _succeeded(fa, env, "C19M16__crop__50", monkeypatch)
    mpath = Path(fa.read_state(env["cf"], row["pid"])["metadata_json"])
    meta = json.loads(mpath.read_text())
    first = meta["calls"]["chip.xcor"][0]
    ex2 = Path(first["callRoot"]) / "attempt-2" / "execution"
    ex2.mkdir(parents=True)
    name = Path(first["outputs"]["score"]).name
    (ex2 / name).write_text("attempt 2 score")
    meta["calls"]["chip.xcor"].append(dict(first, attempt=2, callRoot=str(ex2.parent),
                                           outputs={"score": str(ex2 / name)}))
    mpath.write_text(json.dumps(meta))
    fa.harvest(row, env["cf"])
    recs = {r["role"]: r for r in _harvest_tsv(fa, env, row["pid"])}
    assert "/attempt-2/" in recs["xcor_qc"]["src"]
    assert Path(recs["xcor_qc"]["dest"]).read_text() == "attempt 2 score"


def test_harvest_skips_unsucceeded(fa, env, monkeypatch):
    row = _submitted(fa, env, "C19M16__crop__50", monkeypatch)
    assert fa.harvest(row, env["cf"])["status"] == "submitted"
    assert not fa.paths(env["cf"], row["pid"])["harvest"].exists()


# --- cleanup candidates and CLI ----------------------------------------------------------------

def test_cleanup_candidates_lists_and_deletes_nothing(fa, env, monkeypatch):
    done = _succeeded(fa, env, "C19M16__mapq__0", monkeypatch)
    fa.harvest(done, env["cf"])
    _succeeded(fa, env, "C19M16__mapq__10", monkeypatch)  # succeeded, not harvested
    before = sorted(str(p) for p in env["cf"].rglob("*"))
    out = fa.cleanup_candidates(fa.read_rows(env["rows"]), env["cf"])
    after = sorted(str(p) for p in env["cf"].rglob("*") if p != out)
    assert before == after
    lines = out.read_text().rstrip("\n").split("\n")
    assert lines[0] == "pid\ttree\tbytes\tharvest_verified"
    recs = [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]]
    got = {(r["pid"], Path(r["tree"]).name, r["harvest_verified"]) for r in recs}
    assert got == {("C19M16__mapq__0", "C19M16__mapq__0", "true"),
                   ("C19M16__mapq__10", "C19M16__mapq__10", "false")}
    assert {Path(r["tree"]).parent.name for r in recs} == {"cromwell", "loc"}
    assert all(int(r["bytes"]) > 0 for r in recs)

    # a keeper that changed after harvest is no longer verified
    dest = Path(_harvest_tsv(fa, env, "C19M16__mapq__0")[0]["dest"])
    dest.write_text("corrupted")
    assert fa.harvest_verified(env["cf"], "C19M16__mapq__0") is False


def test_cli_stage_subset(fa, env):
    r = subprocess.run([sys.executable, str(TOOL), "stage", "--rows", str(env["rows"]),
                        "--cf", str(env["cf"]), "--eic", str(env["eic"]), "--refcache", str(env["ref"]),
                        "--pids", "C19M16__crop__36,C19M16__pe__pe"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    states = sorted(p.name for p in (env["cf"] / "fastqarms" / "state").iterdir())
    assert states == ["C19M16__crop__36.json", "C19M16__pe__pe.json"]
    r = subprocess.run([sys.executable, str(TOOL), "poll", "--rows", str(env["rows"]),
                        "--cf", str(env["cf"]), "--pids", "C19M16__nope__x"], capture_output=True, text=True)
    assert r.returncode != 0 and "not in the rows TSV" in r.stderr


def test_cli_stage_genome_tsv(fa, env, gtsv):
    for argv in ([str(TOOL), "--help"], [str(TOOL), "stage", "--help"]):
        r = subprocess.run([sys.executable] + argv, capture_output=True, text=True)
        assert r.returncode == 0 and "--genome-tsv" in r.stdout, argv
    r = subprocess.run([sys.executable, str(TOOL), "stage", "--rows", str(env["rows"]),
                        "--cf", str(env["cf"]), "--eic", str(env["eic"]), "--refcache", str(env["ref"]),
                        "--pids", "C19M16__mapq__10", "--genome-tsv", str(gtsv)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    st = json.loads((env["cf"] / "fastqarms" / "state" / "C19M16__mapq__10.json").read_text())
    assert st["genome_tsv"] == str(gtsv)
    assert st["genome_tsv_md5"] == hashlib.md5(gtsv.read_bytes()).hexdigest()
    assert json.loads((env["cf"] / "fastqarms" / "inputs" / "C19M16__mapq__10.json").read_text()
                      )["chip.genome_tsv"] == str(gtsv)
