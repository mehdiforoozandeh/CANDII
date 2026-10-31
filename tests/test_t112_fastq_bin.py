"""`slurm/t112/fastq_bin.sh` — the FASTQ-route binning array task.

Nothing here reaches Nibi and nothing here runs `bin25.py` (the laptop has no pyBigWig). Two halves
are tested: the row → command logic, through the script's own `DRY_RUN=1` mode on a fake `$CF` with
a fake harvest, and the record-writing python, which is extracted verbatim from the script between
its two marker comments and run against the same fake harvest. The records it writes are then
handed to the real `records.py validate`, so the product schema is checked by the module that owns
it rather than re-asserted here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SH = REPO / "slurm" / "t112" / "fastq_bin.sh"
ARMS = REPO / "tools" / "t112" / "arms.py"
RECORDS = REPO / "tools" / "t112" / "records.py"

SIF = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/sif/chip-seq-pipeline_v2.2.2.sif"
ATAC_SIF = ("/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/sif/"
            "atac-seq-pipeline_v2.2.3.sif")
WFID = "f4e3dc0f-be72-4154-848c-f2d0ad614e7d"
LEADER = "22150583"

#: one line of a real `*.cc.qc` (C19M16's, read on Nibi 2026-09-17): field 3 is the fragment length
#: the pipeline used, and 180 is what `arms.py` records for that track.
CC_QC = ("ENCFF254LWX.merged.trim_50bp.srt.filt.no_chrM.15M.tagAlign.gz\t15000000\t180\t"
         "0.172871051979256\t55\t0.1670181\t1500\t0.1612659\t1.071963\t2.017525\t2\n")

CHIP_ROLES = ("treat_bam", "treat_bai", "ctl_bam", "ctl_bai", "treat_ta", "ctl_ta", "pval_bigwig",
              "xcor_qc", "qc_json", "metadata")
DNASE_ROLES = ("treat_bam", "treat_bai", "treat_ta", "pval_bigwig", "qc_json", "metadata")

TREAT_LINES = 28_000_000
CTL_LINES = 107_039_349


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def records():
    return _load(RECORDS, "t112_records")


def _rows(tmp_path, track, dnase, name):
    tsv = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "fastq", "--dnase", dnase,
                          "--ratio", "yes", "--tracks", track],
                         capture_output=True, text=True, check=True).stdout
    path = tmp_path / name
    path.write_text(tsv)
    return path


def _harvest(cf, pid, roles, prefix):
    """A fake `harvest/<pid>/` with one file per role and the HARVEST.tsv `fastq_arms.py` writes."""
    out = cf / "fastqarms" / "harvest" / pid
    out.mkdir(parents=True)
    names = {
        "treat_bam": f"{prefix}.merged.srt.nodup.bam",
        "treat_bai": f"{prefix}.merged.srt.nodup.bam.bai",
        "ctl_bam": "ENCFF433TZR.merged.srt.nodup.bam",
        "ctl_bai": "ENCFF433TZR.merged.srt.nodup.bam.bai",
        "treat_ta": f"{prefix}.merged.srt.nodup.30M.tagAlign.gz",
        "ctl_ta": "ENCFF433TZR.merged.srt.nodup.tagAlign.gz",
        "pval_bigwig": f"{prefix}.merged.srt.nodup.30M.pval.signal.bigwig",
        "xcor_qc": f"{prefix}.merged.trim_50bp.srt.filt.no_chrM.15M.cc.qc",
        "qc_json": "qc.json",
        "metadata": "metadata.json",
    }
    lines = ["role\tsrc\tdest\tbytes\tmd5"]
    for role in roles:
        dest = out / names[role]
        if role == "xcor_qc":
            dest.write_text(CC_QC)
        elif role == "metadata":
            dest.write_text(json.dumps({"id": WFID, "status": "Succeeded"}) + "\n")
        else:
            dest.write_text(f"{role} bytes for {pid}\n")
        src = cf / "fastqarms" / "cromwell" / pid / "chip" / WFID / f"call-{role}" / dest.name
        lines.append("\t".join([role, str(src), str(dest), str(dest.stat().st_size),
                                hashlib.md5(dest.read_bytes()).hexdigest()]))
    (out / "HARVEST.tsv").write_text("\n".join(lines) + "\n")
    return out


@pytest.fixture()
def env(tmp_path):
    """A fake `$CF` holding one harvested ChIP pid and one harvested DNase pid, plus a chrom.sizes."""
    cf = tmp_path / "cf"
    fa = cf / "fastqarms"
    (fa / "inputs").mkdir(parents=True)
    (fa / "state").mkdir(parents=True)
    (fa / "genome").mkdir(parents=True)
    gtsv = fa / "genome" / "hg38.local.tsv"
    gtsv.write_text("hg38\tbwa_idx_tar\t/scratch/refcache/hg38.bwa.tar\n")

    for pid, pfx, sif, roles, prefix in (
            ("C19M16__crop__36", "chip", SIF, CHIP_ROLES, "ENCFF254LWX"),
            ("C12M02__mapq__0", "atac", ATAC_SIF, DNASE_ROLES, "ENCFF211XVI")):
        ij = fa / "inputs" / f"{pid}.json"
        ij.write_text(json.dumps({f"{pfx}.title": f"CF {pid}", f"{pfx}.singularity": sif,
                                  f"{pfx}.genome_tsv": str(gtsv)}, indent=2) + "\n")
        _harvest(cf, pid, roles, prefix)
        (fa / "state" / f"{pid}.json").write_text(json.dumps({
            "pid": pid, "status": "harvested", "leader_job_id": LEADER, "attempts": 1,
            "workflow_id": WFID,
            "metadata_json": str(fa / "cromwell" / pid / pfx / WFID / "metadata.json"),
            "input_json": str(ij), "loc_dir": str(fa / "loc" / pid),
            "genome_tsv": str(gtsv), "genome_tsv_md5": hashlib.md5(gtsv.read_bytes()).hexdigest(),
            "harvest_tsv": str(fa / "harvest" / pid / "HARVEST.tsv"),
            "updated_utc": "2026-09-18T03:00:00Z"}, indent=2) + "\n")

    chrsz = tmp_path / "GRCh38_EBV.chrom.sizes.tsv"
    chrsz.write_text("chr1\t248956422\nchr21\t46709983\nchrM\t16569\n")
    return {"cf": cf, "chrsz": chrsz, "tmp": tmp_path,
            "chip_rows": _rows(tmp_path, "C19M16", "none", "rows_chip.tsv"),
            "dnase_rows": _rows(tmp_path, "C12M02", "atac", "rows_dnase.tsv")}


def _dry_run(env, rows, index, **extra):
    e = dict(os.environ, DRY_RUN="1", SLURM_ARRAY_TASK_ID=str(index),
             SLURM_TMPDIR=str(env["tmp"] / "slurmtmp"), CHRSZ=str(env["chrsz"]))
    e.update(extra)
    return subprocess.run(["bash", str(SH), str(REPO), str(rows), str(env["cf"])],
                          env=e, capture_output=True, text=True)


def _plus(out):
    return [ln[2:] for ln in out.splitlines() if ln.startswith("+ ")]


# --- the script itself ---------------------------------------------------------------------------

def test_bash_n_is_clean():
    r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sbatch_header_is_the_planned_request():
    head = SH.read_text()
    for line in ("#SBATCH --account=def-maxwl",
                 "#SBATCH --time=4:00:00",
                 "#SBATCH --cpus-per-task=2",
                 "#SBATCH --mem=32G",
                 "#SBATCH --output=/scratch/mforooz/t112_cf/logs/fastq_bin/%x_%A_%a.out",
                 "#SBATCH --error=/scratch/mforooz/t112_cf/logs/fastq_bin/%x_%A_%a.err"):
        assert line in head, line
    # the array loses a comma-valued --export, and the plugin picks the partition from --time
    directives = [ln for ln in head.splitlines() if ln.startswith("#SBATCH")]
    assert not [ln for ln in directives if "--export" in ln or "--partition" in ln]
    assert not [ln for ln in directives if "--gres" in ln]          # CPU work


# --- row -> commands (DRY_RUN) --------------------------------------------------------------------

def test_dry_run_chip_commands_in_order(env):
    r = _dry_run(env, env["chip_rows"], 2)       # index 2 = C19M16__crop__36
    assert r.returncode == 0, r.stderr
    cmds = _plus(r.stdout)
    stage = str(env["tmp"] / "slurmtmp" / "t112_fastq_bin" / "C19M16__crop__36")
    harvest = str(env["cf"] / "fastqarms" / "harvest" / "C19M16__crop__36")

    assert cmds[0].startswith(f"python3 {REPO}/tools/t112/bin25.py counts --ta {harvest}/")
    assert ".30M.tagAlign.gz" in cmds[0] and f"--out {stage}/counts25.npz" in cmds[0]
    assert f"--chrsz {env['chrsz']}" in cmds[0]
    assert "counts --ta" in cmds[1] and "ENCFF433TZR.merged.srt.nodup.tagAlign.gz" in cmds[1]
    assert f"--out {stage}/control_counts25.npz" in cmds[1]
    assert "pval --bigwig" in cmds[2] and ".pval.signal.bigwig" in cmds[2]
    assert f"--tmpdir {stage}/bdg --out {stage}/pval25.npz" in cmds[2]
    assert cmds[3].startswith("python3 - # covariates.json + provenance.json")

    copies = [c for c in cmds if c.startswith("cp ")]
    assert [c.split()[2].split("/")[-1] for c in copies] == [
        "counts25.npz", "control_counts25.npz", "pval25.npz", "covariates.json", "provenance.json"]
    assert all(c.split()[1].startswith(stage) for c in copies)
    assert all(c.split()[2].startswith(str(env["cf"] / "products" / "C19M16__crop__36"))
               for c in copies)

    assert cmds[-1] == (f"python3 {REPO}/tools/t112/records.py validate "
                        f"--products {env['cf']}/products "
                        f"--rows {env['cf']}/fastqarms/bin/C19M16__crop__36/row.tsv --expect 1")
    # a dry run writes nothing
    assert not (env["cf"] / "products").exists()
    assert not (env["cf"] / "fastqarms" / "bin").exists()


def test_dry_run_dnase_has_no_control_and_no_xcor(env):
    r = _dry_run(env, env["dnase_rows"], 2)      # index 2 = C12M02__mapq__0
    assert r.returncode == 0, r.stderr
    cmds = _plus(r.stdout)
    assert len([c for c in cmds if "bin25.py counts" in c]) == 1
    assert not any("control_counts25" in c for c in cmds)
    assert any("bin25.py pval" in c for c in cmds)
    assert [c.split()[2].split("/")[-1] for c in cmds if c.startswith("cp ")] == [
        "counts25.npz", "pval25.npz", "covariates.json", "provenance.json"]


def test_a_kit_without_the_tools_is_refused(env, tmp_path):
    e = dict(os.environ, DRY_RUN="1", SLURM_ARRAY_TASK_ID="0")
    r = subprocess.run(["bash", str(SH), str(tmp_path), str(env["chip_rows"]), str(env["cf"])],
                       env=e, capture_output=True, text=True)
    assert r.returncode == 2 and "has no tools/t112/bin25.py" in r.stderr


def test_a_bam_route_rows_tsv_is_refused(env, tmp_path):
    rows = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "bam", "--dnase", "none",
                           "--ratio", "yes", "--tracks", "C19M16"],
                          capture_output=True, text=True, check=True).stdout
    path = tmp_path / "rows_bam.tsv"
    path.write_text(rows)
    r = _dry_run(env, path, 0)
    assert r.returncode == 2 and "not fastq" in r.stderr


def test_an_unharvested_pid_is_refused(env):
    r = _dry_run(env, env["chip_rows"], 0)       # index 0 = C19M16__pe__pe, never harvested
    assert r.returncode == 2 and "HARVEST.tsv" in r.stderr


def test_a_chip_harvest_missing_a_role_is_refused(env):
    tsv = env["cf"] / "fastqarms" / "harvest" / "C19M16__crop__36" / "HARVEST.tsv"
    kept = [ln for ln in tsv.read_text().rstrip("\n").split("\n") if not ln.startswith("xcor_qc\t")]
    tsv.write_text("\n".join(kept) + "\n")
    r = _dry_run(env, env["chip_rows"], 2)
    assert r.returncode == 2 and "no 'xcor_qc' role" in r.stderr


def test_an_atac_harvest_that_lists_a_control_is_refused(env):
    tsv = env["cf"] / "fastqarms" / "harvest" / "C12M02__mapq__0" / "HARVEST.tsv"
    tsv.write_text(tsv.read_text() + "ctl_ta\t/x/src\t/x/dest\t1\t" + "0" * 32 + "\n")
    r = _dry_run(env, env["dnase_rows"], 2)
    assert r.returncode == 2 and "no control and no xcor" in r.stderr


def test_an_index_past_the_last_row_is_refused(env):
    r = _dry_run(env, env["chip_rows"], 99)
    assert r.returncode == 2 and "no row at index 99" in r.stderr


# --- the records step, extracted from the script --------------------------------------------------

BEGIN = "# >>> records step (extracted verbatim by tests/test_t112_fastq_bin.py) >>>"
END = "# <<< records step <<<"


def _records_step(tmp_path) -> Path:
    body = SH.read_text().split(BEGIN)[1].split(END)[0]
    out = tmp_path / "records_step.py"
    out.write_text(body)
    return out


def _stage(env, pid, control=True):
    """What `bin25.py` leaves behind for the records step: three npz files and their stats."""
    stage = env["tmp"] / "stage" / pid
    stage.mkdir(parents=True)
    names = ["counts25"] + (["control_counts25"] if control else []) + ["pval25"]
    for n in names:
        (stage / f"{n}.npz").write_bytes(b"PK\x03\x04 fake npz for " + n.encode())
    (stage / "counts25_stats.json").write_text(json.dumps(
        {"n_lines": TREAT_LINES, "n_lines_main": TREAT_LINES - 10, "per_chrom": {}}))
    if control:
        (stage / "control_counts25_stats.json").write_text(json.dumps(
            {"n_lines": CTL_LINES, "n_lines_main": CTL_LINES - 10, "per_chrom": {}}))
    (stage / "pval25_stats.json").write_text(json.dumps(
        {"n_lines": 5, "n_lines_main": 5, "per_chrom": {}}))
    (stage / "commands.txt").write_text("python3 bin25.py counts --ta t.tagAlign.gz\n"
                                        "python3 bin25.py pval --bigwig t.bigwig\n")
    return stage


def _run_records_step(env, tmp_path, pid, rows, stage):
    cf = env["cf"]
    e = dict(os.environ,
             T112_KIT=str(REPO), T112_PID=pid,
             T112_ROW=[ln for ln in rows.read_text().splitlines() if ln.startswith(pid + "\t")][0],
             T112_STAGE=str(stage), T112_PRODUCT=str(cf / "products" / pid),
             T112_HARVEST_TSV=str(cf / "fastqarms" / "harvest" / pid / "HARVEST.tsv"),
             T112_STATE=str(cf / "fastqarms" / "state" / f"{pid}.json"),
             T112_CMDLOG=str(stage / "commands.txt"),
             T112_GIT_SHA="d396cc7", SLURM_ARRAY_JOB_ID="22200000", SLURM_ARRAY_TASK_ID="2")
    r = subprocess.run([sys.executable, str(_records_step(tmp_path))], env=e,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return (json.loads((stage / "covariates.json").read_text()),
            json.loads((stage / "provenance.json").read_text()))


def test_records_step_chip(env, tmp_path):
    pid = "C19M16__crop__36"
    stage = _stage(env, pid)
    cov, prov = _run_records_step(env, tmp_path, pid, env["chip_rows"], stage)

    assert cov["pid"] == pid and cov["biosample"] == "CF_C19__crop__36"
    assert cov["assay"] == "H3K27ac" and cov["arm"] == "crop" and cov["level"] == "36"
    assert cov["knob"] == "chip.crop_length" and cov["knob_value"] == 36
    assert cov["depth"] == TREAT_LINES
    assert cov["read_length"] == 36          # the crop value, not the track's 101
    assert cov["run_type"] == "single-ended"
    assert cov["fraglen"] == 180             # field 3 of the cc.qc
    assert cov["control"] == {"accession": "ENCFF433TZR", "source": "matched",
                              "reads": CTL_LINES}

    assert prov["route"] == "fastq"
    assert prov["pipeline"]["repo"] == "ENCODE-DCC/chip-seq-pipeline2"
    assert prov["pipeline"]["release"] == "v2.2.2" and prov["pipeline"]["sif"] == SIF
    assert prov["caper"]["workflow_id"] == WFID
    # the harvested copy, which survives the D3 cleanup — never the one in the Cromwell tree
    assert prov["caper"]["metadata_json"] == str(
        env["cf"] / "fastqarms" / "harvest" / pid / "metadata.json")
    assert prov["caper"]["input_json"].endswith(f"inputs/{pid}.json")
    assert prov["slurm_job_ids"] == [LEADER, "22200000_2"]
    assert prov["code"] == {"snapshot_dir": str(REPO), "git_sha": "d396cc7"}
    assert prov["commands"] == ["python3 bin25.py counts --ta t.tagAlign.gz",
                                "python3 bin25.py pval --bigwig t.bigwig"]
    assert prov["patched_script"] is None and prov["subsample_seed"] == []

    got = {Path(i["path"]).name for i in prov["inputs"]}
    assert got == {f"{pid}.json", "hg38.local.tsv", "metadata.json",
                   "ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz",
                   "ENCFF433TZR.merged.srt.nodup.tagAlign.gz",
                   "ENCFF254LWX.merged.srt.nodup.30M.pval.signal.bigwig",
                   "ENCFF254LWX.merged.trim_50bp.srt.filt.no_chrM.15M.cc.qc"}
    outs = {Path(o["path"]).name: o["md5"] for o in prov["outputs"]}
    assert set(outs) == {"counts25.npz", "control_counts25.npz", "pval25.npz"}
    assert all(str(Path(o["path"]).parent).endswith(f"products/{pid}") for o in prov["outputs"])
    assert outs["counts25.npz"] == hashlib.md5((stage / "counts25.npz").read_bytes()).hexdigest()


def test_records_step_dnase(env, tmp_path):
    pid = "C12M02__mapq__0"
    stage = _stage(env, pid, control=False)
    cov, prov = _run_records_step(env, tmp_path, pid, env["dnase_rows"], stage)
    assert cov["control"] is None
    assert cov["fraglen"] == 150             # atac.smooth_win: a dnase run has no xcor call
    assert cov["read_length"] == 76 and cov["run_type"] == "single-ended"
    assert cov["knob"] == "atac.mapq_thresh" and cov["knob_value"] == 0
    assert prov["pipeline"]["repo"] == "ENCODE-DCC/atac-seq-pipeline"
    assert prov["pipeline"]["sif"] == ATAC_SIF
    assert {Path(o["path"]).name for o in prov["outputs"]} == {"counts25.npz", "pval25.npz"}


def test_records_step_pe_arm_is_paired_ended(env, tmp_path):
    pid = "C19M16__pe__pe"
    _harvest(env["cf"], pid, CHIP_ROLES, "ENCFF254LWX")
    src = env["cf"] / "fastqarms" / "state" / "C19M16__crop__36.json"
    st = json.loads(src.read_text())
    ij = env["cf"] / "fastqarms" / "inputs" / f"{pid}.json"
    ij.write_text((env["cf"] / "fastqarms" / "inputs" / "C19M16__crop__36.json").read_text())
    st.update(pid=pid, input_json=str(ij),
              harvest_tsv=str(env["cf"] / "fastqarms" / "harvest" / pid / "HARVEST.tsv"))
    (env["cf"] / "fastqarms" / "state" / f"{pid}.json").write_text(json.dumps(st, indent=2))
    cov, _ = _run_records_step(env, tmp_path, pid, env["chip_rows"], _stage(env, pid))
    assert cov["run_type"] == "paired-ended"
    assert cov["read_length"] == 101 and cov["knob_value"] is True


def test_records_step_refuses_a_keeper_that_lost_bytes(env, tmp_path):
    pid = "C19M16__crop__36"
    stage = _stage(env, pid)
    ta = next((env["cf"] / "fastqarms" / "harvest" / pid).glob("*.30M.tagAlign.gz"))
    ta.write_text("truncated")
    with pytest.raises(AssertionError):
        _run_records_step(env, tmp_path, pid, env["chip_rows"], stage)


def test_the_written_records_pass_records_py_validate(env, tmp_path, records):
    pid = "C19M16__crop__36"
    stage = _stage(env, pid)
    _run_records_step(env, tmp_path, pid, env["chip_rows"], stage)
    product = env["cf"] / "products" / pid
    product.mkdir(parents=True)
    for n in ("counts25.npz", "control_counts25.npz", "pval25.npz", "covariates.json",
              "provenance.json"):
        (product / n).write_bytes((stage / n).read_bytes())
    row = tmp_path / "row.tsv"
    lines = env["chip_rows"].read_text().splitlines()
    row.write_text(lines[0] + "\n" + next(ln for ln in lines if ln.startswith(pid + "\t")) + "\n")

    n, probs = records.validate(env["cf"] / "products", rows_tsv=row, expect=1)
    assert probs == [] and n == 1
