"""t112 C8 — `tools/t112/bam_arm.py`, the BAM-route arm builder.

What these cover, and why each one is here rather than left to the cluster smoke:

* the names the pipeline's own helpers produce (`human_readable_number`, `subsample_ta_se`), since
  every arm reads or writes a file by a name the pipeline chose and a wrong one fails only at 8 h
  into an 84-task array;
* `plan_commands` per arm: one command list per arm kind, against a fake `$EIC` tree holding the
  recorded `commandLine`, so the substitutions (tagAligns, `--fraglen`, `--chrsz`, `--out-dir`) are
  checked without a cluster;
* the `--ratio` value: computed from the line counts of the two tagAligns actually fed to MACS2 —
  `arms.CONTROLS[*]["reads"]` is a pinned number and was wrong once already;
* the one-line guarantee of the ratio patch, on a real copy of the pipeline's source line.

The laptop has no pyBigWig, so nothing here imports it; `bin25` is imported only inside `run`.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "t112"


def _load(name):
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(f"t112_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bam_arm = _load("bam_arm")
arms = _load("arms")

TRACK = "C19M16"
CHRSZ = bam_arm.CHRSZ

#: the shape of the real recorded command (C19M16, read from Nibi 2026-09-17), with the Cromwell
#: input paths that no longer exist.
RECORDED = (
    "set -e\npython3 $(which encode_task_macs2_signal_track_chip.py) \\\n"
    "    /crom/inputs/1427283357/ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz"
    " /crom/inputs/1411444345/ENCFF433TZR.merged.srt.nodup.tagAlign.gz \\\n"
    "    --gensz hs \\\n    --chrsz /crom/inputs/1706320006/GRCh38_EBV.chrom.sizes.tsv \\\n"
    "    --fraglen 180 \\\n    --pval-thresh 0.01 \\\n    --mem-gb 14.932721015065908"
)


@pytest.fixture
def eic(tmp_path):
    """A fake `$EIC` holding only what `plan` reads: one track's macs2_signal_track commandLine."""
    d = tmp_path / "eic" / "results" / TRACK
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(
        {"calls": {"chip.macs2_signal_track": [{"commandLine": RECORDED,
                                                "executionStatus": "Done"}]}}))
    return str(tmp_path / "eic")


@pytest.fixture
def ta(tmp_path):
    """A `$CF/ta` with tiny stand-ins at the exact names C5 writes, so line counts are real."""
    d = tmp_path / "cf" / "ta"
    (d / "ctldepth" / "ENCFF433TZR__q0.5").mkdir(parents=True)

    def write(path, n, start=1000):
        with gzip.open(path, "wt") as f:
            for i in range(n):
                f.write(f"chr21\t{start + i}\t{start + i + 101}\tN\t1000\t+\n")

    write(d / "ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz", 40)
    write(d / "ENCFF433TZR.merged.srt.nodup.tagAlign.gz", 130)
    write(d / "ENCFF337JNL.merged.srt.nodup.tagAlign.gz", 150)
    write(d / "ctldepth" / "ENCFF433TZR__q0.5" / "ENCFF433TZR.merged.srt.nodup.53M.tagAlign.gz", 65)
    return str(d)


def row(arm, level):
    rows = {(r["arm"], r["level"]): r
            for r in arms.rows("bam", dnase="none", ratio="yes", tracks=[TRACK])}
    r = rows[(arm, level)]
    return {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in r.items()}


def commands(arm, level, eic, ta, tmp_path, **kw):
    return bam_arm.plan_commands(row(arm, level), str(tmp_path / "cf"), eic,
                                 tmp=str(tmp_path / "tmp"), ta_dir=ta, **kw)


# ---------------------------------------------------------------------------- pipeline names

def test_human_readable_number_matches_the_pipeline():
    # encode_lib_common.human_readable_number: integer division at every step, so 7.5M -> '7M'.
    assert bam_arm.human_readable_number(30000000) == "30M"
    assert bam_arm.human_readable_number(15000000) == "15M"
    assert bam_arm.human_readable_number(7500000) == "7M"
    assert bam_arm.human_readable_number(3750000) == "3M"
    assert bam_arm.human_readable_number(53519675) == "53M"
    assert bam_arm.human_readable_number(999) == "999"


def test_subsample_output_matches_subsample_ta_se():
    out = bam_arm.subsample_output("/ta/ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz",
                                   15000000, "/out")
    assert out == "/out/ENCFF254LWX.merged.srt.nodup.30M.15M.tagAlign.gz"


def test_base_tagalign_names_are_c5s(ta):
    assert bam_arm.treatment_tagalign(TRACK, ta) == \
        f"{ta}/ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz"
    assert bam_arm.control_tagalign("ENCFF433TZR", ta) == \
        f"{ta}/ENCFF433TZR.merged.srt.nodup.tagAlign.gz"
    # DNase keeps the .no_chrM_MT stem and the 50M subsample
    assert bam_arm.treatment_tagalign("C12M02", ta).endswith(
        "ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M.tagAlign.gz")


def test_ctldepth_falls_back_to_the_one_file_in_the_directory(ta):
    n = json.loads(row("ctldepth", "q0.5")["knob_value"])
    assert bam_arm.ctldepth_tagalign("ENCFF433TZR", "q0.5", n, ta).endswith(
        "ctldepth/ENCFF433TZR__q0.5/ENCFF433TZR.merged.srt.nodup.53M.tagAlign.gz")
    # a different N: the name no longer matches, but the directory identifies (control, q)
    got = bam_arm.ctldepth_tagalign("ENCFF433TZR", "q0.5", 29036785, ta)
    assert got.endswith("ENCFF433TZR.merged.srt.nodup.53M.tagAlign.gz")


def test_signal_bigwig_name(ta):
    T = bam_arm.treatment_tagalign(TRACK, ta)
    C = bam_arm.control_tagalign("ENCFF433TZR", ta)
    assert bam_arm.signal_bigwig([T, C], "/o") == (
        "/o/ENCFF254LWX.merged.srt.nodup.30M_x_ENCFF433TZR.merged.srt.nodup"
        ".pval.signal.bigwig")
    assert bam_arm.signal_bigwig([T], "/o") == \
        "/o/ENCFF254LWX.merged.srt.nodup.30M.pval.signal.bigwig"


# ---------------------------------------------------------------------------- the recorded command

def test_parse_signal_command_splits_the_recorded_line():
    p = bam_arm.parse_signal_command(RECORDED)
    assert p["invocation"][0] == "python3"
    assert len(p["tas"]) == 2 and all(t.endswith(".tagAlign.gz") for t in p["tas"])
    assert p["flags"] == [("--gensz", "hs"),
                          ("--chrsz", "/crom/inputs/1706320006/GRCh38_EBV.chrom.sizes.tsv"),
                          ("--fraglen", "180"), ("--pval-thresh", "0.01"),
                          ("--mem-gb", "14.932721015065908")]


def test_build_signal_command_replaces_only_what_it_may():
    p = bam_arm.parse_signal_command(RECORDED)
    cmd = bam_arm.build_signal_command(p, ["/ta/T.tagAlign.gz", "/ta/C.tagAlign.gz"], "/out",
                                       fraglen=360)
    assert cmd == ("python3 $(which encode_task_macs2_signal_track_chip.py) "
                   "/ta/T.tagAlign.gz /ta/C.tagAlign.gz --gensz hs "
                   f"--chrsz {CHRSZ} --fraglen 360 --pval-thresh 0.01 "
                   "--mem-gb 14.932721015065908 --out-dir /out")
    # nothing but --fraglen moved when it is not given
    assert "--fraglen 180" in bam_arm.build_signal_command(p, ["/ta/T.tagAlign.gz"], "/out")


def test_build_signal_command_refuses_foreign_recorded_values():
    """`--gensz hs` and `--pval-thresh 0.01` are the plan's pinned values, not just flags."""
    for bad, good in (("--gensz hs", "--gensz mm"), ("--pval-thresh 0.01", "--pval-thresh 0.05")):
        p = bam_arm.parse_signal_command(RECORDED.replace(bad, good))
        with pytest.raises(ValueError, match="recorded --"):
            bam_arm.build_signal_command(p, ["/ta/T.tagAlign.gz"], "/out")


def test_build_signal_command_refuses_a_foreign_chrsz():
    p = bam_arm.parse_signal_command(RECORDED)
    with pytest.raises(ValueError, match="basename"):
        bam_arm.build_signal_command(p, ["/ta/T.tagAlign.gz"], "/out", chrsz="/ref/hg19.sizes")


# ---------------------------------------------------------------------------- per-arm plans

def test_base_runs_the_signal_step_and_nothing_else(eic, ta, tmp_path):
    cs = commands("base", "base", eic, ta, tmp_path)
    assert len(cs) == 4 and cs[0].startswith("mkdir -p ")
    assert "apptainer exec --cleanenv" in cs[1] and "--pwd " in cs[1]
    assert "encode_task_macs2_signal_track_chip.py" in cs[1]
    assert "ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz" in cs[1]
    assert "ENCFF433TZR.merged.srt.nodup.tagAlign.gz" in cs[1]
    assert cs[2].startswith("test -s ")          # nothing leaves $SLURM_TMPDIR unverified
    assert cs[3].startswith("cp ") and cs[3].endswith("/")
    p = bam_arm.plan(row("base", "base"), str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                     ta_dir=ta)
    # base alone bins the pipeline's own bigwig and keeps a rebuild beside it
    assert p["pipeline_bigwig"] == arms.TRACKS[TRACK]["pval_bigwig"]
    assert p["rebuild_npz"].endswith("/C19M16__base__base/rebuild_pval25.npz")


def test_depth_thins_with_the_pipeline_subsampler(eic, ta, tmp_path):
    cs = commands("depth", "15M", eic, ta, tmp_path)
    sub = [c for c in cs if "encode_task_subsample_ctl.py" in c]
    assert len(sub) == 1
    assert "--subsample 15000000" in sub[0]
    assert "ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz --subsample" in sub[0]
    # MACS2 then reads the thinned treatment and the UNCHANGED control
    sig = [c for c in cs if "macs2_signal_track" in c][0]
    assert "ENCFF254LWX.merged.srt.nodup.30M.15M.tagAlign.gz" in sig
    assert "ENCFF433TZR.merged.srt.nodup.tagAlign.gz" in sig


def test_abproxy_mixes_two_draws_to_the_same_total(eic, ta, tmp_path):
    cs = commands("abproxy", "f0.5", eic, ta, tmp_path)
    sub = [c for c in cs if "encode_task_subsample_ctl.py" in c]
    assert len(sub) == 2
    assert "--subsample 15000000" in sub[0] and "ENCFF254LWX" in sub[0]
    assert "--subsample 15000000" in sub[1] and "ENCFF433TZR" in sub[1]
    cat = [c for c in cs if "zcat" in c][0]
    assert "gzip -nc" in cat and "abproxy_f0.5.tagAlign.gz" in cat
    sig = [c for c in cs if "macs2_signal_track" in c][0]
    assert "abproxy_f0.5.tagAlign.gz" in sig
    assert "ENCFF433TZR.merged.srt.nodup.tagAlign.gz" in sig     # control unchanged


def test_abproxy_f09_keeps_the_total_at_30m(eic, ta, tmp_path):
    cs = commands("abproxy", "f0.9", eic, ta, tmp_path)
    ns = [int(c.split("--subsample ")[1].split()[0]) for c in cs if "subsample_ctl" in c]
    assert ns == [3000000, 27000000] and sum(ns) == 30000000


def test_ctlid_other_rotates_the_control_and_none_drops_it(eic, ta, tmp_path):
    other = [c for c in commands("ctlid", "other", eic, ta, tmp_path)
             if "macs2_signal_track" in c][0]
    assert "ENCFF337JNL.merged.srt.nodup.tagAlign.gz" in other
    assert "ENCFF433TZR" not in other
    none = [c for c in commands("ctlid", "none", eic, ta, tmp_path)
            if "macs2_signal_track" in c][0]
    assert ".tagAlign.gz" in none and none.count(".tagAlign.gz") == 1
    p = bam_arm.plan(row("ctlid", "none"), str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                     ta_dir=ta)
    assert p["control"] is None and p["control_accession"] is None
    assert p["kept_bigwig"].endswith("ENCFF254LWX.merged.srt.nodup.30M.pval.signal.bigwig")


def test_ctldepth_uses_c5s_thinned_control(eic, ta, tmp_path):
    sig = [c for c in commands("ctldepth", "q0.5", eic, ta, tmp_path)
           if "macs2_signal_track" in c][0]
    assert "ctldepth/ENCFF433TZR__q0.5/" in sig
    assert "ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz" in sig   # treatment unchanged


def test_extsize_turns_only_fraglen(eic, ta, tmp_path):
    for level, want in (("k0.5", 90), ("k2", 360)):
        sig = [c for c in commands("extsize", level, eic, ta, tmp_path)
               if "macs2_signal_track" in c][0]
        assert f"--fraglen {want}" in sig
    base = [c for c in commands("base", "base", eic, ta, tmp_path) if "macs2_signal_track" in c][0]
    assert "--fraglen 180" in base


# ---------------------------------------------------------------------------- the ratio arm

def test_ratio_is_computed_from_the_files_not_from_a_pinned_count(ta):
    # the fixture is 40 treatment lines against 130 control lines
    assert bam_arm.ratio_for("k1", bam_arm.treatment_tagalign(TRACK, ta),
                             bam_arm.control_tagalign("ENCFF433TZR", ta), sig=None) == 40 / 130
    assert bam_arm.ratio_for("k0.5", bam_arm.treatment_tagalign(TRACK, ta),
                             bam_arm.control_tagalign("ENCFF433TZR", ta)) == \
        float(f"{0.5 * 40 / 130:.10g}")
    assert bam_arm.ratio_k("k0.5") == 0.5 and bam_arm.ratio_k("k2") == 2.0
    with pytest.raises(ValueError):
        bam_arm.ratio_k("other")


def test_ratio_rounding_matches_arms_ratio_value():
    # both round the same quotient the same way, so covariates.knob_value == the rows TSV column
    assert arms.ratio_value(0.5, 107039349) == float(f"{0.5 * 30000000 / 107039349:.10g}")


def test_ratio_arm_copies_patches_and_runs_the_patched_script(eic, ta, tmp_path):
    cs = commands("ratio", "k0.5", eic, ta, tmp_path)
    assert cs[0].startswith("mkdir -p ") and "/patched" in cs[0]
    assert f"cp {bam_arm.RATIO_SCRIPT} " in cs[1]
    assert cs[2].startswith("python3 -c ") and "--SPMR" in cs[2]
    sig = cs[3]
    assert f"PYTHONPATH={bam_arm.CHIP_SRC_DIR} python3 " in sig
    assert "$(which" not in sig
    # --ratio is edited INTO the script, so it is not a CLI flag the signal script would reject
    assert "--ratio" not in sig
    # the value handed to MACS2 is k times the quotient of the two files, not the TSV column
    p = bam_arm.plan(row("ratio", "k0.5"), str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                     ta_dir=ta)
    assert p["knob_value"] == float(f"{0.5 * 40 / 130:.10g}")
    assert p["knob_value"] != json.loads(row("ratio", "k0.5")["knob_value"])
    assert repr(p["knob_value"]) in p["patch"]["new"]


def test_ratio_patch_changes_exactly_one_line(tmp_path, monkeypatch):
    """The real pipeline line, edited by the real one-liner, must move exactly one line."""
    src = tmp_path / "encode_task_macs2_signal_track_chip.py"
    src.write_text(
        "import x\n\n"
        "run_shell_cmd(\n"
        "    ' macs2 callpeak '\n"
        "    '-t {ta} {ctl_param} -f BED -n {prefix} -g {gensz} -p {pval_thresh} '\n"
        "    '--nomodel --shift {shiftsize} --extsize {extsize} --keep-dup all -B --SPMR'.format(\n"
        "        ta=ta,\n    )\n)\n")
    new = bam_arm.ratio_new(0.1401353814)
    text = src.read_text()
    assert text.count(bam_arm.RATIO_OLD) == 1
    src.write_text(text.replace(bam_arm.RATIO_OLD, new))

    original_md5 = hashlib.md5(text.encode()).hexdigest()
    got = bam_arm.apply_patch({"copy": str(src), "old": bam_arm.RATIO_OLD, "new": new,
                               "expect_md5": original_md5})
    assert got["original_md5"] != got["patched_md5"]
    assert got["diff"].count("\n-") + got["diff"].startswith("-") >= 1
    removed = [d for d in got["diff"].splitlines() if d.startswith("-") and not d.startswith("---")]
    added = [d for d in got["diff"].splitlines() if d.startswith("+") and not d.startswith("+++")]
    assert len(removed) == 1 and len(added) == 1
    assert "--ratio 0.1401353814" in added[0]
    # the edit inserts no new format field: `.format(ta=ta)` must still be satisfiable
    assert "{r}" not in added[0]


def test_apply_patch_rejects_a_file_it_did_not_edit(tmp_path):
    src = tmp_path / "s.py"
    src.write_text("nothing to see\n")
    with pytest.raises(SystemExit, match="single expected substitution"):
        bam_arm.apply_patch({"copy": str(src), "old": bam_arm.RATIO_OLD, "new": "x",
                             "expect_md5": bam_arm.RATIO_SCRIPT_MD5})


def test_apply_patch_refuses_a_script_whose_unpatched_md5_moved(tmp_path):
    """The plan pins the SIF's signal script at 6039e1c3...; a copy that reverses to anything else
    is a different recipe, and the arm must stop rather than record a false provenance."""
    body = ("run_shell_cmd(\n    ' macs2 callpeak '\n    '"
            + bam_arm.RATIO_OLD + ".format(\n        ta=ta,\n    )\n)\n")
    new = bam_arm.ratio_new(0.5)
    src = tmp_path / "encode_task_macs2_signal_track_chip.py"
    src.write_text(body.replace(bam_arm.RATIO_OLD, new))
    args = {"copy": str(src), "old": bam_arm.RATIO_OLD, "new": new}

    with pytest.raises(SystemExit, match="!= pinned " + bam_arm.RATIO_SCRIPT_MD5):
        bam_arm.apply_patch(dict(args, expect_md5=bam_arm.RATIO_SCRIPT_MD5))
    # the same file passes once the expectation is the md5 it really reverses to
    got = bam_arm.apply_patch(dict(args, expect_md5=hashlib.md5(body.encode()).hexdigest()))
    assert got["original_md5"] == hashlib.md5(body.encode()).hexdigest()


def test_the_ratio_plan_pins_the_sif_script_md5(eic, ta, tmp_path):
    p = bam_arm.plan(row("ratio", "k2"), str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                     ta_dir=ta)
    assert p["patch"]["expect_md5"] == bam_arm.RATIO_SCRIPT_MD5 == \
        "6039e1c382ef1e8afa3dbfd5ea103ba8"


# ---------------------------------------------------------------------- the DNase (atac) lane
#
# The atac pieces the ChIP tests never touch: a different signal script, `--smooth-win` where chip
# has `--fraglen`, one tagAlign because a DNase run has no control, and the atac image.

#: the real recorded atac command (C12M02, read from Nibi 2026-09-17), with the Cromwell input
#: paths that no longer exist. Note it carries no `--fraglen` at all.
RECORDED_ATAC = (
    "set -e\npython3 $(which encode_task_macs2_signal_track_atac.py) \\\n"
    "    /crom/inputs/-455205790/ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M.tagAlign.gz \\\n"
    "    --gensz hs \\\n    --chrsz /crom/inputs/-2134521918/GRCh38_EBV.chrom.sizes.tsv \\\n"
    "    --pval-thresh 0.01 \\\n    --smooth-win 150 \\\n    --mem-gb 9.398508861660957"
)

#: C5 built this one, at exactly this name (`$CF/ta/LINES.tsv`, 50000000 lines).
DNASE_TA = "ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M.tagAlign.gz"


@pytest.fixture
def atac_eic(tmp_path):
    """A fake `$EIC` holding the C12M02 base run's recorded atac signal commandLine."""
    d = tmp_path / "eic_atac" / "results" / "C12M02"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(
        {"calls": {"atac.macs2_signal_track": [{"commandLine": RECORDED_ATAC,
                                                "executionStatus": "Done"}]}}))
    return str(tmp_path / "eic_atac")


@pytest.fixture
def atac_ta(tmp_path):
    d = tmp_path / "cf_atac" / "ta"
    d.mkdir(parents=True)
    with gzip.open(d / DNASE_TA, "wt") as f:
        for i in range(60):
            f.write(f"chr21\t{1000 + i}\t{1076 + i}\tN\t1000\t+\n")   # 76 bp reads
    return str(d)


def atac_row(arm, level):
    rows = {(r["arm"], r["level"]): r
            for r in arms.rows("bam", dnase="atac", ratio="no", tracks=["C12M02"])}
    r = rows[(arm, level)]
    return {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in r.items()}


def atac_plan(arm, level, atac_eic, atac_ta, tmp_path, **kw):
    return bam_arm.plan(atac_row(arm, level), str(tmp_path / "cf"), atac_eic,
                        tmp=str(tmp_path / "tmp"), ta_dir=atac_ta, **kw)


def test_dnase_base_runs_the_recorded_atac_command_on_one_tagalign(atac_eic, atac_ta, tmp_path):
    p = atac_plan("base", "base", atac_eic, atac_ta, tmp_path)
    assert p["pipeline"] == "atac" and p["sif"].endswith("atac-seq-pipeline_v2.2.3.sif")
    assert p["control"] is None and p["control_accession"] is None
    assert p["fed"] == [f"{atac_ta}/{DNASE_TA}"]              # one tagAlign, no control
    signal = [c for c in p["commands"] if "--gensz" in c]
    assert len(signal) == 1
    cmd = signal[0]
    assert "encode_task_macs2_signal_track_atac.py" in cmd
    assert "encode_task_macs2_signal_track_chip.py" not in cmd
    assert "--smooth-win 150" in cmd and "--fraglen" not in cmd
    assert f"--chrsz {CHRSZ}" in cmd and "--pval-thresh 0.01" in cmd
    assert cmd.count(".tagAlign.gz") == 1
    # the base arm bins the pipeline's own bigwig and keeps a rebuild beside it
    assert p["pipeline_bigwig"] == arms.TRACKS["C12M02"]["pval_bigwig"]
    assert p["kept_bigwig"].endswith(
        "ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M.pval.signal.bigwig")


@pytest.mark.parametrize("level,smooth_win", [("k0.5", 75), ("k2", 300)])
def test_dnase_extsize_moves_smooth_win_only(atac_eic, atac_ta, tmp_path, level, smooth_win):
    p = atac_plan("extsize", level, atac_eic, atac_ta, tmp_path)
    assert p["knob_value"] == smooth_win and p["fraglen"] == smooth_win
    cmd = next(c for c in p["commands"] if "--gensz" in c)
    assert f"--smooth-win {smooth_win}" in cmd and "--smooth-win 150" not in cmd
    assert "--fraglen" not in cmd
    # everything else the base run recorded is carried through untouched
    assert "--pval-thresh 0.01" in cmd and "--mem-gb 9.398508861660957" in cmd


@pytest.mark.parametrize("level,n", [("15M", 15000000), ("7.5M", 7500000), ("3.75M", 3750000)])
def test_dnase_depth_thins_the_50m_base_with_the_pipeline_subsampler(atac_eic, atac_ta, tmp_path,
                                                                    level, n):
    p = atac_plan("depth", level, atac_eic, atac_ta, tmp_path)
    sub = next(c for c in p["commands"] if "encode_task_subsample_ctl.py" in c)
    assert f"{atac_ta}/{DNASE_TA} --subsample {n}" in sub
    assert "atac-seq-pipeline_v2.2.3.sif" in sub          # the atac image, not the chip one
    assert p["subsample_inputs"] == [f"{atac_ta}/{DNASE_TA}"]   # thinned FROM base, so it nests
    assert p["fed"][0].endswith(f".50M.{bam_arm.human_readable_number(n)}.tagAlign.gz")
    assert p["fed"][0] not in p["inputs"]                 # it does not exist until the job runs


def test_dnase_arms_are_only_base_depth_extsize(atac_eic, atac_ta, tmp_path):
    assert bam_arm.DNASE_BAM_ARMS == ("base", "depth", "extsize")
    assert {r["arm"] for r in arms.rows("bam", dnase="atac", ratio="no", tracks=["C12M02"])} \
        == set(bam_arm.DNASE_BAM_ARMS)
    # an arm that turns a control cannot exist here, whatever a hand-made row says
    bogus = dict(atac_row("base", "base"), arm="ctlid", level="other", pid="C12M02__ctlid__other")
    with pytest.raises(ValueError, match="not a atac bam arm"):
        bam_arm.plan(bogus, str(tmp_path / "cf"), atac_eic, tmp=str(tmp_path), ta_dir=atac_ta)


def test_the_cli_knows_every_smoke_part():
    """A part the CLI does not accept is a job that dies after the venv build (22178020)."""
    import argparse
    import contextlib
    import io
    for part in ("ab", "c", "ok", "dnase-a", "dnase-c", "dnase-ok"):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            try:
                bam_arm.main(["smoke", "--part", part, "--cf", "/nonexistent"])
            except (argparse.ArgumentError, SystemExit) as e:
                assert "invalid choice" not in buf.getvalue(), part
                assert getattr(e, "code", 1) != 2, part      # 2 is argparse refusing the argument
            except Exception:
                pass                                        # a missing $CF is fine here
        assert "invalid choice" not in buf.getvalue(), part


def test_dnase_smoke_rows_cover_every_dnase_arm_kind():
    rows = bam_arm.dnase_smoke_rows()
    assert [(r["arm"], r["level"]) for r in rows] == list(bam_arm.DNASE_SMOKE_LEVELS)
    assert {r["arm"] for r in rows} == set(bam_arm.DNASE_BAM_ARMS)
    assert [r["pid"] for r in rows][0] == "C12M02__base__base"


def test_every_dnase_bam_row_plans(atac_eic, atac_ta, tmp_path):
    made = 0
    for r in arms.rows("bam", dnase="atac", ratio="no", tracks=["C12M02"]):
        tr = {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in r.items()}
        cs = bam_arm.plan_commands(tr, str(tmp_path / "cf"), atac_eic, tmp=str(tmp_path / "tmp"),
                                   ta_dir=atac_ta)
        assert cs[0].startswith("mkdir -p ")
        assert cs[-1].startswith("cp ") and ".pval.signal.bigwig" in cs[-1]
        assert sum("--gensz" in c for c in cs) == 1
        assert not [c for c in cs if "chip-seq-pipeline" in c]
        made += 1
    assert made == 6                       # base 1 + depth 3 + extsize 2


# ---------------------------------------------------------------------------- guards

def test_fastq_rows_are_refused(eic, ta, tmp_path):
    fastq = [r for r in arms.rows("fastq", dnase="none", ratio="yes", tracks=[TRACK])][0]
    fastq = {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in fastq.items()}
    with pytest.raises(ValueError, match="route"):
        bam_arm.plan(fastq, str(tmp_path / "cf"), eic, tmp=str(tmp_path), ta_dir=ta)


def test_an_unknown_pipeline_is_refused(eic, ta, tmp_path):
    bogus = dict(row("base", "base"), pipeline="dnase-seq-pipeline")
    with pytest.raises(ValueError, match="pipeline"):
        bam_arm.plan(bogus, str(tmp_path / "cf"), eic, tmp=str(tmp_path), ta_dir=ta)


def test_every_bam_arm_of_every_chip_track_plans(eic, ta, tmp_path):
    """No arm/track combination raises, and every plan ends by keeping its bigwig."""
    made = 0
    for r in arms.rows("bam", dnase="none", ratio="yes"):
        if r["track"] != TRACK:
            continue                       # only C19M16 has a metadata.json in the fake $EIC
        tr = {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in r.items()}
        cs = bam_arm.plan_commands(tr, str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                                   ta_dir=ta)
        assert cs[0].startswith("mkdir -p ")
        assert cs[-1].startswith("cp ") and ".pval.signal.bigwig" in cs[-1]
        assert sum("--gensz" in c for c in cs) == 1   # exactly one signal run
        made += 1
    assert made == 14      # base 1 + depth 3 + abproxy 2 + ratio 2 + ctlid 2 + ctldepth 2 + extsize 2


def test_rows_tsv_round_trip(tmp_path):
    tsv = tmp_path / "rows.tsv"
    tsv.write_text(arms.to_tsv(arms.rows("bam", dnase="none", ratio="yes", tracks=[TRACK])))
    rows = bam_arm.read_rows(tsv)
    assert len(rows) == 14
    assert bam_arm.row_at(tsv, 0)["pid"] == "C19M16__base__base"
    with pytest.raises(SystemExit, match="no row at index"):
        bam_arm.row_at(tsv, 14)


def test_smoke_rows_cover_every_arm_kind():
    rows = bam_arm.smoke_rows()
    assert [(r["arm"], r["level"]) for r in rows] == list(bam_arm.SMOKE_LEVELS)
    assert {r["arm"] for r in rows} == set(bam_arm.BAM_ARMS)


# ---------------------------------------------------------------------------- publishing

def test_publish_refuses_an_empty_file_and_verifies_the_copy(tmp_path):
    src, dst = tmp_path / "s.npz", tmp_path / "out" / "s.npz"
    src.write_bytes(b"")
    with pytest.raises(SystemExit, match="missing or empty"):
        bam_arm.publish([(src, dst)])
    assert not dst.exists()                      # nothing published when one file is bad
    src.write_bytes(b"payload")
    bam_arm.publish([(src, dst)])
    assert dst.read_bytes() == b"payload"


def test_row_at_round_trips_every_knob_value_through_the_real_tsv(tmp_path):
    """`row_at` must hand `plan` the knob_value `arms.py` wrote, ctlid's JSON string included.

    The rows TSV holds `knob_value` as JSON text, so a ctlid row reads `"ENCFF337JNL"` WITH its
    quotes. A csv reader at the default quoting strips them and `plan`'s `json.loads` then dies —
    which is what held all 12 ctlid rows of the first ChIP array (`$CF/logs/bamarms/HELD_ctlid.txt`).
    Every other arm's value is a bare number, `true` or `null` and survives the stripping, so this
    test drives the real file rather than the hand-built dicts the smoke uses.
    """
    rs = arms.rows("bam", dnase="none", ratio="yes")
    tsv = tmp_path / "rows_chip.tsv"
    tsv.write_text(arms.to_tsv(rs))
    assert len(rs) == 84
    seen = set()
    for i, want in enumerate(rs):
        got = bam_arm.row_at(tsv, i)
        assert got["pid"] == want["pid"]
        value = json.loads(got["knob_value"])
        assert value == want["knob_value"] and type(value) is type(want["knob_value"])
        seen.add(want["arm"])
    assert "ctlid" in seen and seen == set(bam_arm.BAM_ARMS)
    # the exact failure, spelled out: the quotes must still be there for the arm that has them
    ctlid = next(bam_arm.row_at(tsv, i) for i, r in enumerate(rs)
                 if (r["arm"], r["level"]) == ("ctlid", "other"))
    assert ctlid["knob_value"] == '"ENCFF337JNL"'


def test_row_at_reads_a_dnase_rows_file(tmp_path):
    tsv = tmp_path / "rows_dnase.tsv"
    rs = arms.rows("all", dnase="atac", ratio="no", tracks=["C12M02"])
    tsv.write_text(arms.to_tsv(rs))
    bam = [r for r in rs if r["route"] == "bam"]
    assert len(rs) == 10 and len(bam) == 6
    assert json.loads(bam_arm.row_at(tsv, 0)["knob_value"]) is None      # base
    assert json.loads(bam_arm.row_at(tsv, 1)["knob_value"]) == 15000000  # depth 15M


def test_sbatch_script_takes_the_kit_root_as_an_argument(tmp_path):
    """SLURM runs a spool copy, so $KIT can never be derived from $BASH_SOURCE (C9, 2026-09-17)."""
    sh = (Path(__file__).resolve().parents[1] / "slurm" / "t112" / "bam_arm.sh").read_text()
    code = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "BASH_SOURCE" in ln]
    assert 'KIT="${1:?' in sh and 'ROWS="${2:?' in sh
    assert '[ -f "$KIT/tools/t112/bam_arm.py" ]' in sh           # fails loudly, not 10 s later

    import subprocess
    script = Path(__file__).resolve().parents[1] / "slurm" / "t112" / "bam_arm.sh"
    for argv in ([], [str(tmp_path)], [str(tmp_path), "rows.tsv"]):
        p = subprocess.run(["bash", str(script)] + argv, capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], "SLURM_ARRAY_TASK_ID": "0"})
        assert p.returncode != 0
        assert "usage: bam_arm.sh <kit_dir> <rows.tsv>" in p.stderr


def test_sbatch_script_matches_the_python_row_index(tmp_path):
    """The `sed -n "$((i + 2))p"` rule in bam_arm.sh must select the row `--index i` builds."""
    import re
    import subprocess
    sh = (Path(__file__).resolve().parents[1] / "slurm" / "t112" / "bam_arm.sh").read_text()
    assert '#SBATCH --account=def-maxwl' in sh
    assert '#SBATCH --time=8:00:00' in sh
    assert '#SBATCH --cpus-per-task=4' in sh
    assert '#SBATCH --mem=48G' in sh
    code = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "--export" in ln]   # arrays lose a comma-valued --export
    assert re.search(r'sed -n "\$\(\(SLURM_ARRAY_TASK_ID \+ 2\)\)p"', sh)
    assert "pip install --no-index pyBigWig h5py" in sh
    assert "module load apptainer/1.3.5" in sh

    tsv = tmp_path / "rows.tsv"
    tsv.write_text(arms.to_tsv(arms.rows("bam", dnase="none", ratio="yes", tracks=[TRACK])))
    for i in (0, 5, 13):
        line = subprocess.run(["sed", "-n", f"{i + 2}p", str(tsv)], capture_output=True,
                              text=True, check=True).stdout.rstrip("\n")
        assert line.split("\t")[0] == bam_arm.row_at(tsv, i)["pid"]


def test_preflight_inputs_are_only_files_that_must_already_exist(eic, ta, tmp_path):
    """`inputs` is checked before the first command runs, so a tagAlign the arm is about to build
    belongs in `fed`, not in `inputs` (job 22149600_0 died on exactly this)."""
    depth = bam_arm.plan(row("depth", "15M"), str(tmp_path / "cf"), eic,
                         tmp=str(tmp_path / "tmp"), ta_dir=ta)
    assert all(Path(f).is_file() for f in depth["inputs"])
    assert depth["fed"][0] not in depth["inputs"]          # the thinned tagAlign does not exist yet
    assert depth["fed"][0].endswith(".30M.15M.tagAlign.gz")

    ab = bam_arm.plan(row("abproxy", "f0.5"), str(tmp_path / "cf"), eic,
                      tmp=str(tmp_path / "tmp"), ta_dir=ta)
    assert all(Path(f).is_file() for f in ab["inputs"])
    assert "abproxy_f0.5" in ab["fed"][0]

    for arm, level in (("base", "base"), ("ratio", "k0.5"), ("ctlid", "other"),
                       ("ctldepth", "q0.5"), ("extsize", "k2")):
        p = bam_arm.plan(row(arm, level), str(tmp_path / "cf"), eic, tmp=str(tmp_path / "tmp"),
                         ta_dir=ta)
        missing = [f for f in p["inputs"] if not Path(f).is_file()]
        assert missing == [f for f in p["inputs"] if f.endswith(".bigwig")]   # only the $EIC one
        assert set(p["fed"]) <= set(p["inputs"])
