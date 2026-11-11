"""D7 (t168): input JSON per assay, arm and unit; arm covariates; harvest and FASTQ cleanup."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

for _p in (Path(__file__).resolve().parents[1] / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import fastq_uniform as F  # noqa: E402

ROOT = "/fq"
GTSV = "/refs/hg38.local.tsv"


def _f(acc, bio, tech, rl, pe="", mate=""):
    return {"accession": acc, "md5": "0" * 32, "bytes": 10, "run_type": "x", "read_length": rl,
            "paired_end": pe, "paired_with": mate, "bio_rep": bio, "tech_rep": tech,
            "platform": "HiSeq", "flowcell": "", "lane": "", "status": "released", "url": ""}


#: single-end histone experiment: bio rep 1 has two libraries (tr1 with 2 runs, tr2 with 1 run)
SE_HISTONE = {"experiment": "ENCSR000AAA", "note": "", "files": [
    _f("ENCFF003", 1, 1, 76), _f("ENCFF001", 1, 1, 76), _f("ENCFF002", 1, 2, 76),
    _f("ENCFF010", 2, 1, 76)]}
#: paired-end ATAC experiment, one replicate, two runs
PE_ATAC = {"experiment": "ENCSR000BBB", "note": "", "files": [
    _f("ENCFF101", 1, 1, 100, "1", "ENCFF102"), _f("ENCFF102", 1, 1, 100, "2", "ENCFF101"),
    _f("ENCFF103", 1, 1, 100, "1", "ENCFF104"), _f("ENCFF104", 1, 1, 100, "2", "ENCFF103")]}
SHORT = {"experiment": "ENCSR000CCC", "note": "", "files": [_f("ENCFF201", 1, 1, 36)]}


def row(exp, assay, cls, unit="pooled", arm="base", **kw):
    return {"row": "1", "experiment": exp, "assay": assay, "assay_class": cls, "unit": unit,
            "arm": arm, **kw}


def js(r, fj, **kw):
    return F.build_input_json(r, fj, GTSV, ROOT, **kw)


def p(exp, acc):
    return f"{ROOT}/{exp}/{acc}.fastq.gz"


def test_histone_pooled_align_only_one_rep_per_bio_rep():
    d = js(row("ENCSR000AAA", "H3K27ac", "narrow_histone"), SE_HISTONE)
    assert d["chip.pipeline_type"] == "histone"
    assert d["chip.align_only"] is True
    assert d["chip.genome_tsv"] == GTSV
    assert d["chip.singularity"].endswith("sif/chip-seq-pipeline_v2.2.2.sif")
    assert d["chip.paired_end"] is False
    e = "ENCSR000AAA"
    assert d["chip.fastqs_rep1_R1"] == [p(e, "ENCFF001"), p(e, "ENCFF002"), p(e, "ENCFF003")]
    assert d["chip.fastqs_rep2_R1"] == [p(e, "ENCFF010")]
    assert not any(k.endswith("_R2") for k in d)
    # default settings: no subsampling, no control, no crop
    assert not any("subsample" in k or "ctl" in k or "crop" in k for k in d)


def test_units_bio_rep_library_and_lane():
    e = "ENCSR000AAA"
    d = js(row(e, "H3K27ac", "narrow_histone", unit="br2"), SE_HISTONE)
    assert d["chip.fastqs_rep1_R1"] == [p(e, "ENCFF010")] and "chip.fastqs_rep2_R1" not in d
    d = js(row(e, "H3K27ac", "narrow_histone", unit="br1tr1"), SE_HISTONE)
    assert d["chip.fastqs_rep1_R1"] == [p(e, "ENCFF001"), p(e, "ENCFF003")]
    d = js(row(e, "H3K27ac", "narrow_histone", unit="br1tr1ln2"), SE_HISTONE)
    assert d["chip.fastqs_rep1_R1"] == [p(e, "ENCFF003")]
    with pytest.raises(SystemExit):
        js(row(e, "H3K27ac", "narrow_histone", unit="br1tr1ln3"), SE_HISTONE)
    with pytest.raises(SystemExit):
        js(row(e, "H3K27ac", "narrow_histone", unit="rep1"), SE_HISTONE)


def test_fastq_files_column_restricts_the_set():
    e = "ENCSR000BBB"
    d = js(row(e, "ATAC-seq", "accessibility_atac", fastq_files="ENCFF104"), PE_ATAC)
    assert d["atac.fastqs_rep1_R1"] == [p(e, "ENCFF103")]
    assert d["atac.fastqs_rep1_R2"] == [p(e, "ENCFF104")]


def test_atac_base_paired_and_adapter_detection():
    e = "ENCSR000BBB"
    d = js(row(e, "ATAC-seq", "accessibility_atac"), PE_ATAC)
    assert d["atac.pipeline_type"] == "atac" and d["atac.align_only"] is True
    assert d["atac.paired_end"] is True and d["atac.auto_detect_adapter"] is True
    assert d["atac.fastqs_rep1_R1"] == [p(e, "ENCFF101"), p(e, "ENCFF103")]
    assert d["atac.fastqs_rep1_R2"] == [p(e, "ENCFF102"), p(e, "ENCFF104")]  # mates in R1 order
    assert not any(k.startswith("chip.") for k in d)
    assert d["atac.singularity"].endswith("sif/atac-seq-pipeline_v2.2.3.sif")


def test_se_arm_drops_mate_two():
    e = "ENCSR000BBB"
    d = js(row(e, "ATAC-seq", "accessibility_atac", arm="se"), PE_ATAC)
    assert d["atac.paired_end"] is False
    assert d["atac.fastqs_rep1_R1"] == [p(e, "ENCFF101"), p(e, "ENCFF103")]
    assert not any(k.endswith("_R2") for k in d)
    with pytest.raises(SystemExit):  # se on a single-end experiment
        js(row("ENCSR000AAA", "H3K27ac", "narrow_histone", arm="se"), SE_HISTONE)


def test_crop_arms_chip_switch_and_applicability():
    d = js(row("ENCSR000AAA", "H3K4me3", "narrow_histone", arm="crop36"), SE_HISTONE)
    assert d["chip.crop_length"] == 36 and d["chip.crop_length_tol"] == 2
    d = js(row("ENCSR000AAA", "H3K4me3", "narrow_histone", arm="crop50"), SE_HISTONE)
    assert d["chip.crop_length"] == 50
    with pytest.raises(SystemExit):  # 36 bp reads cannot be cropped to 36 or 50
        js(row("ENCSR000CCC", "H3K4me3", "narrow_histone", arm="crop36"), SHORT)
    with pytest.raises(SystemExit):
        js(row("ENCSR000AAA", "H3K4me3", "narrow_histone", arm="depth0.50"), SE_HISTONE)


def test_atac_crop_uses_precropped_fastqs_not_a_wdl_input():
    e = "ENCSR000BBB"
    cropped = {a: f"/tmp/crop/{a}.crop.fastq.gz" for a in ("ENCFF101", "ENCFF102", "ENCFF103",
                                                            "ENCFF104")}
    d = js(row(e, "ATAC-seq", "accessibility_atac", arm="crop50"), PE_ATAC, cropped=cropped)
    assert not any("crop_length" in k for k in d)  # atac.wdl v2.2.3 has no crop input
    assert d["atac.fastqs_rep1_R1"] == [cropped["ENCFF101"], cropped["ENCFF103"]]
    assert d["atac.fastqs_rep1_R2"] == [cropped["ENCFF102"], cropped["ENCFF104"]]


def test_control_and_refused_assays():
    d = js(row("ENCSR000AAA", "chipseq-control", "control"), SE_HISTONE)
    assert d["chip.pipeline_type"] == "control" and d["chip.align_only"] is True
    with pytest.raises(SystemExit):
        js(row("ENCSR000AAA", "DNase-seq", "accessibility_dnase"), SE_HISTONE)
    with pytest.raises(SystemExit):  # build list names the wrong pipeline
        js(row("ENCSR000AAA", "H3K27ac", "narrow_histone", pipeline="atac-seq-pipeline"),
           SE_HISTONE)


@pytest.mark.parametrize("arm,knob,value,run_type,rl", [
    ("base", "", "", "paired-ended", 100), ("se", "se", "1", "single-ended", 100),
    ("crop36", "crop", "36", "paired-ended", 36), ("crop50", "crop", "50", "paired-ended", 50)])
def test_arm_covariates(arm, knob, value, run_type, rl):
    r = row("ENCSR000BBB", "ATAC-seq", "accessibility_atac", arm=arm)
    reps = F.unit_reps(PE_ATAC, "pooled")
    a = F.arm_settings(PE_ATAC, reps, arm)
    rec = F.arm_record(r, PE_ATAC, reps, a, job_id="123")
    assert (rec["arm_knob"], rec["arm_value"]) == (knob, value)
    assert rec["run_type"] == run_type and rec["read_length"] == rl
    assert rec["layer"] == "uniform" and rec["dedup"] == "on" and rec["mapq_threshold"] == 30
    assert rec["pipeline"] == "atac-seq-pipeline" and rec["pipeline_version"] == "v2.2.3"
    assert rec["image_digest"] == "sha256:" + F.PIPELINES["atac"]["sha256"]
    assert rec["reps"][0]["bio_rep"] == 1


def test_arm_dir_naming():
    assert F.arm_dir("/w", row("E", "H3K27ac", "narrow_histone")) == Path("/w/E/base")
    assert F.arm_dir("/w", row("E", "H3K27ac", "narrow_histone", unit="br1tr2", arm="se")) == \
        Path("/w/E/br1tr2.se")


def test_genome_tsv_points_at_refs_copy():
    t = F.genome_tsv_text("/refs")
    kv = dict(ln.split("\t") for ln in t.strip().split("\n"))
    assert kv["genome_name"] == "hg38" and kv["chrsz"].startswith("/refs/")
    assert kv["bwa_idx_tar"] == "/refs/8c692fba4640609720272154ab0faa30/ENCFF643CGH.tar.gz"
    assert "googleapis" not in t


def _meta(tmp_path, key, n):
    calls = {}
    for call, outs in (("align", {"bam": "x.bam"}), ("filter", {"nodup_bam": "x.nodup.bam",
                                                                "nodup_bai": "x.nodup.bam.bai"}),
                       ("bam2ta", {"ta": "x.tagAlign.gz"})):
        calls[f"{key}.{call}"] = []
        for s in range(n):
            o = {}
            for k, name in outs.items():
                f = tmp_path / f"s{s}.{call}.{name}"
                f.write_bytes(b"chr1\t0\t10\tN\t1000\t+\n" * (s + 2) if "tagAlign" in name else b"b")
                o[k] = str(f)
            calls[f"{key}.{call}"] += [{"shardIndex": s, "attempt": 1, "executionStatus": "Failed",
                                        "outputs": {}},
                                       {"shardIndex": s, "attempt": 2, "executionStatus": "Done",
                                        "outputs": o}]
    return {"status": "Succeeded", "calls": calls}


def test_harvest_plan_takes_last_done_attempt_per_rep(tmp_path):
    plan = dict(F.harvest_plan(_meta(tmp_path, "atac", 2), "atac", 2))
    assert set(plan) == {f"rep{i}.{s}" for i in (1, 2) for s in
                         ("unfiltered.bam", "filtered.bam", "filtered.bam.bai", "tagAlign.gz")}
    assert plan["rep2.tagAlign.gz"].endswith("s1.bam2ta.x.tagAlign.gz")
    with pytest.raises(SystemExit):
        F.harvest_plan(_meta(tmp_path, "atac", 1), "atac", 2)


def test_verify_and_fastq_cleanup(tmp_path):
    work, fq = tmp_path / "work", tmp_path / "fq"
    r = row("ENCSR000BBB", "ATAC-seq", "accessibility_atac")
    d = F.arm_dir(work, r)
    d.mkdir(parents=True)
    reps = F.unit_reps(PE_ATAC, "pooled")
    (fq / "ENCSR000BBB").mkdir(parents=True)
    (fq / "ENCSR000BBB" / "ENCFF101.fastq.gz").write_bytes(b"x")
    (fq / "ENCSR000BBB" / "ENCFF101.fastq.gz").chmod(0o444)
    assert not F.cleanup_fastq([r], "ENCSR000BBB", work, fq)  # no ALIGN_DONE yet
    lines = ["name\tsrc\tbytes\tmd5"]
    for stem in ("unfiltered.bam", "filtered.bam", "filtered.bam.bai", "tagAlign.gz"):
        (d / f"rep1.{stem}").write_bytes(b"abc")
        lines.append(f"rep1.{stem}\tsrc\t3\t{F.md5_file(d / f'rep1.{stem}')}")
    (d / "HARVEST.tsv").write_text("\n".join(lines) + "\n")
    (d / "arm.json").write_text(json.dumps({"reps": reps}))
    (d / "ALIGN_DONE").write_text("{}")
    assert F.verify_arm(d, md5=True) == []
    (d / "rep1.tagAlign.gz").write_bytes(b"abcd")
    assert F.verify_arm(d) != []
    (d / "rep1.tagAlign.gz").write_bytes(b"abc")
    assert F.cleanup_fastq([r], "ENCSR000BBB", work, fq)
    assert not (fq / "ENCSR000BBB").exists()


def test_count_lines_gz(tmp_path):
    import gzip
    f = tmp_path / "a.tagAlign.gz"
    with gzip.open(f, "wb") as fh:
        fh.write(b"l\n" * 1234)
    assert F.count_lines_gz(f) == 1234


def test_build_list_reader(tmp_path):
    t = tmp_path / "bl.tsv"
    t.write_text("row\texperiment\tassay\tassay_class\tunit\tarm\tpipeline\n"
                 "1\tE1\tH3K27ac\tnarrow_histone\tpooled\tbase\tchip-seq-pipeline2\n"
                 "2\tE2\tATAC-seq\taccessibility_atac\tpooled\tbase\tatac-seq-pipeline\n")
    rows = F.read_build_list(t)
    assert F.get_row(rows, 2)["experiment"] == "E2"
    assert F.pipeline_key(F.get_row(rows, 1)) == ("chip", "histone")
    bad = tmp_path / "bad.tsv"
    bad.write_text("row\texperiment\n1\tE1\n")
    with pytest.raises(SystemExit):
        F.read_build_list(bad)


def test_byte_ranges_cover_the_file_once():
    for size, n in ((10, 3), (1000, 32), (5, 32), (1 << 30, 32)):
        rs = F.byte_ranges(size, n)
        assert rs[0][0] == 0 and rs[-1][1] == size - 1
        assert all(b[0] == a[1] + 1 for a, b in zip(rs, rs[1:]))
        assert len(rs) <= n


# --- KF (t180): mixed run types, not-applicable arms, staging, dropped BAMs ---------------------

#: ENCSR082NQB-like: bio rep 1 one PE pair; bio rep 2 one PE pair plus one SE run (mixed)
MIXED = {"experiment": "ENCSR000MIX", "note": "", "files": [
    _f("ENCFF301", 1, 1, 76, "1", "ENCFF302"), _f("ENCFF302", 1, 1, 76, "2", "ENCFF301"),
    _f("ENCFF401", 2, 1, 76, "1", "ENCFF402"), _f("ENCFF402", 2, 1, 76, "2", "ENCFF401"),
    _f("ENCFF403", 2, 1, 76)]}


def test_mixed_replicate_runs_as_single_end_mate1_like_encode4():
    """ENCODE4's chip-seq-alignment-step v2 aligned these experiments single-end from mate 1 of
    each pair plus the SE runs, for every replicate (portal derived_from, 2026-11-11)."""
    e = "ENCSR000MIX"
    d = js(row(e, "H3K79me2", "broad_histone"), MIXED)
    assert d["chip.paired_end"] is False
    assert d["chip.fastqs_rep1_R1"] == [p(e, "ENCFF301")]                  # PE-only rep too
    assert d["chip.fastqs_rep2_R1"] == [p(e, "ENCFF401"), p(e, "ENCFF403")]
    assert not any(k.endswith("_R2") for k in d)
    reps = F.unit_reps(MIXED, "pooled")
    a = F.arm_settings(MIXED, reps, "base")
    assert a["run_type"] == "single-ended" and "mate 1" in a["mixed_run_types"]
    assert reps[1]["mixed_run_types"] and reps[1]["mates_dropped"] == ["ENCFF402"]
    # a lane of the mixed library is one run type, so it stays what it is
    lane = js(row(e, "H3K79me2", "broad_histone", unit="br2tr1ln1"), MIXED)
    assert lane["chip.paired_end"] is True and lane["chip.fastqs_rep1_R2"] == [p(e, "ENCFF402")]


def test_se_arm_of_a_mixed_experiment_is_not_applicable(tmp_path):
    reps = F.unit_reps(MIXED, "pooled")
    with pytest.raises(F.NotApplicable, match="repeat the base"):
        F.arm_settings(MIXED, reps, "se")
    r = row("ENCSR000MIX", "H3K79me2", "broad_histone", arm="se")
    d = F.arm_dir(tmp_path, r)
    d.mkdir(parents=True)
    F.write_not_applicable(d, r, "x")
    assert F.verify_arm(d) == []                     # counts as verified: FASTQs can be cleaned
    assert json.loads((d / F.NOT_APPLICABLE).read_text())["arm"] == "se"


def test_run_dir_names_exclude_readset_dirs():
    for ok in ("base", "se", "crop36", "br1tr2ln3.base", "br2.crop50"):
        assert F.RUN_DIR_RE.match(ok)
    for no in ("pooled.uniform", "br1.native", "BUILD_COST.uniform.3.json", "base.x"):
        assert not F.RUN_DIR_RE.match(no)


def test_dropped_bams_still_verify(tmp_path):
    d = tmp_path / "base"
    d.mkdir()
    lines = ["name\tsrc\tbytes\tmd5"]
    for stem in ("unfiltered.bam", "filtered.bam", "filtered.bam.bai", "tagAlign.gz"):
        (d / f"rep1.{stem}").write_bytes(b"abc")
        lines.append(f"rep1.{stem}\tsrc\t3\t{F.md5_file(d / f'rep1.{stem}')}")
    (d / "HARVEST.tsv").write_text("\n".join(lines) + "\n")
    (d / "arm.json").write_text(json.dumps({"reps": [{"rep": 1, "bio_rep": 1}]}))
    (d / "ALIGN_DONE").write_text("{}")
    (d / "rep1.unfiltered.bam").unlink()
    assert F.verify_arm(d) != []
    (d / "DROPPED.tsv").write_text("name\tbytes\tmd5\treason\tdropped_utc\n"
                                   "rep1.unfiltered.bam\t3\tx\tkeep_bam=0\tt\n")
    assert F.verify_arm(d) == []


@pytest.fixture
def range_server(tmp_path):
    """A local HTTP server that honours Range, logs every request and fails the first request for
    one path (to exercise retry)."""
    import http.server
    import threading
    root = tmp_path / "srv"
    root.mkdir()
    log = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = root / self.path.lstrip("/")
            log.append((self.path, self.headers.get("Range"), self.headers.get("User-Agent")))
            if self.path.endswith("flaky.bin") and sum(1 for x in log if x[0] == self.path) == 1:
                self.send_response(503)
                self.end_headers()
                return
            data = path.read_bytes()
            rg = self.headers.get("Range")
            if rg:
                lo, hi = (int(x) for x in rg.split("=")[1].split("-"))
                body = data[lo:hi + 1]
                self.send_response(206)
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}", log
    srv.shutdown()


def test_fetch_many_ranges_md5_retry_and_user_agent(range_server, tmp_path, monkeypatch):
    import hashlib
    import os as _os
    root, url, log = range_server
    monkeypatch.setattr(F, "backoff_s", lambda k, base=5.0, cap=300.0: 0.01)
    items = []
    for i, name in enumerate(("a.bin", "b.bin", "flaky.bin")):
        data = _os.urandom(5000 + 997 * i)
        (root / name).write_bytes(data)
        items.append({"url": f"{url}/{name}", "dest": tmp_path / "out" / name,
                      "md5": hashlib.md5(data).hexdigest(), "bytes": len(data)})
    st = F.fetch_many(items, ranges=7, parallel_files=2)
    for it in items:
        assert Path(it["dest"]).read_bytes() == (root / Path(it["dest"]).name).read_bytes()
        assert (Path(it["dest"]).stat().st_mode & 0o777) == 0o444
    assert st["files"] == 3 and st["ranges"] == 7 and st["parallel_files"] == 2
    per_file = {}
    for path, rg, ua in log:
        per_file.setdefault(path, set()).add(rg)
        assert ua == F.USER_AGENT
    assert all(len(v) == 7 for v in per_file.values())        # 7 distinct ranges per file
    # already staged and md5-good: no request
    n = len(log)
    assert F.fetch_many(items, ranges=7, parallel_files=2)["files"] == 0 and len(log) == n
    # a wrong md5 fails after the retries, and leaves no file behind
    bad = dict(items[0], dest=tmp_path / "out" / "bad.bin", md5="0" * 32)
    monkeypatch.setattr(F, "time", type("T", (), {"time": staticmethod(__import__("time").time),
                                                  "sleep": staticmethod(lambda s: None),
                                                  "strftime": staticmethod(__import__("time").strftime)}))
    with pytest.raises(SystemExit, match="md5"):
        F.fetch_many([bad], ranges=3, parallel_files=1)
    assert not (tmp_path / "out" / "bad.bin").exists()


def test_stage_row_routes_native_rows_to_their_bams(range_server, tmp_path, monkeypatch):
    import hashlib
    root, url, _ = range_server
    data = b"BAM" * 1000
    (root / "ENCFF989MMP.bam").write_bytes(data)
    monkeypatch.setattr(F, "portal_file", lambda acc: {
        "accession": acc, "md5": hashlib.md5(data).hexdigest(), "bytes": len(data), "bio_rep": 1,
        "output_type": "redacted unfiltered alignments", "status": "released",
        "assembly": "GRCh38", "file_format": "bam", "mapped_read_length": 36,
        "mapped_run_type": "single-ended", "analysis_step_version": "/x/bwa-alignment-step-v-1-0/",
        "url": f"{url}/{acc}.bam"})
    r = {"row": "1", "experiment": "ENCSR749PIF", "assay": "chipseq-control",
         "assay_class": "control", "unit": "pooled", "arm": "base",
         "fastq_files": "ENCFF989MMP", "pipeline": "native-redacted-bam"}
    assert F.row_kind(r) == "native"
    F.stage_row(r, tmp_path / "fq", ranges=4, parallel_files=1)
    assert (tmp_path / "fq/ENCSR749PIF/ENCFF989MMP.bam").read_bytes() == data
    bj = json.loads((tmp_path / "fq/ENCSR749PIF/bams.json").read_text())
    assert bj["bams"][0]["bio_rep"] == 1
    assert F.row_kind({"assay_class": "accessibility_dnase",
                       "pipeline": "dnase-seq-pipeline v3.0.0"}) == "dnase"


def test_caper_conf_names_local_jars_so_concurrent_jobs_take_no_lock():
    t = F.caper_conf_text()
    assert f"cromwell={F.CROMWELL_JAR}" in t and f"womtool={F.WOMTOOL_JAR}" in t
    assert "http" not in t and t.startswith("backend=local")
