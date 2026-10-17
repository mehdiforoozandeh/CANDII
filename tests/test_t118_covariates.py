"""`tools/t118/build_covariates.py` — the tracked per-product covariate table.

The expected values below are typed in from the t118 plan and the t112 arm table, not derived from
the builder: a wrong base value must fail here. The t112 constants the table must agree with are
read from `tools/t112/arms.py` itself.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t118" / "build_covariates.py"
TABLE = REPO / "tools" / "t118" / "covariates.tsv"
FIX = REPO / "tests" / "fixtures" / "t112_meta"
MANIFEST = FIX / "MANIFEST.tsv"
COVS = FIX / "covariates_by_pid.json"
MANIFEST_MD5 = "599e2ca607961fe550b477558f894edf"

COLUMNS = [
    "pid", "track", "assay", "arm", "level", "usable",
    "depth_reads", "depth_log2", "read_length", "run_type_pe", "dedup_on", "mapq_thresh",
    "has_control", "control_fraction", "ratio_k", "ctl_identity", "control_reads",
    "control_depth_log2", "extsize_k", "fraglen_bp",
    "src_depth", "src_read_length", "src_run_type", "src_dedup", "src_mapq",
    "src_control_fraction", "src_ratio_k", "src_ctl_identity", "src_control_depth",
    "src_extsize_k", "src_fraglen",
]
SRC_PREFIXES = ("manifest:", "covariates.json:", "tools/t112/arms.py:",
                "tools/t112/bam_arm.py:ratio_for", "chip.wdl v2.2.2 default L",
                "atac.wdl v2.2.3 default L", "none: no control")
HISTONE = ("C07M20", "C07M29", "C19M16", "C19M22", "C40M17", "C40M18")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def arms():
    return _load("t112_arms", REPO / "tools" / "t112" / "arms.py")


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST, newline="") as fh:
        return {r["pid"]: r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)}


@pytest.fixture(scope="module")
def covs():
    return json.loads(COVS.read_text())


@pytest.fixture(scope="module")
def table():
    text = TABLE.read_text()
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    return header, {r[0]: dict(zip(header, r)) for r in (ln.split("\t") for ln in lines[1:])}


def _cli(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------- fixtures


def test_manifest_fixture_is_the_verbatim_nibi_manifest():
    assert hashlib.md5(MANIFEST.read_bytes()).hexdigest() == MANIFEST_MD5


def test_covariates_fixture_has_every_manifest_product(manifest, covs):
    assert len(covs) == 130 and len(manifest) == 130
    assert set(covs) == set(manifest)
    assert all(covs[p]["pid"] == p for p in covs)


def test_exactly_52_products_have_their_bases_counts(manifest):
    base = {r["track"]: r["counts25_md5"] for r in manifest.values() if r["arm"] == "base"}
    same = [r for r in manifest.values()
            if r["arm"] != "base" and r["counts25_md5"] == base[r["track"]]]
    assert len(same) == 52
    by = Counter((r["track"] == "C12M02", r["arm"]) for r in same)
    assert by == {(False, "ratio"): 12, (False, "ctlid"): 12, (False, "ctldepth"): 12,
                  (False, "extsize"): 12, (True, "extsize"): 2, (True, "mapq"): 2}
    p_only = [r for r in same if r["arm"] in ("ratio", "ctlid", "ctldepth", "extsize")]
    assert len(p_only) == 50


# ---------------------------------------------------------------------------- rebuild


def test_cli_rebuilds_the_tracked_table_bit_for_bit(tmp_path):
    out = tmp_path / "cov.tsv"
    r = _cli(MANIFEST, COVS, out)
    assert r.returncode == 0, r.stderr
    assert out.read_bytes() == TABLE.read_bytes()


def test_products_dir_form_gives_the_same_bytes(tmp_path, covs):
    d = tmp_path / "products"
    for pid, cov in covs.items():
        (d / pid).mkdir(parents=True)
        (d / pid / "covariates.json").write_text(json.dumps(cov, indent=2))
    out = tmp_path / "cov.tsv"
    r = _cli(MANIFEST, d, out)
    assert r.returncode == 0, r.stderr
    assert out.read_bytes() == TABLE.read_bytes()


def test_table_shape_order_and_encoding(table):
    raw = TABLE.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("ascii").rstrip("\n").split("\n")
    assert len(lines) == 131
    header, rows = table
    assert header == COLUMNS
    assert all(len(ln.split("\t")) == len(COLUMNS) for ln in lines)
    assert list(rows) == sorted(rows)


def test_builder_refuses_a_changed_identical_counts_set(tmp_path, manifest):
    text = MANIFEST.read_text()
    old = manifest["C19M16__ratio__k2"]["counts25_md5"]
    assert text.count(old) > 1
    bad = tmp_path / "MANIFEST.tsv"
    line = next(ln for ln in text.split("\n") if ln.startswith("C19M16__ratio__k2\t"))
    bad.write_text(text.replace(line, line.replace(old, "0" * 32)))
    r = _cli(bad, COVS, tmp_path / "cov.tsv")
    assert r.returncode != 0 and "C19M16__ratio__k2" in r.stderr


# ---------------------------------------------------------------------------- values


def test_usable_is_zero_only_on_the_two_dnase_mapq_arms(table):
    _, rows = table
    assert {p for p, r in rows.items() if r["usable"] == "0"} == {"C12M02__mapq__0",
                                                                   "C12M02__mapq__10"}
    assert all(r["usable"] in ("0", "1") for r in rows.values())


def test_manifest_columns_pass_through(table, manifest):
    _, rows = table
    for pid, r in rows.items():
        m = manifest[pid]
        assert r["depth_reads"] == m["depth"] and r["read_length"] == m["read_length"]
        assert r["fraglen_bp"] == m["fraglen"]
        assert r["run_type_pe"] == ("1" if m["run_type"] == "paired-ended" else "0")
        assert float(r["depth_log2"]) == pytest.approx(math.log2(int(m["depth"])))
    assert {p for p, r in rows.items() if r["run_type_pe"] == "1"} == {
        f"{t}__pe__pe" for t in HISTONE + ("C12M02",)}
    assert Counter(r["read_length"] for r in rows.values()) == {"101": 108, "76": 10,
                                                                 "36": 6, "50": 6}


def test_base_rows_carry_the_pipeline_defaults(table, arms):
    _, rows = table
    for t in HISTONE:
        r = rows[f"{t}__base__base"]
        acc = arms.TRACKS[t]["ctl_acc"]
        assert (r["dedup_on"], r["mapq_thresh"], r["has_control"], r["control_fraction"],
                r["ratio_k"], r["ctl_identity"], r["extsize_k"]) == (
                    "1", "30", "1", "0.0", "1.0", "matched", "1.0")
        assert int(r["control_reads"]) == arms.CONTROLS[acc]["reads"]
        assert r["src_mapq"] == "chip.wdl v2.2.2 default L179 mapq_thresh = 30"
        assert r["src_dedup"].startswith("chip.wdl v2.2.2 default L178 no_dup_removal = false")
    d = rows["C12M02__base__base"]
    assert (d["dedup_on"], d["mapq_thresh"], d["has_control"], d["control_fraction"],
            d["ratio_k"], d["ctl_identity"], d["control_reads"], d["control_depth_log2"],
            d["extsize_k"], d["depth_reads"], d["read_length"], d["fraglen_bp"]) == (
                "1", "30", "0", "0.0", "0.0", "none", "0", "0.0", "1.0", "50000000", "76", "150")
    assert d["src_mapq"].startswith("atac.wdl v2.2.3 default L184 mapq_thresh = 30")
    assert "multimapping = 4" in d["src_mapq"]
    assert d["src_dedup"].startswith("atac.wdl v2.2.3 default L183 no_dup_removal = false")


def test_each_arm_moves_only_its_own_knob(table):
    _, rows = table
    knobs = ("dedup_on", "mapq_thresh", "control_fraction", "ratio_k", "ctl_identity",
             "extsize_k", "has_control")
    moved = {"dedup": {"dedup_on"}, "mapq": {"mapq_thresh"}, "abproxy": {"control_fraction"},
             "ratio": {"ratio_k"}, "extsize": {"extsize_k"},
             "ctlid": {"ctl_identity", "has_control", "ratio_k"},
             "depth": set(), "pe": set(), "crop": set(), "ctldepth": set()}
    for pid, r in rows.items():
        if r["arm"] == "base":
            continue
        b = rows[f"{r['track']}__base__base"]
        diff = {k for k in knobs if r[k] != b[k]}
        allowed = moved[r["arm"]]
        if r["arm"] == "ctlid" and r["level"] == "other":
            allowed = {"ctl_identity"}
        assert diff == allowed, (pid, diff)


def test_arm_values(table):
    _, rows = table
    for t in HISTONE:
        assert rows[f"{t}__dedup__off"]["dedup_on"] == "0"
        assert rows[f"{t}__mapq__0"]["mapq_thresh"] == "0"
        assert rows[f"{t}__mapq__10"]["mapq_thresh"] == "10"
        assert rows[f"{t}__abproxy__f0.5"]["control_fraction"] == "0.5"
        assert rows[f"{t}__abproxy__f0.9"]["control_fraction"] == "0.9"
        assert rows[f"{t}__ratio__k0.5"]["ratio_k"] == "0.5"
        assert rows[f"{t}__ratio__k2"]["ratio_k"] == "2.0"
        assert rows[f"{t}__extsize__k0.5"]["extsize_k"] == "0.5"
        assert rows[f"{t}__extsize__k2"]["extsize_k"] == "2.0"
        assert rows[f"{t}__ctlid__other"]["ctl_identity"] == "other"
        n = rows[f"{t}__ctlid__none"]
        assert (n["has_control"], n["ctl_identity"], n["control_fraction"], n["ratio_k"],
                n["control_reads"], n["control_depth_log2"]) == ("0", "none", "0.0", "0.0", "0",
                                                                 "0.0")
        assert n["src_ctl_identity"] == n["src_control_depth"] == "none: no control"
    assert rows["C12M02__dedup__off"]["dedup_on"] == "0"
    assert rows["C12M02__mapq__0"]["mapq_thresh"] == "0"
    assert rows["C12M02__mapq__10"]["mapq_thresh"] == "10"
    assert rows["C12M02__extsize__k2"]["extsize_k"] == "2.0"
    assert Counter(r["ctl_identity"] for r in rows.values()) == {"matched": 108, "other": 6,
                                                                  "none": 16}


def test_control_reads_agree_with_t112(table, manifest, arms):
    _, rows = table
    for pid, r in rows.items():
        m = manifest[pid]
        if r["has_control"] == "0":
            assert r["control_reads"] == "0" and r["control_depth_log2"] == "0.0"
            continue
        reads = int(r["control_reads"])
        assert float(r["control_depth_log2"]) == pytest.approx(math.log2(reads))
        full = arms.CONTROLS[m["control_accession"]]["reads"]
        if r["arm"] == "ctldepth":
            assert reads == int(m["knob_value"]) < full
        elif m["route"] == "bam":
            assert reads == full, pid
        else:
            # FASTQ-route arms re-processed the control, so its read count moved
            assert reads != full, pid


def test_ratio_and_extsize_knobs_match_the_t112_rule(table, manifest, arms):
    _, rows = table
    for pid, r in rows.items():
        m = manifest[pid]
        if r["arm"] == "ratio":
            acc = arms.TRACKS[r["track"]]["ctl_acc"]
            k = float(r["ratio_k"])
            assert float(m["knob_value"]) == arms.ratio_value(k, arms.CONTROLS[acc]["reads"])
        if r["arm"] == "extsize":
            k = float(r["extsize_k"])
            want = arms.round_half_up(k * arms.TRACKS[r["track"]]["fraglen"])
            assert int(m["knob_value"]) == want == int(r["fraglen_bp"])


def test_every_value_names_its_source(table):
    _, rows = table
    for pid, r in rows.items():
        for c in COLUMNS:
            if c.startswith("src_"):
                assert r[c].startswith(SRC_PREFIXES), (pid, c, r[c])
        if r["has_control"] == "0":
            assert r["src_ratio_k"] == r["src_control_fraction"] == "none: no control"
