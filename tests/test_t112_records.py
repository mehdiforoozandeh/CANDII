"""`tools/t112/records.py` — covariates/provenance records and MANIFEST.tsv for the t112 products.

Later chunks call `records.py validate --rows R --expect N` on Nibi and trust its last line, so the
tests are mostly about what it must REFUSE: a missing key, a control file that disagrees with the
covariates, a record that disagrees with the rows TSV, a count that is not the expected one.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t112" / "records.py"
ARMS = REPO / "tools" / "t112" / "arms.py"


def _load():
    spec = importlib.util.spec_from_file_location("t112_records", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t112_records"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rec():
    return _load()


def _rows_header():
    spec = importlib.util.spec_from_file_location("t112_arms_for_records", ARMS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HEADER


def _cov(track="C19M16", cell="C19", arm="base", level="base", knob="none", knob_value=None,
         control="matched", **over):
    d = dict(pid=f"{track}__{arm}__{level}", biosample=f"CF_{cell}__{arm}__{level}", track=track,
             cell=cell, assay="H3K27ac", arm=arm, level=level, knob=knob, knob_value=knob_value,
             depth=30000000, read_length=101, run_type="single-ended", fraglen=180,
             control=None if control is None else
             {"accession": "ENCFF433TZR", "source": control, "reads": 58073570})
    d.update(over)
    return d


def _prov(pid="C19M16__base__base", route="bam", **over):
    d = dict(pid=pid, route=route,
             pipeline={"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.2.2",
                       "sif": "/project/x/chip-seq-pipeline_v2.2.2.sif",
                       "sif_md5": "f6e408f7e77bafde4556883f33552191", "sif_sha256": None},
             commands=["python3 encode_task_macs2_signal_track_chip.py a b"],
             inputs=[{"path": "/scratch/in.tagAlign.gz", "md5": "0" * 32}],
             outputs=[{"path": "/scratch/out.bigwig", "md5": "a" * 32}],
             subsample_seed=[{"file": "/scratch/in.tagAlign.gz", "uncompressed_bytes": 123}],
             patched_script=None,
             caper=None if route == "bam" else
             {"input_json": "/x/in.json", "workflow_id": "abc", "metadata_json": "/x/m.json"},
             slurm_job_ids=["123", "456_7"], code={"snapshot_dir": "/scratch/code/C8",
                                                   "git_sha": "3fb95b8"},
             created_utc="2026-09-17T12:00:00+00:00")
    d.update(over)
    return d


def _product(rec, root, cov_kw=None, prov_kw=None, control_npz=None):
    cov_kw = cov_kw or {}
    cov = _cov(**cov_kw)
    pdir = Path(root) / cov["pid"]
    pdir.mkdir(parents=True)
    (pdir / "counts25.npz").write_bytes(b"counts-" + cov["pid"].encode())
    (pdir / "pval25.npz").write_bytes(b"pval-" + cov["pid"].encode())
    has_ctl = cov["control"] is not None if control_npz is None else control_npz
    if has_ctl:
        (pdir / "control_counts25.npz").write_bytes(b"ctl-" + cov["pid"].encode())
    rec.write_covariates(pdir / "covariates.json", **cov)
    rec.write_provenance(pdir / "provenance.json", **_prov(pid=cov["pid"], **(prov_kw or {})))
    return pdir


def _rows_tsv(path, covs, route="bam"):
    # written as `arms.py rows` writes it: plain tab joins, knob_value as unquoted-by-csv JSON text
    lines = ["\t".join(_rows_header())]
    for c in covs:
        lines.append("\t".join([c["pid"], c["biosample"], c["track"], c["cell"], c["assay"],
                                c["arm"], c["level"], route, c["knob"], json.dumps(c["knob_value"]),
                                "chip"]))
    Path(path).write_text("\n".join(lines) + "\n")
    return path


# ---- stdlib only ---------------------------------------------------------------------------------

def test_imports_are_stdlib_only_plus_sibling_arms():
    tree = ast.parse(TOOL.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names - {"arms"} <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)


# ---- covariates ----------------------------------------------------------------------------------

def test_write_covariates_fills_fixed_fields_and_round_trips(rec, tmp_path):
    out = tmp_path / "covariates.json"
    rec.write_covariates(out, **_cov())
    d = json.loads(out.read_text())
    assert list(d) == list(rec.COVARIATE_KEYS)
    assert d["schema"] == 1
    assert d["log2_depth"] == math.log2(30000000)
    assert d["count_rule"] == ("store overlap rule: +1 to bins floor(start/25)..floor(end/25) "
                               "inclusive, grid floor(len/25)")
    assert d["wording_note"] == ("t112 says read-start counts; products use the store overlap rule "
                                 "so the loader reads them unchanged")
    assert rec.covariates_problems(d) == []


def test_covariates_use_candi_covariate_names(rec):
    # DATA.md "Four covariate rows": log2(depth), assay_id (from the assay name), read_length,
    # run_type as the file_metadata.json string
    for k in ("depth", "log2_depth", "assay", "read_length", "run_type"):
        assert k in rec.COVARIATE_KEYS
    assert rec.RUN_TYPES == ("single-ended", "paired-ended")


@pytest.mark.parametrize("key", ["pid", "biosample", "track", "cell", "assay", "arm", "level",
                                 "knob", "knob_value", "depth", "read_length", "run_type",
                                 "fraglen", "control"])
def test_write_covariates_missing_key_names_it(rec, tmp_path, key):
    kw = _cov()
    del kw[key]
    out = tmp_path / "covariates.json"
    with pytest.raises(ValueError, match=repr(key)):
        rec.write_covariates(out, **kw)
    assert not out.exists()


@pytest.mark.parametrize("over,key", [
    (dict(run_type="paired-end"), "run_type"),
    (dict(depth=True), "depth"),
    (dict(depth=0), "depth"),
    (dict(read_length=101.0), "read_length"),
    (dict(fraglen="180"), "fraglen"),
    (dict(log2_depth=24.0), "log2_depth"),
    (dict(pid="C19M16__base__other"), "pid"),
    (dict(biosample="C19__base__base"), "biosample"),
    (dict(control={"accession": "ENCFF433TZR", "source": "foreign", "reads": 1}), "control.source"),
    (dict(control={"accession": "ENCFF433TZR", "source": "matched"}), "control"),
    (dict(count_rule="read starts"), "count_rule"),
    (dict(knob_value=[1, 2]), "knob_value"),
    (dict(schema=2), "schema"),
    (dict(extra=1), "extra"),
])
def test_write_covariates_rejects(rec, tmp_path, over, key):
    with pytest.raises(ValueError, match=key):
        rec.write_covariates(tmp_path / "c.json", **_cov(**over))


def test_covariates_accepts_every_knob_value_kind(rec, tmp_path):
    for i, v in enumerate([None, True, 0.5, 15000000, "ENCFF337JNL"]):
        rec.write_covariates(tmp_path / f"c{i}.json", **_cov(knob_value=v))
    rec.write_covariates(tmp_path / "none.json", **_cov(arm="ctlid", level="none", control=None))


# ---- provenance ----------------------------------------------------------------------------------

def test_write_provenance_bam_and_fastq(rec, tmp_path):
    rec.write_provenance(tmp_path / "b.json", **_prov())
    d = json.loads((tmp_path / "b.json").read_text())
    assert list(d) == list(rec.PROVENANCE_KEYS)
    rec.write_provenance(tmp_path / "f.json", **_prov(route="fastq"))
    atac = {"repo": "ENCODE-DCC/atac-seq-pipeline", "release": "v2.2.3", "sif": "/x/atac.sif",
            "sif_md5": "04d9fa482d67cf633845653750f58cef", "sif_sha256": "b" * 64}
    rec.write_provenance(tmp_path / "a.json", **_prov(pipeline=atac, patched_script={
        "original_md5": "6039e1c382ef1e8afa3dbfd5ea103ba8", "patched_md5": "c" * 32,
        "diff": "-a\n+b\n"}))


def test_write_provenance_defaults_created_utc(rec, tmp_path):
    kw = _prov()
    del kw["created_utc"]
    d = rec.write_provenance(tmp_path / "p.json", **kw)
    assert d["created_utc"].endswith("+00:00")


@pytest.mark.parametrize("over,key", [
    (dict(route="bigwig"), "route"),
    (dict(route="fastq", caper=None), "caper"),
    (dict(caper={"input_json": "a", "workflow_id": "b", "metadata_json": "c"}), "caper"),
    (dict(pipeline={"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.2.2", "sif": "/x",
                    "sif_md5": "04d9fa482d67cf633845653750f58cef", "sif_sha256": None}),
     "sif_md5"),
    (dict(pipeline={"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.1.0", "sif": "/x",
                    "sif_md5": "f6e408f7e77bafde4556883f33552191", "sif_sha256": None}),
     "release"),
    (dict(pipeline={"repo": "ENCODE-DCC/dnase-seq-pipeline", "release": "v3.0.0", "sif": "/x",
                    "sif_md5": "f6e408f7e77bafde4556883f33552191", "sif_sha256": None}), "repo"),
    (dict(commands=[]), "commands"),
    (dict(inputs=[{"path": "/x", "md5": "XYZ"}]), "inputs"),
    (dict(outputs=[]), "outputs"),
    (dict(subsample_seed=[{"file": "/x", "uncompressed_bytes": -1}]), "uncompressed_bytes"),
    (dict(patched_script={"original_md5": "0" * 32, "patched_md5": "0" * 32}), "diff"),
    (dict(slurm_job_ids=[123]), "slurm_job_ids"),
    (dict(code={"snapshot_dir": "/x"}), "git_sha"),
    (dict(created_utc="yesterday"), "created_utc"),
])
def test_write_provenance_rejects(rec, tmp_path, over, key):
    with pytest.raises(ValueError, match=key):
        rec.write_provenance(tmp_path / "p.json", **_prov(**over))


def test_write_provenance_missing_key_names_it(rec, tmp_path):
    for key in ("pid", "route", "pipeline", "commands", "inputs", "outputs", "subsample_seed",
                "patched_script", "caper", "slurm_job_ids", "code"):
        kw = _prov()
        del kw[key]
        with pytest.raises(ValueError, match=repr(key)):
            rec.write_provenance(tmp_path / "p.json", **kw)


# ---- validate ------------------------------------------------------------------------------------

def _tree(rec, root):
    kws = [dict(),
           dict(arm="depth", level="15M", knob="treatment_reads", knob_value=15000000),
           dict(arm="ctlid", level="other", knob="control_accession", knob_value="ENCFF337JNL",
                control="other"),
           dict(arm="ctlid", level="none", knob="control_accession", knob_value="none",
                control=None)]
    for kw in kws:
        _product(rec, root, cov_kw=kw)
    return [_cov(**kw) for kw in kws]


def test_validate_rows_ok(rec, tmp_path, capsys):
    covs = _tree(rec, tmp_path / "products")
    rows = _rows_tsv(tmp_path / "rows.tsv", covs)
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows),
                   "--expect", "4"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and out == ["OK 4"]


def test_validate_without_rows_counts_covariate_dirs(rec, tmp_path, capsys):
    _tree(rec, tmp_path / "products")
    (tmp_path / "products" / "stray").mkdir()
    rc = rec.main(["validate", "--products", str(tmp_path / "products")])
    assert rc == 0 and capsys.readouterr().out.strip().splitlines()[-1] == "OK 4"


def test_validate_expect_mismatch_fails(rec, tmp_path, capsys):
    covs = _tree(rec, tmp_path / "products")
    rows = _rows_tsv(tmp_path / "rows.tsv", covs)
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows),
                   "--expect", "5"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1 and out[-1] == "FAIL 4" and "expected 5 products, checked 4" in out


def test_validate_default_products_is_cf(rec):
    assert rec.DEFAULT_PRODUCTS == "/scratch/mforooz/t112_cf/products"


@pytest.mark.parametrize("breakage,needle", [
    (lambda p: (p / "C19M16__base__base" / "pval25.npz").unlink(), "pval25.npz: missing"),
    (lambda p: (p / "C19M16__base__base" / "counts25.npz").write_bytes(b""), "counts25.npz: empty"),
    (lambda p: (p / "C19M16__base__base" / "control_counts25.npz").unlink(),
     "control_counts25.npz: missing"),
    (lambda p: (p / "C19M16__ctlid__none" / "control_counts25.npz").write_bytes(b"x"),
     "control_counts25.npz: present"),
    (lambda p: (p / "C19M16__depth__15M" / "provenance.json").write_text("{"), "unreadable JSON"),
    (lambda p: (p / "C19M16__depth__15M" / "covariates.json").unlink(), "covariates.json: missing"),
    (lambda p: [f.unlink() for f in (p / "C19M16__ctlid__other").iterdir()]
     and (p / "C19M16__ctlid__other").rmdir(), "product dir missing"),
])
def test_validate_rows_catches(rec, tmp_path, capsys, breakage, needle):
    covs = _tree(rec, tmp_path / "products")
    rows = _rows_tsv(tmp_path / "rows.tsv", covs)
    breakage(tmp_path / "products")
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows),
                   "--expect", "4"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1 and out[-1] == "FAIL 4"
    assert any(needle in line for line in out[:-1]), out


def test_validate_catches_record_that_disagrees_with_rows(rec, tmp_path, capsys):
    covs = _tree(rec, tmp_path / "products")
    covs[1] = dict(covs[1], knob_value=7500000)
    rows = _rows_tsv(tmp_path / "rows.tsv", covs)
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows)])
    out = capsys.readouterr().out
    assert rc == 1 and "C19M16__depth__15M: covariates.knob_value" in out


def test_validate_knob_value_bool_is_not_int(rec, tmp_path, capsys):
    root = tmp_path / "products"
    _product(rec, root, cov_kw=dict(arm="dedup", level="off", knob="chip.no_dup_removal",
                                    knob_value=1), prov_kw=dict(route="fastq"))
    rows = _rows_tsv(tmp_path / "rows.tsv", [_cov(arm="dedup", level="off",
                                                  knob="chip.no_dup_removal", knob_value=True)],
                     route="fastq")
    rc = rec.main(["validate", "--products", str(root), "--rows", str(rows)])
    assert rc == 1 and "knob_value" in capsys.readouterr().out


def test_validate_catches_duplicate_pid_and_route(rec, tmp_path, capsys):
    covs = _tree(rec, tmp_path / "products")
    rows = _rows_tsv(tmp_path / "rows.tsv", covs + covs[:1], route="fastq")
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows)])
    out = capsys.readouterr().out
    assert rc == 1 and "duplicate pid" in out and "provenance.route" in out


def test_validate_catches_pipeline_that_disagrees_with_rows(rec, tmp_path, capsys):
    covs = _tree(rec, tmp_path / "products")
    rows = tmp_path / "rows.tsv"
    rows.write_text(_rows_tsv(tmp_path / "r0.tsv", covs).read_text().replace("\tchip\n", "\tatac\n"))
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows)])
    out = capsys.readouterr().out
    assert rc == 1 and "provenance.pipeline.repo" in out


def test_validate_refuses_rows_tsv_without_arms_header(rec, tmp_path, capsys):
    _tree(rec, tmp_path / "products")
    rows = tmp_path / "rows.tsv"
    rows.write_text("pid\nC19M16__base__base\n")
    rc = rec.main(["validate", "--products", str(tmp_path / "products"), "--rows", str(rows)])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1 and out[-1] == "FAIL 0" and "arms.HEADER" in out[0]


def test_cli_products_defaults_to_cf(rec, tmp_path, capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(rec, "validate", lambda p, r, e: seen.setdefault("v", p) and (0, []))
    monkeypatch.setattr(rec, "write_manifest", lambda p, o: seen.setdefault("m", p) and (0, []))
    rec.main(["validate", "--rows", "x.tsv", "--expect", "3"])
    rec.main(["manifest", "--out", str(tmp_path / "M.tsv")])
    assert seen == {"v": "/scratch/mforooz/t112_cf/products", "m": "/scratch/mforooz/t112_cf/products"}


def _arms_product(rec, root, r):
    """A product for one real `arms.py rows` row, with the control each arm implies."""
    import arms
    t = arms.TRACKS[r["track"]]
    kv = json.loads(r["knob_value"])
    if r["pipeline"] == "atac" or (r["arm"] == "ctlid" and r["level"] == "none"):
        control = None
    elif r["arm"] == "ctlid":
        control = {"accession": kv, "source": "other", "reads": arms.CONTROLS[kv]["reads"]}
    else:
        control = {"accession": t["ctl_acc"], "source": "matched",
                   "reads": arms.CONTROLS[t["ctl_acc"]]["reads"]}
    cov = dict(pid=r["pid"], biosample=r["biosample"], track=r["track"], cell=r["cell"],
               assay=r["assay"], arm=r["arm"], level=r["level"], knob=r["knob"], knob_value=kv,
               depth=kv if r["arm"] == "depth" else t["subsample"], read_length=t["read_length"],
               run_type="paired-ended" if r["arm"] == "pe" else "single-ended",
               fraglen=kv if r["arm"] == "extsize" else t["fraglen"], control=control)
    pipeline = {"chip": {"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.2.2",
                         "sif": "/x/chip.sif", "sif_md5": "f6e408f7e77bafde4556883f33552191",
                         "sif_sha256": None},
                "atac": {"repo": "ENCODE-DCC/atac-seq-pipeline", "release": "v2.2.3",
                         "sif": "/x/atac.sif", "sif_md5": "04d9fa482d67cf633845653750f58cef",
                         "sif_sha256": None}}[r["pipeline"]]
    pdir = Path(root) / r["pid"]
    pdir.mkdir(parents=True)
    for name in ("counts25.npz", "pval25.npz") + (("control_counts25.npz",) if control else ()):
        (pdir / name).write_bytes(name.encode())
    rec.write_covariates(pdir / "covariates.json", **cov)
    rec.write_provenance(pdir / "provenance.json",
                         **_prov(pid=r["pid"], route=r["route"], pipeline=pipeline))


def test_validate_every_arms_row_130(rec, tmp_path):
    # the acceptance form: arms.py rows -> one product each -> `validate --rows --expect 130`
    rows = tmp_path / "rows_all.tsv"
    r = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "all", "--dnase", "atac",
                        "--ratio", "yes"], capture_output=True, text=True, check=True)
    rows.write_text(r.stdout)
    for row in csv.DictReader(rows.open(), delimiter="\t", quoting=csv.QUOTE_NONE):
        _arms_product(rec, tmp_path / "products", row)
    r = subprocess.run([sys.executable, str(TOOL), "validate", "--products",
                        str(tmp_path / "products"), "--rows", str(rows), "--expect", "130"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip().splitlines() == ["OK 130"], r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(TOOL), "manifest", "--products",
                        str(tmp_path / "products"), "--out", str(tmp_path / "MANIFEST.tsv")],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "OK 130", r.stdout + r.stderr
    assert len((tmp_path / "MANIFEST.tsv").read_text().splitlines()) == 131


def test_validate_cli_as_script(rec, tmp_path):
    covs = _tree(rec, tmp_path / "products")
    rows = _rows_tsv(tmp_path / "rows.tsv", covs)
    r = subprocess.run([sys.executable, str(TOOL), "validate", "--products",
                        str(tmp_path / "products"), "--rows", str(rows), "--expect", "4"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "OK 4", r.stdout + r.stderr


# ---- manifest ------------------------------------------------------------------------------------

def test_manifest_columns_and_values(rec, tmp_path, capsys):
    root = tmp_path / "products"
    _tree(rec, root)
    out = tmp_path / "MANIFEST.tsv"
    rc = rec.main(["manifest", "--products", str(root), "--out", str(out)])
    assert rc == 0 and capsys.readouterr().out.strip() == "OK 4"
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == [
        "pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
        "knob_value", "depth", "read_length", "run_type", "fraglen", "control_accession",
        "control_source", "pipeline_repo", "pipeline_release", "sif_md5", "counts25_md5",
        "pval25_md5", "control_counts25_md5", "slurm_job_ids", "product_dir"]
    assert '\t"ENCFF337JNL"\t' in out.read_text()  # JSON text, not csv-quoted
    rows = list(csv.DictReader(out.open(), delimiter="\t", quoting=csv.QUOTE_NONE))
    assert [r["pid"] for r in rows] == sorted(r["pid"] for r in rows)
    by = {r["pid"]: r for r in rows}
    base = by["C19M16__base__base"]
    assert base["knob_value"] == "" and base["control_accession"] == "ENCFF433TZR"
    assert base["control_source"] == "matched" and base["depth"] == "30000000"
    assert base["slurm_job_ids"] == "123,456_7"
    assert base["sif_md5"] == "f6e408f7e77bafde4556883f33552191"
    assert base["pipeline_repo"] == "ENCODE-DCC/chip-seq-pipeline2"
    assert base["pipeline_release"] == "v2.2.2" and base["route"] == "bam"
    pdir = root / "C19M16__base__base"
    assert base["counts25_md5"] == hashlib.md5((pdir / "counts25.npz").read_bytes()).hexdigest()
    assert base["control_counts25_md5"] == hashlib.md5(
        (pdir / "control_counts25.npz").read_bytes()).hexdigest()
    assert base["product_dir"] == str(pdir.resolve())
    assert by["C19M16__depth__15M"]["knob_value"] == "15000000"
    assert by["C19M16__ctlid__other"]["knob_value"] == '"ENCFF337JNL"'
    none = by["C19M16__ctlid__none"]
    assert none["control_accession"] == "" and none["control_source"] == ""
    assert none["control_counts25_md5"] == ""


def test_manifest_refuses_invalid_product(rec, tmp_path, capsys):
    root = tmp_path / "products"
    _tree(rec, root)
    (root / "C19M16__base__base" / "pval25.npz").unlink()
    out = tmp_path / "MANIFEST.tsv"
    rc = rec.main(["manifest", "--products", str(root), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and not out.exists() and "pval25.npz: missing" in text
