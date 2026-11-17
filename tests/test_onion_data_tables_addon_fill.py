"""Close-out D2: `build_tables.py addon --catalog` fills lab, antibody, donor and platform of the
appended second-experiment rows by the D1 rule (`fill_covariates.join_rows`). Without `--catalog`
the add-on is unchanged: every file except `build_info.json` is byte-identical to a build with the
option absent, and `build_info.json` gains no key. Base rows are never touched either way."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import build_tables as BT  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, read_tsv  # noqa: E402
from tests.test_onion_data_fill_covariates import _write  # noqa: E402
from tests.test_onion_data_tables_addon import (SXA, SXB, SXE, addon_argv,  # noqa: E402
                                                make_addon_tree)

FILL_KEYS = ("covariates_filled_from", "covariates_filled_columns", "covariates_fill_notes")


def _catalog(d: Path) -> Path:
    """SXA: one lab, one donor, one antibody; SXB: two donors (mixed); SXE: not in the catalog."""
    d.mkdir(parents=True)
    _write(d / "experiments.tsv", [{"experiment": SXA, "lab": "Lab Other"},
                                   {"experiment": SXB, "lab": "Lab Same"}])
    _write(d / "libraries.tsv", [
        {"experiment": SXA, "library": "LSXA1", "bio_rep": "1", "donor": "DONX", "antibody": "ABX"},
        {"experiment": SXB, "library": "LSXB1", "bio_rep": "1", "donor": "DON1", "antibody": "ABY"},
        {"experiment": SXB, "library": "LSXB2", "bio_rep": "2", "donor": "DON2", "antibody": "ABY"}])
    _write(d / "files.tsv", [
        {"file": "FSXA1", "experiment": SXA, "file_format": "fastq", "status": "released",
         "platform": "Illumina HiSeq 2000", "bio_reps": "1"},
        {"file": "FSXB1", "experiment": SXB, "file_format": "fastq", "status": "released",
         "platform": "Illumina HiSeq 4000", "bio_reps": "1"}])
    return d


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    t = make_addon_tree(tmp_path_factory.mktemp("tree"))
    off = t["root"] / "tables" / "x_off"
    off2 = t["root"] / "tables" / "x_off2"
    on = t["root"] / "tables" / "x_on"
    cat = _catalog(t["root"] / "catalog")
    assert BT.main(addon_argv(t, off)) == 0
    assert BT.main(addon_argv(t, off2) + ["--catalog", ""]) == 0
    assert BT.main(addon_argv(t, on) + ["--catalog", str(cat)]) == 0
    return t, off, off2, on, cat


def _rows(d):
    return {r["measurement_id"]: r for r in read_tsv(d / "measurements.tsv", MEASUREMENT_COLUMNS)}


def test_off_switch_writes_the_same_files(built):
    _, off, off2, _, _ = built
    for p in sorted(off.iterdir()):
        if p.name == "build_info.json":
            a, b = json.loads(p.read_text()), json.loads((off2 / p.name).read_text())
            assert not set(FILL_KEYS) & set(a) and not set(FILL_KEYS) & set(b)
            for k in ("created_utc", "slurm_job_id"):
                a.pop(k), b.pop(k)
            assert a.keys() == b.keys()
            continue
        assert p.read_bytes() == (off2 / p.name).read_bytes(), p.name


def test_filled_rows_and_untouched_base(built):
    t, off, _, on, cat = built
    r_off, r_on = _rows(off), _rows(on)
    assert r_off.keys() == r_on.keys()
    base = _rows(t["base"])
    for m in base:                                   # base rows: byte-identical lines
        assert r_on[m] == r_off[m] == base[m]
    a = r_on[f"{SXA}.pooled.uniform.base"]
    # the D1 rule keeps a platform the build wrote ("Illumina": a name with no code, kept, noted)
    assert (a["lab"], a["donor"], a["antibody"], a["platform"]) == (
        "Lab Other", "DONX", "ABX", "Illumina")
    b = r_on[f"{SXB}.pooled.uniform.base"]
    assert b["lab"] == "Lab Same" and b["donor"] == "mixed" and b["antibody"] == "ABY"
    e = r_on[f"{SXE}.pooled.native.base"]
    assert e["lab"] == "" and e["donor"] == ""       # not in the catalog: empty, with a note
    for m, r in r_on.items():                         # every other column as without the switch
        for c in MEASUREMENT_COLUMNS:
            if c not in ("lab", "antibody", "donor", "platform"):
                assert r[c] == r_off[m][c], (m, c)
    info = json.loads((on / "build_info.json").read_text())
    assert info["covariates_filled_from"] == str(cat.resolve())
    notes = info["covariates_fill_notes"]
    assert "donor" in notes[f"{SXB}.pooled.uniform.base"]
    assert "not in the catalog" in notes[f"{SXE}.pooled.native.base"]
    for name in ("pairs.tsv", "evaluation_only.tsv", "missing.tsv", "covariates.json"):
        assert (on / name).read_bytes() == (off / name).read_bytes(), name


# --------------------------------------------- a filled base: fill table and CHANGES.md cover the add-on

def _merge_tsv(dst: Path, src: Path) -> None:
    """Append the data lines of src (same header) to dst."""
    import csv
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(dst, newline="") as fh:
        old = list(csv.DictReader(fh, delimiter="\t"))
    cols = list(old[0])
    _write(dst, old + [{c: r.get(c, "") for c in cols} for r in rows])


@pytest.fixture(scope="module")
def filled_base(tmp_path_factory):
    from onion_data import fill_covariates as FC
    from tests.test_onion_data_fill_covariates import _catalog as base_catalog
    t = make_addon_tree(tmp_path_factory.mktemp("treef"))
    root = t["root"]
    rows = read_tsv(t["base"] / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    cat = base_catalog(root / "catalog_all", sorted({r["experiment"] for r in rows}))
    sx = _catalog(root / "catalog_sx")
    for n in ("experiments.tsv", "libraries.tsv", "files.tsv"):
        _merge_tsv(cat / n, sx / n)
    side = root / "side.tsv"
    assert FC.main(["join", "--tables", str(t["base"]), "--catalog", str(cat), "--out", str(side)]) == 0
    fbase = root / "tables" / "v1_2"
    assert FC.main(["apply", "--base", str(t["base"]), "--side", str(side), "--out", str(fbase)]) == 0
    t["base"] = fbase
    on, off = root / "tables" / "v1_2x_on", root / "tables" / "v1_2x_off"
    assert BT.main(addon_argv(t, on) + ["--catalog", str(cat)]) == 0
    assert BT.main(addon_argv(t, off)) == 0
    return fbase, on, off


def test_fill_table_and_changes_cover_the_addon_rows(filled_base):
    from onion_data import fill_covariates as FC
    fbase, on, off = filled_base
    for name in (FC.SIDE_NAME, FC.CHANGES):
        old, new = (fbase / name).read_bytes(), (on / name).read_bytes()
        assert new.startswith(old) and len(new) > len(old), name
        assert (off / name).read_bytes() == old, name           # switch off: plain copy
    side = {r["measurement_id"]: r for r in read_tsv(on / FC.SIDE_NAME, FC.SIDE_COLUMNS)}
    added = {m: r for m, r in _rows(on).items() if r["in_eic"] == "evalonly"}
    assert len(added) == 3 and set(added) <= set(side)
    for m, r in added.items():
        for c in ("lab", "antibody", "donor", "platform"):
            assert side[m][c] == r[c], (m, c)
    assert "## Add-on: second experiments (close-out D2)" in (on / FC.CHANGES).read_text()


def test_catalog_and_work_snapshot_together(tmp_path):
    """Close-out D2 + D3: `--catalog` (covariate fill) and `--work-snapshot` (QC after the work
    folder left the disk) in one call give the QC of the disk build and the fill of the catalog."""
    t = make_addon_tree(tmp_path / "tree")
    run = t["work"] / SXA / "base"
    run.mkdir(parents=True, exist_ok=True)
    (run / "qc.json").write_text(json.dumps({"align_enrich": {"xcor_score": {"rep1": {
        "NSC": 1.7, "RSC": 1.1, "estimated_fragment_len": 190}}}}))
    cat = _catalog(tmp_path / "catalog")
    disk = tmp_path / "disk"
    assert BT.main(addon_argv(t, disk) + ["--catalog", str(cat)]) == 0
    snap = tmp_path / "work.json.gz"
    assert BT.main(["snapshot-work", "--work", str(t["work"]), "--out", str(snap)]) == 0
    t["work"].rename(tmp_path / "work_archived")
    both = tmp_path / "both"
    assert BT.main(addon_argv(t, both) + ["--catalog", str(cat), "--work-snapshot", str(snap)]) == 0
    for n in ("measurements.tsv", "pairs.tsv", "missing.tsv", "evaluation_only.tsv"):
        assert (disk / n).read_bytes() == (both / n).read_bytes(), n
    a = _rows(both)[f"{SXA}.pooled.uniform.base"]
    assert (a["nsc"], a["lab"], a["donor"]) == ("1.7", "Lab Other", "DONX")
    info = json.loads((both / "build_info.json").read_text())
    assert info["covariates_filled_from"] == str(cat.resolve())
    assert info["work_snapshot_sha256"] == BT.sha256_bytes(snap.read_bytes())
