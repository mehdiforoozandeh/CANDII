"""Tests for the EIC accession ledger and its published-list check (t158, D8).

Synthetic store manifests and portal caches only; no network. The real published fixture is
checked for shape (363 tracks, 267/45/51) because every later chunk relies on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import eic_accessions as E  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "eic_published_accessions.tsv"

PUBLISHED = """# synthetic published list
data_type\tcell_type_id\tmark_id\tcell_type\tassay\tfilename\texperiment
training_data\tC01\tM02\tcellA\tDNase-seq\tC01M02\tENCSR000AAA
training_data\tC01\tM17\tcellA\tH3K27me3\tC01M17\tENCSR000AAB
validation_data\tC02\tM17\tcellB\tH3K27me3\tC02M17\tENCSR000BBB
blind_data\tC02\tM22\tcellB\tH3K4me3\tC02M22\tENCSR000BBC
"""


def _track(assay, exp, f, depth=1e7, rl=36, rt="single-ended", lab="L1", plat="P1", bios="ENCBS1"):
    return {"assay": assay, "exp_accession": exp, "file_accession": f, "bios_accession": bios,
            "depth": depth, "read_length": rl, "run_type": rt, "lab": lab, "platform": plat}


def _manifest():
    return {"biosamples": {
        "T_cellA": {"tracks": [_track("DNase-seq", "ENCSR000AAA", "ENCFF1"),
                               _track("H3K27me3", "ENCSR000AAB", "ENCFF2"),
                               _track("chipseq-control", "ENCSR000CTA", "ENCFF3")]},
        "V_cellB": {"tracks": [_track("H3K27me3", "ENCSR000BBB", "ENCFF4", rl=100,
                                      rt="paired-ended", lab="L2", plat="P2")]},
        "B_cellB": {"tracks": [_track("H3K4me3", "ENCSR000BBC", "ENCFF5", rl=100,
                                      rt="paired-ended", lab="L2", plat="P2"),
                               _track("chipseq-control", "ENCSR000CTB", "ENCFF6")]},
    }}


def _encode():
    def exp(acc, target, ctl=()):
        return {"accession": acc, "status": "released", "target": {"label": target},
                "replication_type": "isogenic",
                "replicates": [{"biological_replicate_number": 1, "technical_replicate_number": 1,
                                "library": {"accession": acc + "L1",
                                            "biosample": {"accession": "ENCBS1"}}}],
                "possible_controls": [{"accession": c} for c in ctl]}
    exps = [exp("ENCSR000AAA", None), exp("ENCSR000AAB", "H3K27me3", ["ENCSR000CTA"]),
            exp("ENCSR000BBB", "H3K27me3", ["ENCSR000CTX"]),
            exp("ENCSR000BBC", "H3K4me3", ["ENCSR000CTB"])]
    files = {f"ENCFF{i}": {"accession": f"ENCFF{i}", "dataset": f"/experiments/{e}/",
                           "biological_replicates": [1]}
             for i, e in enumerate(["", "ENCSR000AAA", "ENCSR000AAB", "ENCSR000CTA",
                                    "ENCSR000BBB", "ENCSR000BBC", "ENCSR000CTB"]) if i}
    return {"experiments": {e["accession"]: e for e in exps}, "files": files, "fastqs": {}}


@pytest.fixture
def setup(tmp_path):
    pub = tmp_path / "pub.tsv"
    pub.write_text(PUBLISHED)
    cache = tmp_path / "encode.json"
    cache.write_text(json.dumps(_encode()))

    def build(manifest, name="ledger.tsv"):
        m = tmp_path / ("m_" + name + ".json")
        m.write_text(json.dumps(manifest))
        out = tmp_path / name
        assert E.main(["build", "--manifest", str(m), "--published", str(pub), "--out", str(out),
                       "--encode-cache", str(cache)]) == 0
        return out

    def check(ledger, capsys):
        capsys.readouterr()  # drop the build's own output
        code = E.main(["check", "--ledger", str(ledger), "--published", str(pub)])
        return code, capsys.readouterr().out.strip().splitlines()

    return build, check


def test_clean_ledger_passes(setup, capsys):
    build, check = setup
    led = build(_manifest())
    code, lines = check(led, capsys)
    assert code == 0 and lines[-1] == "mismatches: 0", lines
    rows = {(r["role"], r["eic_track"]): r for r in E.read_tsv(led)}
    assert set(rows) == {("train", "C01M02"), ("train", "C01M17"), ("val", "C02M17"),
                         ("blind", "C02M22"), ("train", "C01CTL"), ("blind", "C02CTL")}
    assert rows[("train", "C01M17")]["control_experiment"] == "ENCSR000CTA"
    assert rows[("train", "C01M17")]["assay_class"] == "broad_histone"
    assert rows[("blind", "C02M22")]["assay_class"] == "narrow_histone"
    assert rows[("train", "C01M02")]["assay_class"] == "accessibility_dnase"
    assert list(E.read_tsv(led)[0])[:20] == E.LEDGER_COLUMNS[:20]


def test_planted_wrong_accession_fails_and_is_named(setup, capsys):
    build, check = setup
    m = _manifest()
    m["biosamples"]["V_cellB"]["tracks"][0]["exp_accession"] = "ENCSR999ZZZ"
    code, lines = check(build(m), capsys)
    assert code == 1, lines
    named = [ln for ln in lines if ln.startswith("EXPERIMENT C02M17")]
    assert len(named) == 1 and "ENCSR999ZZZ" in named[0] and "ENCSR000BBB" in named[0], lines
    # the store file still belongs to the published experiment, so FILE flags it too
    assert lines[-1] == "mismatches: 2" and any(ln.startswith("FILE C02M17") for ln in lines)


def test_wrong_accession_edited_into_ledger_is_caught(setup, capsys):
    """The check re-reads the published list: editing the ledger's own columns cannot hide it."""
    build, check = setup
    led = build(_manifest())
    rows = E.read_tsv(led)
    for r in rows:
        if r["eic_track"] == "C01M02":
            r["experiment"] = r["published_experiment"] = "ENCSR999ZZZ"
    E.write_tsv(rows, led, E.LEDGER_COLUMNS)
    code, lines = check(led, capsys)
    assert code == 1 and any("C01M02" in ln and "ENCSR999ZZZ" in ln for ln in lines), lines


def test_missing_track_is_a_mismatch(setup, capsys):
    build, check = setup
    m = _manifest()
    m["biosamples"]["B_cellB"]["tracks"] = m["biosamples"]["B_cellB"]["tracks"][1:]
    code, lines = check(build(m), capsys)
    assert code == 1 and any(ln.startswith("MISSING C02M22 role=blind") for ln in lines), lines


def test_missing_ledger_row_is_a_mismatch(setup, capsys):
    build, check = setup
    led = build(_manifest())
    E.write_tsv([r for r in E.read_tsv(led) if r["eic_track"] != "C01M17"], led, E.LEDGER_COLUMNS)
    code, lines = check(led, capsys)
    assert code == 1 and any(ln.startswith("MISSING C01M17") for ln in lines), lines


def test_extra_track_is_a_mismatch(setup, capsys):
    build, check = setup
    m = _manifest()
    m["biosamples"]["T_cellA"]["tracks"].append(_track("H3K4me3", "ENCSR000NEW", "ENCFF9"))
    code, lines = check(build(m), capsys)
    assert code == 1 and lines[-1] == "mismatches: 1", lines
    assert lines[0].startswith("EXTRA C01M22 role=train")


def test_wrong_role_is_a_mismatch(setup, capsys):
    build, check = setup
    m = _manifest()
    m["biosamples"]["T_cellB"] = m["biosamples"].pop("V_cellB")
    code, lines = check(build(m), capsys)
    assert code == 1
    assert any(ln.startswith("ROLE C02M17 ledger role=train") for ln in lines), lines
    assert any(ln.startswith("MISSING C02M17 role=val") for ln in lines), lines


def test_control_outside_possible_controls_is_a_mismatch(setup, capsys):
    build, check = setup
    m = _manifest()
    m["biosamples"]["T_cellA"]["tracks"][2]["exp_accession"] = "ENCSR000OTH"
    code, lines = check(build(m), capsys)
    assert code == 1 and any(ln.startswith("CONTROL C01CTL") for ln in lines), lines


def test_store_file_from_another_experiment_is_a_mismatch(setup, capsys, tmp_path):
    """The accession matches the published list, but the data came from another experiment."""
    build, check = setup
    enc = _encode()
    enc["files"]["ENCFF1"]["dataset"] = "/experiments/ENCSR000NEW/"
    (tmp_path / "encode.json").write_text(json.dumps(enc))
    led = build(_manifest())
    code, lines = check(led, capsys)
    assert code == 1 and lines[-1] == "mismatches: 1", lines
    assert lines[0].startswith("FILE C01M02 role=train") and "ENCSR000NEW" in lines[0]
    row = [r for r in E.read_tsv(led) if r["eic_track"] == "C01M02"][0]
    assert row["match"] == "1" and "proposed fix" in row["note"]


def test_control_of_another_split_is_a_mismatch(setup, capsys):
    """A train control whose only ChIP owners are blind experiments brings a blind-only
    experiment into training (D8)."""
    build, check = setup
    m = _manifest()
    m["biosamples"]["T_cellB"] = {"tracks": [_track("chipseq-control", "ENCSR000CTB", "ENCFF6")]}
    led = build(m)
    code, lines = check(led, capsys)
    assert code == 1 and any(ln.startswith("CONTROL C02CTL role=train") for ln in lines), lines
    note = [r for r in E.read_tsv(led) if r["eic_track"] == "C02CTL" and r["role"] == "train"][0]
    assert "blind ChIP" in note["note"] and "proposed fix" in note["note"]


def test_imbalance_table(setup, tmp_path):
    build, _ = setup
    led = build(_manifest())
    out = tmp_path / "imb.tsv"
    assert E.main(["imbalance", "--ledger", str(led), "--out", str(out)]) == 0
    table = E.read_tsv(out)
    rt = {r["value"]: r for r in table if r["covariate"] == "run_type"}
    assert rt["single-ended"]["n_train"] == "2" and rt["paired-ended"]["n_val"] == "1"
    summ = {r["covariate"]: r for r in E.read_tsv(tmp_path / "imb_summary.tsv")}
    assert float(summ["run_type"]["tvd_train_vs_val"]) == pytest.approx(1.0)
    assert summ["lab"]["blind_tracks_with_value_absent_in_train"] == "1"


def test_real_published_fixture_shape():
    pub = E.load_published(FIXTURE)
    assert len(pub) == 363
    roles = {r: sum(p["role"] == r for p in pub) for r in E.ROLES}
    assert roles == {"train": 267, "val": 45, "blind": 51}
    assert len({p["eic_track"] for p in pub}) == 363
    assert all(p["experiment"].startswith("ENCSR") for p in pub)
    header = FIXTURE.read_text().splitlines()[:4]
    assert any("MOESM5_ESM.csv" in h for h in header) and any("md5" in h for h in header)
