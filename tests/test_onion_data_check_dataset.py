"""t152 chunk D6t: the dataset check passes on tables built from the tiny tree of
`test_onion_data_build_tables.py`, and each planted violation makes its own line nonzero and the
exit code 1."""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import build_tables as BT  # noqa: E402
from onion_data import check_dataset as CD  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, read_tsv  # noqa: E402
from tests.test_onion_data_build_tables import (BLD, TRN, VAL, build_argv,  # noqa: E402
                                                make_tree)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    t = make_tree(tmp_path_factory.mktemp("tree"))
    out = t["root"] / "tables" / "v1"
    assert BT.main(build_argv(t, out)) == 0
    return t, out


def _run(t, tables, capsys, *extra):
    capsys.readouterr()
    code = CD.main(["--tables", str(tables), "--published", str(t["published"]), *extra])
    txt = capsys.readouterr().out
    counts = {m.group(1): int(m.group(2)) for m in re.finditer(r"^([^:\n]+): (\d+)\b", txt, re.M)}
    return code, counts, txt


def _copy(out, tmp_path) -> Path:
    d = tmp_path / "tables"
    shutil.copytree(out, d)
    return d


def _edit(path, columns, fn):
    rows = read_tsv(path, columns)
    rows = fn(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


ZERO = ("measurements outside ledger", "accessions outside the published EIC list",
        "validation or blind measurement in a training pair",
        "withheld combinations in training pairs", "blind rows readable in train or val mode",
        "pairs between two testbed arms in pairs.tsv",
        "arms whose halves do not derive from the parent split", "sha256 mismatches")


def test_clean_tables_pass_with_rung_lines(built, capsys):
    t, out = built
    code, c, txt = _run(t, out, capsys)
    assert code == 0, txt
    for k in ZERO:
        assert c[k] == 0, k
    assert "rung sequencing H3K27ac: 0 pairs, not populated by EIC" in txt
    m = re.search(r"rung pseudoreplicate H3K27ac: (\d+) pairs, .*interval width ([0-9.e+-]+)", txt)
    assert m and int(m.group(1)) == 5 and float(m.group(2)) > 0
    assert re.search(r"rung isogenic H3K27ac: 1 pairs, 1 cells, interval width", txt)


def _plant_and_check(t, out, tmp_path, capsys, rule, plant, *extra):
    d = _copy(out, tmp_path)
    plant(d)
    code, c, txt = _run(t, d, capsys, *extra)
    assert code == 1, txt
    assert c[rule] > 0, txt
    return c


def test_row_outside_ledger_and_published_list(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            r = dict(next(r for r in rows if r["measurement_id"] == f"{TRN}.pooled.uniform.base"))
            r["measurement_id"] = "ENCSRZZZ999.pooled.uniform.base"
            r["experiment"] = "ENCSRZZZ999"
            return rows + [r]
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, fn)
    c = _plant_and_check(t, out, tmp_path, capsys, "measurements outside ledger", plant)
    assert c["accessions outside the published EIC list"] == 1


def test_accession_not_in_published_list(built, tmp_path, capsys):
    t, out = built
    pub = tmp_path / "pub.tsv"
    pub.write_text("".join(ln for ln in Path(t["published"]).read_text().splitlines(True)
                           if TRN not in ln))
    code, c, txt = _run(t, out, capsys, "--published", str(pub))
    assert code == 1 and c["accessions outside the published EIC list"] == 4, txt
    assert c["measurements outside ledger"] == 0


def test_validation_side_in_training_pair(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            for p in rows:
                if p["rung"] == "other_cell_type":
                    p["split"] = "train"
            return rows
        _edit(d / "pairs.tsv", PAIR_COLUMNS, fn)
    _plant_and_check(t, out, tmp_path, capsys,
                     "validation or blind measurement in a training pair", plant)


def test_withheld_combination_in_training_pair(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            a, b = f"{TRN}.pooled.uniform.depth0.25", f"{TRN}.pooled.uniform.depth0.50"
            return rows + [{"pair_id": BT.pair_id("designed", a, b), "a": a, "b": b,
                            "rung": "designed", "assay": "H3K27ac", "cell": "C02",
                            "knobs_differing": "depth", "subset_pair": "1", "split": "train",
                            "never_trained": "0"}]
        _edit(d / "pairs.tsv", PAIR_COLUMNS, fn)
    _plant_and_check(t, out, tmp_path, capsys, "withheld combinations in training pairs", plant)


def test_never_trained_flag_must_match_withheld(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            for p in rows:
                p["never_trained"] = "0"
            return rows
        _edit(d / "pairs.tsv", PAIR_COLUMNS, fn)
    _plant_and_check(t, out, tmp_path, capsys,
                     "never_trained flags that disagree with withheld.tsv", plant)


def test_evaluation_only_row_readable_in_val_mode(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            for r in rows:
                if r["layer"] == "testbed":
                    r["in_eic"] = "val"
            return rows
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, fn)
    c = _plant_and_check(t, out, tmp_path, capsys, "blind rows readable in train or val mode",
                         plant)
    assert c["blind rows readable in train or val mode"] == 2


def test_testbed_row_not_flagged(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        _edit(d / "evaluation_only.tsv", BT.EVAL_ONLY_COLUMNS, lambda rows: rows[1:])
    _plant_and_check(t, out, tmp_path, capsys, "testbed rows not flagged evaluation-only", plant)


def test_testbed_arms_pair_in_pairs_tsv(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        moved = [p for p in read_tsv(d / BT.SHARED, PAIR_COLUMNS) if p["a"].startswith(BLD)]
        _edit(d / "pairs.tsv", PAIR_COLUMNS, lambda rows: rows + moved[:1])
    _plant_and_check(t, out, tmp_path, capsys, "pairs between two testbed arms in pairs.tsv", plant)


def test_arm_with_its_own_split_in_a_designed_pair(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        moved = [p for p in read_tsv(d / BT.SHARED, PAIR_COLUMNS)
                 if p["a"] == f"{TRN}.pooled.uniform.base" and p["b"].endswith(".se")]
        assert moved
        _edit(d / "pairs.tsv", PAIR_COLUMNS, lambda rows: rows + moved)
    c = _plant_and_check(t, out, tmp_path, capsys,
                         "arms whose halves do not derive from the parent split", plant)
    assert c["arms whose halves do not derive from the parent split"] == 2


def test_sha256_mismatch(built, tmp_path, capsys):
    t, out = built

    def plant(d):
        def fn(rows):
            rows[0]["sha256"] = "0" * 64
            return rows
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, fn)
    _plant_and_check(t, out, tmp_path, capsys, "sha256 mismatches", plant)


def test_superset_of_older_tables(built, tmp_path, capsys):
    t, out = built
    v2 = tmp_path / "v2"
    assert BT.main(build_argv(t, v2, "1,2,3", "--superset-of", str(out))) == 0
    code, c, txt = _run(t, v2, capsys, "--superset-of", str(out), "--no-widths")
    assert code == 0 and c["rows of the older tables changed or missing"] == 0, txt
    _edit(v2 / "measurements.tsv", MEASUREMENT_COLUMNS,
          lambda rows: [r for r in rows if r["measurement_id"] != f"{VAL}.pooled.uniform.base"])
    code, c, txt = _run(t, v2, capsys, "--superset-of", str(out), "--no-widths")
    assert code == 1 and c["rows of the older tables changed or missing"] >= 1, txt
