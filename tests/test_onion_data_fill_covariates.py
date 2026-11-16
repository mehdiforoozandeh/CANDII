"""Close-out D1: `fill_covariates.py` fills lab, antibody, donor and a code-only platform into a new
tables version built from the tiny tree of `test_onion_data_build_tables.py`; every other cell and
file stays byte-identical, the relaxed superset check passes, the strict one fails, the
vocabularies hold training values only (a value seen only in validation or blind rows encodes to
0), and the training ids reach the model's covariate vectors."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import build_tables as BT  # noqa: E402
from onion_data import fill_covariates as FC  # noqa: E402
from minicandi.covariates import CATEGORICAL, CAT_TABLE, CovariateEncoder  # noqa: E402
from minicandi.contract import META_COLUMNS  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, read_tsv  # noqa: E402
from tests.test_onion_data_build_tables import (BLD, CTL, TRN, VAL, build_argv,  # noqa: E402
                                                make_tree)
from tests.test_onion_data_check_dataset import _run  # noqa: E402


def _write(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _catalog(d: Path, exps: list) -> Path:
    d.mkdir(parents=True)
    _write(d / "experiments.tsv", [{"experiment": x, "lab": "Lab A" if x != BLD else "Lab B"}
                                   for x in exps])
    libs, files = [], []
    for x in exps:
        for br in ("1", "2"):
            libs.append({"experiment": x, "library": f"LIB{x[-3:]}{br}", "donor":
                         "DON1" if (x != VAL or br == "1") else "DON2", "bio_rep": br,
                         "antibody": "" if x == CTL else ("AB2" if (x, br) == (VAL, "2") else "AB1")})
            files.append({"file": f"F{x[-3:]}{br}", "experiment": x, "file_format": "fastq",
                          "platform": "Illumina HiSeq 4000", "status": "released",
                          "bio_reps": br})
    _write(d / "libraries.tsv", libs)
    _write(d / "files.tsv", files)
    return d


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    t = make_tree(tmp_path_factory.mktemp("tree"))
    base = t["root"] / "tables" / "v1"
    assert BT.main(build_argv(t, base)) == 0
    rows = read_tsv(base / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    cat = _catalog(t["root"] / "catalog", sorted({r["experiment"] for r in rows}))
    side = t["root"] / "side.tsv"
    assert FC.main(["join", "--tables", str(base), "--catalog", str(cat), "--out", str(side)]) == 0
    new = t["root"] / "tables" / "v1_1"
    assert FC.main(["apply", "--base", str(base), "--side", str(side), "--out", str(new)]) == 0
    return t, base, new, side


def test_columns_filled_and_rest_identical(filled):
    t, base, new, _ = filled
    old = read_tsv(base / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    got = {r["measurement_id"]: r for r in read_tsv(new / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)}
    assert len(got) == len(old)
    for r in old:
        n = got[r["measurement_id"]]
        for c in MEASUREMENT_COLUMNS:
            if c not in FC.FILLED:
                assert n[c] == r[c], (r["measurement_id"], c)
        assert n["lab"] == ("Lab B" if r["experiment"] == BLD else "Lab A")
        if r["platform"] in ("", "Illumina HiSeq 4000"):
            assert n["platform"] == "OBI:0002049"
        else:                                            # the tree's "Illumina": no code, kept
            assert n["platform"] == r["platform"]
        if r["experiment"] == CTL:
            assert n["antibody"] == ""                   # a control has no antibody
        elif r["experiment"] == VAL:                     # bio-rep filter; the pool is mixed
            want = {"br1": "AB1", "br2": "AB2"}.get(r["unit"], FC.MIXED)
            assert n["antibody"] == want, r["measurement_id"]
        else:
            assert n["antibody"] == "AB1"
        if r["experiment"] == VAL:                       # two donors: br2 differs
            want = {"br1": "DON1", "br2": "DON2"}.get(r["unit"], FC.MIXED)
            assert n["donor"] == want, r["measurement_id"]
        else:
            assert n["donor"] == "DON1"
    for name in (BT.PAIRS, BT.WITHHELD, BT.MISSING):
        assert (new / name).read_bytes() == (base / name).read_bytes(), name
    vocab = json.loads((new / BT.COVARIATES).read_text())["vocab"]
    assert vocab["antibody"] == ["AB1"]                  # AB2 and mixed occur in VAL rows only
    assert vocab["donor"] == ["DON1"]                    # DON2 and mixed: VAL rows only
    assert vocab["lab"] == ["Lab A"]                     # Lab B: blind rows only
    assert (new / FC.CHANGES).is_file() and (new / FC.SIDE_NAME).is_file()
    assert json.loads((new / BT.BUILD_INFO).read_text())["covariates_of"] == str(base.resolve())


def test_apply_never_overwrites(filled):
    _, base, new, side = filled
    assert FC.main(["apply", "--base", str(base), "--side", str(side), "--out", str(new)]) == 2


def test_relaxed_superset_check(filled, capsys):
    t, base, new, _ = filled
    code, c, txt = _run(t, new, capsys, "--superset-of", str(base), "--no-widths")
    assert code == 1 and c["rows of the older tables changed or missing"] > 0, txt
    code, c, txt = _run(t, new, capsys, "--superset-of", str(base), "--no-widths",
                        "--covariate-columns", ",".join(FC.FILLED))
    assert code == 0 and c["rows of the older tables changed or missing"] == 0, txt


def test_value_outside_training_encodes_to_zero(filled):
    """D1 review finding 1: a value that only validation or blind rows hold gets id 0 (unknown),
    never an embedding row that no training gradient reaches; training values get ids > 0."""
    _, _, new, _ = filled
    enc = CovariateEncoder(json.loads((new / BT.COVARIATES).read_text()))
    rows = read_tsv(new / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    train = {f: {r[f] for r in rows if r["in_eic"] == "train"} for f in CATEGORICAL.values()}
    n_zero = 0
    for r in rows:
        v = enc.c_in(r)
        for col, field in CATEGORICAL.items():
            i = int(v[META_COLUMNS.index(col)])
            assert 0 <= i < CAT_TABLE
            assert (i > 0) == (bool(r[field]) and r[field] in train[field]), (r["measurement_id"], field)
            n_zero += r["in_eic"] != "train" and bool(r[field]) and i == 0
    assert n_zero > 0
    blind = next(r for r in rows if r["experiment"] == BLD)
    assert enc.cat_id("lab", blind["lab"]) == 0 and blind["lab"] == "Lab B"
    val = next(r for r in rows if r["experiment"] == VAL and r["unit"] == "br2")
    assert enc.cat_id("donor", val["donor"]) == 0 and enc.cat_id("antibody", val["antibody"]) == 0


def test_derived_rows_inherit_their_unit(filled):
    """A derived arm (depth thinning) of a unit carries the unit's base-view values."""
    _, _, new, _ = filled
    rows = read_tsv(new / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    base = {(r["experiment"], r["unit"]): r for r in rows if r["arm_knob"] == ""
            and r["layer"] == "uniform"}
    n = 0
    for r in rows:
        b = base.get((r["experiment"], r["unit"]))
        if r["arm_knob"] and b is not None:
            assert all(r[c] == b[c] for c in FC.FILLED), r["measurement_id"]
            n += 1
    assert n > 0


def test_covariate_columns_refuse_a_non_covariate_field(filled, capsys):
    t, base, new, _ = filled
    code, _, txt = _run(t, new, capsys, "--superset-of", str(base), "--no-widths",
                        "--covariate-columns", "lab,in_eic")
    assert code == 2 and "must be categorical covariate fields" in txt


def test_platform_of():
    fq = [{"platform": "Illumina HiSeq 2000", "status": "released"},
          {"platform": "Illumina Genome Analyzer IIx", "status": "released"},
          {"platform": "Illumina NovaSeq 6000", "status": "archived"}]
    assert FC.platform_of({"platform": ""}, fq) == ("OBI:0002000,OBI:0002001",
                                                    "platform: from the catalog FASTQs")
    assert FC.platform_of({"platform": "OBI:0002001,OBI:0000703"}, []) == (
        "OBI:0000703,OBI:0002001", "")
    assert FC.platform_of({"platform": "Illumina HiSeq 4000"}, []) == ("OBI:0002049", "")
    v, note = FC.platform_of({"platform": ""}, fq[2:])
    assert v == "" and "no code" in note
    rep = [dict(fq[0], bio_reps="1"), dict(fq[1], bio_reps="2")]       # native br rows: own FASTQs
    assert FC.platform_of({"platform": "", "unit": "br2", "bio_rep": "2"}, rep)[0] == "OBI:0002000"
    assert FC.platform_of({"platform": "", "unit": "pooled", "bio_rep": ""}, rep)[0] == \
        "OBI:0002000,OBI:0002001"
