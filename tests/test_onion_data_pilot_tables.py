"""Pilot tables over a tiny fake CANDI_STORE (t152 chunk P1): schema, roles, 36 channels, and
the tables load in OnionDataset unchanged."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from onion_data import pilot_tables as P  # noqa: E402
from tests.test_store_reader import make_store  # noqa: E402

RES = 25
SIZES = {"chr1": 600 * RES + 7, "chr2": 300 * RES + 3}
CTX = 128
A35 = C.ASSAYS
#: every panel track appears; V_aa's control is the SAME file as T_aa's (as 20 EIC controls are)
LAYOUT = {
    "T_aa": A35[:12] + (C.CONTROL,),
    "T_bb": A35[12:24] + (C.CONTROL,),
    "T_cc": A35[24:] + A35[:3] + (C.CONTROL,),
    "V_aa": A35[12:15] + (C.CONTROL,),
    "V_bb": A35[:2],
    "B_aa": A35[15:17],
    "B_cc": A35[3:5] + (C.CONTROL,),
}


def _patch_manifest(corpus: Path) -> None:
    """make_store gives every track exp ENCSR1; give each its own, and share T_aa's control."""
    mp = corpus / "manifest.json"
    man = json.loads(mp.read_text())
    for i, (bios, b) in enumerate(sorted(man["biosamples"].items())):
        for t in b["tracks"]:
            t["exp_accession"] = f"ENCSR{i:02d}{C.TRACK_INDEX[t['assay']]:03d}"
    ctl = {b: next(t for t in man["biosamples"][b]["tracks"] if t["assay"] == C.CONTROL)
           for b in ("T_aa", "V_aa")}
    ctl["V_aa"]["exp_accession"] = ctl["T_aa"]["exp_accession"]
    ctl["V_aa"]["file_accession"] = ctl["T_aa"]["file_accession"]
    mp.write_text(json.dumps(man))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pilot")
    corpus = make_store(tmp, tracks=LAYOUT, chrom_sizes=SIZES)
    _patch_manifest(corpus)
    out = tmp / "ONION" / "tables" / "pilot_eic"
    assert P.main(["build", "--corpus", str(corpus), "--out", str(out)]) == 0
    return corpus, out


def _rows(out):
    return D.read_tsv(out / "measurements.tsv", D.MEASUREMENT_COLUMNS)


def test_schema_roles_and_sharing(built):
    corpus, out = built
    rows = _rows(out)
    base = [r for r in rows if not r["arm_knob"]]
    n_store = sum(len(v) for v in LAYOUT.values())
    assert len(base) == n_store - 1                      # the shared control is one measurement
    for r in base:
        _, bios, assay = P.parse_store_path(r["counts_path"])
        assert r["cell"] == bios[2:] and r["assay"] == assay and r["layer"] == "native"
        assert r["in_eic"] == P.PREFIX_ROLE[bios[:2]]
        assert r["measurement_id"] == f"{r['experiment']}.pooled.native.base"
    ctl_aa = [r for r in base if r["cell"] == "aa" and r["assay"] == C.CONTROL]
    assert len(ctl_aa) == 1 and ctl_aa[0]["in_eic"] == "train"
    # a val track of cell aa names that shared control
    v = next(r for r in base if r["cell"] == "aa" and r["assay"] == A35[12])
    assert v["in_eic"] == "val" and v["control_measurement_id"] == ctl_aa[0]["measurement_id"]
    assert {r["assay"] for r in rows} == set(C.TRACKS)
    assert D.read_tsv(out / "withheld.tsv", D.WITHHELD_COLUMNS) == []
    assert (out.parent.parent / "genome").resolve() == (corpus.parent / "genome").resolve()


def test_pairs_subset_flag(built):
    _, out = built
    rows = {r["measurement_id"]: r for r in _rows(out)}
    pairs = D.read_tsv(out / "pairs.tsv", D.PAIR_COLUMNS)
    assert {p["rung"] for p in pairs} == {"other_cell_type", "designed"}
    for p in pairs:
        a, b = rows[p["a"]], rows[p["b"]]
        if p["rung"] == "designed":
            assert p["subset_pair"] == "1" and b["reads_subset_of"] == a["measurement_id"]
            assert b["arm_knob"] == "depth"
        else:
            assert p["subset_pair"] == "0" and a["cell"] != b["cell"]
            assert a["assay"] == b["assay"]
        assert p["never_trained"] == "0"


def test_check_passes_and_iterates_batches(built, capsys):
    _, out = built
    rc = P.main(["check", "--tables", str(out), "--batches", "2", "--batch-size", "2",
                 "--context-bins", str(CTX), "--val-targets", "2"])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "every counts_path readable" in text
    assert "channels: 36 of 36" in text
    assert "train 38+3 controls, val 5+1 controls, blind 4+1 controls" in text
    assert "val batch 0" in text and "check: PASS" in text


def test_training_draws_no_validation_or_blind_row(built):
    _, out = built
    rows = {r["measurement_id"]: r for r in _rows(out)}
    ds = D.OnionDataset(out, "train", {"context_bins": CTX, "batch_size": 2})
    rng = np.random.default_rng(1)
    for _ in range(500):
        plan = ds.plan(rng)
        for _, v_in, v_out, _ in plan["entries"]:
            for m in (v_in, v_out):
                assert m is None or rows[m]["in_eic"] == "train"
    b = ds.batch(2)
    assert tuple(b["x_counts"].shape) == (2, CTX, C.A)
    assert tuple(b["c_out"].shape) == (2, C.A, C.K_META)
    for meta in b["meta"]:
        assert all(rows[m]["in_eic"] == "train" for m in meta["inputs"] + meta["targets"])
    with pytest.raises(D.RoleError):
        ds.read(next(m for m, r in rows.items() if r["in_eic"] == "val"), "chr1", 0)
    with pytest.raises(D.RoleError):
        D.OnionDataset(out, "blind", {"context_bins": CTX})


def test_depth_arm_is_thinned_base(built):
    _, out = built
    rows = _rows(out)
    arm = next(r for r in rows if r["arm_knob"] == "depth" and r["arm_value"] == "0.25"
               and r["in_eic"] == "train")
    ds = D.OnionDataset(out, "train", {"context_bins": CTX})
    base = ds.read(arm["reads_subset_of"], "chr1", 0, 512)
    thin = ds.read(arm["measurement_id"], "chr1", 0, 512)
    assert (thin <= base).all() and thin.sum() < base.sum()
    assert abs(thin.sum() / base.sum() - 0.25) < 0.05
    parent = next(r for r in rows if r["measurement_id"] == arm["reads_subset_of"])
    assert int(arm["depth_reads"]) == round(int(parent["depth_reads"]) * 0.25)
