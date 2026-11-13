"""AGb: Max's signal-space baselines and the replicate-half row (tools/onion_eval/ag2_baselines.py).

Synthetic tables of tools/minicandi/synth.py with 9 cells, ChIP controls wired in, planted
pseudoreplicate signal halves on the validation and blind targets, and a truth directory of pinned
interface F whose truth is a planted copy of one training track's signal."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi.synth import make_evalset, make_synthetic_tables  # noqa: E402
from onion_eval import ag2_baselines as AB  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

RB = 256
REGIME = "leave_one_assay_out"
NOCTL = "s8"                    # this cell's ChIP training tracks get no control
OWN_CELL_COPY = "ENCSR999999.pooled.uniform.base"


def _write_rows(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _plant_halves(row, rng):
    d = Path(row["counts_path"][3:]).parent
    out = {}
    for nm in ("pr1_signal", "pr2_signal"):
        with h5py.File(d / f"{nm}.h5", "w") as f:
            for c, nb in (("chr1", 2048), ("chr2", 1024)):
                v = rng.gamma(1.0, 2.0, nb).astype(np.float32)
                f.create_dataset(c, data=v)
                out[(nm, c)] = v
    return out


def _truth_dir(root, ev_path, split, truths: dict):
    ev = json.loads(Path(ev_path).read_text())
    d = root / f"truth_{split}" / REGIME
    d.mkdir(parents=True)
    for mid, t in truths.items():
        np.save(d / f"{mid}.truth.npy", t.astype(np.float32))
    (d / "tracks.json").write_text(json.dumps({
        "method": "minicandi_final", "kind": "run", "evalset_sha256": E.sha256_file(ev_path),
        "regions_sha256": AB.regions_sha256(ev["regions"], RB), "split": split, "regime": REGIME,
        "region_bins": RB, "n_regions": len(ev["regions"]), "lambda_source": "observed_control",
        "targets": sorted(truths)}))
    return d.parent


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("agb")
    tdir = make_synthetic_tables(root, n_cells=9)
    rows = E.read_tsv(tdir / "measurements.tsv")
    ctl = {r["cell"]: r["measurement_id"] for r in rows
           if r["assay"] == C.CONTROL and not r["arm_knob"]}
    for r in rows:
        if r["assay"] not in ("DNase-seq", C.CONTROL) and r["cell"] != NOCTL:
            r["control_measurement_id"] = ctl[r["cell"]]
    by = {(r["cell"], r["assay"]): r for r in rows if not r["arm_knob"]}
    extra = dict(by[("s1", "H3K4me3")], measurement_id=OWN_CELL_COPY, experiment="ENCSR999999",
                 cell="s0", eic_track="C01M99", in_eic="train", control_measurement_id=ctl["s0"])
    rows.append(extra)
    _write_rows(tdir / "measurements.tsv", rows)
    rng = np.random.default_rng(1)
    halves = {by[k]["measurement_id"]: _plant_halves(by[k], rng)
              for k in (("s0", "H3K4me3"), ("s3", "H3K4me3"), ("s4", "H3K27ac"))}

    ev_val = make_evalset(tdir, n_regions=2, region_bins=RB, seed=3)
    ev = json.loads(ev_val.read_text())
    ev["region_bins"] = RB
    ev_val.write_text(json.dumps(ev))
    regions = [tuple(x) for x in ev["regions"]]
    by_id = {r["measurement_id"]: r for r in rows}
    copy_mid = by[("s1", "H3K4me3")]["measurement_id"]
    copy_sig, why = AB.training_signal(by_id[copy_mid], by_id, regions, RB)
    assert why is None
    val_target = by[("s0", "H3K4me3")]["measurement_id"]
    tf_val = _truth_dir(root, ev_val, "val", {val_target: copy_sig})

    bdir = root / "eval_blind"
    bdir.mkdir()
    ev_b = dict(ev, imputation_targets=[
        {"target": by[k]["measurement_id"], "inputs": [], "regime": REGIME, "split": "blind"}
        for k in (("s3", "H3K4me3"), ("s4", "H3K27ac"))], correction_pairs=[])
    ev_blind = bdir / "evalset_blind.json"
    ev_blind.write_text(json.dumps(ev_b))
    tf_blind = _truth_dir(root / "b", ev_blind, "blind",
                          {t["target"]: np.arcsinh(np.ones(len(regions) * RB))
                           for t in ev_b["imputation_targets"]})
    return dict(root=root, tdir=tdir, rows=rows, by=by, by_id=by_id, regions=regions,
                halves=halves, ev_val=ev_val, tf_val=tf_val, ev_blind=ev_blind, tf_blind=tf_blind,
                val_target=val_target, copy_mid=copy_mid, copy_sig=copy_sig)


@pytest.fixture(scope="module")
def val_out(world):
    out = world["root"] / "out_val"
    AB.run(world["tdir"], world["ev_val"], world["tf_val"], out, split="val", log=lambda *_: None)
    return out


def test_blocks_tail_joins_last():
    assert AB.blocks(16384) == [(0, 8000), (8000, 16384)]
    assert AB.blocks(256) == [(0, 256)]


def test_mean_of_identical_tracks_is_that_track():
    x = np.random.default_rng(0).gamma(1.0, 1.0, 500)
    np.testing.assert_allclose(AB.training_mean([x, x, x]), x, rtol=0, atol=1e-12)
    m, pick = AB.closest_k(x, [x] * 7)
    np.testing.assert_allclose(m, x, rtol=0, atol=1e-12)
    assert len(pick) == 5


def test_training_mean_matches_candidates(world, val_out):
    t = world["val_target"]
    cands = AB.candidate_rows(world["rows"], world["by_id"][t], E.second_experiment_ids(world["tdir"]))
    sigs = [AB.training_signal(r, world["by_id"], world["regions"], RB)[0] for r in cands]
    sigs = [s for s in sigs if s is not None]
    got = np.load(val_out / AB.MEAN / REGIME / f"{t}.signal.npy")
    np.testing.assert_allclose(got, AB.training_mean(sigs).astype(np.float32), rtol=1e-6)
    meta = json.loads((val_out / AB.MEAN / REGIME / "tracks.json").read_text())
    assert meta["kind"] == "oracle_baseline" and meta["targets"] == [t]
    assert meta["n_regions"] == 2 and meta["region_bins"] == RB
    # the ChIP training track with no control is left out and counted
    noctl = world["by"][(NOCTL, "H3K4me3")]["measurement_id"]
    assert noctl in meta["training_tracks_left_out"]
    assert meta["n_training_tracks"][t] == len(sigs) == 6


def test_half_row_is_planted_half_and_truth_is_other_half(world, val_out):
    t = world["val_target"]
    h = world["halves"][t]
    exp1 = np.concatenate([np.arcsinh(h[("pr1_signal", c)][s:s + RB].astype(np.float64))
                           for c, s in world["regions"]])
    exp2 = np.concatenate([np.arcsinh(h[("pr2_signal", c)][s:s + RB].astype(np.float64))
                           for c, s in world["regions"]])
    d = val_out / AB.HALF / REGIME
    np.testing.assert_allclose(np.load(d / f"{t}.signal.npy"), exp1.astype(np.float32))
    np.testing.assert_allclose(np.load(d / f"{t}.truth.npy"), exp2.astype(np.float32))
    assert json.loads((d / "tracks.json").read_text())["kind"] == "replicate_floor"


def test_oracle_picks_planted_copy_and_never_own_cell(world, val_out):
    d = val_out / AB.ORACLE / REGIME
    picks = list(csv.DictReader(open(d / "picks.tsv"), delimiter="\t"))
    assert picks and all(p["picked"].split(";")[0] == world["copy_mid"] for p in picks)
    assert all(OWN_CELL_COPY not in p["picked"] for p in picks)
    assert all(len(p["picked"].split(";")) == 5 for p in picks)
    t = world["val_target"]
    own = {r["cell"] for r in AB.candidate_rows(world["rows"], world["by_id"][t], set())}
    assert "s0" not in own
    oracle = np.load(d / f"{t}.signal.npy")
    mean = np.load(val_out / AB.MEAN / REGIME / f"{t}.signal.npy")
    truth = world["copy_sig"]
    assert np.corrcoef(oracle, truth)[0, 1] >= np.corrcoef(mean, truth)[0, 1]


def test_truth_from_must_match_evalset(world, tmp_path):
    with pytest.raises(AB.Refused):
        AB.run(world["tdir"], world["ev_val"], world["tf_blind"], tmp_path / "o", split="val",
               log=lambda *_: None)


def test_blind_without_flag_raises(world, tmp_path):
    with pytest.raises(PermissionError):
        AB.run(world["tdir"], world["ev_blind"], world["tf_blind"], tmp_path / "o", split="blind",
               log=lambda *_: None)
    assert not (world["ev_blind"].parent / "BLIND_PASSES.tsv").exists()


def test_blind_run_writes_two_ledger_rows(world, tmp_path):
    out = tmp_path / "blind_out"
    AB.run(world["tdir"], world["ev_blind"], world["tf_blind"], out, split="blind",
           allow_blind=True, log=lambda *_: None)
    ledger = world["ev_blind"].parent / "BLIND_PASSES.tsv"
    key = f"oracle_baselines:{out.resolve()}"
    rows = [r for r in csv.DictReader(open(ledger), delimiter="\t") if r["run"] == key]
    assert [r["status"] for r in rows] == ["started", "scored"]
    assert "ag2:" not in key
    with pytest.raises(AB.Refused):
        AB.run(world["tdir"], world["ev_blind"], world["tf_blind"], out, split="blind",
               allow_blind=True, log=lambda *_: None)
    rows = [r for r in csv.DictReader(open(ledger), delimiter="\t") if r["run"] == key]
    assert len(rows) == 2
    for t in json.loads((out / AB.HALF / REGIME / "tracks.json").read_text())["targets"]:
        assert (out / AB.HALF / REGIME / f"{t}.truth.npy").is_file()
    shutil.rmtree(out)
