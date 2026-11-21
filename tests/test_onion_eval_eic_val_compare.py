"""C16 (plan t370, D10): tools/onion_eval/eic_val_compare.py, slurm/onion_eval/eic_val.sh and
slurm/onion_eval/avocado_train_only.sh, off-cluster (tiny folders, DRY_RUN)."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onion_eval import eic_score as ES  # noqa: E402
from onion_eval import eic_val_compare as V  # noqa: E402
from test_onion_eval_eic_score import _section, _write_bigwig  # noqa: E402

EIC_VAL = REPO / "slurm" / "onion_eval" / "eic_val.sh"
AVO = REPO / "slurm" / "onion_eval" / "avocado_train_only.sh"


def test_measures_restated_equal_the_scorer():
    assert V.MEASURES9 == ES.MEASURES9
    assert V.HIGHER_IS_BETTER == ES.HIGHER_IS_BETTER


# ---- table ---------------------------------------------------------------------------------

def _score_file(path, entry, per_exp):
    """per_exp: {exp: {measure: value}} (missing measures = 1.0); adds a boot0 row to ignore."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ES.ROW_KEYS)
        for exp, vals in per_exp.items():
            for cs in ("all23", "boot0"):
                v = [999.0 if cs == "boot0" else vals.get(m, 1.0) for m in ES.MEASURES9]
                w.writerow([entry, "", exp, exp[:3], exp[3:], cs] + [repr(x) for x in v])
    return path


def test_table_means_on_common_tracks_and_pairwise_wins(tmp_path):
    a = _score_file(tmp_path / "a.tsv", "mini", {
        "C01M02": {"mse": 1.0, "gwcorr": 0.5}, "C02M02": {"mse": 3.0, "gwcorr": 0.7},
        "C03M02": {"mse": 100.0, "gwcorr": float("nan")}})          # C03M02: not common
    b = _score_file(tmp_path / "b.tsv", "Average", {
        "C01M02": {"mse": 2.0, "gwcorr": 0.6}, "C02M02": {"mse": 3.0, "gwcorr": float("nan")}})
    out = tmp_path / "t.tsv"
    rows = V.write_table([a, b], out, log=lambda m: None)
    got = {(r["entry"], r["measure"]): r for r in rows}
    m = got[("mini", "mse")]
    assert m["n_tracks"] == 2 and m["n_tracks_entry"] == 3
    assert m["mean"] == pytest.approx(2.0)                         # (1 + 3) / 2, C03 left out
    assert m["beats:Average"] == 1 and m["beats:mini"] == ""        # tie on C02 is no win
    assert got[("Average", "mse")]["beats:mini"] == 0
    g = got[("mini", "gwcorr")]                                      # higher is better
    assert g["mean"] == pytest.approx(0.6) and g["beats:Average"] == 0   # 0.5 < 0.6; NaN no win
    ga = got[("Average", "gwcorr")]
    assert ga["n_nan"] == 1 and ga["mean"] == pytest.approx(0.6) and ga["beats:mini"] == 1
    with open(out) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    assert hdr[:7] == ["entry", "measure", "higher_is_better", "n_tracks", "n_nan", "mean",
                       "n_tracks_entry"]
    assert hdr[7:] == ["beats:mini", "beats:Average"]
    assert len(rows) == 2 * 9


def test_table_refuses_an_entry_in_two_files(tmp_path):
    a = _score_file(tmp_path / "a.tsv", "mini", {"C01M02": {}})
    b = _score_file(tmp_path / "b.tsv", "mini", {"C02M02": {}})
    with pytest.raises(SystemExit):
        V.read_scores([a, b])


def test_table_cli(tmp_path):
    a = _score_file(tmp_path / "a.tsv", "x", {"C01M02": {}})
    b = _score_file(tmp_path / "b.tsv", "y", {"C01M02": {"mse": 0.5}})
    assert V.main(["table", "--scores", str(a), str(b), "--out", str(tmp_path / "t.tsv")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "t.tsv"), delimiter="\t"))
    r = [r for r in rows if r["entry"] == "y" and r["measure"] == "mse"][0]
    assert float(r["mean"]) == 0.5 and r["beats:x"] == "1" and r["beats:y"] == ""


# ---- average -------------------------------------------------------------------------------

def _npy(path, d):
    np.save(path, {c: np.asarray(v, np.float32) for c, v in d.items()}, allow_pickle=True)


def test_average_is_the_per_assay_mean_of_training_tracks_only(tmp_path):
    tr, tv, out = tmp_path / "training_data", tmp_path / "validation_data", tmp_path / "avg"
    tr.mkdir(); tv.mkdir()
    _npy(tr / "C01M02.npy", {"chr1": [1, 2, 3], "chr2": [4]})
    _npy(tr / "C02M02.npy", {"chr1": [3, 4, 5], "chr2": [6]})
    _npy(tr / "C01M16.npy", {"chr1": [10, 10, 10], "chr2": [0]})
    _npy(tr / "C05M22.npy", {"chr1": [7, 7, 7], "chr2": [7]})      # no truth M22: not built
    for e in ("C03M02", "C04M02", "C03M16", "C09M99"):            # M99: no training track
        _npy(tv / f"{e}.npy", {"chr1": [0, 0, 0], "chr2": [0]})
    rec = V.build_average(tr, out, chroms=["chr1", "chr2"], log=lambda m: None)  # default truth
    assert rec["truth"] == str(tv)
    assert rec["members"] == {"M02": ["C01M02", "C02M02"], "M16": ["C01M16"]}
    assert rec["no_training_track"] == ["M99"]
    assert rec["tracks"] == ["C03M02", "C03M16", "C04M02"]
    assert sorted(p.name for p in (out / "assay").iterdir()) == ["M02.npy", "M16.npy"]
    for e in rec["tracks"]:
        assert (out / f"{e}.npy").is_symlink()
        assert os.readlink(out / f"{e}.npy") == os.path.join("assay", f"{e[3:]}.npy")
    d = np.load(out / "C04M02.npy", allow_pickle=True)[()]
    assert d["chr1"].tolist() == [2, 3, 4] and d["chr2"].tolist() == [5]
    assert d["chr1"].dtype == np.float32
    # the scorer reads it as a challenge .npy
    got = ES.load_prediction(out / "C03M16.npy", ["chr1", "chr2"], {})
    assert got["chr1"].tolist() == [10, 10, 10]
    assert json.loads((out / "AVERAGE.json").read_text())["no_training_track"] == ["M99"]
    # a rerun keeps the built files and relinks
    rec2 = V.build_average(tr, out, chroms=["chr1", "chr2"], log=lambda m: None)
    assert rec2["tracks"] == rec["tracks"]


def test_average_refuses_truth_in_the_training_folder(tmp_path):
    tr, tv = tmp_path / "training_data", tmp_path / "validation_data"
    tr.mkdir(); tv.mkdir()
    _npy(tr / "C01M02.npy", {"chr1": [1]})
    _npy(tv / "C01M02.npy", {"chr1": [1]})
    with pytest.raises(SystemExit):
        V.build_average(tr, tmp_path / "o", chroms=["chr1"], log=lambda m: None)


def test_average_of_bigwigs_bins_like_the_scorer_and_pads_short_tracks(tmp_path):
    tr, tv = tmp_path / "training_data", tmp_path / "validation_data"
    tr.mkdir(); tv.mkdir()
    _write_bigwig(tr / "C01M02.bigwig", [("chr1", 50)],
                  [(0, _section(0, 0, 50, 0, 0, 1, [(0, 50, 2.0)]))])
    _write_bigwig(tr / "C02M02.bigwig", [("chr1", 25)],
                  [(0, _section(0, 0, 25, 0, 0, 1, [(0, 25, 4.0)]))])
    _npy(tv / "C03M02.npy", {"chr1": [0, 0]})
    V.build_average(tr, tmp_path / "o", truth=tv, chroms=["chr1"], log=lambda m: None)
    d = np.load(tmp_path / "o" / "C03M02.npy", allow_pickle=True)[()]
    assert d["chr1"].tolist() == pytest.approx([3.0, 1.0])            # (2+4)/2, (2+0)/2


# ---- score, val-cells ----------------------------------------------------------------------

def test_score_calls_the_eic_scorer_with_the_entry_name(tmp_path, monkeypatch, capsys):
    pred, truth = tmp_path / "out" / "signal", tmp_path / "truth"
    pred.mkdir(parents=True); truth.mkdir()
    (tmp_path / "out" / "signal_plug_in").mkdir()
    for e in ("C01M02", "C02M16"):
        (truth / f"{e}.bigwig").write_bytes(b"")
    (pred / "C01M02.npy").write_bytes(b"")
    seen = {}
    monkeypatch.setattr(ES, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    rc = V.main(["score", "--pred", str(pred), "--name", "miniCANDI_expectation",
                 "--truth", str(truth), "--cache", str(tmp_path / "c"),
                 "--out", str(tmp_path / "s.tsv")])
    assert rc == 0
    a = seen["argv"]
    assert a[0] == "score" and a[a.index("--entry") + 1] == "miniCANDI_expectation"
    assert a[a.index("--pred") + 1] == str(pred) and a[a.index("--truth") + 1] == str(truth)
    out = capsys.readouterr().out
    assert "1 of 2 truth tracks" in out and "C02M16" in out and "DONE is missing" in out


def _tables(tmp_path):
    t = tmp_path / "ONION" / "tables" / "eic_v1_2"
    t.mkdir(parents=True)
    with open(t / "measurements.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["measurement_id", "cell", "assay"])
        for mid, cell in (("m1", "B"), ("m2", "A"), ("m3", "A"), ("m4", "C"), ("m5", "D")):
            w.writerow([mid, cell, "x"])
    ev = tmp_path / "ONION" / "eval" / "eic_v1_2" / "evalset.json"
    ev.parent.mkdir(parents=True)
    ev.write_text(json.dumps({"imputation_targets": [
        {"target": "m1", "split": "val", "regime": "leave_one_assay_out"},
        {"target": "m2", "split": "val"},
        {"target": "m3", "split": "val", "regime": "leave_one_assay_out"},
        {"target": "m4", "split": "blind", "regime": "leave_one_assay_out"},
        {"target": "m5", "split": "val", "regime": "other"}]}))
    return t, ev


def test_val_cells_counts_validation_leave_one_assay_out_cells(tmp_path, capsys):
    t, ev = _tables(tmp_path)
    assert V.val_cells(t, ev) == ["A", "B"]
    assert V.main(["val-cells", "--tables", str(t), "--evalset", str(ev)]) == 0
    assert capsys.readouterr().out.strip() == "2"


# ---- eic_val.sh ----------------------------------------------------------------------------

def _clean_env(**kw):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SLURM_") and k not in ("EIC_VAL_CHROMS", "ONION_EXCLUDE")}
    env.update(kw)
    return env


def _sh(script, args, env):
    p = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_eic_val_task_passes_cell_index_and_chroms(tmp_path):
    out = _sh(EIC_VAL, [str(REPO), "/r/run", "/o/out"],
              _clean_env(DRY_RUN="1", SLURM_JOB_ID="7", SLURM_ARRAY_TASK_ID="3",
                         EIC_VAL_CHROMS="chr21"))
    cmd = out.strip().splitlines()[-1].split()
    assert cmd[1].endswith("tools/minicandi/predict.py")
    for flag, val in (("--run", "/r/run"), ("--out", "/o/out"), ("--split", "val"),
                      ("--cell-index", "3"), ("--chroms", "chr21"),
                      ("--tables", "/project/def-maxwl/mforooz/ONION/tables/eic_v1_2"),
                      ("--evalset", "/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset.json")):
        assert cmd[cmd.index(flag) + 1] == val


def test_eic_val_task_without_chroms_predicts_every_chromosome(tmp_path):
    out = _sh(EIC_VAL, [str(REPO), "/r", "/o"],
              _clean_env(DRY_RUN="1", SLURM_JOB_ID="7", SLURM_ARRAY_TASK_ID="0"))
    assert "--chroms" not in out and "--cell-index 0" in out


def test_eic_val_launcher_sizes_the_array_by_validation_cells(tmp_path):
    _tables(tmp_path)
    out = _sh(EIC_VAL, [str(REPO), "/r", "/o"],
              _clean_env(DRY_RUN="1", ONION_ROOT=str(tmp_path / "ONION"),
                         ONION_EXCLUDE="fc1, fc2", EIC_VAL_CHROMS="chr21"))
    line = [l for l in out.splitlines() if l.startswith("sbatch ")][0].split()
    assert "--array=0-1%12" in line and "--job-name=rcp_eicval" in line
    assert "--exclude=fc1,fc2" in line and "--export=ALL" in line
    assert line[-3:] == [str(REPO), "/r", "/o"]
    assert "validation cells: 2; EIC_VAL_CHROMS=chr21" in out


# ---- avocado_train_only.sh -----------------------------------------------------------------

def _fake_005(tmp_path):
    e = tmp_path / "005"
    (e / "scripts").mkdir(parents=True)
    (e / "output" / "inputs" / "bridge").mkdir(parents=True)
    for f in ("hpc_env.sh", "hpc_prepare_grid.py", "hpc_train.py", "hpc_predict.py"):
        (e / "scripts" / f).write_text("# stub\n")
    hdr = ("#SBATCH --output=/scratch/maxwl/logs/x_%A_%a.out\n"
           "NEED_TORCH=1 source /scratch/maxwl/epi_imputation/005_ws/scripts/hpc_env.sh\n"
           "WS=/scratch/maxwl/epi_imputation/005_ws\nSC=/scratch/maxwl/epi_imputation/005\n"
           "DATA=/scratch/maxwl/epi_imputation/data\n--blind-list inputs/bridge/blind_experiments.txt\n")
    for f in ("hpc_bin_tracks.slurm", "hpc_train.slurm", "hpc_predict.slurm",
              "hpc_make_bigwig.slurm"):
        (e / "scripts" / f).write_text(hdr)
    (e / "output" / "inputs" / "bridge" / "eic_bridge.csv").write_text(
        "filename,cell_id,assay_id,split,assay_name\n"
        "C01M02,C01,M02,T,DNase-seq\nC02M16,C02,M16,T,H3K27ac\n"
        "C01M16,C01,M16,V,H3K27ac\nC02M02,C02,M02,V,DNase-seq\nC09M02,C09,M02,B,DNase-seq\n")
    d = tmp_path / "DATA"
    (d / "training_data").mkdir(parents=True)
    for x in ("C01M02", "C02M16"):
        (d / "training_data" / f"{x}.bigwig").write_bytes(b"")
    (d / "training_data" / "metadata_training_data.tsv").write_text("x\n")
    return e, d


def _avo(tmp_path, e, d, **kw):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "sbatch").write_text("#!/bin/sh\necho SBATCH_CALLED >&2\nexit 9\n")
    (fake / "sbatch").chmod(0o755)
    env = _clean_env(AVO_EXP=str(e), EIC_DATA=str(d), AVO_ROOT=str(tmp_path / "root"),
                     LOGS=str(tmp_path / "logs"), PATH=f"{fake}:{os.environ['PATH']}", **kw)
    p = subprocess.run(["bash", str(AVO)], env=env, capture_output=True, text=True)
    return p


def test_avocado_dry_run_stages_training_only_and_prints_the_chain(tmp_path):
    e, d = _fake_005(tmp_path)
    p = _avo(tmp_path, e, d)
    assert p.returncode == 0, p.stderr
    assert "SBATCH_CALLED" not in p.stderr                     # DRY_RUN=1 is the default
    root = tmp_path / "root"
    ws = root / "ws"
    assert (ws / "inputs" / "train_only_experiments.txt").read_text().split() == ["C01M02", "C02M16"]
    assert (ws / "inputs" / "bridge" / "val_experiments.txt").read_text().split() == \
        ["C01M16", "C02M02"]
    tracks = root / "data" / "synapse" / "train_plus_val"
    assert sorted(x.name for x in tracks.iterdir()) == ["C01M02.bigwig", "C02M16.bigwig"]
    bridge = list(csv.DictReader(open(ws / "inputs" / "bridge" / "eic_bridge.csv")))
    assert [r["split"] for r in bridge] == ["T", "T", "VAL_HELDOUT", "VAL_HELDOUT", "B"]
    s = (ws / "scripts" / "hpc_predict.slurm").read_text()
    assert "/scratch/maxwl" not in s and "blind_experiments" not in s
    assert f"WS={ws}\n" in s and f"SC={root / 'sc'}\n" in s and f"DATA={root / 'data'}\n" in s
    assert (e / "scripts" / "hpc_predict.slurm").read_text().count("/scratch/maxwl") == 5  # untouched
    lines = [l for l in p.stdout.splitlines() if l.startswith("sbatch ")]
    names = [next(t for t in l.split() if t.startswith("--job-name=")) for l in lines]
    assert names == ["--job-name=rcp_avocado_" + n for n in
                     ("grid", "bin", "train_shared", "train_genome", "predict", "bigwig")]
    shared, genome = lines[2], lines[3]
    assert "MODE=shared" in shared and "EPOCHS=120" in shared and "chrom20.txt" in shared
    assert "MODE=genome" in genome and "EPOCHS=60" in genome and "--gres=gpu:h100:1" in genome
    assert "--dependency=afterok:<J_SHARED>" in genome
    assert "--array=0-1%12" in lines[5]                         # one task per validation track
    assert "2 training tracks" in p.stdout and "2 validation targets" in p.stdout
    assert "Avocado_train_only" in p.stdout


def test_avocado_refuses_a_bridge_that_disagrees_with_the_training_folder(tmp_path):
    e, d = _fake_005(tmp_path)
    (d / "training_data" / "C05M05.bigwig").write_bytes(b"")
    p = _avo(tmp_path, e, d)
    assert p.returncode == 2 and "differ" in p.stderr
