"""C6 (plan t370, D9): leave-control-out evaluation of the predicted ChIP control.

On the synthetic tables of `minicandi.synth`: the control never reaches the inputs (changing the
control counts does not change the prediction), the output keys exist, `summary` gives mean and
range, and the slurm script's dry run prints the command.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import control_eval as CE  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K27me3"), (3, "DNase-seq"))
CFG = {"context_bins": 128, "batch_size": 4, "control_holdout_cells": ["s1"]}
CONTROL = "chipseq-control"


class Reader(MiniCandi):
    """mu = (1 + mean visible input count) x the target depth from C', on every channel."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, b):
        x = (b["x_counts"].float() * b["x_avail"][:, None, :].float()).mean(-1, keepdim=True)
        depth = (2.0 ** b["c_out"][..., 0] / 2 ** 24)[:, None, :]
        mu = (self.w * (1.0 + x) * depth).clamp_min(1e-3)
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


def _tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def _write_tsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctleval")
    t = make_synthetic_tables(root, n_cells=8, val_tracks=VAL, blind_tracks=())
    rows = _tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    for cell in ("s0", "s1", "s2"):          # the ChIP-seq targets name their cell's control
        for a in ("H3K4me3", "H3K27me3"):
            if (cell, a) in base and base[(cell, a)]["in_eic"] == "val":
                base[(cell, a)]["control_measurement_id"] = base[(cell, CONTROL)]["measurement_id"]
    _write_tsv(t / "measurements.tsv", rows)
    out = root / "eval" / "synth" / "evalset.json"
    E.build(t, out, n_regions=3, n_fit_regions=1, region_bins=256, seed=0)
    return {"root": root, "tables": t, "evalset": out, "base": base}


def _scorer(env, regions=2):
    return CE.build_scorer(env["evalset"], model=Reader(), config=CFG,
                           run_tables_sha256=tables_sha256(env["tables"]), regions=regions,
                           log=lambda *a: None)


@pytest.fixture(scope="module")
def result(env):
    out = env["root"] / "out"
    res = CE.evaluate(env["evalset"], out, model=Reader(), config=CFG,
                      run_tables_sha256=tables_sha256(env["tables"]), regions=2,
                      log=lambda *a: None)
    return res, out


def test_plan_removes_the_control_from_the_inputs(env):
    sc = _scorer(env)
    plan = CE.cell_plan(sc)
    assert set(plan) == {"s0", "s1", "s2"}
    for cell, e in plan.items():
        assert e["control"] == env["base"][(cell, CONTROL)]["measurement_id"]
        assert e["inputs"] and all(sc.ds.rows[m]["assay"] != CONTROL for m in e["inputs"])
        assert all(sc.ds.rows[m]["cell"] == cell for m in e["inputs"] + e["targets"])


def test_control_counts_never_reach_the_prediction(env):
    sc = _scorer(env)
    entry = CE.cell_plan(sc)["s0"]
    a = CE.predict_cell(sc, entry)
    ctl = entry["control"]
    real = sc._ds_read

    def fake(mid, chrom, start, L=None, which="counts"):
        x = real(mid, chrom, start, L, which)
        return (x * 7 + 50).astype(x.dtype) if mid == ctl else x
    sc._ds_read = fake
    sc._region, sc._rc = None, {}
    b = CE.predict_cell(sc, entry)
    assert set(a) == set(b) and len(a) == 2 + len(entry["targets"])
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    # the same perturbation of an input does move the prediction (the model reads its inputs)
    inp = entry["inputs"][0]

    def fake_in(mid, chrom, start, L=None, which="counts"):
        x = real(mid, chrom, start, L, which)
        return (x * 7 + 50).astype(x.dtype) if mid == inp else x
    sc._ds_read = fake_in
    sc._region, sc._rc = None, {}
    c = CE.predict_cell(sc, entry)
    assert not np.array_equal(a["ctl.mu"], c["ctl.mu"])


def test_guard_refuses_a_control_input(env):
    sc = _scorer(env)
    entry = dict(CE.cell_plan(sc)["s0"])
    entry["inputs"] = entry["inputs"] + [entry["control"]]
    with pytest.raises(AssertionError, match="control reached the inputs"):
        CE.predict_cell(sc, entry)


def test_output_keys(result, env):
    res, out = result
    assert (out / CE.DONE).is_file()
    on_disk = json.loads((out / CE.OUT_JSON).read_text())
    assert on_disk["n_cells"] == res["n_cells"] == 3
    for k in ("n_cells", "n_control_targets", "n_chip_targets", "regions_n", "tables_sha256",
              "evalset_sha256"):
        assert k in on_disk
    assert on_disk["regions_n"] == 2
    assert on_disk["tables_sha256"] == tables_sha256(env["tables"])
    assert on_disk["per_cell"]["s1"]["in_sample"] is False
    assert on_disk["per_cell"]["s0"]["in_sample"] is True
    assert on_disk["held_out"]["n_cells"] == 1 and on_disk["in_sample"]["n_cells"] == 2
    for g in CE.GROUPS:
        assert set(CE.COUNT_KEYS) <= set(on_disk[g]["control_count"])
        assert set(CE.SIGNAL_KEYS) <= set(on_disk[g]["signal"])
    blk = on_disk["all"]
    for k in CE.COUNT_KEYS + CE.SIGNAL_KEYS:
        v = blk["control_count"].get(k, blk["signal"].get(k))
        assert v is not None and np.isfinite(v), k
    # obs vs obs would be 1; pred and avg are real comparisons in [-1, 1]
    for k in CE.SIGNAL_KEYS:
        assert -1.0 <= blk["signal"][k] <= 1.0


def test_refuses_a_scorer_directory(env, tmp_path):
    (tmp_path / "SCORED").write_text("x")
    from onion_eval.score import ScoreRefused
    with pytest.raises(ScoreRefused):
        CE.evaluate(env["evalset"], tmp_path, model=Reader(), config=CFG,
                    run_tables_sha256=tables_sha256(env["tables"]), regions=1, log=lambda *a: None)


def test_fold_change_of_the_observed_control_is_one_at_equal_tracks(env):
    """A control equal to the treatment at the same depth and read length gives fold change near 1
    (log2 near 0) wherever the control lambda dominates lambda_bg."""
    from onion_eval import signal_from_counts as SIG
    row = {"frag_len": "200", "depth_reads": "1e8", "read_length": "36"}
    y = np.full(512, 40.0)
    fc = CE.log2_fold_change(y, y, row, row, [("chr1", 0), ("chr1", 256)], 256, SIG)
    assert np.allclose(fc[20:236], 0.0, atol=1e-9)


def test_summary_mean_and_range(tmp_path):
    def blob(v):
        g = {"control_count": {k: v for k in CE.COUNT_KEYS},
             "signal": {k: v / 2 for k in CE.SIGNAL_KEYS}}
        return {"n_cells": 3, "n_control_targets": 3, "n_chip_targets": 4, "regions_n": 64,
                **{x: g for x in CE.GROUPS}}
    for arm, vals in (("armA", (0.1, 0.3, 0.2)), ("armB", (0.5,)), ("zzz", (9.0,))):
        for s, v in enumerate(vals):
            d = tmp_path / "runs" / arm / f"seed{s}"
            d.mkdir(parents=True)
            (d / CE.OUT_JSON).write_text(json.dumps(blob(v)))
    rows = CE.summary(tmp_path / "runs", prefix="arm")
    assert [r["arm"] for r in rows] == ["armA", "armB"]
    a = rows[0]
    assert a["n_seeds"] == 3
    k = "held_out.control_count.crps_skill_vs_training_average"
    assert a[f"{k}_mean"] == pytest.approx(0.2) and a[f"{k}_range"] == pytest.approx(0.2)
    k = "all.signal.pval_arcsinh_pearson_pred_vs_obs"
    assert a[f"{k}_mean"] == pytest.approx(0.1) and a[f"{k}_range"] == pytest.approx(0.1)
    assert rows[1]["n_seeds"] == 1 and rows[1][f"{k}_range"] == 0.0
    out = tmp_path / "s.tsv"
    assert CE.main(["summary", "--runs", str(tmp_path / "runs"), "--prefix", "arm",
                    "--out", str(out)]) == 0
    t = _tsv(out)
    assert len(t) == 2 and t[0]["n_seeds"] == "3" and "regions_n_mean" in t[0]


def test_slurm_dry_run_and_skip(tmp_path):
    sh = REPO / "slurm" / "onion_eval" / "control_eval.sh"
    out = tmp_path / "o"
    envv = dict(os.environ, DRY_RUN="1", ONION_CTL_CELLS="c1,c2")
    envv.pop("SLURM_TMPDIR", None)
    p = subprocess.run(["bash", str(sh), "/kit", "/run", "/ev.json", str(out)], env=envv,
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert ("python3 /kit/tools/onion_eval/control_eval.py run --run /run --evalset /ev.json "
            f"--out {out} --regions 64 --device cpu --infer-batch 8 --cells c1,c2") in p.stdout
    head = sh.read_text()
    for opt in ("--gres=none", "--cpus-per-task=8", "--mem=32000M", "--time=03:00:00"):
        assert f"#SBATCH {opt}" in head
    out.mkdir()
    (out / "CONTROL_EVAL_DONE").write_text("x")
    p = subprocess.run(["bash", str(sh), "/kit", "/run", "/ev.json", str(out)], env=envv,
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and "skip" in p.stdout
