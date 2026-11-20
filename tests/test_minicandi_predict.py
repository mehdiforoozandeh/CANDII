"""PG: genome-wide prediction and EIC export (tools/minicandi/predict.py) on synthetic tables.

Covers the tiling (centre crops against single-window predictions), the challenge .npy format
through eic_score.load_prediction, the lambda sources, that the target's own counts are never
read, and the blind discipline: --allow-blind, the freeze file, the ledger key
`predict:<run dir>` beside the scorer's key, started / predicted rows, resume and repeat."""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import predict as P  # noqa: E402
from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.dataset import OnionDataset, tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import eic_score as EIC  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import score as S  # noqa: E402

CFG = {"context_bins": 128, "batch_size": 4}
STRIDE = 64
QUIET = dict(log=lambda *a: None)


class Local(MiniCandi):
    """mu at a bin depends only on that bin's inputs and on C' (translation-invariant)."""

    def forward(self, b):
        s = b["x_counts"].double().sum(-1, keepdim=True)
        depth = 2.0 ** b["c_out"][..., 0].double() / 2 ** 24
        mu = (0.5 + 0.05 * s) * (1.0 + depth[:, None, :])
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


class Positional(Local):
    """Local plus a ramp over the position inside the window: crops must be mapped exactly."""

    def forward(self, b):
        o = super().forward(b)
        L = o["mu"].shape[1]
        ramp = 1.0 + torch.arange(L, dtype=torch.float64) / L
        return {"n": o["n"], "mu": o["mu"] * ramp[None, :, None]}


def _tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("predict")
    t = make_synthetic_tables(root)          # val: s0 H3K4me3, s1 H3K27ac, s2 DNase; blind: s3, s4
    # one ChIP-seq track with an observed control (s1 H3K27ac); the others have none
    rows = _tsv(t / "measurements.tsv")
    ctl = next(r["measurement_id"] for r in rows if r["cell"] == "s1"
               and r["assay"] == C.CONTROL and not r["arm_knob"])
    for r in rows:
        if r["cell"] == "s1" and r["assay"] == "H3K27ac":
            r["control_measurement_id"] = ctl
    with open(t / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    val = root / "eval" / "synth" / "evalset.json"
    E.build(t, val, n_regions=1, n_fit_regions=1, region_bins=256)
    blind = root / "eval" / "synth_blind" / "evalset_blind.json"
    ev = E.build(t, blind, n_regions=1, n_fit_regions=1, region_bins=256, include_blind=True)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["split"] == "blind" and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"], ev["control_targets"] = [], [], []
    blind.write_text(json.dumps(ev))
    return {"root": root, "tables": t, "val": val, "blind": blind}


def _run_dir(tmp_path, name="rung1_x/seed0", tables=None):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    if tables is not None:
        torch.save({"tables_sha256": tables_sha256(tables)}, run / "ckpt.pt")
    return run


def _freeze(tmp_path, *runs):
    p = tmp_path / "FREEZE.md"
    p.write_text("# Freeze\n\nRuns that may read blind data:\n"
                 + "".join(f"- {r}\n" for r in runs), encoding="utf-8")
    return p


@pytest.fixture()
def fake_registry(monkeypatch):
    fake = types.ModuleType("minicandi.registry")
    fake.load_run = lambda d, map_location="cpu": (Local(), dict(CFG))
    monkeypatch.setitem(sys.modules, "minicandi.registry", fake)
    return fake


@pytest.fixture(scope="module")
def val_pred(env, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("valpred")
    out = tmp / "pred"
    p = P.Predictor(tmp / "run", env["tables"], env["val"], out=out, model=Local(), config=CFG,
                    stride_bins=STRIDE, **QUIET)
    reads = []
    orig = p.ds.read

    def rec(mid, chrom, start, L=None, which="counts"):
        reads.append(mid)
        return orig(mid, chrom, start, L, which)
    p.ds.read = rec
    res = p.run_all()
    return {"p": p, "out": out, "res": res, "reads": reads}


# ------------------------------------------------------------------ tiling

def test_tiles_cover_every_bin_once():
    for n, L, s in ((2048, 128, 64), (1000, 128, 64), (1024, 128, 128), (300, 128, 32),
                    (128, 128, 64)):
        cover = np.zeros(n, int)
        for w, lo, hi in P.tiles(n, L, s):
            assert w <= lo < hi <= w + L
            cover[lo:hi] += 1
        assert (cover == 1).all(), (n, L, s)
    with pytest.raises(ValueError):
        P.tiles(1024, 128, 63)                # L - stride odd
    with pytest.raises(ValueError):
        P.tiles(1024, 128, 256)               # stride > L


def test_centre_crop_equals_single_window_prediction(env, tmp_path):
    """Each tile's kept bins equal the same bins of that window's own full prediction, for a model
    that depends on the position inside the window."""
    p = P.Predictor(tmp_path / "run", env["tables"], env["val"], out=tmp_path / "o",
                    model=Positional(), config=CFG, stride_bins=STRIDE, **QUIET)
    job = p.jobs[0]
    t = C.TRACK_INDEX[job["assay"]]
    n, mu, _ = p.predict_chrom(job, "chr2")
    checked = 0
    for w, lo, hi in P.tiles(p.ds.n_bins["chr2"], p.L, STRIDE):
        with p._hidden_target(job):
            b = C.collate([p.ds.window("chr2", w, job["inputs"], [job["target"]])])
        full = p._forward(b)["mu"][0, :, t]
        np.testing.assert_array_equal(mu[lo:hi], full[lo - w:hi - w])
        checked += 1
    m = (p.L - STRIDE) // 2
    interior = P.tiles(p.ds.n_bins["chr2"], p.L, STRIDE)[3]
    assert interior[1] - interior[0] == m and interior[2] - interior[1] == STRIDE
    assert checked > 3


def test_translation_invariant_model_matches_one_chromosome_window(env, tmp_path):
    """A model local in the bins: the tiled chromosome equals one window over the whole chromosome."""
    p = P.Predictor(tmp_path / "run", env["tables"], env["val"], out=tmp_path / "o",
                    model=Local(), config=CFG, stride_bins=STRIDE, **QUIET)
    job = p.jobs[0]
    t = C.TRACK_INDEX[job["assay"]]
    _, mu, _ = p.predict_chrom(job, "chr2")
    nb = p.ds.n_bins["chr2"]
    big = OnionDataset(env["tables"], "val", dict(CFG, context_bins=nb), seed=0)
    p2 = P.Predictor(tmp_path / "run", env["tables"], env["val"], out=tmp_path / "o2",
                     model=Local(), config=dict(CFG, context_bins=nb), stride_bins=nb, **QUIET)
    assert p2.ds.L == big.L == nb
    _, mu_one, _ = p2.predict_chrom(job, "chr2")
    with p2._hidden_target(job):
        b = C.collate([p2.ds.window("chr2", 0, job["inputs"], [job["target"]])])
    np.testing.assert_allclose(mu_one, p2._forward(b)["mu"][0, :, t], rtol=0, atol=0)
    np.testing.assert_allclose(mu, mu_one, rtol=1e-12)


# ------------------------------------------------------------------ outputs

def test_outputs_load_through_eic_score(val_pred, env):
    out, p = val_pred["out"], val_pred["p"]
    assert val_pred["res"]["done"] and (out / "DONE").is_file()
    sizes = json.loads((env["root"] / "genome" / "chrom_sizes.json").read_text())
    assert p.full == ["chr1", "chr2"]
    for job in p.jobs:
        for d in ("signal", "signal_plug_in"):
            pred = EIC.load_prediction(out / d / f"{job['track']}.npy", p.full, {})
            for c in p.full:
                assert pred[c].dtype == np.float32
                assert pred[c].shape == (P.raw_grid_bins(sizes[c]),)
                assert np.isfinite(pred[c]).all() and (pred[c] >= 0).all()
        with h5py.File(out / "counts" / f"{job['track']}.h5", "r") as f:
            for c in p.full:
                assert f["mu"][c].dtype == np.float16 and f["n"][c].dtype == np.float16
                assert f["mu"][c].shape == (p.ds.n_bins[c],)
        assert not (out / "parts" / job["track"]).exists()
    # the stored counts are the float16 of the model's mean
    job = p.jobs[0]
    _, mu, _ = p.predict_chrom(job, "chr2")
    with h5py.File(out / "counts" / f"{job['track']}.h5", "r") as f:
        np.testing.assert_array_equal(f["mu"]["chr2"][:], mu.astype(np.float16))
    # expectation and plug-in differ (the Jensen gap), and the expectation is signal_from_nb's
    e = np.load(out / "signal" / f"{job['track']}.npy", allow_pickle=True)[()]
    q = np.load(out / "signal_plug_in" / f"{job['track']}.npy", allow_pickle=True)[()]
    assert not np.array_equal(e["chr1"], q["chr1"])


def test_lambda_sources(val_pred):
    rows = {r["cell"]: r for r in _tsv(val_pred["out"] / "lambda_source.tsv")}
    assert rows["s0"]["lambda_source"] == "predicted_control"      # H3K4me3, no control
    assert rows["s1"]["lambda_source"] == "observed_control"       # H3K27ac, control set
    assert rows["s2"]["lambda_source"] == "local_background"       # DNase-seq, MACS2 route
    assert {r["eic_track"] for r in rows.values()} == {j["track"] for j in val_pred["p"].jobs}


def test_target_counts_are_never_read(val_pred):
    p = val_pred["p"]
    assert val_pred["reads"]
    for job in p.jobs:
        row = p.ds.rows[job["target"]]
        track = {m for m, r in p.ds.rows.items()
                 if r["cell"] == row["cell"] and r["assay"] == row["assay"]}
        assert not track & set(val_pred["reads"]), job["target"]


def test_rerun_skips_finished(val_pred, env):
    out = val_pred["out"]
    assert P.main(["--run", str(out.parent / "run"), "--tables", str(env["tables"]),
                   "--evalset", str(env["val"]), "--split", "val", "--out", str(out)]) == 0
    p = P.Predictor(out.parent / "run", env["tables"], env["val"], out=out, model=Local(),
                    config=CFG, stride_bins=STRIDE, **QUIET)
    p.do_chrom = lambda *a: (_ for _ in ()).throw(AssertionError("recomputed a finished target"))
    assert p.run_all()["done"]


def test_predicted_control_needs_predict_control(env, tmp_path):
    p = P.Predictor(tmp_path / "run", env["tables"], env["val"], out=tmp_path / "o",
                    model=Local(), config=dict(CFG, predict_control=False), stride_bins=STRIDE,
                    **QUIET)
    job = next(j for j in p.jobs if j["cell"] == "s0")
    with pytest.raises(P.PredictRefused, match="predicted control"):
        p.predict_chrom(job, "chr2")


# ------------------------------------------------------------------ blind discipline

def _blind_args(env, run, out, freeze=None):
    a = ["--run", str(run), "--tables", str(env["tables"]), "--evalset", str(env["blind"]),
         "--split", "blind", "--allow-blind", "--out", str(out), "--stride-bins", str(STRIDE)]
    return a + (["--freeze", str(freeze)] if freeze else [])


def _ledger(env):
    return S.ledger_rows(env["blind"].parent / S.BLIND_LEDGER)


def test_blind_without_flag_raises(env, tmp_path, fake_registry):
    with pytest.raises(PermissionError, match="allow-blind"):
        P.Predictor(tmp_path / "run", env["tables"], env["blind"], out=tmp_path / "o",
                    split="blind", model=Local(), config=CFG, **QUIET)
    run = _run_dir(tmp_path, tables=env["tables"])
    a = _blind_args(env, run, tmp_path / "o", _freeze(tmp_path, run))
    a.remove("--allow-blind")
    n0 = len(_ledger(env))
    assert P.main(a) == 2
    assert len(_ledger(env)) == n0 and not (tmp_path / "o").exists()


def test_blind_without_freeze_file_refused(env, tmp_path, fake_registry, capsys):
    run = _run_dir(tmp_path, tables=env["tables"])
    out = tmp_path / "o"
    n0 = len(_ledger(env))
    assert P.main(_blind_args(env, run, out)) == 2                       # no --freeze
    assert "--freeze" in capsys.readouterr().err
    assert P.main(_blind_args(env, run, out, tmp_path / "nope.md")) == 2  # missing file
    assert "does not exist" in capsys.readouterr().err
    empty = tmp_path / "empty.md"
    empty.write_text("\n  \n")
    assert P.main(_blind_args(env, run, out, empty)) == 2
    assert "empty" in capsys.readouterr().err
    other = _freeze(tmp_path, tmp_path / "runs" / "rung9_other" / "seed0")
    assert P.main(_blind_args(env, run, out, other)) == 2                # run not on the list
    assert "not named" in capsys.readouterr().err
    assert len(_ledger(env)) == n0 and not out.exists()
    # a validation eval set is refused on --split blind
    a = _blind_args(env, run, out, _freeze(tmp_path, run))
    a[a.index("--evalset") + 1] = str(env["val"])
    assert P.main(a) == 2 and len(_ledger(env)) == n0


def test_blind_ledger_rows_resume_and_repeat(env, tmp_path, fake_registry, capsys):
    run = _run_dir(tmp_path, tables=env["tables"])
    out = tmp_path / "pred"
    fz = _freeze(tmp_path, run)
    key = P.ledger_key(run)
    assert key == "predict:" + str(run.resolve())
    mine = lambda: [r for r in _ledger(env) if r["run"] == key]  # noqa: E731
    # the cost probe: one chromosome writes `started` only
    assert P.main(_blind_args(env, run, out, fz) + ["--chroms", "chr1"]) == 0
    assert [r["status"] for r in mine()] == ["started"] and not (out / "DONE").exists()
    # an array over cells resumes the open pass without a new row; the last one closes it
    assert P.main(_blind_args(env, run, out, fz) + ["--cell-index", "0"]) == 0
    assert [r["status"] for r in mine()] == ["started"]
    assert P.main(_blind_args(env, run, out, fz) + ["--cell-index", "7"]) == 0   # past the end
    assert P.main(_blind_args(env, run, out, fz) + ["--cell-index", "1"]) == 0
    rows = mine()
    assert [r["status"] for r in rows] == ["started", "predicted"]
    assert tuple(_ledger(env)[0]) == S.LEDGER_COLUMNS
    assert rows[0]["out"] == str(out.resolve()) and rows[0]["git_sha"]
    assert rows[0]["tables_sha256"] == tables_sha256(env["tables"])
    assert rows[0]["evalset_sha256"] == S.sha256_file(env["blind"])
    assert (out / "DONE").is_file() and len(list((out / "signal").glob("*.npy"))) == 2
    # DONE: skip before anything is read, no row
    assert P.main(_blind_args(env, run, out, fz)) == 0 and len(mine()) == 2
    # a second prediction of the same key and eval set is refused without a reason
    out2 = tmp_path / "pred2"
    capsys.readouterr()
    assert P.main(_blind_args(env, run, out2, fz)) == 2
    assert "--repeat-blind-reason" in capsys.readouterr().err and len(mine()) == 2
    assert not out2.exists()
    assert P.main(_blind_args(env, run, out2, fz)
                  + ["--repeat-blind-reason", "node died mid-write"]) == 0
    rows = mine()
    assert [r["status"] for r in rows] == ["started", "predicted", "started", "predicted"]
    assert rows[2]["repeat_reason"] == rows[3]["repeat_reason"] == "node died mid-write"


def test_score_row_and_predict_row_do_not_block_each_other(env, tmp_path, fake_registry):
    """One run: the scorer's blind pass (key = run dir) and the prediction (key = predict:<run
    dir>) each get their own started row, in either order, on one eval set and ledger."""
    led = env["blind"].parent / S.BLIND_LEDGER
    kw = dict(split="blind", allow_blind=True, n_boot=0, guardrail_windows=4,
              log=lambda *a: None)
    # scorer first, then predictor
    run_a = _run_dir(tmp_path, "rung1_a/seed0", tables=env["tables"])
    S.score(env["blind"], run_a / "blind", run=run_a, **kw)
    fz = _freeze(tmp_path, run_a, tmp_path / "runs" / "rung1_b" / "seed0")
    assert P.main(_blind_args(env, run_a, tmp_path / "pa", fz)) == 0
    # predictor first, then scorer
    run_b = _run_dir(tmp_path, "rung1_b/seed0", tables=env["tables"])
    assert P.main(_blind_args(env, run_b, tmp_path / "pb", fz)) == 0
    S.score(env["blind"], run_b / "blind", run=run_b, **kw)
    rows = S.ledger_rows(led)
    for run in (run_a, run_b):
        got = {(r["run"], r["status"]) for r in rows if r["run"].endswith(str(run.resolve()))}
        assert got == {(str(run.resolve()), "started"), (str(run.resolve()), "scored"),
                       ("predict:" + str(run.resolve()), "started"),
                       ("predict:" + str(run.resolve()), "predicted")}
    # and each still refuses its own repeat
    with pytest.raises(S.ScoreRefused, match="repeat"):
        S.score(env["blind"], tmp_path / "again" / "blind", run=run_a, **kw)
    assert P.main(_blind_args(env, run_a, tmp_path / "pa2", fz)) == 2
