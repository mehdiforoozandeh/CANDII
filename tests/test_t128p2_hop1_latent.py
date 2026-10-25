"""t128 phase 2: hop-1 scoring of the anchored encoder (`tools/t128p2/hop1.py`, metric 6) and the
latent pass on rung DSANC (`tools/t128/latent.py`, metrics 7 and 8).

The DSanchor run is a 4-step `train_ds.run_training_ds` run on `synth.make_products`; the
identity-initialised DSANC run is built by hand in the t126 run-dir shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data, encoding, model, pairs, score, synth  # noqa: E402
from t128 import latent  # noqa: E402
from t128 import train_ds  # noqa: E402
from t128p2 import hop1  # noqa: E402
from t128p2 import pairs as tp  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only",
                                        "ignore:.*slow, meant for synthetic data")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
SMALL = dict(max_steps=4, eval_every=2, device="cpu", window=256, batch=4, knot_sample=20_000,
             knot_block=256)
SPLIT_KEYS = ("crps_oracle_scaled_all", "scale_error_all", "c_star_all")


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128p2hop1")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed,
            "data": root / "products", "rows": pairs.read_manifest(manifest)}


@pytest.fixture(scope="module")
def anc(prod):
    assert tp.SETTINGS[2]["arm"] == "DSanchor"
    return train_ds.run_training_ds(prod["manifest"], prod["cov"], prod["data"],
                                    prod["root"] / "runs", tp.SETTINGS[2], "T1", "real", 0,
                                    holdout=tp.HOLDOUT, **SMALL)


def _args(prod, run_dir):
    return ["run", str(prod["manifest"]), str(prod["cov"]), str(prod["data"]), str(prod["bed"]),
            str(run_dir)]


def _hand_run(prod, name, rung) -> Path:
    """An identity-initialised run dir in the t126 shape (as tests/test_t128_latent.py)."""
    rows, g = prod["rows"], "T1"
    holdout = [list(h) for h in tp.HOLDOUT]
    trp = pairs.train_pairs(rows, g, holdout)
    fit = pairs.fit_pids(rows, g, holdout)
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    torch.manual_seed(0)
    lad = model.Ladder(rung, "counts", STATS)
    run_dir = prod["root"] / "hand" / name
    run_dir.mkdir(parents=True)
    config = {"run_name": name, "rung": rung, "arm": name.split("_")[0], "g": g,
              "space": "counts", "model": "real", "seed": 0, "n_cov": encoding.N_COV,
              "fit_pids": fit, "train_pairs": trp, "holdout": holdout,
              "heldout_pairs": pairs.heldout_pairs(rows, g, holdout),
              "ids_perm": list(range(len(trp)))}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    torch.save({"g": lad.g.state_dict(), "form": lad.form.state_dict(),
                "stats": {"knots_x": [float(v) for v in STATS["knots_x"]],
                          "n0": STATS["n0"], "sigma0": STATS["sigma0"]},
                "encoder": enc.state(), "config": config, "step": 0, "val_nll": 1.0},
               run_dir / "ckpt.pt")
    (run_dir / "train_log.tsv").write_text("step\tloss\tmse_z\tkl\tval_nll\tseconds\n"
                                           "0\t1.0\t0.0\t0.0\t1.0\t0.1\n")
    (run_dir / "TRAIN_DONE").write_text("synthetic\n")
    return run_dir


def _hop1_pred(prod, run_dir, chunk=model.CHUNK):
    pred = model.load_run(run_dir, prod["data"], prod["cov"], prod["manifest"], device="cpu")
    return hop1.Hop1Predictor(pred, prod["rows"], chunk=chunk)


def test_predict_is_the_hand_padded_encoder_head(prod, anc):
    pred = _hop1_pred(prod, anc)
    arm, base = "T1__pe__pe", "T1__base__base"
    X = pred.corpus.get(arm, "counts", "chr19")
    h = hop1.HALO
    assert h == 30 == latent.HALO
    Xp = np.zeros(X.size + 2 * h)
    Xp[h:h + X.size] = X
    theta_e = pred._theta_t(arm, arm)[:256]
    with torch.no_grad():
        loc, disp = pred.ladder.form.encode_nb(
            torch.from_numpy(model.transform_x(Xp, "counts"))[None], theta_e[None])
    loc0, disp0 = pred.predict(arm, arm, base, "chr19")
    assert loc0.dtype == np.float32 and loc0.shape == X.shape
    np.testing.assert_allclose(loc0, loc[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(disp0, disp[0].numpy(), atol=1e-5)
    for chunk in (97, 256, 999):
        loc1, disp1 = _hop1_pred(prod, anc, chunk=chunk).predict(arm, arm, base, "chr19")
        np.testing.assert_allclose(loc1, loc0, atol=1e-5)
        np.testing.assert_allclose(disp1, disp0, atol=1e-5)


def test_predict_needs_the_base_as_target(prod, anc):
    pred = _hop1_pred(prod, anc)
    with pytest.raises(ValueError):
        pred.predict("T1__pe__pe", "T1__pe__pe", "T1__extsize__extsize", "chr19")
    with pytest.raises(ValueError):
        pred.predict("T1__pe__pe", "T1__pe__pe", "T2__base__base", "chr19")


def test_run_writes_hop1_json(prod, anc):
    assert hop1.main(_args(prod, anc) + ["--workers", "2"]) == 0
    assert (anc / "HOP1_DONE").is_file()
    out = json.loads((anc / "hop1.json").read_text())
    assert {"created_utc", "run", "config", "records", "timing"} <= set(out)
    assert out["config"]["chroms"] == ["chr19", "chr21"] and out["config"]["halo"] == 30
    assert out["config"]["rule"] == hop1.RULE and len(out["config"]["blacklist_sha256"]) == 64
    assert out["run"]["rung"] == "DSANC"
    recs = out["records"]
    rows = prod["rows"]
    base, arms = pairs._by_track(rows, "T1")
    fit = set(pairs.fit_pids(rows, "T1", tp.HOLDOUT))
    n_tr = len([r for r in arms if r["pid"] in fit])
    n_ho = len([r for r in arms if (r["arm"], r["level"]) == ("depth", "7.5M")])
    assert n_ho == 1 and len(recs) == n_tr + n_ho == len(arms)
    assert [r["group"] for r in recs] == ["trained"] * n_tr + ["heldout"] * n_ho
    assert {r["source_pid"] for r in recs if r["group"] == "heldout"} == {"T1__depth__7.5M"}
    for r in recs:
        assert set(pairs.PAIR_KEYS) <= set(r) and set(SPLIT_KEYS) <= set(r)
        assert r["kind"] == "hop1" and r["eval"] == "score" and r["direction"] == "arm_to_base"
        assert r["target_pid"] == base["pid"] == r["cov_tgt_pid"]
        assert r["cov_src_pid"] == r["source_pid"] and "pit_hist" in r
    # one record's CRPS and split, recomputed by hand on the same arrays
    pred = _hop1_pred(prod, anc)
    r = recs[0]
    locs, disps, ys = [], [], []
    for c in ("chr19", "chr21"):
        Y = pred.corpus.get(base["pid"], "counts", c)
        loc, disp = pred.predict(r["source_pid"], r["source_pid"], base["pid"], c)
        keep = ~data.read_blacklist_flags(prod["bed"], c, Y.size)
        locs.append(np.asarray(loc, np.float64)[keep])
        disps.append(np.asarray(disp, np.float64)[keep])
        ys.append(Y[keep])
    want, _ = score.metric_fields(pred, np.concatenate(locs), np.concatenate(disps),
                                  np.concatenate(ys), split=True)
    assert r["crps_all"] == pytest.approx(want["crps_all"], rel=1e-9)
    for k in SPLIT_KEYS:
        assert r[k] == pytest.approx(want[k], rel=1e-9)
    # HOP1_DONE: a second call skips without rewriting
    before = (anc / "hop1.json").read_bytes()
    assert hop1.main(_args(prod, anc)) == 0
    assert (anc / "hop1.json").read_bytes() == before


def test_dsfree_run_exits_2(prod, tmp_path):
    run_dir = tmp_path / "DSfree_T1_counts_real_s0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"rung": "DS"}))
    assert hop1.main(_args(prod, run_dir)) == 2
    assert not (run_dir / "hop1.json").exists() and not (run_dir / "HOP1_DONE").exists()


def test_latent_on_dsanc_writes_three_kinds(prod, anc):
    assert latent.RUNGS == ("DS", "DSBOX", "DSANC")
    assert latent.main(_args(prod, anc) + ["--workers", "1"]) == 0
    out = json.loads((anc / "latent.json").read_text())
    assert out["config"]["sigma_rule"].startswith("DS/DSBOX: median exp(log sigma_z); DSANC:")
    recs, rows = out["records"], prod["rows"]
    n = len([r for r in pairs.usable_products(rows) if r["track"] == "T1"])
    h = tp.HOLDOUT
    n_ident = (len(pairs.train_pairs(rows, "T1", h)) + len(pairs.heldout_pairs(rows, "T1", h))
               + len(pairs.law_pairs(rows, "T1")))
    kinds = [r["kind"] for r in recs]
    assert kinds.count("latent") == n
    assert kinds.count("agree") == n * (n - 1) // 2
    assert kinds.count("ident") == n_ident
    assert all(r["median_sigma_z"] is not None and r["median_sigma_z"] > 0
               for r in recs if r["kind"] == "latent")


def test_identity_init_dsanc_offset(prod):
    run_dir = _hand_run(prod, "DSanchor_T1_counts_real_s0_ident", "DSANC")
    pred = model.load_run(run_dir, prod["data"], prod["cov"], prod["manifest"], device="cpu")
    latent.run_latent(pred, prod["rows"], prod["bed"], run_dir, workers=1)
    lat = [r for r in json.loads((run_dir / "latent.json").read_text())["records"]
           if r["kind"] == "latent"]
    assert lat
    for r in lat:
        xs = []
        for c in ("chr19", "chr21"):
            X = pred.corpus.get(r["pid"], "counts", c)
            xs.append(X[~data.read_blacklist_flags(prod["bed"], c, X.size)])
        X = np.concatenate(xs).astype(np.float64)
        want = float(np.median(np.abs(np.log1p(X + 1) - np.log1p(X))))
        assert r["median_abs_z_minus_x"] == pytest.approx(want, abs=1e-5)


def test_latent_rung_d_still_exits_2(prod, tmp_path):
    run_dir = tmp_path / "D_T1_counts_real_s0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"rung": "D"}))
    assert latent.main(_args(prod, run_dir)) == 2
    assert not (run_dir / "latent.json").exists()
