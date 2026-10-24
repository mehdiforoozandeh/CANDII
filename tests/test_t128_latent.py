"""t128 latent pass (`tools/t128/latent.py`): readings 4-6 of plan/T126_ENCDEC_D2.md.

The run dirs are built by hand (a `model.Ladder("DS", ...)` saved in the t126 run-dir shape that
`model.load_run` reads), not by the training loop, so this test does not depend on it.
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
from t128 import pairs as tp  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}


def _ladder(rung="DS", perturb=True, seed=0) -> model.Ladder:
    torch.manual_seed(seed)
    lad = model.Ladder(rung, "counts", STATS)
    if perturb:
        with torch.no_grad():
            lad.g.mlp[-1].weight.normal_(0.0, 0.05)
            lad.g.cclean.normal_(0.0, 0.5)
            lad.form.f.head.weight.normal_(0.0, 0.1)
            lad.form.f.head.bias.normal_(0.0, 0.1)
    return lad


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128latent")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed,
            "data": root / "products", "rows": pairs.read_manifest(manifest)}


def _make_run(prod, name, rung="DS", model_name="real", perturb=True, g="T1") -> Path:
    """A run dir in the t126 shape: config.json, ckpt.pt, train_log.tsv, TRAIN_DONE."""
    rows = prod["rows"]
    holdout = [list(h) for h in tp.HOLDOUT]
    trp = pairs.train_pairs(rows, g, holdout)
    fit = pairs.fit_pids(rows, g, holdout)
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    lad = _ladder(rung, perturb)
    run_dir = prod["root"] / "runs" / name
    run_dir.mkdir(parents=True)
    config = {"run_name": name, "rung": rung, "arm": name.split("_")[0], "g": g,
              "space": "counts", "model": model_name, "seed": 0, "n_cov": encoding.N_COV,
              "fit_pids": fit, "train_pairs": trp, "holdout": holdout,
              "heldout_pairs": pairs.heldout_pairs(rows, g, holdout),
              "ids_perm": list(range(len(trp)))}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    torch.save({"g": lad.g.state_dict(), "form": lad.form.state_dict(),
                "stats": {"knots_x": [float(v) for v in STATS["knots_x"]],
                          "n0": STATS["n0"], "sigma0": STATS["sigma0"]},
                "encoder": enc.state(), "config": config, "step": 3, "val_nll": 1.0},
               run_dir / "ckpt.pt")
    (run_dir / "train_log.tsv").write_text("step\tloss\tmse_z\tkl\tval_nll\tseconds\n"
                                           "3\t1.0\t0.0\t0.0\t1.0\t0.1\n")
    (run_dir / "TRAIN_DONE").write_text("synthetic\n")
    return run_dir


def _load(prod, run_dir):
    return model.load_run(run_dir, prod["data"], prod["cov"], prod["manifest"], device="cpu")


def test_encode_chrom_matches_hand_padded_encode():
    lad = _ladder("DS").eval()
    X = np.random.default_rng(3).poisson(4.0, 1000).astype(np.float64)
    theta_e = torch.randn(256, generator=torch.Generator().manual_seed(1)) * 0.1 + \
        lad.form.f.init_theta()
    h = latent.HALO
    assert h == 30
    Xp = np.zeros(X.size + 2 * h)
    Xp[h:h + X.size] = X
    with torch.no_grad():
        mu, ls = lad.form.encode(torch.from_numpy(model.transform_x(Xp, "counts"))[None],
                                 theta_e[None])
    mu0, ls0 = latent.encode_chrom(lad, X, theta_e, torch.device("cpu"))
    np.testing.assert_allclose(mu0, mu[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(ls0, ls[0].numpy(), atol=1e-5)
    for chunk in (97, 256, 999):
        mu1, ls1 = latent.encode_chrom(lad, X, theta_e, torch.device("cpu"), chunk=chunk)
        np.testing.assert_allclose(mu1, mu0, atol=1e-5)
        np.testing.assert_allclose(ls1, ls0, atol=1e-5)


def test_theta_e_of_is_the_encoder_film(prod):
    pred = _load(prod, _make_run(prod, "DS_T1_counts_real_s0_theta"))
    pid = "T1__pe__pe"
    c = torch.from_numpy(pred.encoder.encode(pid)).reshape(1, -1)
    with torch.no_grad():
        want = pred.ladder.g.film(c, pred.ladder.g.cclean_of(c))[0]
    assert torch.allclose(latent.theta_e_of(pred, pid), want, atol=1e-6)
    tw = _load(prod, _make_run(prod, "DS_T1_counts_nocov_s0_theta", model_name="nocov"))
    assert torch.allclose(latent.theta_e_of(tw, pid), tw._nocov_theta[:256])


@pytest.mark.parametrize("rung,workers", [("DS", 1), ("DSBOX", 2)])
def test_run_writes_three_kinds(prod, rung, workers):
    run_dir = _make_run(prod, f"{'DS' if rung == 'DS' else 'DSbox'}_T1_counts_real_s0")
    rc = latent.main(["run", str(prod["manifest"]), str(prod["cov"]), str(prod["data"]),
                      str(prod["bed"]), str(run_dir), "--workers", str(workers)])
    assert rc == 0 and (run_dir / "LATENT_DONE").is_file()
    out = json.loads((run_dir / "latent.json").read_text())
    assert {"run", "created_utc", "config", "timing", "records"} <= set(out)
    assert out["config"]["chroms"] == ["chr19", "chr21"]
    recs = out["records"]
    rows = prod["rows"]
    prods = [r for r in pairs.usable_products(rows) if r["track"] == "T1"]
    n = len(prods)
    lat = [r for r in recs if r["kind"] == "latent"]
    agr = [r for r in recs if r["kind"] == "agree"]
    idn = [r for r in recs if r["kind"] == "ident"]
    assert [r["pid"] for r in lat] == [r["pid"] for r in prods]
    assert "T1__depth__7.5M" in {r["pid"] for r in lat}
    assert len(agr) == n * (n - 1) // 2 and all(r["pid_a"] < r["pid_b"] for r in agr)
    h = tp.HOLDOUT
    n_tr, n_ho = len(pairs.train_pairs(rows, "T1", h)), len(pairs.heldout_pairs(rows, "T1", h))
    n_law = len(pairs.law_pairs(rows, "T1"))
    assert len(idn) == n_tr + n_ho + n_law
    assert [r["group"] for r in idn] == ["trained"] * n_tr + ["heldout"] * n_ho + ["law"] * n_law
    assert all(set(pairs.PAIR_KEYS) <= set(r) for r in idn)
    # the identity distance of one law pair, recomputed by hand
    pred = _load(prod, run_dir)
    r = next(r for r in idn if r["group"] == "law")
    ms, xs = [], []
    for c in ("chr19", "chr21"):
        X = pred.corpus.get(r["source_pid"], "counts", c)
        loc, _ = pred.predict(r["source_pid"], r["source_pid"], r["target_pid"], c)
        keep = ~data.read_blacklist_flags(prod["bed"], c, X.size)
        ms.append(pred.mean(loc)[keep])
        xs.append(X[keep])
    want = score.swap_ratio(np.concatenate(ms), np.concatenate(xs))
    assert r[latent.IDENT_KEY] == pytest.approx(want, rel=1e-9, abs=1e-12)
    # agree reuses the latent Z: two products with identical counts (extsize = base) agree exactly
    same = next(a for a in agr if {a["arm_a"], a["arm_b"]} == {"base", "extsize"})
    assert same["median_abs_x_diff"] == 0.0
    # a perturbed model is not the identity
    assert any(x["median_abs_z_minus_x"] > 0 for x in lat)
    # LATENT_DONE: a second call skips without rewriting
    before = (run_dir / "latent.json").read_bytes()
    assert latent.main(["run", str(prod["manifest"]), str(prod["cov"]), str(prod["data"]),
                        str(prod["bed"]), str(run_dir)]) == 0
    assert (run_dir / "latent.json").read_bytes() == before


def test_identity_init_has_no_collapse_gap(prod):
    run_dir = _make_run(prod, "DS_T1_counts_real_s0_ident", perturb=False)
    pred = _load(prod, run_dir)
    latent.run_latent(pred, prod["rows"], prod["bed"], run_dir, workers=1)
    lat = [r for r in json.loads((run_dir / "latent.json").read_text())["records"]
           if r["kind"] == "latent"]
    assert lat and all(r["median_abs_z_minus_x"] == 0.0 for r in lat)
    assert all(r["corr_z_x"] == pytest.approx(1.0) for r in lat)
    assert all(r["median_sigma_z"] == pytest.approx(1.0) for r in lat)


def test_rung_d_exits_2(prod, tmp_path):
    run_dir = tmp_path / "D_T1_counts_real_s0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"rung": "D"}))
    assert latent.main(["run", str(prod["manifest"]), str(prod["cov"]), str(prod["data"]),
                        str(prod["bed"]), str(run_dir)]) == 2
    assert not (run_dir / "latent.json").exists()
