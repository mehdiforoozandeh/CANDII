"""t125-P2f: h scoring, alone (on the true X') and in the chain (on f's predicted NB).

Written from plan_t125 "t125-P2f", the design note "Part 2" / "Scoring" and the PI decisions in
plan/T125_STATUS.md: push only = the exact push-forward of the NB through h (native-scale CRPS on P
and log-scale CRPS on log(1 + P)); own noise = the exact Gaussian-mixture CRPS on log(1 + P), and on
the P scale the point CRPS of the mean; both: MSE on log(1 + P) and native MSE of the mean; all /
nonzero / top 1 % bins by the true P; the oracle-scaled / scale-error split; a CRPS > 20 flag per
record; capped NB rows counted.

Synthetic set-up: `ladder.synth.make_products`, a Poisson control for every T1 product, the
control cache through P2a's `controls`, a 3-step depth-offset A run (`eta.train`), h run dirs
written with P2d's `save_ckpt` (random weights, so h is not the identity), and one h trained by
P2e's `hmap.train` on f's NB (its real checkpoint format).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import baseline_rungs as br  # noqa: E402
from ladder import data as ldata  # noqa: E402
from ladder import encoding, synth  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import score as lscore  # noqa: E402
from eta import model as em  # noqa: E402
from eta import train as et  # noqa: E402
from hmap import controls, features, formula, products  # noqa: E402
from hmap import model as hm  # noqa: E402
from hmap import pushforward as pf  # noqa: E402
from hmap import score  # noqa: E402
from hmap import train as ht  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
SCORE_PY = ROOT / "tools" / "t125" / "hmap" / "score.py"
FAST = {"window": 256, "batch": 8, "knot_sample": 20_000, "knot_block": 256}
N_BINS = {"chr1": 2000, "chr2": 1500, "chr19": 700, "chr21": 600, "chr22": 500, "chrX": 400}
CHROMS = ("chr19", "chr21")
LOG_KEYS = ("crps_log_all", "crps_log_nonzero", "crps_log_top1", "crps_log_oracle_scaled_all",
            "crps_log_scale_error_all", "crps_log_c_star_all")
SHARED_KEYS = ("mse_log1p_all", "mse_log1p_top1", "mse_all", "mse_top1") + LOG_KEYS
#: the native-scale keys: push = the proper CRPS of the pushed set; own noise = the point CRPS of
#: its mean, under its own names (never the push keys)
NATIVE = {False: ("crps_all", "crps_nonzero", "crps_top1", "crps_oracle_scaled_all",
                  "scale_error_all", "c_star_all"),
          True: ("crps_mean_point_all", "crps_mean_point_nonzero", "crps_mean_point_top1",
                 "crps_mean_point_oracle_scaled_all", "crps_mean_point_scale_error_all",
                 "crps_mean_point_c_star_all")}
FLAGS = {False: ("crps_all_gt_20", "n_crps_nonfinite"),
         True: ("crps_mean_point_all_gt_20", "n_crps_mean_point_nonfinite")}


def metric_keys(noise: bool) -> tuple:
    return NATIVE[noise] + SHARED_KEYS


def nat(noise: bool) -> str:
    return "crps_mean_point" if noise else "crps"


ALONE_BASE = {"h_name", "pid", "track", "assay", "mark_class", "arm", "level", "has_control",
              "n_all", "n_top1", "n_crps_log_nonfinite"}
CHAIN_BASE = set(lpairs.PAIR_KEYS) | {"f_run_name", "h_name", "n_all", "n_top1", "n_k_mean",
                                      "n_k_max", "n_capped", "n_crps_log_nonfinite"}


def assert_keys(rec: dict, base: set, noise: bool):
    assert base | set(metric_keys(noise)) | set(FLAGS[noise]) <= set(rec)
    assert not set(NATIVE[not noise]) & set(rec)        # the two native schemas never mix
    assert FLAGS[not noise][0] not in rec


# -- set-up ------------------------------------------------------------------------------------


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _write_tsv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _h_dir(runs: Path, name: str, variant: str, rung: int, noise: bool, seed: int,
           std_width: int | None = None) -> Path:
    """One learned h run dir as P2e writes it: config.json, ckpt.pt (P2d keys), TRAIN_DONE."""
    torch.manual_seed(seed)
    h = hm.HModel(variant, rung, noise, hm.N_IN[rung])
    with torch.no_grad():
        for p in h.parameters():
            p.normal_(0.0, 0.15)
        last = h.lin if hasattr(h, "lin") else h.net[-1]
        last.bias[0] = 0.4
        if noise:
            last.bias[1] = -0.7
    width = h.n_feat if std_width is None else std_width
    rng = np.random.default_rng(seed)
    std = hm.Standardiser().fit(rng.normal(1.0, 2.0, size=(500, width)), hm.continuous_cols(rung))
    d = runs / name
    d.mkdir(parents=True)
    cfg = {"h_name": name, "variant": variant, "rung": rung, "noise": int(noise),
           "source": "true", "seed": seed}
    hm.save_ckpt(d / "ckpt.pt", h, std, cfg, step=5, val=1.0)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "TRAIN_DONE").write_text("x\n")
    return d


@pytest.fixture(scope="module")
def S(tmp_path_factory):
    root = tmp_path_factory.mktemp("h_score")
    prod = root / "products"
    manifest, cov = synth.make_products(prod, 0, n_bins=N_BINS)
    with open(manifest, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols, rows = rd.fieldnames, list(rd)
    rng = np.random.default_rng(11)
    with np.load(prod / "T1__base__base" / "counts25.npz") as z:
        lam = {c: 0.6 * float(z[c].mean()) for c in z.files}
        lens = {c: z[c].shape[0] for c in z.files}
    ctl_rl = []
    for r in rows:
        if r["track"] != "T1":
            continue
        npz = prod / r["pid"] / "control_counts25.npz"
        synth._bin25.write_npz(npz, {c: rng.poisson(lam[c], lens[c]).astype(np.uint32)
                                     for c in lens})
        r["control_counts25_md5"] = _md5(npz)
        ctl_rl.append({"pid": r["pid"], "control_tagalign": "syn.tagAlign.gz",
                       "control_read_length_mean": "101.0", "n_lines_read": "100000"})
    _write_tsv(manifest, cols, rows)
    rl_tsv = root / "control_read_length.tsv"
    _write_tsv(rl_tsv, ["pid", "control_tagalign", "control_read_length_mean", "n_lines_read"],
               ctl_rl)
    ctl = root / "ctl_cache"
    for r in rows:
        controls.build_control_cache(prod, ctl, r["pid"], r)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    f_dir = et.run_training(manifest, cov, prod, root / "eta_runs", "A", "all", "real", 0,
                            max_steps=3, eval_every=3, device="cpu", **FAST)
    runs = root / "h_runs"
    h_affine = _h_dir(runs, "h_affine_r1_push_true_s0", "affine", 1, False, 0)
    h_mlp = _h_dir(runs, "h_mlp_r5_push_true_s1", "mlp", 5, False, 1)
    h_noise = _h_dir(runs, "h_formula_affine_r5_noise_true_s2", "formula_affine", 5, True, 2)
    # one h trained by P2e itself on f's NB (its standardiser fitted at k = f's mean)
    h_fnb = ht.run_training(manifest, cov, prod, ctl, runs, "formula_mlp", 5, 0, "fnb", 0,
                            fnb_run=f_dir, max_steps=3, eval_every=3, device="cpu",
                            control_rl_tsv=rl_tsv, window=256, batch=4)
    specs = ["formula:bin", "formula:frag", str(h_affine), str(h_mlp), str(h_noise), str(h_fnb)]
    infos = products.load(manifest, cov, rl_tsv)
    return {"root": root, "prod": prod, "manifest": manifest, "cov": cov, "ctl": ctl, "bed": bed,
            "rl": rl_tsv, "f_dir": f_dir, "runs": runs, "specs": specs, "infos": infos,
            "rows": lpairs.read_manifest(manifest), "h_affine": h_affine, "h_mlp": h_mlp,
            "h_noise": h_noise, "h_fnb": h_fnb}


def _alone(S, spec, out, workers=1):
    return score.score_alone(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], spec, out,
                             workers=workers, control_rl_tsv=S["rl"])


def _chain(S, specs, out, workers=1):
    return score.score_chain(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], S["f_dir"],
                             out, specs, workers=workers, control_rl_tsv=S["rl"])


@pytest.fixture(scope="module")
def alone(S):
    out = S["root"] / "alone"
    res = {}
    for spec in S["specs"]:
        # the worker pool on the formula h (no torch in the workers: macOS fork-safe)
        path = _alone(S, spec, out, workers=2 if spec == "formula:bin" else 1)
        res[spec] = json.loads(Path(path).read_text())
    return res


@pytest.fixture(scope="module")
def chain(S):
    out = S["root"] / "chain"
    _chain(S, S["specs"], out, workers=1)   # macOS: a forked torch forward can segfault
    d = out / S["f_dir"].name
    return {"dir": d, "res": {p.name: json.loads((p / "scores.json").read_text())
                              for p in d.iterdir() if p.is_dir()}}


def _by_hand_arrays(S, pid):
    """Per chromosome (kept bins only): counts, control (zeros without), P; from the corpora."""
    cp = ldata.Corpus(S["prod"])
    cc = ldata.Corpus(S["ctl"])
    info = S["infos"][pid]
    out = []
    for c in CHROMS:
        x = cp.get(pid, "counts", c).astype(np.float64)
        y = cp.get(pid, "pval", c).astype(np.float64)
        ctl = cc.get(pid, "counts", c).astype(np.float64) if info.has_control else None
        keep = ~ldata.read_blacklist_flags(S["bed"], c, x.size)
        out.append((c, x, y, ctl, keep))
    return out


def _std_apply(std: hm.Standardiser, feat: np.ndarray) -> np.ndarray:
    feat = feat.astype(np.float64).copy()
    for c in std.cols:
        feat[:, c] = 0.0 if std.std[c] < 1e-6 else (feat[:, c] - std.mean[c]) / std.std[c]
    return feat


# -- h specs -----------------------------------------------------------------------------------


def test_load_h_formula_and_learned(S):
    hb = score.load_h("formula:bin")
    hf = score.load_h("formula:frag")
    assert (hb.name, hf.name) == ("h_formula_bin", "h_formula_frag")
    assert not hb.noise and not hf.noise
    ha = score.load_h(S["h_affine"])
    hn = score.load_h(S["h_noise"])
    assert ha.name == "h_affine_r1_push_true_s0" and not ha.noise and ha.rung == 1
    assert hn.name == "h_formula_affine_r5_noise_true_s2" and hn.noise and hn.rung == 5
    hp = score.load_h(S["h_fnb"])                       # P2e's own checkpoint
    assert hp.name == f"h_formula_mlp_r5_push_{S['f_dir'].name}_s0" == S["h_fnb"].name
    assert hp.info()["source"] == f"fnb:{S['f_dir'].name}" and hp.info()["fnb_run"]
    assert hp.std.mean.shape == (22,) and 21 not in hp.std.cols   # formula column never scaled
    with pytest.raises(ValueError):
        score.load_h("formula:window")


def test_load_h_needs_train_done(S, tmp_path):
    d = _h_dir(tmp_path, "h_affine_r1_push_true_s1", "affine", 1, False, 1)
    (d / "TRAIN_DONE").unlink()
    with pytest.raises(FileNotFoundError):
        score.load_h(d)


def test_standardiser_width_without_formula_column(S, tmp_path):
    """P2e may fit the standardiser on the 21 rung-5 columns only: the formula column passes."""
    full = score.load_h(S["h_noise"])
    d = _h_dir(tmp_path, "h_formula_affine_r5_noise_true_s2", "formula_affine", 5, True, 2,
               std_width=21)
    narrow = score.load_h(d)
    narrow.std.mean[:21], narrow.std.std[:21] = full.std.mean[:21], full.std.std[:21]
    feat = np.random.default_rng(0).normal(size=(7, 3, 22)).astype(np.float32)
    np.testing.assert_array_equal(full.standardise(feat), narrow.standardise(feat))


def test_chunks_cover_sorted_and_bounded(monkeypatch):
    monkeypatch.setattr(score, "ELEM_BUDGET", 64)
    monkeypatch.setattr(score, "CHUNK_BINS", 10)
    sizes = np.array([1, 50, 3, 3, 100, 2, 1, 7, 7, 7, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4], np.int64)
    idx = np.arange(sizes.size)[::-1].copy()
    seen = []
    for rows in score.chunk_rows(sizes, idx):
        seen.extend(rows.tolist())
        w = sizes[rows].max()
        assert rows.size <= 10
        assert rows.size == 1 or rows.size * w <= 64
        assert np.all(np.diff(sizes[rows]) >= 0)
    assert sorted(seen) == list(range(sizes.size))


# -- alone -------------------------------------------------------------------------------------


def test_alone_records(S, alone):
    usable = [r["pid"] for r in lpairs.usable_products(S["rows"])]
    for spec, res in alone.items():
        recs = res["records"]
        assert [r["pid"] for r in recs] == usable
        for r in recs:
            noise = res["h"]["noise"] == 1
            assert_keys(r, ALONE_BASE, noise)
            for k in metric_keys(noise):
                assert r[k] is not None and math.isfinite(r[k]), (spec, r["pid"], k)
            assert r[FLAGS[noise][1]] == 0 and r["n_crps_log_nonfinite"] == 0
            assert r[FLAGS[noise][0]] == (r[f"{nat(noise)}_all"] > 20)
            assert r["n_all"] > 0 and r["n_top1"] == math.ceil(0.01 * r["n_all"])
        assert res["h"]["h_name"] == recs[0]["h_name"]
        n = nat(res["h"]["noise"] == 1)
        assert f"n_records_{n}_gt_20" in res and "n_records_crps_log_nonfinite" in res


def test_alone_push_affine_matches_numpy(S, alone):
    """crps_all = |P_hat - y| with P_hat = expm1(mu)_+ of h(log1p k, log1p ctl, assay one-hot)."""
    h, std, *_ = hm.load_ckpt(S["h_affine"] / "ckpt.pt")
    w = h.lin.weight.detach().double().numpy()[0]
    b = float(h.lin.bias.detach()[0])
    recs = {r["pid"]: r for r in alone[str(S["h_affine"])]["records"]}
    for pid, rec in recs.items():
        info = S["infos"][pid]
        ps, ys = [], []
        for _, x, y, ctl, keep in _by_hand_arrays(S, pid):
            feat = np.zeros((x.size, 9))
            feat[:, 0] = np.log1p(x)
            feat[:, 1] = 0.0 if ctl is None else np.log1p(ctl)
            feat[:, 2 + encoding.ASSAYS.index(info.assay)] = 1.0
            mu = _std_apply(std, feat) @ w + b
            ps.append(np.maximum(np.expm1(mu), 0.0)[keep])
            ys.append(y[keep])
        p, y = np.concatenate(ps), np.concatenate(ys)
        sub = br.subset_index(y.astype(np.float32))
        assert rec["n_all"] == y.size
        assert rec["crps_all"] == pytest.approx(np.abs(p - y).mean(), rel=1e-5, abs=1e-9)
        assert rec["crps_top1"] == pytest.approx(np.abs(p - y)[sub["top1"]].mean(), rel=1e-5)
        assert rec["crps_log_all"] == pytest.approx(np.abs(np.log1p(p) - np.log1p(y)).mean(),
                                                    rel=1e-5, abs=1e-9)
        assert rec["mse_log1p_all"] == pytest.approx(((np.log1p(p) - np.log1p(y)) ** 2).mean(),
                                                     rel=1e-5, abs=1e-9)
        assert rec["mse_all"] == pytest.approx(((p - y) ** 2).mean(), rel=1e-5, abs=1e-9)
        split = br.scale_split(p, y, lambda mm, yy: np.abs(mm - yy))
        assert rec["crps_oracle_scaled_all"] == pytest.approx(split["crps_oracle_scaled"],
                                                              rel=1e-5, abs=1e-9)
        assert rec["scale_error_all"] == pytest.approx(split["scale_error"], rel=1e-4, abs=1e-8)


def test_alone_formula_matches_formula_P(S, alone):
    for window in ("bin", "frag"):
        recs = {r["pid"]: r for r in alone[f"formula:{window}"]["records"]}
        for pid, rec in recs.items():
            info = S["infos"][pid]
            ps, ys = [], []
            for _, x, y, ctl, keep in _by_hand_arrays(S, pid):
                ps.append(formula.P(window, info, x, ctl)[keep])
                ys.append(y[keep])
            p, y = np.concatenate(ps), np.concatenate(ys)
            assert rec["crps_all"] == pytest.approx(np.abs(p - y).mean(), rel=1e-9, abs=1e-12)
            assert rec["mse_all"] == pytest.approx(((p - y) ** 2).mean(), rel=1e-9, abs=1e-12)


def test_alone_noise_is_gaussian_on_log1p(S, alone):
    """Own noise alone: crps_log = the closed-form Gaussian CRPS on log1p(P); crps = |mean_P - y|
    with mean_P = exp(mu + sigma^2 / 2) - 1."""
    h, std, *_ = hm.load_ckpt(S["h_noise"] / "ckpt.pt")
    recs = {r["pid"]: r for r in alone[str(S["h_noise"])]["records"]}
    for pid in ("T1__depth__15M", "T2__base__base"):
        info = S["infos"][pid]
        cl, ca, ms = [], [], []
        for _, x, y, ctl, keep in _by_hand_arrays(S, pid):
            fixed = features.fixed_features(info, x, ctl, 5)
            fcol = np.log1p(formula.P("frag", info, x, ctl))
            feat = np.concatenate([np.log1p(x)[:, None], fixed, fcol[:, None]], axis=1)
            with torch.no_grad():
                mu, ls = h(torch.from_numpy(_std_apply(std, feat).astype(np.float32)))
            mu, sg = mu.double().numpy(), np.exp(ls.double().numpy())
            t = np.log1p(y)
            z = (t - mu) / sg
            cl.append((sg * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi)))[keep])
            mean_p = np.exp(mu + sg ** 2 / 2) - 1
            ca.append(np.abs(mean_p - y)[keep])
            ms.append(((np.log1p(mean_p) - t) ** 2)[keep])
        rec = recs[pid]
        assert rec["crps_log_all"] == pytest.approx(np.concatenate(cl).mean(), rel=1e-4)
        assert rec["crps_mean_point_all"] == pytest.approx(np.concatenate(ca).mean(), rel=1e-4)
        assert rec["mse_log1p_all"] == pytest.approx(np.concatenate(ms).mean(), rel=1e-4)


def test_alone_cli_and_marker_skip(S, alone, tmp_path):
    out = tmp_path / "alone_cli"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmd = [sys.executable, str(SCORE_PY), "alone", str(S["manifest"]), str(S["cov"]),
           str(S["prod"]), str(S["ctl"]), str(S["bed"]), "formula:bin", str(out),
           "--workers", "1", "--control-rl-tsv", str(S["rl"])]
    subprocess.run(cmd, check=True, env=env, capture_output=True)
    d = out / "h_formula_bin"
    assert (d / "ALONE_DONE").is_file()
    got = json.loads((d / "scores.json").read_text())["records"]
    want = alone["formula:bin"]["records"]
    assert [r["crps_all"] for r in got] == pytest.approx([r["crps_all"] for r in want], rel=1e-12)
    mtime = (d / "scores.json").stat().st_mtime_ns
    r = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    assert "nothing to do" in r.stdout
    assert (d / "scores.json").stat().st_mtime_ns == mtime


# -- chain -------------------------------------------------------------------------------------


def test_chain_records(S, chain):
    pred = em.load_run(S["f_dir"], S["prod"], S["cov"], S["manifest"], device="cpu")
    tp = pred.train_pairs
    names = {score.load_h(s).name for s in S["specs"]}
    assert set(chain["res"]) == names
    assert (chain["dir"] / "CHAIN_DONE").is_file()
    for name, res in chain["res"].items():
        recs = res["records"]
        assert len(recs) == len(tp)
        assert [(r["source_pid"], r["target_pid"]) for r in recs] == \
            [(p["source_pid"], p["target_pid"]) for p in tp]
        assert res["n_capped_total"] == sum(r["n_capped"] for r in recs) == 0
        assert res["f_run"]["run_name"] == S["f_dir"].name
        for r in recs:
            noise = res["h"]["noise"] == 1
            assert_keys(r, CHAIN_BASE, noise)
            assert r["f_run_name"] == S["f_dir"].name and r["h_name"] == name
            assert r["n_k_mean"] >= 1 and r["n_k_max"] >= 1
            for k in metric_keys(noise):
                assert r[k] is not None and math.isfinite(r[k]), (name, k)
            assert r[FLAGS[noise][0]] == (r[f"{nat(noise)}_all"] > 20)


def test_chain_formula_bin_by_brute_force(S, chain):
    """The pushed set {P(k), NB(k)}: CRPS by the plain double sum, mean, n_k, on every pair."""
    pred = em.load_run(S["f_dir"], S["prod"], S["cov"], S["manifest"], device="cpu")
    recs = chain["res"]["h_formula_bin"]["records"]
    cp, cc = ldata.Corpus(S["prod"]), ldata.Corpus(S["ctl"])
    for p, rec in zip(pred.train_pairs[:6], recs[:6]):
        src, tgt = p["source_pid"], p["target_pid"]
        info = S["infos"][tgt]
        crps, mse, nk, ys = [], [], [], []
        for c in CHROMS:
            loc, disp = pred.predict(src, src, tgt, c)
            mu = pred.mean(loc)
            n = lscore.spread_of("counts", disp)
            ctl = cc.get(tgt, "counts", c).astype(np.float64) if info.has_control else None
            y = cp.get(tgt, "pval", c).astype(np.float64)
            keep = ~ldata.read_blacklist_flags(S["bed"], c, y.size)
            k, w, valid = pf.nb_support(mu, n)
            lam = formula.lambda_track(info, ctl, mu)
            v = formula.P_enumerated("bin", info, k, lam)
            v = np.where(valid, v, 0.0)
            e1 = (w * np.abs(v - y[:, None])).sum(1)
            pair = np.einsum("ri,rj,rij->r", w, w, np.abs(v[:, :, None] - v[:, None, :]))
            crps.append((e1 - 0.5 * pair)[keep])
            m = (w * v).sum(1)
            mse.append(((np.log1p(m) - np.log1p(y)) ** 2)[keep])
            nk.append(valid.sum(1)[keep])
            ys.append(y[keep])
        assert rec["crps_all"] == pytest.approx(np.concatenate(crps).mean(), rel=1e-7, abs=1e-10)
        assert rec["mse_log1p_all"] == pytest.approx(np.concatenate(mse).mean(), rel=1e-9,
                                                     abs=1e-12)
        assert rec["n_k_mean"] == pytest.approx(np.concatenate(nk).mean())
        assert rec["n_k_max"] == int(np.concatenate(nk).max())


def test_chain_noise_is_mixture_crps(S, chain):
    """Own noise in the chain, first pair (T1, with a control): crps_log = the Gaussian-mixture
    CRPS over the NB support on log1p(P); crps = |mixture mean on P - y|. Features per k: the
    bin's k, the fragment window = (k + the neighbours' predicted means) / bins kept, f's mean in
    every other treatment window, the formula column of the fragment variant at k."""
    h, std, *_ = hm.load_ckpt(S["h_noise"] / "ckpt.pt")
    pred = em.load_run(S["f_dir"], S["prod"], S["cov"], S["manifest"], device="cpu")
    p = next(q for q in pred.train_pairs if q["track"] == "T1")
    i = pred.train_pairs.index(p)
    rec = chain["res"]["h_formula_affine_r5_noise_true_s2"]["records"][i]
    src, tgt = p["source_pid"], p["target_pid"]
    info = S["infos"][tgt]
    cp, cc = ldata.Corpus(S["prod"]), ldata.Corpus(S["ctl"])
    cl, ca = [], []
    for c in CHROMS:
        loc, disp = pred.predict(src, src, tgt, c)
        mu_t = pred.mean(loc)
        k, w, valid = pf.nb_support(mu_t, lscore.spread_of("counts", disp))
        ctl = cc.get(tgt, "counts", c).astype(np.float64)
        y = cp.get(tgt, "pval", c).astype(np.float64)
        keep = ~ldata.read_blacklist_flags(S["bed"], c, y.size)
        fixed = features.fixed_features(info, mu_t, ctl, 5, treat_mean=mu_t)
        kfrag = features.k_frag_chain(k, mu_t, features.frag_bins(info.fraglen))
        lam = formula.lambda_track(info, ctl, mu_t)
        pfr = formula.P_enumerated("frag", info, k, lam, formula.frag_rest(mu_t, info))
        n, K = k.shape
        feat = np.concatenate([np.log1p(k)[..., None],
                               np.broadcast_to(fixed[:, None, :8], (n, K, 8)),
                               np.log1p(kfrag)[..., None],
                               np.broadcast_to(fixed[:, None, 9:], (n, K, 11)),
                               np.log1p(pfr)[..., None]], axis=-1)
        flat = _std_apply(std, feat.reshape(n * K, 22)).astype(np.float32)
        with torch.no_grad():
            m, ls = h(torch.from_numpy(flat))
        m = m.double().numpy().reshape(n, K)
        sg = np.exp(ls.double().numpy().reshape(n, K))
        cl.append(pf.crps_gauss_mixture(m, sg, w, np.log1p(y))[keep])
        ca.append(np.abs(pf.mean_gauss_mixture_logp(m, sg, w) - y)[keep])
    assert rec["crps_log_all"] == pytest.approx(np.concatenate(cl).mean(), rel=1e-4)
    assert rec["crps_mean_point_all"] == pytest.approx(np.concatenate(ca).mean(), rel=1e-4)


def test_chain_point_mass_equals_alone(S, alone, monkeypatch, tmp_path):
    """f = a point mass at the true X' (the target's counts): the chain equals `alone` on the target."""
    cp = ldata.Corpus(S["prod"])

    def point_nb(pred, src, tgt, chrom):
        x = cp.get(tgt, "counts", chrom).astype(np.float64)
        return x, np.ones_like(x)

    def point_support(mu, n, tail=pf.TAIL, k_max=pf.K_MAX):
        k = np.rint(np.asarray(mu)).astype(np.int64)[:, None]
        return k, np.ones(k.shape), np.ones(k.shape, bool)

    monkeypatch.setattr(score, "f_nb", point_nb)
    monkeypatch.setattr(score, "nb_support", point_support)
    monkeypatch.setattr(score, "support_size",
                        lambda mu, n, tail=pf.TAIL, k_max=pf.K_MAX: np.ones(np.size(mu), np.int64))
    _chain(S, S["specs"], tmp_path)
    d = tmp_path / S["f_dir"].name
    for spec in S["specs"]:
        a = {r["pid"]: r for r in alone[spec]["records"]}
        name = alone[spec]["h"]["h_name"]
        for rec in json.loads((d / name / "scores.json").read_text())["records"]:
            ref = a[rec["target_pid"]]
            assert rec["n_all"] == ref["n_all"] and rec["n_k_max"] == 1
            for k in metric_keys(alone[spec]["h"]["noise"] == 1):
                assert rec[k] == pytest.approx(ref[k], rel=1e-6, abs=1e-6), (spec, k)


def test_chain_capped_rows_counted(S, monkeypatch, tmp_path):
    monkeypatch.setattr(score, "K_MAX", 3)
    _chain(S, ["formula:frag", str(S["h_mlp"])], tmp_path)
    d = tmp_path / S["f_dir"].name
    for name in ("h_formula_frag", "h_mlp_r5_push_true_s1"):
        res = json.loads((d / name / "scores.json").read_text())
        assert res["n_capped_total"] > 0
        assert res["config"]["k_max"] == 3
        for r in res["records"]:
            assert r["n_k_max"] <= 4
            assert math.isfinite(r["crps_all"])


def test_chain_done_only_when_every_h_is_scored(S, monkeypatch, tmp_path):
    real = score.write_h_scores

    def flaky(d, obj):
        if Path(d).name == "h_formula_frag":
            raise RuntimeError("disk full")
        real(d, obj)

    monkeypatch.setattr(score, "write_h_scores", flaky)
    specs = ["formula:bin", "formula:frag"]
    with pytest.raises(RuntimeError):
        _chain(S, specs, tmp_path)
    d = tmp_path / S["f_dir"].name
    assert (d / "h_formula_bin" / "scores.json").is_file()
    assert not (d / "h_formula_frag" / "scores.json").exists()
    assert not (d / "CHAIN_DONE").exists()
    monkeypatch.setattr(score, "write_h_scores", real)
    mtime = (d / "h_formula_bin" / "scores.json").stat().st_mtime_ns
    _chain(S, specs, tmp_path)
    assert (d / "CHAIN_DONE").is_file()
    assert (d / "h_formula_frag" / "scores.json").is_file()
    assert (d / "h_formula_bin" / "scores.json").stat().st_mtime_ns == mtime   # not re-scored
    assert set((d / "CHAIN_DONE").read_text().split()) >= {"h_formula_bin", "h_formula_frag"}


def test_chain_cli_skips_when_done(S, chain):
    d = chain["dir"]
    mtimes = {p: (p / "scores.json").stat().st_mtime_ns for p in d.iterdir() if p.is_dir()}
    args = ["chain", str(S["manifest"]), str(S["cov"]), str(S["prod"]), str(S["ctl"]),
            str(S["bed"]), str(S["f_dir"]), str(d.parent), "--workers", "1",
            "--control-rl-tsv", str(S["rl"])]
    for s in S["specs"]:
        args += ["--h", s]
    assert score.main(args) == 0
    assert {p: (p / "scores.json").stat().st_mtime_ns for p in mtimes} == mtimes


# -- the log-scale split (PI 2026-10-11), non-finite counts, device default ---------------------


class _ShiftedTruth:
    """A fake h: mu = log1p(y) + c0 ln2 on every bin (1 + P scaled by 2^c0); own noise with a
    tiny sigma, or a point push. `nan_rows` get mu = NaN."""

    kind, rung, formula = "fake", None, False

    def __init__(self, name, c0, noise, nan_rows=()):
        self.name, self.c0, self.noise, self.nan_rows = name, c0, noise, np.asarray(nan_rows, int)

    def __call__(self, batch):
        y = batch.ctx["y"][batch.rows]
        mu = np.log1p(y) + self.c0 * math.log(2.0)
        mu[np.isin(batch.rows, self.nan_rows)] = np.nan
        mu = np.repeat(mu[:, None], batch.k.shape[1], axis=1)
        return mu, (np.full(mu.shape, math.log(1e-3)) if self.noise else None)


def _point_ctx(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.3, 0.0, rng.gamma(1.5, 2.0, n))
    return {"y": y, "mode": "point", "kpt": np.zeros(n, np.int64), "info": None,
            "n_blacklisted": 0}


@pytest.mark.parametrize("noise", [False, True])
def test_log_split_recovers_a_known_shift(noise):
    """Truth shifted by c0 = 0.75 on the log scale: c* = -0.75 and the oracle-scaled log CRPS
    is ~0 (exactly 0 for a point; sigma * (sqrt 2 - 1) / sqrt pi for a Gaussian of sigma 1e-3)."""
    ctx = _point_ctx()
    h = _ShiftedTruth("h_fake", 0.75, noise)
    f = score.score_context([h], ctx)[0]["h_fake"]
    assert f["crps_log_c_star_all"] == pytest.approx(-0.75, abs=1e-9)
    floor = 1e-3 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi) if noise else 0.0
    assert f["crps_log_oracle_scaled_all"] == pytest.approx(floor, abs=1e-9)
    far = 1e-3 / math.sqrt(math.pi) if noise else 0.0      # a Gaussian far from y: |m| - s/sqrt(pi)
    assert f["crps_log_all"] == pytest.approx(0.75 * math.log(2.0) - far, abs=1e-9)
    assert f["crps_log_scale_error_all"] == pytest.approx(0.75 * math.log(2.0) - far - floor,
                                                          abs=1e-9)
    assert f["n_crps_log_nonfinite"] == 0
    if not noise:                    # the native split of push: 1 + P scaled is not P scaled
        assert f["crps_oracle_scaled_all"] <= f["crps_all"]
        assert "crps_mean_point_all" not in f
    else:
        assert "crps_all" not in f and "crps_oracle_scaled_all" not in f
        assert f["crps_mean_point_oracle_scaled_all"] <= f["crps_mean_point_all"]


@pytest.mark.parametrize("noise", [False, True])
def test_log_split_matches_scale_split_on_points(noise):
    """A point push: the log split equals `baseline_rungs.scale_split` run on 1 + P scaled by
    2^c and scored as |log(1 + P) + c ln2 - log(1 + y)|. A mixture of sigma 1e-3 is the same up to
    the Gaussian's -sigma / sqrt(pi) away from the truth."""
    ctx = _point_ctx(seed=9)
    rng = np.random.default_rng(1)
    ly = np.log1p(ctx["y"])
    noisy = ly + rng.normal(0.3, 0.5, ly.size)

    class H(_ShiftedTruth):
        def __call__(self, batch):
            mu = np.maximum(noisy[batch.rows], 0.0)[:, None]
            return mu, (np.full(mu.shape, math.log(1e-3)) if self.noise else None)

    f = score.score_context([H("h_fake", 0.0, noise)], ctx)[0]["h_fake"]
    m = np.maximum(noisy, 0.0)

    def crps(mm, tt):      # in the scale_split form: mm = (1 + m) 2^c, scored on the log scale
        return np.abs(np.log(mm) - tt)

    r = br.scale_split(np.exp(m), ly, crps)      # m = log(1 + P), so 1 + P = exp(m)
    if not noise:
        assert f["crps_log_c_star_all"] == pytest.approx(r["c_star"], abs=1e-9)
        assert f["crps_log_oracle_scaled_all"] == pytest.approx(r["crps_oracle_scaled"],
                                                                abs=1e-12)
        assert f["crps_log_all"] == pytest.approx(r["crps"], abs=1e-12)
    else:
        assert f["crps_log_c_star_all"] == pytest.approx(r["c_star"], abs=0.0101)
        assert f["crps_log_oracle_scaled_all"] == pytest.approx(
            r["crps_oracle_scaled"] - 1e-3 / math.sqrt(math.pi), abs=1e-4)


@pytest.mark.parametrize("noise", [False, True])
def test_nonfinite_log_crps_is_counted(noise):
    ctx = _point_ctx()
    f = score.score_context([_ShiftedTruth("h_fake", 0.5, noise, nan_rows=[3, 50, 77])],
                            ctx)[0]["h_fake"]
    assert f["n_crps_log_nonfinite"] == 3
    assert f["crps_log_all"] is None
    assert f[FLAGS[noise][1]] == 3


def test_cli_device_defaults_to_auto(monkeypatch):
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return Path("x")

    monkeypatch.setattr(score, "score_chain", fake)
    score.main(["chain", "m", "c", "d", "ctl", "bl", "/nonexistent/f_run", "/nonexistent/out",
                "--h", "formula:bin", "--workers", "1"])
    assert seen["device"] == "auto"
