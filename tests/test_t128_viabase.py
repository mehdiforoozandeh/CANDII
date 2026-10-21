"""t128 old D via the base (`tools/t128/viabase.py`) and the opt-in law-pass CRPS split.

  * `score_law` default writes no split keys; `split=True` adds exactly the three split keys to
    every law record and changes nothing else;
  * `ViaBasePredictor.predict` is hop 1 (arm -> base), NB mean, hop 2 (base -> arm) by hand;
  * a pair with the base at either end raises; `run` writes law.json, LAW_DONE, viabase.json and
    leaves the old run dir untouched;
  * the 24-row task table and the Nibi script (DRY_RUN, #SBATCH lines, guards).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# tools/t128/__init__.py belongs to chunk t128-C2, so viabase is imported as a top-level module.
for _p in (REPO / "tools" / "t118", REPO / "tools", REPO / "tools" / "t128"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, score, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t126 import pairs as tp  # noqa: E402
import viabase  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
SPLIT_KEYS = {"crps_oracle_scaled_all", "scale_error_all", "c_star_all"}
SCRIPT = REPO / "slurm" / "t128" / "viabase.sh"


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128vb")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    run_dir = ltrain.run_training(manifest, cov, root / "products", root / "old" / "runs", "D",
                                  "T1", "counts", "real", 0, max_steps=4, eval_every=2,
                                  device="cpu", holdout=tp.HOLDOUT, window=256, batch=4,
                                  knot_sample=20_000, knot_block=256)
    rows = pairs.read_manifest(manifest)
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed, "run_dir": run_dir,
            "rows": rows, "data": root / "products"}


def _pred(prod):
    return model.load_run(prod["run_dir"], prod["data"], prod["cov"], prod["manifest"],
                          device="cpu")


def _law(prod, name, split):
    out = prod["root"] / name
    out.mkdir()
    kw = {} if split is None else {"split": split}
    score.score_law(_pred(prod), prod["rows"], prod["bed"], out, **kw)
    return json.loads((out / "law.json").read_text())["records"]


def test_score_law_split_is_opt_in(prod):
    plain = _law(prod, "law_default", None)
    off = _law(prod, "law_off", False)
    on = _law(prod, "law_on", True)
    assert plain == off
    assert plain and not any(SPLIT_KEYS & set(r) for r in plain)
    key = lambda r: (r["kind"], r["source_pid"], r["target_pid"])     # noqa: E731
    assert [key(r) for r in on] == [key(r) for r in plain]
    for a, b in zip(plain, on):
        if b["kind"] == "law":
            assert set(b) - set(a) == SPLIT_KEYS
        else:                                   # depthlaw copies drop the split, as today
            assert set(b) == set(a)
        assert set(a) <= set(b)
        assert abs(a["crps_all"] - b["crps_all"]) <= 1e-12
        for k in a:
            if k != "crps_all":
                assert a[k] == b[k], k
    assert all(r["kind"] != "law" or r["c_star_all"] is not None for r in on)


def test_viabase_predict_is_two_hops(prod):
    pred = viabase.load_viabase(prod["run_dir"], prod["data"], prod["cov"], prod["manifest"],
                                prod["rows"])
    law = pairs.law_pairs(prod["rows"], "T1")
    p = law[0]
    a, b = p["source_pid"], p["target_pid"]
    base = viabase.base_pid_of(prod["rows"], a)
    assert base == "T1__base__base" or base.startswith("T1")
    assert next(r for r in prod["rows"] if r["pid"] == base)["arm"] == "base"
    plain = _pred(prod)
    X = plain.corpus.get(a, "counts", "chr19")
    loc1, _ = model.predict_chrom(plain.ladder, X, plain._theta_t(a, base), "cpu")
    mid = plain.mean(loc1)
    want = model.predict_chrom(plain.ladder, mid.astype(np.float32), plain._theta_t(base, b), "cpu")
    got = pred.predict(a, a, b, "chr19")
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-6)
    # it is not the direct (never-trained) map
    direct, _ = plain.predict(a, a, b, "chr19")
    assert not np.allclose(direct, got[0])
    with pytest.raises(ValueError):
        pred.predict(base, base, b, "chr19")
    with pytest.raises(ValueError):
        pred.predict(a, a, base, "chr19")


def _mtimes(d: Path):
    return sorted((str(p.relative_to(d)), p.stat().st_mtime_ns) for p in d.rglob("*"))


def test_run_writes_out_dir_only(prod):
    old = prod["run_dir"]
    before = _mtimes(old) + [("", old.stat().st_mtime_ns)]
    out = prod["root"] / "t128" / "viabase" / old.name
    path = viabase.run(prod["manifest"], prod["cov"], prod["data"], prod["bed"], old, out)
    assert path == out / "law.json"
    assert (out / "LAW_DONE").is_file()
    meta = json.loads((out / "viabase.json").read_text())
    assert set(meta) == {"old_run_dir", "old_ckpt_md5", "intermediate", "git_sha"}
    assert meta["intermediate"] == "nb_mean" and meta["old_ckpt_md5"] == model.md5_path(old / "ckpt.pt")
    doc = json.loads(path.read_text())
    assert doc["run"]["run_name"] == old.name
    law = [r for r in doc["records"] if r["kind"] == "law"]
    assert len(law) == len(pairs.law_pairs(prod["rows"], "T1"))
    assert all(SPLIT_KEYS <= set(r) for r in law)
    assert _mtimes(old) + [("", old.stat().st_mtime_ns)] == before
    with pytest.raises(ValueError):
        viabase.run(prod["manifest"], prod["cov"], prod["data"], prod["bed"], old,
                    prod["root"] / "t126" / "D" / old.name)


def test_viabase_runs():
    r = viabase.VIABASE_RUNS
    assert len(r) == 24 == len({x["run_name"] for x in r})
    assert [x["index"] for x in r] == list(range(24))
    assert r[9] == {"index": 9, "run_name": "D_C19M16_counts_real_s0", "g": "C19M16", "seed": 0}
    assert {x["g"] for x in r} == set(pairs.G_IDS)


def _dry(idx, out, old="/old/runs", extra_env=None):
    env = {**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": str(idx),
           "PYTHON": sys.executable, "PYTHONPATH": str(REPO / "src"), **(extra_env or {})}
    return subprocess.run(["bash", str(SCRIPT), "/kit", "/products", "/cache", "/bl.bed", old, out],
                          capture_output=True, text=True, env=env)


def test_script_dry_run():
    r = _dry(9, "/out/t128")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t128/viabase.py run /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /bl.bed /old/runs/D_C19M16_counts_real_s0 "
        "/out/t128/viabase/D_C19M16_counts_real_s0 --workers 8"]
    assert _dry(9, "/out/t126").returncode == 2
    assert _dry(99, "/out/t128").returncode == 2


def test_script_dry_run_skips_done(tmp_path):
    out = tmp_path / "t128"
    (out / "viabase" / "D_C07M20_counts_real_s0").mkdir(parents=True)
    (out / "viabase" / "D_C07M20_counts_real_s0" / "LAW_DONE").write_text("x\n")
    r = _dry(0, str(out))
    assert r.returncode == 0 and r.stdout == ""


def test_script_header():
    text = SCRIPT.read_text()
    sb = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --job-name=t128_viabase" in sb
    assert "#SBATCH --exclude=c128,c166,c537,c371,c329" in sb
    assert "#SBATCH --time=4:00:00" in sb
    assert "#SBATCH --cpus-per-task=8" in sb
    assert "#SBATCH --mem=24000M" in sb
    assert not any("gpu" in l or "gres" in l for l in sb)
    assert "--array=9 " in text and "--array=0-8,10-23%12 " in text
    assert "OLD=/project/def-maxwl/mforooz/t126/D/runs" in text
    assert 'PYTHONPATH="$KIT/src"' in text
    assert "exit 3" in text
