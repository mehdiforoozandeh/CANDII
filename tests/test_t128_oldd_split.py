"""t128 old D law re-scored with the CRPS split (`tools/t128/oldd_split.py`).

  * `run` re-scores the old run's law pairs with the split: same records as a plain `score_law`
    plus the three split keys on every law record, the old run's own "run" block, LAW_DONE,
    oldd_split.json; the old run dir is untouched; a t126 out dir is refused;
  * the 48-row task table (real then nocov) and the Nibi script (DRY_RUN, #SBATCH lines, guards).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools", REPO / "tools" / "t128"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, score, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t126 import pairs as tp  # noqa: E402
import oldd_split  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
SPLIT_KEYS = {"crps_oracle_scaled_all", "scale_error_all", "c_star_all"}
SCRIPT = REPO / "slurm" / "t128" / "oldd_split.sh"


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128od")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    run_dir = ltrain.run_training(manifest, cov, root / "products", root / "old" / "runs", "D",
                                  "T1", "counts", "nocov", 0, max_steps=4, eval_every=2,
                                  device="cpu", holdout=tp.HOLDOUT, window=256, batch=4,
                                  knot_sample=20_000, knot_block=256)
    rows = pairs.read_manifest(manifest)
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed, "run_dir": run_dir,
            "rows": rows, "data": root / "products"}


def _mtimes(d: Path):
    return sorted((str(p.relative_to(d)), p.stat().st_mtime_ns) for p in d.rglob("*"))


def test_run_rescores_with_split_into_out_dir_only(prod):
    old = prod["run_dir"]
    pred = model.load_run(old, prod["data"], prod["cov"], prod["manifest"], device="cpu")
    score.score_law(pred, prod["rows"], prod["bed"], old)        # the old (t126-style) law pass
    before = _mtimes(old) + [("", old.stat().st_mtime_ns)]
    plain = json.loads((old / "law.json").read_text())
    out = prod["root"] / "t128" / "oldD_split" / old.name
    path = oldd_split.run(prod["manifest"], prod["cov"], prod["data"], prod["bed"], old, out)
    assert path == out / "law.json" and (out / "LAW_DONE").is_file()
    assert _mtimes(old) + [("", old.stat().st_mtime_ns)] == before
    meta = json.loads((out / "oldd_split.json").read_text())
    assert set(meta) == {"old_run_dir", "old_ckpt_md5", "old_law_md5", "git_sha"}
    assert meta["old_ckpt_md5"] == model.md5_path(old / "ckpt.pt")
    doc = json.loads(path.read_text())
    assert doc["run"] == plain["run"]                           # the old run's own block
    assert doc["run"]["model"] == "nocov" and doc["run"]["val_nll"] is not None
    key = lambda r: (r["kind"], r["source_pid"], r["target_pid"])     # noqa: E731
    assert [key(r) for r in doc["records"]] == [key(r) for r in plain["records"]]
    law = [r for r in doc["records"] if r["kind"] == "law"]
    assert law and all(SPLIT_KEYS <= set(r) for r in law)
    for a, b in zip(plain["records"], doc["records"]):
        assert abs(a["crps_all"] - b["crps_all"]) <= 1e-12
    with pytest.raises(ValueError):
        oldd_split.run(prod["manifest"], prod["cov"], prod["data"], prod["bed"], old,
                       prod["root"] / "t126" / "D" / old.name)
    with pytest.raises(ValueError):
        oldd_split.run(prod["manifest"], prod["cov"], prod["data"], prod["bed"], old,
                       prod["root"] / "t128" / "oldD_split" / "other_name")


def test_oldd_runs():
    r = oldd_split.OLDD_RUNS
    assert len(r) == 48 == len({x["run_name"] for x in r})
    assert [x["index"] for x in r] == list(range(48))
    assert r[9] == {"index": 9, "run_name": "D_C19M16_counts_real_s0", "g": "C19M16",
                    "model": "real", "seed": 0}
    assert r[33]["run_name"] == "D_C19M16_counts_nocov_s0"
    assert {x["g"] for x in r} == set(pairs.G_IDS)
    assert [x["model"] for x in r] == ["real"] * 24 + ["nocov"] * 24


def _dry(idx, out, old="/old/runs"):
    env = {**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": str(idx),
           "PYTHON": sys.executable, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(["bash", str(SCRIPT), "/kit", "/products", "/cache", "/bl.bed", old, out],
                          capture_output=True, text=True, env=env)


def test_script_dry_run(tmp_path):
    r = _dry(33, "/out/t128")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t128/oldd_split.py run /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /bl.bed /old/runs/D_C19M16_counts_nocov_s0 "
        "/out/t128/oldD_split/D_C19M16_counts_nocov_s0 --workers 8"]
    assert _dry(9, "/out/t126").returncode == 2
    assert _dry(48, "/out/t128").returncode == 2
    out = tmp_path / "t128"
    (out / "oldD_split" / "D_C07M20_counts_real_s0").mkdir(parents=True)
    (out / "oldD_split" / "D_C07M20_counts_real_s0" / "LAW_DONE").write_text("x\n")
    r = _dry(0, str(out))
    assert r.returncode == 0 and r.stdout == ""


def test_script_header():
    text = SCRIPT.read_text()
    sb = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --job-name=t128_oldd_split" in sb
    assert "#SBATCH --exclude=c128,c166,c537,c371,c329" in sb
    assert "#SBATCH --cpus-per-task=8" in sb
    assert not any("gpu" in l or "gres" in l for l in sb)
    assert "--array=0-47%12 " in text
    assert "OLD=/project/def-maxwl/mforooz/t126/D/runs" in text
    assert 'SPLIT="$OUT/oldD_split"' in text
    assert "exit 3" in text and "viabase" not in text
