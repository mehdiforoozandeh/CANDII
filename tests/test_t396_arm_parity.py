"""t396 iteration 2 chunk A3: gate (e) parity check (`tools/t396/arm_parity.py`) and the training
job script (`slurm/t396/train_arm.sh`).

Synthetic tables (`minicandi.synth`) and synthetic count configs only; the four real arm configs
are checked on Fir in chunk A5.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tools" / "t396"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import arm_parity as AP  # noqa: E402

BASE_CFG = {"rung": 4, "context_bins": 64, "param_budget": 1.0e6, "windows_seen": 1000,
            "lr": 0.001, "batch_size": 2, "average_input": False, "train_halves": "pairs",
            "heldout_loss": {"batches": 2, "every": 20}}


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    from minicandi.synth import make_evalset, make_synthetic_tables
    root = tmp_path_factory.mktemp("parity")
    t = make_synthetic_tables(root, n_cells=4, chrom_bins={"chr1": 1024, "chr2": 512})
    make_evalset(t, n_regions=2, region_bins=128)
    return t


@pytest.fixture(scope="module")
def base_hash(tables):
    return AP.stream_hash(BASE_CFG, tables, 0)


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


def test_param_count_of_base_budget():
    n = AP.param_count(BASE_CFG)               # 993,599: average_input off drops 16 parameters
    assert abs(n - AP.TARGET_PARAMS) / AP.TARGET_PARAMS <= AP.PARAM_TOL


def test_different_widths_fail_parity(tables, tmp_path):
    a = _write(tmp_path / "count_in_count_out.json", BASE_CFG)
    narrow = dict(BASE_CFG, param_budget=2.0e5)
    b = _write(tmp_path / "narrow.json", narrow)
    out = tmp_path / "parity.tsv"
    rc = AP.main(["--configs", str(b), str(a), "--tables", str(tables), "--seeds", "0",
                  "--out", str(out)])
    assert rc == 1
    rows = {r["arm"]: r for r in _rows(out)}
    assert list(_rows(out)[0])[-1] == "status"
    assert rows["count_in_count_out"]["status"] == "pass"
    assert rows["narrow"]["status"] == "fail"
    assert int(rows["narrow"]["param_count"]) < AP.TARGET_PARAMS
    assert rows["narrow"]["keys_differ_from_base"] == "param_budget"
    assert rows["narrow"]["hash_seed0"] == rows["count_in_count_out"]["hash_seed0"]


def test_equal_configs_pass_and_reference_diff(tables, tmp_path, base_hash):
    from minicandi.train import full_config
    a = _write(tmp_path / "count_in_count_out.json", BASE_CFG)
    b = _write(tmp_path / "other_lr.json", dict(BASE_CFG, lr=0.003))
    run = tmp_path / "ref_run"
    run.mkdir()
    ref = dict(full_config(BASE_CFG), lr=0.0005)
    (run / "run.json").write_text(json.dumps({"config": ref}), encoding="utf-8")
    out = tmp_path / "parity.tsv"
    rc = AP.main(["--configs", str(a), str(b), "--tables", str(tables), "--seeds", "0",
                  "--reference-run", str(run), "--out", str(out)])
    assert rc == 0
    rows = {r["arm"]: r for r in _rows(out)}
    assert {r["status"] for r in rows.values()} == {"pass"}
    assert rows["other_lr"]["hash_seed0"] == base_hash
    assert rows["other_lr"]["keys_differ_from_base"] == "lr"
    assert rows["count_in_count_out"]["reference_diff"] == "lr"


def test_equal_seeds_equal_hashes(tables, base_hash):
    assert AP.stream_hash(copy.deepcopy(BASE_CFG), tables, 0) == base_hash
    assert AP.stream_hash(BASE_CFG, tables, 1) != base_hash


def test_stream_change_changes_hash(tables, base_hash):
    """A config key that changes the sampler (the whole-track mask range) changes the hash."""
    cfg = dict(BASE_CFG, mask_prob_range=[0.9, 0.95])
    assert AP.stream_hash(cfg, tables, 0) != base_hash


@pytest.fixture(scope="module")
def one_batch(tables):
    from minicandi.train import full_config
    from minicandi.dataset import OnionDataset
    import numpy as np
    ds = OnionDataset(tables, "train", full_config(BASE_CFG), 0)
    for k in range(50):                    # a batch with a same-view target (a bin mask)
        b = ds.batch(2, np.random.default_rng([0, 4, k]))
        if b["x_bin_mask"].any() and any(m["halves_tracks"] for m in b["meta"]):
            return b
    pytest.skip("no batch with a bin mask and a halved track in 50 draws")


def test_digest_covers_mask_ids_and_halves(one_batch):
    d0 = AP.batch_digest(one_batch)
    assert AP.batch_digest(copy.deepcopy(one_batch)) == d0

    b = copy.deepcopy(one_batch)                       # the bin mask
    b["x_bin_mask"].view(-1)[0] = ~b["x_bin_mask"].view(-1)[0]
    assert AP.batch_digest(b) != d0

    b = copy.deepcopy(one_batch)                       # the hidden-track mask
    b["x_avail"].view(-1)[0] = ~b["x_avail"].view(-1)[0]
    assert AP.batch_digest(b) != d0

    b = copy.deepcopy(one_batch)                       # an input measurement id
    m = b["meta"][0]["measurements_in"]
    i = next(j for j, v in enumerate(m) if v)
    m[i] = m[i] + ".other"
    assert AP.batch_digest(b) != d0

    b = copy.deepcopy(one_batch)                       # the half choice
    meta = next(x for x in b["meta"] if x["halves_tracks"])
    meta["halves_tracks"] = meta["halves_tracks"][1:]
    assert AP.batch_digest(b) != d0

    b = copy.deepcopy(one_batch)                       # the input file
    b["meta"][0]["input_file"] = "pr1"
    assert AP.batch_digest(b) != d0


# ------------------------------------------------------------------ slurm/t396/train_arm.sh

SCRIPT = REPO / "slurm" / "t396" / "train_arm.sh"


def _run_script(tmp_path, cpus: str, *extra):
    env = dict(os.environ, DRY_RUN="1", SLURM_CPUS_PER_TASK=cpus, SLURM_TMPDIR=str(tmp_path))
    return subprocess.run(["bash", str(SCRIPT), "/kit", "/kit/arm.json", "3e-4", "/tables", "1",
                           str(tmp_path / "out"), *extra],
                          env=env, capture_output=True, text=True)


def test_train_arm_dry_run(tmp_path):
    r = _run_script(tmp_path, "12", "--max-windows", "800")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 1
    words = lines[0].split()
    assert "/kit/tools/minicandi/train.py" in words
    assert words[words.index("--device") + 1] == "cuda"
    assert words[words.index("--workers") + 1] == "11"
    assert words[words.index("--stage-dir") + 1] == f"{tmp_path}/stage"
    assert words[words.index("--config") + 1] == f"{tmp_path}/out/config.json"
    assert words[-2:] == ["--max-windows", "800"]
    assert not (tmp_path / "out").exists()


def test_train_arm_refuses_one_cpu(tmp_path):
    r = _run_script(tmp_path, "1")
    assert r.returncode == 2
    assert r.stdout == ""


def test_train_arm_header_has_no_gres():
    text = SCRIPT.read_text(encoding="utf-8")
    sb = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    assert "#SBATCH --job-name=sc_train" in sb
    assert not any("--gres" in ln for ln in sb)
