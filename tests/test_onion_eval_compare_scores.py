"""SP2c compare_scores.py: planted differences in scores.json and suffstats.npz are reported."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import compare_scores as C  # noqa: E402

S = {"scorer_version": 2, "evalset_sha256": "E" * 64, "wall_seconds": 12.5,
     "count_crps": {"macro": 0.25, "per_assay": {"H3K4me3": 0.2, "ATAC-seq": None}, "ci": [0.1, 0.4]},
     "copy_leak": {"part1_wiring_pass": True},
     "scorer_settings": {"device": "cuda", "infer_batch": 32}}
A = {"gram/H3K4me3/0": np.arange(12, dtype=np.float64).reshape(3, 4),
     "pair/p1": np.array([1.0, np.nan, 3.0], dtype=np.float32), "names": np.array(["a", "b"])}


def write(d: Path, s=None, arrays=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(S if s is None else s))
    np.savez_compressed(d / "suffstats.npz", **(A if arrays is None else arrays))
    return d


def run(*argv) -> tuple[int, list[str]]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = C.main([str(x) for x in argv])
    return rc, buf.getvalue().splitlines()


def test_identical_dirs(tmp_path):
    rc, out = run(write(tmp_path / "a"), write(tmp_path / "b"))
    assert rc == 0 and out == ["differences: 0"]


def test_changed_leaf(tmp_path):
    s = json.loads(json.dumps(S))
    s["count_crps"]["per_assay"]["H3K4me3"] = 0.2000001
    rc, out = run(write(tmp_path / "a"), write(tmp_path / "b", s))
    assert rc == 1 and out[0] == "differences: 1"
    assert "count_crps.per_assay.H3K4me3" in out[1]
    assert run(tmp_path / "a", tmp_path / "b", "--rtol", "1e-5")[0] == 0     # within rtol


def test_null_against_number_and_bool(tmp_path):
    s = json.loads(json.dumps(S))
    s["count_crps"]["per_assay"]["ATAC-seq"] = 0.0
    s["copy_leak"]["part1_wiring_pass"] = False
    rc, out = run(write(tmp_path / "a"), write(tmp_path / "b", s))
    assert rc == 1 and out[0] == "differences: 2"


def test_missing_key_either_side(tmp_path):
    s = json.loads(json.dumps(S))
    del s["count_crps"]["ci"]
    s["new_metric"] = {"macro": 1.0}
    rc, out = run(write(tmp_path / "a"), write(tmp_path / "b", s))
    assert rc == 1 and out[0] == "differences: 3"
    assert any("count_crps.ci[0]: only in a" in x for x in out)
    assert any("new_metric.macro: only in b" in x for x in out)


def test_string_leaves_compare_presence_only(tmp_path):
    s = json.loads(json.dumps(S))
    s["evalset_sha256"] = "F" * 64
    assert run(write(tmp_path / "a"), write(tmp_path / "b", s))[0] == 0


def test_ignore(tmp_path):
    s = json.loads(json.dumps(S))
    s.update(wall_seconds=99.0, scorer_version=1)
    s["scorer_settings"] = {"device": "cpu", "infer_batch": 8}
    a, b = write(tmp_path / "a"), write(tmp_path / "b", s)
    rc, out = run(a, b)
    assert rc == 1 and out[0] == "differences: 3"
    rc, out = run(a, b, "--ignore", "wall_seconds", "--ignore", "scorer_version",
                  "--ignore", "scorer_settings")
    assert rc == 0, out
    rc, out = run(a, b, "--ignore", "wall_seconds", "--ignore", "scorer_version",
                  "--ignore", "scorer_settings.device")
    assert out[0] == "differences: 1" and "infer_batch" in out[1]


def test_array_value_shape_dtype_and_missing(tmp_path):
    a = write(tmp_path / "a")
    arr = dict(A)
    arr["gram/H3K4me3/0"] = A["gram/H3K4me3/0"].copy()
    arr["gram/H3K4me3/0"][1, 2] += 1e-9
    rc, out = run(a, write(tmp_path / "b", arrays=arr))
    assert rc == 1 and out[0] == "differences: 1" and "1 of 12 values differ" in out[1]
    assert run(a, tmp_path / "b", "--rtol", "1e-6")[0] == 0
    arr = dict(A, **{"gram/H3K4me3/0": np.arange(12, dtype=np.float64).reshape(4, 3)})
    rc, out = run(a, write(tmp_path / "c", arrays=arr))
    assert rc == 1 and "shape (3, 4) vs (4, 3)" in out[1]
    arr = dict(A, **{"pair/p1": A["pair/p1"].astype(np.float64)})
    rc, out = run(a, write(tmp_path / "d", arrays=arr))
    assert rc == 1 and "dtype" in out[1]
    arr = {k: v for k, v in A.items() if k != "pair/p1"}
    arr["pair/p2"] = A["pair/p1"]
    rc, out = run(a, write(tmp_path / "e", arrays=arr))
    assert rc == 1 and out[0] == "differences: 2"
    assert "pair/p1: only in a" in out[1] and "pair/p2: only in b" in out[2]
    arr = dict(A, names=np.array(["a", "c"]))
    rc, out = run(a, write(tmp_path / "f", arrays=arr))
    assert rc == 1 and "names: values differ" in out[1]


def test_missing_file(tmp_path):
    a = write(tmp_path / "a")
    b = tmp_path / "b"
    b.mkdir()
    (b / "scores.json").write_text(json.dumps(S))
    rc, out = run(a, b)
    assert rc == 1 and out == ["differences: 1", "suffstats.npz: only in a"]
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    assert run(tmp_path / "x", tmp_path / "y")[0] == 2


def test_cli_entry(tmp_path):
    import subprocess
    a, b = write(tmp_path / "a"), write(tmp_path / "b")
    p = subprocess.run([sys.executable, str(REPO / "tools" / "onion_eval" / "compare_scores.py"),
                        str(a), str(b)], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("differences: 0")


@pytest.mark.parametrize("path,key,hit", [("refs.sha256_floor", "refs", True),
                                          ("x.refs_dir", "refs", False),
                                          ("timing_seconds.load", "timing_seconds", True),
                                          ("a[0].device", "device", True)])
def test_ignored(path, key, hit):
    assert C.ignored(path, [key]) is hit
