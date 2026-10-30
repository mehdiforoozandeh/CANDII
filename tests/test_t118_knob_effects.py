"""t139: knob_effects.py --peakset; the default stays byte-identical to the t133 code (7004fa4)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "t139_knob"
SCRIPT = ROOT / "tools" / "t118" / "knob_effects.py"
sys.path.insert(0, str(ROOT / "tools" / "t118"))
import knob_effects as ke  # noqa: E402


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(out: Path, *extra) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--products", str(FIX / "products"),
                    "--blacklist", str(FIX / "blacklist.bed"), "--out", str(out), *extra],
                   check=True, capture_output=True)


def _rows(path: Path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("kb")
    _run(d / "base")
    _run(d / "base2", "--peakset", "base")
    _run(d / "shared", "--peakset", "shared")
    return d


def test_fixture_reproducible():
    mk = _module(FIX / "make_fixture.py", "t139_make_fixture")
    md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()  # noqa: E731
    for (pid, space, chrom), arr in mk.arrays().items():
        with np.load(FIX / "products" / pid / ke.SPACES[space]) as z:
            assert z[chrom].dtype == arr.dtype
            assert md5(z[chrom]) == md5(arr), (pid, space, chrom)
    assert (FIX / "products" / "MANIFEST.tsv").read_text() == mk.manifest_text()
    assert (FIX / "blacklist.bed").read_text() == mk.BLACKLIST


def test_default_is_byte_identical_to_golden(runs):
    golden = (FIX / "golden_pairs_default.tsv").read_bytes()
    assert (runs / "base" / "pairs.tsv").read_bytes() == golden
    assert (runs / "base2" / "pairs.tsv").read_bytes() == golden
    for f in ("levels.tsv", "ranking.tsv"):
        assert (runs / "base" / f).read_bytes() == (runs / "base2" / f).read_bytes()
    assert json.loads((runs / "base" / "config.json").read_text())["peakset"] == "base"


def test_shared_peaks_takes_top_of_mean_rank():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(1, 10, 1000), rng.uniform(1, 10, 1000)
    a[:10], b[:10] = 100, 0          # top in a only
    a[10:20], b[10:20] = 0, 100      # top in b only
    a[20:30], b[20:30] = 50, 50      # high in both
    got = ke.shared_peaks({"a": a, "b": b})
    from scipy.stats import rankdata
    want = ke.top_idx((rankdata(a) / a.size + rankdata(b) / b.size) / 2)
    assert set(got) == set(want) == set(range(20, 30))


def test_shared_peaks_depth_invariant():
    rng = np.random.default_rng(1)
    x = np.ones(1000)
    peaks = rng.choice(1000, 10, replace=False)
    x[peaks] = 1000 + np.arange(10) * 10.0
    thin = rng.poisson(0.25 * x).astype(np.float64)
    assert set(ke.shared_peaks({"p": x, "q": thin})) == set(peaks)
    assert set(ke.shared_peaks({"p": x})) == set(ke.top_idx(x)) == set(peaks)


def test_frip_in():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert ke.frip_in(x, np.array([2, 3])) == pytest.approx(0.7)


def test_shared_run(runs):
    base, shared = _rows(runs / "base" / "pairs.tsv"), _rows(runs / "shared" / "pairs.tsv")
    assert len(shared) == len(base) > 0
    assert all(r["peakset"] == "shared" for r in shared)
    assert all(int(r["n_peak_bins"]) == ke.top_idx(np.zeros(int(r["n_fine"]))).size for r in shared)
    assert json.loads((runs / "shared" / "config.json").read_text())["peakset"] == "shared"
    depth = [r for r in shared if r["arm"] == "depth" and r["level"] == "15M"
             and r["space"] == "counts"]
    assert len(depth) == 1
    assert abs(float(depth[0]["log2fc_top1"]) - np.log2(0.5)) < 0.1
    # a p-only knob leaves the counts untouched, so it never enters the counts set or rows
    assert not [r for r in shared if r["arm"] in ke.P_ONLY and r["space"] == "counts"]
    for r in shared:
        if r["space"] == "counts":
            assert float(r["dfrip"]) == pytest.approx(float(r["frip_arm"]) - float(r["frip_base"]))


def test_sbatch_passes_peakset():
    text = (ROOT / "slurm" / "t118" / "knob_effects.sh").read_text()
    assert 'PEAKSET="${5:-base}"' in text
    assert '--peakset "$PEAKSET"' in text
    r = subprocess.run(["bash", "-n", str(ROOT / "slurm" / "t118" / "knob_effects.sh")])
    assert r.returncode == 0
