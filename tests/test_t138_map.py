"""t138 regional map: features at 1 kb and the level-null readout, on synthetic inputs."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "t138"
FEATURES = ROOT / "tools" / "t138" / "features.py"
MAP = ROOT / "tools" / "t138" / "map.py"
N_BINS = 3000
CHROMS = ("chr1", "chr2")
PRODUCTS = [("C19M16__base__base", "C19M16", "H3K27ac", "base", "base"),
            ("C19M16__depth__15M", "C19M16", "H3K27ac", "depth", "15M"),
            ("C12M02__base__base", "C12M02", "DNase-seq", "base", "base")]


def py(*args, env=None):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          env=env)


def tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------- features.py build (fixtures)

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("feat") / "features.npz"
    p = py(FEATURES, "build", FIX / "annot", FIX / "narrowpeak", FIX / "bins1k", out)
    assert p.returncode == 0, p.stderr
    z = np.load(out)
    return {k: z[k] for k in z.files}, json.loads(out.with_suffix(".json").read_text())


def test_gc_of_known_sequence(built):
    f, _ = built
    g1, g2 = f["gc_chr1"], f["gc_chr2"]
    for i, want in [(0, 0.0), (1, 0.1), (2, 0.2), (4, 0.4), (6, 0.6), (10, 1.0), (16, 0.5)]:
        assert abs(g1[i] - want) < 1e-6, i
    assert np.isnan(g1[3]) and np.isnan(g1[5])  # all N; 600 N > 50 %
    assert abs(g2[7] - 0.7) < 1e-6
    assert g1.dtype == np.float32


def test_rep_fraction_of_planted_intervals(built):
    f, _ = built
    r1 = f["rep_chr1"]
    np.testing.assert_allclose(r1[[2, 5, 6, 7]], [0.8, 0.5, 1.0, 0.25], atol=1e-6)
    assert r1[[0, 1, 3, 4, 8]].sum() == 0
    assert abs(f["rep_chr2"][0] - 0.1) < 1e-6


def test_map_mean_of_planted_bedgraph(built):
    f, meta = built
    m1 = f["map_chr1"]
    np.testing.assert_allclose(m1[[0, 1, 10]], [1.0, 0.25, 0.2], atol=1e-6)
    assert m1[2] == 0
    assert abs(f["map_chr2"][19] - 0.6) < 1e-6  # the tail beyond bin 19 is dropped
    assert meta["mappability_missing"] is False


def test_ctx_peak_domain_other(built):
    f, meta = built
    c1, c2 = f["ctx_C19M16_chr1"], f["ctx_C19M16_chr2"]
    assert c2[5] == 2 and c1[26] == 2               # inside the planted peaks
    assert (c1[[27, 28, 29]] == 1).all()            # the planted broad plateau
    assert (c1[:20] == 0).all() and (np.delete(c2, 5) == 0).all()
    d = f["ctx_C12M02_chr1"]
    assert d[1] == 2 and d[2] == 2 and (f["ctx_C12M02_chr2"] == 0).all()  # pr1 file, not pr2
    assert meta["peak_files"]["C12M02"]["basename"].startswith("call_peak_pr1_")
    assert meta["peak_files"]["C19M16"]["basename"].startswith("call_peak_shard0_")
    cc = meta["ctx_counts"]["C19M16"]
    assert (cc["0"], cc["1"], cc["2"]) == (44, 4, 2)


def test_lengths_equal_bins1k(built):
    f, meta = built
    z = np.load(FIX / "bins1k" / "C19M16__base__base.npz")
    for c in CHROMS:
        for k in ("gc", "map", "rep", "ctx_C19M16", "ctx_C12M02"):
            assert f[f"{k}_{c}"].size == z[f"a15_{c}"].size
    assert "chrUn_x_" not in "".join(f) and "gc_chrM" not in f
    assert set(meta["files_sha256"]) >= {"hg38.fa.gz", "rmsk.txt.gz", "k36.umap.bedgraph.gz"}


def test_build_is_atomic_and_logs_peak_file(tmp_path):
    out = tmp_path / "features.npz"
    p = py(FEATURES, "build", FIX / "annot", FIX / "narrowpeak", FIX / "bins1k", out)
    assert p.returncode == 0, p.stderr
    assert out.exists() and not out.with_name("features.npz.tmp").exists()
    assert not (tmp_path / "features.json.tmp").exists()
    assert "call_peak_shard0_x.bfilt.narrowPeak.gz" in p.stderr
    assert "call_peak_pr1_x.no_chrM_MT.bfilt.narrowPeak.gz" in p.stderr


def test_build_refuses_two_peak_candidates(tmp_path):
    np_dir = tmp_path / "narrowpeak"
    shutil.copytree(FIX / "narrowpeak", np_dir)
    src = np_dir / "C19M16" / "call_peak_shard0_x.bfilt.narrowPeak.gz"
    shutil.copy(src, np_dir / "C19M16" / "call_peak_shard0_y.bfilt.narrowPeak.gz")
    out = tmp_path / "features.npz"
    p = py(FEATURES, "build", FIX / "annot", np_dir, FIX / "bins1k", out)
    assert p.returncode != 0
    assert "call_peak_shard0_x" in p.stderr and "call_peak_shard0_y" in p.stderr
    assert not out.exists()


def test_annot_fetch_deletes_corrupt_download(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "curl").write_text('#!/bin/bash\nwhile [[ $# -gt 0 ]]; do\n'
                               '  if [[ "$1" == "-o" ]]; then echo notgzip > "$2"; fi; shift\n'
                               'done\n')
    (fake / "curl").chmod(0o755)
    ann = tmp_path / "t138" / "annot"
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    p = subprocess.run(["bash", str(ROOT / "tools/t138/annot.sh"), str(ann)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 3, (p.stdout, p.stderr)
    assert "gzip -t" in p.stderr
    assert not any(ann.iterdir())  # neither hg38.fa.gz nor hg38.fa.gz.part is left


def test_mappability_missing_build(tmp_path):
    ann = tmp_path / "annot"
    shutil.copytree(FIX / "annot", ann)
    (ann / "k36.umap.bedgraph.gz").unlink()
    (ann / "MAPPABILITY_MISSING").write_text("neither url\n")
    out = tmp_path / "features.npz"
    p = py(FEATURES, "build", ann, FIX / "narrowpeak", FIX / "bins1k", out)
    assert p.returncode == 0, p.stderr
    z = np.load(out)
    assert not any(k.startswith("map_") for k in z.files)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["mappability_missing"] is True and meta["features"] == ["gc", "rep", "ctx"]


# ---------------------------------------------------------------- map.py run (synthetic 1 kb)

def synth(tmp: Path, inflate: bool, with_map: bool = True, seed: int = 1,
          lam_center: float = 60.0, lam_sd: float = 0.7):
    """2 chromosomes x 3000 1 kb bins; halves Poisson from one planted signal. With `inflate`,
    half b is over-dispersed inside the planted low-mappability decile."""
    rng = np.random.default_rng(seed)
    n = len(CHROMS) * N_BINS
    gc = rng.uniform(0.3, 0.6, n)
    low = rng.random(n) < 0.1
    mp = np.where(low, rng.uniform(0.0, 0.3, n), rng.uniform(0.7, 1.0, n))
    rep = rng.uniform(0, 1, n)
    ctx = {}
    for t in ("C19M16", "C12M02"):
        c = np.zeros(n, dtype=np.uint8)
        c[rng.random(n) < 0.1] = 1
        c[rng.random(n) < 0.02] = 2
        ctx[t] = c
    feats = {}
    for i, ch in enumerate(CHROMS):
        sl = slice(i * N_BINS, (i + 1) * N_BINS)
        feats[f"gc_{ch}"] = gc[sl].astype(np.float32)
        feats[f"rep_{ch}"] = rep[sl].astype(np.float32)
        if with_map:
            feats[f"map_{ch}"] = mp[sl].astype(np.float32)
        for t in ctx:
            feats[f"ctx_{t}_{ch}"] = ctx[t][sl]
    np.savez(tmp / "features.npz", **feats)

    b1 = tmp / "bins1k"
    b1.mkdir()
    for pid, track, *_ in PRODUCTS:
        lam = np.exp(rng.normal(np.log(lam_center), lam_sd, n))
        lam = lam * np.where(ctx[track] == 2, 10, np.where(ctx[track] == 1, 3, 1))
        a = rng.poisson(lam)
        lb = lam * np.exp(rng.normal(0, 1.2, n)) if inflate else lam
        b = rng.poisson(np.where(low, lb, lam) if inflate else lam)
        a3, b3 = rng.binomial(a, 0.25), rng.binomial(b, 0.25)
        arr = {}
        for i, ch in enumerate(CHROMS):
            sl = slice(i * N_BINS, (i + 1) * N_BINS)
            for k, v in (("a15", a), ("b15", b), ("a375", a3), ("b375", b3)):
                arr[f"{k}_{ch}"] = v[sl].astype(np.uint32)
        np.savez(b1 / f"{pid}.npz", **arr)
    man = tmp / "MANIFEST.tsv"
    man.write_text("pid\ttrack\tassay\tarm\tlevel\n" +
                   "".join("\t".join(r) + "\n" for r in PRODUCTS))
    return b1, tmp / "features.npz", man


def run_map(tmp, **kw):
    b1, feat, man = synth(tmp, **kw)
    out = tmp / "out"
    p = py(MAP, "run", b1, feat, man, out)
    assert p.returncode == 0, p.stderr
    return out


def test_blacklist_masks_planted_signal(tmp_path):
    """A signal planted only in blacklisted 1 kb bins moves nothing once --blacklist masks them."""
    b1, feat, man = synth(tmp_path, inflate=False)
    bed = tmp_path / "bl.bed"   # chr1 bins 100..199, and bin 500 by a 1 bp overlap of its last 25 bp
    bed.write_text("chr1\t100000\t200000\nchr1\t500999\t501000\n")
    bins = list(range(100, 200)) + [500]
    planted = tmp_path / "planted"
    planted.mkdir()
    for f in b1.glob("*.npz"):
        with np.load(f) as z:
            arr = {k: z[k] for k in z.files}
        for k in ("a15_chr1", "a375_chr1"):
            x = arr[k].copy()
            x[bins] = x[bins] * 20 + 500
            arr[k] = x
        np.savez(planted / f.name, **arr)
    outs = {}
    for name, src, extra in (("a", b1, ["--blacklist", bed]), ("b", planted, ["--blacklist", bed]),
                             ("c", planted, [])):
        p = py(MAP, "run", src, feat, man, tmp_path / name, *extra)
        assert p.returncode == 0, p.stderr
        outs[name] = tsv(tmp_path / name / "map_products.tsv")
    assert outs["a"] == outs["b"]
    assert [r["mean_d"] for r in outs["b"]] != [r["mean_d"] for r in outs["c"]]
    assert all(int(rb["n_bins"]) < int(rc["n_bins"]) for rb, rc in zip(outs["b"], outs["c"]))
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg["blacklist"] == str(bed) and cfg["n_blacklisted_1k_bins"] == len(bins)
    cfg = json.loads((tmp_path / "c" / "config.json").read_text())
    assert cfg["blacklist"] is None and cfg["n_blacklisted_1k_bins"] == 0


def test_level_only_null(tmp_path):
    out = run_map(tmp_path, inflate=False)
    rows = tsv(out / "map_products.tsv")
    assert len(rows) == 2 * len(PRODUCTS)
    for r in rows:
        assert abs(float(r["delta_R2"])) < 0.02, r["pid"]
        for f in ("gc", "map", "rep"):
            assert abs(float(r[f"spearman_r_{f}"])) <= 0.05, (r["pid"], r["depth"], f)
        assert float(r["R2_level"]) >= 0
    check = (out / "map_check.txt").read_text().splitlines()
    assert check[0].startswith("PASS rows 6") and check[1] == "PASS R2_level>=0"


def test_inflated_low_mappability(tmp_path):
    out = run_map(tmp_path, inflate=True)
    rows = tsv(out / "map_products.tsv")
    for r in rows:
        assert float(r["spearman_r_map"]) < -0.1, (r["pid"], r["depth"])
        assert float(r["enr_map_bottom_decile"]) > 1.5, (r["pid"], r["depth"])
    check = (out / "map_check.txt").read_text()
    assert "FAIL" not in check, check
    ctx = tsv(out / "map_context.tsv")
    assert {(r["mark_class"], r["depth"]) for r in ctx} == {
        (m, d) for m in ("narrow", "DNase") for d in ("15", "375")}
    dec = tsv(out / "map_deciles.tsv")
    assert len(dec) == 2 * 2 * 3 * 10
    narrow15 = {int(r["decile"]): float(r["mean_r"]) for r in dec
                if r["mark_class"] == "narrow" and r["depth"] == "15" and r["feature"] == "map"}
    assert narrow15[0] > max(narrow15[k] for k in range(1, 10))
    cls = tsv(out / "map_classes.tsv")
    assert any(r["reading"] == "delta_R2" and r["mark_class"] == "narrow" for r in cls)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["features"] == ["gc", "map", "rep", "ctx"] and cfg["mappability_missing"] is False


def test_sparse_halves_level_null_nonnegative(tmp_path):
    """~1.2 reads per 1 kb bin per half (sparse Poisson): the binned-mean null keeps R2_level >= 0
    and map_check passes, at both depths (375 is ~0.3 reads per bin per half)."""
    out = run_map(tmp_path, inflate=True, lam_center=1.2, lam_sd=0.3)
    rows = tsv(out / "map_products.tsv")
    assert len(rows) == 2 * len(PRODUCTS)
    for r in rows:
        assert float(r["R2_level"]) >= 0, (r["pid"], r["depth"], r["R2_level"])
    check = (out / "map_check.txt").read_text()
    assert "FAIL" not in check, check


def test_level_null_is_binned_mean_step():
    sys.path.insert(0, str(MAP.parent))
    try:
        import map as tmap
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(3)
    level = rng.integers(0, 5, 2000).astype(float)  # heavy ties: many quantile edges collapse
    d = rng.random(2000) + level
    dh = tmap.level_null(d, level)
    for v in np.unique(level):  # one value per level = a step function, equal to the bin mean
        assert np.unique(dh[level == v]).size == 1
    edges = np.unique(np.quantile(level, np.linspace(0, 1, tmap.N_LEVEL_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, edges.size - 2)
    for k in np.unique(idx):
        assert np.isclose(dh[idx == k][0], d[idx == k].mean())
    assert 1 - (d - dh).var() / d.var() >= 0


def test_map_without_mappability(tmp_path):
    out = run_map(tmp_path, inflate=False, with_map=False)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["mappability_missing"] is True and cfg["features"] == ["gc", "rep", "ctx"]
    rows = tsv(out / "map_products.tsv")
    assert all(r["spearman_r_map"] == "nan" for r in rows)
    assert len(rows) == 2 * len(PRODUCTS)


# ---------------------------------------------------------------- slurm scripts

def bash(script, *args, dry=True):
    env = {**os.environ, "DRY_RUN": "1" if dry else "0"}
    return subprocess.run(["bash", str(ROOT / "slurm" / "t138" / script), *args],
                          capture_output=True, text=True, env=env)


def test_map_sh_dry_run_lines():
    p = bash("map.sh", "/kit", "/a/t138/annot", "/np", "/x/t137/floor/bins1k", "/prod",
             "/x/t138/out")
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [
        "python3 /kit/tools/t138/features.py build /a/t138/annot /np /x/t137/floor/bins1k "
        "/x/t138/out/features.npz",
        "python3 /kit/tools/t138/map.py run /x/t137/floor/bins1k /x/t138/out/features.npz "
        "/prod/MANIFEST.tsv /x/t138/out --blacklist "
        "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"]


def test_guards_refuse_other_dirs():
    p = bash("map.sh", "/kit", "/a", "/np", "/b", "/prod", "/x/t137")
    assert p.returncode == 2 and "t138" in p.stderr
    p = bash("annot.sh", "/kit", "/x/t137")
    assert p.returncode == 2
    p = bash("annot.sh", "/kit", "/x/t138/annot")
    assert p.returncode == 0 and p.stdout.strip() == "bash /kit/tools/t138/annot.sh /x/t138/annot"


def test_scripts_parse_and_pin_resources():
    for s in ("slurm/t138/annot.sh", "slurm/t138/map.sh", "tools/t138/annot.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / s)]).returncode == 0
    m = (ROOT / "slurm/t138/map.sh").read_text()
    for line in ("--time=4:00:00", "--mem=32G", "--cpus-per-task=4", "--gres=none",
                 "--exclude=c128,c166,c537,c371,c329", "module load scipy-stack"):
        assert line in m
    a = (ROOT / "slurm/t138/annot.sh").read_text()
    for line in ("--time=2:00:00", "--mem=4G", "--cpus-per-task=1", "--gres=none"):
        assert line in a
    assert "--export" not in m.replace("NO --export", "") + a.replace("NO --export", "")
