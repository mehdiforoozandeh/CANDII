"""t137 — `tools/t137/floor.py`: the resampling floor of the pseudoreplicate halves, on synthetic data.

A tmp pseudoreps tree of 2 products of one narrow track (base + depth 15M), chr19 and chr21 of
4 000 bins drawn Poisson around a planted peak set (plus a short chrM that must be dropped),
pr1 + pr2 = product by construction, a tiny blacklist, and a fake rungs_v2 dir whose oracle records
are computed here from `poisson_crps` on the same bins.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
N_BINS = 4000
PIDS = ("C19M16__base", "C19M16__depth__15M")
BLACKLIST = "chr19\t1000\t3010\nchr21\t50000\t51000\n"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fl = _load("t137_floor", ROOT / "tools" / "t137" / "floor.py")
br = _load("t118_baseline_rungs_for_t137", ROOT / "tools" / "t118" / "baseline_rungs.py")


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _mask(n, intervals):
    """The test's own overlap rule: a 25 bp bin is bad if it touches any interval by >= 1 bp."""
    bad = np.zeros(n, bool)
    for s, e in intervals:
        bad[s // 25: min((e - 1) // 25, n - 1) + 1] = True
    return bad


MASKS = {"chr19": _mask(N_BINS, [(1000, 3010)]), "chr21": _mask(N_BINS, [(50000, 51000)])}


def _kept(arrs):
    return np.concatenate([arrs[c][~MASKS[c]].astype(np.float64) for c in ("chr19", "chr21")])


def make_tree(root: Path, seed: int) -> dict:
    rng = np.random.default_rng(1000 + seed)
    pr, prod = root / "pseudoreps", root / "products"
    halves = {}
    for k, pid in enumerate(PIDS):
        arrs, a1, a2 = {}, {}, {}
        for c, n in (("chr19", N_BINS), ("chr21", N_BINS), ("chrM", 50)):
            level = np.repeat(np.exp(rng.normal(0, 0.7, n // 40 + 1)), 40)[:n] * 60.0
            peaks = rng.choice(n, max(1, n // 100), replace=False)
            level[peaks] *= 10.0
            x = rng.poisson(level * (1.0 if k == 0 else 0.98)).astype(np.uint32)
            h1 = rng.binomial(x.astype(np.int64), 0.5).astype(np.uint32)
            arrs[c], a1[c], a2[c] = x, h1, (x - h1).astype(np.uint32)
        (prod / pid).mkdir(parents=True)
        np.savez(prod / pid / "counts25.npz", **arrs)
        for h, a in (("pr1", a1), ("pr2", a2)):
            (pr / pid / h).mkdir(parents=True)
            np.savez(pr / pid / h / "counts25.npz", **a)
        halves[pid] = {"pr1": a1, "pr2": a2}
    (prod / "MANIFEST.tsv").write_text(
        "pid\ttrack\tassay\tarm\tlevel\n"
        f"{PIDS[0]}\tC19M16\tH3K27ac\tbase\tbase\n{PIDS[1]}\tC19M16\tH3K27ac\tdepth\t15M\n")
    (pr / "MANIFEST.tsv").write_text(
        "pid\thalf\tn_lines\n" + "".join(f"{p}\t{h}\t15000000\n" for p in reversed(PIDS)
                                         for h in ("pr1", "pr2")))
    (root / "blacklist.bed").write_text(BLACKLIST)
    rungs = root / "rungs_v2"
    rungs.mkdir()
    for pid in PIDS:
        a, b = _kept(halves[pid]["pr1"]), _kept(halves[pid]["pr2"])
        recs = []
        for d, (x, y) in (("pr1_to_pr2", (a, b)), ("pr2_to_pr1", (b, a))):
            sp = br.scale_split(np.maximum(x, 1e-3), y, br.poisson_crps)
            recs.append({"space": "counts", "rung": "oracle", "direction": d, "eval": "score",
                         "spread_crps_all": float(np.mean(br.poisson_crps(np.maximum(x, 1e-3), y))),
                         "crps_oracle_scaled_all": sp["crps_oracle_scaled"],
                         "scale_error_all": sp["scale_error"]})
            recs.append({"space": "pval", "rung": "oracle", "direction": d, "eval": "score",
                         "spread_crps_all": 0.5, "spread_crps_top1": 2.0,
                         "crps_oracle_scaled_all": 0.4, "scale_error_all": 0.1})
        (rungs / f"{pid}.json").write_text(json.dumps({"records": recs}))
    return halves


def run_product(root: Path, idx: int) -> int:
    return fl.main(["product", str(root / "pseudoreps"), str(root / "products"),
                    str(root / "blacklist.bed"), str(root / "out" / "t137"), str(idx)])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("t137")
    halves = make_tree(root, 0)
    for i in range(2):
        assert run_product(root, i) == 0
    out = root / "out" / "t137"
    assert fl.main(["aggregate", str(out), str(root / "rungs_v2"), "--expect", "2"]) == 0
    return root, out, halves


# --- thin / rng ---------------------------------------------------------------------------------

def test_thin_sum_within_3_sigma_and_ratio_1_is_identity():
    x = np.random.default_rng(0).poisson(5.0, 200_000).astype(np.uint32)
    for r in (0.5, 0.25, 0.125):
        t = fl.thin(x, r, fl.rng_for(0, 0, 1))
        assert t.dtype == np.uint32 and (t <= x).all()
        n = float(x.sum())
        assert abs(t.sum() - r * n) <= 3 * np.sqrt(n * r * (1 - r))
    assert fl.thin(x, 1.0, fl.rng_for(0, 0, 0)) is x


def test_rng_for_is_reproducible():
    a = fl.rng_for(1, 7, 2).integers(0, 1 << 30, 5)
    b = fl.rng_for(1, 7, 2).integers(0, 1 << 30, 5)
    c = fl.rng_for(0, 7, 2).integers(0, 1 << 30, 5)
    assert (a == b).all() and not (a == c).all()


# --- product ------------------------------------------------------------------------------------

def test_product_writes_14_rows_with_pinned_columns(tree):
    _, out, _ = tree
    for pid in PIDS:
        assert (out / "pairs" / f"{pid}.DONE").exists()
        with open(out / "pairs" / f"{pid}.tsv") as fh:
            header = fh.readline().rstrip("\n").split("\t")
        assert header == list(fl.COLUMNS)
        rows = _rows(out / "pairs" / f"{pid}.tsv")
        assert len(rows) == 14
        assert {(r["thin_ratio"], r["seed"]) for r in rows} == {
            ("1.0", "0"), ("0.5", "0"), ("0.5", "1"), ("0.25", "0"), ("0.25", "1"),
            ("0.125", "0"), ("0.125", "1")}
        assert {r["mark_class"] for r in rows} == {"narrow"}
    rows = _rows(out / "pairs" / f"{PIDS[1]}.tsv")
    assert {(r["arm"], r["level"]) for r in rows} == {("depth", "15M")}


def test_log2fc_flips_sign_and_symmetric_fields_match(tree):
    _, out, _ = tree
    rows = _rows(out / "pairs" / f"{PIDS[0]}.tsv")
    for i in range(0, 14, 2):
        a, b = rows[i], rows[i + 1]
        assert (a["direction"], b["direction"]) == ("pr1_to_pr2", "pr2_to_pr1")
        assert float(a["log2fc_top1"]) == pytest.approx(-float(b["log2fc_top1"]), abs=1e-15)
        for k in ("abs_log2fc_top1", "shape_1k", "divergence_bin", "spearman_fine", "n_coarse"):
            assert a[k] == b[k]


def test_crps_at_ratio_1_matches_own_poisson_crps(tree):
    _, out, halves = tree
    rows = _rows(out / "pairs" / f"{PIDS[0]}.tsv")
    a, b = _kept(halves[PIDS[0]]["pr1"]), _kept(halves[PIDS[0]]["pr2"])
    for r in rows:
        if r["thin_ratio"] != "1.0":
            continue
        x, y = (a, b) if r["direction"] == "pr1_to_pr2" else (b, a)
        want = float(np.mean(br.poisson_crps(np.maximum(x, 1e-3), y)))
        assert float(r["crps_all"]) == pytest.approx(want, abs=1e-9)
        assert int(r["n_all"]) == y.size
        assert int(r["n_top1"]) == int(np.ceil(0.01 * y.size))
        assert float(r["scale_error_all"]) == pytest.approx(
            float(r["crps_all"]) - float(r["crps_oracle_scaled_all"]), abs=1e-12)
        div = float(np.mean(np.abs(np.log2((a + 0.5) / (b + 0.5)))))
        assert float(r["divergence_bin"]) == pytest.approx(div, rel=1e-12)


def test_shape_grows_as_ratio_falls(tmp_path):
    per_ratio = {r: [] for r in fl.RATIOS}
    for s in range(3):
        root = tmp_path / f"s{s}"
        make_tree(root, s + 10)
        assert run_product(root, 0) == 0
        pid = sorted(PIDS)[0]
        rows = _rows(root / "out" / "t137" / "pairs" / f"{pid}.tsv")
        for r in fl.RATIOS:
            per_ratio[r].append(np.mean([float(x["shape_1k"]) for x in rows
                                         if float(x["thin_ratio"]) == r]))
    med = [np.median(per_ratio[r]) for r in fl.RATIOS]
    assert all(med[i] < med[i + 1] for i in range(len(med) - 1)), med


def test_bins1k_arrays(tree):
    _, out, halves = tree
    pid = sorted(PIDS)[0]
    z = np.load(out / "bins1k" / f"{pid}.npz")
    want = {f"{p}{d}_{c}" for p in "ab" for d in ("15", "375") for c in ("chr19", "chr21")}
    assert set(z.files) == want
    h = halves[pid]
    assert all(z[k].dtype == np.uint32 and z[k].size == N_BINS // 40 for k in z.files)
    assert int(z["a15_chr19"].sum()) == int(h["pr1"]["chr19"][: (N_BINS // 40) * 40].sum())
    # the ratio-0.25 seed-0 draw: pr1 then pr2, every chromosome but chrM/chrY, npz order
    rng = fl.rng_for(0, 0, 2)
    for p, key in (("pr1", "a"), ("pr2", "b")):
        for c in ("chr19", "chr21"):
            t = fl.thin(h[p][c], 0.25, rng)
            assert (fl.sums_1k(t) == z[f"{key}375_{c}"]).all()


def test_mismatched_product_exits_3(tmp_path):
    make_tree(tmp_path, 3)
    f = tmp_path / "products" / sorted(PIDS)[0] / "counts25.npz"
    z = dict(np.load(f))
    z["chr19"] = z["chr19"].copy()
    z["chr19"][5] += 1
    np.savez(f, **z)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "t137" / "floor.py"), "product",
                        str(tmp_path / "pseudoreps"), str(tmp_path / "products"),
                        str(tmp_path / "blacklist.bed"), str(tmp_path / "out" / "t137"), "0"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 3, p.stderr
    assert not (tmp_path / "out" / "t137" / "pairs" / f"{sorted(PIDS)[0]}.DONE").exists()


# --- aggregate ----------------------------------------------------------------------------------

def test_aggregate_files(tree):
    _, out, _ = tree
    for f in ("floor_pairs.tsv", "floor_curve.tsv", "floor_at_30M.tsv", "floor_p15M.tsv",
              "floor_check.txt", "config.json"):
        assert (out / f).exists(), f
    pairs = _rows(out / "floor_pairs.tsv")
    assert len(pairs) == 28
    assert [r["pid"] for r in pairs] == sorted(r["pid"] for r in pairs)
    curve = _rows(out / "floor_curve.tsv")
    assert len(curve) == 2 * 5
    for r in curve:
        if r["metric"] in ("shape_1k", "divergence_bin"):
            assert float(r["beta"]) < 0, r
    at30 = _rows(out / "floor_at_30M.tsv")
    shape = [r for r in at30 if r["metric"] == "shape_1k"]
    assert {r["mark_class"] for r in shape} == {"narrow", "all"}
    for r in shape:
        assert float(r["value_30M_extrap"]) < float(r["value_15M"])
        assert int(r["n_products"]) == 2
    p15 = _rows(out / "floor_p15M.tsv")
    assert len(p15) == 4 and {float(r["crps_all"]) for r in p15} == {0.5}
    check = (out / "floor_check.txt").read_text().splitlines()
    assert sum(line.startswith("FAIL") for line in check) == 0
    assert "PASS rows 28" in check
    assert len([line for line in check if line.startswith("PASS C19M16")]) == 4
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["ratios"] == list(fl.RATIOS) and cfg["seeds"] == list(fl.SEEDS)
    assert {r["beta_sign"] for r in curve} <= {"neg", "pos", "nan"}
    assert all(r["beta_sign"] == "neg" for r in curve if r["metric"] in ("shape_1k", "divergence_bin"))


def test_depth_axis_from_n_lines(tmp_path):
    """Two products at different reads per half (15M and 3.75M, the latter the mean of 3.749M and
    3.751M): reads_per_half = n_lines x ratio, and the class rows read every curve at fixed depths."""
    make_tree(tmp_path, 5)
    full = {PIDS[0]: 15_000_000, PIDS[1]: 3_750_000}
    (tmp_path / "pseudoreps" / "MANIFEST.tsv").write_text(
        "pid\thalf\tn_lines\n"
        f"{PIDS[0]}\tpr1\t15000000\n{PIDS[0]}\tpr2\t15000000\n"
        f"{PIDS[1]}\tpr1\t3749000\n{PIDS[1]}\tpr2\t3751000\n")
    for i in range(2):
        assert run_product(tmp_path, i) == 0
    out = tmp_path / "out" / "t137"
    assert fl.main(["aggregate", str(out), str(tmp_path / "rungs_v2"), "--expect", "2"]) == 0
    for r in _rows(out / "floor_pairs.tsv"):
        assert int(r["reads_per_half"]) == round(full[r["pid"]] * float(r["thin_ratio"]))
    curve = _rows(out / "floor_curve.tsv")
    for r in curve:
        assert int(r["reads_per_half_full"]) == full[r["pid"]]
        a, b = float(r["alpha"]), float(r["beta"])
        assert r["beta_sign"] == ("neg" if b < 0 else "pos")
        # the fit is on the product's own axis: refit from obs_* and compare
        x = [np.log2(full[r["pid"]] * q) for q in fl.RATIOS]
        y = [np.log(float(r[f"obs_{fl.RATIO_LABEL[q]}"])) for q in fl.RATIOS]
        bb, aa = np.polyfit(x, y, 1)
        assert (a, b) == pytest.approx((aa, bb), rel=1e-9)
        for k, d in (("15M", 15e6), ("7.5M", 7.5e6), ("3.75M", 3.75e6), ("1.875M", 1.875e6),
                     ("30M_extrap", 30e6)):
            assert float(r[f"value_{k}"]) == pytest.approx(np.exp(a + b * np.log2(d)), rel=1e-9)
    # the 3.75M product read at 3.75M is its own ratio-1 point on the line, not its 0.25 point
    at30 = _rows(out / "floor_at_30M.tsv")
    for m in fl.METRICS_CURVE:
        cr = [r for r in curve if r["metric"] == m]
        cls = next(r for r in at30 if r["metric"] == m and r["mark_class"] == "narrow")
        for k in ("15M", "7.5M", "3.75M", "1.875M", "30M_extrap"):
            want = np.nanmedian([float(r[f"value_{k}"]) for r in cr])
            assert float(cls[f"value_{k}"]) == pytest.approx(want, rel=1e-12)
        assert cls["beta_sign"] in ("neg", "pos", "nan")
        assert int(cls["n_beta_neg"]) + int(cls["n_beta_pos"]) <= 2
    p15 = _rows(out / "floor_p15M.tsv")
    assert {(r["pid"], int(r["reads_per_half"])) for r in p15} == set(full.items())
    check = (out / "floor_check.txt").read_text()
    assert "FAIL" not in check


def test_aggregate_refuses_missing_products(tree, tmp_path):
    _, out, _ = tree
    shutil.copytree(out / "pairs", tmp_path / "pairs")
    (tmp_path / "pairs" / f"{PIDS[0]}.DONE").unlink()
    assert fl.main(["aggregate", str(tmp_path), str(tree[0] / "rungs_v2"), "--expect", "2"]) == 2


def test_check_fails_on_wrong_oracle(tree, tmp_path):
    root, out, _ = tree
    shutil.copytree(out / "pairs", tmp_path / "pairs")
    shutil.copytree(root / "rungs_v2", tmp_path / "rungs")
    f = tmp_path / "rungs" / f"{PIDS[0]}.json"
    doc = json.loads(f.read_text())
    doc["records"][0]["spread_crps_all"] += 1e-3
    f.write_text(json.dumps(doc))
    assert fl.main(["aggregate", str(tmp_path), str(tmp_path / "rungs"), "--expect", "2"]) == 0
    check = (tmp_path / "floor_check.txt").read_text()
    assert check.count("FAIL") == 1


# --- slurm --------------------------------------------------------------------------------------

def _bash(script, *args, **env):
    return subprocess.run(["bash", str(ROOT / "slurm" / "t137" / script), *args],
                          capture_output=True, text=True, env={**os.environ, **env})


def test_floor_sh_dry_run_prints_pinned_line():
    p = _bash("floor.sh", "/kit", "/pr", "/prod", "/bl.bed", "/x/t137/floor",
              DRY_RUN="1", SLURM_ARRAY_TASK_ID="3")
    assert p.returncode == 0, p.stderr
    assert p.stdout == "python3 /kit/tools/t137/floor.py product /pr /prod /bl.bed /x/t137/floor 3\n"


def test_floor_sh_guard_refuses_other_dirs():
    p = _bash("floor.sh", "/kit", "/pr", "/prod", "/bl.bed", "/x/t118/floor",
              DRY_RUN="1", SLURM_ARRAY_TASK_ID="3")
    assert p.returncode == 2 and p.stdout == ""


def test_floor_agg_dry_run_and_headers():
    p = _bash("floor_agg.sh", "/kit", "/x/t137/floor", "/r", DRY_RUN="1")
    assert p.stdout == "python3 /kit/tools/t137/floor.py aggregate /x/t137/floor /r\n"
    assert _bash("floor_agg.sh", "/kit", "/x/t118/floor", "/r", DRY_RUN="1").returncode == 2
    t = (ROOT / "slurm" / "t137" / "floor.sh").read_text()
    for h in ("--account=def-maxwl", "--job-name=t137_floor", "--time=3:00:00",
              "--cpus-per-task=2", "--mem=16G", "--exclude=c128,c166,c537,c371,c329"):
        assert f"#SBATCH {h}\n" in t
    assert "--gres" not in t and 'export PYTHONPATH="$KIT/src"' in t
    a = (ROOT / "slurm" / "t137" / "floor_agg.sh").read_text()
    assert "#SBATCH --time=1:00:00\n" in a and "#SBATCH --mem=8G\n" in a


# --- t142: read thinning (pre-thinned halves) ---------------------------------------------------

def _prethin(root: Path, pid_index: int, fn=None) -> Path:
    """Write <root>/thinned/<pid>/floor_<r>_s<seed>_<half>.npz holding exactly what the unit draw
    gives (same rng, pr1 then pr2, every chromosome but chrM): read mode must then reproduce the
    default rows bit for bit. `fn(name, arrays)` may alter a file."""
    pid = fl.sorted_pids(root / "pseudoreps")[pid_index]
    d = root / "thinned" / pid
    d.mkdir(parents=True, exist_ok=True)
    halves = {h: dict(np.load(root / "pseudoreps" / pid / h / "counts25.npz"))
              for h in ("pr1", "pr2")}
    for di, ratio in enumerate(fl.RATIOS):
        if ratio == 1.0:
            continue
        for seed in fl.SEEDS:
            rng = fl.rng_for(seed, pid_index, di)
            for h in ("pr1", "pr2"):
                arr = {c: fl.thin(x, ratio, rng) for c, x in halves[h].items() if c not in fl.DROP}
                name = f"floor_{fl.RATIO_LABEL[ratio]}_s{seed}_{h}.npz"
                if fn is not None:
                    fn(name, arr)
                np.savez(d / name, **arr)
    return root / "thinned"


def test_read_mode_reproduces_unit_rows_given_the_same_draw(tmp_path):
    make_tree(tmp_path, 7)
    out_u, out_r = tmp_path / "u" / "t137", tmp_path / "r" / "t137"
    args = [str(tmp_path / "pseudoreps"), str(tmp_path / "products"), str(tmp_path / "blacklist.bed")]
    for i in range(2):
        thinned = _prethin(tmp_path, i)
        assert fl.main(["product", *args, str(out_u), str(i)]) == 0
        assert fl.main(["product", *args, str(out_r), str(i), "--thinned", str(thinned)]) == 0
    for pid in PIDS:
        assert (out_u / "pairs" / f"{pid}.tsv").read_text() == (out_r / "pairs" / f"{pid}.tsv").read_text()
        assert (out_u / "pairs" / f"{pid}.DONE").read_text() == "14 rows\n"
        assert (out_r / "pairs" / f"{pid}.DONE").read_text() == "14 rows thinning=read\n"
        zu, zr = np.load(out_u / "bins1k" / f"{pid}.npz"), np.load(out_r / "bins1k" / f"{pid}.npz")
        assert set(zu.files) == set(zr.files) and all((zu[k] == zr[k]).all() for k in zu.files)
    for out, mode in ((out_u, "unit"), (out_r, "read")):
        assert fl.main(["aggregate", str(out), str(tmp_path / "rungs_v2"), "--expect", "2"]) == 0
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["thinning"] == mode
        assert cfg["formulas"]["thin"] == (fl.FORMULAS["thin"] if mode == "unit" else fl.THIN_READ)
    assert (out_u / "floor_pairs.tsv").read_text() == (out_r / "floor_pairs.tsv").read_text()


def test_read_mode_needs_only_scored_chroms_except_bins1k(tmp_path):
    make_tree(tmp_path, 8)
    thinned = _prethin(tmp_path, 0)
    pid = fl.sorted_pids(tmp_path / "pseudoreps")[0]
    f = thinned / pid / "floor_r0.5_s1_pr2.npz"
    z = dict(np.load(f))
    np.savez(f, chr19=z["chr19"], chr21=z["chr21"])          # the scored chromosomes suffice
    assert fl.main(["product", str(tmp_path / "pseudoreps"), str(tmp_path / "products"),
                    str(tmp_path / "blacklist.bed"), str(tmp_path / "o" / "t137"), "0",
                    "--thinned", str(thinned)]) == 0


@pytest.mark.parametrize("defect", ["above", "missing_chrom", "missing_file", "shape"])
def test_read_mode_refuses_bad_thinned_files(tmp_path, defect):
    make_tree(tmp_path, 9)

    def spoil(name, arr):
        if name != "floor_r0.25_s0_pr1.npz":
            return
        if defect == "above":
            arr["chr21"] = arr["chr21"].copy()
            arr["chr21"][10] = arr["chr21"][10] + 10_000
        elif defect == "missing_chrom":
            del arr["chr21"]
        elif defect == "shape":
            arr["chr19"] = arr["chr19"][:-1]

    thinned = _prethin(tmp_path, 0, spoil)
    pid = fl.sorted_pids(tmp_path / "pseudoreps")[0]
    if defect == "missing_file":
        (thinned / pid / "floor_r0.125_s1_pr2.npz").unlink()
    assert fl.main(["product", str(tmp_path / "pseudoreps"), str(tmp_path / "products"),
                    str(tmp_path / "blacklist.bed"), str(tmp_path / "o" / "t137"), "0",
                    "--thinned", str(thinned)]) == 3
    assert not (tmp_path / "o" / "t137" / "pairs" / f"{pid}.DONE").exists()


def test_aggregate_refuses_mixed_thinning(tree, tmp_path):
    _, out, _ = tree
    shutil.copytree(out / "pairs", tmp_path / "pairs")
    (tmp_path / "pairs" / f"{PIDS[0]}.DONE").write_text("14 rows thinning=read\n")
    assert fl.main(["aggregate", str(tmp_path), str(tree[0] / "rungs_v2"), "--expect", "2"]) == 2


def test_default_config_records_unit(tree):
    _, out, _ = tree
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["thinning"] == "unit" and cfg["formulas"]["thin"] == fl.FORMULAS["thin"]
