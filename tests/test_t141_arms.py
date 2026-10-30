"""t141 (q11): the one-warp residual per arm product and its regional readings, on synthetic inputs.

The synthetic world (built in a tmp dir; nothing is read from the cluster):
  2 chromosomes x 3 000 1 kb bins (+ a 25 bp tail that the 1 kb grid drops, + a chrM that is
  dropped); one narrow track C19M16 and one DNase track C12M02; features gc / map / rep / ctx with
  the bottom 10 % of mappability below 0.1.
  crop__36    an exact monotone transform of the base (counts 2x+1 per 25 bp, p 2x) -> residual 0
  mapq__0     the base, x3 inside the low-mappability bins only -> a mappability reading
  depth__15M  the base binomially thinned to half its depth -> sits within the thinning null
  ctlid__none counts bit-identical to the base, p changed -> scored in p only
  C12M02 mapq__0 the DNase mapq arm -> dropped
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHROMS = ("chr1", "chr2")
N1K = 3000
TAIL = 7
PRODUCTS = [  # pid, track, assay, arm, level, depth
    ("C19M16__base__base", "C19M16", "H3K27ac", "base", "base", 30_000_000),
    ("C19M16__crop__36", "C19M16", "H3K27ac", "crop", "36", 30_000_000),
    ("C19M16__mapq__0", "C19M16", "H3K27ac", "mapq", "0", 30_000_000),
    ("C19M16__depth__15M", "C19M16", "H3K27ac", "depth", "15M", 15_000_000),
    ("C19M16__ctlid__none", "C19M16", "H3K27ac", "ctlid", "none", 30_000_000),
    ("C12M02__base__base", "C12M02", "DNase-seq", "base", "base", 50_000_000),
    ("C12M02__mapq__0", "C12M02", "DNase-seq", "mapq", "0", 50_000_000),
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arms = _load("t141_arms", ROOT / "tools" / "t141" / "arms.py")
mk = _load("t141_map_knobs", ROOT / "tools" / "t141" / "map_knobs.py")
M = mk.M  # tools/t138/map.py


def tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _features(rng):
    F = {}
    for c in CHROMS:
        m = rng.uniform(0.3, 1.0, N1K)
        low = rng.random(N1K) < 0.1
        m[low] = rng.uniform(0.0, 0.1, low.sum())
        F[f"map_{c}"] = m.astype(np.float32)
        F[f"gc_{c}"] = rng.uniform(0.3, 0.6, N1K).astype(np.float32)
        F[f"rep_{c}"] = rng.uniform(0, 1, N1K).astype(np.float32)
        for t in ("C19M16", "C12M02"):
            F[f"ctx_{t}_{c}"] = rng.choice(3, N1K, p=[0.8, 0.15, 0.05]).astype(np.uint8)
    return F


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("t141")
    prod, out = root / "products", root / "t141"
    rng = np.random.default_rng(7)
    F = _features(rng)
    np.savez(root / "features.npz", **F)
    n25 = N1K * 40 + TAIL
    base_c, base_p = {}, {}
    for c in CHROMS + ("chrM",):
        lam = np.repeat(rng.lognormal(np.log(10), 1.0, N1K) / 40, 40)
        lam = np.concatenate([lam, np.full(TAIL, 0.3)])
        base_c[c] = rng.poisson(lam).astype(np.uint32)
        sig = np.concatenate([np.repeat(rng.lognormal(0, 0.8, N1K), 40), np.ones(TAIL)])
        base_p[c] = (np.round((sig + rng.uniform(0, 1, n25)) * 8) / 8).astype(np.float32)  # k/8
    low25 = {c: np.concatenate([np.repeat(F[f"map_{c}"] < 0.1, 40), np.zeros(TAIL, bool)])
             for c in CHROMS}
    low25["chrM"] = np.zeros(n25, bool)
    thin_rng = np.random.default_rng(12345)
    arrays = {
        "base": (base_c, base_p),
        "crop": ({c: 2 * x + 1 for c, x in base_c.items()},
                 {c: (2 * x).astype(np.float32) for c, x in base_p.items()}),
        "mapq": ({c: np.where(low25[c], 3 * x, x).astype(np.uint32) for c, x in base_c.items()},
                 {c: np.where(low25[c], 3 * x, x).astype(np.float32) for c, x in base_p.items()}),
        "depth": ({c: thin_rng.binomial(x.astype(np.int64), 0.5).astype(np.uint32)
                   for c, x in base_c.items()},
                  {c: (x + rng.integers(0, 4, n25) / 8).astype(np.float32)
                   for c, x in base_p.items()}),
        "ctlid": (base_c, {c: (x * 1.25 + rng.integers(0, 8, n25) / 8).astype(np.float32)
                           for c, x in base_p.items()}),
    }
    rows = []
    for pid, track, assay, arm, level, depth in PRODUCTS:
        d = prod / pid
        d.mkdir(parents=True)
        cnt, pv = arrays["base" if track == "C12M02" else arm]
        np.savez(d / "counts25.npz", **cnt)
        np.savez(d / "pval25.npz", **pv)
        rows.append({"pid": pid, "track": track, "assay": assay, "arm": arm, "level": level,
                     "knob": arm, "depth": depth})
    with open(prod / "MANIFEST.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    arm_pids = [r["pid"] for r in arms.arm_rows(arms.read_manifest(prod / "MANIFEST.tsv"))]
    for i in range(len(arm_pids)):
        assert arms.arm(prod, out, i) == 0
    # the floor: a stand-in t138 output with one class row at 15M
    t138 = root / "t138"
    t138.mkdir()
    (t138 / "map_classes.tsv").write_text(
        "mark_class\tdepth\treading\tmedian\tmin\tmax\tn_products\n"
        "narrow\t15\tmean_d\t0.42\t0.4\t0.44\t3\nnarrow\t15\tR2_level\t0.3\t0.2\t0.4\t3\n")
    cfg = mk.run(out / "arms1k", root / "features.npz", prod / "MANIFEST.tsv", t138, out / "out")
    return {"root": root, "prod": prod, "out": out, "F": F, "arm_pids": arm_pids, "cfg": cfg,
            "products": tsv(out / "out" / "knob_products.tsv"), "base_c": base_c}


def row(world, arm, space, series="arm"):
    sel = [r for r in world["products"]
           if r["arm"] == arm and r["space"] == space and r["series"] == series
           and r["track"] == "C19M16"]
    assert len(sel) == 1, (arm, space, series, len(sel))
    return sel[0]


# ---------------------------------------------------------------- (a) exact monotone transform

def test_monotone_arm_has_zero_residual(world):
    z = np.load(world["out"] / "arms1k" / "C19M16__crop__36.npz")
    for space in ("counts", "pval"):
        r = np.concatenate([z[f"{space}_r_{c}"] for c in CHROMS])
        assert r.size == 2 * N1K and np.abs(r).max() < 1e-9, space
        rd = row(world, "crop", space)
        assert rd["residual_zero"] == "1"
        for k in ("delta_R2", "delta_R2_map", "delta_R2_gc", "delta_R2_rep", "delta_R2_ctx"):
            assert abs(float(rd[k])) < 1e-9, (space, k)


# ---------------------------------------------------------------- (b) planted in low mappability

def test_low_mappability_arm_reads_mappability(world):
    for space in ("counts", "pval"):
        rd = row(world, "mapq", space)
        dm, dms = float(rd["delta_R2_map"]), float(rd["delta_R2_map_shuffle"])
        assert dm > 0.05, (space, dm)
        assert dm > dms + 0.05, (space, dm, dms)
        for f in ("gc", "rep"):
            assert abs(float(rd[f"delta_R2_{f}"])) < 0.01, (space, f)
        assert float(rd["spearman_r_map"]) < -0.1
        assert float(rd["enr_map_bottom_decile"]) > 1.5
        assert float(rd["delta_R2"]) > float(rd["delta_R2_shuffle"])


# ---------------------------------------------------------------- (c) depth arm vs thinning null

def test_depth_arm_sits_within_thinning_null(world):
    arm = row(world, "depth", "counts")
    nulls = [row(world, "depth", "counts", s) for s in ("null_s0", "null_s1")]
    for k in ("median_d", "mean_d"):          # the residual size, within the two-seed null band
        v = [float(n[k]) for n in nulls]
        pad = 0.1 * np.mean(v)
        assert min(v) - pad <= float(arm[k]) <= max(v) + pad, k
    for k in ("delta_R2", "delta_R2_map", "delta_R2_gc", "delta_R2_rep", "delta_R2_ctx"):
        lo = min(float(n[k]) for n in nulls) - 0.01
        hi = max(float(n[k]) for n in nulls) + 0.01
        assert lo <= float(arm[k]) <= hi, k
    # the null is p-free: no null rows in pval
    assert not [r for r in world["products"] if r["space"] == "pval" and r["series"] != "arm"]


def test_thinning_null_rng_convention(world):
    prod = world["prod"]
    rows = arms.read_manifest(prod / "MANIFEST.tsv")
    pid_index = [r["pid"] for r in rows].index("C19M16__depth__15M")
    z = np.load(world["out"] / "arms1k" / "C19M16__depth__15M.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["pid_index"] == pid_index and meta["null"]["thin_ratio"] == 0.5
    with np.load(prod / "C19M16__base__base" / "counts25.npz") as b:
        for s in (0, 1):
            g = np.random.default_rng([s, pid_index])
            for c in b.files:
                if c == "chrM":
                    continue
                t = g.binomial(b[c].astype(np.int64), 0.5)
                want = t[:N1K * 40].reshape(-1, 40).sum(1)
                np.testing.assert_array_equal(z[f"counts_null{s}_{c}"], want)


def test_depth_meta_has_observed_ratio(world):
    prod = world["prod"]
    meta = json.loads(str(np.load(world["out"] / "arms1k" / "C19M16__depth__15M.npz")["meta"]))
    sums = {}
    for k, pid in (("arm", "C19M16__depth__15M"), ("base", "C19M16__base__base")):
        with np.load(prod / pid / "counts25.npz") as z:
            sums[k] = int(sum(int(z[c].sum()) for c in z.files if c != "chrM"))
    assert meta["null"]["observed_sums"] == sums
    assert meta["null"]["observed_ratio"] == sums["arm"] / sums["base"]
    assert abs(meta["null"]["observed_ratio"] - 0.5) < 0.01          # nominal 15M / 30M


# ---------------------------------------------------------------- (d) space rules

def test_control_knob_only_in_pval_and_dnase_mapq_dropped(world):
    z = np.load(world["out"] / "arms1k" / "C19M16__ctlid__none.npz")
    assert not [k for k in z.files if k.startswith("counts_")]
    assert [k for k in z.files if k.startswith("pval_r_")]
    ctl = [r for r in world["products"] if r["arm"] == "ctlid"]
    assert {r["space"] for r in ctl} == {"pval"}
    assert (world["out"] / "arms1k" / "C12M02__mapq__0.DROPPED").exists()
    assert not (world["out"] / "arms1k" / "C12M02__mapq__0.npz").exists()
    assert not [r for r in world["products"] if r["track"] == "C12M02"]
    assert world["cfg"]["dropped"] == ["C12M02__mapq__0"]
    assert arms.spaces_for("narrow", "extsize") == ("counts", "pval")
    for a in ("ctlid", "ctldepth", "ratio"):
        assert arms.spaces_for("broad", a) == ("pval",) == mk.spaces_for("broad", a)
    assert arms.spaces_for("DNase", "mapq") == () == mk.spaces_for("DNase", "mapq")


def test_grid_drops_chrM_and_tail(world):
    z = np.load(world["out"] / "arms1k" / "C19M16__mapq__0.npz")
    assert not [k for k in z.files if k.endswith("_chrM")]
    for c in CHROMS:
        assert z[f"counts_base_{c}"].size == N1K and z[f"counts_base_{c}"].dtype == np.uint32
        np.testing.assert_array_equal(
            z[f"counts_base_{c}"], world["base_c"][c][:N1K * 40].reshape(-1, 40).sum(1))
    assert not list((world["out"] / "arms1k").glob("*.tmp.npz"))


# ---------------------------------------------------------------- readings are t138's

def test_readings_equal_t138_map_readings():
    rng = np.random.default_rng(3)
    n = 4000
    a = rng.poisson(rng.lognormal(2, 1, n)).astype(float)
    b = rng.poisson(rng.lognormal(2, 1, n)).astype(float) * 0.7
    F = {"gc": rng.uniform(0.3, 0.6, n), "map": rng.uniform(0, 1, n), "rep": rng.uniform(0, 1, n)}
    ctx = rng.choice(3, n)
    feats = ("gc", "map", "rep")
    want = M.readings(a, b, F, ctx, feats)
    got = mk.readings(np.abs(np.log2((a + 0.5) / (b + 0.5))), np.log2(a + b + 1), (a + b) > 0,
                      F, ctx, feats)
    for k, v in want.items():
        assert np.isclose(got[k], v, equal_nan=True, rtol=1e-12, atol=1e-12), k


# ---------------------------------------------------------------- outputs and the floor join

def test_tables_and_floor_join(world):
    o = world["out"] / "out"
    for f in ("knob_products.tsv", "knob_classes.tsv", "knob_context.tsv", "knob_deciles.tsv",
              "config.json"):
        assert (o / f).exists()
    assert not list(o.glob("*.tmp"))
    d = row(world, "depth", "counts")
    assert d["floor_depth"] == "15" and float(d["floor_mean_d"]) == 0.42
    assert d["floor_reads_per_half"] == "15000000" and d["floor_match"] == "exact"
    assert float(d["floor_R2_level"]) == 0.3 and d["floor_delta_R2"] == "nan"
    mq = row(world, "mapq", "counts")                                  # 30M: nearest is 15M
    assert mq["floor_depth"] == "15" and mq["floor_match"] == "nearest"
    assert float(mq["floor_mean_d"]) == 0.42
    dp = row(world, "depth", "pval")                                   # t138 is counts only
    assert dp["floor_depth"] == "counts_only" and dp["floor_mean_d"] == "nan"
    assert dp["floor_reads_per_half"] == "NA" and dp["floor_match"] == "NA"
    cls = tsv(o / "knob_classes.tsv")
    md = [r for r in cls if r["arm"] == "depth" and r["space"] == "counts"
          and r["series"] == "arm" and r["reading"] == "mean_d"]
    assert len(md) == 1 and float(md[0]["floor_median"]) == 0.42
    keys = {(r["arm"], r["space"], r["series"]) for r in cls}
    assert ("depth", "counts", "null_s0") in keys and ("ctlid", "counts", "arm") not in keys
    ctx = tsv(o / "knob_context.tsv")
    assert {r["context_label"] for r in ctx} == {"other", "domain", "peak"}
    dec = tsv(o / "knob_deciles.tsv")
    assert {r["feature"] for r in dec} == {"gc", "map", "rep"}


def test_floor_depth_nearest():
    f = mk.floor_depth
    assert f("30000000", "counts") == ("15", 15_000_000, "nearest")
    assert f("50000000", "counts") == ("15", 15_000_000, "nearest")
    assert f("15000000", "counts") == ("15", 15_000_000, "exact")
    assert f("3750000", "counts") == ("375", 3_750_000, "exact")
    assert f("7500000", "counts") == ("375", 3_750_000, "nearest")    # tie -> the larger floor
    assert f("1000000", "counts") == ("375", 3_750_000, "nearest")
    assert f("30000000", "pval") == ("counts_only", "NA", "NA")
    assert f("", "counts") == ("NA", "NA", "NA")


def test_floor_columns_in_class_tables(world):
    o = world["out"] / "out"
    for name in ("knob_classes.tsv", "knob_context.tsv", "knob_deciles.tsv"):
        r = tsv(o / name)
        assert {"floor_depth", "floor_reads_per_half", "floor_match"} <= set(r[0]), name
    assert world["cfg"]["blacklist"] is None and world["cfg"]["n_blacklisted_1k_bins"] == 0


# ---------------------------------------------------------------- the blacklist mask

BL_BINS = list(range(100, 200)) + [500]       # chr1 1 kb bins; bin 500 by a 1 bp overlap only


def _bed(path):
    path.write_text("chr1\t100000\t200000\nchr1\t500999\t501000\n")
    return path


def test_blacklist_1k_mask(tmp_path):
    bad = M.blacklist_1k(_bed(tmp_path / "bl.bed"), ["chr1", "chr2"], {"chr1": 1000, "chr2": 10})
    assert bad.size == 1010 and sorted(np.flatnonzero(bad)) == BL_BINS


def test_blacklist_masks_planted_signal(world, tmp_path):
    import shutil
    bed = _bed(tmp_path / "bl.bed")
    args = (world["root"] / "features.npz", world["prod"] / "MANIFEST.tsv", tmp_path / "no_t138")
    clean = world["out"] / "arms1k"
    planted = tmp_path / "planted"
    shutil.copytree(clean, planted)
    pid = "C19M16__crop__36"
    with np.load(planted / f"{pid}.npz") as z:
        arr = {k: z[k] for k in z.files}
    for sp in ("counts", "pval"):
        for k in (f"{sp}_arm_chr1", f"{sp}_r_chr1"):
            x = arr[k].copy()
            x[BL_BINS] = x[BL_BINS] + 50
            arr[k] = x
    np.savez(planted / f"{pid}.npz", **arr)
    cfg = mk.run(clean, *args, tmp_path / "a", blacklist=bed)
    mk.run(planted, *args, tmp_path / "b", blacklist=bed)
    mk.run(planted, *args, tmp_path / "c")
    assert cfg["blacklist"] == str(bed) and cfg["n_blacklisted_1k_bins"] == len(BL_BINS)
    a, b, c = (tsv(tmp_path / x / "knob_products.tsv") for x in "abc")
    assert a == b                                    # masked: the planted bins move nothing
    ca = [r for r in a if r["pid"] == pid and r["space"] == "counts"][0]
    cc = [r for r in c if r["pid"] == pid and r["space"] == "counts"][0]
    assert ca["mean_d"] != cc["mean_d"]              # unmasked: they do
    assert int(ca["n_bins"]) < int(cc["n_bins"])


def test_missing_floor_reads_na(world, tmp_path):
    cfg = mk.run(world["out"] / "arms1k", world["root"] / "features.npz",
                 world["prod"] / "MANIFEST.tsv", tmp_path / "no_t138", tmp_path / "out")
    assert not any(cfg["floor_tables_present"].values())
    rows = tsv(tmp_path / "out" / "knob_products.tsv")
    assert all(r["floor_mean_d"] == "nan" for r in rows)


def test_missing_arm_refused(world, tmp_path):
    a1k = tmp_path / "arms1k"
    a1k.mkdir()
    with pytest.raises(SystemExit, match="missing"):
        mk.run(a1k, world["root"] / "features.npz", world["prod"] / "MANIFEST.tsv",
               tmp_path / "t138", tmp_path / "out")


# ---------------------------------------------------------------- slurm scripts

def bash(script, *args, env_extra=None):
    env = {**os.environ, "DRY_RUN": "1", **(env_extra or {})}
    return subprocess.run(["bash", str(ROOT / "slurm" / "t141" / script), *args],
                          capture_output=True, text=True, env=env)


def test_arms_sh_dry_run_and_guard():
    p = bash("arms.sh", "/kit", "/prod", "/x/t141", env_extra={"SLURM_ARRAY_TASK_ID": "4"})
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "python3 /kit/tools/t141/arms.py arm /prod /x/t141 4"
    p = bash("arms.sh", "/kit", "/prod", "/x/t138", env_extra={"SLURM_ARRAY_TASK_ID": "4"})
    assert p.returncode == 2 and "t141" in p.stderr


def test_arms_sh_pid_matches_python_index(world):
    """The shell's awk pid list and arms.py's sorted arm list agree (index -> pid)."""
    awk = ("NR==1{for(i=1;i<=NF;i++){if($i==\"pid\")p=i; if($i==\"arm\")a=i}; next} "
           "$a!=\"base\"{print $p}")
    out = subprocess.run(f"awk -F'\\t' '{awk}' {world['prod']}/MANIFEST.tsv | LC_ALL=C sort -u",
                         shell=True, capture_output=True, text=True).stdout.split()
    assert out == world["arm_pids"]


def test_map_knobs_sh_dry_run_and_guard():
    p = bash("map_knobs.sh", "/kit", "/prod", "/y/t138/out", "/x/t141")
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == ("python3 /kit/tools/t141/map_knobs.py run /x/t141/arms1k "
                                "/y/t138/out/features.npz /prod/MANIFEST.tsv /y/t138/out "
                                "/x/t141/out --blacklist /project/def-maxwl/mforooz/EIC_REPRO/002/"
                                "scripts/hg38_blacklist_v2.bed")
    p = bash("map_knobs.sh", "/kit", "/prod", "/y/t138/out", "/x/t141", "/bl.bed")
    assert p.stdout.strip().endswith("--blacklist /bl.bed")
    p = bash("map_knobs.sh", "/kit", "/prod", "/y/t138/out", "/x/t137")
    assert p.returncode == 2


def test_scripts_parse_and_pin_resources():
    for s in ("arms.sh", "map_knobs.sh"):
        path = ROOT / "slurm" / "t141" / s
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        t = path.read_text()
        for line in ("--gres=none", "--exclude=c128,c166,c537,c371,c329", "--account=def-maxwl"):
            assert line in t, (s, line)
        assert "--export" not in t.replace("NO --export", "")
    assert "module load scipy-stack" in (ROOT / "slurm/t141/map_knobs.sh").read_text()
    assert "requirements-fir.txt" in (ROOT / "slurm/t141/arms.sh").read_text()


# ---------------------------------------------------------------- t142: read-thinned null

DEPTH = "C19M16__depth__15M"


def _null_files(world, d, ratio=0.5, rng_override=None, spoil=None):
    """Stand-in t142 null files: any subset of the base (here a fixed binomial draw)."""
    rows = arms.read_manifest(world["prod"] / "MANIFEST.tsv")
    pid_index = [r["pid"] for r in rows].index(DEPTH)
    (d / DEPTH).mkdir(parents=True, exist_ok=True)
    with np.load(world["prod"] / "C19M16__base__base" / "counts25.npz") as b:
        base = {c: b[c] for c in b.files if c != "chrM"}
    files = {}
    for s in (0, 1):
        g = np.random.default_rng([99, s])
        arr = {c: g.binomial(x.astype(np.int64), ratio).astype(np.uint32) for c, x in base.items()}
        if spoil is not None:
            spoil(arr)
        np.savez(d / DEPTH / f"null_s{s}.npz", **arr)
        rng = rng_override or [s, pid_index, 0]
        (d / DEPTH / f"null_s{s}.json").write_text(json.dumps({"ratio": ratio, "rng": rng}))
        files[s] = arr
    return files, [r["pid"] for r in arms.arm_rows(rows)].index(DEPTH)


def test_read_null_is_read_from_files(world, tmp_path):
    files, idx = _null_files(world, tmp_path / "thinned")
    assert arms.main(["arm", str(world["prod"]), str(tmp_path / "t141"), str(idx),
                      str(tmp_path / "thinned")]) == 0
    z = np.load(tmp_path / "t141" / "arms1k" / f"{DEPTH}.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["null"]["thinning"] == "read" and meta["null"]["formula"] == arms.NULL_READ
    zu = np.load(world["out"] / "arms1k" / f"{DEPTH}.npz")
    assert json.loads(str(zu["meta"]))["null"]["thinning"] == "unit"
    for s in (0, 1):
        for c in CHROMS:
            np.testing.assert_array_equal(z[f"counts_null{s}_{c}"],
                                          files[s][c][:N1K * 40].reshape(-1, 40).sum(1))
    for k in zu.files:                       # everything but the null is unchanged
        if k != "meta" and "null" not in k:
            np.testing.assert_array_equal(z[k], zu[k])


@pytest.mark.parametrize("defect", ["ratio", "rng", "above"])
def test_read_null_refuses_mismatch(world, tmp_path, defect):
    kw = {}
    if defect == "ratio":
        kw["ratio"] = 0.25
    elif defect == "rng":
        kw["rng_override"] = [0, 0, 0]
    else:
        def spoil(arr):
            arr["chr1"] = arr["chr1"].copy()
            arr["chr1"][3] += 10_000
        kw["spoil"] = spoil
    _, idx = _null_files(world, tmp_path / "thinned", **kw)
    assert arms.arm(world["prod"], tmp_path / "t141", idx, tmp_path / "thinned") == 3
    assert not (tmp_path / "t141" / "arms1k" / f"{DEPTH}.npz").exists()
