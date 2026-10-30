"""t142 — `tools/t142/readthin.py` (whole-read thinning) and `tools/t142/compare_real.py`, on
synthetic tagAligns written to a tmp dir. Nothing is read from the cluster.

The counting rule is checked against `tools/t112/bin25.py::counts_from_tagalign`, the function t117
called to make every `pr{1,2}.counts25.npz`.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("t142_readthin", ROOT / "tools" / "t142" / "readthin.py")
cr = _load("t142_compare_real", ROOT / "tools" / "t142" / "compare_real.py")
bin25 = _load("t112_bin25_for_t142", ROOT / "tools" / "t112" / "bin25.py")

SIZES = {"chr19": 40_013, "chr21": 30_007}     # not multiples of 25: a partial last bin


def write_ta(path, lines):
    with gzip.open(path, "wt") as fh:
        for c, s, e in lines:
            fh.write(f"{c}\t{s}\t{e}\tN\t1000\t+\n")


def random_lines(rng, n, length=101, extra=True):
    out = []
    names = list(SIZES)
    for _ in range(n):
        c = names[rng.integers(len(names))]
        s = int(rng.integers(0, SIZES[c] + 50))          # some overhang the chromosome end
        out.append((c, s, s + length))
    if extra:
        out += [("chrUn_KI270438v1", 100, 201), ("chrM", 5, 106), ("chr19", 0, 25),
                ("chr19", 24, 25), ("chr19", 25, 50), ("chr21", SIZES["chr21"] - 1, SIZES["chr21"] + 100)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


@pytest.fixture()
def ta(tmp_path):
    p = tmp_path / "x.tagAlign.gz"
    write_ta(p, random_lines(np.random.default_rng(1), 3000))
    return p


def _sizes_file(tmp_path):
    f = tmp_path / "chrom.sizes"
    f.write_text("".join(f"{c}\t{n}\n" for c, n in SIZES.items()) + "chrM\t16569\n")
    return f


# --- the counting rule --------------------------------------------------------------------------

def test_count_reproduces_t117_rule_bit_exact(ta, tmp_path):
    sizes = bin25.load_chrsz(_sizes_file(tmp_path))
    want = bin25.counts_from_tagalign(ta, sizes)
    for chunk in (4_000_000, 7, 1000):
        got = rt.count(ta, sizes, chunksize=chunk)
        assert list(got) == list(want)
        for c in want:
            assert got[c].dtype == np.uint32
            np.testing.assert_array_equal(got[c], want[c])


def test_like_npz_grid_equals_size_table(ta, tmp_path):
    sizes = bin25.load_chrsz(_sizes_file(tmp_path))
    ref = tmp_path / "ref.npz"
    bin25.write_npz(ref, bin25.counts_from_tagalign(ta, sizes))
    got = rt.count(ta, rt.sizes_like(ref))
    with np.load(ref) as z:
        assert list(got) == list(z.files)
        for c in z.files:
            np.testing.assert_array_equal(got[c], z[c])


def test_cli_count_and_thin(ta, tmp_path):
    sz = _sizes_file(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    py = [sys.executable, str(ROOT / "tools" / "t142" / "readthin.py")]
    p = subprocess.run(py + ["count", str(ta), str(tmp_path / "c.npz"), "--chrsz", str(sz)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    want = bin25.counts_from_tagalign(ta, bin25.load_chrsz(sz))
    with np.load(tmp_path / "c.npz") as z:
        assert all((z[c] == want[c]).all() for c in want)
    p = subprocess.run(py + ["thin", str(ta), "0.5", "1", "4", "2", str(tmp_path / "t.npz"),
                             "--like", str(tmp_path / "c.npz")], capture_output=True, text=True,
                       env=env)
    assert p.returncode == 0, p.stderr
    side = json.loads((tmp_path / "t.json").read_text())
    assert side["rng"] == [1, 4, 2] and side["n_lines"] == 3006
    got, _ = rt.thin([ta], 0.5, np.random.default_rng([1, 4, 2]), bin25.load_chrsz(sz))
    with np.load(tmp_path / "t.npz") as z:
        assert all((z[c] == got[c]).all() for c in got)
    assert not list(tmp_path.glob("*.tmp*"))


# --- the draw -----------------------------------------------------------------------------------

def test_thin_keeps_lines_by_rng_convention(ta, tmp_path):
    """The kept lines are exactly `default_rng([seed, pid, depth]).random(n) < r`, in file order,
    and the counts are the t117 count of those lines; chunking does not change the draw."""
    lines = [ln.split("\t") for ln in gzip.open(ta, "rt").read().splitlines()]
    u = np.random.default_rng([0, 3, 1]).random(len(lines))
    kept = tmp_path / "kept.tagAlign.gz"
    write_ta(kept, [(c, int(s), int(e)) for (c, s, e, *_), k in zip(lines, u < 0.25) if k])
    want = bin25.counts_from_tagalign(kept, SIZES)
    for chunk in (4_000_000, 13):
        got, st = rt.thin([ta], 0.25, rt.rng_for(0, 3, 1), SIZES, chunksize=chunk)
        assert st["kept_lines"] == int((u < 0.25).sum())
        for c in want:
            np.testing.assert_array_equal(got[c], want[c])


def test_union_continues_one_rng(tmp_path):
    rng = np.random.default_rng(5)
    a, b = random_lines(rng, 500, extra=False), random_lines(rng, 400, extra=False)
    write_ta(tmp_path / "a.gz", a)
    write_ta(tmp_path / "b.gz", b)
    write_ta(tmp_path / "ab.gz", a + b)
    got, _ = rt.thin([tmp_path / "a.gz", tmp_path / "b.gz"], 0.3, rt.rng_for(1, 2, 0), SIZES)
    want, _ = rt.thin([tmp_path / "ab.gz"], 0.3, rt.rng_for(1, 2, 0), SIZES)
    for c in want:
        np.testing.assert_array_equal(got[c], want[c])


def _isolated(n, length, gap=1000, chrom="chr19"):
    """n reads each alone in its own window: read i covers [i * gap, i * gap + length)."""
    return [(chrom, i * gap, i * gap + length) for i in range(n)]


def test_whole_reads_drop_together(tmp_path):
    n, L, gap = 35, 101, 1000
    write_ta(tmp_path / "iso.gz", _isolated(n, L, gap))
    sizes = {"chr19": n * gap + 25}
    full = rt.count(tmp_path / "iso.gz", sizes)["chr19"]
    seen = set()
    for seed in range(6):
        t = rt.thin([tmp_path / "iso.gz"], 0.5, rt.rng_for(seed, 0, 1), sizes)[0]["chr19"]
        for i in range(n):
            w = slice(i * gap // 25, (i * gap + L) // 25 + 1)
            assert full[w].min() == 1 and full[w].size == 5          # 101 bp: 5 bins
            k = t[w]
            assert (k == 0).all() or (k == full[w]).all()             # all or nothing
            seen.add(int(k.sum()))
    assert seen == {0, 5}


def test_pairs_drop_together(tmp_path):
    """--pairs: two consecutive lines (the mates of a pair) are kept or dropped together."""
    gap, n = 1000, 40
    lines = []
    for i in range(n):
        lines += [("chr19", 2 * i * gap, 2 * i * gap + 76), ("chr21", (2 * i + 1) * gap,
                                                              (2 * i + 1) * gap + 76)]
    write_ta(tmp_path / "pe.gz", lines)
    sizes = {"chr19": 2 * n * gap + 25, "chr21": 2 * n * gap + 25}
    for chunk in (4_000_000, 6):
        t = rt.thin([tmp_path / "pe.gz"], 0.5, rt.rng_for(0, 1, 1), sizes, pairs=True,
                    chunksize=chunk)[0]
        m1 = np.array([t["chr19"][2 * i * gap // 25] for i in range(n)])
        m2 = np.array([t["chr21"][(2 * i + 1) * gap // 25] for i in range(n)])
        assert (m1 == m2).all() and 0 < m1.sum() < n
        keep = np.random.default_rng([0, 1, 1]).random(n) < 0.5
        assert (m1.astype(bool) == keep).all()
    with pytest.raises(ValueError):
        rt.thin([tmp_path / "pe.gz"], 0.5, rt.rng_for(0, 1, 1), sizes, pairs=True, chunksize=7)


def test_1kb_variance_read_is_k_times_unit(tmp_path):
    """Reads of k = 5 25 bp bins: the variance of 1 kb sums of the thinned track around r x full is
    ~k times larger under whole-read thinning than under per-bin binomial thinning (t137 `unit`)."""
    rng = np.random.default_rng(11)
    nb = 400_000                                       # 25 bp bins: 10 Mb, 10 000 windows of 1 kb
    starts = rng.integers(0, nb - 5, 200_000) * 25     # bin-aligned 100 bp reads: exactly 5 bins
    write_ta(tmp_path / "k5.gz", [("chr19", int(s), int(s) + 100) for s in starts])
    sizes = {"chr19": nb * 25}
    full = rt.count(tmp_path / "k5.gz", sizes)["chr19"].astype(np.int64)
    assert full.sum() == 5 * starts.size
    r = 0.25
    v_read, v_unit = [], []
    for seed in range(3):
        t = rt.thin([tmp_path / "k5.gz"], r, rt.rng_for(seed, 0, 2), sizes)[0]["chr19"]
        u = np.random.default_rng([seed, 0, 2]).binomial(full, r)
        for x, v in ((t, v_read), (u, v_unit)):
            d = (x.astype(np.float64) - r * full).reshape(-1, 40).sum(1)
            v.append(d.var())
    ratio = np.mean(v_read) / np.mean(v_unit)
    # a read inside one window adds k^2 r(1-r) vs k r(1-r); reads crossing a window edge
    # (4 of every 40 starts) less: expected ~ 5 x (1 - 0.1 x 0.4) ~ 4.8
    assert 4.2 < ratio < 5.2, ratio


def test_constants_match_floor():
    src = (ROOT / "tools" / "t137" / "floor.py").read_text()
    assert "RATIOS = (1.0, 0.5, 0.25, 0.125)" in src and "SEEDS = (0, 1)" in src
    assert "BINS1K_RATIO, BINS1K_SEED = 0.25, 0" in src
    assert tuple(r for _, r in rt.FLOOR_RATIOS) == (0.5, 0.25, 0.125)
    assert tuple(d for d, _ in rt.FLOOR_RATIOS) == (1, 2, 3)      # floor.RATIOS.index(r)
    assert rt.BINS1K == (0.25, 0) and rt.SEEDS == (0, 1)


# --- product mode -------------------------------------------------------------------------------

PIDS = ("C19M16__base__base", "C19M16__depth__15M", "C19M16__pe__pe")


def make_world(root: Path, rng_seed: int = 0) -> dict:
    """A pseudoreps tree + products tree of 3 products of one track, each with a tagAlign split in
    two halves (pr1, pr2), and their t117 counts25 made by bin25."""
    rng = np.random.default_rng(rng_seed)
    pr, prod, ta = root / "pseudoreps", root / "products", root / "ta"
    ta.mkdir(parents=True)
    rows_pr, rows_prod, out = [], [], {}
    for pid, depth, n in ((PIDS[0], 30_000_000, 4000), (PIDS[1], 15_000_000, 2000),
                          (PIDS[2], 30_000_000, 3000)):
        pe = pid.endswith("pe__pe")
        lines = random_lines(rng, n, extra=False)
        halves = {"pr1": lines[: n // 2], "pr2": lines[n // 2:]}
        full = {}
        for h, ls in halves.items():
            f = ta / f"{pid}.{h}.tagAlign.gz"
            write_ta(f, ls)
            c = bin25.counts_from_tagalign(f, SIZES)
            bin25.write_npz(pr / pid / h / "counts25.npz", c)
            full[h] = c
            rows_pr.append({"pid": pid, "half": h, "n_lines": len(ls), "paired_end": str(pe),
                            "tagalign": str(f)})
        (prod / pid).mkdir(parents=True)
        bin25.write_npz(prod / pid / "counts25.npz",
                        {c: full["pr1"][c] + full["pr2"][c] for c in SIZES})
        arm, level = pid.split("__")[1:]
        rows_prod.append({"pid": pid, "track": "C19M16", "assay": "H3K27ac", "arm": arm,
                          "level": level, "depth": depth})
        out[pid] = full
    for path, rows in ((pr / "MANIFEST.tsv", rows_pr), (prod / "MANIFEST.tsv", rows_prod)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("t142")
    halves = make_world(root)
    for i in range(3):
        assert rt.product(root / "pseudoreps", root / "products", root / "thinned", i,
                          chunksize=500) == 0
    return root, halves


def test_product_ladder_files(world):
    root, halves = world
    for pid in PIDS:
        d = root / "thinned" / pid
        assert (d / "DONE").exists()
        names = {p.name for p in d.glob("floor_*.npz")}
        assert names == {f"floor_r{r}_s{s}_{h}.npz" for r in ("0.5", "0.25", "0.125")
                         for s in (0, 1) for h in ("pr1", "pr2")}
        meta = json.loads((d / "product.json").read_text())
        assert meta["bit_exact"] == {"pr1": "PASS", "pr2": "PASS"}
        assert meta["paired_end"] == pid.endswith("pe__pe")
        for name in names:
            with np.load(d / name) as z:
                want = list(SIZES) if name.startswith("floor_r0.25_s0_") else ["chr19", "chr21"]
                assert list(z.files) == want
                h = name[-7:-4]
                for c in z.files:
                    assert z[c].dtype == np.uint32 and (z[c] <= halves[pid][h][c]).all()
            side = json.loads((d / (name[:-4] + ".json")).read_text())
            assert side["rng"][1] == PIDS.index(pid)            # sorted pids: same order here
        assert not list(d.glob("*.tmp*"))


def test_product_ladder_follows_floor_order(world):
    """pr1 then pr2 from one rng per (ratio, seed): pr2's draw continues after pr1's lines."""
    root, _ = world
    pid = PIDS[0]
    ta = {h: root / "ta" / f"{pid}.{h}.tagAlign.gz" for h in ("pr1", "pr2")}
    rng = rt.rng_for(1, 0, 3)
    for h in ("pr1", "pr2"):
        got, _ = rt.thin([ta[h]], 0.125, rng, SIZES)
        with np.load(root / "thinned" / pid / f"floor_r0.125_s1_{h}.npz") as z:
            for c in z.files:
                np.testing.assert_array_equal(z[c], got[c])


def test_product_null_for_depth_arm_only(world):
    root, _ = world
    d = root / "thinned" / PIDS[1]
    meta = json.loads((d / "product.json").read_text())
    assert meta["null"]["ratio"] == 0.5 and meta["null"]["base_pid"] == PIDS[0]
    assert meta["null"]["bit_exact_base"] == "PASS"
    base_tas = [str(root / "ta" / f"{PIDS[0]}.{h}.tagAlign.gz") for h in ("pr1", "pr2")]
    assert meta["null"]["tagalign"] == base_tas
    arm_index = sorted(PIDS).index(PIDS[1])
    for s in (0, 1):
        side = json.loads((d / f"null_s{s}.json").read_text())
        assert side["rng"] == [s, arm_index, 0] and side["ratio"] == 0.5
        want, _ = rt.thin(base_tas, 0.5, rt.rng_for(s, arm_index, 0), SIZES)
        with np.load(d / f"null_s{s}.npz") as z:
            assert list(z.files) == list(SIZES)
            for c in z.files:
                np.testing.assert_array_equal(z[c], want[c])
    for pid in (PIDS[0], PIDS[2]):
        assert not list((root / "thinned" / pid).glob("null_*"))


def test_product_refuses_counts_that_do_not_reproduce(tmp_path):
    make_world(tmp_path, 3)
    f = tmp_path / "pseudoreps" / PIDS[0] / "pr2" / "counts25.npz"
    z = dict(np.load(f))
    z["chr21"] = z["chr21"].copy()
    z["chr21"][7] += 1
    np.savez(f, **z)
    assert rt.product(tmp_path / "pseudoreps", tmp_path / "products", tmp_path / "thinned", 0) == 3
    assert not (tmp_path / "thinned" / PIDS[0]).exists()


def test_product_done_skips(world):
    root, _ = world
    assert rt.product(root / "pseudoreps", root / "products", root / "thinned", 0) == 0


# --- compare_real -------------------------------------------------------------------------------

def _pairs_rows(mode):
    rows = []
    # base: 15M per half; depth arm 15M: 7.5M per half (ratio-1 rows = the real halves)
    for pid, arm, level, rph in (("C19M16__base__base", "base", "base", 15_000_000),
                                 ("C19M16__depth__15M", "depth", "15M", 7_500_000)):
        for ratio in (1.0, 0.5, 0.25, 0.125):
            for seed in ((0,) if ratio == 1.0 else (0, 1)):
                for d in ("pr1_to_pr2", "pr2_to_pr1"):
                    # real halves at 7.5M read 0.2; the read floor at 7.5M (base r0.5) too,
                    # the unit floor half that (too smooth)
                    shape = 0.2 * (0.5 / ratio) ** 0.5 if arm == "base" else 0.2 / ratio ** 0.5
                    if arm == "base" and ratio < 1 and mode == "unit":
                        shape = shape / 2.0
                    shape += 0.001 * seed
                    rows.append({"pid": pid, "track": "C19M16", "assay": "H3K27ac",
                                 "mark_class": "narrow", "arm": arm, "level": level,
                                 "thin_ratio": ratio, "reads_per_half": int(rph * ratio),
                                 "seed": seed, "direction": d, "shape_1k": shape,
                                 "abs_log2fc_top1": 0.01, "divergence_bin": 0.5,
                                 "spearman_fine": 0.9, "crps_all": 1.0, "crps_top1": 3.0})
    return rows


def _write_floor(d: Path, mode):
    d.mkdir(parents=True)
    rows = _pairs_rows(mode)
    with open(d / "floor_pairs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (d / "floor_curve.tsv").write_text("pid\tmetric\tvalue_7.5M\n")
    (d / "config.json").write_text(json.dumps({"thinning": mode}))


def test_compare_real_closes_gap(tmp_path):
    _write_floor(tmp_path / "read", "read")
    _write_floor(tmp_path / "unit", "unit")
    assert cr.main([str(tmp_path / "read"), str(tmp_path / "unit"), str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "compare_real.tsv"), delimiter="\t"))
    s = next(r for r in rows if r["metric"] == "shape_1k")
    assert int(s["reads_per_half"]) == 7_500_000 and s["source"] == "ladder"
    assert float(s["real"]) == pytest.approx(0.2)
    assert float(s["read"]) == pytest.approx(0.2005, rel=1e-9)
    assert float(s["unit"]) == pytest.approx(0.1005, rel=1e-9)
    assert float(s["real_over_read"]) == pytest.approx(0.2 / 0.2005, rel=1e-9)
    assert float(s["real_over_unit"]) == pytest.approx(0.2 / 0.1005, rel=1e-9)
    assert float(s["read_seed_spread"]) == pytest.approx(0.001, rel=1e-6)
    cls = list(csv.DictReader(open(tmp_path / "out" / "compare_real_class.tsv"), delimiter="\t"))
    assert {r["mark_class"] for r in cls} == {"narrow", "all"}


def test_compare_real_refuses_swapped_dirs(tmp_path):
    _write_floor(tmp_path / "read", "read")
    _write_floor(tmp_path / "unit", "unit")
    with pytest.raises(SystemExit):
        cr.compare(tmp_path / "unit", tmp_path / "read")


# --- slurm --------------------------------------------------------------------------------------

def _bash(script, *args, **env):
    return subprocess.run(["bash", str(ROOT / "slurm" / "t142" / script), *args],
                          capture_output=True, text=True, env={**os.environ, "DRY_RUN": "1", **env})


def test_slurm_dry_runs_and_guards():
    p = _bash("readthin.sh", "/k", "/pr", "/p", "/x/t142/thinned", SLURM_ARRAY_TASK_ID="3")
    assert p.stdout == "python3 /k/tools/t142/readthin.py product /pr /p /x/t142/thinned 3\n"
    assert _bash("readthin.sh", "/k", "/pr", "/p", "/x/t137/thinned",
                 SLURM_ARRAY_TASK_ID="3").returncode == 2
    p = _bash("floor.sh", "/k", "/pr", "/p", "/bl", "/x/t142/floor", "/x/t142/thinned",
              SLURM_ARRAY_TASK_ID="3")
    assert p.stdout == ("python3 /k/tools/t137/floor.py product /pr /p /bl /x/t142/floor 3 "
                        "--thinned /x/t142/thinned\n")
    assert _bash("floor.sh", "/k", "/pr", "/p", "/bl", "/x/t137/floor", "/x/t142/thinned",
                 SLURM_ARRAY_TASK_ID="3").returncode == 2
    p = _bash("floor_agg.sh", "/k", "/x/t142/floor", "/r")
    assert p.stdout == "python3 /k/tools/t137/floor.py aggregate /x/t142/floor /r\n"
    p = _bash("arms.sh", "/k", "/p", "/x/t142/t141", "/x/t142/thinned", SLURM_ARRAY_TASK_ID="4")
    assert p.stdout == "python3 /k/tools/t141/arms.py arm /p /x/t142/t141 4 /x/t142/thinned\n"
    assert _bash("arms.sh", "/k", "/p", "/x/t141", "/x/t142/thinned",
                 SLURM_ARRAY_TASK_ID="4").returncode == 2


def test_slurm_headers():
    for s in ("readthin.sh", "floor.sh", "floor_agg.sh", "arms.sh"):
        path = ROOT / "slurm" / "t142" / s
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        t = path.read_text()
        for line in ("#SBATCH --gres=none\n", "#SBATCH --exclude=c128,c166,c537,c371,c329\n",
                     "#SBATCH --account=def-maxwl\n"):
            assert line in t, (s, line)
        assert "--export" not in t.replace("NO --export", "")
    assert "module load scipy-stack" in (ROOT / "slurm/t142/readthin.sh").read_text()


def test_arms_sh_depth_index_list_matches_python(tmp_path):
    """The header's awk list of depth-arm indices equals arms.py's sorted arm list positions."""
    rows = [("A__base__base", "base"), ("A__depth__7.5M", "depth"), ("A__crop__36", "crop"),
            ("B__depth__15M", "depth"), ("B__base__base", "base"), ("A__depth__15M", "depth")]
    (tmp_path / "M.tsv").write_text("pid\tarm\n" + "".join(f"{p}\t{a}\n" for p, a in rows))
    cmd = ("awk -F'\\t' 'NR==1{for(i=1;i<=NF;i++){if($i==\"pid\")p=i; if($i==\"arm\")a=i}; next}"
           " $a!=\"base\"{print $p\"\\t\"$a}' " + str(tmp_path / "M.tsv")
           + " | LC_ALL=C sort -u | awk -F'\\t' '$2==\"depth\"{printf \"%s%d\", (n++ ? \",\" : \"\"), NR-1}'")
    got = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    arm_pids = sorted(p for p, a in rows if a != "base")
    want = ",".join(str(i) for i, p in enumerate(arm_pids) if dict(rows)[p] == "depth")
    assert got == want
    assert "$2==\"depth\"" in (ROOT / "slurm/t142/arms.sh").read_text()


# --- end to end: readthin product -> floor.py product --thinned ---------------------------------

def test_floor_reads_readthin_output(world, tmp_path):
    root, halves = world
    fl = _load("t137_floor_for_t142", ROOT / "tools" / "t137" / "floor.py")
    (tmp_path / "bl.bed").write_text("chr19\t100\t300\n")
    out = tmp_path / "t142" / "floor"
    assert fl.main(["product", str(root / "pseudoreps"), str(root / "products"),
                    str(tmp_path / "bl.bed"), str(out), "0", "--thinned",
                    str(root / "thinned")]) == 0
    pid = PIDS[0]
    assert (out / "pairs" / f"{pid}.DONE").read_text() == "14 rows thinning=read\n"
    z = np.load(out / "bins1k" / f"{pid}.npz")
    for h, k in (("pr1", "a"), ("pr2", "b")):
        with np.load(root / "thinned" / pid / f"floor_r0.25_s0_{h}.npz") as t:
            for c in SIZES:
                np.testing.assert_array_equal(z[f"{k}375_{c}"], fl.sums_1k(t[c]))
