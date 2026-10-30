"""t136 — the arm-to-arm grid tables (`tools/t136/pairs.py`) and the two Nibi job scripts.

  * the table: 2 arms (Darm1, Darm2) x the 7 per-track g x seeds 0, 1, 2; anchors 0 / 21; SMOKE =
    the C19M16 seed-0 rows; no name overlaps phase 2's reference runs;
  * the pairs on the t112 manifest: Darm1 adds the 2 ordered depth 15M <-> 3.75M pairs per track,
    Darm2 every arm->arm pair without the held-out 7.5M product (counts computed from
    `usable_products`), no held-out pid in any training pair, every held-out score pair has a 7.5M
    pid; the live t118 tables are untouched;
  * the scripts: #SBATCH lines, the t136 out-dir guard and DRY_RUN command lines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as lp  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t128p3 import pairs as p3  # noqa: E402
from t136 import pairs as tp  # noqa: E402

SLURM = REPO / "slurm" / "t136"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
ROWS = lp.read_manifest(FIXTURE_MANIFEST)
SCRIPTS = {"train": SLURM / "train.sh", "law": SLURM / "law.sh"}
G0 = tp.G_IDS[0]
ANCHORS = {0: f"Darm1_{G0}_counts_real_s0", 21: f"Darm2_{G0}_counts_real_s0"}
OUT = "/project/def-maxwl/mforooz/t136"


# ---- the table -------------------------------------------------------------------------------

def test_constants_and_settings():
    assert tp.G_IDS == lp.TRACKS and len(tp.G_IDS) == 7 and "all" not in tp.G_IDS
    assert tp.HOLDOUT == (("depth", "7.5M"),) and tp.SEEDS == (0, 1, 2) and tp.SPACE == "counts"
    assert tp.ARMS == ("Darm1", "Darm2")
    d = p3.SETTING_BY_ARM_3B["D"]          # phase 2's D recipe
    for s in tp.SETTINGS:
        assert {k: s[k] for k in d if k not in ("number", "arm")} == \
            {k: d[k] for k in d if k not in ("number", "arm")}
    assert tp.ARM_ARRAYS == {"Darm1": "0-20", "Darm2": "21-41"}
    c = tp.G_IDS.index("C19M16")
    assert tp.SMOKE == (3 * c, 21 + 3 * c)


def test_task_table():
    t = tp.tasks(ROWS)
    assert len(t) == 42 == len({r["run_name"] for r in t})
    assert [r["index"] for r in t] == list(range(42))
    assert all(list(r) == list(tp.TASK_KEYS) for r in t)
    assert {i: t[i]["run_name"] for i in ANCHORS} == ANCHORS
    assert [t[i]["run_name"] for i in tp.SMOKE] == ["Darm1_C19M16_counts_real_s0",
                                                    "Darm2_C19M16_counts_real_s0"]
    for ai, s in enumerate(tp.SETTINGS):
        for gi, g in enumerate(tp.G_IDS):
            for seed in tp.SEEDS:
                r = t[ai * 21 + gi * 3 + seed]
                assert (r["rung"], r["g"], r["seed"]) == (s["arm"], g, seed)
                assert r["form_rung"] == "D" and r["model"] == "real" and r["space"] == "counts"
                assert (r["g_hidden"], r["f_channels"]) == (64, 32)
                assert r["run_name"] == ltrain.run_name(s["arm"], g, "counts", "real", seed)
                assert r["n_extra_pairs"] == len(tp.extra_pairs(ROWS, g, s["arm"]))
    assert all(r["n_extra_pairs"] is None for r in tp.tasks())
    names = {r["run_name"] for r in t}
    assert not names & set(p3.REFERENCE_RUNS)


def test_cli_tasks_tsv():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t136" / "pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 43 and lines[0].split("\t") == list(tp.TASK_KEYS)
    assert [lines[1 + i].split("\t")[7] for i in ANCHORS] == list(ANCHORS.values())


# ---- the pairs -------------------------------------------------------------------------------

def _non_base_kept(g):
    """Usable non-base, non-held-out products of track g, counted from the manifest."""
    return [r for r in lp.usable_products(ROWS) if r["track"] == g and r["arm"] != "base"
            and (r["arm"], r["level"]) not in set(tp.HOLDOUT)]


@pytest.mark.parametrize("g", tp.G_IDS)
def test_extra_pair_counts(g):
    d1 = tp.extra_pairs(ROWS, g, "Darm1")
    assert len(d1) == 2
    assert {((p["arm_src"], p["level_src"]), (p["arm_tgt"], p["level_tgt"])) for p in d1} == {
        (("depth", "15M"), ("depth", "3.75M")), (("depth", "3.75M"), ("depth", "15M"))}
    n = len(_non_base_kept(g))
    d2 = tp.extra_pairs(ROWS, g, "Darm2")
    assert len(d2) == n * (n - 1)
    assert {(p["source_pid"], p["target_pid"]) for p in d1} <= \
        {(p["source_pid"], p["target_pid"]) for p in d2}
    for arm in tp.ARMS:
        e = tp.extra_pairs(ROWS, g, arm)
        assert all(p["direction"] == "arm_to_arm" and p["track"] == g for p in e)
        tr = tp.train_pairs(ROWS, g, arm)
        assert tr == lp.train_pairs(ROWS, g, tp.HOLDOUT) + e
        assert not any(lp._held(p, tp.HOLDOUT) for p in tr)
    held = tp.heldout_score_pairs(ROWS, g)
    h75 = {r["pid"] for r in lp.usable_products(ROWS) if r["track"] == g
           and (r["arm"], r["level"]) == ("depth", "7.5M")}
    assert len(h75) == 1 and held
    assert all({p["source_pid"], p["target_pid"]} & h75 for p in held)
    assert len(held) == 2 * n


def test_pinned_counts():
    """The counts G14 reads back (histone 18 kept arms, DNase 6 after the mapq defects)."""
    assert {g: len(tp.extra_pairs(ROWS, g, "Darm2")) for g in tp.G_IDS} == {
        g: (30 if g == "C12M02" else 306) for g in tp.G_IDS}


def test_bad_arm_and_all():
    with pytest.raises(ValueError):
        tp.extra_pairs(ROWS, G0, "D")
    with pytest.raises(ValueError):
        tp.extra_pairs(ROWS, "all", "Darm1")


@pytest.mark.parametrize("args,ok", [(["train-pairs", G0, "Darm1"], 38),
                                     (["extra-pairs", G0, "Darm1"], 2),
                                     (["heldout-pairs", G0], 36)])
def test_cli_pairs(args, ok):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t136" / "pairs.py"), args[0],
                        str(FIXTURE_MANIFEST), *args[1:]], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0].split("\t") == list(lp.PAIR_KEYS) and len(lines) - 1 == ok


def test_live_t118_tables_untouched():
    assert len(lp.tasks(ROWS)) == 576 and len(p3.tasks_3b()) == 42
    assert len(lp.train_pairs(ROWS, "all")) == 242 and len(lp.law_pairs(ROWS, "all")) == 2094


# ---- scripts ---------------------------------------------------------------------------------

def _sbatch(name: str) -> list[str]:
    return [ln for ln in SCRIPTS[name].read_text().splitlines() if ln.startswith("#SBATCH")]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name,time,cpus,mem", [("train", "7:00:00", "2", "10000M"),
                                               ("law", "6:00:00", "8", "24000M")])
def test_sbatch_lines(name, time, cpus, mem):
    lines = _sbatch(name)
    assert f"#SBATCH --job-name=t136_{name}" in lines
    assert "#SBATCH --exclude=c128,c166,c537,c371,c329" in lines
    assert f"#SBATCH --cpus-per-task={cpus}" in lines and f"#SBATCH --mem={mem}" in lines
    assert "fc107" not in SCRIPTS[name].read_text() and "DATA_CANDI_MERGED" not in SCRIPTS[name].read_text()
    assert f"#SBATCH --time={time}" in lines and sum("--time" in ln for ln in lines) == 1
    assert "#SBATCH --account=def-maxwl" in lines
    assert any("gres" in ln for ln in lines) is (name == "train")
    if name == "train":
        assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in lines
    t = SCRIPTS[name].read_text()
    assert 'tools/t136/pairs.py" tasks "$MANIFEST"' in t and "t128p3/pairs.py" not in t
    assert "viabase.py" not in t
    assert "--export" not in t.replace("NO --export", "")


def _dry(name: str, args: list[str], index: int | None) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


HEAD = "/products/MANIFEST.tsv /kit/tools/t118/covariates.tsv /cache"


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_train(index):
    run = ANCHORS[index]
    r = _dry("train", ["/kit", "/products", "/cache", OUT, "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t136/train.py train-index {HEAD} {OUT}/runs {index}",
        f"python3 /kit/tools/t118/ladder/score.py trained {HEAD} /bl.bed {OUT}/runs/{run}"]


def test_dry_run_train_step_overrides():
    r = _dry("train", ["/kit", "/products", "/cache", OUT + "/smoke", "/bl.bed", "300", "100"],
             tp.SMOKE[0])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0] == (f"python3 /kit/tools/t136/train.py train-index {HEAD} "
                                        f"{OUT}/smoke/runs {tp.SMOKE[0]} --max-steps 300 "
                                        "--eval-every 100")


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_law(index):
    run = ANCHORS[index]
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", OUT], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t118/ladder/score.py law {HEAD} /bl.bed {OUT}/runs/{run} "
        "--workers 8 --split"]


REFUSED = ["/project/def-maxwl/mforooz/t128p3", "/project/def-maxwl/mforooz/t128p2",
           "/project/def-maxwl/mforooz/t128p2/viabase", "/project/def-maxwl/mforooz/t135",
           "/x/t136x", "/x/t1360/runs", "/x/t13"]
ACCEPTED = [OUT, OUT + "/smoke"]


@pytest.mark.parametrize("out", REFUSED + ACCEPTED)
def test_out_dir_guard(out):
    ok = out in ACCEPTED
    for r in (_dry("train", ["/kit", "/p", "/c", out, "/bl.bed"], 0),
              _dry("law", ["/kit", "/p", "/c", "/bl.bed", out], 0)):
        assert (r.returncode == 0) is ok, (out, r.returncode, r.stderr)
        if not ok:
            assert r.returncode == 2 and r.stdout == ""


def test_an_index_outside_the_table_fails():
    assert _dry("train", ["/kit", "/p", "/c", OUT, "/bl.bed"], 42).returncode == 2
    assert _dry("law", ["/kit", "/p", "/c", "/bl.bed", OUT], 42).returncode == 2
