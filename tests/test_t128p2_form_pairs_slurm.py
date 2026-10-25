"""t128p2-C1 — rung DSANC, the 4-arm x 96-row phase-2 task table and the four Fir job scripts.

  * DSANC: encoder = old D's NB head of X_base, middle = log1p of its NB mean; z = log1p(X + 1) at
    init; builds GSym (box False), n_theta 512, context 60; predictor path and checkpoint round trip;
  * the table: 96 rows, anchors 9/33/57/81, form_rung per arm, arm D's names equal
    `ladder.train.run_name` and `t128/viabase.VIABASE_RUNS`;
  * the scripts: #SBATCH lines, the t128p2 out-dir guard and, through DRY_RUN=1, the command lines
    and skip rules (the scripts cannot run here: no SLURM, no modules).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding, model  # noqa: E402
from ladder import train as ladder_train  # noqa: E402
from ladder.fforms import form_ds, form_dsanc  # noqa: E402
from t128 import viabase  # noqa: E402
from t128p2 import pairs as tp  # noqa: E402

STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV
C = form_ds.CONTEXT
SLURM = REPO / "slurm" / "t128p2"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
SCRIPTS = {name: SLURM / f"{name}.sh" for name in ("train", "law", "viabase", "verify")}
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
EXCLUDE = "#SBATCH --exclude=fc10713,fc10715"
ANCHORS = {9: "DSfree_C19M16_counts_real_s0", 33: "DSfixed_C19M16_counts_real_s0",
           57: "DSanchor_C19M16_counts_real_s0", 81: "D_C19M16_counts_real_s0"}
RUN57 = ANCHORS[57]


# ---- rung DSANC ------------------------------------------------------------------------------

def _covs(k, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(k, N, generator=g)
    a = torch.randint(0, form_ds.N_ASSAY, (k,), generator=g)
    c[:, form_ds.ASSAY_SLICE] = 0.0
    c[torch.arange(k), form_ds.N_KNOB + a] = 1.0
    return c


def _x(k, w, seed):
    return torch.log1p(torch.poisson(torch.full((k, w), 3.0),
                                     generator=torch.Generator().manual_seed(seed)))


def _ladder(seed=0) -> model.Ladder:
    """A DSANC Ladder with random g last layer, head and C_clean (nothing trivially at init)."""
    torch.manual_seed(seed)
    lad = model.Ladder("DSANC", "counts", STATS)
    with torch.no_grad():
        lad.g.mlp[-1].weight.normal_(0.0, 0.3)
        lad.g.mlp[-1].bias.normal_(0.5, 0.3)
        lad.form.f.head.weight.normal_(0.0, 0.3)
        lad.form.f.head.bias.normal_(0.0, 0.3)
        lad.g.cclean.normal_(0.0, 1.0)
    return lad


def test_mu_bounds_equal_model():
    assert (form_dsanc.MU_MIN, form_dsanc.MU_MAX) == (model.MU_MIN, model.MU_MAX)


def test_dsanc_builds_gsym_unboxed():
    lad = model.Ladder("DSANC", "counts", STATS)
    assert isinstance(lad.form, form_dsanc.FormDSAnc) and isinstance(lad.form, form_ds.FormDS)
    assert lad.form.rung == "DSANC" and lad.form.g_kind == "sym_film"
    assert lad.form.cclean_box is False
    assert isinstance(lad.g, model.GSym) and lad.g.box is False
    assert lad.n_theta == 512 and lad.context == 60 and lad.per_bin is False


def test_middle_is_log1p_of_nb_mean_at_init():
    """At init loc = log1p X, so the NB mean is X + 1 and z = log1p(X + 1): NOT the identity."""
    lad = model.Ladder("DSANC", "counts", STATS)
    cov = torch.cat([_covs(4, 1), _covs(4, 2)], 1)
    th = lad.theta(cov)
    x = _x(4, 300, 0)
    loc, disp = lad.form.encode_nb(x, th[:, :256])
    assert torch.equal(loc, x[:, C // 2:-(C // 2)])
    z, d2 = lad.form.encode(x, th[:, :256])
    X = torch.expm1(x[:, C // 2:-(C // 2)])
    assert torch.allclose(z, torch.log1p(X + 1.0), atol=1e-5)
    assert torch.equal(d2, disp)                     # the NB log-size, not a log sigma
    out, _ = lad(cov, x)
    assert torch.allclose(out, torch.log1p(torch.expm1(x[:, C:-C]) + 1.0), atol=1e-5)


def test_encode_is_middle_of_encode_nb_and_forward_is_decode_of_middle():
    lad = _ladder(1)
    cov = torch.cat([_covs(3, 3), _covs(3, 4)], 1)
    th = lad.theta(cov)
    x = _x(3, 260, 1)
    loc, disp = lad.form.f(x, th[:, :256])
    assert torch.equal(lad.form.encode_nb(x, th[:, :256])[0], loc)
    z, d = lad.form.encode(x, th[:, :256])
    assert torch.equal(z, torch.log1p(torch.exp(loc).clamp(model.MU_MIN, model.MU_MAX)))
    assert torch.equal(d, disp)
    l2, d2 = lad.form.f(z, th[:, 256:])
    l3, d3 = lad(cov, x)
    assert torch.equal(l2, l3) and torch.equal(d2, d3)


def test_middle_clamps():
    z = form_dsanc.FormDSAnc.middle(torch.tensor([-100.0, 100.0]))
    assert torch.allclose(z, torch.log1p(torch.tensor([model.MU_MIN, model.MU_MAX])))


def test_dsanc_predict_chrom_matches_forward():
    lad = _ladder(4)
    c1, c2 = _covs(2, 13), _covs(2, 14)
    x1 = _x(2, 400, 3)
    loc, _ = lad(torch.cat([c1, c2], 1), x1)
    theta = lad.theta(torch.cat([c1[:1], c2[:1]], 1))[0]
    X = torch.expm1(x1[0]).numpy()
    lp, _ = model.predict_chrom(lad, X, theta, "cpu", chunk=100)
    assert np.allclose(lp[C:-C], loc[0].detach().numpy(), atol=1e-5)


def test_dsanc_checkpoint_round_trip(tmp_path):
    lad = _ladder(5)
    path = tmp_path / "ckpt.pt"
    torch.save({"g": lad.g.state_dict(), "form": lad.form.state_dict()}, path)
    ck = torch.load(path)
    new = model.Ladder("DSANC", "counts", STATS)
    new.g.load_state_dict(ck["g"])
    new.form.load_state_dict(ck["form"])
    cov = torch.cat([_covs(3, 15), _covs(3, 16)], 1)
    x = _x(3, 300, 4)
    a, b = lad(cov, x), new(cov, x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ds_rung_is_unchanged():
    lad = model.Ladder("DS", "counts", STATS)
    assert type(lad.form) is form_ds.FormDS
    x = _x(2, 300, 0)
    loc, _ = lad(torch.cat([_covs(2, 1), _covs(2, 2)], 1), x)
    assert torch.equal(loc, x[:, C:-C])


# ---- settings and the task table -------------------------------------------------------------

def test_settings():
    S = tp.SETTINGS
    assert [s["number"] for s in S] == [1, 2, 3, 4]
    assert tp.ARMS == ("DSfree", "DSfixed", "DSanchor", "D")
    keys = ["number", "arm", "rung", "cclean", "loss", "lam", "vae", "twin", "max_steps",
            "val_nll_rule"]
    assert all(list(s) == keys for s in S)
    got = [(s["rung"], s["cclean"], s["loss"], s["lam"], s["vae"], s["twin"], s["max_steps"],
            s["val_nll_rule"]) for s in S]
    assert got == [("DS", "A", "L12", 0.0, False, False, 12000, "L1+L2 NLL"),
                   ("DS", "F", "L12", 0.0, False, False, 12000, "L1+L2 NLL"),
                   ("DSANC", "F", "L12A", 0.0, False, False, 12000, "L1+L2+anchor NLL"),
                   ("D", None, "nb", 0.0, False, False, 12000, "t118 cross NLL")]
    assert tp.SETTING_BY_ARM == {s["arm"]: s for s in S}
    assert not any("_" in a for a in tp.ARMS), "run names split on _"


def test_constants():
    assert tp.HOLDOUT == (("depth", "7.5M"),)
    assert tp.SEEDS == (0, 1, 2) and tp.SPACE == "counts" and tp.MAX_STEPS == 12000
    assert tp.G_IDS == tp._lp.G_IDS and len(tp.G_IDS) == 8 and tp.G_IDS[3] == "C19M16"
    assert tp.TASK_KEYS == ("index", "rung", "form_rung", "g", "space", "model", "seed",
                            "run_name", "setting")
    assert tp.SMOKE == (9, 33, 57, 81) and tp.FULL_ARRAY == "0-95"


def test_task_table():
    t = tp.tasks()
    assert len(t) == 96 == len({r["run_name"] for r in t})
    assert [r["index"] for r in t] == list(range(96))
    assert {i: t[i]["run_name"] for i in ANCHORS} == ANCHORS
    assert all(list(r) == list(tp.TASK_KEYS) for r in t)
    form = {"DSfree": "DS", "DSfixed": "DS", "DSanchor": "DSANC", "D": "D"}
    for ai, arm in enumerate(tp.ARMS):
        for gi, g in enumerate(tp.G_IDS):
            for seed in tp.SEEDS:
                r = t[ai * 24 + gi * 3 + seed]
                assert (r["rung"], r["g"], r["seed"]) == (arm, g, seed)
                assert r["form_rung"] == form[arm] and r["setting"] == ai + 1
                assert r["model"] == "real" and r["space"] == "counts"
                assert r["run_name"] == f"{arm}_{g}_counts_real_s{seed}"
    assert tp.tasks(tp._lp.read_manifest(FIXTURE_MANIFEST)) == t


def test_d_run_names_match_t118_and_viabase():
    d = [r for r in tp.tasks() if r["rung"] == "D"]
    assert len(d) == 24
    for r, v in zip(d, viabase.VIABASE_RUNS):
        assert r["run_name"] == ladder_train.run_name("D", r["g"], "counts", "real", r["seed"])
        assert r["run_name"] == v["run_name"]


def test_cli_tasks_tsv():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t128p2" / "pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 97 and lines[0].split("\t") == list(tp.TASK_KEYS)
    rows = [ln.split("\t") for ln in lines[1:]]
    ri, fi = tp.TASK_KEYS.index("run_name"), tp.TASK_KEYS.index("form_rung")
    assert {int(c[0]): c[ri] for c in rows if int(c[0]) in ANCHORS} == ANCHORS
    assert {int(c[0]): c[fi] for c in rows if int(c[0]) in ANCHORS} == {
        9: "DS", 33: "DS", 57: "DSANC", 81: "D"}


def test_cli_pairs_hold_out_7p5m():
    py, tool = sys.executable, str(REPO / "tools" / "t128p2" / "pairs.py")
    tr = subprocess.run([py, tool, "train-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    ho = subprocess.run([py, tool, "heldout-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    assert tr.returncode == 0 and ho.returncode == 0, tr.stderr + ho.stderr
    assert "7.5M" not in tr.stdout
    held = ho.stdout.splitlines()[1:]
    assert held and all("7.5M" in ln for ln in held)


# ---- script text -----------------------------------------------------------------------------

def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_gres_only_on_train():
    assert [ln for ln in _text("train").splitlines() if "gres" in ln] == [GRES]
    for name in ("law", "viabase", "verify"):
        assert not any("gres" in ln for ln in _sbatch(name))


@pytest.mark.parametrize("name,time,cpus,mem", [("train", "12:00:00", 4, "16000M"),
                                                ("law", "6:00:00", 8, "24000M"),
                                                ("viabase", "4:00:00", 8, "24000M"),
                                                ("verify", "3:00:00", 8, "32000M")])
def test_sbatch_lines(name, time, cpus, mem):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert f"#SBATCH --job-name=t128p2_{name}" in lines
    assert len(re.findall(r"job-name=", _text(name))) == 1
    assert EXCLUDE in lines and sum("--exclude" in ln for ln in lines) == 1
    assert f"#SBATCH --time={time}" in lines
    assert f"#SBATCH --cpus-per-task={cpus}" in lines and f"#SBATCH --mem={mem}" in lines
    assert not any("--partition" in ln or "--export" in ln for ln in lines)
    outs = [ln for ln in lines if "--output" in ln or "--error" in ln]
    if name == "verify":
        assert outs == ["#SBATCH --output=/project/def-maxwl/mforooz/t128p2/verify/%x_%j.out"]
    else:
        assert outs == [], "the submitter gives the log path"


@pytest.mark.parametrize("name", SCRIPTS)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert ("set -euo pipefail" if name != "verify" else "set -uo pipefail") in t
    assert "--export=" not in t
    assert "afterok:" not in t and "/scratch" not in t
    assert "c128" not in t and "EIC_REPRO" not in t, "no Nibi node names or Nibi default paths"


@pytest.mark.parametrize("name", ("train", "law", "viabase"))
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;'):
        assert piece in t, (name, piece)
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


def test_verify_runs_the_four_sets_on_its_own_kit():
    t = _text("verify")
    assert 'K="${1:?' in t
    assert 'export PYTHONPATH="$K/src"' in t and '"$K"/src/*) ;;' in t
    assert "for set in t128p2 t128 t126 t118; do" in t
    assert "tests/test_${set}_*.py" in t
    assert 'echo "RESULT $set rc=$RC' in t
    assert t.index("bin/activate") < t.index("python3 -m pytest")


def test_headers_document_smoke_full_and_aftercorr():
    t = _text("train")
    for piece in ("--array=9,33,57,81%4", "--array=0-95%24", "/smoke", "300 100",
                  "--time=2:00:00", "--dependency=aftercorr:$TRAIN", "LIVE"):
        assert piece in t, piece
    lw = _text("law")
    assert "--array=9,33,57,81%4" in lw and "--array=0-95%24" in lw
    assert "--dependency=aftercorr:$TRAIN" in lw and "LIVE" in lw
    vb = _text("viabase")
    assert "RUNS=/project/def-maxwl/mforooz/t128p2/runs" in vb
    assert "--array=9 " in vb and "--array=0-23%12" in vb


# ---- DRY_RUN ---------------------------------------------------------------------------------

def _dry(name: str, args: list[str], index: int | None = 57) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


TRAIN_ARGS = ["/kit", "/products", "/cache", "/out/t128p2", "/bl.bed"]
TRAIN_LINES = [
    "python3 /kit/tools/t128p2/train.py train-index /products/MANIFEST.tsv "
    "/kit/tools/t118/covariates.tsv /cache /out/t128p2/runs 57",
    "python3 /kit/tools/t118/ladder/score.py trained /products/MANIFEST.tsv "
    f"/kit/tools/t118/covariates.tsv /cache /bl.bed /out/t128p2/runs/{RUN57} --workers 4"]


def test_dry_run_train_prints_the_two_pinned_lines_for_task_57():
    r = _dry("train", TRAIN_ARGS)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == TRAIN_LINES


def test_dry_run_train_step_overrides():
    r = _dry("train", TRAIN_ARGS + ["300", "100"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [TRAIN_LINES[0] + " --max-steps 300 --eval-every 100",
                                     TRAIN_LINES[1]]
    r = _dry("train", TRAIN_ARGS + ["300"])
    assert r.stdout.splitlines()[0] == TRAIN_LINES[0] + " --max-steps 300"
    assert _dry("train", TRAIN_ARGS + ["3x"]).returncode == 2


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_train_maps_every_smoke_index(index):
    r = _dry("train", ["/kit", "/p", "/c", "/out/t128p2", "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-1].endswith(f"/bl.bed /out/t128p2/runs/{ANCHORS[index]} --workers 4")


def test_dry_run_train_skips_a_scored_run(tmp_path):
    out = tmp_path / "t128p2"
    run = out / "runs" / RUN57
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/p", "/c", str(out), "/bl.bed"])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_train_blacklist_is_required():
    assert _dry("train", ["/kit", "/p", "/c", "/out/t128p2"]).returncode != 0
    assert "DEFAULT_BLACKLIST" not in _text("train")


def _law(out, index):
    return _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(out)], index)


def _law_lines(out: str, run: str) -> list[str]:
    tail = f"/products/MANIFEST.tsv /kit/tools/t118/covariates.tsv /cache /bl.bed {out}/runs/{run} --workers 8"
    return [f"python3 /kit/tools/t118/ladder/score.py law {tail} --split",
            f"python3 /kit/tools/t128/latent.py run {tail}",
            f"python3 /kit/tools/t128p2/hop1.py run {tail}"]


@pytest.mark.parametrize("index,n", [(81, 1), (9, 2), (57, 3)])
def test_dry_run_law_steps_per_arm(index, n):
    r = _law("/out/t128p2", index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _law_lines("/out/t128p2", ANCHORS[index])[:n]
    assert "exit 3" in r.stderr   # no SCORE_DONE: a real task would fail loudly


@pytest.mark.parametrize("index,marks", [(81, ["LAW_DONE"]), (9, ["LAW_DONE", "LATENT_DONE"]),
                                         (57, ["LAW_DONE", "LATENT_DONE", "HOP1_DONE"])])
def test_dry_run_law_skip_rules(tmp_path, index, marks):
    out = tmp_path / "t128p2"
    run = out / "runs" / ANCHORS[index]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    want = _law_lines(str(out), ANCHORS[index])[:len(marks)]
    r = _law(out, index)
    assert r.returncode == 0 and r.stdout.splitlines() == want and "exit 3" not in r.stderr
    for k, m in enumerate(marks[:-1]):
        (run / m).touch()
        r = _law(out, index)
        assert r.returncode == 0 and r.stdout.splitlines() == want[k + 1:]
    (run / marks[-1]).touch()
    r = _law(out, index)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_law_d_ignores_latent_marks(tmp_path):
    """Arm D needs LAW_DONE only: with LAW_DONE it is done, without it the law step runs."""
    out = tmp_path / "t128p2"
    run = out / "runs" / ANCHORS[81]
    run.mkdir(parents=True)
    (run / "LATENT_DONE").touch()
    assert len(_law(out, 81).stdout.splitlines()) == 1
    (run / "LAW_DONE").touch()
    assert _law(out, 81).stdout == ""


def test_dry_run_viabase(tmp_path):
    r = _dry("viabase", ["/kit", "/p", "/c", "/bl.bed", "/r", "/out/t128p2"], 9)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t128/viabase.py run /p/MANIFEST.tsv /kit/tools/t118/covariates.tsv "
        "/c /bl.bed /r/D_C19M16_counts_real_s0 /out/t128p2/viabase/D_C19M16_counts_real_s0 --workers 8"]
    assert "exit 3" in r.stderr
    out = tmp_path / "t128p2"
    (out / "viabase" / ANCHORS[81]).mkdir(parents=True)
    (out / "viabase" / ANCHORS[81] / "LAW_DONE").touch()
    r = _dry("viabase", ["/kit", "/p", "/c", "/bl.bed", "/r", str(out)], 9)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


REFUSED = ["/project/def-maxwl/mforooz/t128", "/project/def-maxwl/mforooz/t128/runs",
           "/x/t126/D", "/project/def-maxwl/mforooz/t118/ladder", "/project/def-maxwl/mforooz/t125",
           "/x/t128p2x", "/x/out"]
ACCEPTED = ["/project/def-maxwl/mforooz/t128p2", "/project/def-maxwl/mforooz/t128p2/smoke"]


@pytest.mark.parametrize("out", REFUSED + ACCEPTED)
def test_out_dir_guard(out):
    ok = out in ACCEPTED
    for r in (_dry("train", ["/kit", "/p", "/c", out, "/bl.bed"]),
              _dry("law", ["/kit", "/p", "/c", "/bl.bed", out]),
              _dry("viabase", ["/kit", "/p", "/c", "/bl.bed", "/r", out], 9)):
        assert (r.returncode == 0) is ok, (out, r.returncode, r.stderr)
        if not ok:
            assert r.returncode == 2 and r.stdout == ""


@pytest.mark.parametrize("name,args", [("train", TRAIN_ARGS),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o/t128p2"]),
                                       ("viabase", ["/kit", "/p", "/c", "/b", "/r", "/o/t128p2"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


def test_an_index_outside_the_table_fails():
    assert _dry("train", TRAIN_ARGS, 96).returncode == 2
    assert _dry("law", ["/kit", "/p", "/c", "/b", "/o/t128p2"], 96).returncode == 2
    assert _dry("viabase", ["/kit", "/p", "/c", "/b", "/r", "/o/t128p2"], 24).returncode == 2


def test_missing_positionals_fail():
    assert _dry("law", ["/kit", "/p", "/c", "/o/t128p2"]).returncode != 0
    assert _dry("viabase", ["/kit", "/p", "/c", "/b", "/o/t128p2"], 9).returncode != 0
