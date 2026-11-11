"""EVt: the frozen proxy scorer of the evolution step (tools/onion_eval/auto_proxy.py).

One end-to-end pass on synthetic tables (40 windows, 2 seeds, --local, through the CLI so that
stdout is checked to be exactly one JSON object), then every refusal: forbidden and unknown
config keys, path values, model_file, the module's syntax tree, the parameter count, the wiring
check on a perturbed head, target independence, blind or non-validation eval sets, the budget cap
before and after an attempt, the trained-model gates in `collect`, a kit that differs, a tunnel
that is down, and a job that never ends. Every refusal must print `passed` false and exit non-zero.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import registry  # noqa: E402
from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import auto_proxy as AP  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

PY = sys.executable
SCRIPT = REPO / "tools" / "onion_eval" / "auto_proxy.py"
SH = REPO / "slurm" / "onion_eval" / "auto_proxy.sh"
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
BLIND = ((8, "H3K4me3"),)
L = 128
LOCAL = ["--n-boot", "20", "--guardrail-windows", "8"]


def _rung_cfg(r: int = 2, **over) -> dict:
    cfg = json.loads((REPO / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, **over)
    return cfg


def _recipe(d: Path, cfg: dict, src: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.py").write_text(src if src is not None else
                                (REPO / "tools/minicandi/rungs/conv_ae.py").read_text())
    return d


def _run_cli(args, env=None):
    p = subprocess.run([PY, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src"), **(env or {})},
                       cwd=str(REPO))
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    assert len(lines) == 1, f"stdout must be one JSON object, got {p.stdout!r}"
    return p.returncode, json.loads(lines[0])


def _main(args, capsys):
    rc = AP.main([str(a) for a in args])
    out = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert len(out) == 1, out
    return rc, json.loads(out[0])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("auto_proxy")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    sp = root / "eval" / "strata" / "masks.h5"
    sp.parent.mkdir(parents=True)
    sizes = json.loads((root / "genome" / "chrom_sizes.json").read_text())
    with h5py.File(sp, "w") as f:
        rng = np.random.default_rng(1)
        hi, lo = f.create_group("high"), f.create_group("low")
        for c, bp in sizes.items():
            m = rng.random(bp // 25) < 0.3
            hi.create_dataset(c, data=m)
            lo.create_dataset(c, data=~m)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, strata=sp, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    # keep scoring small (as tests/test_onion_eval_score.py): never-trained non-subset pairs only
    nt = [p for p in ev["correction_pairs"] if p["never_trained"] and not p["subset_pair"]
          and p["cell"] in ("s0", "s1", "s3", "s4")]
    acc = [x for x in ev["imputation_targets"] if x["regime"] == "accessibility_only"]
    ev["imputation_targets"] = [x for x in ev["imputation_targets"] if x not in acc] + [
        x for x in acc if x["cell"] in ("s0", "s3")]
    ev["correction_pairs"] = nt
    out.write_text(json.dumps(ev))
    blind_ev = root / "eval" / "synth" / "evalset_blind.json"
    E.build(t, blind_ev, strata=sp, n_regions=2, n_fit_regions=1, region_bins=256, seed=0,
            include_blind=True)
    return {"root": root, "tables": t, "evalset": out, "ev": ev, "blind_evalset": blind_ev}


@pytest.fixture(scope="module")
def e2e(env, tmp_path_factory):
    """The one full pass: rung 2 recipe (a verbatim copy of the rung module), 40 windows, 2 seeds."""
    d = tmp_path_factory.mktemp("e2e")
    rec = _recipe(d / "recipe", _rung_cfg(2))
    runs = d / "runs"
    rc, res = _run_cli(["--recipe", rec, "--local", "--tables", env["tables"], "--evalset",
                        env["evalset"], "--windows", 40, "--seeds", "0,1", "--runs-root", runs,
                        *LOCAL])
    return {"rc": rc, "res": res, "runs": runs, "recipe": rec}


# ------------------------------------------------------------------------- the passing attempt

def test_local_attempt_passes_and_prints_the_contract(e2e):
    rc, res = e2e["rc"], e2e["res"]
    assert rc == 0 and res["passed"] is True, res
    for k in ("composite", "imputation", "correction", "uncertainty"):
        assert isinstance(res[k]["value"], float) and np.isfinite(res[k]["value"]), (k, res)
    for k in ("imputation", "correction", "uncertainty"):
        assert 0.0 <= res[k]["value"] <= 1.0                     # clipped terms (ruling 18)
    assert len(res["seed_values"]) == 2
    assert res["composite"]["value"] == pytest.approx(np.mean(res["seed_values"]))
    assert res["composite"]["value"] == pytest.approx(
        0.5 * res["imputation"]["value"] + 0.25 * res["correction"]["value"]
        + 0.25 * res["uncertainty"]["value"])
    assert abs(res["param_count"] - 1e6) <= 1e5 and res["flops_per_window"] > 0
    assert res["cpu_hours"] > 0


def test_local_attempt_writes_tally_rows_and_validation_scores(e2e):
    tally = e2e["runs"] / "evolve" / "TALLY.tsv"
    rows = AP.parse_tally(tally.read_text())
    key = Path(e2e["res"]["attempt"]).name
    assert [r["seed"] for r in rows if r["attempt"] == key] == ["0", "1"]
    assert sum(r["cpu_hours"] for r in rows) == pytest.approx(e2e["res"]["cpu_hours"], abs=1e-3)
    for s in (0, 1):
        d = Path(e2e["res"]["attempt"]) / f"seed{s}"
        sc = json.loads((d / "scores.json").read_text())
        assert sc["split"] == "val"
        assert json.loads((d / "proxy_config.json").read_text())["windows_seen"] == 40
        assert json.loads((d / "run.json").read_text())["windows_seen"] >= 40


def test_all_six_rung_modules_pass_the_module_check_except_unet_ladder_import():
    for r, name in registry.RUNGS.items():
        probs = AP.check_module((REPO / f"tools/minicandi/rungs/{name}.py").read_text())
        if name == "unet":   # imports ladder.model.G directly; minicandi.components exports G
            assert probs and all("ladder.model" in p for p in probs), probs
        else:
            assert probs == [], (name, probs)


# ---------------------------------------------------------------------------- config refusals

@pytest.mark.parametrize("key", AP.FORBIDDEN_KEYS)
def test_forbidden_key_is_refused_through_the_cli(tmp_path, key):
    rec = _recipe(tmp_path / "r", _rung_cfg(2, **{key: True}))
    rc, res = _run_cli(["--recipe", rec, "--local", "--tables", tmp_path, "--evalset",
                        tmp_path / "e.json", "--runs-root", tmp_path])
    assert rc != 0 and res["passed"] is False and "forbidden config key" in res["reason"]
    assert res["composite"]["value"] is None


@pytest.mark.parametrize("over,needle", [
    ({"evalset": "x.json"}, "outside the allowed set"),
    ({"genome_dir": "g"}, "outside the allowed set"),
    ({"average_h5": "a.h5"}, "outside the allowed set"),
    ({"region_bins": 512}, "outside the allowed set"),
    ({"tables": "eic_v2"}, "outside the allowed set"),
    ({"count_transform": "../blind/x"}, "may not be a path"),
    ({"lr": [1, 2]}, "must be a scalar"),
    ({"windows_seen": 1.5}, "windows_seen must be an integer"),
])
def test_config_refusals(tmp_path, over, needle):
    with pytest.raises(AP.Reject, match=needle):
        AP.check_config(_rung_cfg(2, **over), tmp_path, local=True)


def test_model_file_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "ROOT", tmp_path)
    rec = tmp_path / "tools" / "minicandi" / "evolve" / "recipe1"
    rec.mkdir(parents=True)
    (rec / "model.py").write_text("")
    rel = "tools/minicandi/evolve/recipe1/model.py"
    assert AP.check_config(_rung_cfg(2, model_file=rel), rec, local=False) == \
        "tools/minicandi/evolve/recipe1"
    with pytest.raises(AP.Reject, match="no model_file"):
        AP.check_config(_rung_cfg(2), rec, local=False)
    with pytest.raises(AP.Reject, match="not the recipe's own model.py"):
        AP.check_config(_rung_cfg(2, model_file="tools/minicandi/rungs/conv_ae.py"), rec,
                        local=False)
    with pytest.raises(AP.Reject, match="relative to the repository root"):
        AP.check_config(_rung_cfg(2, model_file=str(rec / "model.py")), rec, local=False)
    out = tmp_path / "elsewhere"
    out.mkdir()
    (out / "model.py").write_text("")
    with pytest.raises(AP.Reject, match="relative to the repository root"):
        AP.check_config(_rung_cfg(2, model_file=str(out / "model.py")), out, local=False)


# ---------------------------------------------------------------------------- module refusals

HEAD = "import torch\nfrom minicandi import contract as C\n"


@pytest.mark.parametrize("src,needle", [
    ("import os\n", "import os"),
    ("import subprocess\n", "import subprocess"),
    ("import numpy as np\n", "import numpy"),
    ("import importlib\n", "import importlib"),
    ("from minicandi import dataset\n", "from minicandi import dataset"),
    ("from minicandi.dataset import OnionDataset\n", "from minicandi.dataset"),
    ("import torch.hub\n", "import torch.hub"),
    ("from . import x\n", "relative import"),
    ("from math import *\n", "star import"),
    ("x = open('/project/a.h5')\n", "name open"),
    ("x = eval('1')\n", "name eval"),
    ("x = __import__('os')\n", "name __import__"),
    ("x = getattr(torch, 'lo' + 'ad')\n", "non-constant attribute name"),
    ("x = getattr(torch, 'load')\n", "string 'load'"),
    ("f = getattr\n", "non-constant attribute name"),
    ("x = torch.load('ckpt.pt')\n", "attribute .load"),
    ("from pathlib import Path\nx = Path('a').read_text()\n", "attribute .read_text"),
    ("import sys\nx = sys.modules\n", "sys.modules"),
    ("from sys import modules\n", "from sys import modules"),
    ("x = ().__class__.__base__.__subclasses__()\n", "attribute .__subclasses__"),
    ("def f(b):\n    return b['y_counts']\n", "string 'y_counts'"),
    ("def f(b):\n    return b.get('y_base')\n", "string 'y_base'"),
    ("import operator\nf = operator.attrgetter('x')\n", "attribute .attrgetter"),
    ("def f(:\n", "syntax error"),
])
def test_module_refusals(src, needle):
    probs = AP.check_module(HEAD + src)
    assert any(needle in p for p in probs), probs


def test_module_refusal_through_the_cli(tmp_path):
    rec = _recipe(tmp_path / "r", _rung_cfg(2), HEAD + "import os\n")
    rc, res = _run_cli(["--recipe", rec, "--local", "--tables", tmp_path, "--evalset",
                        tmp_path / "e.json", "--runs-root", tmp_path])
    assert rc != 0 and res["passed"] is False and "model.py refused" in res["reason"]


# ---------------------------------------------------------------------------- dynamic refusals

class _Toy(MiniCandi):
    """mu from a learned constant, plus whatever `extra(batch)` reads; ~1e6 parameters."""

    def __init__(self, n_pad=1_000_000):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.pad = torch.nn.Parameter(torch.zeros(n_pad))

    def extra(self, b):
        return 0.0

    def forward(self, b):
        x = b["x_counts"].float()
        mu = torch.nn.functional.softplus(self.w) + 1.0 + 0.0 * x + self.extra(b)
        return {"n": torch.ones_like(mu) * (1.0 + self.pad[0].abs()), "mu": mu}


class _ReadsHidden(_Toy):
    def extra(self, b):
        return 1e-3 * b["x_counts"].float().mean(-1, keepdim=True) * (1 + self.w)


class _ReadsTarget(_Toy):
    def extra(self, b):
        return 1e-3 * b["y_counts"].float() * (1 + self.w)


class _KeepsState(_Toy):
    """Quiet through the wiring check's two calls, then uses the PREVIOUS call's targets: a
    two-call target check (batch, perturbed) would pass it; the four-call check fails it."""

    def __init__(self):
        super().__init__()
        self.prev, self.calls = None, 0

    def extra(self, b):
        self.calls += 1
        out = 0.0 if (self.calls <= 4 or self.prev is None) else 1e-3 * self.prev * (1 + self.w)
        self.prev = b["y_counts"].float()
        return out


def _patched_check(monkeypatch, model):
    monkeypatch.setattr(registry, "build_model", lambda cfg: model)
    return AP.check_model(_rung_cfg(2), Path("model.py"))


def test_dynamic_checks_pass_an_honest_model(monkeypatch):
    st = _patched_check(monkeypatch, _Toy())
    assert st["param_count"] == 1_000_001 and st["flops_per_window"] >= 0


@pytest.mark.parametrize("model,needle", [
    (_ReadsHidden(), "wiring check failed"),
    (_ReadsTarget(), "target independence failed"),
    (_KeepsState(), "target independence failed"),
    (_Toy(n_pad=10), "outside 10%"),
    (_Toy(n_pad=1_200_000), "outside 10%"),
])
def test_dynamic_refusals(monkeypatch, model, needle):
    with pytest.raises(AP.Reject, match=needle):
        _patched_check(monkeypatch, model)


def test_build_failure_is_a_refusal(monkeypatch):
    def boom(cfg):
        raise RuntimeError("bad module")
    monkeypatch.setattr(registry, "build_model", boom)
    with pytest.raises(AP.Reject, match="build_model failed"):
        AP.check_model(_rung_cfg(2), Path("model.py"))


def test_model_file_not_honored_is_refused(tmp_path, monkeypatch):
    """A registry that ignores model_file (the tree before chunk A1) builds the rung module, so
    the candidate's module would never be scored: refused, never scored silently."""
    rec = _recipe(tmp_path / "r", {})
    real = registry.build_model
    monkeypatch.setattr(registry, "build_model",
                        lambda cfg: real({k: v for k, v in cfg.items() if k != "model_file"}))
    with pytest.raises(AP.Reject, match="model_file not honored"):
        AP.check_model(_rung_cfg(2, model_file=str(rec / "model.py")), rec / "model.py")


def test_the_real_rung2_model_passes_the_dynamic_checks():
    st = AP.check_model(_rung_cfg(2), REPO / "tools/minicandi/rungs/conv_ae.py")
    assert abs(st["param_count"] - 1e6) <= 1e5


A1_MERGED = "model_file" in inspect.getsource(registry)


@pytest.mark.skipif(not A1_MERGED, reason="runs after chunk A1 merges (registry model_file)")
def test_model_file_stub_after_a1(tmp_path):
    """A copy of rung 2's module loaded through model_file passes; a copy that reads hidden
    counts fails the wiring check."""
    rec = _recipe(tmp_path / "r", {})
    cfg = _rung_cfg(2, model_file=str(rec / "model.py"))
    (rec / "config.json").write_text(json.dumps(cfg))
    st = AP.check_model(cfg, rec / "model.py")
    assert abs(st["param_count"] - 1e6) <= 1e5
    leak = (REPO / "tools/minicandi/rungs/conv_ae.py").read_text().replace(
        "    def forward(self, batch: dict) -> dict:\n",
        "    def forward(self, batch: dict) -> dict:\n"
        "        batch = dict(batch, x_bin_mask=batch['x_bin_mask'] & False)\n", 1)
    assert leak != (REPO / "tools/minicandi/rungs/conv_ae.py").read_text()
    (rec / "model.py").write_text(leak)
    with pytest.raises(AP.Reject, match="wiring check failed"):
        AP.check_model(cfg, rec / "model.py")


# ------------------------------------------------------------------------- eval-set refusals

def test_validation_evalset_passes_the_input_check(env):
    AP.check_inputs(env["tables"], env["evalset"])


def test_blind_evalset_is_refused(env):
    with pytest.raises(AP.Reject, match="blind eval set"):
        AP.check_inputs(env["tables"], env["blind_evalset"])


def test_renamed_blind_evalset_is_refused(env, tmp_path):
    ev = json.loads(env["blind_evalset"].read_text())
    p = tmp_path / "evalset_x.json"
    p.write_text(json.dumps(ev))
    with pytest.raises(AP.Reject, match="eval set refused"):
        AP.check_inputs(env["tables"], p)


@pytest.mark.parametrize("edit,needle", [
    ("blind_target", "not a validation track"),
    ("train_target", "not a validation track"),
    ("val_input", "not a training track"),
    ("blind_pair", "not validation-only"),
    ("blind_replicate", "replicate pair"),
    ("tables_sha", "tables_sha256 differs"),
])
def test_evalset_edits_are_refused(env, tmp_path, edit, needle):
    ev = json.loads(env["evalset"].read_text())
    rows = AP._tsv(env["tables"] / "measurements.tsv")
    blind = next(r["measurement_id"] for r in rows if r["in_eic"] == "blind" and not r["arm_knob"])
    train = next(r["measurement_id"] for r in rows if r["in_eic"] == "train")
    val = next(r["measurement_id"] for r in rows if r["in_eic"] == "val")
    t0 = ev["imputation_targets"][0]
    if edit == "blind_target":
        ev["imputation_targets"].append(dict(t0, target=blind))
    elif edit == "train_target":
        ev["imputation_targets"].append(dict(t0, target=train))
    elif edit == "val_input":
        ev["imputation_targets"].append(dict(t0, inputs=t0["inputs"] + [val]))
    elif edit == "blind_pair":
        ev["correction_pairs"].append(dict(ev["correction_pairs"][0], target=blind))
    elif edit == "blind_replicate":
        pairs = AP._tsv(env["tables"] / "pairs.tsv")
        pb = [p["pair_id"] for p in pairs if p["split"] == "blind"]
        assert pb
        ev["replicate_pairs"] = pb[:1]
    elif edit == "tables_sha":
        ev["tables_sha256"] = "0" * 64
    p = tmp_path / "evalset.json"
    p.write_text(json.dumps(ev))
    with pytest.raises(AP.Reject, match=needle):
        AP.check_inputs(env["tables"], p)


def test_cluster_mode_needs_the_eic_v1_tables(env):
    with pytest.raises(AP.Reject, match="not the eic_v1 tables"):
        AP.check_inputs(env["tables"], env["evalset"], cluster=True)


def test_run_seed_refuses_a_blind_evalset_before_training(env, tmp_path):
    res = AP.run_seed(_rung_cfg(2), env["tables"], env["blind_evalset"], 0, 40, tmp_path / "s0")
    assert res["status"] == "refused" and "blind" in res["reason"]
    assert not (tmp_path / "s0" / "ckpt.pt").exists()
    assert json.loads((tmp_path / "s0" / AP.PROXY_FILE).read_text())["status"] == "refused"


def test_job_subcommand_refuses_a_forbidden_key(tmp_path):
    rec = _recipe(tmp_path / "r", _rung_cfg(2, train_on_validation=True))
    rc = AP.job_main(["--recipe", str(rec), "--tables", str(tmp_path), "--evalset",
                      str(tmp_path / "e.json"), "--seed", "0", "--windows", "40", "--out",
                      str(tmp_path / "out")])
    assert rc == 1
    assert "forbidden" in json.loads((tmp_path / "out" / AP.PROXY_FILE).read_text())["reason"]


# ---------------------------------------------------------------------------- budget refusals

def test_budget_exhausted_before_the_attempt(env, tmp_path):
    runs = tmp_path / "runs"
    AP.append_local_tally(runs / "evolve" / "TALLY.tsv", AP.tally_line("old", 0, "1", 12, 3600, 0))
    rec = _recipe(tmp_path / "r", _rung_cfg(2))
    rc, res = _run_cli(["--recipe", rec, "--local", "--tables", env["tables"], "--evalset",
                        env["evalset"], "--windows", 40, "--runs-root", runs, *LOCAL],
                       env={AP.CAP_ENV: "12"})
    assert rc != 0 and res["passed"] is False and "budget exhausted" in res["reason"], res
    assert not (runs / "evolve" / "attempts").exists()          # nothing was trained


def test_budget_exhausted_after_the_attempt(e2e, env, tmp_path, monkeypatch, capsys):
    """The attempt that crosses the cap is refused (its CPU-h stay in the tally)."""
    runs = tmp_path / "runs"
    tally = runs / "evolve" / "TALLY.tsv"
    AP.append_local_tally(tally, AP.tally_line("old", 0, "1", 1, 3600, 0))       # 1.0 CPU-h
    seed_dirs = [Path(e2e["res"]["attempt"]) / f"seed{s}" for s in (0, 1)]

    def fake_local(a, cfg, model_py, key, seeds):
        AP.append_local_tally(tally, AP.tally_line(key, 0, "local", 1, 3600, 0))  # +1.0
        return seed_dirs, 1.0
    monkeypatch.setattr(AP, "run_local", fake_local)
    monkeypatch.setattr(AP, "check_model", lambda cfg, mp: {"param_count": 1, "flops_per_window": 1})
    monkeypatch.setenv(AP.CAP_ENV, "1.5")
    rec = _recipe(tmp_path / "r", _rung_cfg(2))
    rc, res = _main(["--recipe", rec, "--local", "--tables", env["tables"], "--evalset",
                     env["evalset"], "--windows", 40, "--runs-root", runs], capsys)
    assert rc == 1 and res["passed"] is False
    assert "budget exhausted" in res["reason"] and "after this attempt" in res["reason"]
    monkeypatch.setenv(AP.CAP_ENV, "100")                       # the same attempt under the cap
    rc, res = _main(["--recipe", rec, "--local", "--tables", env["tables"], "--evalset",
                     env["evalset"], "--windows", 40, "--runs-root", runs], capsys)
    assert rc == 0 and res["passed"] is True, res


def test_bad_cap_and_bad_arguments_are_refusals(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(AP.CAP_ENV, "lots")
    with pytest.raises(AP.Reject, match="not a number"):
        AP.cap_cpuh()
    rc, res = _main(["--windows", "10"], capsys)                 # no --recipe
    assert rc == 1 and res["passed"] is False and "bad arguments" in res["reason"]
    with pytest.raises(AP.Reject):
        AP.parse_seeds("0,0")
    with pytest.raises(AP.Reject):
        AP.parse_seeds("a")


# ---------------------------------------------------------------- collect: trained-model gates

def _copy_seeds(e2e, tmp_path):
    out = []
    for s in (0, 1):
        src = Path(e2e["res"]["attempt"]) / f"seed{s}"
        d = tmp_path / f"seed{s}"
        d.mkdir()
        for f in (AP.PROXY_FILE, "run.json", "scores.json"):
            shutil.copy(src / f, d / f)
        out.append(d)
    return out


def _edit(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


def test_collect_matches_the_cli_result(e2e, tmp_path):
    c = AP.collect(_copy_seeds(e2e, tmp_path), 40)
    assert c["composite"] == pytest.approx(e2e["res"]["composite"]["value"])


@pytest.mark.parametrize("edit,needle", [
    ("wiring", "wiring check failed on the trained model"),
    ("undefined", "composite undefined"),
    ("evalset_sha", "not comparable"),
    ("split", "not comparable"),
    ("refused", "refused: planted"),
    ("short", "trained 10 windows"),
    ("params", "outside 10%"),
    ("missing", "no scores.json"),
    ("no_proxy", "no PROXY.json"),
])
def test_collect_refusals(e2e, tmp_path, edit, needle):
    d0, d1 = _copy_seeds(e2e, tmp_path)
    if edit == "wiring":
        _edit(d1 / "scores.json", lambda o: o["copy_leak"].update(part1_wiring_pass=False))
    elif edit == "undefined":
        _edit(d0 / "scores.json", lambda o: o["selection"].update(imputation_gap_closed=None))
    elif edit == "evalset_sha":
        _edit(d1 / "scores.json", lambda o: o.update(evalset_sha256="f" * 64))
    elif edit == "split":
        _edit(d1 / "scores.json", lambda o: o.update(split="blind"))
    elif edit == "refused":
        _edit(d1 / AP.PROXY_FILE, lambda o: o.update(status="refused", reason="planted"))
    elif edit == "short":
        _edit(d0 / "run.json", lambda o: o.update(windows_seen=10))
    elif edit == "params":
        _edit(d0 / "run.json", lambda o: o.update(param_count=10))
    elif edit == "missing":
        (d1 / "scores.json").unlink()
    elif edit == "no_proxy":
        (d0 / AP.PROXY_FILE).unlink()
    with pytest.raises(AP.Reject, match=needle):
        AP.collect([d0, d1], 40)


# ------------------------------------------------------------------- cluster path (fake hpc)

FAKE = r"""#!/bin/bash
# fake `hpc run <cluster> <cmd>`: log the command, answer from the environment
printf '%s\n' "$3" >> "$FAKE_LOG"
case "$3" in
  *sha256sum*) printf '%b' "${FAKE_SHA:-}" ;;
  *ElapsedRaw*) echo "600|12" ;;
  sbatch*) echo 4242 ;;
  sacct*) echo "4242|${FAKE_STATE:-RUNNING}" ;;
  *TALLY*) printf '%b' "${FAKE_TALLY:-}" ;;
  *"tar -xf"*) cat > /dev/null ;;
esac
exit "${FAKE_RC:-0}"
"""


@pytest.fixture
def cluster(tmp_path, monkeypatch, env):
    """A recipe inside a fake repository root, a fake hpc, and no model build (A1-independent)."""
    root = tmp_path / "repo"
    rec = root / "tools" / "minicandi" / "evolve" / "recipe1"
    _recipe(rec, _rung_cfg(2, model_file="tools/minicandi/evolve/recipe1/model.py"))
    monkeypatch.setattr(AP, "ROOT", root)
    monkeypatch.setattr(AP, "check_model", lambda cfg, mp: {"param_count": 1_000_000,
                                                            "flops_per_window": 1.0})
    fake = tmp_path / "hpc"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    logf = tmp_path / "hpc.log"
    monkeypatch.setenv("ONION_HPC", str(fake))
    monkeypatch.setenv("FAKE_LOG", str(logf))
    monkeypatch.setenv("CRUX_WORKSPACE", str(tmp_path / "ws"))
    args = ["--recipe", rec, "--tables", "/project/def-maxwl/mforooz/ONION/tables/eic_v1",
            "--poll-seconds", "0.01"]
    return {"args": args, "log": logf, "root": root}


@pytest.mark.parametrize("rc_code", [69, 255])
def test_tunnel_down_is_a_refusal(cluster, monkeypatch, capsys, rc_code):
    monkeypatch.setenv("FAKE_RC", str(rc_code))
    rc, res = _main(cluster["args"], capsys)
    assert rc == 1 and res["passed"] is False and res["reason"].startswith("tunnel down"), res
    assert res["composite"]["value"] is None


def test_cluster_budget_exhausted(cluster, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_TALLY", "time\tattempt\tseed\tjob_id\tcpus\twall_seconds\tcpu_hours"
                                     "\texit\\nT\told\t0\t1\t12\t900000\t3000.0\t0\\n")
    rc, res = _main(cluster["args"], capsys)
    assert rc == 1 and "budget exhausted" in res["reason"], res
    assert "sbatch" not in cluster["log"].read_text()


def test_kit_that_differs_is_refused(cluster, monkeypatch, capsys):
    f = cluster["root"] / "tools" / "onion_eval" / "auto_proxy.py"
    f.parent.mkdir(parents=True)
    f.write_text("frozen\n")
    monkeypatch.setattr(AP, "frozen_files", lambda: ["tools/onion_eval/auto_proxy.py"])
    monkeypatch.setenv("FAKE_SHA", "deadbeef  tools/onion_eval/auto_proxy.py\\n")
    rc, res = _main(cluster["args"], capsys)
    assert rc == 1 and "differs from the scorer's tree" in res["reason"], res
    assert "sbatch" not in cluster["log"].read_text()


def test_cluster_tables_other_than_eic_v1_are_refused(cluster, capsys):
    args = list(cluster["args"])
    args[args.index("--tables") + 1] = "/project/def-maxwl/mforooz/ONION/tables/eic_v2"
    rc, res = _main(args, capsys)
    assert rc == 1 and "not the eic_v1 tables" in res["reason"]


def test_cluster_blind_evalset_is_refused(cluster, capsys):
    rc, res = _main(cluster["args"] + ["--evalset", "/x/eval/eic_v2/evalset_blind.json"], capsys)
    assert rc == 1 and "blind eval set" in res["reason"]


def test_a_job_that_never_ends_times_out_and_is_cancelled(cluster, monkeypatch, capsys):
    monkeypatch.setattr(AP, "check_kit", lambda c, k: None)
    rc, res = _main(cluster["args"] + ["--max-wait-minutes", "0.002", "--seeds", "0"], capsys)
    assert rc == 1 and "timed out" in res["reason"], res
    log_ = cluster["log"].read_text()
    assert "sbatch --parsable" in log_ and "scancel 4242" in log_


def test_a_job_that_died_without_output_is_a_refusal(cluster, monkeypatch, capsys):
    monkeypatch.setattr(AP, "check_kit", lambda c, k: None)
    monkeypatch.setenv("FAKE_STATE", "FAILED")
    rc, res = _main(cluster["args"] + ["--seeds", "0"], capsys)
    assert rc == 1 and "job 4242 ended FAILED" in res["reason"], res
    # the job left no tally row, so the scorer wrote one from sacct
    assert any(line.startswith("mkdir -p") and "flock" in line
               for line in cluster["log"].read_text().splitlines())


def test_sbatch_command_and_parsers():
    cmd = AP.sbatch_command(kit="/k", att="/r/evolve/attempts/a", seed=1, windows=20000,
                            tables="/t/eic_v1", evalset="/e/proxy/evalset.json",
                            recipe_rel="tools/minicandi/evolve/recipe1", tally="/r/evolve/TALLY.tsv",
                            job_time="02:00:00", key="20261111T000000Z_abcdef123456",
                            exclude="n1,n2")
    for part in ("--account=def-maxwl", "--time=02:00:00", "--gres=none", "--cpus-per-task=12",
                 "--mem=48000M", "--exclude=n1,n2", "/k/slurm/onion_eval/auto_proxy.sh"):
        assert part in cmd
    assert "allow-blind" not in cmd and "--export" not in cmd
    assert AP.parse_sacct("4242|COMPLETED\n4243|CANCELLED by 1\n") == {"4242": "COMPLETED",
                                                                       "4243": "CANCELLED"}
    rows = AP.parse_tally("\t".join(AP.TALLY_COLUMNS) + "\n" + AP.tally_line("a", 0, 1, 12, 300, 0))
    assert rows[0]["cpu_hours"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------- slurm script

def _sh(*args, env=None):
    return subprocess.run(["bash", str(SH), *map(str, args)], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def test_slurm_script_dry_run_and_refusals():
    good = ["/k", "/r/evolve/attempts/a1", 0, 20000, "/t/eic_v1", "/e/proxy/evalset.json",
            "tools/minicandi/evolve/recipe1", "/r/evolve/TALLY.tsv"]
    p = _sh(*good, env={"DRY_RUN": "1"})
    assert p.returncode == 0, p.stderr
    assert "auto_proxy.py job --recipe tools/minicandi/evolve/recipe1" in p.stdout
    assert "--seed 0 --windows 20000" in p.stdout and "allow-blind" not in p.stdout
    bad = [(5, "/e/eic_v2/evalset_blind.json"), (6, "tools/minicandi/../../etc"),
           (6, "/abs/recipe"), (2, "x"), (3, "-5")]
    for i, v in bad:
        a = list(good)
        a[i] = v
        assert _sh(*a, env={"DRY_RUN": "1"}).returncode == 2, (i, v)
    assert _sh(*good[:7], env={"DRY_RUN": "1"}).returncode == 2
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0
