"""Stage-2 arm configs and the all-arms integration test (chunk S2K; plan table "Stage-2 arms").

(a) The writer: 31 arms, each one change from the incumbent, with the pinned
    `simpler_than_base`, D14 flags, comparisons, `needs`, the arms that never join a design, and
    no json (a reason in `arms.tsv`) for an arm that does not apply or changes nothing.
(b) Integration, on `synth.py` tables with small widths and a 64-bin context: on rungs 2 and 5
    every arm is written, built through the registry by `train.train`, trained 3 steps with a
    finite loss, reloaded with `load_run` and passes `contract.wiring_check` with a perturbed
    head; on rungs 1, 3, 4 and 6 the same for `two_step_native`, `average_input` and
    `control_lambda_head`. One `score.score` call runs on rung 2 with `two_step_native`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import stage2_arms as S2  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
BS = 4
STEPS = 3
#: small widths that every arm of the rung fits within the rung's 10% rule at L = 64
SMALL = {1: {"param_budget": 1e5}, 2: {"param_budget": 1e5}, 3: {"param_budget": 2e5},
         4: {"param_budget": 2e5}, 5: {"param_budget": 1e5},
         6: {"param_budget": 1e5, "d_model": 32, "n_latents": 16, "n_self_layers": 1}}
ALL_ARM_RUNGS = (2, 5)
THREE = ("two_step_native", "average_input", "control_lambda_head")
CASES = [(r, a) for r in ALL_ARM_RUNGS for a in S2.ARMS] + \
        [(r, a) for r in (1, 3, 4, 6) for a in THREE]

TABLE = {  # the plan's table "Stage-2 arms", group by group
    "covariate vector (D6)": ["qc_on_input"],
    "correction route (D15)": ["two_step_native", "two_step_fixed"],
    "ChIP control": ["control_output_only", "control_none", "control_lambda_head"],
    "per-position average (D10)": ["average_input"],
    "DNA source": ["dna_flashzoi", "dna_gnlm", "dna_shuffled", "dna_shift5mb"],
    "covariate embedding and injection": ["inject_adaln_zero", "cov_dropout",
                                          "cov_numeric_only"],
    "depth-controlled input": ["input_depth_fixed"],
    "count transform": ["tf_log1p", "tf_asinh", "tf_anscombe", "tf_pearson", "tf_local_ratio",
                        "tf_quantile", "tf_raw", "tf_soft_clip", "offset_eps0.1"],
    "loss and head": ["loss_poisson_mean", "loss_nb_crps", "loss_impute_x4", "head_shared_size"],
    "masking": ["mask_dirichlet", "mask_panel", "bin_mask_span"],
}


def _incumbent(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/arms/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, batch_size=BS, windows_seen=STEPS * BS, **SMALL[r])
    cfg.update(over)
    return cfg


def _tsv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def _no_build(cfg):
    return None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2")
    td = synth.make_synthetic_tables(root)
    sizes = json.loads((root / "genome" / "chrom_sizes.json").read_text())
    emb = root / "dna_emb"
    emb.mkdir()
    rng = np.random.default_rng(0)
    for c, s in sizes.items():
        with h5py.File(emb / f"{c}.h5", "w") as f:
            f.create_dataset("emb", data=rng.standard_normal(
                (s // C.RES, S2.DNA_EMBEDDING_DIM)).astype(np.float16))
    return {"root": root, "tables": td, "emb": emb, "written": {}}


def _written(env, r: int) -> dict:
    """{arm: row} of `stage2_arms.write` on rung r's small incumbent (cached per module)."""
    if r not in env["written"]:
        out = env["root"] / f"arms_rung{r}"
        S2.write(_incumbent(r), out, {"flashzoi_emb": env["emb"], "gnlm_emb": env["emb"]})
        env["written"][r] = {row["arm"]: row for row in _tsv(out / "arms.tsv")}
        env["written"][r]["_dir"] = out
    return env["written"][r]


# ------------------------------------------------------------------------------ (a) the writer


def test_the_table_has_31_arms_in_its_groups():
    assert len(S2.ARMS) == 31
    got = {}
    for arm, (group, _) in S2.ARMS.items():
        got.setdefault(group, []).append(arm)
    assert got == TABLE


def test_simpler_d14_comparison_needs_and_design_flags():
    true = {"control_none", "control_output_only", "cov_numeric_only", "head_shared_size",
            "tf_raw"}
    false = {"qc_on_input", "two_step_native", "two_step_fixed", "average_input",
             "dna_flashzoi", "dna_gnlm", "control_lambda_head", "inject_adaln_zero"}
    for arm in S2.ARMS:
        assert S2.SIMPLER[arm] is (True if arm in true else False if arm in false else None), arm
    rows = {r["arm"]: r for r in S2.plan(_incumbent(2), build_check=_no_build)}
    assert {a for a, r in rows.items() if r["d14_condition_1"] == "true"} == {
        "qc_on_input", "two_step_native", "two_step_fixed"}
    assert {a: r["comparison incumbent"] for a, r in rows.items()
            if r["comparison incumbent"] != "incumbent"} == {"two_step_fixed": "two_step_native"}
    assert {a: r["needs"] for a, r in rows.items() if r["needs"]} == {
        "qc_on_input": "qc_table", "average_input": "average_h5",
        "dna_flashzoi": "dna_embedding_h5", "dna_gnlm": "dna_embedding_h5"}
    never = {a for a, r in rows.items() if r["joins_design"] != "yes"}
    assert never == {"control_none", "dna_flashzoi", "dna_shuffled", "dna_shift5mb"}
    assert "ruling 26" in rows["dna_flashzoi"]["joins_design"]
    assert "fold-3" in rows["dna_flashzoi"]["joins_design"]
    assert "ruling 14" in rows["control_none"]["joins_design"]


@pytest.mark.parametrize("r", (1, 2, 3, 4, 5, 6))
def test_each_json_is_one_change_from_the_incumbent(env, r):
    inc = _incumbent(r)
    rows = _written(env, r)
    out = rows["_dir"]
    assert sorted(p.stem for p in out.glob("*.json")) == sorted(S2.ARMS)
    for arm, (_, change) in S2.ARMS.items():
        assert rows[arm]["status"] == "write", (r, arm, rows[arm]["status"])
        cfg = json.loads((out / f"{arm}.json").read_text())
        diff = {k for k in set(cfg) | set(inc) if cfg.get(k) != inc.get(k)} - {"simpler_than_base"}
        extra = {"dna_embedding_dim", "dna_embedding_h5"} if arm.startswith("dna_") and \
            change.get("dna") == "precomputed" else set()
        assert diff == {k for k, v in change.items() if inc.get(k) != v} | extra, (arm, diff)
        assert cfg["simpler_than_base"] is S2.SIMPLER[arm]
        for k, v in change.items():
            assert cfg[k] == v
    flz = json.loads((out / "dna_flashzoi.json").read_text())
    assert flz["dna_embedding_dim"] == 128 and flz["dna_embedding_h5"] == str(env["emb"])
    assert rows["dna_flashzoi"]["needs"] == f"dna_embedding_h5={env['emb']}"


def test_qc_on_input_and_two_step_are_exclusive_both_ways(tmp_path):
    rows = {r["arm"]: r for r in S2.write(_incumbent(2, qc_on_input=True), tmp_path / "qc")}
    for arm in ("two_step_native", "two_step_fixed"):
        assert rows[arm]["status"].startswith("not_applicable: two_step (D15) with qc_on_input")
        assert not (tmp_path / "qc" / f"{arm}.json").exists()
    assert rows["qc_on_input"]["status"].startswith("no_change")
    rows = {r["arm"]: r for r in S2.write(_incumbent(2, two_step=True), tmp_path / "ts")}
    assert rows["qc_on_input"]["status"].startswith("not_applicable: two_step (D15)")
    assert rows["two_step_native"]["status"].startswith("no_change")
    # the incumbent already is two_step_native, so two_step_fixed is compared with it directly
    assert rows["two_step_fixed"]["status"] == "write"
    assert rows["two_step_fixed"]["comparison incumbent"] == "incumbent"
    tsv = {r["arm"]: r for r in _tsv(tmp_path / "ts" / "arms.tsv")}
    assert len(tsv) == 31 and tsv["qc_on_input"]["status"] == rows["qc_on_input"]["status"]


def test_offset_eps_needs_the_offset_transform_and_a_held_value_is_no_change(tmp_path):
    rows = {r["arm"]: r for r in S2.plan(_incumbent(2, count_transform="log1p"),
                                         build_check=_no_build)}
    assert rows["offset_eps0.1"]["status"].startswith("not_applicable: offset_eps acts only")
    assert rows["tf_log1p"]["status"].startswith("no_change")
    assert rows["tf_asinh"]["status"] == "write"


def test_a_not_applicable_build_writes_no_json_and_removes_a_stale_one(tmp_path):
    def refuse(cfg):
        return "stub: this hybrid has no shared-size head" \
            if cfg.get("head") == "nb_shared_size" else None

    (tmp_path / "head_shared_size.json").write_text("{}")
    rows = {r["arm"]: r for r in S2.write(_incumbent(2), tmp_path, build_check=refuse)}
    assert rows["head_shared_size"]["status"] == \
        "not_applicable: stub: this hybrid has no shared-size head"
    assert not (tmp_path / "head_shared_size.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == 30


def test_input_paths_go_into_the_json(tmp_path):
    rows = {r["arm"]: r for r in S2.plan(
        _incumbent(2), {"average_h5": "/a.h5", "qc_table": "/q.tsv", "flashzoi_emb": "/f",
                        "gnlm_emb": "/g"}, build_check=_no_build)}
    assert rows["average_input"]["config"]["average_h5"] == "/a.h5"
    assert rows["qc_on_input"]["config"]["qc_table"] == "/q.tsv"
    assert rows["dna_flashzoi"]["config"]["dna_embedding_h5"] == "/f"
    assert rows["dna_gnlm"]["config"]["dna_embedding_h5"] == "/g"
    assert rows["qc_on_input"]["needs"] == "qc_table=/q.tsv"
    bare = {r["arm"]: r for r in S2.plan(_incumbent(2), build_check=_no_build)}
    assert "average_h5" not in bare["average_input"]["config"]
    assert "dna_embedding_h5" not in bare["dna_gnlm"]["config"]


def test_cli_write(tmp_path, capsys):
    inc = tmp_path / "incumbent.json"
    inc.write_text(json.dumps(_incumbent(5)))
    assert S2.main(["write", "--incumbent", str(inc), "--out", str(tmp_path / "o"),
                    "--gnlm-emb", "/g"]) == 0
    assert "wrote 31 arm configs" in capsys.readouterr().out
    rows = _tsv(tmp_path / "o" / "arms.tsv")
    assert [r["arm"] for r in rows] == list(S2.ARMS)
    assert list(rows[0]) == list(S2.ARMS_TSV_COLUMNS)
    assert json.loads((tmp_path / "o" / "dna_gnlm.json").read_text())["dna_embedding_h5"] == "/g"


# ------------------------------------------------------------------------- (b) integration


def _perturb(model, seed: int = 0):
    """Noise on every parameter: the NB head starts at zero, so an untrained model would ignore
    its inputs and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


@pytest.mark.parametrize("r,arm", CASES, ids=[f"rung{r}-{a}" for r, a in CASES])
def test_arm_trains_reloads_and_passes_the_wiring_check(env, r, arm):
    rows = _written(env, r)
    assert rows[arm]["status"] == "write", rows[arm]["status"]
    cfg = json.loads((rows["_dir"] / f"{arm}.json").read_text())
    run = env["root"] / "runs" / f"rung{r}_{arm}"
    info = T.train(cfg, env["tables"], 0, run, log_every=1)
    assert info["windows_seen"] >= STEPS * BS
    losses = [json.loads(x)["loss"] for x in (run / "log.jsonl").read_text().splitlines()]
    assert len(losses) == STEPS and all(math.isfinite(x) for x in losses), losses
    model, saved = R.load_run(run)
    assert saved == T.full_config(cfg)
    b = OnionDataset(env["tables"], "train", saved, 1).batch(BS, np.random.default_rng(1))
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    w = C.wiring_check(_perturb(model), b)
    assert w["pass"], (r, arm, w)
    if arm == "two_step_native":
        assert type(model).__name__ == "TwoStep"
    if arm == "control_lambda_head":
        assert type(model).__name__ != R.model_class(saved).__name__  # wrapped


def test_score_runs_on_rung2_two_step_native(env):
    from onion_eval import evalset as E
    from onion_eval import score as SC
    run = env["root"] / "runs" / "rung2_two_step_native"
    if not (run / "ckpt.pt").is_file():
        cfg = json.loads((_written(env, 2)["_dir"] / "two_step_native.json").read_text())
        T.train(cfg, env["tables"], 0, run, log_every=1)
    ev = env["root"] / "eval" / "s2k" / "evalset.json"
    E.build(env["tables"], ev, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    s = SC.score(ev, run / "scored", run=run, n_boot=0, guardrail_windows=4,
                 log=lambda *a: None)
    assert (run / "scored" / "scores.json").is_file() and (run / "scored" / "SCORED").is_file()
    assert s["split"] == "val" and s["n_targets"] > 0
    assert s["copy_leak"]["part1_wiring_pass"] is True
    assert s["run"] == str(run.resolve())
