"""Ablation 2 (predict_control): the scorer completes for a model run with `predict_control: false`
(control input kept), and the arm changes no input of the composite.

Synthetic tables with a validation ChIP-seq control (so the eval set has one control target and
control correction pairs). The same model weights are scored twice, once with the run config's
`predict_control` true and once false. Expected: both complete on the in-process path and on the
prediction-store path (predict + metrics phases, same-view switch on, as `score_final.sh`); every
scores.json key is equal except (1) the `predicted_control` table, which the OFF arm skips
(`Scorer.jobs` drops the control targets, reason "predict_control is off ..."), and (2) the
non-scorable control row of `subset_pairs_gap_closed`. On the in-process path its depth-only
reference is undefined under OFF, because the dataset drops the control target of a control pair. No code is changed by this test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (4, "chipseq-control"))
L, RB, B = 64, 256, 4
KW = dict(n_boot=5, guardrail_windows=4, log=lambda *a: None)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads", "pred_store")
CONFIG = REPO / "tools/minicandi/configs/ablations/predict_control/rung2/predict_control_off.json"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import tables_sha256
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("abl_pc")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    assert len(ev["control_targets"]) == 1
    head = [p for p in ev["correction_pairs"] if p["never_trained"] and not p["subset_pair"]][:4]
    ctl = [p for p in ev["correction_pairs"] if p["assay"] == S.CONTROL][:2]
    assert head and ctl
    ev["correction_pairs"] = head + ctl
    ev["replicate_pairs"] = []
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    return {"root": root, "evalset": out, "refs": refs, "sha": tables_sha256(t)}


def _model_and_configs():
    from minicandi import registry as R
    off = {**json.loads(CONFIG.read_text()), "context_bins": L, "batch_size": B}
    assert off["predict_control"] is False and off["control_input"] is True
    on = dict(off, predict_control=True)
    torch.manual_seed(2)
    m = R.build_model(off)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval(), on, off


def _scores(d):
    s = json.loads((d / "scores.json").read_text())
    for k in VOLATILE:
        s.pop(k, None)
    return s


def _check_on_off(a, b):
    assert b["predicted_control"]["n_targets"] == 0
    assert "predict_control is off" in b["predicted_control"]["reason"]
    assert a["predicted_control"]["n_targets"] == 1
    assert a["selection"] == b["selection"]
    assert a[S.CORRECTION_ONLY_KEY] == b[S.CORRECTION_ONLY_KEY]
    assert a["copy_leak"] == b["copy_leak"]
    ra = a["subset_pairs_gap_closed"]["per_assay_reason"]
    rb = b["subset_pairs_gap_closed"]["per_assay_reason"]
    assert S.CONTROL in rb          # the non-scorable control row may differ; it is never quoted
    for s in (a, b):
        s.pop("predicted_control")
        s["subset_pairs_gap_closed"]["per_assay_reason"].pop(S.CONTROL, None)
    assert ra.keys() - {S.CONTROL} == rb.keys() - {S.CONTROL}
    assert a == b


def test_in_process_scorer_completes_and_changes_only_the_control_tables(env):
    m, on, off = _model_and_configs()
    for name, cfg in (("on", on), ("off", off)):
        S.score(env["evalset"], env["root"] / name, model=m, config=cfg,
                run_tables_sha256=env["sha"], same_view=True, **KW)
    _check_on_off(_scores(env["root"] / "on"), _scores(env["root"] / "off"))


def test_store_path_completes_and_changes_only_the_control_tables(env):
    from onion_eval import predict_store as P
    m, on, off = _model_and_configs()
    res = {}
    for name, cfg in (("on", on), ("off", off)):
        sc = S.Scorer(env["evalset"], main="model", model=m, config=cfg,
                      run_tables_sha256=env["sha"], same_view=True, **KW)
        reqs = P.requests(sc)
        assert any(r["needs_ctl"] is False for r in reqs)
        store = env["root"] / f"store_{name}"
        P.predict(sc, reqs, store, B)
        out = env["root"] / f"metrics_{name}"
        S.score(env["evalset"], out, phase="metrics", pred_store=store, config=cfg,
                run_tables_sha256=env["sha"], refs_from=env["refs"], same_view=True, **MKW)
        res[name] = _scores(out)
    _check_on_off(res["on"], res["off"])
