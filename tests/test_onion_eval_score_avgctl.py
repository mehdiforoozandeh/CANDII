"""Chunk F4: in the signal space, the position average of a ChIP-seq target with no observed control
takes its lambda from the average of the training controls, labelled `training_average_control`.

Before F4 the training average `a` of r_spec had no control on such a target, so the signal-space
r_spec was undefined on every no-control ChIP target. Targets with an observed control, and ATAC-seq
targets, must score bit-identically to the code at BASE.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import score as S  # noqa: E402

#: the umbrella commit F4 was cut from (the code before F4)
BASE = "e6da76d"
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "DNase-seq"), (4, "DNase-seq"),
       (5, "DNase-seq"), (6, "H3K27me3"), (7, "chipseq-control"))
BLIND = ((8, "H3K4me3"),)
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)
SKIP_KEYS = {"wall_seconds", "timing_seconds", "torch_threads"}
SIG = "signal_arcsinh"
H1 = "specific_variance_recovered_above_noise"
LAB = "lambda_source_training_average"


def old_module(name: str, relpath: str, tmp: Path):
    import importlib.util
    import subprocess
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{relpath}"], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # pragma: no cover - a kit copy without .git
        pytest.skip(f"git history of {BASE} not available: {e}")
    p = tmp / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Reader(MiniCandi):
    """mu = (1 + mean visible input count) x the target depth from C'; no control prediction."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, b):
        x = (b["x_counts"].float() * b["x_avail"][:, None, :].float()).mean(-1, keepdim=True)
        depth = (2.0 ** b["c_out"][..., 0] / 2 ** 24)[:, None, :]
        mu = (self.w * (1.0 + x) * depth).clamp_min(1e-3)
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


class ReaderCtl(Reader):
    """The same, with predict_control on: the scorer asks it for the control channel too."""


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def write_tsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("avgctl")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    rows = tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    # s0's H3K4me3 target has an observed control; s1 and s2 have none
    base[("s0", "H3K4me3")]["control_measurement_id"] = base[("s0", "chipseq-control")]["measurement_id"]
    write_tsv(t / "measurements.tsv", rows)
    targets = {}
    for (cell, a), r in base.items():
        if a in ("H3K4me3", "DNase-seq") and r["in_eic"] in ("val", "blind"):
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
            targets[(cell, a)] = r["measurement_id"]
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["assay"] in ("H3K4me3", "DNase-seq")
                                and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    ev["scorable"]["imputation"] = ["DNase-seq", "H3K4me3"]
    out.write_text(json.dumps(ev))
    return {"root": root, "tables": t, "evalset": out, "targets": targets}


def _model(mod, env, model=None, **kw):
    return mod.Scorer(env["evalset"], main="model", model=model or Reader(), config=CFG,
                      run_tables_sha256=tables_sha256(env["tables"]), **dict(KW, **kw))


def _rspec_records(mod, env, **kw):
    """Run the scorer and capture its imputation records (r_spec is per record)."""
    sc = _model(mod, env, **kw)
    got = []
    orig = sc._imputation_group

    def spy(jobs, regime):
        res = orig(jobs, regime)
        got.append(res)
        return res
    sc._imputation_group = spy
    s = sc.run()
    recs = {}
    for res in got:
        for r in res:
            recs[r["target"]] = r
    return s, recs


@pytest.fixture(scope="module")
def runs(env, tmp_path_factory):
    old = old_module("score_before_f4", "tools/onion_eval/score.py", tmp_path_factory.mktemp("old"))
    return {"old": _rspec_records(old, env), "new": _rspec_records(S, env), "mod_old": old}


def test_training_controls_exist_and_are_train_only(env):
    sc = _model(S, env)
    ctl = {c: m for (c, a), m in sc.ds.train_base.items() if a == S.CONTROL}
    assert len(ctl) == 8 and "s7" not in ctl                # s7's control is a validation track
    assert all(sc.ds.rows[m]["in_eic"] == "train" for m in ctl.values())


def test_rspec_defined_on_no_control_chip_target(runs, env):
    """Fails at BASE (r_spec undefined: the position average has no control), passes after F4."""
    (_, old), (s, new) = runs["old"], runs["new"]
    for cell in ("s1", "s2"):
        t = env["targets"][(cell, "H3K4me3")]
        assert S._undef(old[t]["sig_r_spec"])                    # the defect, reproduced
        assert not S._undef(new[t]["sig_r_spec"]) and np.isfinite(new[t]["sig_r_spec"])
        assert not S._undef(new[t]["sig_pi_r_spec"])
        assert new[t][LAB] == S.TRAINING_AVERAGE_CONTROL
        per = s[H1][SIG]["per_target"][t]
        assert per["lambda_source"] == "predicted_control" or per["lambda_source"] is None
        assert per[LAB] == "training_average_control"
    assert s["r_spec"][SIG]["per_assay"].get("H3K4me3") is not None


def _strip(x, drop=()):
    if isinstance(x, dict):
        return {k: _strip(v, drop) for k, v in x.items() if k not in SKIP_KEYS and k not in drop}
    if isinstance(x, list):
        return [_strip(v, drop) for v in x]
    return x


def _same(a, b):
    return json.dumps(_strip(a), sort_keys=True, default=str) == json.dumps(_strip(b), sort_keys=True,
                                                                             default=str)


def test_observed_control_and_dnase_targets_bit_identical(runs, env):
    (_, old), (_, new) = runs["old"], runs["new"]
    keep = [env["targets"][("s0", "H3K4me3")]] + [env["targets"][(c, "DNase-seq")]
                                                  for c in ("s3", "s4", "s5")]
    for t in keep:
        assert LAB not in new[t]
        assert set(old[t]) == set(new[t])
        for k in old[t]:
            assert _same(old[t][k], new[t][k]), (t, k)
    # the count space of every target is untouched
    for t in old:
        for k in ("level", "E_sp", "V_sp", "N", "recovered", "r_spec"):
            if k in old[t]:
                assert _same(old[t][k], new[t][k]), (t, k)


def test_dnase_per_assay_scores_bit_identical(runs):
    (so, _), (sn, _) = runs["old"], runs["new"]
    for key in (H1, "r_spec"):
        for space in so[key]:
            po, pn = so[key][space].get("per_assay") or {}, sn[key][space].get("per_assay") or {}
            assert _same(po.get("DNase-seq"), pn.get("DNase-seq")), (key, space)
    assert _same(so[H1]["counts_log1p"], sn[H1]["counts_log1p"])


def _flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP_KEYS:
                yield from _flat(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, json.dumps(x, default=str)


def _changed(a, b):
    fa, fb = dict(_flat(a)), dict(_flat(b))
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def test_model_scores_change_only_where_no_control(runs, env):
    """Whole scores.json of the default model run: every changed key is the H3K4me3 signal-space
    r_spec aggregate or a key of a no-control target; nothing of s0 or DNase-seq moves."""
    (so, _), (sn, _) = runs["old"], runs["new"]
    nc = {env["targets"][(c, "H3K4me3")] for c in ("s1", "s2")}
    ch = _changed(so, sn)
    assert ch
    for k in ch:
        assert (k.startswith("/r_spec/signal_arcsinh") and "DNase" not in k) or any(t in k for t in nc), k
    assert not any(env["targets"][("s0", "H3K4me3")] in k for k in ch)


def test_position_average_baseline_labelled(env, runs, tmp_path):
    """The position average as main predictor: no-control targets get a signal, labelled; s0 keeps
    every per-target key except its signal-space `level`, which needs every target of the assay
    (before F4 it was undefined because s1 and s2 had no signal)."""
    old = runs["mod_old"]
    a = old.score(env["evalset"], tmp_path / "old", baseline="position_average", config=CFG, **KW)
    b = S.score(env["evalset"], tmp_path / "new", baseline="position_average", config=CFG, **KW)
    for space in (SIG, SIG + "_plug_in"):
        po, pn = a[H1][space]["per_target"], b[H1][space]["per_target"]
        for cell in ("s1", "s2"):
            t = env["targets"][(cell, "H3K4me3")]
            assert po[t]["lambda_source"] is None
            assert pn[t]["lambda_source"] == "training_average_control" == pn[t][LAB]
        t0 = env["targets"][("s0", "H3K4me3")]
        assert po[t0]["level"] is None and np.isfinite(pn[t0]["level"])
        assert _same({k: v for k, v in po[t0].items() if k != "level"},
                     {k: v for k, v in pn[t0].items() if k != "level"})
        for c in ("s3", "s4", "s5"):
            t = env["targets"][(c, "DNase-seq")]
            assert _same(po[t], pn[t]), t
    for k in _changed(a, b):
        assert "DNase" not in k and "counts_log1p" not in k, k
