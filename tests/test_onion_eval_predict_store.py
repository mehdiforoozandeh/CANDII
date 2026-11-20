"""Prediction store (plan t152, Amendment 8, pinned interface L, first item; chunk SP1).

On synthetic tables and a small perturbed model of each of the six rungs: at the legacy batch the
stored `n`, `mu` and `ctl_mu` are bit-identical to the arrays the legacy `_job_arrays` gets from
the model; 2 shards equal 1; a rerun skips stored requests; both TF32 flags are false inside
`predict`; `complete` refuses a store of another evaluation set or run."""
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
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import registry as R  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import predict_store as P  # noqa: E402
from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
L, RB, LEGACY_B = 64, 256, 4          # 4 windows per region; the legacy batch fills one region
KW = dict(n_boot=0, guardrail_windows=8, log=lambda *a: None)


def _tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("pstore")
    t = make_synthetic_tables(root, n_cells=8, val_tracks=VAL, blind_tracks=())
    pairs = _tsv(t / "pairs.tsv")
    meas = {r["measurement_id"]: r for r in _tsv(t / "measurements.tsv")}
    extra = []                 # replicate-ladder pairs, so replicate pair jobs exist
    for r in meas.values():
        if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3":
            extra.append(dict(pairs[0], pair_id=f"rep{len(extra)}", a=r["measurement_id"],
                              b=r["measurement_id"].replace(".base", ".mapq0"), rung="technical",
                              assay=r["assay"], cell=r["cell"], subset_pair="0", split="val",
                              never_trained="0", knobs_differing="mapq"))
            d = Path(r["counts_path"][3:]).parent      # signal tracks: the predicted-control route
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
    with open(t / "pairs.tsv", "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]), delimiter="\t", lineterminator="\n")
        w.writerows(extra)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = ev["correction_pairs"][:4]
    out.write_text(json.dumps(ev))
    other = out.with_name("evalset_other.json")              # same targets, another file hash
    other.write_text(json.dumps(dict(ev, note="another evaluation set")))
    sub = out.with_name("evalset_pair_regions.json")          # pairs on region 1 only
    sub.write_text(json.dumps(dict(ev, pair_regions=[1])))
    return {"tables": t, "evalset": out, "other": other, "sub": sub}


def _model(r: int):
    cfg = {**json.loads((REPO / f"tools/minicandi/configs/rung{r}/base.json").read_text()),
           "context_bins": L, "batch_size": LEGACY_B}
    torch.manual_seed(r)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(r)
    with torch.no_grad():       # the NB head starts at zero: perturb so outputs read the inputs
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval(), cfg


def _scorer(env, r=1, evalset=None):
    m, cfg = _model(r)
    return S.Scorer(evalset or env["evalset"], main="model", model=m, config=cfg,
                    run_tables_sha256=tables_sha256(env["tables"]), **KW)


@pytest.fixture(scope="module")
def sc1(env):
    return _scorer(env, 1)


def test_requests_follow_the_scorer_job_order(env, sc1):
    reqs = P.requests(sc1)
    imp, cor, rep, ctl = sc1.jobs()
    assert len(reqs) == len(imp) + len(cor) + len(rep) + len(ctl)
    kinds = [r["key"].split("/")[0] for r in reqs]
    assert kinds == sorted(kinds, key=lambda k: k != "item")          # items, then pairs
    assert [r["kind"] for r in reqs[len(imp) + len(ctl):]] == (["correction"] * len(cor)
                                                               + ["replicate"] * len(rep))
    assert rep and ctl and any(r["needs_ctl"] for r in reqs)
    assert all(r["region_indices"] == [0, 1] for r in reqs)
    sub = P.requests(_scorer(env, 1, env["sub"]))
    assert all(r["region_indices"] == ([1] if r["key"].startswith("pair/") else [0, 1])
               for r in sub)


@pytest.mark.parametrize("r", sorted(R.RUNGS))
def test_legacy_batch_is_bit_identical_to_job_arrays(env, tmp_path, r):
    sc = _scorer(env, r)
    assert sc.tile_bs == LEGACY_B
    reqs = P.requests(sc)
    P.predict(sc, reqs, tmp_path / "store", LEGACY_B, guardrail=False)
    st = P.PredictionStore.for_scorer(tmp_path / "store", sc)
    assert st.complete(reqs)
    n_ctl = 0
    for q in reqs:
        A = sc._job_arrays(q["job"], ())
        got = st.read(q["key"])
        assert set(got) == ({"n", "mu", "ctl_mu"} if q["needs_ctl"] else {"n", "mu"})
        for k in got:
            assert got[k].dtype == np.float32 and got[k].shape == (2, RB)
            assert np.array_equal(got[k].reshape(-1).astype(np.float64), A[f"main.{k}"]), (q["key"], k)
        assert np.ptp(got["mu"]) > 0          # the model reads its inputs: the check has teeth
        n_ctl += "ctl_mu" in got
        assert ("main.ctl_mu" in A) == ("ctl_mu" in got)
    assert n_ctl >= 1


def test_two_shards_equal_one_and_rerun_skips(env, tmp_path, sc1):
    reqs = P.requests(sc1)
    one = P.predict(sc1, reqs, tmp_path / "one", LEGACY_B, guardrail=False)
    assert one["stored"] == len(reqs) and one["skipped"] == 0
    st2 = P.PredictionStore.for_scorer(tmp_path / "two", sc1)
    a = P.predict(sc1, reqs, tmp_path / "two", LEGACY_B, shard=(0, 2))
    assert a["n_requests"] == len(reqs[0::2]) and not st2.complete(reqs)    # shard 1 missing
    P.predict(sc1, reqs, tmp_path / "two", LEGACY_B, shard=(1, 2))
    assert st2.complete(reqs) and (tmp_path / "two" / P.GUARDRAIL_FILE).is_file()
    assert not (tmp_path / "one" / P.GUARDRAIL_FILE).is_file()
    st1 = P.PredictionStore(tmp_path / "one")
    for q in reqs:
        x, y = st1.read(q["key"]), st2.read(q["key"])
        assert set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x)
    # rerun: every request is skipped, no array file is rewritten
    files = sorted((tmp_path / "one").glob("*.npz"))
    stamp = {f: f.stat().st_mtime_ns for f in files}
    again = P.predict(sc1, reqs, tmp_path / "one", LEGACY_B, guardrail=False)
    assert again["stored"] == 0 and again["skipped"] == len(reqs)
    assert {f: f.stat().st_mtime_ns for f in files} == stamp
    # a rerun after a lost file stores that request only
    files[0].unlink()
    assert not st1.complete(reqs)
    assert P.predict(sc1, reqs, tmp_path / "one", LEGACY_B, guardrail=False)["stored"] == 1
    assert st1.complete(reqs)


def test_larger_batch_spans_regions_and_keeps_the_layout(tmp_path, sc1):
    reqs = P.requests(sc1)[:3]
    assert [len(b) for b in P._batches(sc1, [0, 1], 3)] == [3, 1, 3, 1]       # per region
    assert [len(b) for b in P._batches(sc1, [0, 1], 6)] == [6, 2]             # end to end
    P.predict(sc1, reqs, tmp_path / "b8", 8, guardrail=False)
    P.predict(sc1, reqs, tmp_path / "b4", LEGACY_B, guardrail=False)
    for q in reqs:
        x = P.PredictionStore(tmp_path / "b8").read(q["key"])
        y = P.PredictionStore(tmp_path / "b4").read(q["key"])
        for k in x:
            assert x[k].shape == y[k].shape
            np.testing.assert_allclose(x[k], y[k], rtol=1e-5)


def test_tf32_flags_are_false_inside_predict(tmp_path, sc1):
    seen = []
    model = sc1.model_pred.model
    hook = model.register_forward_pre_hook(lambda m, a: seen.append(P.tf32_flags()))
    old = P.tf32_flags()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        P.predict(sc1, P.requests(sc1)[:2], tmp_path / "s", LEGACY_B, guardrail=False)
        assert P.tf32_flags() == (True, True)                # restored after the phase
    finally:
        hook.remove()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    assert seen and all(f == (False, False) for f in seen)
    head = json.loads((tmp_path / "s" / "PREDICTED_0of1.json").read_text())
    assert head["tf32"] is False and head["amp"] is False and head["infer_batch"] == LEGACY_B


def test_complete_refuses_another_evalset_run_or_settings(env, tmp_path, sc1):
    reqs = P.requests(sc1)
    d = tmp_path / "s"
    P.predict(sc1, reqs, d, LEGACY_B, run=tmp_path / "runA", guardrail=False)
    assert P.PredictionStore.for_scorer(d, sc1, run=tmp_path / "runA").complete(reqs)
    with pytest.raises(S.ScoreRefused, match="run"):
        P.PredictionStore.for_scorer(d, sc1, run=tmp_path / "runB").complete(reqs)
    other = _scorer(env, 1, env["other"])
    with pytest.raises(S.ScoreRefused, match="evalset_sha256"):
        P.PredictionStore.for_scorer(d, other, run=tmp_path / "runA").complete(P.requests(other))
    with pytest.raises(S.ScoreRefused, match="tables_sha256"):
        P.PredictionStore(d, tables_sha256="0" * 64).complete(reqs)
    with pytest.raises(S.ScoreRefused, match="infer_batch"):
        P.PredictionStore(d, settings={"infer_batch": 16}).complete(reqs)
    with pytest.raises(S.ScoreRefused, match="requests_sha256"):     # another request list
        P.PredictionStore(d).complete(reqs[:-1])
    # predict refuses to write into a store of another scoring
    with pytest.raises(S.ScoreRefused, match="run"):
        P.predict(sc1, reqs, d, LEGACY_B, run=tmp_path / "runB", guardrail=False)
    with pytest.raises(S.ScoreRefused, match="infer_batch"):
        P.predict(sc1, reqs, d, 8, run=tmp_path / "runA", guardrail=False)
    with pytest.raises(S.ScoreRefused, match="evalset_sha256"):
        P.predict(other, P.requests(other), d, LEGACY_B, run=tmp_path / "runA", guardrail=False)
