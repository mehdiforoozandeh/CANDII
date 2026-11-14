"""BRt retrieval baseline: neighbor choice on two planted cell groups, and the scorer route."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import baseline_retrieval as BR  # noqa: E402

A, K, K_META = 36, 18, 11
L, N_WIN = 200, 5
GROUP1, GROUP2 = ("c0", "c1", "c2", "c3"), ("c4", "c5", "c6", "c7")
ASSAYS = (0, 1, 2)
LOADING = {0: 1.0, 1: 1.3, 2: -0.7}


def _batch(rows, *, kind=1, x_assays=()):
    """rows: list of (cell, start_bin, {assay: counts [L]}, {assay: log2 depth})."""
    B = len(rows)
    y = np.zeros((B, L, A), np.int32)
    avail = np.zeros((B, A), bool)
    c_out = np.zeros((B, A, K_META), np.float32)
    x_avail = np.zeros((B, A), bool)
    meta = []
    for b, (cell, start, ys, lds) in enumerate(rows):
        for a, v in ys.items():
            y[b, :, a] = v
            avail[b, a] = True
        for a, d in lds.items():
            c_out[b, a, 0] = d
        x_avail[b, list(x_assays)] = True
        meta.append({"cell": cell, "chrom": "chr1", "start_bin": start})
    return {"x_counts": torch.zeros(B, L, A, dtype=torch.int32), "x_avail": torch.from_numpy(x_avail),
            "c_in": torch.zeros(B, A, K), "c_out": torch.from_numpy(c_out),
            "y_counts": torch.from_numpy(y), "y_avail": torch.from_numpy(avail),
            "y_kind": torch.from_numpy(np.where(avail, kind, 0).astype(np.uint8)), "meta": meta}


@pytest.fixture(scope="module")
def planted():
    """Two cell groups: each group shares a latent field z_g over the bins; every track of a cell
    is Poisson(depth x exp(loading_a z_g + small cell noise))."""
    rng = np.random.default_rng(0)
    nb = L * N_WIN
    z = {g: rng.normal(0, 1, nb) for g in (1, 2)}
    depth = {}
    tracks = {}
    for g, cells in ((1, GROUP1), (2, GROUP2)):
        for c in cells:
            for a in ASSAYS:
                ld = float(rng.uniform(-1, 1))
                depth[(c, a)] = ld
                lam = 2.0 ** ld * 5 * np.exp(LOADING[a] * z[g] + rng.normal(0, 0.2, nb))
                tracks[(c, a)] = rng.poisson(lam)
    batches = [_batch([(c, w * L, {a: tracks[(c, a)][w * L:(w + 1) * L]}, {a: depth[(c, a)]})
                       for (c, a) in sorted(tracks)]) for w in range(N_WIN)]
    r = BR.Retrieval().fit(batches)
    return {"r": r, "tracks": tracks, "depth": depth}


def test_neighbor_comes_from_the_target_group_and_never_itself(planted):
    r = planted["r"]
    for g, cells in ((GROUP1, GROUP1), (GROUP2, GROUP2)):
        for c in cells:
            for a in ASSAYS:
                d, score, shared = r.neighbor(c, a, [b for b in ASSAYS if b != a])
                assert d != c                      # the target cell is never a candidate
                assert d in g, (c, a, d)           # the neighbor is from the target's group
                assert score > 0.5 and a not in shared and len(shared) == 2
    # the NB-size fit used the same rule, also never the cell itself
    for (c, a), (d, _, _) in r.fit_choices.items():
        assert d != c and (d in GROUP1) == (c in GROUP1)
    assert set(r.size) == set(ASSAYS) and all(np.isfinite(v) and v > 0 for v in r.size.values())


def test_predict_copies_the_neighbor_scaled_to_the_target_depth(planted):
    r, tracks, depth = planted["r"], planted["tracks"], planted["depth"]
    ld_t = 1.5
    batch = _batch([("c0", L, {2: np.zeros(L, np.int32)}, {2: ld_t})], x_assays=(0, 1, 2))
    out = r.predict(batch)
    d = r.choices[("c0", 2, (0, 1))][0]
    assert d in GROUP1 and d != "c0"
    want = np.maximum(tracks[(d, 2)][L:2 * L] * 2.0 ** (ld_t - depth[(d, 2)]), BR.PRED_FLOOR)
    np.testing.assert_allclose(out["mu"][0, :, 2].numpy(), want, rtol=1e-5)
    assert torch.all(out["n"][0, :, 2] == torch.tensor(r.size[2], dtype=torch.float32))
    # every other track, and a pair entry (y_kind 2), is NaN
    assert torch.isnan(out["mu"][0, :, :2]).all()
    pair = _batch([("c0", L, {2: np.zeros(L, np.int32)}, {2: ld_t})], kind=2, x_assays=(0, 1))
    assert torch.isnan(r.predict(pair)["mu"]).all()


def test_no_shared_input_assay_is_undefined(planted):
    r = planted["r"]
    assert r.neighbor("c0", 2, [])[0] is None
    batch = _batch([("c0", 0, {2: np.zeros(L, np.int32)}, {2: 0.0})], x_assays=())
    assert torch.isnan(r.predict(batch)["mu"]).all()


# ---------------------------------------------------------------- the scorer route, end to end

def test_cli_writes_scores_with_the_baseline_keys(tmp_path):
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    from onion_eval import score as S
    val = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"))
    t = make_synthetic_tables(tmp_path, n_cells=6, val_tracks=val, blind_tracks=())
    ev_path = tmp_path / "eval" / "synth" / "evalset.json"
    ev = E.build(t, ev_path, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    ev["correction_pairs"] = ev["correction_pairs"][:1]
    ev_path.write_text(json.dumps(ev))
    kw = dict(n_boot=10, log=lambda *a: None)
    cfg = {"context_bins": 128, "batch_size": 4}
    ref = S.score(ev_path, tmp_path / "pa", baseline="position_average", config=cfg, **kw)
    out = tmp_path / "eval" / "synth" / "baselines" / "retrieval" / "seed0"
    s = BR.run(t, ev_path, out, config=cfg, **kw)
    assert set(s) == set(ref)
    assert set(s["selection"]) == set(ref["selection"])
    assert s["predictor"] == "retrieval" and s["copy_leak"]["copy_leak_pass"] is None
    assert s["n_targets"] == ref["n_targets"] > 0
    crps = s["count_crps"]["non_scorable_per_assay"]     # defined on every assay with a target
    assert set(crps) == set(ref["count_crps"]["non_scorable_per_assay"])
    assert crps and all(np.isfinite(v) and v > 0 for v in crps.values())
    assert (out / "scores.json").is_file() and (out / S.SCORED).is_file()
    rows = list(csv.DictReader(open(out / BR.NEIGHBORS_TSV, encoding="utf-8"), delimiter="\t"))
    pred = [x for x in rows if x["use"] == "predict"]
    assert pred and all(x["neighbor"] and x["neighbor"] != x["cell"] for x in pred)
    assert all(x["neighbor"] != x["cell"] for x in rows)
    # the CLI refuses tables that are not the eval set's, and skips a scored directory
    assert BR.main(["run", "--tables", str(tmp_path), "--evalset", str(ev_path),
                    "--out", str(tmp_path / "x")]) == 2
    assert BR.main(["run", "--tables", str(t), "--evalset", str(ev_path), "--out", str(out)]) == 0
