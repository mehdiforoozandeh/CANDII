"""SM: the scorer's memory bound, and its outputs unchanged by the streaming restructure.

Pins:
  - identity: on a synthetic case with 6 regions, imputation targets, correction pairs and the
    `--dump-signal` hook, scores.json (volatile keys dropped), suffstats.npz arrays and every dumped
    method track file equal digests taken with the scorer of the base commit fb3eeb7 (before SM),
    one set per platform (the last float bits differ between the laptop and the Nibi nodes);
  - memory: the traced peak (tracemalloc) of one scoring grows with the number of regions by at most
    the arrays of one target group, never by a per-region cache held across the run.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_onion_eval_ag2_dump as T  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

RB = T.RB
N_REGIONS = 6
N_PAIRS = 3
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)

#: digests of `recipes` with the scorer of fb3eeb7 (before SM); darwin: the laptop (candii env);
#: linux: Nibi (kit SMd0 = fb3eeb7 plus this file, job 27189666)
_DIGESTS = {
    "darwin": {"depth_only_scaling": {"dump": "2e8aec7813b0635827d51ad453dc5ec214b30c82eb2a2014fa1b0cdd2dba9767", "scores": "233a2ed4acb4187f60aa314e1cbc2d82ea2cb3605af8f5083015c7695873a204", "suff": "67456a031cefdc83226b03467d5ca3abdd3bb74c8c02b284bfe6e80afb80d8a6"}, "marginal_nb": {"dump": "fa9f5cc44e501d229b217c70544a8391aca385c35f1de25edbff1822da0e7f6c", "scores": "411e76c7252b18c6b296bad923903a1a2019a3d9ae9d13bb48e0a809237a8574", "suff": "86c179fd442de5f485f66699a140f85b8e3aca7170004f692200814ae9b1586e"}, "model_pc": {"dump": "d6e3c6411e1729ecbefe1383ede0f21474618397618b0dd7dc74cde28b42166e", "scores": "4237a5db2be4ff7f2e8dcab95b4f47ef703a9291d3b7e2701695ede472ce7090", "suff": "352d68f26eecd26d13b3c1f799f8f7f06158c4499c15393355ce545557873653"}, "position_average": {"dump": "7a3c96e4e0d79927cf5d9f5b8bb716cfcec4b7630c1dbdddff7a7ff814c1ba0e", "scores": "73c4bdf6d8045649e0b10e5c6316167239bd08dd481417e405c02c58040ac996", "suff": "78382455c394ef248d060eb86e833645c74e9b368fbd61a6170c210fa0b0f5fa"}, "pseudoreplicate_floor": {"dump": "fa8947270b25a8122da2a4baf61f9a3cedbcbdc0fc47ff900781b6efcf19f455", "scores": "3eda3e0c2682c79baaf5f1377972800f8de2a43e990970c8c3899d0374048db1", "suff": "3e4415ac215c58c1bdc47b559c721ba15c0df29d7281e9080084aaf7819a85d2"}},
    "linux": {"depth_only_scaling": {"dump": "2e8aec7813b0635827d51ad453dc5ec214b30c82eb2a2014fa1b0cdd2dba9767", "scores": "39aaf9aa365c7d1f950d426fae9cf6c96d2c8cc8063936ee971749a44f7158cd", "suff": "2b74de6fa969c6c983b55bc8e7b96f577e3bcb5d1a1bdcff811bea6cdcd4f6c0"}, "marginal_nb": {"dump": "fa9f5cc44e501d229b217c70544a8391aca385c35f1de25edbff1822da0e7f6c", "scores": "590caef4e61b35d4b8bab4d64f89a70b1ce2a3df1f2fb89f9ced3989669d0878", "suff": "0bfbd56dbad2f78313fa10229ea2510f3291eb7bd441d37d750ca6b832d2f7f7"}, "model_pc": {"dump": "d6e3c6411e1729ecbefe1383ede0f21474618397618b0dd7dc74cde28b42166e", "scores": "30d0cb0a4fbec2f3e1e32e1fe686929c7d2128364bf0428f70b7198c9351d146", "suff": "d2076a93db8009f48f6994bb50bd83a598e21a516ec14fe50a9fb4aef0a11a93"}, "position_average": {"dump": "7a3c96e4e0d79927cf5d9f5b8bb716cfcec4b7630c1dbdddff7a7ff814c1ba0e", "scores": "532be8cfac5a6dda8100a998a1a66c48e9c6eaaa33f10ff465bcd14b5e5d0cba", "suff": "ff5df35e860483a59fdbd9607da6ce2b7982601ef219a14873828246d7460ea2"}, "pseudoreplicate_floor": {"dump": "fa8947270b25a8122da2a4baf61f9a3cedbcbdc0fc47ff900781b6efcf19f455", "scores": "359bd56f210f150a024863ad3a42ae4eab9db4e749267e56fb50a20c62cfdb79", "suff": "c0ddba4581ebd420ca966a089c8bbeb41ae1d91684cf2915fe7be675ca2fc61e"}},
}
BASE_DIGESTS = _DIGESTS.get(sys.platform, _DIGESTS["linux"])


def _trim(ev, n_pairs=N_PAIRS):
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["assay"] in ("H3K4me3", "DNase-seq")]
    ev["correction_pairs"] = ev["correction_pairs"][:n_pairs]
    ev["replicate_pairs"] = []
    ev["scorable"]["imputation"] = ["DNase-seq", "H3K4me3"]
    return ev


def make_env(root: Path, n_regions: int = N_REGIONS, n_pairs: int = N_PAIRS,
             chrom_bins=None) -> dict:
    """The AGd fixture (`test_onion_eval_ag2_dump.make_env`) with more regions and a few pairs."""
    t = make_synthetic_tables(root, n_cells=9, val_tracks=T.VAL, blind_tracks=T.BLIND,
                              chrom_bins=chrom_bins)
    rows = T._tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    for cell in ("s0", "s8"):
        base[(cell, "H3K4me3")]["control_measurement_id"] = base[(cell, "chipseq-control")][
            "measurement_id"]
    T._write_tsv(t / "measurements.tsv", rows)
    import h5py
    for (cell, a), r in base.items():
        if a in ("H3K4me3", "DNase-seq") and r["in_eic"] in ("val", "blind"):
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
    val = root / "eval" / "synth" / "evalset.json"
    val.parent.mkdir(parents=True, exist_ok=True)
    val.write_text(json.dumps(_trim(E.build(t, val, n_regions=n_regions, n_fit_regions=1,
                                            region_bins=RB, seed=0), n_pairs)))
    return {"root": root, "tables": t, "evalset": val}


def dump_digest(d: Path, root: Path) -> str:
    """Every file of a --dump-signal directory: .npy bytes, tracks.json normalized."""
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(d))
        if p.name == "tracks.json":
            m = json.loads(p.read_text(encoding="utf-8"))
            m.pop("evalset_sha256", None)
            body = T._normalize(json.dumps(m, sort_keys=True), root).encode()
        else:
            body = p.read_bytes()
        h.update(f"{rel}|".encode() + hashlib.sha256(body).digest())
    return h.hexdigest()


def _score(env, out, **kw):
    from onion_eval import score as S
    model = kw.pop("model", None)
    base = kw.pop("baseline", None)
    if model is None and base is None:
        model = T.Reader()
    return S.score(env["evalset"], out, model=model, baseline=base, config=dict(T.CFG),
                   run_tables_sha256=tables_sha256(env["tables"]) if model is not None else None,
                   **KW, **kw)


RECIPES = (
    ("model_pc", dict(lambda_source="predicted_control")),
    ("marginal_nb", dict(baseline="marginal_nb")),
    ("position_average", dict(baseline="position_average")),
    ("depth_only_scaling", dict(baseline="depth_only_scaling")),
    ("pseudoreplicate_floor", dict(baseline="pseudoreplicate_floor")),
)


def recipes(env, out_root: Path, dump_root: Path) -> dict:
    got = {}
    for name, kw in RECIPES:
        o, d = Path(out_root) / name, Path(dump_root) / name
        _score(env, o, dump_signal=d, **kw)
        got[name] = {"scores": T.scores_digest(o, env["root"]), "suff": T.suff_digest(o),
                     "dump": dump_digest(d, env["root"])}
    return got


@pytest.fixture(scope="module")
def digests(tmp_path_factory):
    root = tmp_path_factory.mktemp("sm")
    env = make_env(root / "env")
    return recipes(env, root / "out", root / "dump")


def test_outputs_identical_to_base(digests):
    """scores.json, suffstats.npz and the dumped track files equal the fb3eeb7 digests."""
    assert BASE_DIGESTS is not None, "base digests not pinned for this platform"
    assert digests == BASE_DIGESTS


def _traced(root: Path, n_regions: int) -> dict:
    """Traced (tracemalloc) memory of one position-average scoring: the most held right after a
    target's vectors are built (`_job_arrays`, i.e. inside a group) and right after a group closes
    (`_imputation_group`). Transient peaks inside a metric (bounded by constants such as the signal
    module's _MAX_CELLS) are not what grows with the region count, so they are not asserted."""
    from onion_eval import score as S
    env = make_env(root, n_regions=n_regions, n_pairs=1, chrom_bins={"chr1": 4096, "chr2": 2048})
    got = {"in_group": 0, "after_group": 0, "pa_after_group": 0, "pa_foreign_cells": 0}
    orig = {nm: getattr(S.Scorer, nm) for nm in ("_job_arrays", "_imputation_group")}

    def spy(nm, key):
        def w(self, *a, **k):
            r = orig[nm](self, *a, **k)
            got[key] = max(got[key], tracemalloc.get_traced_memory()[0])
            if key == "after_group":      # position-average fits still held once the group closed
                got["pa_after_group"] = max(got["pa_after_group"], len(self._pa_cache))
            else:                         # cells kept by a cached fit outside the group's targets
                group = getattr(self, "_pa_keep_cells", None) or set()
                got["pa_foreign_cells"] = max([got["pa_foreign_cells"]] + [
                    len(set(t["cells"]) - group) for pa in self._pa_cache.values()
                    for t in pa.table.values()])
            return r
        return w
    S.Scorer._job_arrays = spy("_job_arrays", "in_group")
    S.Scorer._imputation_group = spy("_imputation_group", "after_group")
    tracemalloc.start()
    try:
        _score(env, root / "out", baseline="position_average")
    finally:
        tracemalloc.stop()
        for nm, f in orig.items():
            setattr(S.Scorer, nm, f)
    return got


def test_memory_held_not_linear_in_regions(tmp_path):
    """From 3 to 12 regions: after a group closes, no vector of it (and no per-region fit of the
    position average) is still held, so the held memory grows by a few float64 vectors at most;
    inside a group it grows by at most the vectors of one group (3 targets x at most 24 float64
    vectors, a generous count). The old scorer kept one position-average fit per region of the
    assay, with every training cell's per-region means, after the group (the first assert)."""
    small, big = _traced(tmp_path / "a", 3), _traced(tmp_path / "b", 12)
    vec = 8 * (12 - 3) * RB                       # bytes of one float64 vector's extra bins
    assert big["pa_after_group"] == 0 and big["pa_foreign_cells"] == 0, big
    # held across groups: only the training-average control rate (one vector) of each cell that a
    # later group still reads (2 cells here: the H3K4me3 targets without an observed control), plus
    # allocator slack (Nibi measured 4.02 vectors on 2026-11-14)
    assert big["after_group"] - small["after_group"] < 6 * vec, (small, big, vec)
    assert big["in_group"] - small["in_group"] <= 3 * 24 * vec, (small, big, vec)
