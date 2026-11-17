"""Close-out item A7 (resolution 1 to 125 bp): the in-run read-set check of the recount, the
store-only verify, the switch-off identity of the recount outputs, and `res_bp` on rung 3.

The recount fixture is the one of tests/test_onion_data_recount.py (synthetic tagAligns whose
eic_v1 counts.h5 is made with the overlap rule).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tools" / "t112", REPO / "tests", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import recount as RC  # noqa: E402
from test_onion_data_recount import EXP, MID, RB, SHIFT_FL, tree  # noqa: E402,F401
from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.registry import build_model  # noqa: E402


def _run(t, out, extra=(), rc=0):
    args = ["run", "--work", str(t["tmp"] / "work"), "--experiment", EXP, "--regions",
            str(t["tmp"] / "regions.tsv"), "--frag-len", str(SHIFT_FL), "--halves", "1", "--out",
            str(out), "--chrom-sizes", str(t["tmp"] / "genome" / "chrom_sizes.json"), *extra]
    assert RC.main(args) == rc
    return Path(out) / MID


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_switch_off_outputs_identical(tree):
    """--verify-counts changes no data file; without it provenance has no new key."""
    a = _run(tree, tree["tmp"] / "res_a")
    b = _run(tree, tree["tmp"] / "res_b", ["--verify-counts", str(tree["v1m"] / "counts.h5")])
    for name in ("point1bp.h5", "point1bp_pr1.h5", "point1bp_pr2.h5", "counts.h5", "pr1.h5",
                 "pr2.h5"):
        with h5py.File(a / name, "r") as fa, h5py.File(b / name, "r") as fb:
            assert sorted(fa.keys()) == sorted(fb.keys())
            for k in fa.keys():
                assert np.array_equal(fa[k][()], fb[k][()])
    pa = json.loads((a / "provenance.json").read_text())
    pb = json.loads((b / "provenance.json").read_text())
    assert "readset_check" not in pa and "n_reads_in_regions_halves" not in pa
    assert pb["readset_check"]["result"] == "pass"
    assert pb["outputs"] == pa["outputs"]          # the same sha256 of every data file
    assert sum(pb["n_reads_in_regions_halves"].values()) == pb["n_reads_in_regions"]


def test_in_run_check_fails_on_a_dropped_read(tree, capsys):
    pr1 = tree["rs"] / "pr1.tagAlign.gz"
    lines = gzip.open(pr1, "rt").read().splitlines(keepends=True)
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("chr2\t") and int(ln.split("\t")[1]) < RB - 1000)   # eval chr2:0
    with gzip.open(pr1, "wt") as fh:
        fh.writelines(lines[:i] + lines[i + 1:])
    d = _run(tree, tree["tmp"] / "res", ["--verify-counts", str(tree["v1m"] / "counts.h5")],
             rc=1)
    assert "overlap recount differs" in capsys.readouterr().out
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["readset_check"]["result"] != "pass"
    # the store-only verify refuses it
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5"),
                    "--no-rescan"]) == 1


def test_no_rescan_passes_after_the_tagaligns_leave(tree, capsys):
    d = _run(tree, tree["tmp"] / "res", ["--verify-counts", str(tree["v1m"] / "counts.h5")])
    shutil.rmtree(tree["tmp"] / "work")                     # the work folder goes to tape
    capsys.readouterr()
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5"),
                    "--no-rescan"]) == 0
    assert "verify failures: 0" in capsys.readouterr().out
    # a tampered point file is still caught from the store alone
    with h5py.File(d / "point1bp_pr1.h5", "a") as f:
        f["r0"][7] += 1
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5"),
                    "--no-rescan"]) == 1
    assert "sha256 differs" in capsys.readouterr().out


def test_no_rescan_refuses_a_run_without_the_check(tree, capsys):
    d = _run(tree, tree["tmp"] / "res")
    assert RC.main(["verify", "--out", str(d), "--counts", str(tree["v1m"] / "counts.h5"),
                    "--no-rescan"]) == 1
    assert "no passed in-run read-set check" in capsys.readouterr().out


def test_no_rescan_refuses_another_counts_file(tree, capsys):
    d = _run(tree, tree["tmp"] / "res", ["--verify-counts", str(tree["v1m"] / "counts.h5")])
    other = tree["tmp"] / "other_counts.h5"
    shutil.copy(tree["v1m"] / "counts.h5", other)
    with h5py.File(other, "a") as f:
        f["chr1"][0] += 1
    assert RC.main(["verify", "--out", str(d), "--counts", str(other), "--no-rescan"]) == 1
    assert "another counts file" in capsys.readouterr().out


# ----------------------------------------------------------------------------- rung 3, res_bp


def _batch(L, res, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = 2
    return {"x_counts": torch.randint(0, 6, (B, L, C.A), generator=g, dtype=torch.int32),
            "x_avail": torch.ones(B, C.A, dtype=torch.bool),
            "x_bin_mask": torch.rand(B, L, C.A, generator=g) < 0.3,
            "x_dna": torch.randint(0, 5, (B, L * res), generator=g).to(torch.uint8),
            "c_in": torch.randn(B, C.A, C.k_input({}), generator=g) + 24.0,
            "c_out": torch.randn(B, C.A, C.K_META, generator=g) + 24.0,
            "y_avail": torch.ones(B, C.A, dtype=torch.bool)}


@pytest.mark.parametrize("res", [1, 5, 50, 125])
def test_rung3_builds_res_bp_with_the_base_parameter_count(res):
    cfg = {"rung": 3, "context_bins": 64, "param_budget": 2.0e5}
    torch.manual_seed(0)
    base = build_model(cfg)
    torch.manual_seed(0)
    m = build_model({**cfg, "res_bp": res})
    assert m.param_count() == base.param_count()        # the token size adds no parameter
    assert m.dna.res_bp == res
    m.eval()
    with torch.no_grad():
        o = m(_batch(64, res))
    assert o["mu"].shape == (2, 64, C.A) and torch.isfinite(o["mu"]).all()


def test_rung3_res25_is_the_default_model():
    cfg = {"rung": 3, "context_bins": 64, "param_budget": 2.0e5}
    torch.manual_seed(0)
    a = build_model(cfg).eval()
    torch.manual_seed(0)
    b = build_model({**cfg, "res_bp": 25}).eval()
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    bt = _batch(64, 25)
    with torch.no_grad():
        oa, ob = a(bt), b(bt)
    assert torch.equal(oa["mu"], ob["mu"]) and torch.equal(oa["n"], ob["n"])


@pytest.mark.parametrize("key", [{"res_in_sub": 5}, {"res_out_sub": 5}])
def test_rung3_refuses_the_decoupled_arms(key):
    with pytest.raises(M.NotApplicable):
        build_model({"rung": 3, "context_bins": 64, "param_budget": 2.0e5, **key})
    from minicandi.rungs import conv_attn_ae
    with pytest.raises(M.NotApplicable):
        conv_attn_ae.Model({"context_bins": 64, "param_budget": 2.0e5, **key}, width=32)
