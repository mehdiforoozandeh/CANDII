"""Chunk RM: miniCANDI at another bin size, on a small synthetic region store.

The store is written with the recount tool's own writer (`recount.write_region_h5`): every 25 bp
count of the synthetic tables is spread over its 25 bases at random, so `pool(point1bp, 25)`
equals the tables' counts.h5 inside each region, as `recount.py verify` demands of the real store.
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
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.dataset import OnionDataset, read_tsv, tables_sha256  # noqa: E402
from minicandi.registry import build_model, load_run  # noqa: E402
from minicandi.res_adapter import ResAdapter  # noqa: E402
from minicandi.res_dataset import POINT_FILES, ResDataset, res_total_loss  # noqa: E402
from minicandi.res_dataset import train as res_train  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_data import recount as R  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import res_score as RS  # noqa: E402
from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"))
RB = R.REGION_BINS                       # 16,384 bins of 25 bp: one store region
EB = 2048                                # eval regions of 2,048 bins (51.2 kb) inside it: the
                                         # scorer's CRPS costs ~1 s per 1,000 bins and target
QUIET = dict(log=lambda *a: None)


class InputMean(C.MiniCandi):
    """Reads only the visible input counts (any token size): mu = target size factor x (0.5 + the
    mean over visible tracks of count / input size factor); n = 4. Element-wise, so its output
    does not depend on how windows are batched."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, b):
        vis = (b["x_avail"][:, None, :] & ~b["x_bin_mask"]).float()
        s_in = torch.exp2(b["c_in"][..., 0] - 24.0)[:, None, :]
        s_out = torch.exp2(b["c_out"][..., 0] - 24.0)[:, None, :]
        r = (b["x_counts"].float() * vis / s_in.clamp_min(1e-6)).sum(-1, keepdim=True)
        r = r / vis.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (s_out * (0.5 + r) * self.w).clamp_min(1e-3)
        return {"n": torch.full_like(mu, 4.0), "mu": mu}


class Const(C.MiniCandi):
    def __init__(self, c=2.0, k=4.0):
        super().__init__()
        self.c, self.k = c, k

    def forward(self, b):
        shape = b["x_counts"].shape
        return {"n": torch.full(shape, self.k), "mu": torch.full(shape, self.c)}


def _write_tsv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=(),
                              chrom_bins={"chr1": 2 * RB, "chr2": RB})
    rows = [r for r in read_tsv(t / "measurements.tsv") if r["arm_knob"] == ""]
    cols = list(read_tsv(t / "measurements.tsv")[0])
    _write_tsv(t / "measurements.tsv", rows, cols)
    with open(t / "pairs.tsv", encoding="utf-8") as fh:
        header = fh.readline()
    (t / "pairs.tsv").write_text(header, encoding="utf-8")
    evp = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, evp, n_regions=1, n_fit_regions=1, region_bins=EB, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    # 3 training tracks per assay are too few for the builder's rule; the test declares both
    # validation assays imputation-scorable so that headline 1 is defined
    ev["scorable"]["imputation"] = ["H3K27ac", "H3K4me3"]
    evp.write_text(json.dumps(ev), encoding="utf-8")
    ec, es = ev["regions"][0]
    store_eval = [ec, es // RB * RB]                     # the store region holding the eval region
    train = [b for b in E.candidate_blocks(root / "genome", RB) if b != store_eval]
    regions = ([{"region": 0, "chrom": ec, "start_bp": store_eval[1] * 25, "role": "eval"}]
               + [{"region": 1, "chrom": train[0][0], "start_bp": train[0][1] * 25,
                   "role": "train"}])
    _write_tsv(t / "regions.tsv", [{k: str(v) for k, v in r.items()} for r in regions],
               R.REGION_COLUMNS)
    rng = np.random.default_rng(7)
    for r in rows:
        d = Path(r["counts_path"][3:]).parent
        for which, name in POINT_FILES.items():
            src = d / ("counts.h5" if which == "counts" else f"{which}.h5")
            parts = []
            with h5py.File(src, "r") as f:
                for g in regions:
                    c = np.asarray(f[g["chrom"]][g["start_bp"] // 25:g["start_bp"] // 25 + RB],
                                   np.int64)
                    pos = np.repeat(np.arange(RB) * 25, c) + rng.integers(0, 25, int(c.sum()))
                    parts.append(np.bincount(pos, minlength=RB * 25))
            R.write_region_h5(d / name, np.concatenate(parts), regions, {"rule": "test"})
    return {"root": root, "tables": t, "evalset": evp, "ev": ev, "regions": regions}


def _target_job(env):
    return env["ev"]["imputation_targets"][0]


# --------------------------------------------------------------------------- components, registry

def test_res_defaults_build_the_same_components():
    assert M.component_kwargs({}) == M.component_kwargs({"res_bp": 25})
    assert M.local_windows() == M.LOCAL_WINDOWS == M.local_windows(25)
    assert M.local_windows(5) == (200, 2000) and M.local_windows(125) == (8, 80)
    torch.manual_seed(0)
    a = M.DnaTower("own", 8)
    torch.manual_seed(0)
    b = M.DnaTower("own", 8, res_bp=25)
    x = torch.randint(0, 5, (2, 64 * 25)).to(torch.uint8)
    assert torch.equal(a(x), b(x))
    t5 = M.DnaTower("own", 8, **M.component_kwargs({"res_bp": 5})["dna"])
    assert t5(torch.randint(0, 5, (2, 64 * 5)).to(torch.uint8)).shape == (2, 64, 8)
    with pytest.raises(ValueError):
        M.component_kwargs({"res_bp": 10})
    with pytest.raises(ValueError):
        M.res_subs_of({"res_in_sub": 5, "res_bp": 5})


def test_other_rungs_refuse_the_resolution_keys():
    for rung in (1, 3, 5, 6):
        for key in ({"res_bp": 5}, {"res_in_sub": 5}, {"res_out_sub": 5}):
            with pytest.raises(M.NotApplicable):
                build_model({"rung": rung, **key})


# --------------------------------------------------------------------------- dataset

def test_res25_dataset_equals_onion_dataset(env):
    job = _target_job(env)
    chrom, s25 = env["ev"]["regions"][0]
    cfg = {"context_bins": 1024, "evalset": str(env["evalset"])}
    od = OnionDataset(env["tables"], "val", cfg)
    rd = ResDataset(env["tables"], "val", cfg)
    m = InputMean()
    for k in (0, 7, 15):
        a = od.window(chrom, s25 + k * 1024, job["inputs"], [job["target"]])
        b = rd.window(chrom, (s25 + k * 1024) * 25, job["inputs"], [job["target"]])
        for key in a:
            if key != "meta":
                assert torch.equal(a[key], b[key]), key
        assert b["meta"]["start_bin"] == s25 + k * 1024
        ba, bb = C.collate([a]), C.collate([b])
        C.validate_batch(bb)
        assert torch.equal(m(ba)["mu"], m(bb)["mu"])


def test_res_dataset_tokens_and_train_regions(env):
    train_r = env["regions"][1]
    for res, L in ((1, 256), (5, 64), (50, 64), (125, 32)):
        ds = ResDataset(env["tables"], "train", {"context_bins": L, "res_bp": res,
                                                 "batch_size": 2}, seed=1)
        b = ds.batch(2, np.random.default_rng(0))
        C.validate_batch(b, res_bp=res)
        for meta in b["meta"]:
            assert meta["chrom"] == train_r["chrom"]
            assert train_r["start_bp"] <= meta["start_bp"]
            assert meta["start_bp"] + L * res <= train_r["start_bp"] + R.REGION_BP
        # counts are exact sums of the 1 bp store
        meta = b["meta"][0]
        t = next(i for i, m in enumerate(meta["measurements_out"]) if m)
        v = ds.read_bp(meta["measurements_out"][t], meta["chrom"], meta["start_bp"], L * res,
                       meta["target_file"])
        assert np.array_equal(R.pool(v, res), b["y_counts"][0, :, t].numpy())
    vds = ResDataset(env["tables"], "val", {"context_bins": 32, "res_bp": 125,
                                            "evalset": str(env["evalset"]), "region_bins": EB})
    items = vds.evalset_items()
    s0 = env["ev"]["regions"][0][1] * 25
    assert items[0]["start_bin"] == s0                    # tiled from the region's first base
    assert all((it["start_bin"] - s0) % 4000 == 0 for it in items)
    assert len({it["start_bin"] for it in items}) == EB * 25 // 4000
    assert max(it["start_bin"] for it in items) + 4000 <= s0 + EB * 25


# --------------------------------------------------------------------------- adapter

def _batch25(env, L=64):
    job = _target_job(env)
    chrom, s25 = env["ev"]["regions"][0]
    od = OnionDataset(env["tables"], "val", {"context_bins": L})
    bm = np.zeros((L, C.A), bool)
    bm[::3, C.TRACK_INDEX["H3K27ac"]] = True
    wins = [od.window(chrom, s25 + k * L, job["inputs"], [job["target"]], x_bin_mask=bm)
            for k in range(2)]
    return C.collate(wins)


def test_constant_5bp_model_gives_exact_sums_at_25bp(env):
    b = _batch25(env)
    ad = ResAdapter(Const(2.0, 4.0), {"res_bp": 5, "context_bins": 64}, tables=env["tables"])
    o = ad(b)
    assert o["mu"].shape == b["x_counts"].shape
    assert torch.equal(o["mu"], torch.full_like(o["mu"], 10.0))       # 5 x 2
    assert torch.equal(o["n"], torch.full_like(o["n"], 20.0))         # (5c)^2 / (5 c^2 / k)
    # a fine model that reads its input: 1 bp tokens summed back give the 25 bp input
    class Echo(C.MiniCandi):
        def forward(self, fb):
            vis = fb["x_avail"][:, None, :] & ~fb["x_bin_mask"]
            return {"n": torch.full(fb["x_counts"].shape, 1e4),
                    "mu": fb["x_counts"].float() * vis + 1.0}
    o1 = ResAdapter(Echo(), {"res_bp": 1, "context_bins": 160}, tables=env["tables"])(b)
    vis = b["x_avail"][:, None, :] & ~b["x_bin_mask"]
    assert torch.allclose(o1["mu"], b["x_counts"].float() * vis + 25.0)
    # hidden bins: the adapter never reads them (a perturbed hidden input moves nothing)
    pert = dict(b, x_counts=torch.where(b["x_bin_mask"], torch.full_like(b["x_counts"], 77),
                                        b["x_counts"]))
    assert torch.equal(ResAdapter(Echo(), {"res_bp": 5, "context_bins": 64},
                                  tables=env["tables"])(pert)["mu"],
                       ResAdapter(Echo(), {"res_bp": 5, "context_bins": 64},
                                  tables=env["tables"])(b)["mu"])


# --------------------------------------------------------------------------- scoring

@pytest.fixture(scope="module")
def scored(env, tmp_path_factory):
    """InputMean scored by score.py and by res_score (predict, then the 25 bp table)."""
    run = tmp_path_factory.mktemp("runs") / "res25" / "seed0"
    cfg = {"context_bins": 1024, "batch_size": 8, "res_bp": 25}
    sc = S.score(env["evalset"], run / "score", model=InputMean(), config=cfg,
                 run_tables_sha256=tables_sha256(env["tables"]), n_boot=0, guardrail_windows=2,
                 **QUIET)
    RS.predict(run, env["tables"], env["evalset"], model=InputMean(), config=cfg, **QUIET)
    tab = RS.table(run, 25, run / "table25.json", n_boot=0, **QUIET)
    return {"run": run, "score": sc, "table": tab, "cfg": cfg}


def test_table_at_25bp_equals_score_py(scored):
    sc, ind = scored["score"], scored["table"]["bounds"]["independent"]
    pairs = {"imputation_gap_closed": sc["selection"]["imputation_gap_closed"],
             "uncertainty_gap_closed": sc["selection"]["uncertainty_gap_closed"],
             "pit_deviation_pooled": sc["pit_deviation_pooled"]["macro"],
             "count_crps": sc["count_crps"]["macro"],
             "count_crps_oracle_scaled": sc["count_crps_oracle_scaled"]["macro"],
             "count_crps_scale_error": sc["count_crps_scale_error"]["macro"]}
    assert pairs["imputation_gap_closed"] is not None and pairs["pit_deviation_pooled"] is not None
    for k, v in pairs.items():
        if v is None:
            assert ind[k] is None, k
        else:
            assert abs(ind[k] - v) <= 1e-6, (k, ind[k], v)
    # at 25 bp from a 25 bp output the two size bounds coincide
    com = scored["table"]["bounds"]["comonotone"]
    assert abs(com["count_crps"] - ind["count_crps"]) <= 1e-9


def test_table_at_50bp_pools_and_bounds_differ(env, scored, tmp_path):
    tab = RS.table(scored["run"], 50, tmp_path / "t50.json", n_boot=0, **QUIET)
    assert tab["haar_depth_J"] == 10 and tab["span_bp_per_region"] == EB * 25   # 1,024 x 50 bp
    ind, com = tab["bounds"]["independent"], tab["bounds"]["comonotone"]
    assert ind["count_crps"] is not None and com["count_crps"] is not None
    assert ind["count_crps"] != com["count_crps"]
    assert RS.span_of(50, RB) == (16384, 13)
    assert RS.span_of(125, RB) == (10240, 11) and RS.span_of(250, RB) == (10240, 10)


def test_table_refuses_and_never_predicts(env, scored, tmp_path):
    with pytest.raises(FileNotFoundError):
        RS.table(tmp_path / "no_run", 25)
    assert RS.main(["table", "--run", str(tmp_path / "no_run"), "--at", "25",
                    "--out", str(tmp_path / "x.json")]) == 1
    run50 = tmp_path / "res50"
    cfg = {"context_bins": 64, "res_bp": 50}
    RS.predict(run50, env["tables"], env["evalset"], model=InputMean(), config=cfg, **QUIET)
    info = json.loads((run50 / RS.CACHE / RS.INFO).read_text())
    assert info["output_bin_bp"] == 50 and info["cache_bin_bp"] == 50
    with pytest.raises(S.ScoreRefused):
        RS.table(run50, 125)
    with pytest.raises(S.ScoreRefused):
        RS.table(run50, 25)
    assert RS.main(["table", "--run", str(run50), "--at", "125",
                    "--out", str(tmp_path / "y.json")]) == 2
    # one cache bin per 50 bp of the eval region
    z = np.load(run50 / RS.CACHE / f"{_target_job(env)['target']}.npz")
    assert z["mu_sum"].shape == (1, EB * 25 // 50)


# --------------------------------------------------------------------------- the decoupled arms

@pytest.mark.parametrize("rung", [2, 4])
@pytest.mark.parametrize("arm", ["in5_out25", "in25_out5"])
def test_decoupled_arms_train_two_steps(env, tmp_path, rung, arm):
    sub = {"in5_out25": {"res_in_sub": 5}, "in25_out5": {"res_out_sub": 5}}[arm]
    cfg = {"rung": rung, "windows_seen": 4, "batch_size": 2, "context_bins": 32,
           "param_budget": 2.0e5, "res_bp": 25, **sub}
    out = tmp_path / f"r{rung}_{arm}"
    info = res_train(cfg, env["tables"], 0, out)
    assert info["steps"] == 2 and info["done"]
    assert info["flops_per_window"] > 0 and np.isfinite(info["last_train_loss"])
    assert json.loads((out / "run.json").read_text())["flops_per_window"] > 0
    model, config = load_run(out)
    ds = ResDataset(env["tables"], "val", {**config, "evalset": str(env["evalset"]),
                                           "region_bins": EB})
    b = C.collate([ds.item_window(it) for it in ds.evalset_items()[:2]])
    inner = model.inner if isinstance(model, ResAdapter) else model
    with torch.no_grad():
        o = inner(b)
    assert o["mu"].shape == b["x_counts"].shape and torch.isfinite(o["mu"]).all()
    if arm == "in5_out25":
        assert isinstance(model, ResAdapter)              # load_run presents it at 25 bp
        hid = (b["x_bin_mask"] | ~b["x_avail"][:, None, :])[..., None].expand_as(b["x_counts_sub"])
        pert = dict(b, x_counts_sub=torch.where(hid, torch.full_like(b["x_counts_sub"], 99),
                                                b["x_counts_sub"]))
        with torch.no_grad():
            assert torch.equal(inner(pert)["mu"], o["mu"])   # hidden sub-bins never reach it
    else:
        assert o["mu_sub"].shape == b["x_counts"].shape + (5,)
        assert torch.allclose(o["mu_sub"].sum(-1), o["mu"])
        loss, _ = res_total_loss(o, b, config)
        assert torch.isfinite(loss)


@pytest.mark.parametrize("rung", [2, 4])
@pytest.mark.parametrize("res", [1, 5, 50, 125])
def test_res_arms_build_and_run(env, rung, res):
    L = {1: 256, 5: 64, 50: 32, 125: 32}[res]
    cfg = {"rung": rung, "context_bins": L, "param_budget": 2.0e5, "res_bp": res}
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    ds = ResDataset(env["tables"], "train", {**cfg, "batch_size": 2}, seed=0)
    b = ds.batch(2, np.random.default_rng(1))
    with torch.no_grad():
        o = model(b)
    assert o["mu"].shape == (2, L, C.A)
    assert C.wiring_check(model, b)["pass"]
