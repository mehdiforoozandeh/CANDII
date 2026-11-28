"""t396 C6: `tools/t396/dump_native.py` on synthetic data.

The synthetic tables of `tests/test_t392_end_to_end.py`, with one change: the overlap counts of
every measurement are rewritten as the read starts convolved with the stored span kernel (rounded),
so that the truth's overlap and start arrays are related as on real data. A tiny read-start run is
trained, its prediction store is made at an inference batch smaller than a region (so a region
has batch edges inside it), and both dump commands run on it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tools" / "t392", REPO / "tools" / "t396"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dump_native as D  # noqa: E402
from conv_sparsity import conv  # noqa: E402

CONFIG = REPO / "tools" / "minicandi" / "configs" / "t392" / "rs_halves_mix_avg_pearson.json"
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K27ac"), (3, "H3K27ac"), (4, "DNase-seq"))
L = 64
RB = 256
INFER_BATCH = 2            # 4 windows per region: two forward batches, one batch edge inside
CONV_FLOOR = 0.05          # normalized MSE of y against conv(s): rounding and region edges only

TRUTH_KEYS = {"mid", "assay", "cell", "stranded", "read_length", "chrom", "start_bin",
              "depth_reads", "shares", "shares_starts", "kernel", "n_reads_strand", "y", "y_pr1",
              "y_pr2", "s", "s_pr1", "s_pr2", "pa_counts", "pa_starts", "pa_counts_ids",
              "pa_starts_ids"}
MODEL_KEYS = {"mu_sub", "n_sub", "mu", "n", "max_abs_rel_diff"}


def _kernel() -> np.ndarray:
    k = np.zeros((2, 17))
    k[0, 8:12] = (1.0, 0.8, 0.5, 0.2)
    k[1, 5:9] = (0.2, 0.5, 0.8, 1.0)
    return k


def _write_starts(path: Path, arrays: dict, mid: str, depth: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_reads = np.zeros(2, np.int64)
    for v in arrays.values():
        n_reads += v.sum(0)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint16))
        f.attrs.update({"schema": 1, "counting_rule": "read_start", "resolution": 25,
                        "strand_columns": "plus,minus", "measurement_id": mid,
                        "depth_reads": depth, "kernel": _kernel(), "n_reads_strand": n_reads})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import build_average_h5 as onion_average_h5
    from minicandi.dataset import read_tsv
    from minicandi.rs_dataset import build_average_h5
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("t396_dump")
    tables = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=(),
                                   chrom_bins={"chr1": 1024, "chr2": 512})
    store = root / "rs_store"
    rng = np.random.default_rng(3)
    w = _kernel()
    for r in read_tsv(tables / "measurements.tsv"):
        mid = r["measurement_id"]
        d = Path(r["counts_path"].split(":", 1)[1]).parent
        halves = {}
        for which in ("pr1", "pr2"):
            with h5py.File(d / f"{which}.h5", "r") as f:
                halves[which] = {}
                for c in f:
                    tot = np.asarray(f[c][:], np.int64)
                    plus = rng.binomial(tot, 0.5)
                    halves[which][c] = np.stack([plus, tot - plus], 1)
        depth = float(r["depth_reads"] or 0) or 1.0
        both = {c: halves["pr1"][c] + halves["pr2"][c] for c in halves["pr1"]}
        _write_starts(store / mid / STORE_FILES["counts"], both, mid, depth)
        for which in ("pr1", "pr2"):
            _write_starts(store / mid / STORE_FILES[which], halves[which], mid, depth)
        # overlap counts = starts convolved with the span kernel (map (1) of the instructions)
        for fname, src in (("pr1.h5", halves["pr1"]), ("pr2.h5", halves["pr2"]), ("counts.h5", both)):
            with h5py.File(d / fname, "r+") as f:
                for c in f:
                    f[c][...] = np.rint(conv(src[c].astype(np.float64), w)).astype(f[c].dtype)
    ref_avg = onion_average_h5(tables, root / "average.h5")
    rs_avg = build_average_h5(ref_avg, store, tables, root / "rs_average.h5")
    evp = root / "eval" / "synth" / "evalset.json"
    ev = E.build(tables, evp, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"] = []
    ev["replicate_pairs"] = []
    ev["pair_regions"] = [1]
    evp.write_text(json.dumps(ev), encoding="utf-8")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.update({"context_bins": L, "param_budget": 2.0e5, "windows_seen": 16, "batch_size": 8,
                "heldout_loss": None, "average_h5": str(ref_avg),
                "read_start_store": str(store), "read_start_average_h5": str(rs_avg)})
    return {"root": root, "tables": tables, "store": store, "evalset": evp, "ev": ev, "cfg": cfg,
            "rs_avg": rs_avg}


@pytest.fixture(scope="module")
def run(env):
    from minicandi.res_dataset import train
    out = env["root"] / "runs" / "rs" / "seed0"
    torch.manual_seed(0)
    train(env["cfg"], env["tables"], 0, out, log_every=1000)
    return out


@pytest.fixture(scope="module")
def pred_store(env, run):
    from minicandi.registry import load_run
    from onion_eval import predict_store as PS
    from onion_eval import score as S
    model, config = load_run(run)
    sc = S.Scorer(env["evalset"], main="model", model=model, config=config,
                  run_tables_sha256=S.run_tables_sha256(run), n_boot=0, guardrail_windows=2,
                  log=lambda *a: None)
    d = env["root"] / "pred_store"
    PS.predict(sc, PS.requests(sc), d, INFER_BATCH, run=run, guardrail=False,
               log=lambda *a: None)
    return d


@pytest.fixture(scope="module")
def truth(env, run):
    out = env["root"] / "truth"
    assert D.main(["truth", "--evalset", str(env["evalset"]), "--tables", str(env["tables"]),
                   "--rs-config", str(run), "--out", str(out), "--shard", "0/1"]) == 0
    return out


@pytest.fixture(scope="module")
def dumped(env, run, pred_store):
    out = env["root"] / "arrays"
    for i in range(2):          # two shards cover every target once
        assert D.main(["model", "--run", str(run), "--evalset", str(env["evalset"]),
                       "--pred-store", str(pred_store), "--out", str(out),
                       "--shard", f"{i}/2"]) == 0
    return out


def _targets(env):
    return [t["target"] for t in env["ev"]["imputation_targets"]
            if t["regime"] == "leave_one_assay_out" and t["split"] == "val"]


def test_truth_keys_shapes_dtypes(env, truth):
    tg = _targets(env)
    assert tg and sorted(p.stem for p in truth.glob("*.npz")) == sorted(tg)
    R = len(env["ev"]["regions"])
    for mid in tg:
        with np.load(truth / f"{mid}.npz") as z:
            assert set(z.files) == TRUTH_KEYS
            assert str(z["mid"]) == mid
            for k in ("y", "y_pr1", "y_pr2"):
                assert z[k].dtype == np.int32 and z[k].shape == (R, RB)
            for k in ("s", "s_pr1", "s_pr2"):
                assert z[k].dtype == np.uint16 and z[k].shape == (R, RB, 2)
            for k in ("pa_counts", "pa_starts"):
                assert z[k].dtype == np.float32 and z[k].shape == (R, RB)
                assert np.all(np.isfinite(z[k])) and np.all(z[k] > 0)
            assert z["kernel"].shape == (2, 17) and z["n_reads_strand"].shape == (2,)
            assert z["shares"].shape == (2,) and z["shares_starts"].shape == (2,)
            assert np.isclose(z["shares"].sum(), 1) and np.isclose(z["shares_starts"].sum(), 1)
            assert z["chrom"].shape == (R,) and z["start_bin"].shape == (R,)
            assert [(str(c), int(s)) for c, s in zip(z["chrom"], z["start_bin"])] == \
                [(c, int(s)) for c, s in env["ev"]["regions"]]
            assert list(z["pa_counts_ids"]) == list(z["pa_starts_ids"]) and len(z["pa_counts_ids"])
            assert bool(z["stranded"]) == (str(z["assay"]) != "DNase-seq")
            # halves sum to the whole, in both spaces
            assert np.array_equal(z["s_pr1"].astype(int) + z["s_pr2"], z["s"].astype(int))


def test_truth_y_is_the_scorer_target_and_conv_of_starts(env, truth):
    from onion_eval import score as S
    sc = S.Scorer(env["evalset"], main="position_average", n_boot=0, log=lambda *a: None)
    for mid in _targets(env):
        with np.load(truth / f"{mid}.npz") as z:
            y, s, w = z["y"], z["s"], z["kernel"]
            job = next(j for j in D.target_jobs(sc) if j["target"] == mid)
            c0, s0 = env["ev"]["regions"][0]
            win = sc.ds.window(c0, int(s0), job["inputs"], [mid])
            t = sc.C.TRACK_INDEX[str(z["assay"])]
            assert np.array_equal(np.asarray(win["y_counts"])[:, t], y[0, :sc.L])
            for r in range(len(y)):
                cv = conv(s[r].astype(np.float64), w)
                nmse = float(np.mean((y[r] - cv) ** 2) / np.mean(cv ** 2))
                assert nmse <= CONV_FLOOR, (mid, r, nmse)


def test_model_keys_shapes_and_gate(env, dumped):
    R = len(env["ev"]["regions"])
    for mid in _targets(env):
        with np.load(dumped / f"{mid}.npz") as z:
            assert MODEL_KEYS <= set(z.files)
            for k in ("mu_sub", "n_sub"):
                assert z[k].dtype == np.float32 and z[k].shape == (R, RB, 2)
            for k in ("mu", "n"):
                assert z[k].dtype == np.float32 and z[k].shape == (R, RB)
            assert float(z["max_abs_rel_diff"]) <= 1e-4
            assert list(z["region_indices"]) == list(range(R))


def test_mu_sub_converted_equals_adapter_output(env, run, dumped):
    """One forward of a whole region (one stitched run) through the adapter against convert_nb
    of the dumped mu_sub of that region."""
    from minicandi import read_start as RS
    from minicandi.registry import load_run
    from onion_eval import score as S
    model, config = load_run(run)
    sc = S.Scorer(env["evalset"], main="model", model=model, config=config,
                  run_tables_sha256=S.run_tables_sha256(run), n_boot=0, log=lambda *a: None)
    for job in D.target_jobs(sc):
        mid = job["target"]
        row = sc._row(mid)
        t = sc.C.TRACK_INDEX[row["assay"]]
        stranded = row["assay"] not in RS.UNSTRANDED_TRACKS
        with np.load(dumped / f"{mid}.npz") as z:
            mu_sub, n_sub = z["mu_sub"], z["n_sub"]
        for r, (c, s) in enumerate(sc.regions):
            sc._set_region(c, s)
            wins = [sc.ds.window(c, s + k * L, job["inputs"], [mid]) for k in range(RB // L)]
            batch = sc.C.collate(wins)
            mu_adapter = sc.model_pred(batch)["mu"][:, :, t].reshape(-1)
            k = model._window_kernels(batch["meta"][0])[t]
            mu_dump, _ = RS.convert_nb(torch.from_numpy(mu_sub[r]), torch.from_numpy(n_sub[r]),
                                       k, stranded=stranded)
            assert D.rel_diff(mu_dump.numpy(), mu_adapter) <= 1e-4, (mid, r)
        if not stranded:
            assert np.all(mu_sub[..., 1] == 0)


def test_model_pair_regions(env, run, pred_store, tmp_path):
    out = tmp_path / "pair"
    assert D.main(["model", "--run", str(run), "--evalset", str(env["evalset"]),
                   "--pred-store", str(pred_store), "--out", str(out), "--regions", "pair",
                   "--shard", "0/4"]) == 0
    files = sorted(out.glob("*.npz"))
    assert files
    with np.load(files[0]) as z:
        assert z["mu_sub"].shape == (1, RB, 2) and list(z["region_indices"]) == [1]
        assert float(z["max_abs_rel_diff"]) <= 1e-4


def test_differing_average_ids_exit_1(env, run, tmp_path):
    bad_avg = tmp_path / "rs_average_bad.h5"
    shutil.copyfile(env["rs_avg"], bad_avg)
    with h5py.File(bad_avg, "r+") as f:
        for g in f.values():
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in g.attrs["measurement_ids"]]
            g.attrs["measurement_ids"] = np.asarray(ids[1:], dtype=h5py.string_dtype())
            g.attrs["n_cells"] = int(g.attrs["n_cells"]) - 1
    rdir = tmp_path / "run"
    rdir.mkdir()
    d = json.loads((run / "run.json").read_text(encoding="utf-8"))
    d["config"]["read_start_average_h5"] = str(bad_avg)
    (rdir / "run.json").write_text(json.dumps(d), encoding="utf-8")
    out = tmp_path / "truth"
    assert D.main(["truth", "--evalset", str(env["evalset"]), "--tables", str(env["tables"]),
                   "--rs-config", str(rdir), "--out", str(out)]) == 1
    assert not list(out.glob("*.npz")) if out.exists() else True
