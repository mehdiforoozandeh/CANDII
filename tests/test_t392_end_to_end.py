"""t392 chunk C10a: the whole read-start path end to end on synthetic data.

Synthetic tables (`minicandi.synth`) plus a synthetic read-start store derived from them; the
t392 training config shrunk (context, budget, windows); `res_dataset.train` for 16 windows;
`registry.load_run` returns the read-start adapter; `score.py` (`--run`, the legacy single-job
path) on a synthetic evaluation set writes `scores.json` whose headline terms are finite; and
`tools/t392/sanity.py` runs on the same run.

Headline terms in scores.json: `selection.imputation_gap_closed` (imputation),
`selection.correction_gap_closed` (correction), `selection.uncertainty_gap_closed`
(calibration). select_rule's `correction_v3_raw` needs an oracle-floor reference and is not
computed here.
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

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tools" / "t392"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from minicandi import contract as C  # noqa: E402

CONFIG = REPO / "tools" / "minicandi" / "configs" / "t392" / "rs_halves_mix_avg_pearson.json"
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K27ac"), (3, "H3K27ac"), (4, "DNase-seq"))
L = 64
QUIET = dict(log=lambda *a: None)
HEADLINE = {"imputation": "imputation_gap_closed", "correction": "correction_gap_closed",
            "calibration": "uncertainty_gap_closed"}


def _kernel() -> np.ndarray:
    """[2, 17]: a span kernel that covers the start bin and the next 3 bins on the read's side."""
    k = np.zeros((2, 17))
    k[0, 8:12] = (1.0, 0.8, 0.5, 0.2)          # plus strand: reads extend downstream
    k[1, 5:9] = (0.2, 0.5, 0.8, 1.0)           # minus strand: upstream
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
                        "depth_reads": depth, "kernel": _kernel(),
                        "n_reads_strand": n_reads})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import build_average_h5 as onion_average_h5
    from minicandi.dataset import read_tsv
    from minicandi.rs_dataset import build_average_h5
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("rs_e2e")
    tables = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=(),
                                   chrom_bins={"chr1": 1024, "chr2": 512})
    # the store: each measurement's 25 bp counts split binomially into two strands; pr1 and pr2
    # are split from the tables' own pr1 and pr2, so starts = pr1 + pr2 as in the real store
    store = root / "rs_store"
    rng = np.random.default_rng(3)
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
    ref_avg = onion_average_h5(tables, root / "average.h5")
    rs_avg = build_average_h5(ref_avg, store, tables, root / "rs_average.h5")
    evp = root / "eval" / "synth" / "evalset.json"
    ev = E.build(tables, evp, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"] = [p for p in ev["correction_pairs"]
                              if p["never_trained"] and not p["subset_pair"]]
    ev["replicate_pairs"] = []
    # 5 training tracks per assay are too few for the builder's rule; declare the two histone
    # validation assays imputation-scorable so that the imputation and calibration terms exist
    ev["scorable"]["imputation"] = ["H3K27ac", "H3K4me3"]
    evp.write_text(json.dumps(ev), encoding="utf-8")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.update({"context_bins": L, "param_budget": 2.0e5, "windows_seen": 16, "batch_size": 8,
                "heldout_loss": None, "average_h5": str(ref_avg),
                "read_start_store": str(store), "read_start_average_h5": str(rs_avg)})
    return {"root": root, "tables": tables, "store": store, "evalset": evp, "ev": ev, "cfg": cfg}


@pytest.fixture(scope="module")
def run(env):
    from minicandi.res_dataset import train
    out = env["root"] / "runs" / "rs" / "seed0"
    torch.manual_seed(0)
    info = train(env["cfg"], env["tables"], 0, out, log_every=1000)
    return {"out": out, "info": info}


@pytest.fixture(scope="module")
def scored(env, run):
    from onion_eval import score as S
    out = run["out"] / "score"
    S.score(env["evalset"], out, run=run["out"], n_boot=0, guardrail_windows=2, **QUIET)
    return json.loads((out / "scores.json").read_text(encoding="utf-8"))


def test_config_is_the_t392_config_shrunk(env):
    ref = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert ref["read_start"] is True and ref["rung"] == 4
    changed = {k for k in ref if env["cfg"][k] != ref[k]}
    assert changed <= {"context_bins", "param_budget", "windows_seen", "heldout_loss",
                       "average_h5", "read_start_store", "read_start_average_h5"}


def test_train_runs_16_windows(run):
    info = run["info"]
    assert info["done"] and info["steps"] == 2
    assert math.isfinite(info["last_train_loss"])
    assert (run["out"] / "ckpt.pt").is_file() and (run["out"] / "DONE").is_file()


def test_load_run_returns_the_read_start_adapter(env, run):
    from minicandi.registry import load_run
    from minicandi.rs_adapter import ReadStartAdapter
    model, config = load_run(run["out"])
    assert isinstance(model, ReadStartAdapter) and config["read_start"]
    assert config["rung"] == 4 and config["context_bins"] == L


def test_score_writes_finite_headline_terms(env, scored):
    sel = scored["selection"]
    vals = {name: sel[key] for name, key in HEADLINE.items()}
    print("headline terms:", json.dumps(vals))
    for name, v in vals.items():
        assert isinstance(v, (int, float)) and math.isfinite(v), (name, v, sel)
    assert scored["n_targets"] == len(env["ev"]["imputation_targets"]) > 0
    assert scored["n_pairs"] == len(env["ev"]["correction_pairs"]) > 0


def test_sanity_tool_reports_finite_numbers(env, run, tmp_path):
    import sanity
    out = tmp_path / "sanity.json"
    assert sanity.main(["--run", str(run["out"]), "--evalset", str(env["evalset"]),
                        "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    print("sanity:", res["nb_deviance_mean"], res["pearson_log1p"])
    assert math.isfinite(res["nb_deviance_mean"]) and res["nb_deviance_mean"] >= 0
    assert math.isfinite(res["pearson_log1p"]) and -1 <= res["pearson_log1p"] <= 1
    n_regions = len(env["ev"]["regions"])
    assert res["n_regions"] == n_regions and res["n_targets"] == len(env["ev"]["imputation_targets"])


def test_sanity_deviance_is_the_nb_unit_deviance():
    import sanity
    y = np.array([0.0, 3.0, 10.0])
    mu = np.array([1.0, 3.0, 4.0])
    n = np.array([2.0, 5.0, 1.5])
    ref = []
    for yi, mi, ni in zip(y, mu, n):
        t1 = yi * math.log(yi / mi) if yi > 0 else 0.0
        ref.append(2 * (t1 - (yi + ni) * math.log((yi + ni) / (mi + ni))))
    assert np.allclose(sanity.nb_deviance(y, mu, n), ref)
    assert sanity.nb_deviance(np.array([3.0]), np.array([3.0]), np.array([5.0]))[0] == 0.0
