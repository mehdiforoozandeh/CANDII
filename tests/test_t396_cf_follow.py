"""t396 B2: tools/t396/cf_follow.py on synthetic tables and stub models (no trained run)."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tools" / "t396"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cf_follow as F  # noqa: E402

L = 64
RB = 256
DEDUP = F.COL["dedup_on"]


# ------------------------------------------------------------------------ pairs on tables only
def _row(mid, exp, assay, cell, layer, knob="", value="", parent="", **kw):
    r = {"measurement_id": mid, "experiment": exp, "assay": assay, "cell": cell, "layer": layer,
         "in_eic": "blind", "arm_knob": knob, "arm_value": value, "parent_measurement_id": parent,
         "depth_reads": "30000000", "run_type": "single-ended", "read_length": "101",
         "dedup": "1", "mapq_threshold": "30"}
    r.update(kw)
    return r


def _tables_7(root: Path) -> Path:
    from minicandi.synth import MEASUREMENT_COLUMNS
    rows = []
    for k in range(7):
        exp, cell = f"ENCSR{k:03d}AAA", f"C{k:02d}"
        base = f"{exp}.pooled.testbed.base"
        rows.append(_row(base, exp, "H3K4me1", cell, "testbed"))
        rows.append(_row(f"{exp}.pooled.testbed.nodedup", exp, "H3K4me1", cell, "testbed",
                         "nodedup", "", base, dedup="0"))
        rows.append(_row(f"{exp}.pooled.testbed.pe", exp, "H3K4me1", cell, "testbed", "se", "pe",
                         base, run_type="paired-ended",
                         read_length="50" if k == 0 else "101",
                         depth_reads="60000000" if k == 1 else "30000000"))
        rows.append(_row(f"{exp}.pooled.testbed.depth0.50", exp, "H3K4me1", cell, "testbed",
                         "depth", "0.50", base, depth_reads="15000000"))
        # a non-testbed nodedup arm: never a controlled pair
        ub = f"{exp}.pooled.uniform.base"
        rows.append(_row(ub, exp, "H3K4me1", cell, "uniform", in_eic="train"))
        rows.append(_row(f"{exp}.pooled.uniform.nodedup", exp, "H3K4me1", cell, "uniform",
                         "nodedup", "", ub, in_eic="train", dedup="0"))
    d = root / "tables7"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MEASUREMENT_COLUMNS), delimiter="\t",
                           lineterminator="\n", restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return d


def test_pairs_gives_28_rows_with_fields(tmp_path):
    tables = _tables_7(tmp_path)
    out = tmp_path / "pairs.tsv"
    assert F.main(["pairs", "--tables", str(tables), "--out", str(out)]) == 0
    rows = F.read_tsv(out)
    assert len(rows) == 28
    assert tuple(rows[0]) == F.PAIR_COLUMNS
    assert {r["covariate"] for r in rows} == {"dedup", "run_type"}
    assert sum(r["direction"] == "base_to_variant" for r in rows) == 14
    assert all(".testbed." in r["source"] and ".testbed." in r["target"] for r in rows)
    fc = {(r["track"], r["covariate"]): r["fields_changed"] for r in rows}
    assert fc[("ENCSR000AAA", "run_type")] == "run_type_pe,read_length"
    assert fc[("ENCSR001AAA", "run_type")] == "log2_depth,run_type_pe"
    assert fc[("ENCSR002AAA", "run_type")] == "run_type_pe"
    assert all(fc[(f"ENCSR{k:03d}AAA", "dedup")] == "dedup_on" for k in range(7))
    for r in rows:
        if r["direction"] == "base_to_variant":
            assert r["source"].endswith(".base")
        else:
            assert r["target"].endswith(".base")


# ------------------------------------------------------------------- synthetic data and stubs
def _write_h5(path: Path, arrays: dict, mid: str, depth: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint32))
        f.attrs.update({"schema": 1, "measurement_id": mid, "depth_reads": depth,
                        "depth_fragments": depth, "resolution": 25, "counting_rule": "synthetic"})


def _kernel() -> np.ndarray:
    k = np.zeros((2, 17))
    k[0, 8:12] = (1.0, 0.8, 0.5, 0.2)
    k[1, 5:9] = (0.2, 0.5, 0.8, 1.0)
    return k


def _write_starts(path: Path, arrays: dict, mid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_reads = np.zeros(2, np.int64)
    for v in arrays.values():
        n_reads += v.sum(0)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint16))
        f.attrs.update({"schema": 1, "counting_rule": "read_start", "resolution": 25,
                        "strand_columns": "plus,minus", "measurement_id": mid,
                        "depth_reads": 1.0, "kernel": _kernel(), "n_reads_strand": n_reads})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import read_tsv
    from minicandi.synth import MEASUREMENT_COLUMNS, make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("t396_cf_follow")
    tables = make_synthetic_tables(root, n_cells=6, val_tracks=((0, "H3K4me3"),),
                                   blind_tracks=((3, "H3K4me3"),),
                                   chrom_bins={"chr1": 1024, "chr2": 512})
    rows = read_tsv(tables / "measurements.tsv")
    blind = next(r for r in rows if r["in_eic"] == "blind" and not r["arm_knob"])
    exp = blind["experiment"]
    src_dir = Path(blind["counts_path"].split(":", 1)[1]).parent
    # deep testbed data (the halves floor beats a 2x scale error): Poisson counts, binomial halves
    drng = np.random.default_rng(11)
    base = {}
    with h5py.File(src_dir / "counts.h5", "r") as f:
        for c in f:
            lam = 40.0 * np.exp(np.sin(np.arange(f[c].shape[0]) / 9.0))
            tot = drng.poisson(lam)
            p1 = drng.binomial(tot, 0.5)
            base[c] = {"counts.h5": tot, "pr1.h5": p1, "pr2.h5": tot - p1}
    new = []
    # nodedup = 2 x base (and twice the depth: log2_depth differs but is never swapped);
    # pe = base exactly, read length 50 (fields run_type_pe, read_length)
    for arm, factor, upd in (("base", 1, {"run_type": "single-ended", "read_length": "100"}),
                             ("nodedup", 2, {"run_type": "single-ended", "read_length": "100",
                                             "dedup": "0", "arm_knob": "nodedup"}),
                             ("pe", 1, {"run_type": "paired-ended", "read_length": "50",
                                        "arm_knob": "se", "arm_value": "pe"})):
        mid = f"{exp}.pooled.testbed.{arm}"
        d = root / "measurements" / mid
        dep = float(blind["depth_reads"]) * factor
        for f in ("counts.h5", "pr1.h5", "pr2.h5"):
            _write_h5(d / f, {c: v[f] * factor for c, v in base.items()}, mid,
                      dep if f == "counts.h5" else dep / 2)
        r = dict(blind, measurement_id=mid, layer="testbed", arm_value="",
                 parent_measurement_id="" if arm == "base" else f"{exp}.pooled.testbed.base",
                 depth_reads=str(int(float(blind["depth_reads"]) * factor)),
                 counts_path=f"h5:{(d / 'counts.h5').resolve()}")
        r.update(upd)
        new.append(r)
    with open(tables / "measurements.tsv", "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MEASUREMENT_COLUMNS), delimiter="\t",
                           lineterminator="\n", restval="")
        for r in new:
            w.writerow({k: r.get(k, "") for k in MEASUREMENT_COLUMNS})
    # read-start store of every row: strands split from each half
    rng = np.random.default_rng(3)
    store = root / "rs_store"
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
        both = {c: halves["pr1"][c] + halves["pr2"][c] for c in halves["pr1"]}
        _write_starts(store / mid / "starts.h5", both, mid)
        for which in ("pr1", "pr2"):
            _write_starts(store / mid / f"starts_{which}.h5", halves[which], mid)
    evp = root / "eval" / "synth" / "evalset.json"
    ev = E.build(tables, evp, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    evp.write_text(json.dumps(ev), encoding="utf-8")
    pairs = F.build_pairs(read_tsv(tables / "measurements.tsv"), F._encoder(tables))
    return {"root": root, "tables": tables, "store": store, "evalset": evp, "ev": ev,
            "pairs": pairs, "exp": exp}


def _g(dedup_on: torch.Tensor) -> torch.Tensor:
    """The stub's covariate law: 2x the reads with dedup off."""
    return 2.0 - dedup_on


class CountStub(torch.nn.Module):
    """mu = source input x g(C' dedup) / g(C dedup) when `follow`, else the input itself; it
    records every C' it is given."""

    def __init__(self, follow: bool):
        super().__init__()
        self.follow, self.calls = follow, []

    def forward(self, b):
        self.calls.append(b["c_out"].clone())
        x = b["x_counts"].double()
        if self.follow:
            x = x * _g(b["c_out"][..., DEDUP].double())[:, None, :] / _g(
                b["c_in"][..., DEDUP].double())[:, None, :]
        return {"mu": x, "n": torch.full_like(x, 10.0)}


class StartStub(torch.nn.Module):
    """The read-start form of `CountStub(follow=True)`: on `x_counts_sub` [B, L, A, 2]."""

    def forward(self, b):
        x = b["x_counts_sub"].double()
        s = _g(b["c_out"][..., DEDUP].double()) / _g(b["c_in"][..., DEDUP].double())
        mu = x * s[:, None, :, None]
        return {"mu_sub": mu, "n_sub": torch.full_like(mu, 10.0)}


def _cfg():
    return {"context_bins": L, "batch_size": 2}


def _predict(env, model, out, seed="0", name="stub", config=None):
    meta = {"model": name, "seed": seed, "run": "stub"}
    return F.predict_all(model, config or _cfg(), env["tables"], env["ev"], env["evalset"],
                         env["pairs"], out, (0, 1), 2, meta, log=lambda *a: None)


@pytest.fixture(scope="module")
def follow_dirs(env):
    root = env["root"] / "pred"
    dirs = []
    for seed in ("0", "1"):
        d = root / "follow" / f"seed{seed}"
        _predict(env, CountStub(True), d, seed, "follow")
        dirs.append(d)
    return dirs


@pytest.fixture(scope="module")
def ignore_dir(env):
    d = env["root"] / "pred" / "ignore" / "seed0"
    _predict(env, CountStub(False), d, "0", "ignore")
    return d


def _table(dirs, **kw):
    table, per_track = F.build_table([str(d) for d in dirs], **kw)
    return table, per_track


def _get(table, **key):
    out = [r for r in table if all(r[k] == v for k, v in key.items())]
    assert len(out) == 1, (key, out)
    return out[0]


# -------------------------------------------------------------------------------------- tests
def test_fixture_has_4_directed_pairs(env):
    assert len(env["pairs"]) == 4
    fc = {p["covariate"]: p["fields_changed"] for p in env["pairs"]}
    assert fc == {"dedup": "dedup_on", "run_type": "run_type_pe,read_length"}


def test_following_stub_scores_near_1(follow_dirs):
    table, _ = _table(follow_dirs)
    for d in F.DIRECTIONS:
        for scope in F.SCOPES:
            r = _get(table, model="follow", space="overlap_count", covariate="dedup",
                     direction=d, scope=scope)
            assert r["follow_score"] == pytest.approx(1.0, abs=1e-6)
            assert r["seed_range"] == pytest.approx(0.0, abs=1e-9)
            assert r["tracks_following"] == 1 and r["tracks_excluded"] == 0
            if scope == "top1pct":
                assert r["follows"] == "yes"
    up = _get(table, model="follow", space="overlap_count", covariate="dedup",
              direction="base_to_variant", scope="genome")
    assert up["total_ratio"] == pytest.approx(2.0, rel=1e-6)


def test_ignoring_stub_scores_0(ignore_dir):
    table, _ = _table([ignore_dir])
    for d in F.DIRECTIONS:
        r = _get(table, model="ignore", space="overlap_count", covariate="dedup", direction=d,
                 scope="top1pct")
        assert r["follow_score"] == pytest.approx(0.0, abs=1e-9)
        assert r["follows"] == "no" and r["tracks_following"] == 0
        assert r["total_ratio"] == pytest.approx(1.0)


def test_denominator_le_0_is_excluded_and_counted(follow_dirs):
    # pe = base exactly and the stub ignores run type: D_source_cov = D_halves, denominator 0
    table, per_track = _table(follow_dirs)
    for d in F.DIRECTIONS:
        r = _get(table, model="follow", space="overlap_count", covariate="run_type",
                 direction=d, scope="genome")
        assert r["tracks_excluded"] == 1
        assert np.isnan(r["follow_score"])
    pt = [x for x in per_track if x["covariate"] == "run_type"]
    assert pt and all(x["excluded"] == 1 for x in pt)


def test_source_covariates_never_reach_the_target_call(env):
    from minicandi.dataset import OnionDataset
    from minicandi.contract import TRACK_INDEX
    cfg = dict(_cfg(), evalset=str(env["evalset"]), region_bins=RB, allow_blind=True)
    ds = OnionDataset(env["tables"], "blind", cfg, seed=0)
    enc = ds.encoder
    for p in env["pairs"]:
        stub = CountStub(True)
        src, tgt = ds.rows[p["source"]], ds.rows[p["target"]]
        t = TRACK_INDEX[tgt["assay"]]
        F.predict_pair(stub, ds, None, ds.rows, enc, p, [tuple(env["ev"]["regions"][0])], RB, L,
                       2, F.other_inputs(list(ds.rows.values()), tgt["cell"], tgt["assay"]))
        v_tgt = np.asarray(enc.c_out(tgt, 0.5), np.float32)
        v_src = np.asarray(enc.c_out(src, 0.5), np.float32)
        named = [F.COL[f] for f in p["fields_changed"].split(",")]
        assert len(stub.calls) % 2 == 0 and stub.calls
        for k in range(0, len(stub.calls), 2):
            ct, cs = stub.calls[k], stub.calls[k + 1]
            assert np.array_equal(ct[:, t].numpy(), np.broadcast_to(v_tgt, ct[:, t].shape))
            # every other track identical between the two calls
            other = [i for i in range(ct.shape[1]) if i != t]
            assert torch.equal(ct[:, other], cs[:, other])
            diff = np.nonzero((ct[:, t] != cs[:, t]).any(0).numpy())[0].tolist()
            assert diff == sorted(named)
            for i in named:
                assert np.allclose(cs[:, t, i].numpy(), v_src[i])
        if p["covariate"] == "dedup":       # depth differs between the rows but is not swapped
            assert v_tgt[F.COL["log2_depth"]] != v_src[F.COL["log2_depth"]]
            assert np.allclose(stub.calls[1][:, t, F.COL["log2_depth"]].numpy(),
                               v_tgt[F.COL["log2_depth"]])


def test_fields_changed_mismatch_refused(env):
    from minicandi.dataset import OnionDataset
    cfg = dict(_cfg(), evalset=str(env["evalset"]), region_bins=RB, allow_blind=True)
    ds = OnionDataset(env["tables"], "blind", cfg, seed=0)
    p = dict(next(p for p in env["pairs"] if p["covariate"] == "run_type"),
             fields_changed="run_type_pe")
    with pytest.raises(ValueError, match="fields_changed"):
        F.predict_pair(CountStub(True), ds, None, ds.rows, ds.encoder, p,
                       [tuple(env["ev"]["regions"][0])], RB, L, 2, [])


def test_read_start_stub_scored_in_both_spaces(env):
    from minicandi.rs_adapter import ReadStartAdapter
    cfg = dict(_cfg(), read_start=True, read_start_store=str(env["store"]))
    model = ReadStartAdapter(StartStub(), cfg, tables=env["tables"])
    d = env["root"] / "pred" / "rs" / "seed0"
    assert _predict(env, model, d, "0", "rs", config=cfg) == len(env["pairs"])
    meta = json.loads((d / "meta.json").read_text())
    assert meta["spaces"] == ["overlap_count", "start_sum"]
    with np.load(d / f"{F.pair_key(env['pairs'][0])}.npz") as z:
        n = 2 * RB
        for s in ("overlap_count", "start_sum"):
            for k in ("mu_tgt", "mu_src", "n_tgt", "n_src", "y", "y1", "x"):
                assert z[f"{k}_{s}"].shape == (n,)
    table, _ = _table([d])
    for d_ in F.DIRECTIONS:
        r = _get(table, model="rs", space="start_sum", covariate="dedup", direction=d_,
                 scope="genome")
        assert r["follow_score"] > 0.95
        r = _get(table, model="rs", space="overlap_count", covariate="dedup", direction=d_,
                 scope="genome")
        assert np.isfinite(r["follow_score"])
    up = _get(table, model="rs", space="start_sum", covariate="dedup",
              direction="base_to_variant", scope="genome")
    assert up["total_ratio"] == pytest.approx(2.0, rel=0.01)


def test_rerun_skips_and_table_cli_with_caps(env, follow_dirs, tmp_path):
    assert _predict(env, CountStub(True), follow_dirs[0], "0", "follow") == 0
    caps = tmp_path / "caps.tsv"
    p = next(p for p in env["pairs"] if p["covariate"] == "dedup"
             and p["direction"] == "variant_to_base")
    caps.write_text("source\ttarget\tcovariate\tgap_closed\n"
                    f"{p['source']}\t{p['target']}\tdedup\t0.7\n", encoding="utf-8")
    out = tmp_path / "tab"
    assert F.main(["table", "--pred", *map(str, follow_dirs), "--caps-table", str(caps),
                   "--out", str(out)]) == 0
    text = (out / "table.tsv").read_text().splitlines()
    assert text[0] == F.LIMIT
    rows = F.read_tsv(out / "table.tsv")
    r = next(x for x in rows if x["covariate"] == "dedup" and x["direction"] == "variant_to_base"
             and x["scope"] == "top1pct" and x["space"] == "overlap_count")
    assert float(r["caps_gap_closed"]) == pytest.approx(0.7) and r["caps_n"] == "1"
    assert r["per_seed"].count(":") == 2
    assert (out / "per_track.tsv").is_file()


def test_table_refuses_missing_pair(env, follow_dirs, tmp_path):
    d = tmp_path / "partial"
    shutil.copytree(follow_dirs[0], d)
    next(d.glob("*.npz")).unlink()
    with pytest.raises(SystemExit, match="no npz"):
        F.build_table([str(d)])
