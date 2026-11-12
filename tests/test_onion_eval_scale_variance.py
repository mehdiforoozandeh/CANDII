"""Tests for tools/onion_eval/scale_variance.py (plan Amendment 3, chunk RVt)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from onion_eval import scale_variance as SV  # noqa: E402

CHAINS = ["1,5,25,125", "1,5,25,50"]


def _direct(x, chain):
    """Second method: explicit block means expanded back to 1 bp, after the span rule."""
    ch = SV.parse_chain(chain)
    x = np.atleast_2d(np.asarray(x, float))
    x = x[:, : (x.shape[1] // ch[-1]) * ch[-1]]
    mu = x.mean()
    out = []
    for t in ch[1:]:
        m = np.repeat(x.reshape(x.shape[0], -1, t).mean(-1), t, axis=1)
        out.append(dict(within=np.mean((x - m) ** 2), between=np.mean((m - mu) ** 2),
                        total=np.mean((x - mu) ** 2)))
    return out


@pytest.mark.parametrize("chain", CHAINS)
def test_block_identity_exact_and_matches_direct(chain):
    rng = np.random.default_rng(0)
    x = rng.poisson(rng.gamma(0.5, 4.0, size=(3, 1130)))     # span 1125 (to 125) or 1100 (to 50)
    rows = SV.block_identity(x, chain)
    assert [(r["from_bp"], r["to_bp"]) for r in rows] == list(zip(SV.parse_chain(chain), SV.parse_chain(chain)[1:]))
    for r, d in zip(rows, _direct(x, chain)):
        assert abs(r["within"] + r["between"] - r["total"]) <= 1e-10 * r["total"]
        for k in ("within", "between", "total"):
            assert r[k] == pytest.approx(d[k], rel=1e-10)
    w = [r["within"] for r in rows]
    assert all(b >= a for a, b in zip(w, w[1:]))              # counts: cumulative below to_bp


def test_span_rule_section_c():
    assert SV.span(409_600, SV.parse_chain("1,5,25,125")) == 409_500
    assert SV.span(409_600, SV.parse_chain("1,5,25,50")) == 409_600
    with pytest.raises(ValueError):
        SV.parse_chain("1,5,25,60")


def test_pool_is_exact_sum():
    x = np.arange(260)
    for b in (5, 25, 50, 125):
        p = SV.pool(x, b)
        assert p.sum() == x[: (260 // b) * b].sum() and len(p) == 260 // b


def test_log1p_step_identity():
    rng = np.random.default_rng(1)
    x = rng.poisson(3.0, size=(2, 1250))
    for r in SV.block_identity(x, "1,5,25,125", scale="log1p"):
        assert abs(r["within"] + r["between"] - r["total"]) <= 1e-10 * r["total"]
        assert 0 <= r["share_raw"] <= 1
        u = np.log1p(SV.pool(x, r["from_bp"]).astype(float))
        k = r["to_bp"] // r["from_bp"]
        assert r["total"] == pytest.approx(u.var(), rel=1e-10)
        m = np.repeat(u.reshape(2, -1, k).mean(-1), k, axis=1)
        assert r["within"] == pytest.approx(np.mean((u - m) ** 2), rel=1e-10)


def test_cross_identity_adds_up_and_self_cross_is_variance():
    rng = np.random.default_rng(2)
    full = rng.poisson(4.0, size=(2, 1000))
    a = rng.binomial(full, 0.5)
    b = full - a
    for scale in SV.SCALES:
        for r in SV.reproducible_share(a, b, "1,5,25,50", scale=scale):
            assert abs(r["cross_within"] + r["cross_between"] - r["cross_total"]) <= 1e-10 * abs(r["cross_total"]) + 1e-12
    same = SV.reproducible_share(a, a, "1,5,25,125", full=a)
    for r, q in zip(same, SV.block_identity(a, "1,5,25,125")):
        assert r["cross_within"] == pytest.approx(q["within"], rel=1e-10)
        assert r["share_reproducible"] == pytest.approx(q["share_raw"], rel=1e-10)


# --- planted signal: the estimator against a known answer -------------------------------------------

R, L, N_CELLS = 4, 25_000, 8


def _truth():
    """Broad piecewise-constant signal (125 bp blocks) times a 10 bp periodic signal."""
    rng = np.random.default_rng(999)
    broad = np.repeat(rng.lognormal(0, 0.5, size=(R, L // 125)), 125, axis=1)
    return 2.0 * broad * (1 + 0.8 * np.cos(2 * np.pi * np.arange(L) / 10))


def _records(lam, seed):
    rng = np.random.default_rng(seed)
    recs = []
    for c in range(N_CELLS):
        full = rng.poisson(lam)
        a = rng.binomial(full, 0.5)                           # pseudoreplicate halves
        for r in SV.reproducible_share(a, full - a, "1,5,25,125"):
            recs.append(dict(assay="H3K4me3", layer="pseudoreplicate", chain="1,5,25,125",
                             scale="counts", from_bp=r["from_bp"], to_bp=r["to_bp"], cell=f"c{c}",
                             share_raw=r["share_raw"], share_reproducible=r["share_reproducible"]))
    return {(g["from_bp"], g["to_bp"]): g for g in SV.aggregate(recs, n_boot=1000, seed=seed)}


def test_planted_10bp_signal_reproducible_share_below_25():
    lam = _truth()
    # the planted share: signal variance below 25 bp over the full track's expected total variance
    # (signal variance + Poisson variance = mean)
    p = SV.block_identity(lam, "1,5,25,125")[1]
    assert (p["from_bp"], p["to_bp"]) == (5, 25)
    planted = p["within"] / (p["total"] + lam.mean())
    g = _records(lam, seed=3)[(5, 25)]
    assert g["n_cells"] == N_CELLS
    assert g["ci_low"] <= planted <= g["ci_high"], (planted, g)
    assert g["ci_high"] - g["ci_low"] < 0.02                  # the interval is informative
    assert g["share_raw"] > g["share_reproducible"] + 0.1     # raw also holds the sampling noise


def test_pure_noise_reproducible_share_is_zero():
    lam = np.full((R, L), 2.0)
    for step, g in _records(lam, seed=4).items():
        assert g["ci_low"] <= 0.0 <= g["ci_high"], (step, g)
        assert g["share_raw"] > 0.7                           # raw counts the noise as signal


# --- the CLI on a tiny region store -----------------------------------------------------------------

def _write_tsv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def test_cli_run_on_tiny_store(tmp_path):
    h5py = pytest.importorskip("h5py")
    store, tables = tmp_path / "res", tmp_path / "eic_res_v1"
    tables.mkdir()
    n_reg, Lr = 3, 2500
    _write_tsv(tables / "regions.tsv", ["region", "chrom", "start_bp", "role"],
               [[0, "chr1", 0, "eval"], [1, "chr1", 5000, "train"], [2, "chr2", 0, "eval"]])
    rng = np.random.default_rng(5)
    lam = 3.0 * (1 + 0.8 * np.cos(2 * np.pi * np.arange(Lr) / 10))

    def put(mid, name, arrs):
        (store / mid).mkdir(parents=True, exist_ok=True)
        with h5py.File(store / mid / f"{name}.h5", "w") as f:
            for i, x in enumerate(arrs):
                f.create_dataset(f"r{i}", data=x.astype(np.uint16))

    meas, pairs = [], []
    for k, cell in enumerate(["cA", "cB", "cC"]):
        exp = f"ENCSR{k}"
        mid = f"{exp}.pooled.uniform.base"
        br = [rng.poisson(np.tile(lam, (n_reg, 1)) / 2) for _ in range(2)]
        full = br[0] + br[1]
        a = rng.binomial(full, 0.5)
        put(mid, "point1bp", list(full))
        put(mid, "point1bp_pr1", list(a))
        put(mid, "point1bp_pr2", list(full - a))
        meas.append([mid, exp, "H3K4me3", "pooled", cell, "val", ""])
        for i in (1, 2):
            put(f"{exp}.br{i}.uniform.base", "point1bp", list(br[i - 1]))
            meas.append([f"{exp}.br{i}.uniform.base", exp, "H3K4me3", f"br{i}", cell, "val", ""])
        pairs.append([f"iso{k}", f"{exp}.br1.uniform.base", f"{exp}.br2.uniform.base", "isogenic",
                      "H3K4me3", cell])
    _write_tsv(tables / "measurements.tsv",
               ["measurement_id", "experiment", "assay", "unit", "cell", "in_eic", "arm_knob"], meas)
    _write_tsv(tables / "pairs.tsv", ["pair_id", "a", "b", "rung", "assay", "cell"], pairs)

    out = tmp_path / "scale_variance.tsv"
    assert SV.main(["run", "--store", str(store), "--tables", str(tables), "--chains", *CHAINS,
                    "--out", str(out), "--n-boot", "200"]) == 0
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert list(rows[0]) == SV.COLUMNS
    got = {(r["layer_or_all"], r["chain"], r["from_bp"], r["to_bp"], r["scale"]): r for r in rows}
    for layer in ("all", "pseudoreplicate", "isogenic", "other_cell_type"):
        r = got[(layer, "1,5,25,125", "5", "25", "counts")]
        assert int(r["n_tracks"]) == 3 and int(r["n_cells"]) >= 2 and r["ci_low"] != ""
    for layer in ("technical", "sequencing"):
        r = got[(layer, "1,5,25,50", "5", "25", "log1p")]
        assert r["n_tracks"] == "0" and r["share_reproducible"] == ""
    # the 10 bp signal is reproducible across the halves and across the replicates
    for layer in ("pseudoreplicate", "isogenic", "other_cell_type"):
        assert float(got[(layer, "1,5,25,125", "5", "25", "counts")]["share_reproducible"]) > 0.05
