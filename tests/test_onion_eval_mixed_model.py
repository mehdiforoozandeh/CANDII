"""Tests for tools/onion_eval/mixed_model.py (chunk MM): REML variance components on simulated data."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny matrices; threads only add overhead

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import mixed_model as mm  # noqa: E402


def simulate(seed, G=12, E=2, R=2, H=2, K=200, s=(0.5, 0.2, 0.15), se=0.15, beta=0.3, depth=True):
    """Nested cell / experiment / replicate with H observations per replicate (one library each)."""
    rng = np.random.default_rng(seed)
    lab = {f: [] for f in mm.FACTORS}
    Y, d = [], []
    mu = rng.normal(0, 1, K)
    for g in range(G):
        ug = rng.normal(0, np.sqrt(s[0]), K)
        for e in range(E):
            ue = rng.normal(0, np.sqrt(s[1]), K)
            for r in range(R):
                ur = rng.normal(0, np.sqrt(s[2]), K)
                dep = rng.uniform(20, 24) if depth else 22.0
                for _ in range(H):
                    Y.append(mu + ug + ue + ur + beta * dep + rng.normal(0, np.sqrt(se), K))
                    d.append(dep)
                    lab["cell"].append(g)
                    lab["experiment"].append((g, e))
                    lab["replicate"].append((g, e, r))
                    lab["library"].append((g, e, r))
    Y = np.array(Y)
    truth = np.array(list(s) + [se])
    return Y, lab, np.array(d), truth / truth.sum()


def test_known_components_recovered_within_interval_and_library_flagged():
    Y, lab, d, truth = simulate(0)
    rows, beta = mm.decompose(Y @ Y.T, Y.sum(1), Y.shape[1], lab, d, n_boot=60, seed=0)
    by = {r["factor"]: r for r in rows}
    for f, t in zip(["cell", "experiment", "replicate", "residual"], truth):
        r = by[f]
        assert r["estimable"] == 1
        assert r["ci_low"] <= t <= r["ci_high"], (f, t, r)
        assert abs(r["variance_fraction"] - t) < 0.05
    # one library per replicate: library is not separable from replicate
    assert by["library"]["estimable"] == 0
    assert "confounded with replicate" in by["library"]["note"]
    assert np.isnan(by["library"]["variance_fraction"])
    assert abs(beta - 0.3) < 0.02
    assert sum(r["variance_fraction"] for r in rows if r["estimable"]) == pytest.approx(1.0)


def test_one_experiment_per_cell_is_not_estimable():
    Y, lab, d, _ = simulate(1, E=1, s=(0.5, 0.0, 0.15))
    rows, _ = mm.decompose(Y @ Y.T, Y.sum(1), Y.shape[1], lab, d, n_boot=0)
    by = {r["factor"]: r for r in rows}
    assert by["experiment"]["estimable"] == 0
    assert "confounded with cell" in by["experiment"]["note"]
    assert by["cell"]["estimable"] == 1 and by["replicate"]["estimable"] == 1


def test_estimability_rules():
    lab = {"cell": [0, 0, 1, 1], "experiment": [0, 0, 1, 1], "replicate": [0, 1, 2, 3],
           "library": [0, 1, 2, 3]}
    est = mm.estimability(lab)
    assert est["cell"][0] == 1
    assert est["experiment"][0] == 0 and "cell" in est["experiment"][1]
    assert est["replicate"][0] == 0 and "residual" in est["replicate"][1]
    assert est["library"][0] == 0
    assert mm.estimability({"cell": [0, 0], "experiment": [0, 1], "replicate": [0, 1],
                            "library": [0, 1]})["cell"][0] == 0


def test_reml_equals_anova_in_balanced_design():
    """Second method: in a balanced nested design (no depth slope) REML equals the ANOVA
    (method of moments) estimator when every estimate is positive."""
    G, E, H = 10, 3, 2
    Y, lab, _, _ = simulate(2, G=G, E=E, R=1, H=H, s=(0.4, 0.3, 0.0), se=0.3, depth=False)
    lab = {f: (lab[f] if f in ("cell", "experiment") else list(range(len(Y)))) for f in mm.FACTORS}
    K = Y.shape[1]
    Yc = Y - Y.mean(0)  # remove the per-coefficient mean
    A = Yc.reshape(G, E, H, K)
    m_ge = A.mean(2)
    m_g = A.mean((1, 2))
    ms_e = ((A - m_ge[:, :, None]) ** 2).sum() / (K * G * E * (H - 1))
    ms_exp = H * ((m_ge - m_g[:, None]) ** 2).sum() / (K * G * (E - 1))
    ms_cell = E * H * (m_g ** 2).sum() / (K * (G - 1))
    anova = np.array([(ms_cell - ms_exp) / (E * H), (ms_exp - ms_e) / H, ms_e])
    assert np.all(anova > 0)
    Zs = [mm._incidence(lab["cell"]), mm._incidence(lab["experiment"])]
    sig, beta, _ = mm.reml_fit(Y @ Y.T, Y.sum(1), K, Zs, depth=None)
    assert np.isnan(beta)
    np.testing.assert_allclose(sig, anova, rtol=1e-4)


MEAS_COLS = ["measurement_id", "experiment", "assay", "assay_class", "cell", "eic_track", "in_eic",
             "unit", "bio_rep", "tech_rep", "library", "layer", "arm_knob", "arm_value",
             "parent_measurement_id", "reads_subset_of", "depth_reads", "depth_fragments", "run_type",
             "read_length", "platform", "lab", "pipeline", "pipeline_version", "image_digest", "dedup",
             "mapq_threshold", "antibody", "donor", "control_measurement_id", "has_pseudoreps",
             "has_signal", "nsc", "rsc", "nrf", "pbc1", "pbc2", "frip", "frag_len", "counts_path",
             "sha256"]


def _write_h5(path, rng, rate, n_bins):
    import h5py
    with h5py.File(path, "w") as f:
        f["chr1"] = rng.poisson(rate, n_bins).astype(np.uint16)


def test_cli_end_to_end_on_tiny_tables(tmp_path):
    rng = np.random.default_rng(0)
    n_bins = 2 * 2 ** mm.J
    base = rng.gamma(2.0, 2.0, n_bins)
    rows = []
    for c in range(4):
        cell_rate = base * rng.gamma(4.0, 0.25, n_bins)
        for br in (1, 2):
            exp = f"ENCSR{c:03d}"
            mid = f"{exp}.br{br}.uniform.base"
            d = tmp_path / "m" / mid
            d.mkdir(parents=True)
            rate = cell_rate * rng.gamma(10.0, 0.1, n_bins)
            for h in ("counts", "pr1", "pr2"):
                _write_h5(d / f"{h}.h5", rng, rate if h == "counts" else rate / 2, n_bins)
            role = "blind" if c == 3 else ("val" if c == 2 else "train")
            r = {k: "" for k in MEAS_COLS}
            r.update(measurement_id=mid, experiment=exp, assay="H3K4me3", assay_class="narrow_histone",
                     cell=f"C{c}", in_eic=role, unit=f"br{br}", bio_rep=str(br), layer="uniform",
                     arm_knob="", depth_reads=str(int(rate.sum())), lab=f"lab{c}", has_pseudoreps="1",
                     counts_path="h5:" + str(d / "counts.h5"))
            rows.append(r)
    tables = tmp_path / "tables"
    tables.mkdir()
    with open(tables / "measurements.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, MEAS_COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ev = tmp_path / "evalset.json"
    ev.write_text(json.dumps({"tables": str(tables), "regions": [["chr1", 0], ["chr1", 2 ** mm.J]]}))
    out = tmp_path / "mm.tsv"
    assert mm.main(["run", "--tables", str(tables), "--evalset", str(ev), "--band", "peak",
                    "--out", str(out), "--n-boot", "10"]) == 0
    with open(out, newline="") as fh:
        res = list(csv.DictReader(fh, delimiter="\t"))
    assert list(res[0]) == mm.COLUMNS
    by = {r["factor"]: r for r in res}
    assert by["cell"]["n_levels"] == "3"  # the blind cell is never read
    assert by["residual"]["n_levels"] == "12"  # 3 cells x 2 replicates x 2 halves
    assert by["experiment"]["estimable"] == "0" and "cell" in by["experiment"]["note"]
    assert by["library"]["estimable"] == "0"
    assert by["lab"]["estimable"] == "0" and "not separable" in by["lab"]["note"]
    assert by["replicate"]["estimable"] == "1"
    assert "REML" in by["residual"]["note"]
    fr = sum(float(r["variance_fraction"]) for r in res if r["estimable"] == "1"
             and r["factor"] != "log2_depth")
    assert fr == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(SystemExit):
        mm.main(["run", "--tables", str(tables), "--evalset", str(ev), "--band", "nope",
                 "--out", str(out)])
