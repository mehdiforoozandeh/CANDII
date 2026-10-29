"""t128 phase 3b readings (tools/t128p3/readings3b/analyze.py): the SE bar, size bar, labels, CI,
replication and both controls on fabricated per-seed data, against numbers worked by hand
(plan/T126_ENCDEC_D2.md § Phase 3b, "Reading" block). The last test re-reads seeds 0-2 from the
phase-3 mirror (read-only; skipped when absent) and checks they equal the phase-3 m1 table."""
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "t128p3b_readings"
P3_MIRROR = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p3")

_spec = importlib.util.spec_from_file_location(
    "t128p3b_analyze", ROOT / "tools" / "t128p3" / "readings3b" / "analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

CASES = json.loads((FIX / "cases.json").read_text())


def case(name):
    c = CASES[name]
    return ({i: v for i, v in enumerate(c["f"]) if v is not None},
            {i: v for i, v in enumerate(c["ref"]) if v is not None})


def test_gain_by_hand():
    # f mean 0.51, ref mean 0.55; both sd^2 = 4e-4 / 5 = 8e-5; R6 = 0.02
    r = A.reading(*case("gain"))
    assert r["gap"] == pytest.approx(0.04)
    se = math.sqrt(8e-5 / 6 + 8e-5 / 6)                     # 0.0051640
    assert r["se"] == pytest.approx(se)
    assert r["real_bar"] == pytest.approx(0.0206559, abs=1e-7)
    assert r["size_bar"] == pytest.approx(2 * 0.669 * 0.02)  # 0.02676 > 0.0154, > 0.015
    assert r["df"] == pytest.approx(10.0)                    # equal variances, 6 + 6 seeds
    half = 2.2281389 * se                                    # t(0.975, 10)
    assert r["ci_lo"] == pytest.approx(0.04 - half, abs=1e-7)
    assert r["ci_hi"] == pytest.approx(0.04 + half, abs=1e-7)
    assert r["label"] == "gain"
    # seeds 3-5: f 0.50, 0.52, 0.51 / ref 0.54, 0.56, 0.55 -> sd 0.01 each
    assert r["gap3"] == pytest.approx(0.04)
    assert r["rep_bar"] == pytest.approx(2 * math.sqrt(2 * 1e-4 / 3))   # 0.016330
    assert r["replication"] == "PASS"
    assert r["drift_delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["drift_bar"] == pytest.approx(4 * math.sqrt(2 * 1e-4 / 3))
    assert r["control_a"] == "PASS" and r["flag"] == ""
    assert "new_gap" not in r


def test_loss_is_the_mirror():
    f, ref = case("gain")
    r = A.reading(ref, f)
    assert r["gap"] == pytest.approx(-0.04)
    assert r["label"] == "loss"
    assert r["replication"] == "FAIL"


def test_gain_real_below_the_size_bar():
    # gap 0.015; sd^2 = 4e-6 / 5 = 8e-7 each -> SE 5.164e-4, real bar 0.0020656;
    # R6 0.002 -> size bar max(2 max(0.0077, 0.001338), 0.015) = 0.0154 > 0.015
    r = A.reading(*case("gain_below_size"))
    assert r["gap"] == pytest.approx(0.015)
    assert r["real_bar"] == pytest.approx(4 * math.sqrt(2 * 8e-7 / 6))
    assert r["size_bar"] == pytest.approx(0.0154)
    assert r["label"] == "gain real, below the size bar"
    f, ref = case("gain_below_size")
    assert A.reading(ref, f)["label"] == "loss real, below the size bar"


def test_size_bar_minimum_and_floor_overrides():
    f = {s: 0.5 for s in range(6)}
    r = {s: 0.6 for s in range(6)}
    f[0], r[0] = 0.5001, 0.6001                  # tiny ranges: floor 0.0077 -> 0.0154
    assert A.compare(f, r)["size_bar"] == pytest.approx(0.0154)
    assert A.compare(f, r, floor=0.0, minimum=0.0)["size_bar"] == pytest.approx(2 * 0.669 * 1e-4)


def test_within_seed_noise():
    # gap 0.005; sd_f^2 = 0.0028/5, sd_ref^2 = 0.00175/5 -> SE 0.012315, real bar 0.04926
    r = A.reading(*case("within"))
    assert r["gap"] == pytest.approx(0.005)
    assert r["se"] == pytest.approx(math.sqrt(0.00056 / 6 + 0.00035 / 6))
    assert r["label"] == "within seed noise"
    assert r["ci_lo"] < 0 < r["ci_hi"]
    assert r["gap3"] == pytest.approx(0.5366667 - 0.5333333, abs=1e-6)
    assert r["replication"] == "FAIL"


def test_kit_drift_fail_reads_new_seeds_only():
    # ref seeds 0-2 mean 0.55, seeds 3-5 mean 0.61: delta 0.06 > 4 SE3 = 4 * 0.0081650 = 0.03266
    r = A.reading(*case("drift"))
    assert r["drift_delta"] == pytest.approx(0.06)
    assert r["drift_bar"] == pytest.approx(4 * math.sqrt(2 * 1e-4 / 3))
    assert r["control_a"] == "FAIL"
    assert r["flag"].startswith("kit drift")
    assert r["gap"] == pytest.approx(0.07)                  # six seeds: 0.58 - 0.51
    # seeds 3-5: f 0.50, 0.51, 0.52 / ref 0.60, 0.61, 0.62 -> gap 0.10, SE 0.0081650,
    # size bar max(2 max(0.0077, 0.02, 0.02), 0.015) = 0.04 (range factor 1 at 3 seeds)
    assert r["new_gap"] == pytest.approx(0.10)
    assert r["new_se"] == pytest.approx(math.sqrt(2 * 1e-4 / 3))
    assert r["new_size_bar"] == pytest.approx(0.04)
    assert r["new_df"] == pytest.approx(4.0)
    assert r["new_label"] == "gain"


def test_incomplete_seeds():
    r = A.reading(*case("incomplete"))
    assert r["label"] == "incomplete (n_seeds=5)"
    assert r["replication"] == "n/a (n_seeds=2)"


def _recorded(name):
    return pd.read_csv(FIX / name, sep="\t")


def test_readout_control_pass_fail_and_not_run(tmp_path):
    ok = A.readout(FIX / "rescore", _recorded("recorded_viabase.tsv"))
    assert ok["control_b"] == "PASS"
    assert ok["run"] == "D_C07M20_counts_real_s0"
    assert (ok["n_pairs"], ok["n_unmatched"]) == (3, 0)        # law + eval score only
    assert ok["max_abs_delta"] == pytest.approx(5e-7, abs=1e-12)
    off = A.readout(FIX / "rescore", _recorded("recorded_viabase_off.tsv"))
    assert off["control_b"] == "FAIL"
    assert off["max_abs_delta"] == pytest.approx(2e-6, abs=1e-12)
    assert A.readout(tmp_path / "absent", _recorded("recorded_viabase.tsv"))["control_b"] == "not run"


@pytest.mark.skipif(not (P3_MIRROR / "m1_never_table.tsv").is_file(), reason="no phase-3 mirror")
def test_seeds_0_2_from_the_phase3_mirror(tmp_path):
    for f in ("records_fir.tsv.gz", "viabase_fir.tsv.gz", "runs_fir.tsv", "m1_never_table.tsv"):
        shutil.copy(P3_MIRROR / f, tmp_path / f"p3_{f}")
    for f in ("records_fir.tsv.gz", "viabase_fir.tsv.gz", "runs_fir.tsv"):
        shutil.copy(P3_MIRROR / f"ref_{f}", tmp_path / f"ref_{f}")
    A.main(tmp_path)
    ck = (tmp_path / "checks.txt").read_text()
    assert "PASS seeds 0-2 per-seed metric 1" in ck
    assert "INFO control (b) readout: not run" in ck
    h = pd.read_csv(tmp_path / "headline_table.tsv", sep="\t")
    p3 = pd.read_csv(P3_MIRROR / "m1_never_table.tsv", sep="\t")
    sel = (p3.g_version == "per_track") & (p3.knob == "all") & (p3.subgroup == "never")
    row = h[h.cell == "all"].iloc[0]
    assert row.mean_f == pytest.approx(float(p3[sel & (p3.arm == "Dbigfviabase")].crps_all.iloc[0]))
    assert row.mean_ref == pytest.approx(float(p3[sel & (p3.arm == "Dviabase")].crps_all.iloc[0]))
    assert row.label == "incomplete (n_seeds=3)"
    for f in ("m1_direct_table.tsv", "m3_trained_table.tsv", "m4_heldout_table.tsv",
              "m5_depthlaw_table.tsv", "m9_steps_table.tsv", "per_seed_table.tsv"):
        assert (tmp_path / f).is_file()
