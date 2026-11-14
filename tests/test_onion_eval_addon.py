"""OXe: the add-on eval set (`evalset.py addon`) and add-on scoring (`score.py --addon`,
`baseline_retrieval.py run --addon`) on synthetic tables with evaluation-only second experiments.

Pins (plan, Amendment 4, chunk OXe):
  - the frozen-form evalset.json and scores.json of the synthetic fixture are byte-identical (after the
    tmp root path and the wall-clock keys are normalized) to digests taken on the base commit 5bfad89;
  - on add-on tables the per-target values of every shared target equal its frozen values exactly;
  - per_rung["other_experiment"], pass_rate and the experiment row of explained_fraction_per_layer
    are finite;
  - a second experiment is never an input, under every build option;
  - an add-on eval set built from tables with 3 training-track pairs and 1 validation-track pair
    lists 1 pair id;
  - the retrieval baseline runs with --addon.
The add-on score tests need OXd's `minicandi/dataset.py` (in_eic = evalonly readable in mode val)."""
from __future__ import annotations

import csv
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
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

# 3 validation tracks of H3K4me3 and of H3K27ac (both imputation-scorable), one blind track
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"))
BLIND = ((8, "H3K4me3"),)
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=20, guardrail_windows=4, log=lambda *a: None)
#: sha256 of the normalized frozen-form outputs, taken on the base commit 5bfad89 (before OXe). The
#: scores digest depends on the float library of the machine, so it was taken on both machines that
#: run this file: the laptop (candii env) and Nibi (the kit venv, check job 26979901); the eval set
#: digest is the same on both.
FROZEN_EVALSET_SHA256 = "1ae55d858a8aeceb5a433796015bd387cb801e3edb8e0ec202459521e0f86175"
FROZEN_SCORES_SHA256 = {"laptop": "a71d6341e4cd46f10dc2a0b84d513c52f661875dd78401f41545757574f69a48",
                        "nibi": "0cf176f9826007d91ea629c6bd64be62ccc7a29d84bba869e76df569b36efe95"}
#: wall-clock keys of scores.json, left out of the digest
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads", "evalset_sha256")
USE_OTHER = "other_experiment: evaluation use only, never training"


class Trivial(MiniCandi):
    """Reads no count input: mu = a per-track constant rate times the target depth from C'."""

    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.full((36,), 3.0), requires_grad=False)

    def forward(self, b):
        L = b["y_counts"].shape[1]
        mu = (2.0 ** b["c_out"][..., 0] / 2 ** 24 * self.rate).clamp_min(1e-3)
        mu = mu[:, None, :].expand(-1, L, -1).contiguous()
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


def _tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _normalize(text: str, root: Path) -> str:
    """The tmp root path and the tables sha256 (it hashes the absolute counts paths) masked."""
    tsha = tables_sha256(Path(root) / "tables" / "synth")
    text = text.replace(tsha, "<TABLES_SHA256>")
    for p in sorted({str(root), str(Path(root).resolve())}, key=len, reverse=True):
        text = text.replace(p, "<ROOT>")
    return text


def evalset_digest(path, root) -> str:
    return hashlib.sha256(_normalize(Path(path).read_text(encoding="utf-8"), root).encode()).hexdigest()


def scores_digest(path, root) -> str:
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in VOLATILE:
        s.pop(k, None)
    s.pop("scorer_version", None)   # SF0 (Amendment 8): the one key added after the pinned base
    text = json.dumps(s, indent=1, sort_keys=True, allow_nan=False)
    return hashlib.sha256(_normalize(text, root).encode()).hexdigest()


def make_frozen(root: Path) -> dict:
    """Synthetic tables with isogenic pairs on the validation tracks (base view vs its mapq0 view),
    the frozen eval set built from them, and a scored copy trimmed for speed (every
    leave-one-assay-out target, no accessibility-only item, two never-trained correction pairs)."""
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    pairs = _tsv(t / "pairs.tsv")
    meas = _tsv(t / "measurements.tsv")
    extra = []
    for r in meas:
        if r["in_eic"] == "val" and r["arm_knob"] == "":
            extra.append(dict(pairs[0], pair_id=f"iso{len(extra)}", a=r["measurement_id"],
                              b=r["measurement_id"].replace(".base", ".mapq0"), rung="isogenic",
                              assay=r["assay"], cell=r["cell"], subset_pair="0", split="val",
                              never_trained="0", knobs_differing="mapq"))
    _write(t / "pairs.tsv", PAIR_COLUMNS, pairs + extra)
    full = root / "eval" / "synth" / "evalset_full.json"
    E.build(t, full, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    ev = json.loads(full.read_text(encoding="utf-8"))
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"] = [p for p in ev["correction_pairs"] if p["never_trained"]
                              and not p["subset_pair"] and p["cell"] in ("s0", "s3")][:2]
    ev_path = root / "eval" / "synth" / "evalset.json"
    ev_path.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return {"root": root, "tables": t, "evalset_full": full, "evalset": ev_path, "ev": ev}


def score_frozen(fz: dict, out: Path) -> dict:
    from onion_eval import score as S
    return S.score(fz["evalset"], out, model=Trivial(), config=dict(CFG),
                   run_tables_sha256=tables_sha256(fz["tables"]), **KW)


def _second_experiment(root: Path, row: dict, k: int, rng, *, role="evalonly") -> dict:
    """A second experiment of the track of `row` at the same depth: its counts with a seeded share
    (30%) of each bin moved to the next bin, so the totals match the EIC track's (the 5% rule of
    `metrics.variance_ratio` always holds) and the pair value is not 0; its own halves, another lab
    and experiment accession."""
    src = Path(row["counts_path"][3:])
    mid = f"ENCSRX{k:05d}.pooled.uniform.base"
    d = root / "measurements" / mid
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    with h5py.File(src, "r") as fi:
        for c in fi:
            x = np.asarray(fi[c][:], np.int64)
            mv = rng.binomial(x, 0.3)
            counts[c] = x - mv + np.roll(mv, 1)
    pr1 = {c: rng.binomial(v, 0.5) for c, v in counts.items()}
    dep = float(row["depth_reads"])
    for nm, arr, dd in (("counts", counts, dep), ("pr1", pr1, dep / 2),
                        ("pr2", {c: counts[c] - pr1[c] for c in counts}, dep / 2)):
        with h5py.File(d / f"{nm}.h5", "w") as f:
            for c, v in arr.items():
                f.create_dataset(c, data=v.astype(np.uint32))
            f.attrs.update({"schema": 1, "measurement_id": mid, "depth_reads": dd,
                            "depth_fragments": dd, "resolution": 25})
    new = dict(row, measurement_id=mid, experiment=mid.split(".")[0], in_eic=role,
               lab="labZ", depth_reads=f"{dep:.0f}", depth_fragments=f"{dep / 2:.0f}",
               counts_path=f"h5:{(d / 'counts.h5').resolve()}", control_measurement_id="")
    return new


def make_addon(fz: dict, name: str, tracks, *, blind_role="blind") -> Path:
    """Add-on tables of the frozen tables: every base file byte for byte, then one second
    experiment per EIC track of `tracks` ((cell, assay) of base views), its other_experiment pair
    (split val, or blind for a blind track), evaluation_only.tsv and build_info.json."""
    root, base = fz["root"], fz["tables"]
    out = base.parent / name
    shutil.copytree(base, out)
    meas = _tsv(base / "measurements.tsv")
    by = {(r["cell"], r["assay"]): r for r in meas if r["arm_knob"] == ""}
    rng = np.random.default_rng(7)
    rows, pairs, eo = [], [], []
    for k, key in enumerate(tracks):
        r = by[key]
        role = blind_role if r["in_eic"] == "blind" else "evalonly"
        new = _second_experiment(root, r, len(list((root / "measurements").glob("ENCSRX*"))) + k,
                                 rng, role=role)
        rows.append(new)
        eo.append({"measurement_id": new["measurement_id"], "use": USE_OTHER})
        pairs.append({"pair_id": f"ox{k:03d}", "a": r["measurement_id"], "b": new["measurement_id"],
                      "rung": "other_experiment", "assay": r["assay"], "cell": r["cell"],
                      "knobs_differing": "experiment;lab", "subset_pair": "0",
                      "split": "blind" if r["in_eic"] == "blind" else "val", "never_trained": "0"})
    with open(out / "measurements.tsv", "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=list(MEASUREMENT_COLUMNS), delimiter="\t",
                       lineterminator="\n").writerows(rows)
    with open(out / "pairs.tsv", "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=list(PAIR_COLUMNS), delimiter="\t",
                       lineterminator="\n").writerows(pairs)
    _write(out / "evaluation_only.tsv", ("measurement_id", "use"), eo)
    (out / "build_info.json").write_text(json.dumps({
        "addon_of": str(base), "addon_of_tables_sha256": tables_sha256(base)}), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("oxe")
    fz = make_frozen(root)
    fz["scores_dir"] = root / "runs" / "arm" / "seed0"
    fz["scores"] = score_frozen(fz, fz["scores_dir"])
    return fz


@pytest.fixture(scope="module")
def addon(frozen):
    from onion_eval import score as S
    val = [(f"s{i}", a) for i, a in VAL]
    tx = make_addon(frozen, "synth_x", val + [("s6", "H3K4me3")])
    evx = frozen["root"] / "eval" / "synth_x" / "evalset.json"
    E.addon(frozen["evalset"], tx, evx)
    out = frozen["scores_dir"] / "ox"
    s = S.score(evx, out, model=Trivial(), config=dict(CFG),
                run_tables_sha256=tables_sha256(frozen["tables"]), addon=True, **KW)
    return {"tables": tx, "evalset": evx, "ev": json.loads(evx.read_text()), "scores": s,
            "out": out}


# ------------------------------------------------------------------------------------- frozen form

def test_frozen_outputs_byte_identical_to_base_commit(frozen):
    assert evalset_digest(frozen["evalset_full"], frozen["root"]) == FROZEN_EVALSET_SHA256
    assert scores_digest(frozen["scores_dir"] / "scores.json", frozen["root"]) in FROZEN_SCORES_SHA256.values()


# ------------------------------------------------------------------------------------- eval set

def _evalonly(tables):
    return {r["measurement_id"] for r in _tsv(Path(tables) / "measurements.tsv")
            if r["in_eic"] == "evalonly"}


def test_build_never_lists_a_second_experiment(frozen, tmp_path):
    tx = make_addon(frozen, "synth_b", [("s0", "H3K4me3"), ("s6", "H3K4me3"), ("s8", "H3K4me3")])
    eo = _evalonly(tx) | {r["measurement_id"] for r in _tsv(tx / "evaluation_only.tsv")}
    assert len(eo) == 3
    out_dir = frozen["root"] / "eval" / "synth_b"
    for kw, name in (({}, "evalset.json"),
                     ({"include_blind": True}, "evalset_blind.json"),
                     ({"include_blind": True, "inputs_roles": ("train", "val")}, "evalset_blind_tv.json")):
        ev = E.build(tx, out_dir / name, n_regions=2, n_fit_regions=1, region_bins=256, **kw)
        listed = {t["target"] for t in ev["imputation_targets"] + ev["control_targets"]}
        listed |= {m for t in ev["imputation_targets"] + ev["control_targets"] for m in t["inputs"]}
        listed |= {m for p in ev["correction_pairs"] for m in (p["source"], p["target"])}
        assert not listed & eo, (name, listed & eo)
        # the blind EIC track itself stays a target of the blind eval set
        if kw.get("include_blind"):
            assert any(t["split"] == "blind" for t in ev["imputation_targets"])
        # other_experiment pairs enter replicate_pairs under the usual split rule
        ox = {p["pair_id"] for p in _tsv(tx / "pairs.tsv") if p["rung"] == "other_experiment"}
        assert ox <= set(ev["replicate_pairs"])


def test_addon_evalset_keys_and_one_pair_id(frozen):
    # 3 training-track second experiments and 1 validation-track one -> one pair id
    tx = make_addon(frozen, "synth_c", [("s6", "H3K4me3"), ("s7", "H3K4me3"), ("s6", "H3K27ac"),
                                        ("s3", "H3K27ac")])
    out = frozen["root"] / "eval" / "synth_c" / "evalset.json"
    ev = E.addon(frozen["evalset"], tx, out)
    base = frozen["ev"]
    assert ev["regions"] == base["regions"] and ev["fit_regions"] == base["fit_regions"]
    assert ev["strata"] == base["strata"] and ev["scorable"] == base["scorable"]
    assert len(ev["replicate_pairs"]) == 1
    second = sorted(_evalonly(tx))
    flagged = [t for t in ev["imputation_targets"] if t.get("evaluation_only")]
    assert len(flagged) == 1 and flagged[0]["target"] in second
    eic = [t for t in ev["imputation_targets"] if not t.get("evaluation_only")]
    assert len(eic) == 1 and (eic[0]["cell"], eic[0]["assay"]) == ("s3", "H3K27ac")
    assert flagged[0]["inputs"] == eic[0]["inputs"] and flagged[0]["split"] == eic[0]["split"]
    assert flagged[0]["regime"] == "leave_one_assay_out"
    assert ev["correction_pairs"] == [] and ev["control_targets"] == []
    assert ev["addon_of"] == hashlib.sha256(frozen["evalset"].read_bytes()).hexdigest()
    assert ev["tables_sha256"] == tables_sha256(tx)
    assert E.check(out) == []
    # no input of any item is a second experiment
    assert not {m for t in ev["imputation_targets"] for m in t["inputs"]} & set(second)


def test_addon_refuses_tables_of_another_base(frozen, tmp_path):
    tx = make_addon(frozen, "synth_d", [("s1", "H3K4me3")])
    info = json.loads((tx / "build_info.json").read_text())
    info["addon_of_tables_sha256"] = "0" * 64
    (tx / "build_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="addon_of_tables_sha256"):
        E.addon(frozen["evalset"], tx, frozen["root"] / "eval" / "synth_d" / "evalset.json")


def test_check_flags_a_second_experiment_as_input(addon, tmp_path):
    ev = json.loads(addon["evalset"].read_text())
    second = sorted(_evalonly(addon["tables"]))[0]
    ev["imputation_targets"][0]["inputs"] = ev["imputation_targets"][0]["inputs"] + [second]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(ev))
    probs = E.check(bad, tables=addon["tables"])
    assert any(second in p for p in probs)


# ------------------------------------------------------------------------------------- add-on scores

def test_shared_targets_equal_their_frozen_values(frozen, addon):
    s0, s1 = frozen["scores"], addon["scores"]
    h0 = s0["specific_variance_recovered_above_noise"]["counts_log1p"]["per_target"]
    h1 = s1["specific_variance_recovered_above_noise"]["counts_log1p"]["per_target"]
    shared = [t["target"] for t in addon["ev"]["imputation_targets"] if not t.get("evaluation_only")]
    assert len(shared) == 6 and set(shared) <= set(h0)
    for tg in shared:
        assert h1[tg] == h0[tg], tg
    v0 = s0["predicted_over_replicate_variance"]["per_target"]
    v1 = s1["predicted_over_replicate_variance"]["per_target"]
    for tg in shared:
        assert v1[tg]["per_rung"]["pseudoreplicate"] == v0[tg]["per_rung"]["pseudoreplicate"]
    f0 = np.load(frozen["scores_dir"] / "suffstats.npz")
    f1 = np.load(addon["out"] / "suffstats.npz")
    for tg in shared:
        k = f"crps/leave_one_assay_out/{tg}"
        assert np.array_equal(f0[k], f1[k], equal_nan=True)


def test_two_bounds_and_experiment_layer_are_finite(addon):
    s = addon["scores"]
    h7 = s["predicted_over_replicate_variance"]
    assert np.isfinite(h7["per_rung"]["other_experiment"]["macro"])
    assert np.isfinite(h7["pass_rate"]["macro"])
    assert h7["n_undefined_upper_bound"] == 0
    for a in ("H3K4me3", "H3K27ac"):
        rows = s["explained_fraction_per_layer"]["per_assay"][a]["rows"]
        exp = [r for r in rows if r["rung"] == "other_experiment"]
        assert exp and np.isfinite(exp[0]["explained_fraction"]), rows


def test_flagged_targets_only_in_per_target(addon):
    s = addon["scores"]
    flagged = {t["target"] for t in addon["ev"]["imputation_targets"] if t.get("evaluation_only")}
    assert len(flagged) == 6
    assert s["n_targets"] == 6 and s["addon"]["n_evaluation_only_targets"] == 6
    h1 = s["specific_variance_recovered_above_noise"]["counts_log1p"]
    assert flagged <= set(h1["per_target"])
    assert all(h1["per_target"][m]["evaluation_only"] is True for m in flagged)
    assert all(v["n_targets"] == 3 for v in h1["per_assay_status"].values())
    h7 = s["predicted_over_replicate_variance"]
    assert flagged <= set(h7["per_target"])
    assert h7["n"] == 6 and h7["per_rung"]["other_experiment"]["n"] == 6
    # the flagged targets' reason texts; every undefined ratio states its reason
    for tg, v in h7["per_target"].items():
        for rung, val in v["per_rung"].items():
            if val is None:
                assert v["per_rung_reason"][rung]
    assert s["copy_leak"]["n_reconstruction_items"] > 0


def test_addon_refusals(frozen, addon, tmp_path):
    from onion_eval import score as S
    sha = tables_sha256(frozen["tables"])
    # an add-on eval set needs --addon, and --addon needs an add-on eval set
    with pytest.raises(S.ScoreRefused, match="--addon"):
        S.score(addon["evalset"], tmp_path / "a", baseline="marginal_nb", config=dict(CFG), **KW)
    with pytest.raises(S.ScoreRefused, match="addon_of"):
        S.score(frozen["evalset"], tmp_path / "b", baseline="marginal_nb", config=dict(CFG),
                addon=True, **KW)
    # the run must be trained on the tables the add-on extends
    with pytest.raises(S.ScoreRefused, match="addon_of_tables_sha256"):
        S.score(addon["evalset"], tmp_path / "c", model=Trivial(), config=dict(CFG),
                run_tables_sha256=tables_sha256(addon["tables"]), addon=True, **KW)
    # never over a scores.json of another eval set (the frozen score of the seed dir)
    with pytest.raises(S.ScoreRefused, match="evalset_sha256"):
        S.score(addon["evalset"], frozen["scores_dir"], model=Trivial(), config=dict(CFG),
                run_tables_sha256=sha, addon=True, **KW)
    assert scores_digest(frozen["scores_dir"] / "scores.json", frozen["root"]) in FROZEN_SCORES_SHA256.values()


def test_retrieval_baseline_runs_with_addon(frozen, addon):
    from onion_eval import baseline_retrieval as BR
    out = frozen["root"] / "eval" / "synth_x" / "baselines" / "retrieval" / "seed0"
    rc = BR.main(["run", "--tables", str(addon["tables"]), "--evalset", str(addon["evalset"]),
                  "--out", str(out), "--addon"])
    assert rc == 0
    s = json.loads((out / "scores.json").read_text())
    assert s["predictor"] == "retrieval" and s["addon"]["n_evaluation_only_targets"] == 6
    assert s["n_targets"] == 6
    second = _evalonly(addon["tables"])
    used = {r["neighbor"] for r in _tsv(out / BR.NEIGHBORS_TSV)}
    assert not used & {m.split(".")[0] for m in second}
    # without --addon the add-on eval set is refused
    assert BR.main(["run", "--tables", str(addon["tables"]), "--evalset", str(addon["evalset"]),
                    "--out", str(out.parent / "seed1")]) == 2
