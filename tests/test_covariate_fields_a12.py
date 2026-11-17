"""Close-out item A12: the arm-config generator (covariate_field_arms.py) and the ranked list and
exclusion curve (covariate_ranking.py), on synthetic scores with a known answer."""
from __future__ import annotations

import json
import math
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import covariate_field_arms as G  # noqa: E402
from minicandi.contract import META_COLUMNS, QC_COLUMNS  # noqa: E402
from onion_eval import covariate_ranking as CR  # noqa: E402

BASES = [REPO / f"tools/minicandi/configs/rung{r}/base_heldout.json" for r in (2, 3, 4)]


# ------------------------------------------------------------------------------------------ generator

def test_lofo_from_any_base(tmp_path):
    rows = G.write_arms("lofo", BASES, tmp_path)
    assert len(rows) == 3 * len(META_COLUMNS)
    for r in (2, 3, 4):
        base = json.loads(BASES[r - 2].read_text())
        files = sorted((tmp_path / f"rung{r}").glob("*.json"))
        assert [f.stem for f in files] == sorted(f"lofo_{c}" for c in META_COLUMNS)
        for f in files:
            cfg = json.loads(f.read_text())
            assert {k: v for k, v in cfg.items() if k != "covariate_drop_fields"} == base
            assert cfg["covariate_drop_fields"] == [f.stem[len("lofo_"):]]
    tsv = (tmp_path / "arms.tsv").read_text().splitlines()
    assert tsv[0].split("\t")[:4] == ["arm", "rung", "kind", "fields"] and len(tsv) == 1 + len(rows)
    # idempotent; another content at the same path is refused
    G.write_arms("lofo", BASES, tmp_path)
    (tmp_path / "rung2" / "lofo_lab_id.json").write_text("{}\n")
    with pytest.raises(FileExistsError):
        G.write_arms("lofo", BASES, tmp_path)


def test_qc_skip_groups_curve_and_refusals(tmp_path):
    base = {**json.loads(BASES[0].read_text()), "qc_on_input": True}
    assert len(G.arms_for("lofo", base)) == len(META_COLUMNS) + len(QC_COLUMNS)
    assert len(G.arms_for("lofo", base, skip=["frip"])) == len(META_COLUMNS) + len(QC_COLUMNS) - 1
    g = dict(G.arms_for("groups", base))
    assert g["group_pipeline"] == ["pipeline_id", "pipeline_version_id"]
    assert g["group_lab_platform"] == ["lab_id", "platform_id"]
    assert g["group_qc"] == list(QC_COLUMNS)
    assert "group_qc" not in dict(G.arms_for("groups", json.loads(BASES[0].read_text())))
    c = G.arms_for("curve", base, order=["dedup_on", "pipeline_id", "lab_id"])
    assert c == [("curve_k02", ["dedup_on", "pipeline_id"]),
                 ("curve_k03", ["dedup_on", "pipeline_id", "lab_id"])]
    for kw in ({"order": None}, {"order": ["lab_id", "lab_id"]}, {"order": ["lab"]}):
        with pytest.raises(ValueError):
            G.arms_for("curve", base, **kw)
    ts = {**json.loads(BASES[0].read_text()), "two_step": True}       # two-step bases accepted
    assert len(G.arms_for("lofo", ts)) == len(META_COLUMNS)
    with pytest.raises(ValueError):
        G.arms_for("lofo", {**base, "covariate_drop_fields": ["lab_id"]})
    with pytest.raises(ValueError):
        G.arms_for("lofo", base, skip=["not_a_field"])


def test_generated_config_builds_and_drops(tmp_path):
    import torch
    from minicandi import registry as R
    G.write_arms("lofo", BASES[2:], tmp_path)
    cfg = {**json.loads((tmp_path / "rung4" / "lofo_donor_id.json").read_text()), "context_bins": 64}
    torch.manual_seed(0)
    m = R.build_model(cfg)
    assert m.cov_drop == {"drop": ((10,), (10,))}


def test_cli(tmp_path, capsys):
    assert G.main(["groups", "--base", str(BASES[1]), "--out", str(tmp_path)]) == 0
    assert "arm configs (groups)" in capsys.readouterr().out
    assert (tmp_path / "rung3" / "group_all_but_depth.json").is_file()


# ------------------------------------------------------------------------------------------ ranking

FULL = {"convae": 0.50, "convattn": 0.50, "unet": 0.50}
#: condition -> model -> shift of every rule term (the full set's seed range is 0.02)
SHIFTS = {
    "lofo_lab_id": {"convae": -0.05, "convattn": -0.05, "unet": 0.0},     # useful (2 models)
    "lofo_antibody_id": {"convae": 0.0, "convattn": 0.0, "unet": -0.06},  # useful (2x, 1 model)
    "lofo_donor_id": {"convae": -0.03, "convattn": 0.0, "unet": 0.0},     # unclear (1 model, <2x)
    "lofo_mapq_threshold": {"convae": 0.0, "convattn": -0.002, "unet": 0.0},  # safe alone, S 0.1
    "lofo_dedup_on": {"convae": 0.001, "convattn": -0.005, "unet": 0.0},  # safe alone, S 0.25
    "lofo_pipeline_id": {"convae": 0.0, "convattn": 0.0, "unet": -0.01},  # safe alone, S 0.5
    "group_pipeline": {"convae": 0.0, "convattn": -0.004, "unet": 0.0},   # group safe
    "group_view_knobs": {"convae": -0.05, "convattn": -0.05, "unet": 0.0},  # group useful
    "curve_k02": {"convae": -0.004, "convattn": -0.01, "unet": -0.012},   # within
    "curve_k03": {"convae": -0.04, "convattn": -0.03, "unet": -0.05},     # beyond
}


def _rows(shifts=SHIFTS, undefined=None, seeds=(-0.01, 0.0, 0.01), models=CR.MODELS):
    rows = []
    for cond, sh in [("full", {m: 0.0 for m in FULL})] + list(shifts.items()):
        for m in models:
            for seed, e in enumerate(seeds):                                 # range 0.02
                r = {"condition": cond, "model": m, "seed": seed}
                for t in CR.REPORTED:
                    r[t] = FULL[m] + sh[m] + e
                if undefined and (cond, m) == undefined:
                    r["correction_only"] = None
                rows.append(r)
    return rows


def _cls(rows):
    return {r["condition"]: r["class"] for r in CR.classify(CR.effects(CR.summarize(rows)))}


def test_effects_classes_rank_and_groups():
    ranked = CR.classify(CR.effects(CR.summarize(_rows())))
    cls = {r["condition"]: r["class"] for r in ranked}
    assert cls["lofo_lab_id"] == "useful" and cls["lofo_antibody_id"] == "useful"
    assert cls["lofo_donor_id"] == "unclear"
    assert cls["lofo_pipeline_id"] == "safe to remove"                  # its group is safe
    assert cls["group_view_knobs"] == "useful"
    assert cls["lofo_dedup_on"] == "shared: keep"                         # largest S of the group
    assert cls["lofo_mapq_threshold"] == "shared: removable if dedup_on kept"
    S = {r["condition"]: r["S"] for r in ranked}
    assert S["lofo_antibody_id"] == pytest.approx(3.0) and S["lofo_lab_id"] == pytest.approx(2.5)
    assert S["lofo_dedup_on"] == pytest.approx(0.25)
    order = [r["condition"] for r in ranked]
    assert order[0] == "lofo_antibody_id" and set(order[1:3]) == {"group_view_knobs", "lofo_lab_id"}
    # the useful group's carrier goes last
    assert CR.curve_order(ranked) == ["mapq_threshold", "pipeline_id", "donor_id", "lab_id",
                                      "antibody_id", "dedup_on"]


def test_group_unclear_or_not_scored():
    sh = {k: v for k, v in SHIFTS.items() if k != "group_pipeline"}
    assert _cls(_rows(sh))["lofo_pipeline_id"] == "safe alone (group not scored)"
    sh["group_pipeline"] = {"convae": -0.03, "convattn": 0.0, "unet": 0.0}  # unclear group
    assert _cls(_rows(sh))["lofo_pipeline_id"] == "unclear (group pipeline)"


def test_undefined_cells():
    ranked = CR.classify(CR.effects(CR.summarize(_rows(undefined=("lofo_lab_id", "unet")))))
    assert ranked[-1]["condition"] == "lofo_lab_id" and ranked[-1]["class"] == "undefined"
    one = CR.effects(CR.summarize(_rows(seeds=(0.0,))))                  # 1 seed: undefined
    assert all(e["z"] is None and e["lower_beyond"] is None for e in one)
    flat = CR.effects(CR.summarize(_rows(seeds=(0.0, 0.0))))             # zero range: undefined
    assert all(e["z"] is None for e in flat)
    assert not any(e["z"] == math.inf for e in flat)


def test_single_model_rule():
    cls = _cls(_rows(models=("unet",)))
    assert cls["lofo_antibody_id"] == "useful" and cls["lofo_pipeline_id"] == "safe to remove"
    cls = _cls(_rows(models=("convae",)))
    assert cls["lofo_donor_id"] == "useful"                               # beyond on the only model


def test_report_curve_minimum_set_and_figure(tmp_path):
    res = CR.report(_rows(), tmp_path)
    assert res["order"][:3] == ["mapq_threshold", "pipeline_id", "donor_id"]
    # k = 1 (lofo_mapq_threshold) within, k = 2 (curve_k02) within, k = 3 (curve_k03) beyond;
    # the kept list is every META field minus the two removed (undefined fields stay)
    ms = res["minimum_set"]
    assert ms["k"] == 2 and ms["removed"] == ["mapq_threshold", "pipeline_id"]
    assert ms["kept"] == [c for c in META_COLUMNS if c not in ("mapq_threshold", "pipeline_id")]
    assert CR.report(_rows(), tmp_path / "qc", qc=True)["minimum_set"]["kept"][-len(QC_COLUMNS):] \
        == list(QC_COLUMNS)
    cur = (tmp_path / "curve.tsv").read_text().splitlines()
    assert cur[0].startswith("k\tremoved") and {ln.split("\t")[0] for ln in cur[1:]} == {"0", "1", "2", "3"}
    svg = xml.dom.minidom.parse(str(tmp_path / "curve.svg"))
    assert len(svg.getElementsByTagName("polyline")) == len(CR.REPORTED) * len(CR.MODELS)
    md = (tmp_path / "RANKING.md").read_text()
    assert "Minimum set: log2_depth, run_type_pe" in md
    assert (tmp_path / "order.txt").read_text().strip().startswith("mapq_threshold,pipeline_id")
    ranked = (tmp_path / "ranked.tsv").read_text().splitlines()
    assert ranked[1].split("\t")[1] == "lofo_antibody_id" and ranked[0].endswith("group_note")


def test_tsv_cli(tmp_path, capsys):
    rows = _rows()
    p = tmp_path / "seeds.tsv"
    cols = ["condition", "model", "seed", *CR.REPORTED]
    p.write_text("\t".join(cols) + "\n" + "".join(
        "\t".join(str(r[c]) for c in cols) + "\n" for r in rows))
    assert CR.main(["tsv", "--scores", str(p), "--out", str(tmp_path / "o")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["minimum_set"]["k"] == 2 and out["curve_svg"] is True


def test_screen_condition_names():
    # a screening sub-directory p256_foreign_<mode> is the condition <mode>
    assert "p256_foreign_out_keep_depth.lab_id".split("foreign_", 1)[-1] == "out_keep_depth.lab_id"
