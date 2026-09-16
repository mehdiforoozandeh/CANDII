"""The two things `plan/BENCHMARK_DESIGN.md` asks of an eDICE retrain: §5 selection, D32 loci.

Run from `competitors/edice/`:

    PYTHONPATH=.:../../src pytest tests/ -q

Unlike `test_edice_model.py` these are not synthetic-tensor checks. §5 is a claim about which
PANEL a run reads and D32 is a claim about which BINS it reads, and neither can be proved against
a tensor — so the end-to-end tests build a real `CANDI_STORE` with the repo's own writer, run
`run_eic.py train` over it on CPU, and read the answers back off disk.

Tests are written against the requirement, not the implementation: the derived eval set must
contain no `B_` target, the best weights must be on disk before the run ends, and under a
`regions` regime every training bin must be inside the BED. How `run_eic.py` arranges that is its
own business.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "src"), str(Path(__file__).resolve().parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_eic                                                              # noqa: E402
from eic_panel import derive_v_only_regime, train_bin_spans                 # noqa: E402

from candi.store.regime import Regime, RegionSet                            # noqa: E402
from tests.test_store_reader import ASSAYS, RES, make_store                 # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: One `V_` pair and one `B_` pair, so the filter has something to drop. Each prompt cell LACKS the
#: assay its truth cell carries, which is the only layout that poses imputation.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq"),
    "T_bb": ("ATAC-seq", "DNase-seq"),
    "V_aa": ("H3K4me3",),
    "B_bb": ("H3K4me3",),
}
PAIRS = [["T_aa", "V_aa"], ["T_bb", "B_bb"]]


def regime_obj(store: Path, **over) -> dict:
    obj = {
        "store": str(store),
        "assays": list(ASSAYS),
        "biosamples": {"train": ["T_aa", "T_bb"], "eval": ["V_aa", "B_bb"]},
        "eval_pairs": [list(p) for p in PAIRS],
        "context_bins": 64,
        "train_chroms": ["chr1"],
        "eval_chroms": ["chr2"],
        "window_plan": {"type": "tile", "stride_bins": 64, "min_valid_frac": 0.9},
        "dsf": {"policy": "discrete", "levels": [1, 2, 4, 8]},
        "kinds": ["counts", "peaks", "pval"],
        "seed": 42,
    }
    obj.update(over)
    return obj


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("edicesel"), tracks=TRACKS)


def write_regime(d: Path, obj: dict) -> Path:
    p = d / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------------------
# §5 — the derived selection panel is V_ only
# ---------------------------------------------------------------------------------------------


def test_the_derived_selection_regime_declares_no_blind_target(store, tmp_path) -> None:
    """§5, and the PI's ruling of 2026-08-31: `B_` is not merely unranked, it is never read."""
    src = write_regime(tmp_path, regime_obj(store))
    got = derive_v_only_regime(json.loads(src.read_text()), src)
    assert got["eval_pairs"] == [["T_aa", "V_aa"]]
    assert [t for _, t in got["eval_pairs"] if not t.startswith("V_")] == []
    assert got["biosamples"]["eval"] == ["V_aa"]


def test_the_derived_selection_regime_still_names_every_training_cell(store, tmp_path) -> None:
    """Only the EVAL side is filtered. Dropping a training cell would change what is fitted."""
    src = write_regime(tmp_path, regime_obj(store))
    got = derive_v_only_regime(json.loads(src.read_text()), src)
    assert got["biosamples"]["train"] == ["T_aa", "T_bb"]
    assert got["assays"] == list(ASSAYS)
    assert got["train_chroms"] == ["chr1"]


def test_a_regime_with_no_validation_pair_cannot_select_a_checkpoint(store, tmp_path) -> None:
    src = write_regime(tmp_path, regime_obj(store, eval_pairs=[["T_bb", "B_bb"]]))
    with pytest.raises(ValueError, match="no `V_` eval pair"):
        derive_v_only_regime(json.loads(src.read_text()), src)


def test_the_derived_regime_survives_the_beds_hash_gate_from_another_directory(store, tmp_path,
                                                                              bed) -> None:
    """`regions.bed` resolves against the regime file's own directory, and the derived copy is
    written beside the run, not beside the source. An unrewritten relative path fails the D32
    sha256 check — which is the failure this rewrite exists to prevent."""
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    got = derive_v_only_regime(json.loads(src.read_text()), src)
    assert Path(got["regions"]["bed"]).is_absolute()
    elsewhere = tmp_path / "run"
    elsewhere.mkdir()
    (elsewhere / "derived.json").write_text(json.dumps(got))
    r = Regime.from_file(elsewhere / "derived.json")
    assert r.regions is not None and r.eval_pairs == (("T_aa", "V_aa"),)


# ---------------------------------------------------------------------------------------------
# D32 — training loci, at eDICE's own unit
# ---------------------------------------------------------------------------------------------

#: Edges deliberately OFF the 25 bp bin grid, so a containment rule and a division disagree.
BED_TEXT = "chr1\t1010\t5240\tR1\nchr1\t20003\t26000\tR2\n"


@pytest.fixture()
def bed(tmp_path) -> Path:
    p = tmp_path / "regions.bed"
    p.write_text(BED_TEXT, encoding="utf-8")
    return p


def bed_decl(bed: Path) -> dict:
    return {"bed": str(bed), "sha256": hashlib.sha256(bed.read_bytes()).hexdigest(),
            "policy": "contain"}


def test_without_a_regions_key_the_whole_chromosome_is_the_training_scope(store, tmp_path) -> None:
    src = write_regime(tmp_path, regime_obj(store))
    assert train_bin_spans(json.loads(src.read_text()), src, "chr1", 2000, RES) == [(0, 2000)]


def test_a_training_bin_that_straddles_a_region_edge_is_dropped(store, tmp_path, bed) -> None:
    """Containment, not overlap. `1010 // 25` is bin 40, whose bases 1000-1025 reach outside the
    region, so the first training bin is 41; `5240 // 25` is bin 209, wholly inside, so the last
    is 209 and the span ends at 210."""
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    spans = train_bin_spans(json.loads(src.read_text()), src, "chr1", 2000, RES)
    assert spans == [(41, 209), (801, 1040)]


def test_every_training_bin_lies_wholly_inside_a_declared_region(store, tmp_path, bed) -> None:
    """The requirement itself, checked in base pairs against the BED rather than against the
    span arithmetic that produced it."""
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    intervals = [(int(f[1]), int(f[2])) for f in
                 (ln.split() for ln in BED_TEXT.splitlines() if ln.strip())]
    for a, b in train_bin_spans(json.loads(src.read_text()), src, "chr1", 2000, RES):
        for b_idx in (a, b - 1):
            lo, hi = b_idx * RES, (b_idx + 1) * RES
            assert any(s <= lo and hi <= e for s, e in intervals), f"bin {b_idx} escapes the BED"


def test_the_scope_is_not_re_anchored_at_each_regions_own_first_bin(store, tmp_path,
                                                                   bed) -> None:
    """D32 keeps the chromosome's bin grid (§3.1's 1,294-vs-1,328 ruling). Every returned bin
    index is therefore a multiple-of-nothing offset from bin 0, and the span for a region starting
    at 20,003 bp starts at bin 801 — `ceil(20003/25)` — not at a bin renumbered from the region."""
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    spans = train_bin_spans(json.loads(src.read_text()), src, "chr1", 2000, RES)
    assert spans[1][0] == -(-20003 // RES) == 801


def test_the_scope_never_runs_past_the_end_of_the_chromosome(store, tmp_path, bed) -> None:
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    spans = train_bin_spans(json.loads(src.read_text()), src, "chr1", 900, RES)
    assert spans[-1][1] == 900


def test_the_pilot_bed_gives_edice_the_regimes_own_declared_training_scope() -> None:
    """`plan/BENCHMARK_DESIGN.md` §3.1: 40 regions, 25,588,197 bp, **1,023,489** contained 25 bp
    bins after the Rule 2 chromosome cut. Pinned here for the same reason CANDI pins 1,294 —
    eDICE's window is one bin, so this is the scope it sees, and CANDI's 993,792 (97.10 %) is the
    smaller number a 768-bin window can fit inside the same regions.
    """
    cfg = json.loads((REPO / "configs/regime.eic_pilot.json").read_text())
    regions = RegionSet.from_obj(cfg["regions"], base=REPO / "configs")
    train = set(cfg["train_chroms"])
    spans = [(c, s) for c in regions.chroms if c in train for s in regions.bin_spans(c, RES)]
    assert len(spans) == 40
    assert sum(b - a for _, (a, b) in spans) == 1_023_489


# ---------------------------------------------------------------------------------------------
# end to end: a real store, a real training run, on CPU
# ---------------------------------------------------------------------------------------------


def run_train(store: Path, out: Path, regime_path: Path, *, epochs: int, extra=()) -> dict:
    args = run_eic.build_parser().parse_args([
        "train", "--regime", str(regime_path), "--out", str(out), "--device", "cpu",
        "--n-targets", "1", "--epochs", str(epochs), "--batch-size", "64",
        "--eval-batch-size", "256", "--slab", "500", *extra,
    ])
    assert run_eic.cmd_train(args) == 0
    return json.loads((out / "train.json").read_text())


@pytest.fixture(scope="module")
def trained(store, tmp_path_factory):
    d = tmp_path_factory.mktemp("edicerun")
    src = write_regime(d, regime_obj(store))
    return run_train(store, d / "run", src, epochs=3, extra=["--eval-every", "1",
                                                             "--early-stop-epochs", "0"]), d / "run"


def test_the_run_selects_on_the_validation_panel_and_never_opens_a_blind_one(trained) -> None:
    rec, out = trained
    sel = rec["selection"]
    assert sel["panel"] == "V_" and sel["scored_by"] == "candi.bench.external.score_external"
    derived = json.loads(Path(sel["regime"]).read_text())
    assert [t for _, t in derived["eval_pairs"] if t.startswith("B_")] == []
    assert sel["n_tracks"] == 1
    assert sorted(p.name for p in (out / "preds_select").iterdir() if p.is_dir()) == \
        ["T_aa__V_aa__H3K4me3"]


def test_the_best_weights_are_on_disk_before_the_run_ends(trained) -> None:
    """The property that survives a walltime kill: `model.best.pt` is written inside the check
    that improved, not in the tidy-up after the last epoch."""
    rec, out = trained
    assert (out / "model.best.pt").exists()
    improved = [c["epoch"] for c in rec["selection"]["curve"] if c["improved"]]
    assert improved and rec["selection"]["best_epoch"] == improved[-1]


def test_the_selected_checkpoint_is_the_best_epochs_weights_and_not_the_last(trained) -> None:
    import torch

    rec, out = trained
    if rec["selection"]["best_epoch"] == rec["config"]["epochs"] - 1:
        pytest.skip("the last epoch won, so best and last coincide")
    best = torch.load(out / "model.best.pt", map_location="cpu")
    sel = torch.load(out / "model.selected.pt", map_location="cpu", weights_only=False)
    last = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    for k, v in best.items():
        assert torch.equal(sel["state_dict"][k], v)
    assert any(not torch.equal(last["state_dict"][k], v) for k, v in best.items())


def test_every_selection_check_scores_the_same_positions(trained) -> None:
    """Selection compares epoch 1 against epoch 3, which is only a paired comparison if both saw
    the same bins. One source, opened once, is what pins that."""
    rec, _out = trained
    assert rec["selection"]["chroms"] == ["chr2"]
    assert {c["n_tracks"] for c in rec["selection"]["curve"]} == {1}


def test_a_run_that_stalls_stops_early_and_keeps_the_checkpoint_it_had(store,
                                                                      tmp_path) -> None:
    """Patience is counted in epochs, not in checks — so at a cadence of 1 a patience of 1 ends
    the run two epochs after the best one, and `model.best.pt` is still that epoch's."""
    src = write_regime(tmp_path, regime_obj(store))
    rec = run_train(store, tmp_path / "stall", src, epochs=8,
                    extra=["--eval-every", "1", "--early-stop-epochs", "1"])
    curve = rec["selection"]["curve"]
    assert len(curve) < 8, "a stalled run must not spend the whole epoch budget"
    assert not curve[-1]["improved"]
    assert (tmp_path / "stall" / "model.best.pt").exists()


def test_a_run_with_no_cadence_selects_nothing_and_says_so(store, tmp_path) -> None:
    src = write_regime(tmp_path, regime_obj(store))
    rec = run_train(store, tmp_path / "nosel", src, epochs=1, extra=["--eval-every", "0"])
    assert rec["selection"] is None
    assert not (tmp_path / "nosel" / "model.best.pt").exists()


def test_a_regions_regime_trains_on_the_contained_bins_and_no_others(store, tmp_path,
                                                                    bed) -> None:
    """The D32 end of the requirement, read back off the run record: the training matrix holds
    exactly as many rows as there are bins wholly inside the BED."""
    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    rec = run_train(store, tmp_path / "pilot", src, epochs=1, extra=["--eval-every", "0"])
    want = sum(b - a for a, b in train_bin_spans(json.loads(src.read_text()), src, "chr1",
                                                 2000, RES))
    assert rec["n_train_bins"] == rec["n_train_scope_bins"] == want
    assert want < 2000, "the BED must actually restrict something"
    assert rec["regions"]["sha256"] == bed_decl(bed)["sha256"]


def test_a_regions_regime_reads_the_same_values_the_store_holds_at_those_bins(store, tmp_path,
                                                                             bed) -> None:
    """`n_train_bins` counts rows; this checks they are the RIGHT rows. The training matrix is
    `arcsinh` of the panel's pval at the contained bins, so re-reading the store at those bins and
    at bins just outside must reproduce the one and not the other."""
    from eic_panel import load_regime, read_slab, training_panel

    src = write_regime(tmp_path, regime_obj(store, regions=bed_decl(bed)))
    regime = load_regime(src)
    corpus = run_eic._open_store(regime)
    panel = training_panel(corpus, regime)
    spans = train_bin_spans(regime, src, "chr1", 2000, RES)
    inside = np.arcsinh(np.concatenate(
        [read_slab(corpus, panel, "chr1", a, b) for a, b in spans], axis=0))
    a0 = spans[0][0]
    outside = np.arcsinh(read_slab(corpus, panel, "chr1", a0 - 1, a0))
    assert inside.shape == (sum(b - a for a, b in spans), panel.n_tracks)
    assert not np.array_equal(inside[:1], outside), "bin 40 straddles the edge and must be out"


def test_the_board_prediction_pass_still_emits_every_declared_track(trained, store,
                                                                   tmp_path) -> None:
    """Selection reads `V_` only; the board pass reads the regime as shipped and must still emit
    the `B_` tracks too. The two roots are written by the same code, so this is the guard against
    a selection-shaped narrowing leaking into the run that gets scored."""
    from candi.bench.external import score_external
    from candi.bench.harness import open_source

    _rec, out = trained
    src = write_regime(tmp_path, regime_obj(store))
    args = run_eic.build_parser().parse_args([
        "predict", "--regime", str(src), "--model", str(out / "model.selected.pt"),
        "--out", str(tmp_path / "preds"), "--device", "cpu", "--slab", "500",
        "--batch-size", "256"])
    assert run_eic.cmd_predict(args) == 0
    assert sorted(p.name for p in (tmp_path / "preds").iterdir() if p.is_dir()) == \
        ["T_aa__V_aa__H3K4me3", "T_bb__B_bb__H3K4me3"]
    source = open_source(store=str(src))
    try:
        scored = score_external(source, tmp_path / "preds", seed=0)
    finally:
        source.close()
    assert scored["provenance"]["missing_tracks"] == []
    assert scored["provenance"]["method"] == run_eic.METHOD


def test_the_selection_metric_is_our_instruments_number_not_edices_own(store, tmp_path) -> None:
    """§5's "same rule" is the point. `edice_torch.metrics` is eDICE's training diagnostic, and a
    run that selected on it would not be comparable with CANDI's row."""
    src = write_regime(tmp_path, regime_obj(store))
    rec = run_train(store, tmp_path / "metric", src, epochs=1,
                    extra=["--eval-every", "1", "--select-metric", "gwspear"])
    assert rec["selection"]["metric"] == "gwspear"
    assert all(c["metric"] == "gwspear" for c in rec["selection"]["curve"])
    assert set(rec["selection"]["curve"][0]) >= {"epoch", "value", "improved", "seconds"}


def test_a_metric_the_instrument_cannot_produce_for_a_point_track_is_refused() -> None:
    """eDICE emits `signal_mu` and no `signal_sigma`, so `score_external` records a point-only
    track and never computes `crps`. Offering it as a choice would select on a nan."""
    assert "crps" not in run_eic.SELECT_METRICS
    with pytest.raises(SystemExit):
        run_eic.build_parser().parse_args(
            ["train", "--regime", "r.json", "--out", "o", "--n-targets", "1",
             "--select-metric", "crps"])
