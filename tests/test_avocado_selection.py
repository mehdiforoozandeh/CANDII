"""t81 — Avocado selects its checkpoint on the `V_` panel (`plan/BENCHMARK_DESIGN.md` §5).

§5 asks every trainable method to select its best checkpoint on `V_`, by the same rule. Avocado
selected nothing: it ran a fixed number of epochs and kept the last state, monitoring a 1-in-50
(position, track) entry mask over its own training columns — a number no other method computes.

Two claims are defended here and they are separate.

**The panel.** Both live regimes declare all 38 pairs in one file, 26 `V_` and 12 `B_`. The PI's
ruling of 2026-08-31 is that `B_` is not read at all, not merely kept out of the argmax, so the loop
scores a DERIVED `V_`-only regime. A `B_` target reaching the selection panel is invisible until
publication, which is why it is checked against the shipped configs and not only against a fixture.

**The write.** The best weights go to disk the moment the metric improves, not when the run ends.
A run killed by walltime must still leave a selected checkpoint behind. That is checked by driving
the loop with a scripted metric curve and looking at the file after every check — the point is
*when* the bytes appear, so a stub metric is the right instrument and the real scorer is exercised
separately, end to end, in `test_the_selection_metric_comes_from_the_scorer_that_ranks_the_board`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_AVO = Path(__file__).resolve().parents[1] / "competitors" / "avocado"
sys.path.insert(0, str(_AVO / "vendor"))
sys.path.insert(0, str(_AVO))

import bin_store                                                            # noqa: E402
import train as avo_train                                                   # noqa: E402
from index import select_pairs, write_select_regime                         # noqa: E402

from tests.test_store_reader import make_store                              # noqa: E402
from tests.test_store_regime import regime_dict                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LIVE_REGIMES = ("configs/regime.eic_19.json", "configs/regime.eic_pilot.json")

TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq"),
    "T_bb": ("ATAC-seq", "DNase-seq"),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "V_bb": ("ATAC-seq", "H3K4me3"),
    "B_aa": ("ATAC-seq", "H3K4me3"),
}


@pytest.fixture(scope="module")
def scope(tmp_path_factory):
    """A store and a regime declaring BOTH panels, as the live regimes do."""
    d = tmp_path_factory.mktemp("avoselect")
    store = make_store(d / "s", tracks=TRACKS)
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"],
                                         "eval": ["V_aa", "V_bb", "B_aa"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"], ["T_aa", "B_aa"]])
    rp = d / "regime.json"
    rp.write_text(json.dumps(obj), encoding="utf-8")
    out = d / "binned"
    assert bin_store.main(["--regime", str(rp), "--out", str(out), "--chrom", "chr2"]) == 0
    return d, rp, out


# ---------------------------------------------------------------------------
# the panel — B_ is never read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", LIVE_REGIMES)
def test_the_derived_selection_panel_carries_no_test_biosample(cfg, tmp_path):
    """The check that matters on the shipped configs: 26 pairs in, every target `V_`, zero `B_`."""
    src = REPO / cfg
    d = write_select_regime(src, tmp_path / "sel.json")
    targets = [t for _, t in d["eval_pairs"]]
    assert len(targets) == 26
    assert all(t.startswith("V_") for t in targets)
    assert not [t for t in targets if t.startswith("B_")]
    assert d["biosamples"]["eval"] == sorted(set(targets))
    # and the file on disk says the same thing, since that is what the loop opens
    assert json.loads((tmp_path / "sel.json").read_text())["eval_pairs"] == d["eval_pairs"]


def test_the_shipped_regimes_really_do_declare_both_panels():
    """Without this the test above could pass on a regime that had no `B_` to drop."""
    for cfg in LIVE_REGIMES:
        obj = json.loads((REPO / cfg).read_text(encoding="utf-8"))
        kept, dropped = select_pairs(obj)
        assert len(kept) == 26 and len(dropped) == 12
        assert all(t.startswith("B_") for _, t in dropped)


def test_the_derived_regime_carries_an_absolute_bed_path(tmp_path):
    """`regions.bed` resolves against the regime file's directory, and the derived copy lands on
    /scratch beside a checkpoint. A relative path there fails D32's sha256 gate."""
    from candi.store.regime import Regime

    d = write_select_regime(REPO / "configs" / "regime.eic_pilot.json", tmp_path / "sel.json")
    assert Path(d["regions"]["bed"]).is_absolute()
    r = Regime.from_file(tmp_path / "sel.json")           # the hash gate, run for real
    assert {t.split("_", 1)[0] for _, t in r.eval_pairs} == {"V"}


def test_no_test_biosample_is_predicted_during_a_run(scope, tmp_path):
    """End to end: the prediction root the loop scores holds only `V_` tracks."""
    _, rp, binned = scope
    out = tmp_path / "c.pt"
    assert avo_train.main([
        "--regime", str(rp), "--chrom", "chr2", "--mode", "shared",
        "--data-root", str(binned), "--out", str(out), "--epochs", "1",
        "--batch-positions", "64", "--select-every", "1"]) == 0
    written = sorted(p.name for p in Path(str(out) + ".select_pred").iterdir() if p.is_dir())
    assert written and not [d for d in written if "B_" in d]


# ---------------------------------------------------------------------------
# the write — the moment the metric improves
# ---------------------------------------------------------------------------

def _run_with_curve(rp, binned, out, curve, monkeypatch, extra=()):
    """Drive the loop with a scripted metric, recording `.best.pt` after every check."""
    import candi.bench.external as ext

    seen, calls = [], {"i": 0}

    def fake(source, pred_root, **kw):
        v = curve[calls["i"]]
        calls["i"] += 1
        return {"macro": {"pval": {"mse": v, "n_tracks": 2}}}

    def spy(*a, **kw):
        res = fake(*a, **kw)
        p = Path(str(out) + ".best.pt")
        seen.append((res["macro"]["pval"]["mse"], p.read_bytes() if p.exists() else None))
        return res

    # Recorded BEFORE the value is used, so `seen[i]` is the file as it stood going INTO check i.
    monkeypatch.setattr(ext, "score_external", spy)
    rc = avo_train.main([
        "--regime", str(rp), "--chrom", "chr2", "--mode", "shared", "--data-root", str(binned),
        "--out", str(out), "--epochs", str(len(curve)), "--batch-positions", "64",
        "--select-every", "1", *extra])
    return rc, seen


def test_the_best_weights_are_written_the_moment_the_metric_improves(scope, tmp_path, monkeypatch):
    """Improvement at check 0 and check 1, none at check 2 — so the bytes change exactly twice."""
    _, rp, binned = scope
    out = tmp_path / "c.pt"
    rc, seen = _run_with_curve(rp, binned, out, [3.0, 1.0, 2.0], monkeypatch)
    assert rc == 0
    assert seen[0][1] is None, "nothing has been selected before the first check"
    assert seen[1][1] is not None, "check 0 improved and its weights were not on disk"
    assert seen[2][1] is not None and seen[2][1] != seen[1][1], "check 1 improved and did not write"
    # check 2 was worse, so the run ends with check 1's weights, moved into place as `--out`.
    assert out.read_bytes() == seen[2][1]
    assert torch.load(out, map_location="cpu", weights_only=False)["selection"]["value"] == 1.0


def test_the_checkpoint_the_run_ships_is_the_selected_one_not_the_last(scope, tmp_path, monkeypatch):
    """`--out` is what `predict.py` is pointed at, so `--out` must be the selected weights."""
    _, rp, binned = scope
    out = tmp_path / "c.pt"
    rc, _ = _run_with_curve(rp, binned, out, [1.0, 4.0, 9.0], monkeypatch)
    assert rc == 0
    sel = torch.load(out, map_location="cpu", weights_only=False)
    last = torch.load(str(out) + ".last.pt", map_location="cpu", weights_only=False)
    assert sel["selection"]["epoch"] == 0 and sel["selection"]["value"] == 1.0
    assert last["steps"] > sel["steps"], "the last epoch was not kept beside the selected one"
    assert not torch.equal(sel["model"]["cell.weight"], last["model"]["cell.weight"])


def test_a_stalled_run_stops_and_keeps_the_weights_it_already_selected(scope, tmp_path, monkeypatch):
    """Patience is counted in epochs, and stopping loses nothing: the best is already on disk."""
    _, rp, binned = scope
    out = tmp_path / "c.pt"
    rc, seen = _run_with_curve(rp, binned, out, [1.0, 4.0, 4.0, 4.0, 4.0], monkeypatch,
                               extra=("--select-patience", "1"))
    assert rc == 0
    assert len(seen) == 3, "the run should have stopped two epochs after the best, not run on"
    assert torch.load(out, map_location="cpu", weights_only=False)["selection"]["epoch"] == 0


# ---------------------------------------------------------------------------
# the instrument — the same scorer that ranks the board
# ---------------------------------------------------------------------------

def test_the_selection_metric_comes_from_the_scorer_that_ranks_the_board(scope, tmp_path):
    """No stub: the value logged by the loop must be `score_external`'s own macro key.

    Recomputed here by scoring the run's own prediction root through the public entry point, which
    is what makes "uniform selection" a fact about which code ran rather than a claim.
    """
    from candi.bench.external import score_external
    from candi.bench.harness import open_source

    _, rp, binned = scope
    out = tmp_path / "c.pt"
    log = tmp_path / "t.jsonl"
    assert avo_train.main([
        "--regime", str(rp), "--chrom", "chr2", "--mode", "shared", "--data-root", str(binned),
        "--out", str(out), "--epochs", "1", "--batch-positions", "64",
        "--select-every", "1", "--log", str(log)]) == 0

    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    logged = [r for r in rows if "select_value" in r][-1]["select_value"]
    src = open_source(store=str(out) + ".select_regime.json", chroms=("chr2",))
    try:
        again = score_external(src, Path(str(out) + ".select_pred"))
    finally:
        src.close()
    assert logged == pytest.approx(again["macro"]["pval"]["mse"])
    assert again["provenance"]["method"] == "Avocado"


def test_a_distributional_metric_is_refused_while_avocado_has_no_sigma(scope, tmp_path):
    """Avocado emits a point. `crps` without a §6.1 σ-table is not a small number, it is no number."""
    _, rp, binned = scope
    with pytest.raises(SystemExit, match="sigma-table"):
        avo_train.main(["--regime", str(rp), "--chrom", "chr2", "--mode", "shared",
                        "--data-root", str(binned), "--out", str(tmp_path / "c.pt"),
                        "--epochs", "1", "--select-every", "1", "--select-metric", "crps"])


def test_a_bed_scoped_fit_refuses_to_claim_it_selected(scope, tmp_path):
    """The one stage that cannot select says so and stops, rather than quietly keeping the last."""
    _, rp, binned = scope
    with pytest.raises(SystemExit, match="no V_ panel"):
        avo_train.main(["--regime", str(rp), "--chrom", "regions", "--mode", "shared",
                        "--data-root", str(binned), "--out", str(tmp_path / "c.pt"),
                        "--positions", str(binned / "regions_layout.csv"),
                        "--epochs", "1", "--select-every", "1"])


def test_a_run_without_selection_keeps_the_last_epoch_and_writes_no_best(scope, tmp_path):
    """005's behaviour, still reachable — and still the thing §5 forbids for a trainable method."""
    _, rp, binned = scope
    out = tmp_path / "c.pt"
    assert avo_train.main([
        "--regime", str(rp), "--chrom", "chr2", "--mode", "shared", "--data-root", str(binned),
        "--out", str(out), "--epochs", "1", "--batch-positions", "64"]) == 0
    assert not Path(str(out) + ".best.pt").exists()
    assert torch.load(out, map_location="cpu", weights_only=False)["selection"] is None
