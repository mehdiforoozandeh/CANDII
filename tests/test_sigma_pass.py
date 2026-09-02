"""t91 / D2 — the TRAINING-RESIDUAL σ pass, against a REAL store on disk.

Nothing is mocked. `make_store` writes an actual `CANDI_STORE`, `tools/sigma_training_regime.py`
derives a regime from a realistic source one, and `competitors/sigma_pass.py` fits σ over
predictions written in the §4.1 on-disk format — the same thing a rival's `predict` script emits.

Three properties carry the whole point of the pass and each has a test that would fail without it:

* **σ is the root mean squared TRAINING residual.** A prediction offset from truth by a known
  constant must produce exactly that constant, per assay, pooled over tracks and bins.
* **No `V_`/`B_` track is ever opened (Rule 1).** Asserted twice: the fitter exits 3 on a regime
  that names one, and a spy on `CorpusStore.__getitem__` proves the happy path never reaches one
  even though the store holds one.
* **`fitted_on` says which.** The prefix is the only thing that tells a leak-free table from a leaky
  one months later, so it is checked as a string, and the four eval-pair fitters that could write
  the leaky one are asserted gone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from candi.bench.external import read_sigma_table
from candi.bench.harness import open_source
from candi.store.reader import CorpusStore
from candi.store.regime import Regime, RegimeError
from competitors import sigma_pass as SP
from tests.test_store_reader import make_store
from tests.test_store_regime import CTX, regime_dict

import importlib.util as _ilu

REPO = Path(__file__).resolve().parent.parent

#: `T_aa` and `T_bb` are the training cells the σ pass fits on; `V_aa` exists ONLY so that "the
#: fitter never opens it" is a fact about a store that has one rather than about an empty shelf.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "H3K4me3"),
    "V_aa": ("DNase-seq", "H3K4me3"),
}
#: The constant each assay's prediction is offset from truth by, so σ is known in closed form.
OFFSET = {"ATAC-seq": 2.0, "DNase-seq": 0.5, "H3K4me3": 3.0}

MANIFEST = {
    "method": "test-rival",
    "version": "0.0.1",
    "generated_by": "tests/test_sigma_pass.py",
    "date": "2026-09-01",
    "arms": ["pval"],
    "notes": "training-track predictions, offset from truth by a known constant",
}


def _load_tool():
    """`tools/sigma_training_regime.py` — a script, not a package module."""
    path = REPO / "tools" / "sigma_training_regime.py"
    spec = _ilu.spec_from_file_location("sigma_training_regime", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOL = _load_tool()


# ---------------------------------------------------------------------------
# a real store and a realistic SOURCE regime (the thing configs/regime.eic_19.json is)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("sigmastore"), tracks=TRACKS)


def _source_regime(store: Path, d: Path, *, regions=None) -> Path:
    obj = regime_dict(store,
                      biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"]])
    if regions is not None:
        obj["regions"] = regions
    p = d / "regime.source.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _bed(d: Path, name: str = "scope.bed") -> Path:
    """A BED holding whole 64-bin windows of chr1 — 20 of the 32 the chromosome carries."""
    p = d / name
    p.write_text("chr1\t0\t32000\tscope\n", encoding="utf-8")
    return p


def _derive(source: Path, out: Path, *, n_cells: int = 2, seed: int = 890217) -> Path:
    assert TOOL.main(["--regime", str(source), "--n-cells", str(n_cells),
                      "--seed", str(seed), "--out", str(out)]) == 0
    return out


def _truth(store: Path, cell: str, assay: str, chrom: str = "chr1") -> np.ndarray:
    """Truth read straight off the store — the independent path the fitter is checked against."""
    c = CorpusStore(store)
    try:
        return np.asarray(c[cell][assay].pval(chrom, 0), dtype=np.float32)
    finally:
        c.close()


def _write_pred(root: Path, store: Path, derived: Path, *, offset=OFFSET,
                manifest=None) -> Path:
    """A §4.1 root of TRAINING-track predictions: `signal_mu = truth + offset[assay]`."""
    from candi.bench.external import _expected

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST if manifest is None else manifest),
                                        encoding="utf-8")
    source = open_source(store=derived)
    try:
        chroms = list(source.eval_chroms)
        for dirname, (pair, assay) in sorted(_expected(source).items()):
            d = root / dirname
            d.mkdir(exist_ok=True)
            for c in chroms:
                mu = _truth(store, pair.target_biosample, assay, c) + np.float32(offset[assay])
                np.savez(d / f"{c}.npz", signal_mu=mu.astype(np.float32))
    finally:
        source.close()
    return root


# ---------------------------------------------------------------------------
# the derived regime
# ---------------------------------------------------------------------------

def test_the_derived_regime_self_pairs_the_sampled_training_cells(store, tmp_path):
    src = _source_regime(store, tmp_path)
    out = _derive(src, tmp_path / "regime.sigma.json")
    obj = json.loads(out.read_text())

    assert obj["_comment"].startswith("DERIVED training-residual sigma regime")
    assert obj["biosamples"]["eval"] == ["T_aa", "T_bb"]
    assert obj["eval_pairs"] == []
    # The source's TRAINING slice is what gets scored, and nothing trains — the two lists may not
    # overlap, which is why `train_chroms` empties rather than being copied across.
    assert obj["eval_chroms"] == ["chr1"]
    assert obj["train_chroms"] == []
    assert "V_" not in out.read_text() and "B_" not in out.read_text()

    source = open_source(store=out)
    try:
        pairs = source.pairs("impute")
        assert [(p.input_biosample, p.target_biosample) for p in pairs] == \
            [("T_aa", "T_aa"), ("T_bb", "T_bb")]
        # `targets` on a self-pair is every assay the cell holds — the whole training panel.
        assert [source.assays[a] for a in source.targets(pairs[0], "impute")] == \
            ["ATAC-seq", "DNase-seq", "H3K4me3"]
    finally:
        source.close()


def test_a_declared_self_pair_does_not_load_which_is_why_eval_pairs_is_empty(store, tmp_path):
    """The spelling the derived file CANNOT use, pinned so a future edit does not try it again."""
    obj = regime_dict(store, biosamples={"train": ["T_aa"], "eval": ["T_aa"]},
                      kinds=["counts", "peaks", "pval"], eval_pairs=[["T_aa", "T_aa"]])
    p = tmp_path / "selfpair.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RegimeError, match="with itself"):
        Regime.from_file(p)


def test_the_draw_is_seeded_and_re_derivable(store, tmp_path):
    src = _source_regime(store, tmp_path)
    a = json.loads(_derive(src, tmp_path / "a.json", n_cells=1, seed=7).read_text())
    b = json.loads(_derive(src, tmp_path / "b.json", n_cells=1, seed=7).read_text())
    assert a["biosamples"]["eval"] == b["biosamples"]["eval"]
    assert len(a["biosamples"]["eval"]) == 1
    assert set(a["biosamples"]["eval"]) <= {"T_aa", "T_bb"}
    assert a["_source_regime"]["sample"] == {"n_cells": 1, "seed": 7,
                                             "rule": "default_rng(seed).permutation"}
    assert TOOL.sample_cells(["T_aa", "T_bb"], 1, 7) == TOOL.sample_cells(["T_aa", "T_bb"], 1, 7)


def test_asking_for_more_cells_than_the_regime_trains_on_is_refused(store, tmp_path):
    src = _source_regime(store, tmp_path)
    with pytest.raises(SystemExit, match="without replacement"):
        TOOL.main(["--regime", str(src), "--n-cells", "9", "--seed", "1",
                   "--out", str(tmp_path / "x.json")])


def test_an_eval_panel_cell_in_the_training_pool_exits_3(store, tmp_path):
    obj = regime_dict(store, biosamples={"train": ["T_aa", "V_aa"], "eval": ["T_bb"]},
                      kinds=["counts", "peaks", "pval"])
    p = tmp_path / "leaky.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    assert TOOL.main(["--regime", str(p), "--n-cells", "1", "--seed", "1",
                      "--out", str(tmp_path / "y.json")]) == TOOL.EXIT_EVAL_PANEL


def test_the_derived_regime_makes_the_region_bed_absolute(store, tmp_path):
    """`eic_pilot` declares `regions/…bed` relative to `configs/`; the derived file lives elsewhere."""
    bed = _bed(tmp_path)
    import hashlib
    sha = hashlib.sha256(bed.read_bytes()).hexdigest()
    src = _source_regime(store, tmp_path,
                         regions={"bed": bed.name, "sha256": sha, "policy": "contain"})
    out = _derive(src, tmp_path / "sub" / "regime.sigma.json")
    obj = json.loads(out.read_text())
    assert Path(obj["regions"]["bed"]).is_absolute()
    assert obj["regions"]["sha256"] == sha
    assert Regime.from_file(out).regions.resolved == str(bed.resolve())


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def test_the_self_pair_truth_is_the_cells_own_pval_off_the_store(store, tmp_path):
    """`stream_truth` reads `y_*_imp`, which a self-pair batch does not carry; the alias is exact."""
    from candi.bench.external import stream_truth

    out = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    source = open_source(store=out)
    try:
        pair = source.pairs("impute")[0]
        cols = source.targets(pair, "impute")
        got = stream_truth(SP._SelfPairTruth(source, "chr1"), pair, cols, batch_windows=2)
        for col in cols:
            assay = source.assays[col]
            assert np.array_equal(got[col]["chr1"]["pval"],
                                  _truth(store, pair.target_biosample, assay))
    finally:
        source.close()


def test_sigma_is_the_root_mean_squared_training_residual(store, tmp_path):
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived)
    out = tmp_path / "sigma.json"
    assert SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out),
                    "--method", "Avocado"]) == 0

    table = json.loads(out.read_text())
    assert table["method"] == "Avocado"
    for assay, k in OFFSET.items():
        assert table["sigma"][assay] == pytest.approx(k, rel=1e-4)
    # ATAC-seq and H3K4me3 are on both cells; DNase-seq only on T_aa. 2000 bins on chr1.
    assert table["n_tracks"] == {"ATAC-seq": 2, "DNase-seq": 1, "H3K4me3": 2}
    assert table["n_points"] == {"ATAC-seq": 4000, "DNase-seq": 2000, "H3K4me3": 4000}
    assert table["cells"] == ["T_aa", "T_bb"]


def test_the_table_says_it_was_fitted_on_training_residuals(store, tmp_path):
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived)
    out = tmp_path / "sigma.json"
    SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out),
             "--method", "eDICE"])
    table = json.loads(out.read_text())

    assert SP.SIGMA_FITTED_ON_PREFIX == "training-residuals:"
    assert table["fitted_on"].startswith(SP.SIGMA_FITTED_ON_PREFIX)
    assert "eval_pairs" not in table["fitted_on"]
    assert derived.name in table["fitted_on"]
    assert "2 cells" in table["fitted_on"] and "chr1" in table["fitted_on"]
    import hashlib
    assert table["regime_sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
    assert table["pred_manifest_sha256"] == \
        hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_the_table_is_accepted_by_the_bench_reader(store, tmp_path):
    """The two sides of the §4.2 boundary must agree on the shape, not just on our reading of it."""
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived)
    out = tmp_path / "sigma.json"
    SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out),
             "--method", "Lavawizard"])
    table = read_sigma_table(out)
    assert table["sigma"]["H3K4me3"] == pytest.approx(3.0, rel=1e-4)


def test_a_zero_residual_takes_the_floor_and_the_floor_is_recorded(store, tmp_path):
    """`read_sigma_table` refuses σ <= 0, so a perfect track must still carry a positive width."""
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived,
                       offset={a: 0.0 for a in OFFSET})
    out = tmp_path / "sigma.json"
    SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out), "--method", "avg"])
    table = json.loads(out.read_text())

    from competitors.baselines.heads import SIGMA_FLOOR
    assert table["sigma_floor"] == SIGMA_FLOOR
    assert table["sigma_floor_source"] == "competitors.baselines.heads.SIGMA_FLOOR"
    assert sorted(table["floored_assays"]) == sorted(OFFSET)
    assert all(v == SIGMA_FLOOR for v in table["sigma"].values())
    read_sigma_table(out)          # positive, so the bench still accepts it


def test_eval_regions_cuts_the_residual_to_the_bed(store, tmp_path):
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived)
    bed = _bed(tmp_path)
    out = tmp_path / "sigma.json"
    assert SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out),
                    "--method", "ChromImpute", "--eval-regions", str(bed)]) == 0
    table = json.loads(out.read_text())

    # 20 whole 64-bin windows inside chr1:0-32000, against 2000 bins at full coverage.
    assert table["n_points"]["DNase-seq"] == 20 * CTX
    assert table["eval_scope"]["name"] == "regions"
    import hashlib
    assert f"regions {hashlib.sha256(bed.read_bytes()).hexdigest()[:12]}" in table["fitted_on"]
    # The offset is constant, so cutting the positions must not move σ.
    assert table["sigma"]["DNase-seq"] == pytest.approx(0.5, rel=1e-4)


def test_a_prediction_root_that_covers_none_of_the_training_tracks_is_refused(store, tmp_path):
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = tmp_path / "empty"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    with pytest.raises(SystemExit, match="training-track prediction root"):
        SP.main(["--regime", str(derived), "--pred", str(root), "--out",
                 str(tmp_path / "s.json"), "--method", "Avocado"])


# ---------------------------------------------------------------------------
# Rule 1 — no V_/B_ track is opened, ever
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    {"biosamples": {"train": ["T_aa"], "eval": ["V_aa"]}},
    {"eval_pairs": [["T_aa", "V_aa"]]},
])
def test_the_fitter_exits_3_on_a_regime_that_names_an_eval_panel_cell(store, tmp_path, edit):
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    obj = json.loads(derived.read_text())
    obj.update(edit)
    leaky = tmp_path / "leaky.sigma.json"
    leaky.write_text(json.dumps(obj), encoding="utf-8")
    assert SP.main(["--regime", str(leaky), "--pred", str(tmp_path), "--out",
                    str(tmp_path / "s.json"), "--method", "Avocado"]) == SP.EXIT_EVAL_PANEL
    assert not (tmp_path / "s.json").exists()


def test_the_fitter_never_opens_a_V_or_B_biosample(store, tmp_path, monkeypatch):
    """Rule 1 as a fact about what was read, not about what the regime file says.

    The store HOLDS `V_aa`. Every `CorpusStore[name]` the whole run performs is recorded, and none
    of them may be an eval-panel cell.
    """
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    root = _write_pred(tmp_path / "pred", store, derived)

    seen: list = []
    real = CorpusStore.__getitem__
    monkeypatch.setattr(CorpusStore, "__getitem__",
                        lambda self, name: (seen.append(name), real(self, name))[1])
    out = tmp_path / "sigma.json"
    assert SP.main(["--regime", str(derived), "--pred", str(root), "--out", str(out),
                    "--method", "Avocado"]) == 0
    assert seen, "the spy caught nothing, so it proves nothing"
    assert [n for n in seen if n.startswith(("V_", "B_"))] == []


# ---------------------------------------------------------------------------
# the four eval-pair fitters are retired
# ---------------------------------------------------------------------------

def test_the_eval_pair_fitters_are_gone():
    """They wrote `fitted_on = "<regime> eval_pairs"`, which Rule 1 voids. One fitter replaces them."""
    left = [m for m in ("avocado", "edice", "lavawizard", "chromimpute")
            if (REPO / "competitors" / m / "fit_sigma.py").exists()]
    assert left == []


def test_no_rival_readme_still_points_at_a_deleted_fitter():
    hits = [str(p) for p in sorted((REPO / "competitors").glob("*/README.md"))
            if "fit_sigma.py" in p.read_text(encoding="utf-8")]
    assert hits == []
