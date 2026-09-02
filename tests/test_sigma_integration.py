"""The σ chain end to end, over the three files that have to agree about ONE regime shape.

    tools/sigma_training_regime.py            writes the derived regime
    competitors.baselines.generate            writes the avg-arcsinh training-track root
    competitors.sigma_pass                    fits σ on the residual between them

Each of those is unit-tested where it lives (`tests/test_sigma_pass.py`, `tests/test_baselines.py`).
What is tested HERE is the seam: the derived regime declares **no `eval_pairs` at all**, because
`candi.store.regime` refuses a `[c, c]` literal outright, and every consumer has to read that shape
the way `bench.harness.StoreSource` does — self-pair `biosamples.eval`, and a self-pair's panel is
every assay the cell holds.

Two consumers did not, and both failures were silent:

* `competitors/baselines/generate.py` read `eval_pairs` alone, so `--methods avg-arcsinh --store
  <derived>` wrote a root with no track in it and a manifest saying so. The σ pass then had nothing
  to take a residual against, and the `avg-arcsinh` tier could get no σ table.
* `competitors/lavawizard/store_eic.py::_declared_tracks` did the same, so `write_predictions`
  wrote zero tracks while its caller still wrote a manifest — a σ prediction root that looks
  finished and holds nothing.

A test per consumer would have caught neither: each one's own contract was satisfied. So the chain
is run, once, on a real synthetic store.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path

import numpy as np
import pytest

from candi.bench.external import _expected
from candi.bench.harness import Pair, open_source
from candi.store.reader import CorpusStore
from competitors import sigma_pass as SP
from competitors.baselines import generate as Gen
from tests.test_store_reader import make_store
from tests.test_store_regime import regime_dict

REPO = Path(__file__).resolve().parent.parent

#: Three training cells and one eval cell. `T_bb` lacks DNase-seq, so the self-paired panels are not
#: all the same size — a σ table whose `n_tracks` is uniform could not tell a per-cell panel from a
#: per-regime one. Cell types differ ("aa", "bb", "cc"), so §5's exclusion leaves every self-paired
#: track at least one contributor and nothing is skipped.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "H3K4me3"),
    "T_cc": ("ATAC-seq", "DNase-seq", "H3K4me3"),
    "V_aa": ("DNase-seq", "H3K4me3"),
}
TRAIN = ["T_aa", "T_bb", "T_cc"]


def _load_tool():
    """`tools/sigma_training_regime.py` — a script, not a package module."""
    path = REPO / "tools" / "sigma_training_regime.py"
    spec = _ilu.spec_from_file_location("sigma_training_regime", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOL = _load_tool()


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("sigmaint"), tracks=TRACKS)


def _source_regime(store: Path, d: Path) -> Path:
    """What `configs/regime.eic_19.json` is: a declared `T_ -> V_` panel over a training pool."""
    p = d / "regime.source.json"
    p.write_text(json.dumps(regime_dict(store, biosamples={"train": TRAIN, "eval": ["V_aa"]},
                                        kinds=["counts", "peaks", "pval"],
                                        eval_pairs=[["T_aa", "V_aa"]])), encoding="utf-8")
    return p


def _derive(source: Path, out: Path, *, n_cells: int = 2) -> Path:
    assert TOOL.main(["--regime", str(source), "--n-cells", str(n_cells),
                      "--seed", "890217", "--out", str(out)]) == 0
    return out


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def test_the_derived_regime_is_the_no_pairing_shape_every_consumer_must_read(store, tmp_path):
    """The premise of everything below, asserted once so a shape change fails here first."""
    derived = json.loads(_derive(_source_regime(store, tmp_path),
                                 tmp_path / "regime.sigma.json").read_text(encoding="utf-8"))
    assert derived["eval_pairs"] == []
    assert derived["train_chroms"] == []
    assert derived["biosamples"]["train"] == TRAIN
    assert set(derived["biosamples"]["eval"]) <= set(TRAIN) and derived["biosamples"]["eval"]


def test_the_baselines_write_the_training_tracks_the_sigma_pass_asks_for(store, tmp_path):
    """`avg-arcsinh` under the derived regime covers exactly `bench.external._expected`'s list."""
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    roots = Gen.main(["--store", str(derived), "--out", str(tmp_path / "pred"),
                      "--methods", "avg-arcsinh", "--quiet"])
    assert roots == 0
    root = tmp_path / "pred" / "avg-arcsinh"
    source = open_source(store=derived)
    try:
        want = set(_expected(source))
    finally:
        source.close()
    got = {d.name for d in root.iterdir() if d.is_dir()}
    assert want, "the derived regime declares no track, so this proves nothing"
    assert got == want, f"root holds {sorted(got)}, the σ pass will ask for {sorted(want)}"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skipped_tracks"] == [], \
        "a self-paired track with no contributor would make the σ pass need --allow-missing"


def test_the_sigma_pass_fits_a_training_residual_table_on_that_root(store, tmp_path):
    """The whole chain, and the two facts a σ table is quoted on: the prefix and `n_tracks`."""
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    assert Gen.main(["--store", str(derived), "--out", str(tmp_path / "pred"),
                     "--methods", "avg-arcsinh", "--quiet"]) == 0
    out = tmp_path / "sigma.json"
    assert SP.main(["--regime", str(derived), "--pred", str(tmp_path / "pred" / "avg-arcsinh"),
                    "--out", str(out), "--method", "avg-arcsinh"]) == 0

    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["fitted_on"].startswith(SP.SIGMA_FITTED_ON_PREFIX)
    assert table["method"] == "avg-arcsinh"

    # Every assay the drawn cells hold, and every one of them fitted on at least one track. An
    # empty root used to produce no table at all; a root missing a cell would show up as a 1 here.
    cells = json.loads(derived.read_text(encoding="utf-8"))["biosamples"]["eval"]
    with CorpusStore(store) as corpus:
        want = {a: sum(1 for c in cells if a in set(corpus[c].assays()))
                for a in {a for c in cells for a in TRACKS[c]}}
    assert table["n_tracks"] == want
    assert all(v > 0 for v in table["n_tracks"].values())
    assert all(np.isfinite(v) and v > 0 for v in table["sigma"].values())
    assert set(table["cells"]) == set(cells)


def test_no_eval_panel_track_is_named_anywhere_in_the_chain(store, tmp_path):
    """Rule 1, on the artefacts rather than on the code path: `V_aa` appears in neither."""
    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    assert Gen.main(["--store", str(derived), "--out", str(tmp_path / "pred"),
                     "--methods", "avg-arcsinh", "--quiet"]) == 0
    out = tmp_path / "sigma.json"
    assert SP.main(["--regime", str(derived), "--pred", str(tmp_path / "pred" / "avg-arcsinh"),
                    "--out", str(out), "--method", "avg-arcsinh"]) == 0
    names = [d.name for d in (tmp_path / "pred" / "avg-arcsinh").iterdir() if d.is_dir()]
    assert names and not [n for n in names if "V_" in n or "B_" in n]
    assert "V_aa" not in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lavawizard reads the same shape through its own function
# ---------------------------------------------------------------------------

def test_lavawizard_declared_tracks_self_pairs_the_eval_biosamples(store, tmp_path):
    """`_declared_tracks` must name what `bench.external._expected` will ask the root for.

    Count is Σ over the drawn cells of the assays that cell holds — the self-pair's panel — and NOT
    zero, which is what reading `eval_pairs` alone returned on this shape.
    """
    from competitors.lavawizard.store_eic import _declared_tracks

    derived = _derive(_source_regime(store, tmp_path), tmp_path / "regime.sigma.json")
    obj = json.loads(derived.read_text(encoding="utf-8"))
    cells = obj["biosamples"]["eval"]
    with CorpusStore(store) as corpus:
        got = _declared_tracks(obj, corpus)
        want = [(c, c, a) for c in cells for a in obj["assays"]
                if a in set(corpus[c].assays())]
    assert got == want
    assert len(got) == sum(len(set(TRACKS[c]) & set(obj["assays"])) for c in cells)
    assert all(src == tgt for src, tgt, _ in got), "a σ regime declares no cross-cell pair"

    source = open_source(store=derived)
    try:
        expected = {(p.input_biosample, p.target_biosample, source.assays[a])
                    for p in source.pairs("impute") for a in source.targets(p, "impute")}
    finally:
        source.close()
    assert set(got) == expected, \
        "Lavawizard would write a track set the scorer does not ask for, or miss one it does"


def test_lavawizard_declared_tracks_still_reads_a_declared_pairing(store, tmp_path):
    """The fallback is a FALLBACK: a regime that declares pairs is unaffected by it."""
    from competitors.lavawizard.store_eic import _declared_tracks

    obj = json.loads(_source_regime(store, tmp_path).read_text(encoding="utf-8"))
    with CorpusStore(store) as corpus:
        got = _declared_tracks(obj, corpus)
    assert got == [("T_aa", "V_aa", a) for a in obj["assays"] if a in set(TRACKS["V_aa"])]
    assert {tgt for _, tgt, _ in got} == {"V_aa"}


def test_the_two_pair_readers_agree_on_the_declared_shape_too(store, tmp_path):
    """Both readers, one regime with pairs: the generator's panel is the harness's panel."""
    source_regime = _source_regime(store, tmp_path)
    panel = Gen.Panel(source_regime)
    source = open_source(store=source_regime)
    try:
        assert panel.pairs == [("T_aa", "V_aa")] == [(p.input_biosample, p.target_biosample)
                                                     for p in source.pairs("impute")]
        want = sorted(source.assays[a] for a in source.targets(Pair("T_aa", "V_aa"), "impute"))
        assert sorted(panel.targets(("T_aa", "V_aa"))) == want
    finally:
        source.close()
        panel.close()
