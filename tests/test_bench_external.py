"""t47 — the external-track entry (`RIVALS_PLAN.md` §4), against a REAL store on disk.

Nothing is mocked. `make_store` writes an actual `CANDI_STORE`, a real `CandiModel` is streamed
over it by `harness.stream_tracks`, and the resulting predictions are written out to the §4.1
on-disk format with `np.savez` — the same thing a competitor's `predict_*` script will do.

The headline test is the §4.3 acceptance gate and it is an EQUALITY, not a smoke test: score the
model the normal way, stream its own predictions to disk, score them back through the external
entry, and every shared numeric key must agree. That is what makes "the external path is the same
instrument" a fact rather than a claim — if the two paths ever diverge (a different truth read, a
re-derived metric, a grid shifted by one bin) this test is the thing that says so.

The rest of the file is the refusals. A prediction root is written by code we do not own, so every
way it can be wrong quietly — an array a bin short, a track naming a biosample the regime never
declared, a panel with holes in it — must be loud and must name the offending track.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from candi.bench import harness as H
from candi.bench.external import (
    WITHHELD_WITHOUT_PEAK_TRUTH, ExternalError, main, read_sigma_table, score_external,
    track_dirname,
)
from candi.bench.harness import Pair, open_source, track_key
from candi.model import build_model
from candi.store import layout as L

from tests.test_store_reader import ASSAYS, N_BINS, make_store
from tests.test_store_regime import CTX, regime_dict

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: Two declared pairs, and in each the prompt cell LACKS the assay the truth cell carries — the only
#: layout that poses imputation. Both target cells carry H3K4me3, so `panel_specificity` has two
#: cell types to compare rather than declining.
PAIRED_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_bb": ("ATAC-seq", "H3K4me3"),
}
PAIRS = (Pair("T_aa", "V_aa"), Pair("T_bb", "V_bb"))
TARGET_ASSAY = "H3K4me3"

MANIFEST = {
    "method": "candi-roundtrip",
    "version": "0.0.1",
    "generated_by": "tests/test_bench_external.py",
    "date": "2026-08-25",
    "arms": ["pval", "count"],
    "notes": "CANDI's own predictions, streamed to the §4.1 format",
}

#: `--c-index-pairs` for every scored run here. The C-index is the one sampled measure (D3); the
#: two paths must be given the SAME budget and the same seed or they sample different pairs and the
#: round-trip would compare two estimates instead of one number.
C_PAIRS = 2_000


# ---------------------------------------------------------------------------
# a real store, a real regime, a real checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("extstore"), tracks=PAIRED_TRACKS)


@pytest.fixture(scope="module")
def regime_file(store, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("extregime")
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]])
    p = d / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def model():
    torch_seed = 0
    import torch
    torch.manual_seed(torch_seed)
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, heads=("count", "signal", "peak"))
    m.eval()
    return m


def _source(regime_file, **kw):
    return open_source(store=regime_file, **kw)


# ---------------------------------------------------------------------------
# writing the §4.1 format — this is what a competitor's predict script does
# ---------------------------------------------------------------------------

def write_root(root: Path, recs, *, keep=("signal_mu", "signal_sigma", "mu", "n", "peak_score"),
               manifest=None, mangle=None) -> Path:
    """Stream `TrackRecord`s out to `<root>/<in>__<out>__<assay>/chr*.npz` + `manifest.json`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST if manifest is None else manifest),
                                        encoding="utf-8")
    for rec in recs:
        d = root / track_dirname(rec.pair, rec.assay)
        d.mkdir(exist_ok=True)
        for c in rec.chroms:
            payload = {}
            for name in keep:
                src = getattr(rec, name)
                if c in src:
                    payload[name] = np.asarray(src[c], dtype=np.float32)
            if mangle is not None:
                payload = mangle(d.name, c, payload)
            np.savez(d / f"{c}.npz", **payload)
    return root


@pytest.fixture(scope="module")
def recs(regime_file, model):
    """CANDI's own predictions for both declared pairs, as the harness assembles them."""
    src = _source(regime_file)
    try:
        return list(H.stream_tracks(model, src, "cpu", kind="impute", batch_windows=4))
    finally:
        src.close()


@pytest.fixture(scope="module")
def model_scores(regime_file, model):
    """The normal bench path. `C` is off: it perturbs a prompt and re-decodes, which needs a model,
    so the external entry cannot emit it and there is nothing to compare."""
    src = _source(regime_file)
    try:
        return H.run_bench(model, src, "cpu", kinds=("impute",), batch_windows=4,
                           blocks=("E", "P", "D", "B"), c_index_pairs=C_PAIRS, seed=0)
    finally:
        src.close()


@pytest.fixture(scope="module")
def full_root(tmp_path_factory, recs) -> Path:
    return write_root(tmp_path_factory.mktemp("extpred") / "full", recs)


@pytest.fixture(scope="module")
def external_scores(regime_file, full_root):
    src = _source(regime_file)
    try:
        return score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def _numeric(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and
            not isinstance(v, bool)}


def _same(a, b) -> bool:
    """Exact equality with `nan == nan`. Several blocks emit a real nan (an annotation set that
    selects no bin, a region correlation with nothing to correlate), and `nan != nan` would turn a
    bit-identical row into a failure."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


# ---------------------------------------------------------------------------
# 1 — §4.3, the acceptance gate: the same instrument, proven
# ---------------------------------------------------------------------------

def test_the_stream_covers_both_declared_pairs(recs) -> None:
    assert {r.key for r in recs} == {f"{p.input_biosample}|{p.target_biosample}|{TARGET_ASSAY}"
                                     for p in PAIRS}


def test_the_external_result_has_run_benchs_shape(external_scores, model_scores) -> None:
    assert set(external_scores) == {"provenance", "tracks", "per_track", "macro", "panels",
                                    "panel", "ranking"}
    # And against `run_bench` itself rather than only against a list retyped here. `model_scores`
    # runs without the C block (it needs the model) and without a split scope, which is exactly
    # what the external path can produce — so the two key sets are EQUAL, and a block added to one
    # path and forgotten on the other fails here.
    assert set(model_scores) == set(external_scores)
    assert external_scores["ranking"] is None
    assert external_scores["tracks"] == sorted(external_scores["per_track"])
    assert set(external_scores["panels"]) == {"count", "pval"}


def test_every_shared_numeric_key_survives_the_round_trip(model_scores, external_scores) -> None:
    """§4.3 — the acceptance gate. Same tracks, same arms, same keys, same numbers."""
    assert external_scores["tracks"] == model_scores["tracks"]
    for key in model_scores["tracks"]:
        mine, theirs = model_scores["per_track"][key], external_scores["per_track"][key]
        assert set(mine) == set(theirs) == {"count", "pval"}, key
        for arm in ("count", "pval"):
            a, b = _numeric(mine[arm]), _numeric(theirs[arm])
            assert set(a) == set(b), f"{key}/{arm} key sets differ"
            for k in sorted(a):
                if np.isfinite(a[k]) or np.isfinite(b[k]):
                    assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"{key}/{arm}/{k}"
                else:
                    assert not np.isfinite(a[k]) and not np.isfinite(b[k]), f"{key}/{arm}/{k}"


def test_the_macro_and_the_panel_survive_the_round_trip(model_scores, external_scores) -> None:
    for arm in ("count", "pval"):
        a, b = _numeric(model_scores["macro"][arm]), _numeric(external_scores["macro"][arm])
        assert set(a) == set(b)
        for k in sorted(a):
            assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"macro/{arm}/{k}"
    assert set(external_scores["panel"]) == set(model_scores["panel"])
    mine = model_scores["panel"][TARGET_ASSAY]
    theirs = external_scores["panel"][TARGET_ASSAY]
    assert theirs["n_cell_types"] == mine["n_cell_types"] == 2
    assert theirs["tracks"] == mine["tracks"]
    for key in mine["tracks"]:
        a, b = _numeric(mine[key]), _numeric(theirs[key])
        assert set(a) == set(b)
        for k in sorted(a):
            if np.isfinite(a[k]) or np.isfinite(b[k]):
                assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"panel/{key}/{k}"
            else:
                assert not np.isfinite(a[k]) and not np.isfinite(b[k]), f"panel/{key}/{k}"


def test_a_real_peak_score_is_recorded_as_a_real_peak_head(external_scores) -> None:
    """The model has a peak head, so `peak_score` is a probability and `bernoulli_nll` is defined."""
    for arms in external_scores["per_track"].values():
        assert np.isfinite(float(arms["pval"]["bernoulli_nll"]))


# ---------------------------------------------------------------------------
# 2 — provenance: a score file traceable to the code that made it (§4.1)
# ---------------------------------------------------------------------------

def test_the_manifest_is_copied_verbatim_and_names_the_method(external_scores) -> None:
    prov = external_scores["provenance"]
    assert prov["method"] == MANIFEST["method"]
    assert prov["manifest"] == MANIFEST
    assert prov["suite"] == "candi.bench.external"
    assert prov["missing_tracks"] == [] and prov["declared_tracks"] == 2
    assert prov["signal_target_transform"] == "none"
    assert prov["pred_inversion"] == "external", (
        "§4.1 — an external signal_mu arrives in -log10 p already, and a reader must be able to "
        "tell that from a CANDI head that was trained in the eval space")


def test_a_root_with_no_manifest_is_refused(regime_file, recs, tmp_path) -> None:
    root = write_root(tmp_path / "nomanifest", recs)
    (root / "manifest.json").unlink()
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match="manifest.json"):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# 3 — the refusals (§4.3): loud, and naming the offending track
# ---------------------------------------------------------------------------

def test_an_array_off_the_grid_is_refused_by_name(regime_file, recs, tmp_path) -> None:
    """One bin short is not nearly right — it is every later bin scored at the wrong position."""
    victim = track_dirname(recs[0].pair, recs[0].assay)

    def clip(dirname, chrom, payload):
        if dirname == victim and chrom == "chr2":
            return {k: v[:-1] for k, v in payload.items()}
        return payload

    root = write_root(tmp_path / "shortgrid", recs, mangle=clip)
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError) as exc:
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    assert victim in str(exc.value)
    assert str(N_BINS["chr2"]) in str(exc.value), "the message must say what the grid is"


def test_a_track_naming_an_undeclared_biosample_is_refused_by_name(regime_file, recs,
                                                                   tmp_path) -> None:
    root = write_root(tmp_path / "stranger", recs)
    stranger = f"T_aa__V_zz__{TARGET_ASSAY}"
    (root / stranger).mkdir()
    for c, n in N_BINS.items():
        np.savez(root / stranger / f"{c}.npz", signal_mu=np.zeros(n, dtype=np.float32))
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=stranger):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def test_an_assay_the_prompt_cell_already_carries_is_not_a_declared_track(regime_file, recs,
                                                                          tmp_path) -> None:
    """ATAC-seq is in both cells, so predicting it measures copying — it is not a scorable track."""
    root = write_root(tmp_path / "shared", recs)
    shared = "T_aa__V_aa__ATAC-seq"
    (root / shared).mkdir()
    for c, n in N_BINS.items():
        np.savez(root / shared / f"{c}.npz", signal_mu=np.zeros(n, dtype=np.float32))
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=shared):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def test_a_missing_declared_pair_fails_the_run_unless_it_is_allowed(regime_file, recs,
                                                                    tmp_path) -> None:
    """D2's lesson applied to a producer we do not control: a partial panel is not a panel."""
    root = write_root(tmp_path / "partial", recs)
    gone = track_dirname(PAIRS[1], TARGET_ASSAY)
    for f in (root / gone).iterdir():
        f.unlink()
    (root / gone).rmdir()

    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=gone):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()

    src = _source(regime_file)
    try:
        got = score_external(src, root, c_index_pairs=C_PAIRS, allow_missing=True)
    finally:
        src.close()
    assert got["provenance"]["missing_tracks"] == [gone]
    assert got["provenance"]["allow_missing"] is True
    assert got["tracks"] == [track_key(PAIRS[0], TARGET_ASSAY, "impute")]


def test_a_missing_chromosome_is_refused_even_when_the_track_is_present(regime_file, recs,
                                                                        tmp_path) -> None:
    root = write_root(tmp_path / "onechrom", recs)
    victim = track_dirname(recs[0].pair, recs[0].assay)
    (root / victim / "chr2.npz").unlink()
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=victim):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def test_half_a_negative_binomial_is_refused(regime_file, recs, tmp_path) -> None:
    def drop_n(dirname, chrom, payload):
        return {k: v for k, v in payload.items() if k != "n"}

    root = write_root(tmp_path / "halfnb", recs, mangle=drop_n)
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match="NEGATIVE BINOMIAL"):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def test_a_regime_with_no_declared_pairs_cannot_score_external_tracks(store, tmp_path,
                                                                      recs) -> None:
    """The format names an input cell; a leave-one-assay-out regime has none to name."""
    p = tmp_path / "nopairs.json"
    p.write_text(json.dumps(regime_dict(store,
                                        biosamples={"train": ["T_aa"], "eval": ["V_aa"]},
                                        kinds=["counts", "peaks", "pval"])), encoding="utf-8")
    root = write_root(tmp_path / "nopairs_root", recs)
    src = _source(p)
    try:
        with pytest.raises(ExternalError, match="eval_pairs"):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# 4 — the arms a producer did not predict: ABSENT, never nan (§4.2)
# ---------------------------------------------------------------------------

GAUSS_ONLY = ("crps", "pit_ks", "coverage_95", "c_index", "c_index_se", "gaussian_nll")
E_KEYS = ("mse", "gwcorr", "gwspear", "mse1obs", "mse1imp")
P_KEYS = ("acc_by_obs_strength", "acc_by_imp_strength")


@pytest.fixture(scope="module")
def point_only(regime_file, recs, tmp_path_factory):
    """A rival's typical output: a `-log10 p` point track and nothing else."""
    root = write_root(tmp_path_factory.mktemp("extpoint"), recs, keep=("signal_mu",))
    src = _source(regime_file)
    try:
        return root, score_external(src, root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()


def test_a_point_only_track_carries_the_point_blocks_and_no_others(point_only) -> None:
    _root, got = point_only
    for key, arms in got["per_track"].items():
        assert set(arms) == {"pval"}, f"{key}: no mu/n was supplied, so there is no count arm"
        arm = arms["pval"]
        for k in E_KEYS + P_KEYS:
            assert k in arm, f"{key}: the E and P blocks need only a point prediction"
        for k in GAUSS_ONLY:
            assert k not in arm, (
                f"{key}: `{k}` is a property of a forecast DISTRIBUTION and this track predicted a "
                f"point. §4.2 — absent keys, never nan.")


def test_no_absent_key_is_smuggled_in_as_a_nan(point_only, external_scores) -> None:
    """A nan is skipped by `macro_mean`'s finiteness filter, so it reads exactly like a real score
    that happened to be undefined. Dropping an arm must never produce one.

    The reference is the SAME predictions scored with their spread: the E-block emits real nans of
    its own (quirk 8 — an annotation set that selects zero bins on this fixture's chromosome), and
    those are a property of the annotations, not of the missing sigma. So the bar is that the
    point-only row introduces no nan the full row did not already have.
    """
    _root, got = point_only
    for key, arms in got["per_track"].items():
        for arm, row in arms.items():
            ref = _numeric(external_scores["per_track"][key][arm])
            for k, v in _numeric(row).items():
                assert np.isfinite(v) or not np.isfinite(ref[k]), f"{key}/{arm}/{k} is {v}"
    for k, v in _numeric(got["macro"]["pval"]).items():
        assert np.isfinite(v), f"macro/pval/{k} is {v}"


def test_a_point_only_macro_has_no_count_arm_at_all(point_only) -> None:
    _root, got = point_only
    assert got["macro"]["count"] == {}
    assert got["macro"]["pval"]["n_tracks"] == 2


def test_without_a_peak_score_the_row_is_a_coverage_ranking(point_only) -> None:
    """B3 — no rival has a peak head, so `auprc` ranks by the predicted LEVEL and says so by
    withholding `bernoulli_nll`, exactly as the harness's own fallback does."""
    _root, got = point_only
    for arms in got["per_track"].values():
        assert np.isfinite(float(arms["pval"]["auprc"]))
        assert "bernoulli_nll" not in arms["pval"]
    assert got["provenance"]["point_only_tracks"] == got["tracks"]


def test_a_count_only_track_has_no_pval_arm(regime_file, recs, tmp_path) -> None:
    root = write_root(tmp_path / "countonly", recs, keep=("mu", "n"))
    src = _source(regime_file)
    try:
        got = score_external(src, root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    for arms in got["per_track"].values():
        assert set(arms) == {"count"}
        assert np.isfinite(float(arms["count"]["nb_nll"]))
    assert got["panel"] == {}, "the panel measure binarises a p-value; a count arm has none"


def test_a_track_that_predicts_nothing_recognised_is_refused(regime_file, recs, tmp_path) -> None:
    def junk(dirname, chrom, payload):
        return {"whatever": payload["mu"]}

    root = write_root(tmp_path / "junk", recs, mangle=junk)
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match="none of"):
            score_external(src, root, c_index_pairs=C_PAIRS)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# 5 — the sigma table (B1a, §6.1)
# ---------------------------------------------------------------------------

def test_the_sigma_table_gives_a_point_only_track_its_distributional_keys(regime_file, point_only,
                                                                         tmp_path) -> None:
    root, without = point_only
    table = {"method": "candi-roundtrip", "fitted_on": "regime.eic_val eval_pairs",
             "sigma": {TARGET_ASSAY: 0.75}}
    p = tmp_path / "sigma.json"
    p.write_text(json.dumps(table), encoding="utf-8")

    src = _source(regime_file)
    try:
        got = score_external(src, root, seed=0, c_index_pairs=C_PAIRS,
                             sigma_table=read_sigma_table(p), sigma_table_path=str(p))
    finally:
        src.close()
    for key, arms in got["per_track"].items():
        for k in GAUSS_ONLY:
            assert k in arms["pval"], f"{key}: the table supplies the spread `{k}` needs"
            assert np.isfinite(float(arms["pval"][k]))
        # The point blocks are untouched — a sigma decides the spread, never the mean.
        for k in E_KEYS:
            assert arms["pval"][k] == pytest.approx(float(without["per_track"][key]["pval"][k]))
        assert got["provenance"]["sigma_source"][key] == "sigma_table"
    assert got["provenance"]["point_only_tracks"] == []
    assert got["provenance"]["sigma_table"]["fitted_on"] == "regime.eic_val eval_pairs"


def test_a_tracks_own_sigma_beats_the_table(regime_file, full_root, tmp_path) -> None:
    """§4.2 fills a constant sigma only when the track has none; a heteroscedastic producer (§5.2)
    must not have its per-bin spread overwritten by a pooled constant."""
    p = tmp_path / "sigma.json"
    p.write_text(json.dumps({"method": "x", "fitted_on": "y", "sigma": {TARGET_ASSAY: 99.0}}),
                 encoding="utf-8")
    src = _source(regime_file)
    try:
        got = score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS,
                             sigma_table=read_sigma_table(p), sigma_table_path=str(p))
    finally:
        src.close()
    assert set(got["provenance"]["sigma_source"].values()) == {"track"}


def test_a_sigma_table_without_a_positive_sigma_is_refused(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"method": "x", "sigma": {TARGET_ASSAY: 0.0}}), encoding="utf-8")
    with pytest.raises(ExternalError, match="positive"):
        read_sigma_table(p)
    p.write_text(json.dumps({"method": "x"}), encoding="utf-8")
    with pytest.raises(ExternalError, match="sigma"):
        read_sigma_table(p)


# ---------------------------------------------------------------------------
# 6 — the CLI, driven exactly as a shell drives it
# ---------------------------------------------------------------------------

def test_the_cli_writes_the_same_result_it_computes(regime_file, full_root, external_scores,
                                                    tmp_path) -> None:
    out = tmp_path / "scores.json"
    rc = main(["--store", str(regime_file), "--pred", str(full_root), "--out", str(out),
               "--chroms", "chr2", "--c-index-pairs", str(C_PAIRS), "--quiet"])
    assert rc == 0
    got = json.loads(out.read_text())
    assert got["provenance"]["method"] == MANIFEST["method"]
    assert got["tracks"] == external_scores["tracks"]
    for key in got["tracks"]:
        assert got["per_track"][key]["pval"]["mse"] == pytest.approx(
            float(external_scores["per_track"][key]["pval"]["mse"]), rel=1e-6, abs=1e-6)


def test_the_harness_gates_are_inert_on_the_model_path(recs) -> None:
    """Every record `stream_tracks` builds has both a count prediction and a spread, so the two
    properties `bench.external` added to `TrackRecord` cannot change a model-path number."""
    for rec in recs:
        assert rec.has_count is True
        assert rec.has_sigma is rec.has_pval is True


# ---------------------------------------------------------------------------
# 7 — t56: the sampled CRPS, end to end through this entry
# ---------------------------------------------------------------------------
# The unit-level validation of the estimator lives in `tests/test_crps_sampled.py`. What is pinned
# HERE is the integration: the flag is off by default, on it reaches only the count arm's CRPS
# family, and a score file it produced says so.

def test_the_default_run_stamps_no_crps_estimator(external_scores) -> None:
    """OFF BY DEFAULT, and silent when off. The presence of the key is the flag, so a default run
    must not carry it — otherwise every pre-t56 score file reads as a different kind of file."""
    prov = external_scores["provenance"]
    for key in ("crps_estimator", "crps_k", "crps_seed"):
        assert key not in prov, key


def test_the_sampled_run_stamps_the_estimator_k_and_seed(regime_file, full_root) -> None:
    src = _source(regime_file)
    try:
        got = score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS,
                             crps_approx=64, crps_seed=5)
    finally:
        src.close()
    prov = got["provenance"]
    assert prov["crps_estimator"] == "fair_sampled"
    assert prov["crps_k"] == 64 and prov["crps_seed"] == 5


def test_sampling_reproduces_the_exact_crps_and_leaves_every_other_key_alone(
        regime_file, full_root, external_scores) -> None:
    """The whole claim of t56, on a real store: swap the instrument, keep the number.

    `c_star`/`n_star_log2` are the oracle ARGMIN, so they may land one grid step away — they are
    listed with the moved keys rather than asserted equal. Everything else in the count arm, and
    the entire pval arm, must be untouched: no other measure ever called `nb_crps`.
    """
    src = _source(regime_file)
    try:
        got = score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS,
                             crps_approx=512, crps_seed=0)
    finally:
        src.close()
    moved = {"crps", "crps_oracle_scaled", "crps_oracle_scaled_and_n", "scale_error",
             "c_star", "n_star_log2"}
    for key in external_scores["tracks"]:
        exact, sampled = (external_scores["per_track"][key], got["per_track"][key])
        pa, pb = _numeric(exact["pval"]), _numeric(sampled["pval"])
        assert set(pa) == set(pb)
        for k in sorted(pa):
            assert pb[k] == pa[k] or (np.isnan(pa[k]) and np.isnan(pb[k])), f"{key}/pval/{k}"
        a, b = _numeric(exact["count"]), _numeric(sampled["count"])
        assert set(a) == set(b)
        for k in sorted(set(a) - moved):
            assert b[k] == a[k] or (np.isnan(a[k]) and np.isnan(b[k])), f"{key}/count/{k}"
        for k in ("crps", "crps_oracle_scaled", "crps_oracle_scaled_and_n"):
            assert b[k] == pytest.approx(a[k], rel=0.02), f"{key}/count/{k}"
        assert sampled["count"]["beats_marginal"] == exact["count"]["beats_marginal"], key


# ---------------------------------------------------------------------------
# 8 — t89: a rival selects on the SAME positions CANDI does
# ---------------------------------------------------------------------------
# §5's whole point is that every method selects on one number. A cheap selection scope that only
# `candi.monitor` could reach would give CANDI the fast path and leave the rivals on the slow one,
# and the two numbers would stop being the same measurement. The scope therefore lives on the
# EvalSource, which both entry points already take, and these tests are what says the external path
# honours it bin for bin rather than merely accepting the flag.

@pytest.fixture(scope="module")
def ext_scope_bed(tmp_path_factory) -> Path:
    """Part of the eval chromosome, edges off the 25 bp bin grid, as the hg38 Pilot Regions are."""
    p = tmp_path_factory.mktemp("extscope") / "scope.bed"
    p.write_text("chr2\t3210\t11190\tR0\n", encoding="utf-8")
    return p


def test_a_rival_scored_under_a_scope_is_scored_on_exactly_the_scoped_bins(
        regime_file, full_root, ext_scope_bed, recs) -> None:
    """THE UNIFORMITY PROPERTY. The rival still hands over FULL-LENGTH arrays — §4.1's length
    assertion is what makes bin `i` the bin at `i * 25` bp — and the cut happens on our side, with
    the same index the model path uses. Compared against the same records compacted by hand, so
    nothing here can pass by both paths making the same mistake about which bins those are.
    """
    from candi.bench import annotations as ann

    src = _source(regime_file, eval_regions=ext_scope_bed)
    try:
        got = score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS)
        idx = {c: src.scored_bins(c) for c in src.eval_chroms}
        want = {}
        for rec in recs:
            cut = H.TrackRecord(pair=rec.pair, assay=rec.assay, kind=rec.kind,
                                chroms=rec.chroms, has_peak_head=rec.has_peak_head,
                                bin_scope="regions")
            for name in ("mu", "n", "counts", "signal_mu", "signal_sigma", "pval",
                         "peak_score", "peaks"):
                src_d, dst_d = getattr(rec, name), getattr(cut, name)
                for c in rec.chroms:
                    if c in src_d:
                        dst_d[c] = np.asarray(src_d[c])[idx[c]]
            want[rec.key] = H.score_track(
                cut, gene_annotations=ann.gene_annotations(),
                enh_annotations=ann.enhancer_annotations(), seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()

    assert set(got["per_track"]) == set(want)
    for key in want:
        for arm in ("count", "pval"):
            a, b = _numeric(want[key][arm]), _numeric(got["per_track"][key][arm])
            assert set(a) == set(b), f"{key}/{arm} key sets differ"
            for k in sorted(a):
                if np.isfinite(a[k]) or np.isfinite(b[k]):
                    assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"{key}/{arm}/{k}"


def test_a_rivals_score_file_says_which_positions_it_was_measured_over(
        regime_file, full_root, ext_scope_bed, external_scores) -> None:
    """A leaderboard that mixed a scoped row with a full-coverage one would rank two exams. The
    scope block is the same shape on both paths, so one reader can tell them apart."""
    src = _source(regime_file, eval_regions=ext_scope_bed)
    try:
        scoped = score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    assert external_scores["provenance"]["eval_scope"]["name"] == "full"
    sc = scoped["provenance"]["eval_scope"]
    assert sc["name"] == "regions" and 0.0 < sc["fraction"] < 1.0
    assert sc["bed"] == str(ext_scope_bed)
    for key, arms in scoped["per_track"].items():
        assert arms["count"]["bin_scope"] == "regions", key


def test_the_scope_carries_the_key_the_rivals_actually_select_on(regime_file, recs,
                                                                 ext_scope_bed,
                                                                 tmp_path_factory) -> None:
    """`pval:mse`, on a POINT-ONLY track, under a scope.

    The selection key is NOT uniform across methods (PI ruling, 2026-09-01): CANDI selects on
    count-arm `crps`, and Avocado / eDICE / Lavawizard select on `pval:mse`, because no rival has a
    count head and the pval CRPS would need a σ-table that Rule 1 forbids fitting on `V_`. A scope
    that only kept CANDI's key working would give the rivals a cheap eval that cannot select.

    It holds for a structural reason worth stating: the scope lives on `EvalSource` and cuts
    POSITIONS, while a selection key is a function of the two vectors at those positions. The keys
    the scope withholds are exactly the ones that read a genomic coordinate out of the array, and
    neither `count:crps` nor `pval:mse` is one of them.
    """
    root = write_root(tmp_path_factory.mktemp("extpointscope"), recs, keep=("signal_mu",))
    src = open_source(store=regime_file, eval_regions=ext_scope_bed)
    try:
        got = score_external(src, root, seed=0, c_index_pairs=C_PAIRS)
        idx = {c: src.scored_bins(c) for c in src.eval_chroms}
    finally:
        src.close()

    assert "count" not in got["macro"] or not got["macro"]["count"], "a point track has no count arm"
    sel = got["macro"]["pval"]["mse"]
    assert np.isfinite(sel), "the rivals' selection key is not finite under the scope"

    # and it is the number a hand-compacted rescore gives, not merely a finite one
    per = [got["per_track"][k]["pval"]["mse"] for k in sorted(got["per_track"])]
    want = []
    for rec in sorted(recs, key=lambda r: r.key):
        y = np.concatenate([np.asarray(rec.pval[c])[idx[c]] for c in rec.chroms])
        p = np.concatenate([np.asarray(rec.signal_mu[c])[idx[c]] for c in rec.chroms])
        want.append(float(((y - p) ** 2).mean()))
    assert per == pytest.approx(want, rel=1e-6, abs=1e-9)
    assert sel == pytest.approx(float(np.mean(want)), rel=1e-6, abs=1e-9)
    # the scope is on the row for a rival exactly as it is for a checkpoint
    for arms in got["per_track"].values():
        assert arms["pval"]["bin_scope"] == "regions"


def test_a_rival_opens_the_shipped_selection_scope_and_its_hash_proves_which_one(
        regime_file, full_root, recs, tmp_path) -> None:
    """The COMMITTED scope, opened by the external entry, on a point-only rival root.

    All four rivals now select on `V_` through this path, and §5 only holds if every method scores
    the same POSITIONS. `provenance.eval_scope.sha256` is what proves they did — a method that
    quietly opened a different BED produces a different hash, and a leaderboard can refuse it.
    That is why this asserts the hash of the shipped file rather than merely that a scope exists.

    The shipped BED names chr20/21/22 and the test store names chr1/chr2, so the BED is REBUILT
    here for this store from the same generator, same seed, same rule. What is pinned is the
    mechanism and the hash contract, not a coordinate that no test corpus could carry.
    """
    import hashlib
    import numpy as np

    import tools.make_eval_scope_bed as G
    from candi.bench.harness import full_tiling
    from tests.test_store_reader import N_BINS

    src0 = open_source(store=regime_file)
    try:
        nb = {c: src0.n_bins(c) for c in src0.eval_chroms}
        ctx = src0.context_bins
    finally:
        src0.close()
    n_win = max(2, len(full_tiling(nb[list(nb)[0]], ctx)) // 4)
    bed = tmp_path / "eval_random_seedtest.bed"
    bed.write_text(G.build(nb, windows=n_win, seed=890217, context_bins=ctx, resolution=25),
                   encoding="utf-8")
    want_sha = hashlib.sha256(bed.read_bytes()).hexdigest()

    root = write_root(tmp_path / "pointrival", recs, keep=("signal_mu",))
    src = open_source(store=regime_file, eval_regions=bed)
    try:
        got = score_external(src, root, seed=0, c_index_pairs=C_PAIRS)
        idx = {c: src.scored_bins(c) for c in src.eval_chroms}
    finally:
        src.close()

    sc = got["provenance"]["eval_scope"]
    assert sc["name"] == "regions"
    assert sc["sha256"] == want_sha, "the hash a leaderboard would check does not name this BED"
    assert sc["n_regions"] == n_win and sc["scored_bins"] == n_win * ctx

    # the rival's own selection key, over exactly those bins
    sel = got["macro"]["pval"]["mse"]
    want = []
    for rec in sorted(recs, key=lambda r: r.key):
        y = np.concatenate([np.asarray(rec.pval[c])[idx[c]] for c in rec.chroms])
        p = np.concatenate([np.asarray(rec.signal_mu[c])[idx[c]] for c in rec.chroms])
        want.append(float(((y - p) ** 2).mean()))
    assert sel == pytest.approx(float(np.mean(want)), rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# 9 — the three panel numbers, on a rival's row (plan/BENCHMARK_DESIGN.md §5.2)
# ---------------------------------------------------------------------------

def test_a_rivals_result_carries_the_same_three_panels_run_bench_does(external_scores,
                                                                      model_scores) -> None:
    """`panels` is not a leaderboard-side derivation. §5.2's three numbers are aggregations of the
    per-track rows, so they are computed where the rows are, on both paths, by the same function —
    otherwise a board would re-derive them from a macro that has already lost the panel labels."""
    for arm in ("count", "pval"):
        assert set(external_scores["panels"][arm]) == {"V_breadth", "V_matched", "B"}
        mine, theirs = model_scores["panels"][arm], external_scores["panels"][arm]
        for name in ("V_breadth", "V_matched", "B"):
            a, b = _numeric(mine[name]), _numeric(theirs[name])
            assert set(a) == set(b), f"panels/{arm}/{name}"
            for k in sorted(a):
                if np.isfinite(a[k]) or np.isfinite(b[k]):
                    assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"{arm}/{name}/{k}"

    pv = external_scores["panels"]["pval"]
    # Both target cells are `V_`, so the breadth panel is the whole run and `B` is empty. What is
    # pinned is that the middle number EXISTS and says it is not ranked — a board that ever read a
    # V_->B_ delta straight off `V_breadth` would be reading two different exams.
    assert pv["V_breadth"]["n_experiments"] == 2 and pv["V_breadth"]["ranked"] is True
    assert pv["B"]["n_experiments"] == 0 and pv["B"]["ranked"] is True
    assert pv["V_matched"]["ranked"] is False and pv["V_matched"]["matched_to"] == []


# ---------------------------------------------------------------------------
# 10 — §4 — one pass, two aggregations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_chrom_regime(store, tmp_path_factory) -> Path:
    """The same two declared pairs over BOTH chromosomes, so a scope can be split in two.

    `train_chroms` is empty on purpose: a regime refuses to share a chromosome between the two
    splits, and what is under test here is the aggregation, not the training plan.
    """
    d = tmp_path_factory.mktemp("exttwochrom")
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]],
                      train_chroms=[], eval_chroms=["chr1", "chr2"])
    p = d / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def two_chrom_root(tmp_path_factory) -> Path:
    """A point-only rival root over both chromosomes. Written by hand rather than streamed: what
    §4 claims is about the AGGREGATION, and a model pass would only make the fixture slower."""
    root = tmp_path_factory.mktemp("exttwochrompred") / "pred"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    rng = np.random.default_rng(4)
    for pair in PAIRS:
        d = root / track_dirname(pair, TARGET_ASSAY)
        d.mkdir()
        for c, n in N_BINS.items():
            np.savez(d / f"{c}.npz", signal_mu=rng.gamma(1.0, 2.0, n).astype(np.float32))
    return root


def test_held_out_chroms_yields_the_genome_wide_block_a_whole_pass_would_have(
        two_chrom_regime, two_chrom_root) -> None:
    """The headline of §4: ONE pass, two aggregations, and the genome-wide half is the number a
    separate all-chromosome run would have produced — not a rescaling of the held-out one."""
    src = _source(two_chrom_regime)
    try:
        whole = score_external(src, two_chrom_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    src = _source(two_chrom_regime)
    try:
        split = score_external(src, two_chrom_root, seed=0, c_index_pairs=C_PAIRS,
                               held_out_chroms=["chr2"])
    finally:
        src.close()

    assert "genome_wide" not in whole, "one scope, so §4's blanking rule leaves no block"
    assert set(split["genome_wide"]) == {"chroms", "per_track", "macro", "panels", "note"}
    assert split["genome_wide"]["chroms"] == ["chr1", "chr2"]
    for arm in ("count", "pval"):
        assert split["genome_wide"]["macro"][arm] == whole["macro"][arm], arm
        assert split["genome_wide"]["panels"][arm] == whole["panels"][arm], arm
    # and the ranked half really is the narrower scope, not a copy of the same numbers
    assert split["macro"]["pval"]["mse"] != whole["macro"]["pval"]["mse"]
    for key, arms in split["per_track"].items():
        assert arms["pval"]["chroms"] == ["chr2"], key
        assert split["genome_wide"]["per_track"][key]["pval"]["chroms"] == ["chr1", "chr2"]


def test_the_scope_block_says_which_half_is_ranked(two_chrom_regime, two_chrom_root,
                                                   external_scores) -> None:
    src = _source(two_chrom_regime)
    try:
        split = score_external(src, two_chrom_root, seed=0, c_index_pairs=C_PAIRS,
                               held_out_chroms=["chr2"])
    finally:
        src.close()
    sc = split["provenance"]["scope"]
    assert sc["ranked"] == H.SCOPE_HELD_OUT
    assert sc["held_out_chroms"] == ["chr2"] and sc["scored_chroms"] == ["chr1", "chr2"]
    assert sc["genome_wide_computed"] is True
    # a run that was given no split says so in the same key rather than by an absent one
    assert external_scores["provenance"]["scope"]["genome_wide_computed"] is False


def test_a_held_out_chromosome_the_run_never_scored_is_refused(two_chrom_regime,
                                                               two_chrom_root) -> None:
    src = _source(two_chrom_regime)
    try:
        with pytest.raises(ExternalError, match="chr9"):
            score_external(src, two_chrom_root, c_index_pairs=C_PAIRS,
                           held_out_chroms=["chr2", "chr9"])
    finally:
        src.close()


def test_a_split_scope_and_an_eval_region_scope_cannot_be_combined(two_chrom_regime,
                                                                   two_chrom_root,
                                                                   ext_scope_bed) -> None:
    """`genome_wide` means every bin of every scored chromosome. Under a region scope it would be a
    region cut carrying that name, which is the one thing §4 exists to keep apart."""
    src = _source(two_chrom_regime, eval_regions=ext_scope_bed)
    try:
        with pytest.raises(ExternalError, match="genome_wide"):
            score_external(src, two_chrom_root, c_index_pairs=C_PAIRS, held_out_chroms=["chr2"])
    finally:
        src.close()


# ---------------------------------------------------------------------------
# 11 — --truth-root: somebody else's truth, our instrument
# ---------------------------------------------------------------------------

TRUTH_MANIFEST = {
    "kind": "truth",
    "truth": "challenge",
    "source_dir": "tests/test_bench_external.py",
    "bridge_sha256": None,
    "chroms": ["chr2"],
    "bin_rule": "mean of each 25 bp bin, NaN->0, floor(chr_len/25), 0-anchored",
    "generated_by": "tests/test_bench_external.py",
    "date": "2026-09-01",
}


def write_truth_root(root: Path, recs, *, manifest=None, drop=()) -> Path:
    """A §4.1-layout TRUTH root holding the store's own pval layer as `signal_mu`.

    Built from the very truth the store path scores against, so a difference between the two runs
    can only be the code — never the data.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(TRUTH_MANIFEST if manifest is None else manifest), encoding="utf-8")
    for rec in recs:
        name = track_dirname(rec.pair, rec.assay)
        if name in drop:
            continue
        d = root / name
        d.mkdir(exist_ok=True)
        for c in rec.chroms:
            np.savez(d / f"{c}.npz", signal_mu=np.asarray(rec.pval[c], dtype=np.float32))
    return root


@pytest.fixture(scope="module")
def challenge_scores(regime_file, full_root, recs, tmp_path_factory):
    troot = write_truth_root(tmp_path_factory.mktemp("exttruth") / "truth", recs)
    src = _source(regime_file)
    try:
        return troot, score_external(src, full_root, seed=0, c_index_pairs=C_PAIRS,
                                     truth_root=troot)
    finally:
        src.close()


def test_the_same_truth_from_a_root_gives_the_same_pval_numbers(challenge_scores,
                                                                external_scores) -> None:
    """THE ACCEPTANCE GATE FOR `--truth-root`. The root holds the store's OWN pval layer, so every
    key that survives the swap must be the store path's number to the last bit. Anything that
    moved would be the truth reader, not the truth."""
    _troot, got = challenge_scores
    assert got["tracks"] == external_scores["tracks"]
    for key in got["tracks"]:
        mine = external_scores["per_track"][key]["pval"]
        theirs = got["per_track"][key]["pval"]
        assert set(theirs) <= set(mine), f"{key}: the challenge row invented a key"
        for k in sorted(theirs):
            assert _same(mine[k], theirs[k]), f"{key}/{k}"


def test_challenge_truth_withholds_exactly_the_keys_that_read_a_peak_call(
        challenge_scores, external_scores) -> None:
    """The 2019 challenge distributed signal tracks and no peak calls, so the count arm and every
    peak-derived key are ABSENT — not nan, and not computed against a stand-in. `peak_base_rate`
    is the one that matters most: a finite 0.0 there is a number `macro_mean` would average.

    `nb_nll` goes too, and by a different route: the loss tier is arm-independent and is spread
    into BOTH arms, so the count likelihood rides on the pval row. With no count truth it is
    withheld by `has_count`, not by the peak rule — hence it is named here and not in the list."""
    _troot, got = challenge_scores
    for key, arms in got["per_track"].items():
        assert set(arms) == {"pval"}, f"{key}: challenge truth carries no counts"
        gone = set(external_scores["per_track"][key]["pval"]) - set(arms["pval"])
        assert gone == (set(WITHHELD_WITHOUT_PEAK_TRUTH) | {"nb_nll"}) & set(
            external_scores["per_track"][key]["pval"]), key
        assert "peak_overlap_0.01" in arms["pval"], (
            "peak_overlap ranks the TRUTH SIGNAL by the prediction and reads no peak call")
    assert got["macro"]["count"] == {}
    assert got["panel"] == {}, "panel_specificity binarises against MACS2 calls, which are absent"
    assert got["panels"]["pval"]["V_breadth"]["n_experiments"] == 2


def test_a_score_file_names_the_truth_it_was_measured_against(challenge_scores,
                                                              external_scores) -> None:
    import hashlib
    troot, got = challenge_scores
    want = hashlib.sha256((troot / "manifest.json").read_bytes()).hexdigest()
    assert got["provenance"]["truth"] == {"source": "challenge", "root": str(troot),
                                          "manifest_sha256": want}
    assert external_scores["provenance"]["truth"] == {"source": "store"}


def test_a_prediction_root_passed_as_the_truth_is_refused(regime_file, full_root) -> None:
    """Scoring a method against its own output is a perfect row and a meaningless one, so the
    truth manifest carries `kind: "truth"` and nothing else is accepted."""
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match="kind"):
            score_external(src, full_root, c_index_pairs=C_PAIRS, truth_root=full_root)
    finally:
        src.close()


def test_a_track_the_truth_root_does_not_cover_is_a_hole_like_any_other(regime_file, full_root,
                                                                        recs, tmp_path) -> None:
    gone = track_dirname(PAIRS[1], TARGET_ASSAY)
    troot = write_truth_root(tmp_path / "partialtruth", recs, drop=(gone,))
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=gone):
            score_external(src, full_root, c_index_pairs=C_PAIRS, truth_root=troot)
    finally:
        src.close()
    src = _source(regime_file)
    try:
        got = score_external(src, full_root, c_index_pairs=C_PAIRS, truth_root=troot,
                             allow_missing=True)
    finally:
        src.close()
    assert got["provenance"]["missing_tracks"] == [gone]
    assert got["tracks"] == [track_key(PAIRS[0], TARGET_ASSAY, "impute")]


def test_a_count_only_prediction_has_no_truth_under_a_challenge_root(regime_file, recs,
                                                                     tmp_path) -> None:
    root = write_root(tmp_path / "counttruth", recs, keep=("mu", "n"))
    troot = write_truth_root(tmp_path / "counttruth_t", recs)
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match="SIGNAL track only"):
            score_external(src, root, c_index_pairs=C_PAIRS, truth_root=troot)
    finally:
        src.close()


def test_a_truth_root_array_off_the_grid_is_refused_by_name(regime_file, full_root, recs,
                                                            tmp_path) -> None:
    troot = write_truth_root(tmp_path / "shorttruth", recs)
    victim = track_dirname(recs[0].pair, recs[0].assay)
    short = np.asarray(recs[0].pval["chr2"], dtype=np.float32)[:-1]
    np.savez(troot / victim / "chr2.npz", signal_mu=short)
    src = _source(regime_file)
    try:
        with pytest.raises(ExternalError, match=victim):
            score_external(src, full_root, c_index_pairs=C_PAIRS, truth_root=troot)
    finally:
        src.close()


def test_the_cli_takes_both_new_flags(regime_file, full_root, challenge_scores,
                                      tmp_path) -> None:
    troot, direct = challenge_scores
    out = tmp_path / "challenge.json"
    rc = main(["--store", str(regime_file), "--pred", str(full_root), "--out", str(out),
               "--truth-root", str(troot), "--held-out-chroms", "chr2",
               "--c-index-pairs", str(C_PAIRS), "--quiet"])
    assert rc == 0
    got = json.loads(out.read_text())
    assert got["provenance"]["truth"]["source"] == "challenge"
    # chr2 is every scored chromosome here, so §4's blanking rule leaves no genome-wide block
    assert "genome_wide" not in got
    assert got["provenance"]["scope"]["genome_wide_computed"] is False
    for key in got["tracks"]:
        assert got["per_track"][key]["pval"]["mse"] == pytest.approx(
            float(direct["per_track"][key]["pval"]["mse"]), rel=1e-9, abs=1e-12)
