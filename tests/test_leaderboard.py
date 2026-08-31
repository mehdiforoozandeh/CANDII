"""The rivals leaderboard compiler, pinned (plan/BENCHMARK_DESIGN.md, t58 -> t82).

Everything here runs on the synthetic fixtures in `tests/fixtures/leaderboard/` — four fake
methods (`fixture-a` … `fixture-d`) with hand-chosen numbers, so every rank and every tie below
is arithmetic a reader can redo on paper. Nothing depends on `cruxvault/results/`, which is
untracked and per-machine by design.

Two vocabularies the fixtures use throughout. `BOARD` is a regime id — the retired ids
`main` / `dev` / `entrants` are gone (§9), and so is the `protocol` field that carried
P1 / P2 / P3. `VIEW` is one address: truth, panel and scope joined by dots (§1). The `root`
fixture also flips `noise_floor.measured` on, because the committed regimes have no measured
noise floor and so rank nothing (§15) — the ranking tests below need a board that ranks.

The fixture geometry, once: `fixture-a` is best on every composite metric, `b` second, `c` third,
`d` worst. On the count arm `a` and `b` sit 0.05 apart — under the 0.09 macro-CRPS floor, so they
tie. `fixture-d` is pval-only (no count arm, no peak metrics), which is what exercises the
partial-coverage rule: d is missing the peaks category, so it gets no composite and is left out of
the headline ranking, while still ranking inside pointwise and distributional. Peaks stays in the
composite for a/b/c. The count sub-board ranks three rows, not four.
"""
from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "leaderboard"
BOARD = "eic.19"
VIEW = "store.V_breadth.held-out"
ADDRESS = ["--truth", "store", "--panel", "V_breadth", "--scope", "held-out"]

_spec = importlib.util.spec_from_file_location("leaderboard_tool", REPO / "tools" / "leaderboard.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


# ---------------------------------------------------------------- fixtures ---

@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A leaderboard root: the real registry + the real boards with fixture-frozen hashes."""
    root = tmp_path / "leaderboard"
    (root / "rows").mkdir(parents=True)
    root.joinpath("registry.json").write_text(
        (REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for b in list(boards["boards"].values()) + [boards["anchor"]]:
        b["frozen"]["store_manifest_hash"] = "fixhash-store"
        b["frozen"]["regime_sha256"] = "fixhash-regime"
    for b in boards["boards"].values():
        b["noise_floor"] = {"measured": True, "note": "fixture floor"}
        # the fixture score jsons name the regime the pre-t79 tree had; the gate compares
        # basenames, so point the board at it rather than editing four fixtures
        b["eval_set"]["regime"] = "configs/regime.eic_val.json"
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")
    return root


def add(root: Path, score: str, method: str, *extra: str) -> None:
    lb.main(["--root", str(root), "add", str(FIX / score), *ADDRESS,
             "--board", BOARD, "--method", method, "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", *extra])


def add_all(root: Path) -> None:
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    add(root, "score_fixture_c.json", "fixture-c", "--allow-missing")
    add(root, "score_fixture_d.json", "fixture-d", "--allow-missing")


def board_view(root: Path) -> dict:
    return lb.compile_leaderboard(root)["boards"][BOARD]["views"][VIEW]


def row_path(root: Path, name: str) -> Path:
    return root / "rows" / BOARD / VIEW / f"{name}.json"


# ---------------------------------------------------------------- schema ---

def test_registry_and_boards_round_trip() -> None:
    """The committed registry and boards load through their own gates."""
    reg = lb.load_registry(REPO / "leaderboard")
    boards = lb.load_boards(REPO / "leaderboard")
    ids = {lb.metric_id(m) for m in reg["metrics"]}
    for required in ("pval/mse", "pval/gwcorr", "pval/gwspear", "pval/mse1obs", "pval/crps",
                     "pval/pit_ks", "pval/coverage_95", "pval/auprc", "pval/gaussian_nll",
                     "count/crps", "count/crps_oracle_scaled", "count/scale_error",
                     "count/nb_nll"):
        assert required in ids, f"registry lost {required}"
    count_crps = next(m for m in reg["metrics"] if lb.metric_id(m) == "count/crps")
    assert count_crps["floor"] == 0.09
    assert count_crps["companions"] == ["crps_oracle_scaled", "scale_error"]
    pval_crps = next(m for m in reg["metrics"] if lb.metric_id(m) == "pval/crps")
    assert pval_crps["floor"] is None, "the pval floor waits for t57; it must not be invented"
    assert pval_crps["companions"] == ["pit_ks", "coverage_95"]
    # the CANDI self-comparison bar is its own field — it is NOT a cross-method floor
    bar = reg["candi_self_comparison_bar"]
    assert bar["value"] == 0.1195 and bar["value"] != count_crps["floor"]
    # §9 — the two live regimes are the boards; `entrants` is no longer one of them.
    assert set(boards["boards"]) == {"eic.19", "eic.pilot"}
    assert lb.ANCHOR not in boards["boards"]
    for bid, b in boards["boards"].items():
        assert "protocol" not in b, f"{bid} still carries a retired P1/P2/P3 protocol"
        assert b["regime_id"].startswith(bid)
        assert set(b["truths"]) == {"store", "challenge"}
    # §1's vocabulary, and the views derived from it — never a hand-written list
    assert set(boards["truths"]) == {"store", "challenge"}
    assert set(boards["panels"]) == {"V_breadth", "V_matched", "B_"}
    assert set(boards["scopes"]) == {"held-out", "genome-wide"}
    assert boards["panels"]["V_matched"]["ranked"] is False
    assert boards["scopes"]["genome-wide"]["ranked"] is False
    assert set(boards["markers"]) == {"edice-embedding-substitution", "no-selection"}
    assert boards["method_markers"]["eDICE"] == ["edice-embedding-substitution"]
    assert "no-selection" in boards["method_markers"]["ChromImpute"]
    assert boards["truths"]["challenge"]["arms"] == ["pval"]
    assert len(view_keys := lb.view_keys(boards, "eic.19")) == 12
    assert lb.CANONICAL_VIEW in view_keys
    assert lb.view_keys(boards, lb.ANCHOR) == ["challenge.B_.held-out"]
    assert [d["id"] for d in boards["deferred_regimes"]][0] == "eic.gw→20,21,22"
    cov = reg["categories"]["covariate_diagnostics"]
    assert "absent_note" in cov and "candi.bench.external" in cov["absent_note"]
    assert "will_populate" in cov and "candi.bench" in cov["will_populate"]
    assert "C-block" in cov["will_populate"]


def test_committed_rows_all_pass_their_own_gates() -> None:
    """Every committed row loads through the reader's gates against the frozen boards.

    (Until M4 this asserted the dir was empty; real rows entered at M4, all via `add`.)"""
    reg = lb.load_registry(REPO / "leaderboard")
    boards = lb.load_boards(REPO / "leaderboard")
    for container in ("rows", lb.ANCHOR):
        base = REPO / "leaderboard" / container
        for p in base.glob("*/*.json") if container == lb.ANCHOR else base.glob("*/*/*.json"):
            row = json.loads(p.read_text(encoding="utf-8"))
            lb.gate_row_shape(row, reg)
            bid = lb.ANCHOR if container == lb.ANCHOR else p.parent.parent.name
            assert row["board"] == bid
            assert lb.gate_row_address(row, boards, bid, reg) == p.parent.name
            lb.gate_row_against_board(row, lb.board_spec(boards, bid), bid)
        stray = [q.name for q in base.rglob("*")
                 if q.is_file() and q.suffix != ".json"
                 and q.name not in (".gitkeep", "README.md")]
        assert stray == [], (container, stray)
    # §3.3 void rows are still shape-gated, so they cannot rot in the corner
    void = sorted((REPO / "leaderboard" / "void").glob("*/*.json"))
    assert len(void) == 37, "the retired rows are the record; do not delete them"
    for p in void:
        row = json.loads(p.read_text(encoding="utf-8"))
        lb.gate_row_shape(row, reg)
        assert row["board"] == p.parent.name


def test_add_round_trips_a_row(root: Path) -> None:
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["metrics"]["pval"]["mse"] == 1.0
    assert row["metrics"]["count"]["crps"] == 0.40
    assert row["badges"] == {"position": "position-generalizing",
                             "cell_types": "zero-shot cell types"}
    prov = row["provenance"]
    assert prov["scoring_sha"] == "deadbeef" and prov["fir_path"] == "fake:/scratch/fixture"
    assert prov["sigma_table"] is None
    assert prov["has_peak_head"] is False
    assert prov["flags"]["pval_pred_space"] == "-log10p"
    reg = lb.load_registry(root)
    lb.gate_row_shape(row, reg)  # what `add` wrote passes the reader's own gate


def test_add_extracts_the_sigma_table_id(root: Path) -> None:
    """B1a rows carry their σ-table id, mechanically, from the score json's provenance."""
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    row = json.loads(row_path(root, "fixture-b@v1").read_text(encoding="utf-8"))
    assert row["provenance"]["sigma_table"] == {"method": "fixture-b",
                                                "fitted_on": "regime.eic_val eval_pairs"}
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-b@v1")
    assert compiled["provenance"]["sigma_table"]["method"] == "fixture-b"
    assert compiled["has_peak_head"] is False


def test_add_extracts_has_peak_head_from_bernoulli_nll(root: Path, tmp_path: Path) -> None:
    """A real peak head is the presence of bernoulli_nll in the score macro, not a value range."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["macro"]["pval"]["bernoulli_nll"] = 0.42
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(score_path),
             *ADDRESS, "--board", BOARD, "--method", "fixture-a", "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["provenance"]["has_peak_head"] is True
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-a@v1")
    assert compiled["has_peak_head"] is True


def test_add_carries_contributor_mode_in_flags(root: Path, tmp_path: Path) -> None:
    """contributor_mode and clip are FLAG_KEYS members: add copies them and compile keeps them."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["contributor_mode"] = True
    score["provenance"]["clip"] = "p99"
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(score_path),
             *ADDRESS, "--board", BOARD, "--method", "fixture-a", "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["provenance"]["flags"]["contributor_mode"] is True
    assert row["provenance"]["flags"]["clip"] == "p99"
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-a@v1")
    assert compiled["provenance"]["flags"]["contributor_mode"] is True
    assert compiled["provenance"]["flags"]["clip"] == "p99"


# ---------------------------------------------------------------- refusals ---

def test_add_refuses_nan(root: Path) -> None:
    with pytest.raises(SystemExit, match="non-finite"):
        add(root, "score_nan.json", "fixture-nan", "--allow-missing")


def test_add_refuses_a_lone_count_crps(root: Path) -> None:
    """count `crps` never enters without `crps_oracle_scaled` + `scale_error`."""
    with pytest.raises(SystemExit, match="scale_error"):
        add(root, "score_bad_companion.json", "fixture-bad", "--allow-missing")


def test_add_refuses_missing_metrics_unless_declared(root: Path) -> None:
    with pytest.raises(SystemExit, match="allow-missing"):
        add(root, "score_fixture_d.json", "fixture-d")
    add(root, "score_fixture_d.json", "fixture-d", "--allow-missing")
    row = json.loads(row_path(root, "fixture-d@v1").read_text(encoding="utf-8"))
    assert "count/crps" in row["missing_metrics"] and "pval/auprc" in row["missing_metrics"]


def test_add_refuses_an_unfrozen_board(tmp_path: Path) -> None:
    """A board whose hashes are still TODO refuses every add (the pre-M4 state)."""
    root = tmp_path / "leaderboard"
    (root / "rows").mkdir(parents=True)
    root.joinpath("registry.json").write_text(
        (REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for b in list(boards["boards"].values()) + [boards["anchor"]]:
        b["frozen"]["store_manifest_hash"] = "TODO-freeze-at-stamping"
        b["frozen"]["regime_sha256"] = "TODO-freeze-at-stamping"
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")
    with pytest.raises(SystemExit, match="not frozen"):
        add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")


def test_committed_boards_refuse_every_add_until_they_are_frozen() -> None:
    """The two live regimes are deliberately NOT frozen, and that is the §3.3 enforcement.

    The old hashes digested a store manifest from before t78's DNase rebuild and a regime json
    from before t79's rewrite. Freezing them again is part of stamping the first retrained row;
    until then the TODO hash is what stops a row landing against the wrong data.
    """
    boards = lb.load_boards(REPO / "leaderboard")
    for bid, b in list(boards["boards"].items()) + [(lb.ANCHOR, boards["anchor"])]:
        for field in ("store_manifest_hash", "regime_sha256"):
            assert b["frozen"][field].startswith(lb.TODO_HASH), (bid, field)
        assert b["frozen"].get("frozen_note"), f"board {bid} must say what its hashes will digest"


def test_placement_mode_stamps_an_anchor_row(tmp_path: Path) -> None:
    """`add --placement-method` copies macro_all out of a t54-style placement file, and
    refuses a file that does not digest to the board's frozen regime hash."""
    placement = {
        "aggregation": "bootstrap mean; median within assay; mean over assay medians",
        "methods": {"fixture-entrant": {
            "n_experiments": 48,
            # per_assay NaN (prom_corr outside H3K4me3) must not block macro_all extraction
            "per_assay": {"H3K27me3": {"prom_corr": float("nan")}},
            "macro_all": {"n_assays": 7, "mse": 1.5, "gwcorr": 0.4, "gwspear": 0.3,
                          "mse1obs": 20.0}}},
    }
    pfile = tmp_path / "placement.json"
    pfile.write_text(json.dumps(placement), encoding="utf-8")
    root = tmp_path / "leaderboard"
    (root / "rows").mkdir(parents=True)
    root.joinpath("registry.json").write_text(
        (REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for b in list(boards["boards"].values()) + [boards["anchor"]]:
        b["frozen"]["store_manifest_hash"] = "fixhash-store"
        b["frozen"]["regime_sha256"] = "fixhash-regime"
    boards["anchor"]["frozen"]["regime_sha256"] = lb.sha256_file(pfile)
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")

    argv = ["--root", str(root), "add", str(pfile), "--board", lb.ANCHOR,
            "--truth", "challenge", "--panel", "B_", "--scope", "held-out",
            "--method", "fixture-entrant", "--version", "round2-2019",
            "--date", "2026-08-26", "--lineage", "entrant",
            "--position-class", "unrecorded", "--cell-class", "unrecorded",
            "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
            "--fir-path", "fake:/x", "--allow-missing",
            "--placement-method", "fixture-entrant"]
    lb.main(argv)
    row = json.loads((root / lb.ANCHOR / "challenge.B_.held-out"
                      / "fixture-entrant@round2-2019.json").read_text(encoding="utf-8"))
    # §6 — an anchor row carries no regime, because we never trained it
    assert row["address"] == {"regime": None, "truth": "challenge",
                              "panel": "B_", "scope": "held-out"}
    assert row["metrics"]["pval"] == {"mse": 1.5, "gwcorr": 0.4, "gwspear": 0.3,
                                      "mse1obs": 20.0}
    assert "count" not in row["metrics"]
    assert row["provenance"]["flags"]["placement_method"] == "fixture-entrant"
    # a tampered placement file no longer digests to the frozen hash and is refused
    pfile.write_text(json.dumps(placement).replace("1.5", "1.4"), encoding="utf-8")
    with pytest.raises(SystemExit, match="digests to"):
        lb.main(argv + ["--force"])


def test_add_refuses_a_wrong_store_hash(root: Path) -> None:
    with pytest.raises(SystemExit, match="does not match"):
        lb.main(["--root", str(root), "add", str(FIX / "score_fixture_a.json"),
                 *ADDRESS, "--board", BOARD, "--method", "fixture-a", "--version", "v1",
                 "--date", "2026-08-27", "--lineage", "baseline",
                 "--position-class", "generalizing", "--cell-class", "zero-shot",
                 "--scoring-sha", "deadbeef", "--store-manifest-hash", "some-other-store",
                 "--fir-path", "fake:/x", "--allow-missing"])


# ---------------------------------------------------------------- composite ---

def test_plain_ranks_and_the_missing_arm_rule(root: Path) -> None:
    add_all(root)
    view = board_view(root)
    # fixture-d has no auprc (peaks). Peaks stays in the composite for everyone who
    # covers it; d is incomplete — no composite, no headline rank (PI 2026-08-27).
    assert view["categories_in_composite"] == ["distributional", "peaks", "pointwise"]
    by_id = {r["id"]: r for r in view["rows"]}
    assert by_id["fixture-a@v1"]["rank"] == [1, 1]
    assert by_id["fixture-b@v1"]["rank"] == [2, 2]
    assert by_id["fixture-c@v1"]["rank"] == [3, 3]
    assert by_id["fixture-d@v1"]["rank"] is None
    assert by_id["fixture-d@v1"]["composite"] is None
    assert by_id["fixture-d@v1"]["partial_coverage"] is True
    assert by_id["fixture-d@v1"]["missing_composite_categories"] == ["peaks"]
    # d still ranks inside every category it covers
    assert by_id["fixture-d@v1"]["metric_ranks"]["pval/mse"] == [4, 4]
    assert by_id["fixture-d@v1"]["metric_ranks"]["pval/crps"] == [4, 4]
    assert "pval/auprc" not in by_id["fixture-d@v1"]["metric_ranks"]
    assert "peaks" not in by_id["fixture-d@v1"]["category_subscores"]
    assert "pointwise" in by_id["fixture-d@v1"]["category_subscores"]
    # a/b/c keep a composite that now includes peaks
    for rid in ("fixture-a@v1", "fixture-b@v1", "fixture-c@v1"):
        assert by_id[rid]["partial_coverage"] is False
        assert by_id[rid]["composite"] is not None
        assert "peaks" in by_id[rid]["category_subscores"]
    # the count arm is a sub-board, not a composite category, and d is simply not on it
    sub = view["sub_boards"]["count_arm"]
    assert [r["id"] for r in sub["rows"]] == ["fixture-a@v1", "fixture-b@v1", "fixture-c@v1"]
    # a and b sit 0.05 apart on count crps — under the 0.09 floor, so they share "1-2"
    by_id = {r["id"]: r["rank"] for r in sub["rows"]}
    assert by_id == {"fixture-a@v1": [1, 2], "fixture-b@v1": [1, 2], "fixture-c@v1": [3, 3]}


def test_partial_coverage_blanks_composite_without_poisoning_peers(root: Path, tmp_path: Path) -> None:
    """A method missing an entire composite category (distributional) gets a dash,
    not a zeroed composite, and does not drop that category for complete peers."""
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    score = json.loads((FIX / "score_fixture_c.json").read_text(encoding="utf-8"))
    for key in ("crps", "pit_ks", "coverage_95", "gaussian_nll"):
        score["macro"]["pval"].pop(key, None)
    path = tmp_path / "score_partial.json"
    path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(path),
             *ADDRESS, "--board", BOARD, "--method", "fixture-partial", "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    view = board_view(root)
    assert "distributional" in view["categories_in_composite"]
    by_id = {r["id"]: r for r in view["rows"]}
    partial = by_id["fixture-partial@v1"]
    assert partial["composite"] is None and partial["rank"] is None
    assert partial["partial_coverage"] is True
    assert "distributional" in partial["missing_composite_categories"]
    assert "pval/mse" in partial["metric_ranks"]  # still ranks in pointwise
    assert by_id["fixture-a@v1"]["composite"] is not None
    assert by_id["fixture-a@v1"]["rank"] == [1, 1]
    assert "distributional" in by_id["fixture-a@v1"]["category_subscores"]


def test_missing_count_arm_alone_does_not_blank_composite(root: Path, tmp_path: Path) -> None:
    """Count space is a sub-board, not a composite category (B1b). A pval-only row
    that fully covers pointwise + distributional + peaks still gets a composite."""
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    score = json.loads((FIX / "score_fixture_b.json").read_text(encoding="utf-8"))
    score["macro"]["count"] = {}
    path = tmp_path / "score_pval_only.json"
    path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(path),
             *ADDRESS, "--board", BOARD, "--method", "fixture-pval-only", "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    view = board_view(root)
    by_id = {r["id"]: r for r in view["rows"]}
    assert by_id["fixture-pval-only@v1"]["partial_coverage"] is False
    assert by_id["fixture-pval-only@v1"]["composite"] is not None
    assert by_id["fixture-pval-only@v1"]["rank"] is not None
    assert "count" not in by_id["fixture-pval-only@v1"]["metrics"]
    assert [r["id"] for r in view["sub_boards"]["count_arm"]["rows"]] == ["fixture-a@v1"]


def test_floor_ties_propagate_into_composite_spreads(root: Path) -> None:
    """PRD §5.4 — a floored metric ties into a rank interval, the interval into the
    category mean, the mean into the composite spread."""
    reg = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    for m in reg["metrics"]:
        if m["key"] == "crps" and m["arm"] == "pval":
            m["floor"] = 0.05  # a-b gap is 0.01: tied; every other gap clears it
    (root / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    add_all(root)
    rows = {r["id"]: r for r in board_view(root)["rows"]}
    assert rows["fixture-a@v1"]["metric_ranks"]["pval/crps"] == [1, 2]
    assert rows["fixture-b@v1"]["metric_ranks"]["pval/crps"] == [1, 2]
    # peaks now stays in the composite (d is incomplete, not a veto).
    # a: pointwise (1,1) + crps (1,2) + peaks (1,1) → (1.0, 4/3)
    # b: pointwise (2,2) + crps (1,2) + peaks (2,2) → (5/3, 2.0)
    assert rows["fixture-a@v1"]["composite"] == [1.0, 4 / 3]
    assert rows["fixture-b@v1"]["composite"] == [5 / 3, 2.0]
    assert rows["fixture-a@v1"]["rank"] == [1, 1]
    assert rows["fixture-b@v1"]["rank"] == [2, 2]
    assert rows["fixture-c@v1"]["rank"] == [3, 3]
    assert rows["fixture-d@v1"]["rank"] is None
    assert rows["fixture-d@v1"]["composite"] is None


def test_lineage_sub_board_takes_only_candi_diagnostics(root: Path) -> None:
    add_all(root)
    view = board_view(root)
    assert view["sub_boards"]["candi_lineage"]["rows"] == []  # no candi row, no lineage board
    cov = lb.compile_leaderboard(root)["covariate_coverage"]
    assert cov["n_rows_with_diagnostics"] == 0
    assert "candi" not in cov["lineages"]


def test_candi_diagnostics_surface_when_present(root: Path, tmp_path: Path) -> None:
    """A CANDI row whose score json carries a C block of scalars lands on the lineage board."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["suite"] = "candi.bench"
    score["C"] = {
        "covuse": 0.91, "covshare": 0.40, "depthdir": 0.80,
        "depthcounterfact": 0.50, "covspec": 0.70,
        "depthblind": 0.60, "biokeep": 0.55,
    }
    path = tmp_path / "score_candi.json"
    path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(path),
             *ADDRESS, "--board", BOARD, "--method", "CANDI", "--version", "v0",
             "--date", "2026-08-27", "--lineage", "candi",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    compiled = lb.compile_leaderboard(root)
    lin = compiled["boards"][BOARD]["views"][VIEW]["sub_boards"]["candi_lineage"]["rows"]
    assert len(lin) == 1 and lin[0]["id"] == "CANDI@v0"
    assert lin[0]["metrics"]["diagnostics"]["covuse"] == 0.91
    assert lin[0]["metrics"]["diagnostics"]["biokeep"] == 0.55
    assert compiled["covariate_coverage"]["n_rows_with_diagnostics"] == 1
    assert "candi.bench" in compiled["covariate_coverage"]["scorers"]


def test_nested_c_block_flattens_to_registry_scalars(root: Path, tmp_path: Path) -> None:
    """A real `harness.c_block` score (nested covuse, combined depthblind_biokeep) stamps.

    Headlines the instrument names: depthdir.monotone_frac, depthcounterfact.frac_min_at_true,
    covspec.mean_gap, depthblind_biokeep.bio_silhouette → biokeep. covuse, covshare, and
    depthblind have no code-defined panel scalar and stay off the row rather than being averaged.
    """
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["suite"] = "candi.bench"
    score["C"] = {
        "n_units": 4, "n_windows": 8, "n_contexts": 2,
        "covariates": ["depth", "assay_id", "read_length", "run_type"],
        "kind": "impute",
        "covuse": {
            "depth": {"crps_observed": 1.0, "marginal_p": 0.01, "marginal_mean_d_crps": 0.4,
                      "uses_covariate": True, "within_batch_d_crps": 0.0},
            "assay_id": {"crps_observed": 1.0, "marginal_p": 1.0, "marginal_mean_d_crps": 0.0,
                         "uses_covariate": False, "within_batch_d_crps": 0.0},
            "read_length": {"crps_observed": 1.0, "marginal_p": 1.0, "marginal_mean_d_crps": 0.0,
                            "uses_covariate": False, "within_batch_d_crps": 0.0},
            "run_type": {"crps_observed": 1.0, "marginal_p": 1.0, "marginal_mean_d_crps": 0.0,
                         "uses_covariate": False, "within_batch_d_crps": 0.0},
        },
        "covshare": {"depth": 0.85, "assay_id": 0.10, "read_length": 0.03, "run_type": 0.02,
                     "total_variance": 1.2, "total_variance_unclamped": 1.2},
        "depthdir": {"monotone_frac": 0.80, "mean_dose_response_corr": 0.99,
                     "mean_level_by_told_depth": [1.0, 0.5, 0.25, 0.125],
                     "n_flat_units": 0, "ladder": [0.0, -1.0, -2.0, -3.0]},
        "depthcounterfact": {"frac_min_at_true": 0.50, "frac_beats_told1": 0.75,
                             "n_levels": 4, "constant_answer_value": 0.25,
                             "crps_at_true_level": {"1.0": 0.1, "2.0": 0.2}},
        "covspec": {"mean_gap": 0.70,
                    "gap_by_aspect": {"level": 0.8, "shape": 0.6, "dispersion": 0.7, "tail": 0.7},
                    "owner_by_aspect": {"level": "depth"},
                    "matrix": [[1.0, 0.1, 0.1, 0.1]],
                    "covariates": ["depth"], "aspects": ["level", "shape", "dispersion", "tail"]},
        "depthblind_biokeep": {"kbet_rejection_rate": 0.10, "ilisi": 3.2, "batch_asw": 0.85,
                               "bio_silhouette": 0.55, "effective_rank": 4.0,
                               "invariance_ok": True},
    }
    path = tmp_path / "score_nested_c.json"
    path.write_text(json.dumps(score), encoding="utf-8")
    lb.main(["--root", str(root), "add", str(path),
             *ADDRESS, "--board", BOARD, "--method", "CANDI", "--version", "nested-c",
             "--date", "2026-08-27", "--lineage", "candi",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", "--allow-missing"])
    row = json.loads(row_path(root, "CANDI@nested-c").read_text(encoding="utf-8"))
    diag = row["metrics"]["diagnostics"]
    assert diag["depthdir"] == 0.80
    assert diag["depthcounterfact"] == 0.50
    assert diag["covspec"] == 0.70
    assert diag["biokeep"] == 0.55
    assert "covuse" not in diag
    assert "covshare" not in diag
    assert "depthblind" not in diag
    lin = lb.compile_leaderboard(root)["boards"][BOARD]["views"][VIEW][
        "sub_boards"]["candi_lineage"]["rows"]
    assert len(lin) == 1 and lin[0]["metrics"]["diagnostics"] == diag


def test_committed_rows_have_no_covariate_diagnostics() -> None:
    """The empty covariate tab is a fact about the stamped rows, not a missing widget."""
    compiled = lb.compile_leaderboard(REPO / "leaderboard")
    assert compiled["covariate_coverage"]["n_rows_with_diagnostics"] == 0
    assert "candi.bench" not in compiled["covariate_coverage"]["scorers"]
    assert "candi" not in compiled["covariate_coverage"]["lineages"]
    for base in ("rows", lb.ANCHOR):
        for p in (REPO / "leaderboard" / base).rglob("*.json"):
            row = json.loads(p.read_text(encoding="utf-8"))
            assert not row["metrics"].get("diagnostics")
            assert row["provenance"]["scorer"] != "candi.bench"


# ---------------------------------------------------------------- build ---

def test_compile_is_deterministic(root: Path) -> None:
    add_all(root)
    once = lb.dump_json(lb.compile_leaderboard(root))
    twice = lb.dump_json(lb.compile_leaderboard(root))
    assert once == twice
    assert "NaN" not in once


def test_empty_state_compiles(root: Path) -> None:
    """Zero rows is a valid board — CI runs on the committed, empty-rows repo."""
    compiled = lb.compile_leaderboard(root)
    for bid, board in compiled["boards"].items():
        for view in board["views"].values():
            assert view["rows"] == []
        assert board["climb"] == {}
        assert isinstance(board["pending"], list)
    assert compiled["covariate_coverage"]["n_rows_with_diagnostics"] == 0
    assert set(compiled["boards"]) == {"eic.19", "eic.pilot", lb.ANCHOR}
    assert compiled["boards"][lb.ANCHOR]["kind"] == "anchor"
    assert compiled["boards"]["eic.19"]["kind"] == "regime"


def test_pending_travels_into_the_payload_and_drops_when_stamped(root: Path) -> None:
    """Absence is a first-class row: pending lists ride into the payload, and a stamped
    method is dropped so a later `add` does not need a boards.json edit to un-grey it."""
    compiled = lb.compile_leaderboard(root)
    for bid in ("eic.19", "eic.pilot"):
        assert {p["method"] for p in compiled["boards"][bid]["pending"]} == {
            "CANDI", "Avocado", "ChromImpute", "eDICE", "Lavawizard",
            "avg", "avg-arcsinh", "marginal", "knn1", "knn5"}
    # the 2019 field is 25 rows, all awaiting their rescore on our grid (§6, §13.4)
    assert len(compiled["boards"][lb.ANCHOR]["pending"]) == 25
    candi = next(p for p in compiled["boards"]["eic.19"]["pending"] if p["method"] == "CANDI")
    assert candi["lineage"] == "candi" and candi["markers"] == []
    # §5 and §7 — a pending row already says which markers it will carry
    edice = next(p for p in compiled["boards"]["eic.19"]["pending"] if p["method"] == "eDICE")
    assert edice["markers"] == ["edice-embedding-substitution"]
    ci = next(p for p in compiled["boards"]["eic.19"]["pending"] if p["method"] == "ChromImpute")
    assert ci["markers"] == ["no-selection"]
    add(root, "score_fixture_a.json", "Lavawizard", "--allow-missing")
    compiled = lb.compile_leaderboard(root)
    assert all(p["method"] != "Lavawizard" for p in compiled["boards"]["eic.19"]["pending"])
    assert any(p["method"] == "Lavawizard" for p in compiled["boards"]["eic.pilot"]["pending"])


def test_site_is_a_nested_plain_language_view() -> None:
    """v5: gated three-layer picker; v4 science rendering stays behind the pick."""
    index = (REPO / "leaderboard" / "site" / "index.html").read_text(encoding="utf-8")
    js = (REPO / "leaderboard" / "site" / "app.js").read_text(encoding="utf-8")
    assert 'id: "eval-tabs"' in js and 'id: "head-tabs"' in js and 'id: "family-tabs"' in js
    assert "derivedHeads" in js and "headIdOf" in js
    assert "COVARIATE_HEAD" in js and 'COVARIATE_HEAD = "count"' in js
    assert "covariate_diagnostics" in js
    assert "data-summary" in js and "headSummaryBody" in js
    assert "All methods, one table" not in js
    assert "Is CANDI climbing?" not in index and "Is CANDI climbing?" not in js
    assert "strict (minus chr19)" not in index and "strict (minus chr19)" not in js
    assert "results computing" in js
    assert "helpBtn" in js and "rankBarChart" in js and "radarCard" in js
    assert "PVAL_RADAR_EDGES" in js and "COUNT_RADAR_EDGES" in js
    assert "radarCountBar" in js and "radarPolygon" in js
    assert 'RADAR_EDGES = ["pointwise", "distributional", "peaks", "count_arm"]' not in js
    assert "never share an axis" in js
    assert "ordering only" in js
    assert "oracle-scaled" in js
    assert "partial coverage" in js
    assert "not scored at this address" in js
    assert "compositeCell" in js
    assert "absent_note" in js and "will_populate" in js
    assert "No covariate-sensitivity numbers" in js
    assert "outerEval: null" in js and "showClimb: false" in js
    assert "comboComplete" in js and "writeHash" in js and "applyHash" in js
    assert "CANDI progress over time" in js
    assert "stroke-dasharray" not in js
    assert "methodLink" in js and "comboHelpBtn" in js and "cardOverlay" in js
    assert "help.json" in js
    assert "${m.label} (${truthSpace(bid)})" in js


def test_the_site_carries_no_retired_vocabulary() -> None:
    """§9 — P1/P2/P3, Dataset 2/3 and the board ids main/dev/entrants are gone from the board.

    boards.json and help.json are allowed to name a retired code once, to say it is retired;
    app.js and index.html must not carry one at all.
    """
    site = REPO / "leaderboard" / "site"
    for name in ("app.js", "index.html", "style.css"):
        text = (site / name).read_text(encoding="utf-8")
        for tok in ("Dataset-2", "Dataset 2", "Dataset-3", "Dataset 3",
                    '"main"', '"dev"', '"entrants"', "Full genome", "Chromosome 21"):
            assert tok not in text, f"{name} still carries the retired {tok!r}"
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for bid, b in boards["boards"].items():
        blob = json.dumps(b, ensure_ascii=False)
        for tok in ("Dataset-2", "Dataset 2", "Dataset-3", "Dataset 3"):
            assert tok not in blob, (bid, tok)


def test_the_site_expresses_the_address_the_truth_toggle_and_the_two_markers() -> None:
    """The five pieces of t82, each pinned to the code that renders it."""
    js = (REPO / "leaderboard" / "site" / "app.js").read_text(encoding="utf-8")
    index = (REPO / "leaderboard" / "site" / "index.html").read_text(encoding="utf-8")
    # §1 — the address, and the address bar that sets it
    assert "addressBar" in js and 'id: "address-bar"' in js and "addressLine" in js
    assert 'truth: "store"' in js and 'panel: "V_breadth"' in js and 'scope: "held-out"' in js
    assert "viewKeyFor" in js
    assert "regime, truth, panel, scope" in index or "regime, truth, panel, scope" in js
    # §6 — the truth toggle, and the arms it greys out
    assert "truthArms" in js and "headIsLive" in js and "headDeadReason" in js
    assert "tab-greyed" in js
    # §6 — the anchor block and its non-independence warning
    assert "anchorPanel" in js and 'id: "anchor-block"' in js
    assert "non_independence" in js and "anchor — we did not run these" in js
    # §5, §7 — the two row markers
    assert "markerBadges" in js and "badge-marker" in js
    # §15 — unranked is a state, not an error
    assert "unrankedBanner" in js and "cell-unranked" in js
    assert "rankingOf" in js and "isRanked" in js
    # §3.3 — void rows are named, never numbered
    assert "voidPanel" in js and 'id: "void-block"' in js


def test_the_compiled_payload_makes_the_address_rule_structural() -> None:
    """§1 — a row cannot reach the ranked table without all six fields, and §15's unranked
    state travels in the payload rather than being a front-end decision."""
    compiled = lb.compile_leaderboard(REPO / "leaderboard")
    assert compiled["canonical_view"] == lb.CANONICAL_VIEW
    assert set(compiled["address"]["fields"]) == {
        "method", "regime", "truth", "panel", "scope", "metric"}
    for bid, board in compiled["boards"].items():
        for key, view in board["views"].items():
            assert view["ranking"]["state"] == "unranked", (bid, key)
            assert view["ranking"]["reason"], (bid, key)
    anchor = compiled["boards"][lb.ANCHOR]
    assert list(anchor["views"]) == ["challenge.B_.held-out"]
    assert "no regime" in anchor["views"]["challenge.B_.held-out"]["ranking"]["reason"]
    # §3.3 — the void rows are named and dated, and carry no number at all
    assert len(compiled["void"]["rows"]) == 37
    blob = json.dumps(compiled["void"])
    for field in ("metrics", "mse", "gwcorr", "crps"):
        assert field not in blob, f"a void row leaked `{field}` into the payload"
    for r in compiled["void"]["rows"]:
        assert set(r) == {"method", "version", "date", "lineage", "former_board", "reason"}


def test_add_refuses_a_row_whose_address_does_not_resolve(root: Path) -> None:
    """§1 — if any field is unknown, the row does not go in the ranked table. `add` has no
    default for truth, panel or scope, and it checks each against boards.json's vocabulary."""
    base = ["--root", str(root), "add", str(FIX / "score_fixture_a.json"),
            "--board", BOARD, "--method", "fixture-a", "--version", "v1",
            "--date", "2026-08-27", "--lineage", "baseline",
            "--position-class", "generalizing", "--cell-class", "zero-shot",
            "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
            "--fir-path", "fake:/x", "--allow-missing"]
    with pytest.raises(SystemExit):  # argparse: the three address flags are required
        lb.main(base)
    with pytest.raises(SystemExit, match="unknown truth"):
        lb.main(base + ["--truth", "guess", "--panel", "V_breadth", "--scope", "held-out"])
    with pytest.raises(SystemExit, match="unknown panel"):
        lb.main(base + ["--truth", "store", "--panel", "V_", "--scope", "held-out"])
    with pytest.raises(SystemExit, match="unknown scope"):
        lb.main(base + ["--truth", "store", "--panel", "V_breadth", "--scope", "chr21"])
    lb.main(base + ADDRESS)
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["address"] == {"regime": "eic.19→20,21,22", "truth": "store",
                              "panel": "V_breadth", "scope": "held-out"}


def test_challenge_truth_refuses_a_count_or_peak_metric(root: Path) -> None:
    """§7 — the 2019 data has no counts and no peak calls, so two truths never share a row."""
    with pytest.raises(SystemExit, match="challenge truth"):
        lb.main(["--root", str(root), "add", str(FIX / "score_fixture_a.json"),
                 "--board", BOARD, "--truth", "challenge",
                 "--panel", "V_breadth", "--scope", "held-out",
                 "--method", "fixture-a", "--version", "v1",
                 "--date", "2026-08-27", "--lineage", "baseline",
                 "--position-class", "generalizing", "--cell-class", "zero-shot",
                 "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
                 "--fir-path", "fake:/x", "--allow-missing"])


def test_a_reported_only_address_keeps_its_numbers_and_drops_its_order(root: Path) -> None:
    """§4, §5.2, §15 — three separate reasons an address does not rank, and none of them
    is an error or an empty table. The numbers stay; only the order goes."""
    boards = json.loads((root / "boards.json").read_text(encoding="utf-8"))
    assert boards["boards"][BOARD]["noise_floor"]["measured"] is True
    add_all(root)
    compiled = lb.compile_leaderboard(root)
    views = compiled["boards"][BOARD]["views"]
    assert views[VIEW]["ranking"]["state"] == "ranked"
    assert views[VIEW]["rows"][0]["rank"] == [1, 1]
    # V_ matched is reported, never ranked (§5.2)
    matched = views["store.V_matched.held-out"]
    assert matched["ranking"]["state"] == "unranked"
    # genome-wide is reported, never ranked (§4)
    assert views["store.V_breadth.genome-wide"]["ranking"]["state"] == "unranked"
    # and with the floor unmeasured, even the ranked address stops ranking (§15)
    boards["boards"][BOARD]["noise_floor"] = {"measured": False, "note": "t86 has not landed"}
    (root / "boards.json").write_text(json.dumps(boards), encoding="utf-8")
    view = lb.compile_leaderboard(root)["boards"][BOARD]["views"][VIEW]
    assert view["ranking"]["state"] == "unranked"
    assert view["ranking"]["reason"] == "t86 has not landed"
    assert len(view["rows"]) == 4, "an unranked address still shows every row"
    for r in view["rows"]:
        assert r["rank"] is None and r["composite"] is None
        assert r["metric_ranks"] == {} and r["category_subscores"] == {}
        assert r["metrics"]["pval"]["mse"] is not None, "the number itself is never dropped"
    assert all(r["rank"] is None for r in view["sub_boards"]["count_arm"]["rows"])


def test_a_stamped_row_carries_its_method_markers(root: Path) -> None:
    """§5, §7 — markers come from boards.json, not from the stamper's memory."""
    add(root, "score_fixture_a.json", "eDICE", "--allow-missing")
    add(root, "score_fixture_b.json", "ChromImpute", "--allow-missing")
    add(root, "score_fixture_c.json", "fixture-c", "--allow-missing")
    rows = {r["method"]: r for r in board_view(root)["rows"]}
    assert rows["eDICE"]["markers"] == ["edice-embedding-substitution"]
    assert rows["ChromImpute"]["markers"] == ["no-selection"]
    assert rows["fixture-c"]["markers"] == []


def test_v4_head_family_mapping_matches_registry() -> None:
    """Heads partition registry metrics. Covariate has arm=null; it sits under Count
    because harness.c_block predicts NB (mu, n) against count truth."""
    reg = json.loads((REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"))
    js = (REPO / "leaderboard" / "site" / "app.js").read_text(encoding="utf-8")
    assert 'COVARIATE_HEAD = "count"' in js
    cats_by_head: dict[str, set[str]] = {"count": set(), "pval": set(), "peak": set()}
    for m in reg["metrics"]:
        if m["category"] == "peaks":
            cats_by_head["peak"].add(m["category"])
        elif m["category"] == "covariate_diagnostics":
            cats_by_head["count"].add(m["category"])
        elif m["arm"] == "count":
            cats_by_head["count"].add(m["category"])
        elif m["arm"] == "pval":
            cats_by_head["pval"].add(m["category"])
    assert cats_by_head["pval"] == {"pointwise", "distributional", "loss"}
    assert cats_by_head["count"] == {"count_arm", "loss", "covariate_diagnostics"}
    assert cats_by_head["peak"] == {"peaks"}
    pval_ids = {(m["arm"], m["key"]) for m in reg["metrics"]
                if m["arm"] == "pval" and m["category"] != "peaks"}
    count_ids = {(m["arm"], m["key"]) for m in reg["metrics"] if m["arm"] == "count"}
    assert not (pval_ids & count_ids)


# ---------------------------------------------------------------- site ---

import shutil  # noqa: E402  (kept with the site tests that need it)


def with_site(root: Path) -> Path:
    shutil.copytree(REPO / "leaderboard" / "site", root / "site")
    return root


def test_build_ships_the_site_beside_the_payload(root: Path, tmp_path: Path) -> None:
    add_all(with_site(root))
    out = tmp_path / "_site"
    lb.main(["--root", str(root), "build", "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["app.js", "help.json", "index.html",
                                                     "leaderboard.json", "style.css"]
    payload = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert set(payload["boards"]) == {"eic.19", "eic.pilot", lb.ANCHOR}
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'src="app.js"' in index and 'href="style.css"' in index
    assert "leaderboard.json" in (out / "app.js").read_text(encoding="utf-8")


def test_build_is_bit_identical_between_reruns(root: Path, tmp_path: Path) -> None:
    add_all(with_site(root))
    a, b = tmp_path / "a", tmp_path / "b"
    lb.main(["--root", str(root), "build", "--out", str(a)])
    lb.main(["--root", str(root), "build", "--out", str(b)])
    for name in ("leaderboard.json", "index.html", "app.js", "style.css", "help.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_site_makes_no_external_requests() -> None:
    """PRD §8 — no framework, no chart library, no external requests. The W3C namespace
    identifier is a name, not a request, and is the only thing allowed to look like a URL."""
    for name in ("index.html", "app.js", "style.css", "help.json"):
        text = (REPO / "leaderboard" / "site" / name).read_text(encoding="utf-8") \
            .replace("http://www.w3.org/", "").replace("http%3A%2F%2Fwww.w3.org%2F", "")
        assert "http://" not in text and "https://" not in text, name


def test_check_passes_on_the_committed_repo_state() -> None:
    """`check` is what CI runs; it must pass on the empty-rows repo as committed."""
    assert lb.main(["check"]) == 0


def test_check_passes_with_fixture_rows(root: Path) -> None:
    add_all(with_site(root))
    assert lb.main(["--root", str(root), "check"]) == 0


# ---------------------------------------------------------------- help.json ---

SITE_HEAD_FAMILIES = {
    "count": ["count_arm", "loss", "covariate_diagnostics"],
    "pval": ["pointwise", "distributional", "loss"],
    "peak": ["peaks"],
}


def site_combo_keys() -> set[str]:
    keys = set()
    for bid in ("eic.19", "eic.pilot", lb.ANCHOR):
        keys.add(f"{bid}/summary")
        keys.add(f"{bid}/radar")
        for head, fams in SITE_HEAD_FAMILIES.items():
            keys.add(f"{bid}/{head}/summary")
            keys.update(f"{bid}/{head}/{fam}" for fam in fams)
    return keys


def test_help_json_covers_combos_and_stamped_methods() -> None:
    """help.json parses; every site combo has a card; every stamped method has a card."""
    path = REPO / "leaderboard" / "site" / "help.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) >= {"methods", "combos", "metrics"}
    methods, combos = payload["methods"], payload["combos"]
    expected = site_combo_keys()
    assert set(combos) == expected, (
        f"help combos extra={set(combos) - expected} missing={expected - set(combos)}")
    for key, entry in combos.items():
        for field in ("question", "truth", "instrument", "devices", "caveats"):
            assert entry.get(field), f"{key} missing {field}"
        blob = " ".join(str(entry[f]) for f in ("question", "truth", "instrument", "devices", "caveats"))
        assert "UNKNOWN" not in blob and "UNVERIFIED" not in blob, key
    stamped = set()
    for p in (REPO / "leaderboard" / "rows").rglob("*.json"):
        stamped.add(json.loads(p.read_text(encoding="utf-8"))["method"])
    missing = sorted(stamped - set(methods))
    assert missing == [], f"stamped methods with no help card: {missing}"
    for name, card in methods.items():
        for field in ("what", "training", "classes", "scoring", "caveats"):
            assert card.get(field), f"{name} missing {field}"
        blob = " ".join(str(card[f]) for f in ("what", "training", "classes", "scoring", "caveats"))
        assert "UNKNOWN" not in blob and "UNVERIFIED" not in blob, name


def _metric_id(m: dict) -> str:
    slot = m["arm"] if m["arm"] in ("pval", "count") else "diagnostics"
    return f"{slot}/{m['key']}"


def test_help_metrics_cover_registry_and_mathml_is_xml() -> None:
    """Every registry metric has a help entry; every formula_mathml is well-formed MathML."""
    help_payload = json.loads(
        (REPO / "leaderboard" / "site" / "help.json").read_text(encoding="utf-8"))
    metrics = help_payload["metrics"]
    reg = json.loads((REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"))
    expected = {_metric_id(m) for m in reg["metrics"]}
    assert set(metrics) == expected, (
        f"help metrics extra={set(metrics) - expected} missing={expected - set(metrics)}")
    pit = metrics["pval/pit_ks"]
    for field in ("question", "formula_mathml", "estimator_notes", "read_rules"):
        assert pit.get(field), f"pval/pit_ks missing {field}"
    assert "Uniform(0,1)" in pit["question"] or "Uniform" in pit["question"]
    assert "Kolmogorov" in pit["estimator_notes"] or "PIT" in pit["question"]
    for mid, entry in metrics.items():
        for field in ("question", "formula_mathml", "estimator_notes", "read_rules"):
            assert entry.get(field), f"{mid} missing {field}"
        blob = " ".join(str(entry[f]) for f in
                        ("question", "formula_mathml", "estimator_notes", "read_rules"))
        assert "UNKNOWN" not in blob and "UNVERIFIED" not in blob, mid
        root = ET.fromstring(entry["formula_mathml"])
        tag = root.tag.split("}")[-1]
        assert tag == "math", f"{mid} root is {root.tag}"
        assert list(root) or (root.text and root.text.strip()), f"{mid} empty math"
    js = (REPO / "leaderboard" / "site" / "app.js").read_text(encoding="utf-8")
    assert "metricHelpBtn" in js and "metricCardBody" in js
    assert 'kind === "metric"' in js or 'card.kind === "metric"' in js
    assert "formula_mathml" in js
    assert "familyMetricHelps" in js
    assert 'state.midHead === "radar"' in js and "${state.outerEval}/radar" in js

