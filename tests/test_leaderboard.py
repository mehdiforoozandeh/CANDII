"""The rivals leaderboard compiler, pinned (LEADERBOARD_PRD.md §7.6, t58).

Everything here runs on the synthetic fixtures in `tests/fixtures/leaderboard/` — four fake
methods (`fixture-a` … `fixture-d`) with hand-chosen numbers, so every rank and every tie below
is arithmetic a reader can redo on paper. Nothing depends on `cruxvault/results/`, which is
untracked and per-machine by design.

The fixture geometry, once: `fixture-a` is best on every composite metric, `b` second, `c` third,
`d` worst. On the count arm `a` and `b` sit 0.05 apart — under the 0.09 macro-CRPS floor, so they
tie. `fixture-d` is pval-only (no count arm, no peak metrics), which is what exercises the §5.2
missing-arm rule: the peaks category drops out of the composite for everyone, and the count
sub-board ranks three rows, not four.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "leaderboard"

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
    for b in boards["boards"].values():
        b["frozen"]["store_manifest_hash"] = "fixhash-store"
        b["frozen"]["regime_sha256"] = "fixhash-regime"
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")
    return root


def add(root: Path, score: str, method: str, *extra: str) -> None:
    lb.main(["--root", str(root), "add", str(FIX / score),
             "--board", "dev", "--method", method, "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", *extra])


def add_all(root: Path) -> None:
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    add(root, "score_fixture_c.json", "fixture-c", "--allow-missing")
    add(root, "score_fixture_d.json", "fixture-d", "--allow-missing")


def dev_view(root: Path) -> dict:
    return lb.compile_leaderboard(root)["boards"]["dev"]["views"]["default"]


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
    assert set(boards["boards"]) == {"main", "dev", "entrants"}
    assert boards["boards"]["main"]["views"] == ["default", "strict"]


def test_committed_rows_all_pass_their_own_gates() -> None:
    """Every committed row loads through the reader's gates against the frozen boards.

    (Until M4 this asserted the dir was empty; real rows entered at M4, all via `add`.)"""
    reg = lb.load_registry(REPO / "leaderboard")
    boards = lb.load_boards(REPO / "leaderboard")
    rows_dir = REPO / "leaderboard" / "rows"
    for p in rows_dir.rglob("*.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        lb.gate_row_shape(row, reg)
        bid = p.parent.name
        assert row["board"] == bid
        lb.gate_row_against_board(row, boards["boards"][bid], bid)
    stray = [p.name for p in rows_dir.rglob("*")
             if p.is_file() and p.suffix != ".json" and p.name not in (".gitkeep", "README.md")]
    assert stray == []


def test_add_round_trips_a_row(root: Path) -> None:
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    row = json.loads((root / "rows" / "dev" / "fixture-a@v1.json").read_text(encoding="utf-8"))
    assert row["metrics"]["pval"]["mse"] == 1.0
    assert row["metrics"]["count"]["crps"] == 0.40
    assert row["badges"] == {"position": "position-generalizing",
                             "cell_types": "zero-shot cell types"}
    prov = row["provenance"]
    assert prov["scoring_sha"] == "deadbeef" and prov["fir_path"] == "fake:/scratch/fixture"
    assert prov["sigma_table"] is None
    assert prov["flags"]["pval_pred_space"] == "-log10p"
    reg = lb.load_registry(root)
    lb.gate_row_shape(row, reg)  # what `add` wrote passes the reader's own gate


def test_add_extracts_the_sigma_table_id(root: Path) -> None:
    """B1a rows carry their σ-table id, mechanically, from the score json's provenance."""
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    row = json.loads((root / "rows" / "dev" / "fixture-b@v1.json").read_text(encoding="utf-8"))
    assert row["provenance"]["sigma_table"] == {"method": "fixture-b",
                                                "fitted_on": "regime.eic_val eval_pairs"}


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
    row = json.loads((root / "rows" / "dev" / "fixture-d@v1.json").read_text(encoding="utf-8"))
    assert "count/crps" in row["missing_metrics"] and "pval/auprc" in row["missing_metrics"]


def test_add_refuses_an_unfrozen_board(tmp_path: Path) -> None:
    """A board whose hashes are still TODO refuses every add (the pre-M4 state)."""
    root = tmp_path / "leaderboard"
    (root / "rows").mkdir(parents=True)
    root.joinpath("registry.json").write_text(
        (REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for b in boards["boards"].values():
        b["frozen"]["store_manifest_hash"] = "TODO-freeze-at-stamping"
        b["frozen"]["regime_sha256"] = "TODO-freeze-at-stamping"
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")
    with pytest.raises(SystemExit, match="not frozen"):
        add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")


def test_committed_boards_are_frozen() -> None:
    """M4 froze all three boards; a TODO hash regressing in would silently block adds."""
    boards = lb.load_boards(REPO / "leaderboard")
    for bid, b in boards["boards"].items():
        for field in ("store_manifest_hash", "regime_sha256"):
            v = b["frozen"][field]
            assert len(v) == 64 and not v.startswith("TODO"), (bid, field)
        assert b["frozen"].get("frozen_note"), f"board {bid} must say what its hashes digest"


def test_placement_mode_stamps_a_dataset3_row(tmp_path: Path) -> None:
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
    for b in boards["boards"].values():
        b["frozen"]["store_manifest_hash"] = "fixhash-store"
        b["frozen"]["regime_sha256"] = "fixhash-regime"
    boards["boards"]["entrants"]["frozen"]["regime_sha256"] = lb.sha256_file(pfile)
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")

    argv = ["--root", str(root), "add", str(pfile), "--board", "entrants",
            "--method", "fixture-entrant", "--version", "round2-2019",
            "--date", "2026-08-26", "--lineage", "entrant",
            "--position-class", "unrecorded", "--cell-class", "unrecorded",
            "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
            "--fir-path", "fake:/x", "--allow-missing",
            "--placement-method", "fixture-entrant"]
    lb.main(argv)
    row = json.loads((root / "rows" / "entrants" / "fixture-entrant@round2-2019.json")
                     .read_text(encoding="utf-8"))
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
                 "--board", "dev", "--method", "fixture-a", "--version", "v1",
                 "--date", "2026-08-27", "--lineage", "baseline",
                 "--position-class", "generalizing", "--cell-class", "zero-shot",
                 "--scoring-sha", "deadbeef", "--store-manifest-hash", "some-other-store",
                 "--fir-path", "fake:/x", "--allow-missing"])


# ---------------------------------------------------------------- composite ---

def test_plain_ranks_and_the_missing_arm_rule(root: Path) -> None:
    add_all(root)
    view = dev_view(root)
    # fixture-d has no auprc, so peaks leaves the composite for every row (§5.2);
    # pointwise and distributional stay because every row has them.
    assert view["categories_in_composite"] == ["distributional", "pointwise"]
    ranks = {r["id"]: r["rank"] for r in view["rows"]}
    assert ranks == {"fixture-a@v1": [1, 1], "fixture-b@v1": [2, 2],
                     "fixture-c@v1": [3, 3], "fixture-d@v1": [4, 4]}
    # the count arm is a sub-board, not a composite category, and d is simply not on it
    sub = view["sub_boards"]["count_arm"]
    assert [r["id"] for r in sub["rows"]] == ["fixture-a@v1", "fixture-b@v1", "fixture-c@v1"]
    # a and b sit 0.05 apart on count crps — under the 0.09 floor, so they share "1-2"
    by_id = {r["id"]: r["rank"] for r in sub["rows"]}
    assert by_id == {"fixture-a@v1": [1, 2], "fixture-b@v1": [1, 2], "fixture-c@v1": [3, 3]}


def test_floor_ties_propagate_into_composite_spreads(root: Path) -> None:
    """PRD §5.4 — a floored metric ties into a rank interval, the interval into the
    category mean, the mean into the composite spread."""
    reg = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    for m in reg["metrics"]:
        if m["key"] == "crps" and m["arm"] == "pval":
            m["floor"] = 0.05  # a-b gap is 0.01: tied; every other gap clears it
    (root / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    add_all(root)
    rows = {r["id"]: r for r in dev_view(root)["rows"]}
    assert rows["fixture-a@v1"]["metric_ranks"]["pval/crps"] == [1, 2]
    assert rows["fixture-b@v1"]["metric_ranks"]["pval/crps"] == [1, 2]
    assert rows["fixture-a@v1"]["composite"] == [1.0, 1.5]   # pointwise (1,1) + crps (1,2)
    assert rows["fixture-b@v1"]["composite"] == [1.5, 2.0]
    assert rows["fixture-a@v1"]["rank"] == [1, 2]
    assert rows["fixture-b@v1"]["rank"] == [1, 2]
    assert rows["fixture-c@v1"]["rank"] == [3, 3]
    assert rows["fixture-d@v1"]["rank"] == [4, 4]


def test_lineage_sub_board_takes_only_candi_diagnostics(root: Path) -> None:
    add_all(root)
    view = dev_view(root)
    assert view["sub_boards"]["candi_lineage"]["rows"] == []  # no candi row, no lineage board


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


def test_pending_travels_into_the_payload_and_drops_when_stamped(root: Path) -> None:
    """Absence is a first-class row: pending lists ride into the payload, and a stamped
    method is dropped so a later `add` does not need a boards.json edit to un-grey it."""
    compiled = lb.compile_leaderboard(root)
    assert {p["method"] for p in compiled["boards"]["main"]["pending"]} >= {
        "Lavawizard", "avg", "knn5"}
    assert {p["method"] for p in compiled["boards"]["dev"]["pending"]} == {"Lavawizard"}
    assert compiled["boards"]["entrants"]["pending"] == []
    add(root, "score_fixture_a.json", "Lavawizard", "--allow-missing")
    compiled = lb.compile_leaderboard(root)
    assert all(p["method"] != "Lavawizard" for p in compiled["boards"]["dev"]["pending"])
    assert any(p["method"] == "Lavawizard" for p in compiled["boards"]["main"]["pending"])


def test_site_is_a_merged_plain_language_view() -> None:
    """v2: one table, no board tabs, no climbing headline, no jargon labels in the HTML."""
    index = (REPO / "leaderboard" / "site" / "index.html").read_text(encoding="utf-8")
    js = (REPO / "leaderboard" / "site" / "app.js").read_text(encoding="utf-8")
    assert 'id="board-tabs"' not in index
    assert "Is CANDI climbing?" not in index and "Is CANDI climbing?" not in js
    assert "strict (minus chr19)" not in index and "strict (minus chr19)" not in js
    assert "Dataset-3" not in index
    assert "results computing" in js
    assert "helpBtn" in js and "rankBarChart" in js and "radarCard" in js
    assert "ordering only" in js
    assert "oracle-scaled" in js


# ---------------------------------------------------------------- site ---

import shutil  # noqa: E402  (kept with the site tests that need it)


def with_site(root: Path) -> Path:
    shutil.copytree(REPO / "leaderboard" / "site", root / "site")
    return root


def test_build_ships_the_site_beside_the_payload(root: Path, tmp_path: Path) -> None:
    add_all(with_site(root))
    out = tmp_path / "_site"
    lb.main(["--root", str(root), "build", "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["app.js", "index.html",
                                                     "leaderboard.json", "style.css"]
    payload = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert set(payload["boards"]) == {"main", "dev", "entrants"}
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'src="app.js"' in index and 'href="style.css"' in index
    assert "leaderboard.json" in (out / "app.js").read_text(encoding="utf-8")


def test_build_is_bit_identical_between_reruns(root: Path, tmp_path: Path) -> None:
    add_all(with_site(root))
    a, b = tmp_path / "a", tmp_path / "b"
    lb.main(["--root", str(root), "build", "--out", str(a)])
    lb.main(["--root", str(root), "build", "--out", str(b)])
    for name in ("leaderboard.json", "index.html", "app.js", "style.css"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_site_makes_no_external_requests() -> None:
    """PRD §8 — no framework, no chart library, no external requests. The W3C namespace
    identifier is a name, not a request, and is the only thing allowed to look like a URL."""
    for name in ("index.html", "app.js", "style.css"):
        text = (REPO / "leaderboard" / "site" / name).read_text(encoding="utf-8") \
            .replace("http://www.w3.org/", "").replace("http%3A%2F%2Fwww.w3.org%2F", "")
        assert "http://" not in text and "https://" not in text, name


def test_check_passes_on_the_committed_repo_state() -> None:
    """`check` is what CI runs; it must pass on the empty-rows repo as committed."""
    assert lb.main(["check"]) == 0


def test_check_passes_with_fixture_rows(root: Path) -> None:
    add_all(with_site(root))
    assert lb.main(["--root", str(root), "check"]) == 0
