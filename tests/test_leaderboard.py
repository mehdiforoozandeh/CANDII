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

The committed fixtures predate §5.2's three panels, so `panelled()` gives each one the `panels`
block `harness.panel_macros` would have written, with the SAME numbers under all three panels —
`add` reading a panel is a lookup, and a test of a lookup must be able to say which key was read.
The last section of this file drops the synthetic fixtures entirely and stamps a score json the
real `candi.bench.external` produced over a real store, under both truths and both scopes.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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

def panelled(score: dict) -> dict:
    """A fixture score json, plus the `panels` block §5.2 requires.

    Every panel gets the arm's own macro numbers, so a test can assert WHICH panel `add` read only
    by the key it used — nothing here is invented, and `n_experiments` / `ranked` / `matched_to`
    are the non-metric keys `harness.panel_macros` really writes. `matched_to` is non-empty because
    the fixture stands for a pass that scored `B_` rows: an empty one is the unfilled panel `add`
    refuses, and the real-score section of this file builds that from the scorer itself.
    """
    out = json.loads(json.dumps(score))
    out["panels"] = {
        arm: {"V_breadth": {**block, "n_experiments": 1, "ranked": True},
              "V_matched": {**block, "n_experiments": 1, "ranked": False,
                            "matched_to": ["H3K4me3"]},
              "B": {**block, "n_experiments": 1, "ranked": True}}
        for arm, block in (out.get("macro") or {}).items()
    }
    return out


def write_score(path: Path, score: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(panelled(score)), encoding="utf-8")
    return path


#: The fixtures were written when a σ table could be fitted on eval pairs; §7 now refuses that, so
#: `fixture_score` rewrites `fitted_on` into the form `competitors/sigma_pass.py` produces. The raw
#: fixture value is fed in untouched by `test_add_refuses_a_sigma_table_fitted_on_the_eval_panel`.
FITTED_ON = "training-residuals: regime.eic_19.sigma.json T_ self-pairs, 12 cells, chroms chr19"


def fixture_score(root: Path, name: str) -> Path:
    """One committed fixture, panelled and with a legal σ `fitted_on`, beside the root."""
    score = json.loads((FIX / name).read_text(encoding="utf-8"))
    sigma = (score.get("provenance") or {}).get("sigma_table")
    if isinstance(sigma, dict):
        sigma["fitted_on"] = FITTED_ON
    return write_score(root.parent / "scores" / name, score)


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


def edit_boards(root: Path, edit) -> None:
    """Rewrite the root's boards.json through `edit`, which mutates the loaded dict in place."""
    boards = json.loads((root / "boards.json").read_text(encoding="utf-8"))
    edit(boards)
    (root / "boards.json").write_text(json.dumps(boards), encoding="utf-8")


def add(root: Path, score: str, method: str, *extra: str) -> None:
    lb.main(["--root", str(root), "add", str(fixture_score(root, score)), *ADDRESS,
             "--board", BOARD, "--method", method, "--version", "v1",
             "--date", "2026-08-27", "--lineage", "baseline",
             "--position-class", "generalizing", "--cell-class", "zero-shot",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/scratch/fixture", *extra])


def add_path(root: Path, score: Path, method: str, *extra: str) -> None:
    """`add` on an already-written score json — the address flags default to the canonical view."""
    argv = ["--root", str(root), "add", str(score),
            "--board", BOARD, "--method", method, "--version", "v1",
            "--date", "2026-08-27", "--lineage", "baseline",
            "--position-class", "generalizing", "--cell-class", "zero-shot",
            "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
            "--fir-path", "fake:/scratch/fixture", *extra]
    if "--panel" not in extra:
        argv += ADDRESS
    lb.main(argv)


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
    assert set(boards["markers"]) == {"edice-embedding-substitution", "no-selection",
                                      "selection-key-pval-mse"}
    assert "edice-embedding-substitution" in boards["method_markers"]["eDICE"]
    assert "no-selection" in boards["method_markers"]["ChromImpute"]
    # the 2026-09-01 ruling: the three trainable rivals select on pval mse, CANDI on count crps
    for method in ("Avocado", "eDICE", "Lavawizard"):
        assert "selection-key-pval-mse" in boards["method_markers"][method], method
    assert "selection-key-pval-mse" not in boards["method_markers"].get("CANDI", [])
    key_marker = boards["markers"]["selection-key-pval-mse"]["eli5"]
    assert "count-arm CRPS" in key_marker and "handicaps CANDI" in key_marker
    assert "2026-09-01" in key_marker
    # §4's blanking rule and the code that enforces it name the same three methods
    rule = boards["scopes"]["genome-wide"]["blanking_rule"]
    assert set(lb.GENOME_WIDE_BLANKED) == {"Avocado", "ChromImpute", "Lavawizard"}
    for method in lb.GENOME_WIDE_BLANKED:
        assert method in rule, method
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
                             "cell_types": "zero-shot cell types",
                             "sigma": lb.SIGMA_BADGE_NATIVE}
    assert row["ranked"] is True
    assert row["in_sample_fraction"] is None
    prov = row["provenance"]
    assert prov["scoring_sha"] == "deadbeef" and prov["fir_path"] == "fake:/scratch/fixture"
    assert prov["sigma_table"] is None
    assert prov["has_peak_head"] is False
    assert prov["flags"]["pval_pred_space"] == "-log10p"
    assert "truth_manifest_hash" not in prov, "a store row pins the store manifest, nothing else"
    reg = lb.load_registry(root)
    lb.gate_row_shape(row, reg)  # what `add` wrote passes the reader's own gate


def test_add_extracts_the_sigma_table_id_and_badges_the_fitted_spread(root: Path) -> None:
    """§7 — a σ table on the row is the fitted-flat-σ badge, and its `fitted_on` travels with it."""
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    row = json.loads(row_path(root, "fixture-b@v1").read_text(encoding="utf-8"))
    assert row["provenance"]["sigma_table"] == {"method": "fixture-b", "fitted_on": FITTED_ON}
    assert row["badges"]["sigma"] == lb.SIGMA_BADGE_FITTED == "fitted flat σ"
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-b@v1")
    assert compiled["provenance"]["sigma_table"]["method"] == "fixture-b"
    assert compiled["badges"]["sigma"] == lb.SIGMA_BADGE_FITTED
    assert compiled["has_peak_head"] is False


def test_add_refuses_a_sigma_table_fitted_on_the_eval_panel(root: Path) -> None:
    """The committed fixture's σ table says `regime.eic_val eval_pairs` — a Rule 1 leak. §7 fits σ
    on training residuals only, so the row is refused rather than badged."""
    raw = json.loads((FIX / "score_fixture_b.json").read_text(encoding="utf-8"))
    assert raw["provenance"]["sigma_table"]["fitted_on"] == "regime.eic_val eval_pairs"
    with pytest.raises(SystemExit, match="training-residuals:"):
        add_path(root, write_score(root.parent / "sigma_leak.json", raw), "fixture-b",
                 "--allow-missing")
    assert not row_path(root, "fixture-b@v1").exists()


def test_add_extracts_has_peak_head_from_bernoulli_nll(root: Path) -> None:
    """A real peak head is the presence of bernoulli_nll in the score macro, not a value range."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["macro"]["pval"]["bernoulli_nll"] = 0.42
    add_path(root, write_score(root.parent / "peakhead.json", score), "fixture-a",
             "--allow-missing")
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["provenance"]["has_peak_head"] is True
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-a@v1")
    assert compiled["has_peak_head"] is True


def test_add_carries_contributor_mode_in_flags(root: Path) -> None:
    """contributor_mode and clip are FLAG_KEYS members: add copies them and compile keeps them."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["contributor_mode"] = True
    score["provenance"]["clip"] = "p99"
    add_path(root, write_score(root.parent / "flags.json", score), "fixture-a", "--allow-missing")
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


TRUTH_SHA = "b99dde1107125311d5af3b68964f56b77cc5d568c7e846d76aa1718290216284"


def entrant_score(root: Path, *, manifest_sha: str = TRUTH_SHA) -> Path:
    """One 2019 entrant's own score json, as `candi.bench.external --truth-root` writes it.

    pval only, `truth.source == "challenge"`, and a truth manifest hash — which is what the anchor
    block freezes in place of a regime, because we never trained these rows (§6).
    """
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["macro"]["count"] = {}
    for key in ("auprc", "peak_base_rate"):  # no 2019 peak calls to score against
        score["macro"]["pval"].pop(key, None)
    score["provenance"]["truth"] = {"source": "challenge", "root": "/x/truth",
                                    "manifest_sha256": manifest_sha}
    return write_score(root.parent / f"entrant_{manifest_sha[:8]}.json", score)


def anchor_root(tmp_path: Path, *, truth_sha: str = TRUTH_SHA) -> Path:
    """A leaderboard root whose anchor block is frozen to a challenge truth root's manifest."""
    root = tmp_path / "leaderboard"
    (root / "rows").mkdir(parents=True)
    root.joinpath("registry.json").write_text(
        (REPO / "leaderboard" / "registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    boards = json.loads((REPO / "leaderboard" / "boards.json").read_text(encoding="utf-8"))
    for b in list(boards["boards"].values()) + [boards["anchor"]]:
        b["frozen"]["store_manifest_hash"] = "fixhash-store"
        b["frozen"]["regime_sha256"] = "fixhash-regime"
    boards["anchor"]["frozen"]["regime_sha256"] = truth_sha
    root.joinpath("boards.json").write_text(json.dumps(boards), encoding="utf-8")
    return root


def add_entrant(root: Path, score: Path, method: str, *extra: str) -> None:
    lb.main(["--root", str(root), "add", str(score), "--board", lb.ANCHOR,
             "--truth", "challenge", "--panel", "B_", "--scope", "held-out",
             "--method", method, "--version", "2019-submission",
             "--date", "2026-09-02", "--lineage", "entrant",
             "--position-class", "unrecorded", "--cell-class", "unrecorded",
             "--scoring-sha", "deadbeef", "--store-manifest-hash", "fixhash-store",
             "--fir-path", "fake:/x", "--allow-missing", *extra])


def test_an_anchor_row_is_one_entrants_own_score_json(tmp_path: Path) -> None:
    """§6 — the 25 anchor rows are rescored through our scorer, one score json each. No regime,
    `--lineage entrant`, and pval only."""
    root = anchor_root(tmp_path)
    add_entrant(root, entrant_score(root), "fixture-entrant")
    row = json.loads((root / lb.ANCHOR / "challenge.B_.held-out"
                      / "fixture-entrant@2019-submission.json").read_text(encoding="utf-8"))
    assert row["address"] == {"regime": None, "truth": "challenge",
                              "panel": "B_", "scope": "held-out"}
    assert row["lineage"] == "entrant"
    assert row["metrics"]["pval"]["mse"] == 1.0
    assert "count" not in row["metrics"]
    assert row["ranked"] is False, "§6 — an anchor row shares no denominator with ours"
    assert row["provenance"]["truth_manifest_hash"] == TRUTH_SHA
    compiled = lb.compile_leaderboard(root)["boards"][lb.ANCHOR]
    ids = [r["id"] for r in compiled["views"]["challenge.B_.held-out"]["rows"]]
    assert ids == ["fixture-entrant@2019-submission"]


def test_the_anchor_gate_is_the_truth_roots_manifest_hash(tmp_path: Path) -> None:
    """The anchor has no regime to freeze, so `frozen.regime_sha256` is the sha256 of the challenge
    truth root's manifest.json. A row rescored against another build of that truth is refused."""
    root = anchor_root(tmp_path)
    other = "0" * 64
    with pytest.raises(SystemExit, match="truth manifest hash"):
        add_entrant(root, entrant_score(root, manifest_sha=other), "fixture-entrant")
    assert not (root / lb.ANCHOR / "challenge.B_.held-out").exists()
    # and an unfrozen anchor takes no row at all
    root2 = anchor_root(tmp_path / "second", truth_sha="TODO-anchor-truth-manifest")
    with pytest.raises(SystemExit, match="not frozen"):
        add_entrant(root2, entrant_score(root2), "fixture-entrant")


def test_an_anchor_row_carries_no_method_marker_of_ours(tmp_path: Path) -> None:
    """`Lavawizard` is both a retrained rival and a frozen 2019 entrant. Every marker names
    something a method did while WE trained it, so the anchor block takes none of them."""
    root = anchor_root(tmp_path)
    add_entrant(root, entrant_score(root), "Lavawizard")
    row = json.loads((root / lb.ANCHOR / "challenge.B_.held-out"
                      / "Lavawizard@2019-submission.json").read_text(encoding="utf-8"))
    assert row["markers"] == []
    anchor = lb.compile_leaderboard(root)["boards"][lb.ANCHOR]
    assert all(p["markers"] == [] for p in anchor["pending"])


def test_an_anchor_row_cannot_be_addressed_off_the_anchor_block(tmp_path: Path) -> None:
    """§6 — the anchor block is one address: `B_`, held-out. A row anywhere else in that directory
    would rank beside ours by living in the same tree."""
    root = anchor_root(tmp_path)
    score = entrant_score(root)
    with pytest.raises(SystemExit, match="panel `V_breadth`"):
        add_entrant(root, score, "fixture-entrant", "--panel", "V_breadth")
    with pytest.raises(SystemExit, match="scope `genome-wide`"):
        add_entrant(root, score, "fixture-entrant", "--scope", "genome-wide")


def test_the_anchor_offers_only_the_truths_boards_json_lists(tmp_path: Path) -> None:
    """The committed anchor names one truth — the 2019 measurement the field was ranked under — so
    a store-truth anchor row is refused. It lands the moment boards.json offers both, and nothing
    in the code hard-codes which."""
    root = anchor_root(tmp_path)
    store_score = fixture_score(root, "score_fixture_a.json")  # no truth block → store
    with pytest.raises(SystemExit, match="not offered by"):
        add_entrant(root, store_score, "fixture-entrant", "--truth", "store")
    assert lb.view_keys(lb.load_boards(root), lb.ANCHOR) == ["challenge.B_.held-out"]
    edit_boards(root, lambda b: b["anchor"].update(truths=["challenge", "store"]))
    assert lb.view_keys(lb.load_boards(root), lb.ANCHOR) == ["challenge.B_.held-out",
                                                             "store.B_.held-out"]
    add_entrant(root, store_score, "fixture-entrant", "--truth", "store")
    row = json.loads((root / lb.ANCHOR / "store.B_.held-out"
                      / "fixture-entrant@2019-submission.json").read_text(encoding="utf-8"))
    # a store-truth row is pinned by the store manifest; the truth-manifest gate is the
    # challenge row's, because that is the only truth a manifest hash names
    assert row["address"]["truth"] == "store"
    assert "truth_manifest_hash" not in row["provenance"]
    anchor = lb.compile_leaderboard(root)["boards"][lb.ANCHOR]
    assert [r["id"] for r in anchor["views"]["store.B_.held-out"]["rows"]] == [
        "fixture-entrant@2019-submission"]


def test_the_placement_path_is_gone() -> None:
    """§6 retires the vendored 001 scorer from the board, so the t54 placement path goes with it:
    no flag, no shaper, no `check` branch keyed on the old scorer's name."""
    src = (REPO / "tools" / "leaderboard.py").read_text(encoding="utf-8")
    assert "--placement-method" not in lb.build_parser().format_help()
    assert not hasattr(lb, "placement_score")
    assert not hasattr(lb, "PLACEMENT_SCORER")
    assert "placement_method" not in lb.FLAG_KEYS
    # at most one line may still say the word, and only to record the retirement
    said = [ln for ln in src.splitlines() if "placement" in ln]
    assert len(said) <= 1, said
    for ln in said:
        assert ln.lstrip().startswith("#"), ln


def test_add_refuses_a_wrong_store_hash(root: Path) -> None:
    with pytest.raises(SystemExit, match="does not match"):
        lb.main(["--root", str(root), "add", str(fixture_score(root, "score_fixture_a.json")),
                 *ADDRESS, "--board", BOARD, "--method", "fixture-a", "--version", "v1",
                 "--date", "2026-08-27", "--lineage", "baseline",
                 "--position-class", "generalizing", "--cell-class", "zero-shot",
                 "--scoring-sha", "deadbeef", "--store-manifest-hash", "some-other-store",
                 "--fir-path", "fake:/x", "--allow-missing"])


def test_add_refuses_a_score_json_with_no_panels_block(root: Path) -> None:
    """§5.2 — `macro` pools every scored track and is not any one panel. A pre-panels file has
    nothing to stamp under a panel label, and reading its macro as one would print a 22-assay
    number beneath an 8-assay heading."""
    raw = root.parent / "nopanels.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text((FIX / "score_fixture_a.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="no `panels`"):
        add_path(root, raw, "fixture-a", "--allow-missing")


def test_add_refuses_a_blanked_genome_wide_cell(root: Path) -> None:
    """§4 — Avocado, ChromImpute and Lavawizard were fit at every position, so their genome-wide
    number is a memorisation score. A blanked cell is not computed, and is not stamped."""
    score = fixture_score(root, "score_fixture_a.json")
    for method in lb.GENOME_WIDE_BLANKED:
        with pytest.raises(SystemExit, match="no genome-wide cell") as e:
            add_path(root, score, method, "--panel", "V_breadth", "--scope", "genome-wide",
                     "--truth", "store", "--allow-missing")
        assert "blank" in str(e.value), "the refusal must name §4's blanking rule"
    # a method that is out-of-sample somewhere is refused for a different reason: this score json
    # was never given --held-out-chroms, so its genome-wide number was NOT COMPUTED
    with pytest.raises(SystemExit, match="genome_wide"):
        add_path(root, score, "eDICE", "--panel", "V_breadth", "--scope", "genome-wide",
                 "--truth", "store", "--allow-missing")


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


def test_partial_coverage_blanks_composite_without_poisoning_peers(root: Path) -> None:
    """A method missing an entire composite category (distributional) gets a dash,
    not a zeroed composite, and does not drop that category for complete peers."""
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    add(root, "score_fixture_b.json", "fixture-b", "--allow-missing")
    score = json.loads((FIX / "score_fixture_c.json").read_text(encoding="utf-8"))
    for key in ("crps", "pit_ks", "coverage_95", "gaussian_nll"):
        score["macro"]["pval"].pop(key, None)
    add_path(root, write_score(root.parent / "score_partial.json", score), "fixture-partial",
             "--allow-missing")
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


def test_missing_count_arm_alone_does_not_blank_composite(root: Path) -> None:
    """Count space is a sub-board, not a composite category (B1b). A pval-only row
    that fully covers pointwise + distributional + peaks still gets a composite."""
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    score = json.loads((FIX / "score_fixture_b.json").read_text(encoding="utf-8"))
    score["macro"]["count"] = {}
    score["provenance"]["sigma_table"]["fitted_on"] = FITTED_ON
    add_path(root, write_score(root.parent / "score_pval_only.json", score),
             "fixture-pval-only", "--allow-missing")
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
    path = write_score(tmp_path / "score_candi.json", score)
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
    path = write_score(tmp_path / "score_nested_c.json", score)
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
    assert edice["markers"] == ["edice-embedding-substitution", "selection-key-pval-mse"]
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


def test_the_site_expresses_the_address_the_truth_toggle_and_the_row_markers() -> None:
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
    # §5, §7 — the row markers, and §7's mandatory spread badge read off the stamped row
    assert "markerBadges" in js and "badge-marker" in js
    assert "row.badges && row.badges.sigma" in js
    assert "fitted flat σ" in js and "native heteroscedastic" in js
    assert "TRAINING-set residuals" in js, "the fitted-σ badge must name §7's rule"
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
    base = ["--root", str(root), "add", str(fixture_score(root, "score_fixture_a.json")),
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
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["truth"] = {"source": "challenge", "root": "/x",
                                    "manifest_sha256": "fixhash-regime"}
    with pytest.raises(SystemExit, match="challenge truth"):
        add_path(root, write_score(root.parent / "challenge_counts.json", score), "fixture-a",
                 "--truth", "challenge", "--panel", "V_breadth", "--scope", "held-out",
                 "--allow-missing")


def test_a_challenge_row_must_come_from_a_challenge_truth_pass(root: Path) -> None:
    """§6 — the toggle measures the pipeline and nothing else, so a row's truth is the score json's
    own `provenance.truth.source`, never a label the stamper chose."""
    with pytest.raises(SystemExit, match="measured against `store` truth"):
        add(root, "score_fixture_a.json", "fixture-a", "--allow-missing",
            "--truth", "challenge")
    # ... and a challenge pass cannot be stamped as a store row either
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"]["truth"] = {"source": "challenge", "root": "/x",
                                    "manifest_sha256": "abc"}
    path = write_score(root.parent / "challenge_pass.json", score)
    with pytest.raises(SystemExit, match="measured against `challenge` truth"):
        add_path(root, path, "fixture-a", "--allow-missing")


def test_a_challenge_row_names_which_build_of_the_2019_truth_it_read(root: Path) -> None:
    """A challenge row copies `provenance.truth.manifest_sha256` onto itself: the store manifest
    pins the grid, and only the truth manifest says what the numbers were compared against."""
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["macro"]["count"] = {}
    for key in ("auprc", "peak_base_rate"):
        score["macro"]["pval"].pop(key, None)
    score["provenance"]["truth"] = {"source": "challenge", "root": "/x/truth",
                                    "manifest_sha256": TRUTH_SHA}
    add_path(root, write_score(root.parent / "challenge_ok.json", score), "fixture-a",
             "--truth", "challenge", "--panel", "B_", "--scope", "held-out", "--allow-missing")
    row = json.loads((root / "rows" / BOARD / "challenge.B_.held-out"
                      / "fixture-a@v1.json").read_text(encoding="utf-8"))
    assert row["provenance"]["truth_manifest_hash"] == TRUTH_SHA
    assert "count" not in row["metrics"], "no count truth, so no count arm — not a zero"
    assert "auprc" not in row["metrics"]["pval"]
    # the same file with the hash removed cannot say which truth it read, and is refused
    score["provenance"]["truth"].pop("manifest_sha256")
    with pytest.raises(SystemExit, match="manifest_sha256"):
        add_path(root, write_score(root.parent / "challenge_nohash.json", score), "fixture-a",
                 "--truth", "challenge", "--panel", "B_", "--scope", "held-out",
                 "--allow-missing")


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
    """§5, §7 and the 2026-09-01 ruling — markers come from boards.json, not from the stamper's
    memory, and a rival that selects on pval mse says so on every row."""
    add(root, "score_fixture_a.json", "eDICE", "--allow-missing")
    add(root, "score_fixture_b.json", "ChromImpute", "--allow-missing")
    add(root, "score_fixture_c.json", "Avocado", "--allow-missing")
    add(root, "score_fixture_d.json", "fixture-d", "--allow-missing")
    rows = {r["method"]: r for r in board_view(root)["rows"]}
    assert rows["eDICE"]["markers"] == ["edice-embedding-substitution",
                                        "selection-key-pval-mse"]
    assert rows["ChromImpute"]["markers"] == ["no-selection"]
    assert rows["Avocado"]["markers"] == ["selection-key-pval-mse"]
    assert rows["fixture-d"]["markers"] == []


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


def test_the_in_sample_fraction_is_copied_when_a_producer_writes_one(root: Path) -> None:
    """§4 — the genome-wide cell carries the per-cell in-sample fraction. Nothing writes it today,
    so the row records an explicit `null`; the moment a producer does, `add` copies it."""
    assert lb.IN_SAMPLE_KEY == "in_sample_fraction"
    add(root, "score_fixture_a.json", "fixture-a", "--allow-missing")
    assert json.loads(row_path(root, "fixture-a@v1").read_text())["in_sample_fraction"] is None
    score = json.loads((FIX / "score_fixture_a.json").read_text(encoding="utf-8"))
    score["provenance"][lb.IN_SAMPLE_KEY] = 1 / 23
    add_path(root, write_score(root.parent / "insample.json", score), "fixture-a",
             "--allow-missing", "--force")
    row = json.loads(row_path(root, "fixture-a@v1").read_text(encoding="utf-8"))
    assert row["in_sample_fraction"] == 1 / 23
    compiled = next(r for r in board_view(root)["rows"] if r["id"] == "fixture-a@v1")
    assert compiled["in_sample_fraction"] == 1 / 23
    score["provenance"][lb.IN_SAMPLE_KEY] = "most of it"
    with pytest.raises(SystemExit, match="in_sample_fraction"):
        add_path(root, write_score(root.parent / "insample_bad.json", score), "fixture-a",
                 "--allow-missing", "--force")


def test_a_reported_only_panel_is_stamped_ranked_false(root: Path) -> None:
    """§5.2 — `V_matched` exists so the V_→B_ step is readable and is never ranked. The row says so
    itself; the compiler then drops every order at that address."""
    score = fixture_score(root, "score_fixture_a.json")
    for panel, ranked in (("V_breadth", True), ("V_matched", False), ("B_", True)):
        add_path(root, score, "fixture-a", "--truth", "store", "--panel", panel,
                 "--scope", "held-out", "--allow-missing")
        row = json.loads((root / "rows" / BOARD / f"store.{panel}.held-out"
                          / "fixture-a@v1.json").read_text(encoding="utf-8"))
        assert row["ranked"] is ranked, panel
        assert row["address"]["panel"] == panel
    views = lb.compile_leaderboard(root)["boards"][BOARD]["views"]
    matched = views["store.V_matched.held-out"]
    assert matched["ranking"]["state"] == "unranked"
    assert matched["rows"][0]["ranked"] is False
    assert matched["rows"][0]["rank"] is None
    assert matched["rows"][0]["metrics"]["pval"]["mse"] == 1.0, "the number itself stays"
    assert views["store.V_breadth.held-out"]["rows"][0]["ranked"] is True


# ------------------------------------------------- a real candi.bench.external score json ---
#
# Everything above runs on hand-written fixtures. This section stamps the thing `add` will really
# be given: a score json `candi.bench.external` produced over a real store, with §5.2's `panels`,
# §4's `genome_wide` block, and both truths of §6. It is what makes "the address is a lookup" a
# fact about the scorer's own output rather than about `panelled()`.

REAL_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq"),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "DNase-seq"),
    "B_bb": ("ATAC-seq", "H3K4me3"),
}
REAL_ASSAY = "H3K4me3"


@pytest.fixture(scope="module")
def real_scores(tmp_path_factory) -> dict:
    """`{truth: path}` — two score jsons written by `candi.bench.external`'s own CLI.

    Built from `tests/test_bench_external.py`'s own pieces: its §4.1 prediction manifest, its
    `kind: "truth"` manifest, the real store writer and the real regime shape. The files are
    written by `external.main`, so they are byte-for-byte what a Fir scoring pass hands `add` —
    including `null` where a metric is undefined, which is the spelling `jsonable` uses.

    One `V_` target and one `B_` target, so all three panels of §5.2 carry a number, and chr2 of
    two chromosomes is held out, so §4's `genome_wide` block exists.
    """
    np = pytest.importorskip("numpy", reason="a real score json needs the scorer")
    external = pytest.importorskip("candi.bench.external", reason="the scorer needs candi")
    from candi.bench.harness import Pair
    from tests.test_bench_external import MANIFEST, TRUTH_MANIFEST
    from tests.test_store_reader import N_BINS, make_store
    from tests.test_store_regime import regime_dict

    tmp = tmp_path_factory.mktemp("lbreal")
    store = make_store(tmp / "s", tracks=REAL_TRACKS)
    regime = tmp / "regime.json"
    regime.write_text(json.dumps(regime_dict(
        store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "B_bb"]},
        kinds=["counts", "peaks", "pval"], train_chroms=[], eval_chroms=["chr1", "chr2"],
        eval_pairs=[["T_aa", "V_aa"], ["T_bb", "B_bb"]])), encoding="utf-8")

    pred, truth = tmp / "pred", tmp / "truth"
    for d, man in ((pred, MANIFEST), (truth, {**TRUTH_MANIFEST, "chroms": ["chr1", "chr2"]})):
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    rng = np.random.default_rng(4)
    for pair in (Pair("T_aa", "V_aa"), Pair("T_bb", "B_bb")):
        # the §4.1 directory name is the scorer's own, never spelled out here
        name = external.track_dirname(pair, REAL_ASSAY)
        (pred / name).mkdir()
        (truth / name).mkdir()
        for c, n in N_BINS.items():
            np.savez(pred / name / f"{c}.npz",
                     signal_mu=rng.gamma(1.0, 2.0, n).astype(np.float32),
                     signal_sigma=np.full(n, 0.7, np.float32),
                     mu=rng.gamma(2.0, 3.0, n).astype(np.float32),
                     n=np.full(n, 5.0, np.float32),
                     peak_score=rng.random(n).astype(np.float32))
            np.savez(truth / name / f"{c}.npz",
                     signal_mu=rng.gamma(1.0, 2.0, n).astype(np.float32))

    # `v_only` and `b_only` are the two passes the real programme runs: `V_` and `B_` are predicted
    # and scored separately, off panel-derived regimes and separate prediction roots. They are what
    # makes the unfilled `V_matched` block a fact about the pipeline, not a hand-edited json.
    roots = {"store": pred, "challenge": pred}
    for name, pair, ev in (("v_only", Pair("T_aa", "V_aa"), ["V_aa"]),
                           ("b_only", Pair("T_bb", "B_bb"), ["B_bb"])):
        (tmp / f"regime.{name}.json").write_text(json.dumps(regime_dict(
            store, biosamples={"train": ["T_aa", "T_bb"], "eval": ev},
            kinds=["counts", "peaks", "pval"], train_chroms=[], eval_chroms=["chr1", "chr2"],
            eval_pairs=[[pair.input_biosample, pair.target_biosample]])),
            encoding="utf-8")
        panel_root = tmp / f"pred.{name}"
        panel_root.mkdir()
        (panel_root / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        shutil.copytree(pred / external.track_dirname(pair, REAL_ASSAY),
                        panel_root / external.track_dirname(pair, REAL_ASSAY))
        roots[name] = panel_root

    out = {"truth_root": truth}
    for name in ("store", "challenge", "v_only", "b_only"):
        path = tmp / f"score.{name}.json"
        cfg = regime if name in ("store", "challenge") else tmp / f"regime.{name}.json"
        argv = ["--store", str(cfg), "--pred", str(roots[name]), "--out", str(path),
                "--held-out-chroms", "chr2", "--c-index-pairs", "2000", "--quiet"]
        if name == "challenge":
            argv += ["--truth-root", str(truth)]
        assert external.main(argv) == 0
        out[name] = path
    return out


@pytest.fixture
def real_root(root: Path, real_scores: dict) -> Path:
    """The fixture leaderboard root — site copied in, anchor frozen to the real truth manifest,
    and both regimes pointed at the real regime json the pass actually read."""
    sha = lb.sha256_file(real_scores["truth_root"] / "manifest.json")

    def edit(b: dict) -> None:
        b["anchor"]["frozen"]["regime_sha256"] = sha
        for board in b["boards"].values():
            board["eval_set"]["regime"] = "regime.json"

    edit_boards(root, edit)
    return with_site(root)


def real_score(real_scores: dict, truth: str) -> dict:
    return json.loads(real_scores[truth].read_text(encoding="utf-8"))


def point_the_board_at(root: Path, regime_name: str) -> None:
    """Name the regime file a pass actually read, on every board of `root`.

    A panel pass reads a DERIVED regime (`tools/declare_eval_pairs.py split` writes
    `regime.<name>.<panel>.json`), and `gate_row_against_board` matches the row's recorded regime
    against `eval_set.regime` by basename — so the two must be the same file, not two spellings.
    """
    edit_boards(root, lambda b: [board["eval_set"].update(regime=regime_name)
                                 for board in b["boards"].values()])


def test_the_address_reads_the_panel_and_scope_the_scorer_wrote(real_root: Path,
                                                                real_scores: dict) -> None:
    """§1, §4, §5.2 — four addresses over ONE score json, and each row carries exactly the numbers
    that address names. Nothing is recomputed: every value is compared back to the source block."""
    score = real_score(real_scores, "store")
    path = real_scores["store"]
    for panel, scope in (("V_breadth", "held-out"), ("V_matched", "held-out"),
                         ("B_", "held-out"), ("B_", "genome-wide")):
        add_path(real_root, path, "CANDI", "--truth", "store", "--panel", panel,
                 "--scope", scope, "--lineage", "candi", "--allow-missing")
        row = json.loads((real_root / "rows" / BOARD / f"store.{panel}.{scope}"
                          / "CANDI@v1.json").read_text(encoding="utf-8"))
        block = score if scope == "held-out" else score["genome_wide"]
        key = lb.PANEL_JSON_KEY[panel]
        for arm in ("pval", "count"):
            for metric, val in row["metrics"].get(arm, {}).items():
                assert block["panels"][arm][key][metric] == val, (panel, scope, arm, metric)
        assert row["metrics"]["pval"]["mse"] == block["panels"]["pval"][key]["mse"]
        # the count arm rides on the same lookup, companions and all (registry companion rule)
        assert set(row["metrics"]["count"]) >= {"crps", "crps_oracle_scaled", "scale_error"}
        assert row["metrics"]["count"]["crps"] == block["panels"]["count"][key]["crps"]
        assert row["ranked"] is (panel != "V_matched" and scope == "held-out")
    # the four rows are four different numbers, not one number stamped four times
    seen = {(p, s): json.loads((real_root / "rows" / BOARD / f"store.{p}.{s}"
                                / "CANDI@v1.json").read_text())["metrics"]["pval"]["mse"]
            for p, s in (("V_breadth", "held-out"), ("B_", "held-out"), ("B_", "genome-wide"))}
    assert len(set(seen.values())) == 3, seen
    assert lb.main(["--root", str(real_root), "check"]) == 0


def test_a_real_challenge_row_carries_neither_the_count_nor_the_peak_arm(real_root: Path,
                                                                        real_scores: dict) -> None:
    """§7 — under challenge truth `panel_macros` writes a count block with no metrics in it and the
    scorer withholds every peak-derived key. The row carries none of them: absent, not zero."""
    score = real_score(real_scores, "challenge")
    assert score["panels"]["count"]["B"]["n_experiments"] == 0
    path = real_scores["challenge"]
    add_path(real_root, path, "CANDI", "--truth", "challenge", "--panel", "B_",
             "--scope", "held-out", "--lineage", "candi", "--allow-missing")
    row = json.loads((real_root / "rows" / BOARD / "challenge.B_.held-out"
                      / "CANDI@v1.json").read_text(encoding="utf-8"))
    assert "count" not in row["metrics"]
    assert "auprc" not in row["metrics"]["pval"] and "peak_base_rate" not in row["metrics"]["pval"]
    assert row["metrics"]["pval"]["mse"] == score["panels"]["pval"]["B"]["mse"]
    assert "count/crps" in row["missing_metrics"] and "pval/auprc" in row["missing_metrics"]
    assert row["provenance"]["truth_manifest_hash"] == \
        score["provenance"]["truth"]["manifest_sha256"]


def test_a_real_entrant_score_json_stamps_an_anchor_row(real_root: Path,
                                                        real_scores: dict) -> None:
    """§6 — the anchor block, end to end: one entrant's own challenge-truth score json, gated on
    the sha256 of the truth root's manifest.json rather than on a regime."""
    add_entrant(real_root, real_scores["challenge"], "Lavawizard")
    row = json.loads((real_root / lb.ANCHOR / "challenge.B_.held-out"
                      / "Lavawizard@2019-submission.json").read_text(encoding="utf-8"))
    assert row["address"]["regime"] is None and row["lineage"] == "entrant"
    assert row["markers"] == [], "the 2019 submission is not the rival we retrained"
    assert row["provenance"]["truth_manifest_hash"] == lb.sha256_file(
        real_scores["truth_root"] / "manifest.json")
    assert lb.main(["--root", str(real_root), "check"]) == 0


def test_add_refuses_a_V_matched_panel_no_B_pass_ever_filled(real_root: Path,
                                                             real_scores: dict) -> None:
    """§5.2 — the matched assay set is MEASURED from the pass's own `B_` rows. The real programme
    scores `V_` and `B_` in separate passes, so a `V_` json's `V_matched` block comes out empty:
    no experiments, no `matched_to`. That is no number, not a number of zero, and `add` refuses it
    and names the verb that fixes it. The same file's `V_breadth` is untouched and still lands."""
    point_the_board_at(real_root, "regime.v_only.json")
    score = real_score(real_scores, "v_only")
    matched = score["panels"]["pval"]["V_matched"]
    assert matched["matched_to"] == [] and matched["n_experiments"] == 0
    assert score["panels"]["pval"]["V_breadth"]["n_experiments"] > 0
    with pytest.raises(SystemExit, match="fill-panels") as e:
        add_path(real_root, real_scores["v_only"], "CANDI", "--truth", "store",
                 "--panel", "V_matched", "--scope", "held-out", "--lineage", "candi",
                 "--allow-missing")
    assert "matched_to" in str(e.value) and "no scored experiments" in str(e.value)
    assert not (real_root / "rows" / BOARD / "store.V_matched.held-out").exists()
    add_path(real_root, real_scores["v_only"], "CANDI", "--truth", "store",
             "--panel", "V_breadth", "--scope", "held-out", "--lineage", "candi",
             "--allow-missing")
    row = json.loads((real_root / "rows" / BOARD / "store.V_breadth.held-out"
                      / "CANDI@v1.json").read_text(encoding="utf-8"))
    assert row["metrics"]["pval"]["mse"] == score["panels"]["pval"]["V_breadth"]["mse"]
    assert "panels_from" not in row["provenance"], "this pass measured its own panels"


def test_a_filled_V_matched_row_records_where_its_panels_came_from(real_root: Path,
                                                                   real_scores: dict) -> None:
    """`fill-panels` re-measures `panels` over the union of a `V_` and a `B_` pass's `per_track`
    and records `provenance.panels_from`. `add` then stamps the matched cell — unranked (§5.2) —
    and carries that provenance onto the row, so a reader is told the middle number's `B_` rows
    came from a sibling pass. The union here is `harness.panel_macros`' own, not a hand-written
    block: the same function `fill-panels` calls, over the same two jsons."""
    point_the_board_at(real_root, "regime.v_only.json")
    panel_macros = pytest.importorskip("candi.bench.harness").panel_macros
    v, b = real_score(real_scores, "v_only"), real_score(real_scores, "b_only")
    filled = json.loads(json.dumps(v))
    union = {**v["per_track"], **b["per_track"]}
    filled["panels"] = {arm: panel_macros(union, arm) for arm in ("pval", "count")}
    filled["provenance"]["panels_from"] = {
        "v": str(real_scores["v_only"]), "b": str(real_scores["b_only"]),
        "tool": "candi.bench.external fill-panels"}
    path = real_root.parent / "filled.json"
    path.write_text(json.dumps(filled), encoding="utf-8")  # NOT panelled(): these are real panels
    add_path(real_root, path, "CANDI", "--truth", "store", "--panel", "V_matched",
             "--scope", "held-out", "--lineage", "candi", "--allow-missing")
    row = json.loads((real_root / "rows" / BOARD / "store.V_matched.held-out"
                      / "CANDI@v1.json").read_text(encoding="utf-8"))
    assert row["ranked"] is False, "§5.2 — the matched panel is reported, never ranked"
    assert row["provenance"]["panels_from"]["tool"] == "candi.bench.external fill-panels"
    assert row["metrics"]["pval"]["mse"] == filled["panels"]["pval"]["V_matched"]["mse"]
    assert filled["panels"]["pval"]["V_matched"]["matched_to"] == [REAL_ASSAY]
    compiled = lb.compile_leaderboard(real_root)["boards"][BOARD]["views"][
        "store.V_matched.held-out"]
    assert compiled["ranking"]["state"] == "unranked"
    assert compiled["rows"][0]["provenance"]["panels_from"]["v"].endswith("score.v_only.json")


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

