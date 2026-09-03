"""t87 — `tools/declare_eval_pairs.py split`: cutting a declared pairing down to one panel.

Two same-named pairing tools used to ship. The retired one held a baked-in
`{"validation": "V_", "test": "B_"}` table and wrote one regime per split; this verb takes the
prefix as an argument, like `declare` does, and D16/D31 are the reason.

The claim these tests defend is not that prefix matching works. It is that a derived regime carries
ONE panel. A held-out target reaching a regime that drives best-checkpoint selection would put the
test set inside the loop, and nothing downstream could get it back out. The second claim is the one
that only shows up on a cluster: the derived file lives beside the RUN, not beside the source, so a
relative `regions.bed` (D32 resolves it against the regime file's own directory) would resolve
somewhere else or nowhere, and the training scope would silently move.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import tools.declare_eval_pairs as tool                                  # noqa: E402
from tools.declare_eval_pairs import main as tool_main, pairs_on_panel   # noqa: E402

from candi.store.regime import Regime                                    # noqa: E402

PILOT = REPO / "configs" / "regime.eic_pilot.json"

ASSAYS = ["DNase-seq", "H3K4me3", "H3K27ac"]
PAIRS = [["T_K562", "V_K562"], ["T_HepG2", "V_HepG2"],
         ["T_K562", "B_K562"], ["T_lonely", "B_lonely"]]


def _regime(tmp_path: Path, **over) -> Path:
    """A minimal loadable regime. `Regime.from_file` needs store/assays/context_bins and no store."""
    obj = {
        "_comment": "a fixture",
        "store": "/nowhere/CANDI_STORE/eic",
        "assays": ASSAYS,
        "biosamples": {"train": ["T_K562", "T_HepG2", "T_lonely"],
                       "eval": ["V_K562", "V_HepG2", "B_K562", "B_lonely"]},
        "eval_pairs": PAIRS,
        "context_bins": 768,
        "train_chroms": ["chr1", "chr2"],
        "eval_chroms": ["chr20", "chr21", "chr22"],
        "window_plan": {"type": "tile", "stride_bins": 768, "min_valid_frac": 0.9},
        "dsf": {"policy": "discrete", "levels": [1, 2]},
        "kinds": ["counts", "peaks"],
        "seed": 42,
    }
    obj.update(over)
    p = tmp_path / "regime.src.json"
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return p


def _split(src: Path, panel: str, out: Path) -> int:
    return tool_main(["split", "--regime", str(src), "--panel", panel, "--out", str(out)])


# ---------------------------------------------------------------------------------------------
# the cut itself
# ---------------------------------------------------------------------------------------------

def test_the_target_end_decides_the_panel_not_the_prompt() -> None:
    """`T_K562` prompts both panels, so a rule reading the input would keep everything."""
    kept, dropped = pairs_on_panel([tuple(p) for p in PAIRS], "B_")
    assert kept == [("T_K562", "B_K562"), ("T_lonely", "B_lonely")]
    assert dropped == [("T_K562", "V_K562"), ("T_HepG2", "V_HepG2")]


def test_no_other_panel_can_reach_a_derived_regime(tmp_path) -> None:
    """The separation that keeps a held-out panel clean. A leak here is invisible until publication."""
    out = tmp_path / "derived.V_.json"
    assert _split(_regime(tmp_path), "V_", out) == 0
    obj = json.loads(out.read_text())
    assert [t for _, t in obj["eval_pairs"]] == ["V_K562", "V_HepG2"]
    assert not any(t.startswith("B_") for _, t in obj["eval_pairs"])
    assert not any(b.startswith("B_") for b in obj["biosamples"]["eval"])


def test_the_eval_pool_is_exactly_the_kept_targets(tmp_path) -> None:
    """A biosample in `eval` that is in no pair is a target the loader would slot and never score."""
    out = tmp_path / "derived.B_.json"
    _split(_regime(tmp_path), "B_", out)
    obj = json.loads(out.read_text())
    assert obj["biosamples"]["eval"] == [t for _, t in obj["eval_pairs"]] == ["B_K562", "B_lonely"]


def test_the_training_split_and_every_other_key_survive_verbatim(tmp_path) -> None:
    src = _regime(tmp_path)
    out = tmp_path / "derived.json"
    _split(src, "V_", out)
    before, after = json.loads(src.read_text()), json.loads(out.read_text())
    for k in ("store", "assays", "context_bins", "train_chroms", "eval_chroms",
              "window_plan", "dsf", "kinds", "seed"):
        assert after[k] == before[k], k
    assert after["biosamples"]["train"] == before["biosamples"]["train"]


def test_a_panel_that_keeps_nothing_exits_2_and_writes_no_file(tmp_path) -> None:
    """An empty `eval_pairs` reads as 'declared, none' and would silently disable the eval."""
    out = tmp_path / "nope.json"
    assert _split(_regime(tmp_path), "Z_", out) == 2
    assert not out.exists()


def test_a_regime_declaring_no_pairs_exits_2_rather_than_inventing_one(tmp_path) -> None:
    out = tmp_path / "nope.json"
    assert _split(_regime(tmp_path, eval_pairs=[]), "V_", out) == 2
    assert not out.exists()


def test_the_panel_is_required_and_has_no_default(tmp_path) -> None:
    """D31's rule applied to the cut: a default panel would make the choice inferred."""
    with pytest.raises(SystemExit) as e:
        tool_main(["split", "--regime", str(_regime(tmp_path)),
                   "--out", str(tmp_path / "x.json")])
    assert e.value.code == 2                                    # argparse's "required" exit
    assert not (tmp_path / "x.json").exists()


def test_the_comment_says_it_is_derived_and_by_what(tmp_path) -> None:
    """Every consumer of a derived regime reads this line first; the launchers grep for it."""
    out = tmp_path / "derived.json"
    _split(_regime(tmp_path), "V_", out)
    c = json.loads(out.read_text())["_comment"]
    assert c.startswith("DERIVED by tools/declare_eval_pairs.py split")
    assert "--panel V_" in c and "regime.src.json" in c


# ---------------------------------------------------------------------------------------------
# the write itself — one path, many writers (K16)
# ---------------------------------------------------------------------------------------------
# `--out` is a SHARED path. `slurm/t81_predict_candi.sh` derives the panel regime into
# `$WORKSPACE/regime.<name>.<panel>.json`, the same name for every task of a sharded array, because
# each shard's manifest records `regime` and the shards of one root must not differ in it. So the
# 12 tasks that start together all run this verb over the same destination, and a `write_text`
# there is a truncate followed by a fill: on 2026-09-03 six of the twelve re-read a sibling's
# half-written file and exited 1. The launcher's path is not the bug and does not move; the write
# is. Every writer produces identical bytes, so making the swap indivisible is the whole fix.
#
# Both checks are deterministic. A race is not reproducible on demand, but the rename is: patch
# `os.replace` and the ordering is fixed by the patch rather than by the scheduler.

def test_the_derived_file_lands_by_rename_never_by_writing_the_shared_path(tmp_path,
                                                                           monkeypatch) -> None:
    """The destination is only ever touched by one `os.replace`, from a temp file beside it."""
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((Path(src), Path(dst), Path(dst).exists()))
        real_replace(src, dst)

    monkeypatch.setattr(tool.os, "replace", spy)
    out = tmp_path / "workspace" / "regime.eic_19.B_.json"
    assert _split(_regime(tmp_path), "B_", out) == 0
    assert len(seen) == 1, f"the split renamed {len(seen)} times, expected exactly 1"
    src, dst, existed_before = seen[0]
    assert dst == out
    assert src != out, "the split wrote the shared path directly; a reader can see it truncated"
    assert src.parent == out.parent, (
        f"the temp file is in {src.parent}, not beside the destination -- a rename across a "
        f"filesystem is a copy, and a copy is not atomic")
    assert not existed_before, "nothing reached the destination before the rename"
    assert not list(out.parent.glob(".*.tmp")), "the temp file was left behind"
    Regime.from_file(out)


def test_a_second_writer_to_the_same_out_never_exposes_a_partial_file(tmp_path,
                                                                     monkeypatch) -> None:
    """What a concurrent reader gets at the last instant before writer two lands: writer one, whole.

    This is the shard case in one process: two `split` runs, same `--out`, and the destination is
    a loadable regime at every point where the second one could be interrupted.
    """
    src_regime = _regime(tmp_path)
    out = tmp_path / "regime.shared.json"
    assert _split(src_regime, "B_", out) == 0
    first_bytes = out.read_bytes()

    observed = []
    real_replace = os.replace

    def spy(tmp_name, dst):
        observed.append(json.loads(Path(dst).read_text(encoding="utf-8")))
        real_replace(tmp_name, dst)

    monkeypatch.setattr(tool.os, "replace", spy)
    assert _split(src_regime, "B_", out) == 0
    assert len(observed) == 1
    assert [t for _, t in observed[0]["eval_pairs"]] == ["B_K562", "B_lonely"], (
        "the shared path held something other than a complete derived regime mid-write")
    assert out.read_bytes() == first_bytes, (
        "two writers of the same panel disagreed on the bytes -- then which one wins matters")


# ---------------------------------------------------------------------------------------------
# the derived file lives somewhere else — which is what the absolute BED is for
# ---------------------------------------------------------------------------------------------

def test_the_bed_is_rewritten_absolute_so_the_derived_file_can_live_anywhere(tmp_path) -> None:
    """D32 resolves `regions.bed` against the REGIME's directory. Beside the run, that is wrong."""
    out = tmp_path / "elsewhere" / "regime.eic_pilot.B_.json"
    assert _split(PILOT, "B_", out) == 0
    obj = json.loads(out.read_text())
    bed = obj["regions"]["bed"]
    assert Path(bed).is_absolute() and Path(bed).is_file()
    assert obj["regions"]["sha256"] == json.loads(PILOT.read_text())["regions"]["sha256"]
    Regime.from_file(out)                       # the sha256 is checked at load; it must still pass


def test_a_regime_with_no_regions_key_does_not_grow_one(tmp_path) -> None:
    out = tmp_path / "derived.json"
    _split(_regime(tmp_path), "V_", out)
    assert "regions" not in json.loads(out.read_text())


# ---------------------------------------------------------------------------------------------
# the shipped corpus — the numbers the launchers are sized against
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("panel,n", [("V_", 26), ("B_", 12)])
def test_the_shipped_pilot_regime_splits_26_and_12(tmp_path, panel, n) -> None:
    out = tmp_path / f"regime.eic_pilot.{panel}.json"
    assert _split(PILOT, panel, out) == 0
    obj = json.loads(out.read_text())
    assert len(obj["eval_pairs"]) == n == len(obj["biosamples"]["eval"])
    assert all(t.startswith(panel) for _, t in obj["eval_pairs"])
    assert len(json.loads(PILOT.read_text())["eval_pairs"]) == 38


def test_the_two_panels_of_a_regime_partition_its_declared_pairs(tmp_path) -> None:
    """26 + 12 = 38, and no pair is in both. The cut loses nothing and duplicates nothing."""
    got = []
    for panel in ("V_", "B_"):
        out = tmp_path / f"{panel}.json"
        _split(PILOT, panel, out)
        got.append([tuple(p) for p in json.loads(out.read_text())["eval_pairs"]])
    v, b = got
    assert not set(v) & set(b)
    assert set(v) | set(b) == {tuple(p) for p in json.loads(PILOT.read_text())["eval_pairs"]}
