"""t125-P2i: the chain's pairs split across tasks, with no change to any number.

`score.py chain --pair-shard K/N` scores only the pairs i with i % N == K that have no kept
`pair_<i>.json` and writes nothing else; `--assemble-only` scores nothing and, once every pair is
kept, writes scores.json, merge_error.json and CHAIN_DONE. Sharded (N = 3) then assembled must
equal the one-task run: the same file set, CHAIN_DONE and merge_error.json byte for byte, every
scores.json equal but for its timestamp and timing. No two shards score the same pair; a kept pair
is never scored again.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hmap import score  # noqa: E402
from test_t125_h_score import S, _h_dir  # noqa: E402,F401  (the shared synthetic set-up)

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
N_SHARDS = 3
MIX = 4                       # small, so the own-noise h merges and merge_error.json is written
VOLATILE = ("created_utc", "timing")


def _specs(S):
    return ["formula:bin", str(S["h_mlp"]), str(S["h_noise"])]


def _run(S, out, **kw) -> Path:
    return score.score_chain(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], S["f_dir"],
                             out, _specs(S), workers=1, control_rl_tsv=S["rl"], device="cpu",
                             mix_groups=MIX, **kw)


def _files(d: Path) -> set:
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


def _no_scoring(pair):
    raise AssertionError(f"pair {pair['source_pid']} -> {pair['target_pid']} scored again")


REAL_CHAIN_JOB = score.chain_job


def _spy(calls: list):
    def spy(pair):
        calls.append(score._CTX["pred"].train_pairs.index(pair))
        return REAL_CHAIN_JOB(pair)
    return spy


@pytest.fixture(scope="module")
def runs(S, tmp_path_factory):
    root = tmp_path_factory.mktemp("shard")
    seq = _run(S, root / "seq")
    calls = {}
    with pytest.MonkeyPatch.context() as mp:
        for k in range(N_SHARDS):
            calls[k] = []
            mp.setattr(score, "chain_job", _spy(calls[k]))
            fd = _run(S, root / "sharded", pair_shard=(k, N_SHARDS))
        after_shards = _files(fd)
        mp.setattr(score, "chain_job", _no_scoring)
        _run(S, root / "sharded", assemble_only=True)
    name = S["f_dir"].name
    return {"seq": seq, "fd": fd, "calls": calls, "after_shards": after_shards, "root": root,
            "part_seq": root / "seq" / f"{name}.partial",
            "part_sh": root / "sharded" / f"{name}.partial"}


def test_shards_score_every_pair_once(runs):
    n = len(list(runs["part_seq"].glob("pair_*.json")))
    assert n >= 2 * N_SHARDS, "too few pairs for the split to mean anything"
    got = [i for k in range(N_SHARDS) for i in runs["calls"][k]]
    assert sorted(got) == list(range(n)) and len(set(got)) == len(got)   # disjoint, complete
    for k in range(N_SHARDS):
        assert runs["calls"][k] and all(i % N_SHARDS == k for i in runs["calls"][k])


def test_shards_write_only_pair_files(runs):
    assert runs["after_shards"] == set()       # no scores.json, merge_error.json or CHAIN_DONE
    assert {p.name for p in runs["part_sh"].iterdir()} == {p.name for p in
                                                            runs["part_seq"].iterdir()}


def _scores(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*/scores.json")):
        obj = json.loads(p.read_text())
        for k in VOLATILE:
            obj.pop(k)
        out[p.parent.name] = obj
    return out


def assert_same_outputs(a: Path, b: Path):
    assert _files(a) == _files(b)
    assert "merge_error.json" in _files(a) and "CHAIN_DONE" in _files(a)
    for f in ("CHAIN_DONE", "merge_error.json"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    sa, sb = _scores(a), _scores(b)
    assert sa.keys() == sb.keys() and len(sa) == 3
    for name in sa:
        # bit-equal: the serialised records, not a tolerance
        assert json.dumps(sa[name]["records"]) == json.dumps(sb[name]["records"]), name
        assert sa[name] == sb[name], name


def test_sharded_then_assembled_equals_the_one_task_run(runs):
    assert_same_outputs(runs["seq"], runs["fd"])


def test_one_task_run_over_kept_pairs_just_assembles(S, runs, tmp_path):
    name = S["f_dir"].name
    shutil.copytree(runs["part_sh"], tmp_path / f"{name}.partial")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "chain_job", _no_scoring)
        fd = _run(S, tmp_path)
    assert_same_outputs(runs["seq"], fd)


def test_a_kept_pair_is_never_scored_again(S, runs, tmp_path):
    name = S["f_dir"].name
    part = tmp_path / f"{name}.partial"
    shutil.copytree(runs["part_seq"], part)
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in part.iterdir()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "chain_job", _no_scoring)
        for k in range(N_SHARDS):
            _run(S, tmp_path, pair_shard=(k, N_SHARDS))
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in part.iterdir()} == before


def test_a_shard_scores_only_its_missing_pairs(S, runs, tmp_path):
    name = S["f_dir"].name
    part = tmp_path / f"{name}.partial"
    shutil.copytree(runs["part_seq"], part)
    for i in (1, 2, 4):
        (part / f"pair_{i:04d}.json").unlink()
    calls = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "chain_job", _spy(calls))
        _run(S, tmp_path, pair_shard=(1, N_SHARDS))
    assert calls == [1, 4]
    assert (part / "pair_0001.json").is_file() and not (part / "pair_0002.json").exists()


def test_assemble_with_a_pair_missing_writes_nothing(S, runs, tmp_path):
    name = S["f_dir"].name
    part = tmp_path / f"{name}.partial"
    shutil.copytree(runs["part_seq"], part)
    (part / "pair_0001.json").unlink()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "chain_job", _no_scoring)
        fd = _run(S, tmp_path, assemble_only=True)
    assert not fd.exists() or _files(fd) == set()


def test_shard_arguments(tmp_path, monkeypatch):
    got = {}

    def fake(*a, **k):
        got.update(k)
        return tmp_path
    monkeypatch.setattr(score, "score_chain", fake)
    base = ["chain", "m", "c", "d", "ctl", "bl", "f_run", str(tmp_path), "--h", "formula:bin"]
    score.main(base)
    assert got["pair_shard"] is None and got["assemble_only"] is False
    score.main(base + ["--pair-shard", "2/3"])
    assert got["pair_shard"] == (2, 3)
    score.main(base + ["--assemble-only"])
    assert got["assemble_only"] is True and got["pair_shard"] is None
    for bad in (["--pair-shard", "3/3"], ["--pair-shard", "x"], ["--pair-shard", "-1/3"],
                ["--pair-shard", "0/3", "--assemble-only"]):
        with pytest.raises(SystemExit):
            score.main(base + bad)


def test_shard_and_assemble_exclude_each_other(tmp_path):
    with pytest.raises(ValueError):
        score.score_chain("m", "c", "d", "ctl", "bl", "f", tmp_path, [], pair_shard=(0, 2),
                          assemble_only=True)
