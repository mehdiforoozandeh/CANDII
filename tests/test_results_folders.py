"""Tracked experiment folders under `cruxvault/results/` keep the runall contract (AGENTS.md §4.11).

The old repo's results grew to 132 MB because nothing capped them, so this repo ignored them
all, and every finding's evidence link broke on a fresh clone. Folders are now tracked one at a
time, once converted. These tests are what keep a converted folder small, rerunnable and frozen:

- no tracked file is over 1 MB;
- every tracked folder has `runall.sh` and `MANIFEST.tsv`;
- every `git` row of the manifest is tracked and still has its recorded sha256 (frozen);
- every finding's `evidence:` path inside a tracked folder is tracked.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = "cruxvault/results/"
MAX_BYTES = 1 << 20


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", RESULTS], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.split()


def _folders(tracked: list[str]) -> set[str]:
    return {p[len(RESULTS):].split("/", 1)[0] for p in tracked}


def test_no_tracked_result_file_over_cap() -> None:
    big = [(p, (ROOT / p).stat().st_size) for p in _tracked()
           if (ROOT / p).stat().st_size > MAX_BYTES]
    assert not big, f"over 1 MB; keep these on the cluster as MANIFEST.tsv rows: {big}"


def test_tracked_folders_have_runall_and_manifest() -> None:
    tracked = set(_tracked())
    for d in _folders(tracked):
        for f in ("runall.sh", "MANIFEST.tsv"):
            assert f"{RESULTS}{d}/{f}" in tracked, f"{RESULTS}{d}/ is tracked without {f}"


def test_manifest_git_rows_are_tracked_and_unchanged() -> None:
    tracked = set(_tracked())
    for d in _folders(tracked):
        man = ROOT / RESULTS / d / "MANIFEST.tsv"
        rows = [ln.split("\t") for ln in man.read_text().splitlines()[1:] if ln]
        for path, nbytes, sha, where in rows:
            if where != "git":
                continue
            rel = f"{RESULTS}{d}/{path}"
            assert rel in tracked, f"{rel} is a git row of {man.name} but is not tracked"
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            assert got == sha, f"{rel} changed since MANIFEST.tsv was written; the folder is frozen"


def test_finding_evidence_in_tracked_folders_is_tracked() -> None:
    tracked = set(_tracked())
    folders = _folders(tracked)
    for f in sorted((ROOT / "cruxvault").glob("f*.md")):
        line = next((ln for ln in f.read_text().splitlines() if ln.startswith("evidence:")), "")
        for ev in (e.strip() for e in line[len("evidence:"):].split(",")):
            if ev.startswith("results/") and ev.split("/")[1] in folders:
                assert f"cruxvault/{ev}" in tracked, f"{f.name} cites {ev}, which is not tracked"
