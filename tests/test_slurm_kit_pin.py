"""t39 — every cluster job must run the checkout it was submitted from, not the shared one.

`candi_venv` on Fir carries an editable install whose `.pth` is a HARD PATH to the shared kit's
`src`. So `$KIT` and `cd "$KIT"` select which SCRIPTS run while the LIBRARY still comes from that
one clone, parked on whatever branch someone pinned for some other job.

It surfaced when t31's calibration ran this branch's tool against a `candi.eval` three branches old
and died on an ImportError -- because the branch had ADDED a function. Had it merely CHANGED one,
the job would have produced numbers and nothing in the output would have said which code made them.
That is why this is a test and not a note: the loud version of the failure was luck.

These are text checks on the scripts, deliberately. The real assertion lives inside
`slurm/_kit_pin.sh`, which refuses to run when `candi.__file__` is not under `$KIT`; what a test
here can add is that every job actually goes through it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
GUARD = SLURM / "_kit_pin.sh"
#: Scripts that pin PYTHONPATH themselves, in the form they already used before the guard existed.
SELF_PINNED = ('PYTHONPATH="$KIT/src"', "PYTHONPATH=src")


def _jobs():
    return sorted(p for p in SLURM.glob("*.sh") if p.name != GUARD.name)


def test_there_are_job_scripts_to_check():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert len(_jobs()) >= 10


@pytest.mark.parametrize("job", _jobs(), ids=lambda p: p.name)
def test_every_job_pins_candi_to_its_own_checkout(job: Path):
    text = job.read_text(encoding="utf-8")
    assert ("_kit_pin.sh" in text or any(f in text for f in SELF_PINNED)), (
        f"{job.name} neither sources slurm/_kit_pin.sh nor pins PYTHONPATH itself, so it would "
        f"import candi from the shared kit whatever $KIT says")


@pytest.mark.parametrize("job", [p for p in _jobs() if "_kit_pin.sh" in p.read_text()],
                         ids=lambda p: p.name)
def test_the_guard_is_sourced_after_the_venv_is_active(job: Path):
    """It imports `candi`, so it needs the interpreter it is guarding -- ordering is load-bearing."""
    text = job.read_text(encoding="utf-8")
    assert text.index("bin/activate") < text.index('source "$KIT/slurm/_kit_pin.sh"')


def test_the_guard_sets_pythonpath_and_then_verifies_it():
    """Setting it is not enough. The failure being prevented is one nobody would notice."""
    g = GUARD.read_text(encoding="utf-8")
    assert 'export PYTHONPATH="$KIT/src"' in g
    assert "import candi" in g and "candi.__file__" in g
    assert "sys.exit(" in g, "the guard must be fatal, not a warning"


def test_the_guard_runs_before_anything_is_allocated():
    """A job that discovers the wrong library halfway through has already spent the GPU."""
    g = GUARD.read_text(encoding="utf-8")
    assert "|| exit 1" in g
