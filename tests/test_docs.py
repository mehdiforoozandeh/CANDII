"""The one doc with a size budget, and the reason it needs one.

`CLAUDE.md` is loaded into the agent's context on EVERY turn. `AGENTS.md`, `DATA.md`, `EVAL.md`
and `README.md` are not — they cost nothing until something opens them. So a line added to
CLAUDE.md is a line paid for on every request forever, and a line added to AGENTS.md is a line
paid for only when it is needed.

That asymmetry is invisible while writing, which is exactly why it needs a test. AGENTS.md was
small once; it is 510 lines now. Nothing stopped it, because nothing was watching. This is the
thing watching.

When this fails, the fix is almost never a bigger number. It is to move the new content into the
file that owns that subject and leave a router line in CLAUDE.md pointing at it. Raise the cap
only as a deliberate decision, not to make a red test go away.
"""
from __future__ import annotations

from pathlib import Path

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"

# A front door, not a spec. See the module docstring before changing this.
MAX_LINES = 120


def test_claude_md_stays_a_front_door() -> None:
    n = len(CLAUDE_MD.read_text(encoding="utf-8").splitlines())
    assert n <= MAX_LINES, (
        f"CLAUDE.md is {n} lines, over the {MAX_LINES}-line budget. It loads on every turn. "
        f"Move the new content into AGENTS.md / DATA.md / EVAL.md and leave a "
        f"router line here, rather than raising the cap."
    )


def test_claude_md_routes_to_every_companion_doc() -> None:
    """A router that has stopped naming a doc has stopped routing to it."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for doc in ("AGENTS.md", "DATA.md", "EVAL.md", "README.md", "STORE.md"):
        assert doc in text, f"CLAUDE.md no longer mentions {doc} — the routing table has a hole."
