"""t125 part 2 — the map h from a predicted NB and the control to P = -log10 p.

**Import convention (pinned for every t125 module and test):** put `<repo>/tools/t118` AND
`<repo>/tools/t125` on `sys.path`, then `from ladder import ...`, `from eta import ...`,
`from hmap import ...`. Importing this package does that for `tools/t118` (and `tools/t125`), so
`from hmap import features` works once `tools/t125` itself is on the path; the CLI modules
(`controls.py`, `tasks.py`) set both up in script mode.

Modules of the data side (chunk t125-P2a): `caller` (the caller's constants and scaling),
`products` (per-product numbers), `features` (rung-1 and rung-5 feature arrays), `controls`
(the control-count cache), `tasks` (the h task tables).
"""
from __future__ import annotations

import sys
from pathlib import Path

_T125 = Path(__file__).resolve().parents[1]
for _p in (_T125.parent / "t118", _T125):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
