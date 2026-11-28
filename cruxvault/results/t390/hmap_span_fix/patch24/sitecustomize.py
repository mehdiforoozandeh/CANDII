"""Bit-exact check only (runall.sh puts this folder first on PYTHONPATH): every python process sets
`hmap.caller.SPAN_EXTRA` and `hmap.formula.SPAN_EXTRA` back to 24, the pre-fix value."""
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
for _p in (_TOOLS / "t118", _TOOLS / "t125"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from hmap import caller as _c, formula as _f  # noqa: E402

_c.SPAN_EXTRA = 24
_f.SPAN_EXTRA = 24
