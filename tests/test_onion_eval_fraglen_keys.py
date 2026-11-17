"""FL review fix 1: `fraglen_table.sha256` is a comparability key of select_rule and arm_report.

Pins: a score with `--fraglen-table` and one without are refused as a mix, with a message that
names the option and says signal-space values differ; a score made before the key existed reads
None, equal to a score without the option; `--allow-mix fraglen_table.sha256` is accepted and
warns that signal-space values differ; two different tables are a mix too."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import arm_report as AR  # noqa: E402
from onion_eval import select_rule as SR  # noqa: E402

KEY = "fraglen_table.sha256"


def _s(path, fl=None, **kw):
    s = {"__path__": path, "evalset_sha256": "e" * 64, "tables_sha256": "t" * 64,
         "scorer_version": 2, "scorer_settings": {"device": "cpu", "infer_batch": 8}}
    if fl is not None:
        s["fraglen_table"] = {"path": "/x.tsv", "sha256": fl}
    s.update(kw)
    return s


def _items(*scores):
    return [(s["__path__"], SR.scorer_keys(s)) for s in scores]


def test_key_listed_and_read():
    assert KEY in SR.SCORER_KEYS and KEY in SR.MIXABLE
    assert SR.scorer_keys(_s("a"))[KEY] is None                       # old score: no table
    assert SR.scorer_keys(_s("a", fl="f" * 64))[KEY] == "f" * 64


def test_old_score_equals_flag_absent():
    SR.check_scorer_keys(_items(_s("old"), _s("new_without_flag", fraglen_table=None)))


def test_mix_refused_with_message():
    with pytest.raises(SR.Refused) as e:
        SR.check_scorer_keys(_items(_s("a"), _s("b", fl="f" * 64)))
    msg = str(e.value)
    assert KEY in msg and "--fraglen-table" in msg and "signal-space values differ" in msg
    with pytest.raises(SR.Refused):
        SR.check_scorer_keys(_items(_s("a", fl="f" * 64), _s("b", fl="g" * 64)))
    SR.check_scorer_keys(_items(_s("a", fl="f" * 64), _s("b", fl="f" * 64)))


def test_allow_mix_names_the_key(capsys):
    SR.check_scorer_keys(_items(_s("a"), _s("b", fl="f" * 64)), allow=(KEY,))
    assert "signal-space values differ" in capsys.readouterr().err
    with pytest.raises(SR.Refused):            # the other keys are still checked
        SR.check_scorer_keys(_items(_s("a"), _s("b", fl="f" * 64, tables_sha256="u" * 64)),
                             allow=(KEY,))


def test_arm_report_sees_the_key(capsys):
    arms = {"x": [_s("x/seed0")], "y": [_s("y/seed0", fl="f" * 64)]}
    with pytest.raises(AR.Refused) as e:
        AR.check_arm_keys(arms)
    assert KEY in str(e.value)
    AR.check_arm_keys(arms, allow=(KEY,))
    assert "signal-space values differ" in capsys.readouterr().err


def test_signal_pileup_is_a_comparability_key(capsys):
    """E2: `signal_pileup` (absent = inbin) is refused as a mix unless --allow-mix names it."""
    PK = "signal_pileup"
    assert SR.scorer_keys(_s("old"))[PK] == "inbin"
    SR.check_scorer_keys(_items(_s("old"), _s("new", signal_pileup="inbin")))
    with pytest.raises(SR.Refused, match="signal_pileup.*count-space values do not depend"):
        SR.check_scorer_keys(_items(_s("old"), _s("new", signal_pileup="box_d")))
    SR.check_scorer_keys(_items(_s("old"), _s("new", signal_pileup="box_d")), allow=(PK,))
    assert "signal_pileup" in capsys.readouterr().err
