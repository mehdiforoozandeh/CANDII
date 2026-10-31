"""t117 — the pure parts of `tools/t117/pseudoreps.py`: which read set each product splits, the
task tables, the recorded-spr parser and the half names. The cluster parts (spr, MACS2, binning)
are proven by the smoke and the checks on Nibi, not here."""
from __future__ import annotations

import collections
import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "t117" / "pseudoreps.py"
spec = importlib.util.spec_from_file_location("pseudoreps", TOOL)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def test_readsets_and_halves_cover_all_130_products():
    rs, hs = P.task_tables(P.all_rows())
    assert len(hs) == 260 and len({(h["pid"], h["half"]) for h in hs}) == 260
    assert [h["index"] for h in hs] == [str(i) for i in range(260)]
    # 7 base + 21 depth + 12 abproxy + 40 FASTQ-route read sets
    assert len(rs) == 80
    assert collections.Counter(r["route"] for r in rs) == {"bam": 40, "fastq": 40}
    users = [p for r in rs for p in r["users"].split(",")]
    assert sorted(users) == sorted({h["pid"] for h in hs})


def test_treatment_unchanged_arms_reuse_the_base_read_set():
    rows = {r["pid"]: r for r in P.all_rows()}
    for pid, r in rows.items():
        rid = P.readset_of(r)
        if r["arm"] in ("base", "ratio", "ctlid", "ctldepth", "extsize"):
            assert rid == f"{r['track']}__base__base"
        else:
            assert rid == pid


def test_spr_flags_reads_the_recorded_command():
    se = ("set -e\npython3 $(which encode_task_spr.py) \\\n    /x/A.30M.tagAlign.gz \\\n"
          "    --pseudoreplication-random-seed 0")
    assert P.spr_flags(se) == {"seed": 0, "paired_end": False}
    assert P.spr_flags(se + " \\\n    --paired-end") == {"seed": 0, "paired_end": True}


def test_half_names_follow_encode_task_spr():
    assert (P.pr_name("/a/ENCFF254LWX.merged.srt.nodup.30M.tagAlign.gz", "pr1")
            == "ENCFF254LWX.merged.srt.nodup.30M.pr1.tagAlign.gz")
