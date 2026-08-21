"""What a SEED change moves, measured with the instrument you are actually quoting.

`AGENTS.md` §7.2 says a seed change moves pooled imputation CRPS by **0.1195**, Spearman by
0.0562, ECE by 0.0354. Those were measured with `candi.eval`. `candi.bench` is a different
instrument on a different population — whole chromosomes, a track-mean rather than an assay-mean —
and there is no reason its seed sensitivity should be the same number. Quoting the old floor beside
a new number is the exact failure §7.2 exists to prevent, one level up.

So: train the same recipe at two seeds, score both, and read the spread off.

    python tools/seed_floor.py run_s0.bench.json run_s1.bench.json
    python tools/seed_floor.py run_s0.json run_s1.json --suite eval

**What this can and cannot say.** Two seeds give ONE paired difference, not a distribution — the
same standing as §7.2's own numbers, which is why they are comparable at all. It is a magnitude,
not an interval. Three or more seeds give a range and the tool prints it, but a range over three
draws is still not a confidence interval and this tool will not call it one.

The per-track block matters more than the macro block. A macro that holds still while its tracks
swing by an order of magnitude more is a macro whose stability is an averaging artefact, and the
number a reader should weigh an arm-vs-arm claim against is the track spread.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Headline scalars, per suite. `bench` keys are dotted paths; `eval` keys likewise.
KEYS = {
    "bench": [
        ("macro count CRPS", "macro.count.crps"),
        ("  ... oracle-scaled", "macro.count.crps_oracle_scaled"),
        ("  ... scale error", "macro.count.scale_error"),
        ("macro count gwspear", "macro.count.gwspear"),
        ("macro count gwcorr", "macro.count.gwcorr"),
        ("macro count mse", "macro.count.mse"),
        ("macro count ECE", "macro.count.ece"),
        ("macro count C-index", "macro.count.c_index"),
        ("macro count coverage95", "macro.count.coverage_95"),
        ("macro count AUPRC", "macro.count.auprc"),
        ("macro pval CRPS", "macro.pval.crps"),
        ("macro pval gwcorr", "macro.pval.gwcorr"),
        ("macro denoise CRPS", "macro_denoise.count.crps"),
        ("macro denoise gwspear", "macro_denoise.count.gwspear"),
    ],
    "eval": [
        ("imp macro CRPS", "M1.imp_macro_crps"),
        ("  ... oracle-scaled", "M1.imp_macro_crps_oracle_scaled"),
        ("  ... scale error", "M1.imp_macro_scale_error"),
        ("imp macro Spearman", "M1.imp_macro_spearman_raw"),
        ("imp pooled CRPS", "M1.imp.crps"),
        ("imp pooled ECE", "M1.imp.ece"),
        ("den macro CRPS", "M1.den_macro_crps"),
        ("den macro Spearman", "M1.den_macro_spearman_raw"),
    ],
}

#: Where each suite keeps its per-track scores, and under what field.
PER_TRACK = {
    "bench": ("per_track", lambda rec: (rec.get("count") or {})),
    "eval": ("M1.imp_per_target", lambda rec: rec),
}


def _get(d: Any, dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def detect(doc: Dict[str, Any]) -> str:
    return "bench" if "per_track" in doc else "eval"


def spread(vals: List[float]) -> str:
    """`|b - a|` for a pair, `max - min` for more. Never called a confidence interval."""
    if len(vals) == 2:
        return f"{abs(vals[1] - vals[0]):.5f}"
    return f"{max(vals) - min(vals):.5f}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="two or more result JSONs, same recipe, different seed")
    p.add_argument("--suite", choices=["bench", "eval"], default=None,
                   help="default: detected from the first file")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("two seeds at minimum; one run has no spread to report")

    docs = [json.loads(Path(r).read_text()) for r in args.runs]
    suite = args.suite or detect(docs[0])
    names = [Path(r).stem for r in args.runs]

    L = [f"# What a seed change moves — `candi.{suite}`\n",
         f"{len(docs)} runs of one recipe at different seeds: " + ", ".join(f"`{n}`" for n in names),
         "",
         "Two seeds give ONE paired difference, not a distribution. That is the same standing as "
         "`AGENTS.md` §7.2's own seed numbers, which is what makes the two comparable — and it is "
         "a magnitude, never an interval.\n",
         "## Headline scalars\n",
         "| key | " + " | ".join(names) + " | spread |",
         "|---|" + "---|" * (len(names) + 1)]
    for label, path in KEYS[suite]:
        vals = [_get(d, path) for d in docs]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in vals):
            continue
        L.append(f"| {label} | " + " | ".join(f"{v:.5f}" for v in vals) + f" | **{spread(vals)}** |")

    block, pick = PER_TRACK[suite]
    per = [_get(d, block) or {} for d in docs]
    common = sorted(set(per[0]).intersection(*[set(x) for x in per[1:]])) if per[0] else []
    field = "crps"
    rows = []
    for k in common:
        vals = [pick(x[k]).get(field) for x in per]
        if all(isinstance(v, (int, float)) for v in vals):
            rows.append((k, vals))
    if rows:
        L += ["", f"## Per track — CRPS across {len(rows)} tracks common to every run\n",
              "The macro is a mean of these. A macro that holds still while its tracks swing is a "
              "macro whose stability is an averaging artefact, and the track spread is what an "
              "arm-vs-arm claim has to clear.\n",
              "| track | " + " | ".join(names) + " | spread |",
              "|---|" + "---|" * (len(names) + 1)]
        sp = []
        for k, vals in rows:
            s = (abs(vals[1] - vals[0]) if len(vals) == 2 else max(vals) - min(vals))
            sp.append(s)
            L.append(f"| {k.replace('|', chr(92) + '|')} | "
                     + " | ".join(f"{v:.5f}" for v in vals) + f" | {s:.5f} |")
        L += ["", f"**Track spread: median {statistics.median(sp):.5f}, "
                  f"max {max(sp):.5f} on `{rows[sp.index(max(sp))][0]}`.** "
                  f"The macro spread above is a mean of this column, so it is smaller by "
                  f"construction; a claim about a single track has to clear the track number, not "
                  f"the macro one.\n"]

    text = "\n".join(L)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
