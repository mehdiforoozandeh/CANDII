"""What a SEED change moves, measured with the instrument you are actually quoting.

`AGENTS.md` §7.2 says a seed change moves pooled imputation CRPS by **0.1195**, Spearman by
0.0562, ECE by 0.0354. Those were measured with `candi.eval`, since deleted (D15). `candi.bench` is a different
instrument on a different population — whole chromosomes, a track-mean rather than an assay-mean —
and there is no reason its seed sensitivity should be the same number. Quoting the old floor beside
a new number is the exact failure §7.2 exists to prevent, one level up.

So: train the same recipe at two seeds, score both, and read the spread off.

    python tools/seed_floor.py run_s0.bench.json run_s1.bench.json
    python tools/seed_floor.py run_s0.json run_s1.json --suite eval

**One floor per panel, because they are not one exam.** A `candi.bench` score json carries three
panel aggregations of the same scored pass (`plan/BENCHMARK_DESIGN.md` §5.2): `V_breadth` over the
22 assays the `V_` cells pose, `B` over the 8 the `B_` cells pose, and `V_matched` — the same `V_`
tracks restricted to `B_`'s assay set. They do not have the same resolution: fewer, differently
chosen tracks per panel means a different mean over a different population, so a spread measured on
`V_breadth` is not the floor a `B` number has to clear. `--panel` reads one of the three; without it
the tool reads the whole-pass `macro`, as it always has.

    python tools/seed_floor.py s0.store.V_.json s1.store.V_.json --suite bench --panel V_breadth
    python tools/seed_floor.py s0.store.V_.json s1.store.V_.json --suite bench --panel B \
        --scope genome-wide

**What this can and cannot say.** Two seeds give ONE paired difference, not a distribution — the
same standing as §7.2's own numbers, which is why they are comparable at all. It is a magnitude,
not an interval. Three or more seeds give a range, and an sd over three draws, and the tool prints
both — but neither is a confidence interval and this tool will not call one that.

The per-track block matters more than the macro block. A macro that holds still while its tracks
swing by an order of magnitude more is a macro whose stability is an averaging artefact, and the
number a reader should weigh an arm-vs-arm claim against is the track spread.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # HISTORICAL: the h5-era `candi.eval` keys, for reading an ARCHIVED run json. Nothing writes
    # `M1` any more (D15); `--suite bench` above is the live path.
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

#: The three panel aggregations of one scoring pass, spelled as `harness.PANELS` spells them.
PANELS = ("V_breadth", "V_matched", "B")

#: The two scopes of one scoring pass. `held-out` is the ranked one and sits at the top of the score
#: json; `genome-wide` sits under `genome_wide` and is emitted only when the run scored more
#: chromosomes than it holds out.
SCOPES = ("held-out", "genome-wide")

#: What must be quoted WITH a `crps`, per arm — `EVAL.md` "Rules for quoting any number" 4 for the
#: count arm, and the board's own companion rule for the pval arm. All of them or none of the three:
#: a raw CRPS alone cannot say whether the model got the shape wrong or only the scale.
CRPS_COMPANIONS = {
    "count": ("crps_oracle_scaled", "scale_error"),
    "pval": ("pit_ks", "coverage_95"),
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


def panel_of(target_biosample: str) -> Optional[str]:
    """Which panel a scored track belongs to, from its TARGET cell's prefix.

    `harness.panel_of`, restated here so the tool reads a finished score json without importing
    `candi` (and so it can read a json written by a version of the harness this checkout does not
    have). Anything that is neither `V_` nor `B_` — a self-paired denoise record, typically —
    belongs to no panel.
    """
    if target_biosample.startswith("V_"):
        return "V"
    if target_biosample.startswith("B_"):
        return "B"
    return None


def panel_keys(vals: List[Dict[str, Any]], arm: str
               ) -> Tuple[List[Tuple[str, List[float]]], List[str]]:
    """The keys of one arm's panel block worth tabulating, in reading order, and what was dropped.

    `crps` and its companions lead, because a reader who sees the raw number first and the split
    three rows down has already formed the wrong impression. Every other numeric key that EVERY run
    carries follows, scores before track counts — a key one run has and another lacks has no
    spread, and printing it would compare a number against nothing.

    The drop is `EVAL.md` rule 4 held mechanically: a `crps` whose companions are missing takes the
    companions' silence with it rather than travelling alone.
    """
    shared: Optional[set] = None
    for b in vals:
        here = {k for k, v in b.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        shared = here if shared is None else (shared & here)
    shared = shared or set()
    lead = ("crps",) + CRPS_COMPANIONS.get(arm, ())
    dropped: List[str] = []
    if "crps" in shared and not all(c in shared for c in CRPS_COMPANIONS.get(arm, ())):
        dropped = [k for k in lead if k in shared]
        shared = shared - set(lead)
        lead = ()
    rest = shared - set(lead)
    # Counts of WHAT WAS AGGREGATED, not scores. They belong in the table -- `EVAL.md` rule 8 says
    # check `n_tracks` before comparing two panels, and a panel that lost a track between seeds is
    # exactly what a floor reader needs to see -- but at the BOTTOM of it. Sorted in among the
    # metrics they bury three real numbers under seven copies of the same track count.
    counts = {k for k in rest
              if k.endswith("_n_tracks") or k in ("n_tracks", "n_experiments", "n_points")}
    order = [k for k in lead if k in shared] + sorted(rest - counts) + sorted(counts)
    return [(k, [float(b[k]) for b in vals]) for k in order], dropped


def _panel_section(docs: List[Dict[str, Any]], names: List[str], panel: str, pre: str,
                   scope: str) -> List[str]:
    """The headline block when `--panel` is given: one table per arm, off `panels[arm][panel]`."""
    L = [f"## Panel `{panel}` — {scope} scope\n",
         "The three panels aggregate DIFFERENT track populations out of one scored pass, so each "
         "has its own resolution and a spread measured here is the floor for this panel's numbers "
         "and no other. `V_breadth` → `V_matched` is the exam changing, `V_matched` → `B` is the "
         "generalization gap, and `V_breadth` is never subtracted from `B` (§5.3's reading rule). "
         "`V_matched` itself is not ranked.\n"]
    for a in ("count", "pval"):
        blocks = [_get(d, f"{pre}panels.{a}.{panel}") for d in docs]
        if any(not isinstance(b, dict) for b in blocks):
            L.append(f"**`{a}` arm: absent from at least one run, so there is nothing to "
                     f"compare.** A count arm is absent by design under challenge truth, which "
                     f"carries signal only.\n")
            continue
        if all(float(b.get("n_experiments", 0)) == 0 for b in blocks):
            L.append(f"**`{a}` arm: panel `{panel}` scored no experiments in any run.**\n")
            continue
        rows, dropped = panel_keys(blocks, a)
        if not rows:
            L.append(f"**`{a}` arm: no numeric key is carried by every run.**\n")
            continue
        L.append(f"### `{panel}` · `{a}` arm\n")
        # MEASURED, never listed. Which assays this panel actually posed is read off the blocks in
        # front of us: a hard-coded "22 and 8" would go stale the first time the panel moved, and
        # would go stale silently, which is the failure `panel_macros` itself refuses to risk.
        seen = sorted({str(x) for b in blocks for x in (b.get("assays") or [])})
        L.append(f"{len(seen)} assays" + (f" — {', '.join(seen)}" if seen else "")
                 + ". Experiments per run: "
                 + ", ".join(f"`{n}` {int(b.get('n_experiments', 0))}"
                             for b, n in zip(blocks, names)) + ".\n")
        if dropped:
            L.append("`" + "`, `".join(dropped) + "` " + ("is" if len(dropped) == 1 else "are")
                     + " WITHHELD here: `crps` travels with "
                     + " and ".join(f"`{c}`" for c in CRPS_COMPANIONS[a])
                     + " or it does not travel (`EVAL.md` rule 4). Not every run carries the "
                       "whole set, so none of the three is printed.\n")
        col = "paired \\|Δ\\|" if len(docs) == 2 else "range"
        cols = ["key", *names, col] + (["sd"] if len(docs) >= 3 else [])
        L += ["| " + " | ".join(cols) + " |", "|---|" + "---|" * (len(cols) - 1)]
        for key, vals in rows:
            cells = [f"`{key}`"] + [f"{v:.5f}" for v in vals] + [f"**{spread(vals)}**"]
            if len(docs) >= 3:
                cells.append(f"{statistics.stdev(vals):.5f}")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    while L and L[-1] == "":                    # the per-track block supplies its own blank line
        L.pop()
    return L


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="two or more result JSONs, same recipe, different seed")
    p.add_argument("--suite", choices=["bench", "eval"], default=None,
                   help="default: detected from the first file")
    p.add_argument("--arm", choices=["impute", "denoise", "all"], default="impute",
                   help="which tracks the per-track block covers. `impute` by default, because "
                        "bench keeps both arms in one map and denoising is the easier task -- a "
                        "median over the union is not comparable to eval.py's imputation-only one.")
    p.add_argument("--panel", choices=list(PANELS), default=None,
                   help="read `panels[arm][PANEL]` instead of the whole-pass `macro`. The three "
                        "panels of plan/BENCHMARK_DESIGN.md 5.2 aggregate different track "
                        "populations -- V_ poses 22 assays, B_ poses 8 -- so each has its own "
                        "seed floor and one panel's spread is not another's.")
    p.add_argument("--scope", choices=list(SCOPES), default=SCOPES[0],
                   help="`held-out` (default) reads the ranked block at the top of the score json; "
                        "`genome-wide` reads the same three blocks under `genome_wide`, which "
                        "exists only when the run scored more chromosomes than it held out.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("two seeds at minimum; one run has no spread to report")

    docs = [json.loads(Path(r).read_text()) for r in args.runs]
    arm = args.arm
    suite = args.suite or detect(docs[0])
    names = [Path(r).stem for r in args.runs]
    pre = "" if args.scope == SCOPES[0] else "genome_wide."

    # REFUSE, do not shrug. A `--panel` the file cannot answer must not come back as an empty table:
    # an empty table reads as "the seeds agreed", which is the one wrong answer a floor tool can
    # give. The eval-era jsons predate the panels entirely, so name that separately.
    if args.panel and suite != "bench":
        raise SystemExit(
            f"--panel {args.panel} needs a `candi.bench` score json; a `--suite {suite}` file "
            f"carries no `panels` block at all (the three panels of plan/BENCHMARK_DESIGN.md 5.2 "
            f"postdate it). Drop --panel to read that file's macro block.")
    for name, doc in zip(names, docs):
        if pre and not isinstance(doc.get("genome_wide"), dict):
            raise SystemExit(
                f"`{name}` carries no `genome_wide` block, so --scope genome-wide has nothing to "
                f"read. A run that scored exactly the chromosomes it holds out has ONE scope, and "
                f"`provenance.scope.genome_wide_computed` says so.")
        if args.panel and not isinstance(_get(doc, f"{pre}panels"), dict):
            raise SystemExit(
                f"`{name}` carries no `{pre}panels` block, so --panel {args.panel} has nothing to "
                f"read. Score it with a `candi.bench` that emits "
                f"`panels[arm][V_breadth|V_matched|B]`, or drop --panel to read `macro`.")

    L = [f"# What a seed change moves — `candi.{suite}`\n",
         f"{len(docs)} runs of one recipe at different seeds: " + ", ".join(f"`{n}`" for n in names)]
    if args.panel or pre:
        # The address the numbers below came from, and what the spread column IS, in the header --
        # not in a footnote. A panel floor quoted against the wrong panel is the failure this
        # option exists to prevent, and a reader who skims one line must still see which panel.
        if len(docs) == 2:
            standing = f"**{len(docs)} seeds: a paired |Δ|, not an interval.**"
        else:
            standing = (f"**{len(docs)} seeds: a range and an sd over {len(docs)} draws, "
                        f"not an interval.**")
        address = f"{pre}panels[arm][{args.panel}]" if args.panel else f"{pre}macro[arm]"
        L += ["", f"{standing} Read from `{address}`, the {args.scope} scope."]
    L += ["",
          "Two seeds give ONE paired difference, not a distribution. That is the same standing as "
          "`AGENTS.md` §7.2's own seed numbers, which is what makes the two comparable — and it is "
          "a magnitude, never an interval.\n"]
    if args.panel:
        L += _panel_section(docs, names, args.panel, pre, args.scope)
    else:
        L += ["## Headline scalars\n",
              "| key | " + " | ".join(names) + " | spread |",
              "|---|" + "---|" * (len(names) + 1)]
        for label, path in KEYS[suite]:
            vals = [_get(d, f"{pre}{path}") for d in docs]
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in vals):
                continue
            L.append(f"| {label} | " + " | ".join(f"{v:.5f}" for v in vals)
                     + f" | **{spread(vals)}** |")

    block, pick = PER_TRACK[suite]
    per = [_get(d, f"{pre}{block}") or {} for d in docs]
    common = sorted(set(per[0]).intersection(*[set(x) for x in per[1:]])) if per[0] else []
    # SPLIT BY ARM. bench keeps imputation and denoising in one `per_track` map, and denoising is
    # the easier task with the smaller spread and usually the larger count -- on the t22 panel, 26
    # denoise tracks against 12 impute. A median over the union is a median of the denoise arm
    # wearing both names, and it is not comparable to eval.py's, which is imputation only.
    if arm == "impute":
        common = [k for k in common if not k.endswith("|denoise")]
    elif arm == "denoise":
        common = [k for k in common if k.endswith("|denoise")]
    # AND SPLIT BY PANEL when one was asked for. Leaving the per-track block over every track while
    # the headline shows one panel is the worse of the two errors available here: the reader would
    # weigh a `B` macro against a track floor that is mostly `V_` tracks. Membership is read off the
    # track key's TARGET cell, exactly as `harness.panel_of` reads it.
    if args.panel:
        matched = set()
        if args.panel == "V_matched":
            # `B_`'s assay set, MEASURED from the B_ rows these runs share -- never a listed set,
            # which would go stale silently the first time the panel moved (`harness.panel_macros`
            # measures it the same way, and for the same reason).
            for k in common:
                fields = k.split("|")
                if len(fields) >= 3 and panel_of(fields[1]) == "B":
                    matched.add(str(pick(per[0][k]).get("assay", fields[2])))
        keep = []
        for k in common:
            fields = k.split("|")
            if len(fields) < 3:
                continue
            side = panel_of(fields[1])
            assay = str(pick(per[0][k]).get("assay", fields[2]))
            if (args.panel == "B" and side == "B") or (args.panel == "V_breadth" and side == "V") \
                    or (args.panel == "V_matched" and side == "V" and assay in matched):
                keep.append(k)
        common = keep
    rows = []
    for k in common:
        recs = [pick(x[k]) for x in per]
        vals = [r.get("crps") for r in recs]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        # AGENTS.md 7.2: raw CRPS never travels without its split, and a seed floor is the one
        # place the split earns its keep hardest -- a track whose RAW number doubles while its
        # oracle-scaled number barely moves has changed its scale, not its ranking.
        extra = {}
        for f in ("crps_oracle_scaled", "scale_error"):
            fv = [r.get(f) for r in recs]
            extra[f] = fv if all(isinstance(v, (int, float)) for v in fv) else None
        rows.append((k, vals, extra))
    if rows:
        where = f" on panel `{args.panel}`" if args.panel else ""
        L += ["", f"## Per track — CRPS across {len(rows)} `{arm}` tracks common to every run"
                  f"{where}\n",
              "The macro is a mean of these. A macro that holds still while its tracks swing is a "
              "macro whose stability is an averaging artefact, and the track spread is what an "
              "arm-vs-arm claim has to clear.\n",
              "The last two columns are the split `AGENTS.md` §7.2 requires beside any raw CRPS, "
              "and here they do real work: a track whose raw spread is large while its "
              "oracle-scaled spread is small moved its SCALE, not its ranking, and the two are "
              "not the same failure.\n",
              "| track | " + " | ".join(names) + " | spread | oracle-scaled sp. | scale-error sp. |",
              "|---|" + "---|" * (len(names) + 3)]
        sp = []
        def _sp(v):
            return abs(v[1] - v[0]) if len(v) == 2 else max(v) - min(v)
        for k, vals, extra in rows:
            d = _sp(vals)
            sp.append(d)
            os_ = extra["crps_oracle_scaled"]
            se_ = extra["scale_error"]
            L.append(f"| {k.replace('|', chr(92) + '|')} | "
                     + " | ".join(f"{v:.5f}" for v in vals)
                     + f" | {d:.5f} | {('%.5f' % _sp(os_)) if os_ else '—'} | "
                     + f"{('%.5f' % _sp(se_)) if se_ else '—'} |")
        worst = rows[sp.index(max(sp))]
        L += ["", f"**Track spread: median {statistics.median(sp):.5f}, "
                  f"max {max(sp):.5f} on `{worst[0]}`.** "
                  f"The macro spread above is a mean of this column, so it is smaller by "
                  f"construction; a claim about a single track has to clear the track number, not "
                  f"the macro one.\n"]
        os_w = worst[2]["crps_oracle_scaled"]
        if os_w and _sp(os_w) < max(sp) / 2:
            L.append(f"On that worst track the oracle-scaled spread is only {_sp(os_w):.5f}, so "
                     f"most of the {max(sp):.5f} is SCALE and not ranking. Quoting the raw number "
                     f"alone would read as the model falling apart on a seed change when what "
                     f"moved was its calibration -- which is the whole reason §7.2 forbids it.\n")

    text = "\n".join(L)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
