"""CANDI inside the 2019 field — one run of the challenge ranker, and the figure it produces.

`plan/BENCHMARK_DESIGN.md` §6 lifts the 25 anchor entrants out of the ranked table into a separate
block, because we never trained them. That has a cost the board cannot pay off: CANDI and the 2019
submissions never share a ranking denominator, so **"CANDI would have placed Nth" cannot be read
off the board**. This tool computes that one number separately.

It is **a labelled figure, never a board row**, and the reason is mechanical rather than editorial:
the challenge ranker's stage 2 ranks *across methods within a cell*, so adding CANDI to the field
changes every entrant's rank denominator. That is the same reason §7 allows exactly one CANDI row.
A figure can carry "this is a second, differently-denominated ranking"; a board row cannot.

**One ranker.** The ranking is `candi.bench.ranking.aggregate_ranks` — the challenge's own
four-stage procedure and the only ranker in this repo. This file computes no placement of its own;
it loads score jsons, shapes them into that function's input contract, and formats what comes back.

**What it reads, and what it refuses.** The `pval` arm only: challenge truth is bigwig signal, so
there are no counts and no peak calls to score, and the count and peak arms are absent from a
`--truth-root` score json rather than NaN. It refuses a score json whose
`provenance.truth.source` is not `"challenge"` and one whose scored panel is not `B_` — the 51
blind experiments are the only place CANDI and an entrant were measured on the same exam.

**What it can and cannot say.** Three limits travel with the placement, and all three are printed
with it rather than left to a reader:

1. The ORDER reproduces the published ranking and the SCORES do not (archive `h71`/`h77`). The
   resolution limit is ~0.005 correlation units and 5 of 24 adjacent pairs in the reference invert
   on three or more of the ten chromosome subsets. A placement that separates two entries by less
   than that is not a placement.
2. The field has fewer distinct methods than it has rows — three groups of entrants submitted
   byte-identical tracks (`competitors/entrants/README.md` §7). Any "beats N methods" claim must be
   COUNTED over distinct submissions, never read off the row count. This tool counts it.
3. One bootstrap, not ten. A score json carries one number per experiment per measure; the
   challenge's ten position bootstraps were never re-drawn here, so stage 4's "second-best of ten
   bootstrap ranks" degenerates to the single bootstrap's rank. `n_bootstraps` in the output says so.

    python tools/candi_in_field.py --candi <CANDI challenge.B_.json> \
        --anchors /project/def-maxwl/mforooz/t81_scores/anchor \
        --out cruxvault/results/t81
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candi.bench import ranking                                    # noqa: E402
from candi.bench.eic import MEASURES                               # noqa: E402
from candi.bench.harness import panel_of                           # noqa: E402

#: A score json that is not challenge truth, or not the `B_` panel, exits with this.
EXIT_REFUSED = 3
#: Nothing to rank — no anchor file resolved, or a file that is not a score json.
EXIT_INPUT = 2

#: The panel this figure is about. `B_` is the challenge's 51 blind experiments (§6).
PANEL = "B"

#: `msevar` is excluded EVERYWHERE (`competitors/entrants/README.md` §1): no variance vector 001
#: tried got closer than median ratio 0.19, so eight of the nine measures are reported. Dropping it
#: here as well is what keeps this ranking on the same measure set as the entrant tables.
EXCLUDED_MEASURES: Tuple[str, ...] = ("msevar",)

#: The byte-identical entrant groups, verified by md5 2026-08-26 and recorded in
#: `competitors/entrants/README.md` §7 "Entrants that are not independent". THREE groups, not four:
#: the README's `CUImpute1`+`ICU` H3K27ac row has the same *members* as a subset of group A, so it
#: is recorded in group A's assay list rather than counted as a group of its own.
#:
#: Overridable with `--identical-groups <json>` so the count can be recomputed when the detector
#: (`placement_table.py`, which works from the score CSVs alone) flags a new tie — including a tie
#: involving one of our own methods, which would be a bug worth seeing.
IDENTICAL_GROUPS: Tuple[Dict[str, Any], ...] = (
    {"tag": "A",
     "members": ["CUImpute1", "CUWA", "ICU"],
     "assays": "all 3 broad marks, H3K4me3, ATAC-seq (26 broad-mark experiments byte-identical); "
               "CUImpute1 and ICU are additionally identical on H3K27ac",
     "evidence": "competitors/entrants/README.md §7, md5 2026-08-26"},
    {"tag": "B",
     "members": ["Avocado_p0", "ICU"],
     "assays": "H3K4me1 — ICU submitted the organizers' Avocado baseline tracks, byte-identical, "
               "for all 7 H3K4me1 experiments",
     "evidence": "competitors/entrants/README.md §7, md5 2026-08-26"},
    {"tag": "C",
     "members": ["Hongyang_Li_and_Yuanfang_Guan", "Hongyang_Li_and_Yuanfang_Guan_v1"],
     "assays": "ATAC-seq — identical, and the same single file (md5 50258e27ee7b) for all three "
               "ATAC cells, i.e. a prediction that cannot be cell-type specific",
     "evidence": "competitors/entrants/README.md §7, md5 2026-08-26"},
)

#: Copied from `plan/BENCHMARK_DESIGN.md` §7.3 / the `candi.bench.ranking` docstring, and BUILT FROM
#: the module's own constants so the sentence cannot drift away from the code that enforces it.
RESOLUTION_LIMIT_SENTENCE = (
    f"Ranking resolution limit: ≈ {ranking.RESOLUTION_LIMIT_CORR} correlation units; "
    f"{ranking.UNSEPARABLE_ADJACENT_PAIRS} of 24 adjacent pairs invert on ≥ 3 of the ten "
    f"chromosome subsets. A placement that separates two entries by less than that is not a "
    f"placement."
)

COUNTED_SENTENCE = ("Any “beats N methods” claim must be counted, never read off the "
                    "table.")


class FieldError(RuntimeError):
    """Nothing to rank — a missing file, an unreadable json, a malformed group list."""


class Refusal(FieldError):
    """The wrong truth or the wrong panel: the inputs are readable and must still not be ranked.

    Its own class rather than a message the CLI pattern-matches on, so the exit code is decided
    where the refusal is raised.
    """


# ---------------------------------------------------------------------------
# loading, and the two refusals
# ---------------------------------------------------------------------------

def load_score(path: Path | str, *, arm: str = "pval") -> Dict[str, Any]:
    """One score json, checked for truth and panel, reduced to `{method, cells}`.

    `cells[experiment][measure]` for the requested arm, over impute tracks only. The experiment id
    is the bench `track_key` (`T_cell|B_cell|assay`), which is what makes two methods' cells line
    up: the anchor prediction roots and the CANDI dump were both written from the same `B_`-derived
    regime, so the same experiment carries the same key in every file.
    """
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FieldError(f"{p}: not a readable score json — {exc}") from exc
    if not isinstance(obj, dict) or "per_track" not in obj:
        raise FieldError(f"{p}: no `per_track` block — this is not a candi.bench score json")

    prov = obj.get("provenance") or {}
    truth = (prov.get("truth") or {}).get("source")
    if truth != "challenge":
        raise Refusal(
            f"{p}: provenance.truth.source is {truth!r}, not 'challenge'. This figure places CANDI "
            f"in the 2019 field, and the 2019 field was scored against the organizers' own blind "
            f"truth. A store-truth row is a different measurement of the same experiments "
            f"(plan/BENCHMARK_DESIGN.md §6) and cannot be ranked against it.")

    method = prov.get("method")
    if not method:
        raise FieldError(f"{p}: provenance.method is empty — a ranked row needs a method name")

    cells: Dict[str, Dict[str, float]] = {}
    panels_seen: Dict[str, int] = {}
    for key, arms in (obj["per_track"] or {}).items():
        row = (arms or {}).get(arm)
        if row is None or row.get("kind") != "impute":
            continue
        fields = str(key).split("|")
        if len(fields) < 3:
            continue
        panel = panel_of(fields[1])
        panels_seen[str(panel)] = panels_seen.get(str(panel), 0) + 1
        cells[str(key)] = {m: float(row[m]) for m in MEASURES
                           if m in row and m not in EXCLUDED_MEASURES}

    off_panel = {k: v for k, v in panels_seen.items() if k != PANEL}
    if off_panel or PANEL not in panels_seen:
        raise Refusal(
            f"{p}: scored panel is {panels_seen or '{}'}, not {PANEL!r} alone. `B_` is the "
            f"challenge's 51 blind experiments and the only exam CANDI and a 2019 entrant both "
            f"sat; a `V_` row was never submitted to the challenge and under challenge truth is "
            f"excluded by ruling D3.")

    return {"method": str(method), "cells": cells, "path": str(p),
            "missing_tracks": list(prov.get("missing_tracks") or []),
            "allow_missing": bool(prov.get("allow_missing", False)),
            "truth_manifest_sha256": (prov.get("truth") or {}).get("manifest_sha256"),
            "pred_root": prov.get("pred_root")}


def resolve_anchors(spec: str) -> List[Path]:
    """`--anchors` is a directory of per-entrant score jsons, or a glob.

    A directory is read as the pinned anchor layout first
    (`<dir>/<entrant>/challenge.B_.json`) and falls back to `<dir>/*.json`, so both the Fir tree and
    a flat rsync of it work with no flag. Every resolved path is reported in the output, because
    "which 25 files" is the one thing a reader cannot reconstruct from the table.
    """
    if any(ch in spec for ch in "*?["):
        return sorted(Path(p) for p in _glob.glob(spec, recursive=True))
    d = Path(spec)
    if not d.is_dir():
        raise FieldError(f"--anchors {spec} is neither a glob nor a directory")
    found = sorted(d.glob(f"*/challenge.{PANEL}_.json"))
    return found or sorted(d.glob("*.json"))


# ---------------------------------------------------------------------------
# the ranker's input contract
# ---------------------------------------------------------------------------

def build_table(entries: Sequence[Mapping[str, Any]]
                ) -> Tuple[Dict[int, Dict[str, Dict[str, Dict[str, float]]]],
                           List[str], List[str]]:
    """`table[bootstrap][experiment][team][measure]`, with the usable measures decided by the data.

    A measure is kept only when EVERY team carries it in EVERY cell that team scored. Keeping a
    partly-present measure would not drop it quietly — `rank_within_cell` ranks a missing value
    LAST — so a team that happened to lose one key would take a rank penalty for the gap rather than
    for its predictions. Dropped measures are returned and printed, never swallowed.

    A team absent from an experiment is not repaired here: `aggregate_ranks` scores it
    `MISSING_SCORE` (0.5, equal to the cap), which is the challenge's own rule for an absent team.
    `UIOWA_Michaelson` is the real case — it never submitted `C38M18`.
    """
    experiments = sorted({e for ent in entries for e in ent["cells"]})
    candidates = [m for m in MEASURES if m not in EXCLUDED_MEASURES]
    keep: List[str] = []
    for m in candidates:
        if all(m in cell for ent in entries for cell in ent["cells"].values()):
            keep.append(m)
    dropped = [m for m in candidates if m not in keep]

    table: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {0: {}}
    for e in experiments:
        cell: Dict[str, Dict[str, float]] = {}
        for ent in entries:
            if e in ent["cells"]:
                cell[ent["method"]] = {m: ent["cells"][e][m] for m in keep}
        table[0][e] = cell
    return table, experiments, dropped


# ---------------------------------------------------------------------------
# the non-independence count
# ---------------------------------------------------------------------------

def normalise_groups(raw: Any) -> List[Dict[str, Any]]:
    """Accept either the constant's shape or a bare list of member lists."""
    if not isinstance(raw, list):
        raise FieldError("--identical-groups must hold a JSON list")
    out: List[Dict[str, Any]] = []
    for i, g in enumerate(raw):
        if isinstance(g, list):
            g = {"members": g}
        if not isinstance(g, dict) or not isinstance(g.get("members"), list) \
                or len(g["members"]) < 2:
            raise FieldError(f"--identical-groups entry {i} needs a `members` list of >= 2 names")
        out.append({"tag": str(g.get("tag") or chr(ord("A") + i)),
                    "members": [str(m) for m in g["members"]],
                    "assays": str(g.get("assays") or ""),
                    "evidence": str(g.get("evidence") or "")})
    return out


def count_distinct(names: Iterable[str], groups: Sequence[Mapping[str, Any]]
                   ) -> Tuple[int, List[List[str]]]:
    """How many DISTINCT submissions a set of names holds, and which names merged.

    Each group is a clique over its members; the answer is the number of connected components
    among the names given. Components rather than a per-group subtraction because the groups
    OVERLAP — `ICU` is in two of them — and subtracting per group would double-count it.

    This is a LOWER BOUND on the number of distinct methods, and deliberately so: the byte
    identities hold per assay, so treating `Avocado_p0` and `CUWA` as one method (they are linked
    only through `ICU`, on different marks) under-counts rather than over-counts. Under-counting is
    the safe direction for a "beats N" claim.
    """
    present = list(dict.fromkeys(names))
    parent = {n: n for n in present}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        members = [m for m in g["members"] if m in parent]
        for m in members[1:]:
            a, b = find(members[0]), find(m)
            if a != b:
                parent[b] = a
    comps: Dict[str, List[str]] = {}
    for n in present:
        comps.setdefault(find(n), []).append(n)
    merged = sorted((sorted(v) for v in comps.values() if len(v) > 1), key=lambda v: v[0])
    return len(comps), merged


# ---------------------------------------------------------------------------
# the markdown
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{nd}f}"


def render_md(*, result: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
              candi: str, experiments: Sequence[str], measures: Sequence[str],
              dropped: Sequence[str], groups: Sequence[Mapping[str, Any]],
              anchor_paths: Sequence[Path], candi_path: Path) -> str:
    order = sorted(result["final_rank"], key=lambda t: result["final_rank"][t])
    tag_of: Dict[str, List[str]] = {}
    for g in groups:
        for m in g["members"]:
            tag_of.setdefault(m, []).append(g["tag"])

    below = [t for t in order if result["final_rank"][t] > result["final_rank"][candi]]
    n_distinct, merged = count_distinct(below, groups)
    field_distinct, field_merged = count_distinct(order, groups)
    by_path = {e["method"]: e for e in entries}

    lines: List[str] = []
    lines.append("# CANDI inside the 2019 field")
    lines.append("")
    lines.append(f"One run of the challenge ranker over the `{PANEL}_` panel under **challenge "
                 f"truth**, with CANDI added to the {len(entries) - 1}-entrant anchor field. "
                 f"**A figure, never a board row** (`plan/BENCHMARK_DESIGN.md` §6): stage 2 of the "
                 f"ranker ranks across methods within a cell, so adding CANDI changes every "
                 f"entrant's rank denominator — which is why the board lifts the entrants into a "
                 f"separate anchor block and why this number lives here instead.")
    lines.append("")
    lines.append("## What was ranked")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| ranker | `candi.bench.ranking.aggregate_ranks` — the challenge's four-stage "
                 "procedure, and the only ranker in this repo |")
    lines.append("| truth | **challenge** (2019 organizers' blind bigwigs); "
                 "`provenance.truth.source == \"challenge\"` verified on every file |")
    lines.append(f"| panel | `{PANEL}_` — the 51 blind experiments |")
    lines.append("| arm | `pval` only. Challenge truth carries no counts and no peak calls, so the "
                 "count and peak arms are ABSENT from these score jsons rather than zero |")
    lines.append(f"| methods | {len(entries)} (1 CANDI + {len(entries) - 1} anchor entrants) |")
    lines.append(f"| distinct submissions in the field | **{field_distinct}** of "
                 f"{len(entries)} rows |")
    lines.append(f"| experiments | {len(experiments)} |")
    lines.append(f"| measures used | {len(measures)} — {', '.join(f'`{m}`' for m in measures)} |")
    lines.append(f"| measures dropped | "
                 f"{', '.join(f'`{m}`' for m in dropped) if dropped else 'none'}"
                 f" — `msevar` is excluded everywhere (`competitors/entrants/README.md` §1); "
                 f"anything else here was absent from at least one team's cell |")
    lines.append(f"| bootstraps | {result['n_bootstraps']} |")
    lines.append("")
    lines.append("**One bootstrap, not ten.** A score json carries one number per experiment per "
                 "measure — the challenge's ten position bootstraps were never re-drawn here — so "
                 "stages 1–3 run once and stage 4's *second-best of ten bootstrap ranks* "
                 "degenerates to that single bootstrap's rank. The cap `min(0.5, r/n)` of stage 3 "
                 "is applied exactly as the challenge applies it.")
    lines.append("")
    lines.append("## The resolution limit, quoted with the placement")
    lines.append("")
    lines.append(f"> {RESOLUTION_LIMIT_SENTENCE}")
    lines.append("")
    lines.append("The ORDER reproduces the published ranking (archive `h77`: 16 of 25 exact "
                 "published ranks, no team moving more than two places); the absolute SCORES do "
                 "not (archive `h71`, refuted at 3.514e-02 against a 1e-4 bar). Read the column "
                 "below as a placement, never as a score.")
    lines.append("")
    lines.append("## The ranked table")
    lines.append("")
    lines.append("| rank | method | ident. group | 2nd-best bootstrap rank | "
                 "mean capped score (lower is better) | experiments scored | |")
    lines.append("|---:|---|:---:|---:|---:|---:|---|")
    for t in order:
        is_candi = t == candi
        name = f"**{t}**" if is_candi else t
        n_exp = len(by_path[t]["cells"]) if t in by_path else 0
        lines.append(
            f"| {result['final_rank'][t]} | {name} | "
            f"{'/'.join(tag_of.get(t, [])) or ''} | "
            f"{_fmt(result['second_best_bootstrap_rank'][t], 0)} | "
            f"{_fmt(result['mean_bootstrap_score'][t])} | {n_exp} | "
            f"{'**← CANDI**' if is_candi else ''} |")
    lines.append("")
    lines.append(f"CANDI places **{result['final_rank'][candi]} of {len(entries)}**. "
                 f"{RESOLUTION_LIMIT_SENTENCE}")
    lines.append("")
    lines.append("A team absent from an experiment scores `MISSING_SCORE` = 0.5, which is exactly "
                 "the stage-3 cap — the challenge's own rule, and why `experiments scored` is on "
                 "the row. `UIOWA_Michaelson` is the real case: it never submitted `C38M18` "
                 "(`competitors/entrants/README.md` §7).")
    lines.append("")
    lines.append("## Non-independence in the field")
    lines.append("")
    lines.append(f"**Byte-identical entrant groups: {len(groups)}.** Verified by md5 on the "
                 f"downloaded tracks (`competitors/entrants/README.md` §7); the field therefore "
                 f"contains fewer distinct methods than it has rows.")
    lines.append("")
    lines.append("| group | members | identical on |")
    lines.append("|---|---|---|")
    for g in groups:
        lines.append(f"| {g['tag']} | {', '.join(f'`{m}`' for m in g['members'])} | "
                     f"{g['assays'] or '—'} |")
    lines.append("")
    lines.append(f"**{COUNTED_SENTENCE}** Counted for CANDI at rank "
                 f"{result['final_rank'][candi]}:")
    lines.append("")
    lines.append(f"- rows below CANDI: **{len(below)}**")
    lines.append(f"- **distinct submissions below CANDI: {n_distinct}** — the number to quote")
    if merged:
        lines.append("- collapsed below CANDI: "
                     + "; ".join("{" + ", ".join(f"`{m}`" for m in c) + "}" for c in merged))
    lines.append(f"- distinct submissions in the whole field: {field_distinct} of {len(entries)} "
                 f"rows"
                 + ("" if not field_merged else
                    " (collapsed: "
                    + "; ".join("{" + ", ".join(f"`{m}`" for m in c) + "}" for c in field_merged)
                    + ")"))
    lines.append("")
    lines.append("The count is a **lower bound** on distinct methods: the byte identities hold per "
                 "assay, and `ICU` links two groups on different marks, so merging their "
                 "connected component under-counts rather than over-counts. Under-counting is the "
                 "safe direction for a “beats N” claim.")
    lines.append("")
    lines.append("## Files read")
    lines.append("")
    lines.append(f"- CANDI: `{candi_path}`")
    for p in anchor_paths:
        lines.append(f"- anchor: `{p}`")
    gaps = [f"`{e['method']}` ({len(e['missing_tracks'])} missing)" for e in entries
            if e["missing_tracks"]]
    if gaps:
        lines.append("")
        lines.append("Panels with recorded holes (`provenance.missing_tracks`): " + ", ".join(gaps))
    lines.append("")
    lines.append("Figure: `candi_in_2019_field.svg`.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

#: The dataviz skill's EMPHASIS form: one accent hue for the series that is the point, the
#: de-emphasis ink for the context rows. Accent = categorical slot 1, stepped per mode
#: (`#2a78d6` light / `#3987e5` dark); both clear every check of `validate_palette.js` against
#: their own surface. Greys are the palette's own ink tokens, so nothing here is eyeballed.
_SVG_CSS = """
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --accent: #2a78d6; --context: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --accent: #3987e5; --context: #c3c2b7;
  }
}
.surface { fill: var(--surface); }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.sub { fill: var(--ink2); font-size: 11px; }
.cap { fill: var(--ink2); font-size: 10.5px; }
.name { fill: var(--ink2); font-size: 11px; }
.name-hi { fill: var(--ink); font-size: 11px; font-weight: 600; }
.rank { fill: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.tick { fill: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.axlabel { fill: var(--ink2); font-size: 10.5px; }
.val { fill: var(--ink); font-size: 10px; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.ring { fill: none; stroke: var(--surface); stroke-width: 2; }
.dot { fill: var(--context); }
.dot-hi { fill: var(--accent); }
.key { fill: var(--context); }
.key-hi { fill: var(--accent); }
.tagbox { fill: none; stroke: var(--axis); stroke-width: 1; }
.tag { fill: var(--ink2); font-size: 8.5px; font-weight: 600; }
"""


def _nice_ticks(lo: float, hi: float) -> List[float]:
    span = max(hi - lo, 1e-9)
    for step in (0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2):
        if span / step <= 6.0:
            break
    start = (int(lo / step)) * step
    out: List[float] = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 6))
        v += step
    return out or [round(lo, 6), round(hi, 6)]


#: Advance width of one character, as a fraction of the font size, for the system sans at these
#: sizes. Used to MEASURE before placing: an SVG has no layout engine, so a label that is not
#: measured is a label that silently overflows the canvas or collides with the next column. The
#: figure is a report artefact, not a live page — nothing reflows it after the fact.
_CHAR_W = 0.56


def _text_w(s: str, size: float) -> float:
    return len(s) * size * _CHAR_W


def _wrap(text: str, width_px: float, size: float) -> List[str]:
    """Greedy word wrap to a pixel width. Long unbreakable tokens are allowed to run over."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and _text_w(trial, size) > width_px:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def render_svg(*, result: Mapping[str, Any], candi: str,
               groups: Sequence[Mapping[str, Any]], n_experiments: int,
               n_distinct_below: int, n_rows_below: int) -> str:
    """A horizontal DOT plot, ordered by placement, CANDI in the accent hue.

    A dot plot rather than bars: the ranked quantity is stage 3's capped mean rank fraction, which
    lives in a narrow band well away from zero. A bar has to start at zero to be honest, and 26
    bars from zero would differ by a few percent of their own length; a dot plot puts the axis
    where the data is without lying about a baseline. Lower is better, and the axis says so.

    Identical-submission groups are tagged with a lettered box beside the name rather than a
    bracket: the rows are ordered by placement, so a group's members need not be adjacent and a
    span bracket would have nothing contiguous to span.
    """
    order = sorted(result["final_rank"], key=lambda t: result["final_rank"][t])
    score = {t: float(result["mean_bootstrap_score"][t]) for t in order}
    tag_of: Dict[str, List[str]] = {}
    for g in groups:
        for m in g["members"]:
            if m in score:
                tag_of.setdefault(m, []).append(g["tag"])

    n = len(order)
    ROW, PAD, FS_NAME, FS_SUB, FS_CAP = 21, 24, 11.0, 11.0, 10.5

    # --- columns, measured from the longest name rather than guessed ---
    max_tags = max((len(v) for v in tag_of.values()), default=0)
    x_name = PAD + 26                                       # rank digits sit right-aligned before
    w_names = max(_text_w(t, FS_NAME) for t in order)
    x_tag = x_name + w_names + 10
    x0 = x_tag + (max_tags * 17 if max_tags else 0) + 16
    x1 = x0 + 520
    W = int(x1 + PAD + 44)                                  # 44 = room for a value label at the tip
    text_w = W - 2 * PAD

    # --- the head block, wrapped to the canvas ---
    sub_lines = (
        _wrap(f"Challenge ranker over the B_ panel ({n_experiments} blind experiments) under "
              f"challenge truth, pval arm. CANDI added to the {n - 1}-entrant field, which changes "
              f"every entrant’s rank denominator — a figure, not a board row.", text_w, FS_SUB)
        + _wrap("Placement is reproducible; the score is not. Read the axis as an ordering.",
                text_w, FS_SUB))
    y_legend = 49 + 16 * len(sub_lines) + 6
    y0 = y_legend + 26
    y_axis = y0 + n * ROW + 10

    # --- the caption block: one line per group, so a group is never split across lines ---
    cap_lines: List[str] = []
    cap_lines += _wrap(RESOLUTION_LIMIT_SENTENCE, text_w, FS_CAP)
    cap_lines += _wrap(f"{COUNTED_SENTENCE} CANDI at rank {result['final_rank'][candi]} of {n}: "
                       f"{n_rows_below} rows below it, {n_distinct_below} distinct submissions "
                       f"(byte-identical groups collapsed; a lower bound).", text_w, FS_CAP)
    cap_lines.append("Byte-identical entrant groups (competitors/entrants/README.md §7):")
    for g in groups:
        # One line per group. SVG collapses leading whitespace, so the group lines are not indented;
        # the header line above them does the grouping. Never joined into one line: a wrap would
        # then cut a group's member list in half.
        cap_lines += _wrap(f"{g['tag']} = " + ", ".join(g["members"]), text_w, FS_CAP)
    cap_lines += _wrap(f"One bootstrap: score jsons carry one number per experiment, so stage 4's "
                       f"second-best of ten degenerates to a single bootstrap rank.", text_w,
                       FS_CAP)
    cap_top = y_axis + 48
    H = int(cap_top + 15 * len(cap_lines) + 8)

    lo_d, hi_d = min(score.values()), max(score.values())
    pad = max((hi_d - lo_d) * 0.10, 0.002)
    lo, hi = lo_d - pad, hi_d + pad
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    def sx(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    p: List[str] = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="CANDI placed {result["final_rank"][candi]} of {n} in the 2019 '
             f'challenge field, ranked by the challenge ranker over the B_ panel.">')
    p.append(f"<style>{_SVG_CSS}</style>")
    p.append(f'<rect class="surface" x="0" y="0" width="{W}" height="{H}"/>')
    p.append(f'<text class="title" x="{PAD}" y="30">CANDI inside the 2019 challenge field</text>')
    for k, line in enumerate(sub_lines):
        p.append(f'<text class="sub" x="{PAD}" y="{49 + 16 * k}">{xml_escape(line)}</text>')
    # legend — always present for two series
    lx = PAD
    p.append(f'<circle class="key-hi" cx="{lx + 5}" cy="{y_legend}" r="5"/>')
    p.append(f'<text class="sub" x="{lx + 16}" y="{y_legend + 4}">CANDI</text>')
    lx += 16 + _text_w("CANDI", FS_SUB) + 26
    p.append(f'<circle class="key" cx="{lx + 4:.0f}" cy="{y_legend}" r="4"/>')
    p.append(f'<text class="sub" x="{lx + 14:.0f}" y="{y_legend + 4}">2019 entrant</text>')
    lx += 14 + _text_w("2019 entrant", FS_SUB) + 26
    p.append(f'<rect class="tagbox" x="{lx:.0f}" y="{y_legend - 6.5}" width="13" height="13" '
             f'rx="2"/>')
    p.append(f'<text class="tag" x="{lx + 3.5:.0f}" y="{y_legend + 3}">A</text>')
    p.append(f'<text class="sub" x="{lx + 19:.0f}" y="{y_legend + 4}">byte-identical submission '
             f'group</text>')

    for t in ticks:
        x = sx(t)
        p.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y_axis}"/>')
        p.append(f'<text class="tick" x="{x:.1f}" y="{y_axis + 15}" text-anchor="middle">'
                 f'{t:.3f}</text>')
    p.append(f'<line class="axis" x1="{x0}" y1="{y_axis}" x2="{x1}" y2="{y_axis}"/>')
    p.append(f'<text class="axlabel" x="{(x0 + x1) / 2:.0f}" y="{y_axis + 33}" '
             f'text-anchor="middle">stage-3 mean capped rank fraction — '
             f'lower is better</text>')

    labelled = {candi, order[0], order[-1]}
    for i, t in enumerate(order):
        y = y0 + i * ROW + ROW / 2
        hi_row = t == candi
        p.append(f'<text class="rank" x="{x_name - 12:.0f}" y="{y + 3.5}" text-anchor="end">'
                 f'{result["final_rank"][t]}</text>')
        p.append(f'<text class="{"name-hi" if hi_row else "name"}" x="{x_name}" y="{y + 3.5}">'
                 f'{xml_escape(t)}</text>')
        for j, tag in enumerate(tag_of.get(t, [])):
            bx = x_tag + j * 17
            p.append(f'<rect class="tagbox" x="{bx:.0f}" y="{y - 6.5:.1f}" width="13" '
                     f'height="13" rx="2"/>')
            p.append(f'<text class="tag" x="{bx + 3.5:.0f}" y="{y + 3:.1f}">'
                     f'{xml_escape(tag)}</text>')
        x = sx(score[t])
        r = 5.0 if hi_row else 4.0
        p.append(f'<circle class="ring" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')
        p.append(f'<circle class="{"dot-hi" if hi_row else "dot"}" cx="{x:.1f}" '
                 f'cy="{y:.1f}" r="{r}"/>')
        if t in labelled:
            # MEASURED, not assumed: the label goes outside the mark on whichever side has room.
            lab = f"{score[t]:.4f}"
            right = x + 10 + _text_w(lab, 10.0) <= W - PAD
            p.append(f'<text class="val" x="{x + (10 if right else -10):.1f}" y="{y + 3.5:.1f}" '
                     f'text-anchor="{"start" if right else "end"}">{lab}</text>')

    for k, line in enumerate(cap_lines):
        p.append(f'<text class="cap" x="{PAD}" y="{cap_top + k * 15}">{xml_escape(line)}</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/candi_in_field.py",
        description="Rank CANDI inside the 2019 challenge field over B_ under challenge truth, "
                    "and draw the labelled figure. A figure, never a board row.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candi", required=True,
                   help="CANDI's challenge.B_.json (challenge truth, B_ panel)")
    p.add_argument("--anchors", required=True,
                   help="directory of per-entrant score jsons "
                        "(<dir>/<entrant>/challenge.B_.json, or <dir>/*.json), or a glob")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--identical-groups", default=None,
                   help="JSON file overriding the byte-identical entrant groups. A list of "
                        "{tag, members, assays, evidence} objects, or of bare member lists. "
                        "Default: the three groups of competitors/entrants/README.md §7")
    p.add_argument("--basename", default="candi_in_2019_field",
                   help="output stem (default: candi_in_2019_field)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        groups = (normalise_groups(json.loads(Path(args.identical_groups).read_text(
            encoding="utf-8"))) if args.identical_groups
            else normalise_groups([dict(g) for g in IDENTICAL_GROUPS]))
        candi_path = Path(args.candi).resolve()
        anchor_paths = [q for q in (p.resolve() for p in resolve_anchors(args.anchors))
                        if q != candi_path]
        if not anchor_paths:
            raise FieldError(f"--anchors {args.anchors} resolved no score json")
        candi_entry = load_score(candi_path)
        entries = [candi_entry] + [load_score(p) for p in anchor_paths]
    except Refusal as exc:
        print(f"[candi_in_field] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except FieldError as exc:
        print(f"[candi_in_field] {exc}", file=sys.stderr)
        return EXIT_INPUT

    names = [e["method"] for e in entries]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        print(f"[candi_in_field] REFUSED: two score jsons name the same method {dupes} — a ranked "
              f"table cannot carry a method twice (plan/BENCHMARK_DESIGN.md §7)", file=sys.stderr)
        return EXIT_REFUSED
    candi = candi_entry["method"]

    table, experiments, dropped = build_table(entries)
    result = ranking.aggregate_ranks(table)

    order = sorted(result["final_rank"], key=lambda t: result["final_rank"][t])
    below = [t for t in order if result["final_rank"][t] > result["final_rank"][candi]]
    n_distinct, _ = count_distinct(below, groups)
    measures = sorted({m for cell in table[0].values() for row in cell.values() for m in row})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render_md(result=result, entries=entries, candi=candi, experiments=experiments,
                   measures=measures, dropped=dropped, groups=groups,
                   anchor_paths=anchor_paths, candi_path=candi_path)
    svg = render_svg(result=result, candi=candi, groups=groups, n_experiments=len(experiments),
                     n_distinct_below=n_distinct, n_rows_below=len(below))
    md_path = out / f"{args.basename}.md"
    svg_path = out / f"{args.basename}.svg"
    md_path.write_text(md, encoding="utf-8")
    svg_path.write_text(svg, encoding="utf-8")

    print(md)
    print(f"[candi_in_field] byte-identical entrant groups: {len(groups)} "
          f"({', '.join(g['tag'] + '=' + '/'.join(g['members']) for g in groups)})")
    print(f"[candi_in_field] CANDI rank {result['final_rank'][candi]} of {len(entries)}; "
          f"{len(below)} rows below, {n_distinct} distinct submissions below (counted)")
    print(f"[candi_in_field] wrote {md_path} and {svg_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
