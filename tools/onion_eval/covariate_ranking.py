"""Close-out item A12: the ranked list of covariate fields and the exclusion curve, from scored runs.

    # confirmation step: leave-one-field-out (and group) trainings against the full-field runs
    python tools/onion_eval/covariate_ranking.py runs --runs <runs root> --sub p256_sv \\
        --ref convae=<dir> --ref convattn=<dir> --ref unet=<dir> --prefix a12_ --out <dir>
    # screening step: the full-field runs scored with one field foreign or neutral
    python tools/onion_eval/covariate_ranking.py screen --runs <runs root> --sub p256_sv \\
        --ref convae=<dir> --ref convattn=<dir> --ref unet=<dir> --out <dir> \\
        --screen-sub p256_foreign_out_keep_depth.lab_id [--screen-sub ...]
    # any step, from a per-seed TSV (condition, model, seed and the TERMS columns)
    python tools/onion_eval/covariate_ranking.py tsv --scores <seeds.tsv> --out <dir>

Rule (stated in the item file before any run; DECISION_RULE). Per condition (a dropped field, a
group, a curve step or a screening mode), model m and term t of TERMS: delta = mean(condition) -
mean(full), over the done seeds; R = the larger of the two seed ranges (max - min); the loss in
range units is z = -delta / R. A cell is "lower, beyond" when z > 1 and "2x" when z > 2.
- useful: lower beyond on at least 2 of the 3 models (any term), or 2x on any one model;
- safe to remove: no cell lower beyond (every model, every term), i.e. score S <= 1;
- unclear: otherwise (lower beyond on one model, below 2x); kept by default.
Score S = the largest z over the (model, term) cells; the ranked list sorts by S, most useful
first. The curve order is the single-field conditions by S, least useful first. The composite is
reported beside the terms, and is not a rule term (it is a weighted sum of three of them).

Outputs in --out: `ranked.tsv`, `effects.tsv` (every cell), `curve.tsv` and `curve.svg` (when
curve conditions exist, the x axis is the number of fields removed), `order.txt` (the curve order,
for covariate_field_arms.py curve --order) and `RANKING.md` (the rule, the ranked list, the minimum
set). Writes nothing else.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

#: the rule terms: imputation (variance recovered), gap closed on crop pairs with the oracle
#: reference, calibration, and the correction-only term
TERMS = ("imputation", "correction", "calibration", "correction_only")
REPORTED = TERMS + ("composite",)
#: select_rule.table column of each term (composite version 3, as arm-summary reads it)
TABLE_COLUMNS = {"imputation": "imputation_raw",
                 "correction": "correction_gap_closed_oracle_no_nodedup_side",
                 "calibration": "uncertainty_raw", "correction_only": "correction_only",
                 "composite": "composite_v3"}
MODELS = ("convae", "convattn", "unet")
MODEL_NAMES = {"convae": "conv autoencoder", "convattn": "conv + attention", "unet": "U-Net"}
FULL = "full"
DECISION_RULE = (
    "delta = mean(condition) - mean(full set) over seeds; R = larger seed range (max - min) of the "
    "two; z = -delta / R. Lower beyond: z > 1; 2x: z > 2. Useful: lower beyond on >= 2 of 3 models "
    "(any of imputation, gap closed, calibration, correction-only), or 2x on one model. Safe to "
    "remove: no cell lower beyond (S = max z <= 1). Unclear: otherwise; kept by default.")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summarize(rows) -> dict:
    """{(condition, model, term): {"mean", "range", "n"}} over seeds; a term undefined in any seed
    of a cell gives mean and range None."""
    acc = {}
    for r in rows:
        for t in REPORTED:
            acc.setdefault((r["condition"], r["model"], t), []).append(_f(r.get(t)))
    out = {}
    for k, v in acc.items():
        ok = [x for x in v if x is not None]
        full = bool(ok) and len(ok) == len(v)
        out[k] = {"mean": sum(ok) / len(ok) if full else None,
                  "range": (max(ok) - min(ok)) if full else None, "n": len(v)}
    return out


def effects(summ: dict, ref: str = FULL) -> list:
    """One row per (condition, model, term) of every non-reference condition."""
    out = []
    for (c, m, t), s in sorted(summ.items()):
        if c == ref:
            continue
        r0 = summ.get((ref, m, t))
        if r0 is None:
            raise ValueError(f"no reference ({ref}) scores for model {m}")
        if s["mean"] is None or r0["mean"] is None:
            d = R = z = None
        else:
            d = s["mean"] - r0["mean"]
            R = max(s["range"], r0["range"])
            z = (-d / R) if R > 0 else (math.inf if d < 0 else 0.0)
        out.append({"condition": c, "model": m, "term": t, "mean": s["mean"], "range": s["range"],
                    "n_seeds": s["n"], "ref_mean": r0["mean"], "ref_range": r0["range"],
                    "delta": d, "R": R, "z": z,
                    "lower_beyond": None if z is None or t not in TERMS else z > 1,
                    "lower_2x": None if z is None or t not in TERMS else z > 2,
                    "higher_beyond": None if z is None or t not in TERMS else z < -1})
    return out


def classify(eff: list) -> list:
    """Per condition (curve steps `curve_k*` excluded; they are read by `curve`): S, the class of
    DECISION_RULE, the models lower beyond, and the cell with the largest loss. Sorted by S, most useful first; a condition with an undefined rule cell is
    `undefined` and last."""
    by = {}
    for e in eff:
        if e["term"] in TERMS and not e["condition"].startswith("curve_k"):   # curve steps: curve.tsv
            by.setdefault(e["condition"], []).append(e)
    out = []
    for c, es in by.items():
        if any(e["z"] is None for e in es):
            out.append({"condition": c, "S": None, "class": "undefined", "models_lower_beyond": "",
                        "worst_cell": "", "n_cells_lower_beyond": None})
            continue
        top = max(es, key=lambda e: e["z"])
        mb = sorted({e["model"] for e in es if e["lower_beyond"]})
        two = any(e["lower_2x"] for e in es)
        cls = ("useful" if len(mb) >= 2 or two else "safe to remove" if not mb else "unclear")
        out.append({"condition": c, "S": top["z"], "class": cls, "models_lower_beyond": ",".join(mb),
                    "worst_cell": f"{top['model']}:{top['term']}",
                    "n_cells_lower_beyond": sum(bool(e["lower_beyond"]) for e in es)})
    out.sort(key=lambda r: (r["S"] is None, -(r["S"] if r["S"] is not None else 0), r["condition"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def single_field(cond: str):
    """The field of a leave-one-field-out condition `lofo_<field>`; None for any other."""
    return cond[len("lofo_"):] if cond.startswith("lofo_") else None


def curve_order(ranked: list) -> list:
    """The single-field (lofo_<field>) conditions by S, least useful first (ties: by name)."""
    rs = [r for r in ranked if single_field(r["condition"]) and r["S"] is not None]
    return [single_field(r["condition"]) for r in sorted(rs, key=lambda r: (r["S"], r["condition"]))]


def curve(summ: dict, order: list) -> list:
    """Rows (k, fields removed, model, term, mean, range, delta, R, within) for k = 0 (full set),
    1 (lofo_<order[0]>) and k >= 2 (curve_k<kk>) where scores exist."""
    names = {0: FULL}
    if order:
        names[1] = f"lofo_{order[0]}"
    for k in range(2, len(order) + 1):
        names[k] = f"curve_k{k:02d}"
    out = []
    for k, c in names.items():
        for m in MODELS:
            for t in REPORTED:
                s, r0 = summ.get((c, m, t)), summ.get((FULL, m, t))
                if s is None or r0 is None:
                    continue
                d = R = w = None
                if s["mean"] is not None and r0["mean"] is not None:
                    d, R = s["mean"] - r0["mean"], max(s["range"], r0["range"])
                    w = (d >= -R) if t in TERMS else None
                out.append({"k": k, "removed": ",".join(order[:k]), "condition": c, "model": m,
                            "term": t, "mean": s["mean"], "range": s["range"], "delta": d,
                            "R": R, "within": w})
    return out


def minimum_set(curve_rows: list, order: list, all_fields: list):
    """The largest k such that every rule cell of steps 1..k is within the seed range of the full
    set (and every model has a score at each step); the minimum set is all_fields minus order[:k].
    None when the curve has no step."""
    ks = sorted({r["k"] for r in curve_rows if r["k"] > 0})
    best = 0
    for k in ks:
        rs = [r for r in curve_rows if r["k"] == k and r["term"] in TERMS]
        if {r["model"] for r in rs} != set(MODELS) or any(r["within"] is not True for r in rs):
            break
        if k != best + 1:
            break
        best = k
    if not ks:
        return None
    return {"k": best, "removed": order[:best], "kept": [f for f in all_fields if f not in order[:best]]}


# ------------------------------------------------------------------------------------------ I/O

def _write_tsv(path: Path, rows: list, cols: list) -> None:
    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")


COLORS = {"convae": "#1f77b4", "convattn": "#d62728", "unet": "#2ca02c"}


def plot_curve(curve_rows: list, order: list, path: Path) -> bool:
    """One figure (SVG, no plotting dependency): one panel per reported term, one line per model,
    mean over seeds with the seed range as a bar, x = number of fields removed (tick label: the
    field removed at that step). Returns True when written."""
    from xml.sax.saxutils import escape
    W, H, L, T, PW, PH = 250, 300, 55, 30, 180, 170
    ks = sorted({r["k"] for r in curve_rows})
    if not ks:
        return False
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * len(REPORTED)}" height="{H}" '
           f'font-family="sans-serif" font-size="10">',
           f'<rect width="100%" height="100%" fill="white"/>']
    xs = {k: L + (PW * (i / max(len(ks) - 1, 1))) for i, k in enumerate(ks)}
    for pi, t in enumerate(REPORTED):
        ox = pi * W
        rs = [r for r in curve_rows if r["term"] == t and r["mean"] is not None]
        lo = [r["mean"] - r["range"] / 2 for r in rs] or [0.0]
        hi = [r["mean"] + r["range"] / 2 for r in rs] or [1.0]
        y0, y1 = min(lo), max(hi)
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 0.5, y1 + 0.5
        pad = 0.05 * (y1 - y0)
        y0, y1 = y0 - pad, y1 + pad
        Y = lambda v: T + PH * (1 - (v - y0) / (y1 - y0))  # noqa: E731
        svg.append(f'<text x="{ox + L + PW / 2}" y="{T - 12}" text-anchor="middle" '
                   f'font-weight="bold">{escape(t.replace("_", " "))}</text>')
        svg.append(f'<rect x="{ox + L}" y="{T}" width="{PW}" height="{PH}" fill="none" '
                   f'stroke="#888"/>')
        for v in (y0 + pad, (y0 + y1) / 2, y1 - pad):
            svg.append(f'<text x="{ox + L - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:.3f}</text>')
        for k in ks:
            lab = "none" if k == 0 else f"{k}: -{order[k - 1]}"
            x = ox + xs[k]
            svg.append(f'<text transform="translate({x:.1f},{T + PH + 10}) rotate(60)" '
                       f'>{escape(lab)}</text>')
        for m in MODELS:
            pts = sorted((r for r in rs if r["model"] == m), key=lambda r: r["k"])
            if not pts:
                continue
            c = COLORS[m]
            svg.append('<polyline fill="none" stroke="{}" stroke-width="1.5" points="{}"/>'.format(
                c, " ".join(f"{ox + xs[r['k']]:.1f},{Y(r['mean']):.1f}" for r in pts)))
            for r in pts:
                x = ox + xs[r["k"]]
                svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(r["mean"] - r["range"] / 2):.1f}" '
                           f'y2="{Y(r["mean"] + r["range"] / 2):.1f}" stroke="{c}"/>')
                svg.append(f'<circle cx="{x:.1f}" cy="{Y(r["mean"]):.1f}" r="2.5" fill="{c}"/>')
    for i, m in enumerate(MODELS):
        svg.append(f'<text x="{L + 120 * i}" y="{H - 6}" fill="{COLORS[m]}">'
                   f'{escape(MODEL_NAMES[m])}</text>')
    svg.append(f'<text x="{W * len(REPORTED) - 6}" y="{H - 6}" text-anchor="end">x: fields removed '
               f'(cumulative); dot: mean over seeds; bar: seed range</text>')
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg) + "\n", encoding="utf-8")
    return True


def report(rows: list, out: Path, all_fields=None, order=None) -> dict:
    """Write every output of the module docstring from per-seed rows; return a summary dict."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summ = summarize(rows)
    eff = effects(summ)
    ranked = classify(eff)
    _write_tsv(out / "effects.tsv", eff, ["condition", "model", "term", "mean", "range", "n_seeds",
                                          "ref_mean", "ref_range", "delta", "R", "z",
                                          "lower_beyond", "lower_2x", "higher_beyond"])
    _write_tsv(out / "ranked.tsv", ranked, ["rank", "condition", "class", "S", "models_lower_beyond",
                                            "n_cells_lower_beyond", "worst_cell"])
    order = list(order) if order else curve_order(ranked)
    (out / "order.txt").write_text(",".join(order) + "\n", encoding="utf-8")
    cur = curve(summ, order)
    has_curve = any(r["k"] >= 2 for r in cur)
    ms, png = None, False
    if has_curve:
        _write_tsv(out / "curve.tsv", cur, ["k", "removed", "condition", "model", "term", "mean",
                                            "range", "delta", "R", "within"])
        png = plot_curve(cur, order, out / "curve.svg")
        ms = minimum_set(cur, order, list(all_fields or order))
    lines = ["# Covariate fields: ranked list and exclusion curve", "",
             f"Rule (stated before the runs): {DECISION_RULE}", "",
             "| rank | condition | class | S | models lower beyond | worst cell |",
             "|---|---|---|---|---|---|"]
    for r in ranked:
        s = "" if r["S"] is None else f"{r['S']:.2f}"
        lines.append(f"| {r['rank']} | {r['condition']} | {r['class']} | {s} | "
                     f"{r['models_lower_beyond'] or '-'} | {r['worst_cell'] or '-'} |")
    lines += ["", f"Curve order (least useful first): {', '.join(order) or '-'}."]
    if ms is not None:
        lines.append(f"Minimum set: {', '.join(ms['kept'])} ({ms['k']} fields removed within the "
                     f"seed range: {', '.join(ms['removed']) or 'none'}).")
    else:
        lines.append("Minimum set: not yet (no curve step is scored).")
    (out / "RANKING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ranked": ranked, "order": order, "minimum_set": ms, "curve_svg": png}


def rows_from_tsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def _table(runs, d, sub):
    from onion_eval.select_rule import table
    return [r for r in table(runs, d, sub, "auto", 3) if r["arm"] == d and r["state"] == "done"]


def _as_rows(trows, condition, model):
    return [{"condition": condition, "model": model, "seed": r["seed"],
             **{t: r.get(c) for t, c in TABLE_COLUMNS.items()}} for r in trows]


def rows_from_runs(runs, sub, refs: dict, prefix: str) -> list:
    """Per-seed rows of the full-set runs (`refs`: model -> run dir name) and of every run dir
    `<prefix><condition>_<model>` under `runs`, all read in score sub-directory `sub`."""
    rows = []
    for m, d in refs.items():
        rows += _as_rows(_table(runs, d, sub), FULL, m)
    pat = re.compile(re.escape(prefix) + r"(.+)_(" + "|".join(MODELS) + r")$")
    for p in sorted(Path(runs).iterdir()):
        mm = pat.fullmatch(p.name)
        if p.is_dir() and mm:
            rows += _as_rows(_table(runs, p.name, sub), mm.group(1), mm.group(2))
    return rows


def rows_from_screen(runs, sub, refs: dict, screen_subs) -> list:
    """Per-seed rows of the full-set runs in `sub` (the reference) and in each screening
    sub-directory (condition = the sub-directory name after 'foreign_')."""
    rows = []
    for m, d in refs.items():
        rows += _as_rows(_table(runs, d, sub), FULL, m)
        for s in screen_subs:
            rows += _as_rows(_table(runs, d, s), s.split("foreign_", 1)[-1], m)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("runs", "screen", "tsv"):
        p = sp.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--order", default="", help="curve order, least useful first (default: "
                                                  "from the single-field conditions)")
        p.add_argument("--fields", default="", help="all fields, for the minimum set")
        if name == "tsv":
            p.add_argument("--scores", required=True)
            continue
        p.add_argument("--runs", required=True)
        p.add_argument("--sub", default="p256_sv")
        p.add_argument("--ref", action="append", required=True, help="<model>=<run dir name>")
        if name == "runs":
            p.add_argument("--prefix", default="a12_")
        else:
            p.add_argument("--screen-sub", action="append", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tsv":
        rows = rows_from_tsv(a.scores)
    else:
        refs = dict(x.split("=", 1) for x in a.ref)
        bad = set(refs) - set(MODELS)
        if bad:
            ap.error(f"--ref models {sorted(bad)} not in {MODELS}")
        rows = (rows_from_runs(a.runs, a.sub, refs, a.prefix) if a.cmd == "runs"
                else rows_from_screen(a.runs, a.sub, refs, a.screen_sub))
    order = [s for s in a.order.split(",") if s] or None
    fields = [s for s in a.fields.split(",") if s] or None
    res = report(rows, Path(a.out), fields, order)
    print(json.dumps({"n_rows": len(rows), "order": res["order"], "minimum_set": res["minimum_set"],
                      "curve_svg": res["curve_svg"],
                      "ranked": [(r["condition"], r["class"], r["S"]) for r in res["ranked"]]},
                     default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
