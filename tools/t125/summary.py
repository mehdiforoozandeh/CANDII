"""t125 — assemble GRID_SUMMARY.md from the part-1 (eta) and part-2 (h) aggregate outputs.

    python tools/t125/summary.py <eta_agg_dir> <h_agg_dir> <out_md> [--old-grid-tsv PATH]
                                 [--run-device PATH]

Inputs: `<eta_agg_dir>/summary_data.json` (`eta/aggregate.py`) and `<h_agg_dir>/summary_data_h.json`
(`hmap/aggregate.py`). A missing file is not an error: its sections say so. `--old-grid-tsv` is
only recorded (path and md5) under "What was run"; the old-grid values come from the aggregates.

The sections follow handoff Phase 7, in this order: What was run; Drafted readings (a placeholder
list the N6 agent fills); Part 1: each design against the old grid; Part 1: each model against
its no-covariates twin; Part 1: mean or spread; Part 1: depth law; Part 1: exploding pairs
(CRPS > 20); Part 2: complexity x inputs; Part 2: push only against push + own noise; Part 2:
repair (h versions on f's NB); Part 2: the chain against the old grid's p results; Part 2:
exploding pairs (CRPS > 20); Notes on how it was run.

Rules (handoff Phase 7, plan t125-E1):
- every value is quoted as `mean ± seed wobble` (the largest pairwise |delta| of the per-seed
  values), or `mean (one seed)` when there is one seed;
- every CRPS is quoted with its split (oracle-scaled part + scale-error part);
- each comparison row carries "gain exceeds 2 × the larger seed wobble: yes / no" (the aggregate's
  own flag; when the bar cannot be formed it says no and gives the reason);
- the explosion counts (CRPS > 20, and non-finite CRPS, apart, by kind) of both parts, the
  capped NB row counts of part 2, and every skipped reading with its reason;
- no Spearman value in any table, and no verdict words: free text copied from the inputs is
  passed through `clean()`, and the writer raises if a verdict word is left in the output;
- beside every CRPS mean, the count of the (pair, seed) records with CRPS > 20 inside it (part 1:
  the trained-pair records of each arm's cell; part 2: the records of the h's kind);
- part 2 on a partial chain set (PI 2026-10-17): a banner "<n> of <m> designs" naming the designs
  whose chain is not finished, at the top and in every part-2 section;
- the device of every part-1 run (`--run-device`, default `<eta_agg_dir>/run_device.tsv`): GPU and
  CPU counts per rung, and the note that cross-row design comparisons also cross devices;
- the merged mixture (G and each chain's largest per-bin and averaged error) and the learned h
  fits at the step cap, with their capped NB rows (from `hmap/aggregate.py`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ETA_FILE = "summary_data.json"
H_FILE = "summary_data_h.json"
N_TASKS = 432
SECTIONS = (
    "What was run",
    "Drafted readings",
    "Part 1: each design against the old grid",
    "Part 1: each model against its no-covariates twin",
    "Part 1: mean or spread",
    "Part 1: depth law",
    "Part 1: exploding pairs (CRPS > 20)",
    "Part 2: complexity × inputs",
    "Part 2: push only against push + own noise",
    "Part 2: repair (h versions on f's NB)",
    "Part 2: the chain against the old grid's p results",
    "Part 2: exploding pairs (CRPS > 20)",
    "Notes on how it was run",
)
EXCEEDS = "gain exceeds 2 × the larger seed wobble"
#: words that must never appear in the summary (a summary reports, it does not conclude)
VERDICT_RE = re.compile(r"\b(supported|refuted|pass|passes|passed|fail|fails|failed|failure|"
                        r"confirmed)\b", re.IGNORECASE)
_REPLACE = {"pass": "give", "passes": "gives", "passed": "given", "fail": "miss",
            "fails": "misses", "failed": "did not work", "failure": "problem",
            "supported": "backed", "refuted": "contradicted", "confirmed": "checked"}
CLASS_ORDER = ("DNase", "narrow", "broad", "all")
GV_LABEL = {"per_track": "one g per track", "across": "one g across tracks"}
METRIC_LABEL = {"crps_all": "CRPS, all bins", "crps_top1": "CRPS, top 1 % bins",
                "mse_log1p_all": "MSE on log(1 + value) of the predicted mean, all bins",
                "mse_log1p_top1": "MSE on log(1 + value) of the predicted mean, top 1 % bins",
                "crps_log_all": "CRPS on log(1 + P), all bins",
                "crps_log_top1": "CRPS on log(1 + P), top 1 % bins",
                "crps_oracle_scaled_all": "CRPS split: oracle-scaled part",
                "scale_error_all": "CRPS split: scale-error part",
                "crps_log_oracle_scaled_all": "CRPS on log(1 + P) split: oracle-scaled part",
                "crps_log_scale_error_all": "CRPS on log(1 + P) split: scale-error part",
                "crps_mean_point_all": "point CRPS of the mean, all bins",
                "crps_mean_point_top1": "point CRPS of the mean, top 1 % bins",
                "crps_mean_point_oracle_scaled_all": "point CRPS split: oracle-scaled part",
                "crps_mean_point_scale_error_all": "point CRPS split: scale-error part"}
PARTS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all", "crps_top1", "mse_log1p_all",
         "mse_log1p_top1")
EX_COL = "records with CRPS > 20 in the mean (this / other)"
CRPS_PARTS = ("crps_all", "crps_top1", "crps_oracle_scaled_all", "scale_error_all")
PART_LABEL = {"crps_all": "CRPS all", "crps_oracle_scaled_all": "oracle-scaled part",
              "scale_error_all": "scale-error part", "crps_top1": "CRPS top 1 %",
              "mse_log1p_all": "MSE all", "mse_log1p_top1": "MSE top 1 %"}


# ---------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------


def clean(text) -> str:
    """Free text from the inputs, with any verdict word replaced and table pipes escaped."""
    if text is None:
        return ""
    s = VERDICT_RE.sub(lambda m: _REPLACE[m.group(0).lower()], str(text))
    return s.replace("|", "/").replace("\n", " ")


def f(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    return f"{float(v):.4g}"


def pm(mean, wob) -> str:
    """`mean ± wobble`, or `mean (one seed)` when the wobble does not exist."""
    if mean is None:
        return "—"
    if wob is None:
        return f"{f(mean)} (one seed)"
    return f"{f(mean)} ± {f(wob)}"


def yn(flag, reason=None) -> str:
    out = "yes" if flag else "no"
    if reason and not flag:
        out += f" ({clean(reason)})"
    return out


def gv(g) -> str:
    return GV_LABEL.get(g, str(g)) if g is not None else "—"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def _ckey(mc) -> int:
    return CLASS_ORDER.index(mc) if mc in CLASS_ORDER else len(CLASS_ORDER)


def _load(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# part 1
# ---------------------------------------------------------------------------------------------


def _pairs_text(diag: dict | None) -> str:
    if not diag:
        return "—"
    if diag.get("pair_matched") is False:
        return "class means (old-grid TSV, not pair-matched)"
    return (f"{diag.get('n_pairs_used', '—')} used, {diag.get('n_pairs_dropped_nonfinite', 0)} "
            f"dropped as non-finite")


def _split_text(comp: dict) -> str:
    sg = comp.get("split_gain") or {}
    bits = []
    for k, lab in (("crps_oracle_scaled_all", "oracle-scaled"), ("scale_error_all", "scale error")):
        p = sg.get(k)
        if p:
            bits.append(f"{lab} {pm(p['this'], p['this_wobble'])} vs "
                        f"{pm(p['other'], p['other_wobble'])}, gain {f(p['gain'])}")
    return "; ".join(bits) or "—"


def _eta_rows(readings: list[dict], name: str) -> list[dict]:
    rows = [r for r in readings if r["reading"] == name]
    return sorted(rows, key=lambda r: (r["rung"], r["g_version"], _ckey(r["mark_class"]),
                                       r["metric"]))


def _p1_index(rows) -> dict:
    return {(e.get("rung"), e.get("g_version"), e.get("mark_class"), e.get("model"), e.get("kind"),
             e.get("eval")): int(e.get("n_crps_gt_20") or 0) for e in rows or []}


def _p1_counts(eta: dict, r: dict, comparison: str) -> tuple:
    """(n this, n other): the trained-pair (score) records with CRPS > 20 in the two cells of a
    part-1 comparison (both arms over their 3 seeds)."""
    t125, old = _p1_index(eta.get("explosions")), _p1_index(eta.get("oldgrid_explosions"))
    rung, g, mc = r["rung"], r["g_version"], r["mark_class"]
    this = t125.get((rung, g, mc, r.get("model") or "real", "trained", "score"))
    if comparison == "beatstwin":
        other = t125.get((rung, g, mc, "nocov", "trained", "score"))
    elif comparison == "vs_oldspread":
        other = t125.get(("Aold", g, mc, "real", "trained", "score"))
    elif comparison == "vs_oldgrid":
        orung = (r.get("components") or {}).get("oldgrid_rung") or ("A" if rung == "Aold"
                                                                    else rung)
        other = old.get((orung, g, mc, "real", "trained", "score"))
    else:
        other = None
    return this, other


def _ex_text(n_this, n_other) -> str:
    return f"{f(n_this)} / {f(n_other)}"


def _cmp_table(rows: list[dict], this_lab: str, other_lab: str, gain_lab: str,
               extra=None, counts=None) -> list[str]:
    header = ["rung", "g", "class", "metric", this_lab, other_lab, EX_COL, gain_lab,
              "bar (2 × wobble)", EXCEEDS, "CRPS split (this vs other)", "pairs"]
    if extra:
        header.append(extra[0])
    body = []
    for r in rows:
        c = r["components"]
        line = [r["rung"], gv(r["g_version"]), r["mark_class"], METRIC_LABEL.get(r["metric"],
                                                                                r["metric"]),
                pm(c.get("this"), c.get("wobble_this")), pm(c.get("other"), c.get("wobble_other")),
                (_ex_text(*counts(r)) if counts and r["metric"] in CRPS_PARTS else "—"),
                f(r["value"]), f(r["bar"]), yn(r["met"], c.get("met_reason")),
                _split_text(c) if r["metric"] == "crps_all" else "—", _pairs_text(c.get("pairs"))]
        if extra:
            line.append(extra[1](r))
        body.append(line)
    return table(header, body)


def part1_oldgrid(eta: dict) -> list[str]:
    rd = eta["readings"]
    out = ["Each t125 design (real model, trained pairs, score chromosomes) against the same "
           "design in the old grid (counts space). Aold is compared with the old grid's A: the "
           "depth offset alone. gain = old grid − t125, positive = t125 lower. The column \""
           + EX_COL + "\" counts the (pair, seed) records with CRPS > 20 in each arm's cell; a "
           "nonzero count inflates that CRPS mean.", ""]
    out += _cmp_table(_eta_rows(rd, "vs_oldgrid"), "t125", "old grid", "gain (old − t125)",
                      counts=lambda r: _p1_counts(eta, r, "vs_oldgrid"))
    out += ["A against Aold: the new spread rule alone (both with the depth offset). "
            "gain = Aold − A, positive = A lower.", ""]
    out += _cmp_table(_eta_rows(rd, "vs_oldspread"), "A", "Aold", "gain (Aold − A)",
                      counts=lambda r: _p1_counts(eta, r, "vs_oldspread"))
    return out


def part1_twin(eta: dict) -> list[str]:
    out = [f"Note: {clean(eta.get('twin_note'))}.", "",
           "gain = twin − real, positive = the real model lower. The old grid's own gap (twin − "
           "real) is beside it for scale only.", ""]
    out += _cmp_table(_eta_rows(eta["readings"], "beatstwin"), "real", "no-covariates twin",
                      "gain (twin − real)",
                      ("old grid gap", lambda r: f(r["components"].get("oldgrid_gap"))),
                      counts=lambda r: _p1_counts(eta, r, "beatstwin"))
    return out


def part1_mean_spread(eta: dict) -> list[str]:
    out = ["Where each gain sits: in the mean (MSE of the predicted mean) or in the spread (the "
           "CRPS split). Each cell: gain (bar) — " + EXCEEDS + ": yes / no. The old grid has no "
           "MSE.", ""]
    header = ["comparison", "rung", "g", "class"] + [PART_LABEL[p] for p in PARTS]
    body = []
    for r in sorted(_eta_rows(eta["readings"], "mean_or_spread"),
                    key=lambda r: (r["metric"], r["rung"], r["g_version"], _ckey(r["mark_class"]))):
        parts = r["components"].get("parts", {})
        cells = []
        for p in PARTS:
            x = parts.get(p)
            if not x or x.get("gain") is None:
                cells.append("—")
            else:
                ex = ""
                if p in CRPS_PARTS:
                    nt, no = _p1_counts(eta, r, r["metric"])
                    if nt or no:
                        ex = f" [records with CRPS > 20: this {f(nt)}, other {f(no)}]"
                cells.append(f"{f(x['gain'])} ({f(x['bar'])}) — {yn(x['gain_exceeds_bar'])}; "
                             f"this {pm(x['this'], x['this_wobble'])}, other "
                             f"{pm(x['other'], x['other_wobble'])}{ex}")
        body.append([r["metric"], r["rung"], gv(r["g_version"]), r["mark_class"]] + cells)
    return out + table(header, body)


def part1_depth(eta: dict) -> list[str]:
    out = ["Value = the largest |2^(mean over seeds of log2 predicted − log2 true) − 1| over the "
           "depth→depth law pairs (real model). Within the bar: value ≤ 0.10. Beside it: the same "
           "over the trained base↔depth pairs, the no-covariates twin, and the old grid's value "
           "with the gain (old − t125).", ""]
    header = ["rung", "g", "class", "law pairs", "value", "within 0.10", "trained pairs value",
              "twin value", "old grid value", "gain (old − t125)", "bar (2 × wobble)", EXCEEDS]
    body = []
    rows = sorted(eta.get("depth_law", []), key=lambda r: (r["rung"], r["g_version"],
                                                           _ckey(r["mark_class"])))
    for r in rows:
        c = r["components"]
        body.append([r["rung"], gv(r["g_version"]), r["mark_class"], f(c.get("n_pairs")),
                     pm(r["value"], r["seed_wobble"]), yn(r["met"], c.get("met_reason")),
                     pm(c.get("trained_value"), c.get("trained_seed_wobble")),
                     pm(c.get("nocov_value"), c.get("nocov_seed_wobble")),
                     pm(c.get("oldgrid_value"), c.get("oldgrid_seed_wobble")),
                     f(c.get("gain_vs_oldgrid")), f(c.get("bar_vs_oldgrid")),
                     yn(c.get("gain_vs_oldgrid_exceeds_bar"), c.get("oldgrid_note"))])
    return out + table(header, body)


def _explode_totals(rows: list[dict], key_fn, gt, nonfin, extra=None) -> dict:
    tot: dict = defaultdict(lambda: [0, 0, 0, 0])
    for e in rows:
        t = tot[key_fn(e)]
        t[0] += e.get("n_records", 0)
        t[1] += e.get(gt, 0)
        t[2] += e.get(nonfin, 0)
        if extra:
            t[3] += e.get(extra, 0)
    return tot


def part1_explosions(eta: dict) -> list[str]:
    ex = eta.get("explosions", [])
    tot = _explode_totals(ex, lambda e: f"{e['kind']}/{e['eval']}", "n_crps_gt_20",
                          "n_crps_nonfinite")
    n_gt = sum(t[1] for t in tot.values())
    n_nf = sum(t[2] for t in tot.values())
    out = [f"Part 1 explosion counts: {n_gt} (pair, seed) records with CRPS > 20 and {n_nf} with a "
           "non-finite CRPS bin, counted apart, by record kind (a non-finite record is never in a "
           "mean; a depth-law record is a copy and is not counted).", ""]
    out += table(["kind/eval", "records", "CRPS > 20", "non-finite"],
                 [[k, f(v[0]), f(v[1]), f(v[2])] for k, v in sorted(tot.items())])
    nz = [e for e in ex if e.get("n_crps_gt_20") or e.get("n_crps_nonfinite")]
    out += ["Cells with a nonzero count:", ""]
    out += table(["rung", "g", "class", "model", "kind/eval", "records", "CRPS > 20", "non-finite"],
                 [[e["rung"], gv(e["g_version"]), e["mark_class"], e["model"],
                   f"{e['kind']}/{e['eval']}", f(e["n_records"]), f(e["n_crps_gt_20"]),
                   f(e["n_crps_nonfinite"])]
                  for e in sorted(nz, key=lambda e: (e["rung"], e["g_version"],
                                                     _ckey(e["mark_class"]), e["model"]))])
    old = eta.get("oldgrid_explosions") or []
    if old:
        out += ["The old grid (counts space, real model), the same counts:", ""]
        tot = _explode_totals(old, lambda e: f"{e.get('kind')}/{e.get('eval')}", "n_crps_gt_20",
                              "n_crps_nonfinite")
        out += table(["kind/eval", "records", "CRPS > 20", "non-finite"],
                     [[k, f(v[0]), f(v[1]), f(v[2])] for k, v in sorted(tot.items())])
    return out


# ---------------------------------------------------------------------------------------------
# part 2
# ---------------------------------------------------------------------------------------------


def part2_banner(h: dict | None) -> list[str]:
    """"<n> of <m> designs" when some expected chain f is not finished (PI 2026-10-17)."""
    if not h or not h.get("chain_f_runs_missing") or not h.get("chain_f_expected_runs"):
        return []
    fin = [x.split("_")[0] for x in h.get("chain_f_finished_runs", [])]
    mis = [x.split("_")[0] for x in h["chain_f_runs_missing"]]
    n = len(h["chain_f_expected_runs"])
    return [f"**Part 2: {len(fin)} of {n} designs.** The chain readings (repair, the chain "
            f"against the old grid, push only against own noise in the chain) cover the f of "
            f"track {h.get('chain_track') or 'all'}, seed 0, for designs {', '.join(fin) or 'none'}. "
            f"Missing (chain not finished): {', '.join(mis)}. These tables are refreshed when "
            "they land; the alone readings are complete.", ""]


def _h_ex_index(h: dict) -> dict:
    return {(e.get("scope"), e.get("f_rung"), e.get("f_g_version"), e.get("h_name"),
             e.get("mark_class")): e for e in h.get("explosions", [])}


def _hkey(r):
    return (r.get("f_rung") or "", r.get("f_g_version") or "", r.get("h_name") or "",
            _ckey(r.get("mark_class")), r.get("metric") or "")


def part2_grid(h: dict) -> list[str]:
    cells = h.get("alone_cells", [])
    idx = {(c["h_name"], c["mark_class"], c["metric"]): c for c in cells}
    out = ["Each h alone on the true X′ (every usable product; mean over the products of a class, "
           "per h seed). Push only: the proper CRPS on P; push + own noise: the point CRPS of the "
           "P-scale mean (never the same number as a proper CRPS). The CRPS on log(1 + P) is "
           "proper for both. Full tables: grid_h.md, h_alone_summary.tsv; the chain values per f: "
           "h_chain_summary.tsv.", ""]
    header = ["h", "class", "CRPS on P, all bins", "its split (oracle-scaled / scale error)",
              "CRPS on P, top 1 %", "CRPS on log(1 + P), all bins", "its split",
              "CRPS on log(1 + P), top 1 %", "MSE log(1 + P) of the mean, all bins",
              "MSE, top 1 %", "products", "records with CRPS > 20 in the CRPS on P means",
              "records with a non-finite bin (on P / on log(1 + P))"]
    body = []
    for name, mc in sorted({(c["h_name"], c["mark_class"]) for c in cells},
                           key=lambda k: (k[0], _ckey(k[1]))):
        noise = any(idx.get((name, mc, m)) for m in ("crps_mean_point_all",))
        nat = "crps_mean_point" if noise else "crps"
        sp = (("crps_mean_point_oracle_scaled_all", "crps_mean_point_scale_error_all") if noise
              else ("crps_oracle_scaled_all", "scale_error_all"))

        def v(m):
            c = idx.get((name, mc, m))
            return pm(c["mean"], c["seed_wobble"]) if c else "—"

        n = max((c.get("n") or 0 for (hn, cl, _), c in idx.items() if hn == name and cl == mc),
                default=0)
        ex = next((c.get("explosions") for (hn, cl, _), c in idx.items()
                   if hn == name and cl == mc and c.get("explosions")), None) or {}
        body.append([name + (" (own noise: point CRPS)" if noise else ""), mc, v(f"{nat}_all"),
                     f"{v(sp[0])} / {v(sp[1])}", v(f"{nat}_top1"), v("crps_log_all"),
                     f"{v('crps_log_oracle_scaled_all')} / {v('crps_log_scale_error_all')}",
                     v("crps_log_top1"), v("mse_log1p_all"), v("mse_log1p_top1"), f(n),
                     f"{f(ex.get('n_gt_20'))} of {f(ex.get('n_records'))}",
                     f"{f(ex.get('n_native_nonfinite'))} / {f(ex.get('n_log_nonfinite'))}"])
    return out + table(header, body)


def _readings(h: dict, name: str) -> list[dict]:
    return sorted((r for r in h.get("readings", []) if r["reading"] == name), key=_hkey)


def _fcell(r) -> str:
    if r.get("f_rung") is None:
        return "alone"
    return f"{r['f_rung']}, {gv(r['f_g_version'])}"


def part2_push_noise(h: dict) -> list[str]:
    out = ["The two uncertainty variants of the same h (variant, rung, training source), on the "
           "log scale and the MSE (PI 2026-10-11). gain = push − own noise, positive = push + own "
           "noise lower. An fnb h has one h seed: its wobble is the true-X′ h's (PI).", ""]
    header = ["scope / f", "variant", "rung", "source", "class", "metric", "push only",
              "push + own noise", "gain (push − noise)", "bar (2 × wobble)", EXCEEDS,
              "log-scale CRPS split gain", "note"]
    body = []
    for r in _readings(h, "push_vs_noise"):
        sg = (r.get("components") or {}).get("split_gain") or {}
        split = "; ".join(f"{'oracle-scaled' if 'oracle' in k else 'scale error'} gain "
                          f"{f(p.get('gain'))} (bar {f(p.get('bar'))})" for k, p in sorted(sg.items()))
        body.append([_fcell(r), r["variant"], f(r["rung"]), r["source"], r["mark_class"],
                     METRIC_LABEL.get(r["metric"], r["metric"]),
                     pm(r["push_mean"], r["push_wobble"]), pm(r["noise_mean"], r["noise_wobble"]),
                     f(r["gain"]), f(r["bar"]), yn(r["gain_exceeds_bar"]), split or "—",
                     clean(r.get("wobble_note"))])
    return out + table(header, body)


def part2_repair(h: dict) -> list[str]:
    out = ["The h trained on the true X′ against the h trained on f's NB, both in the chain on "
           "the same f. delta = true − fnb, positive = the fnb-trained h lower. The fnb h has one "
           "h seed (PI decision (ii) b); the bar uses the true-X′ h's seed wobble.", ""]
    header = ["f", "class", "variant", "rung", "noise", "metric", "true-X′ h", "fnb h",
              "records with CRPS > 20 in the mean (true / fnb)",
              "delta (true − fnb)", "bar (2 × wobble)", "|delta| exceeds bar", EXCEEDS, "pairs"]
    body = []
    exi = _h_ex_index(h)
    for r in _readings(h, "repair"):
        p = (r.get("components") or {}).get("pairs") or {}
        ex = "—"
        if not r["metric"].startswith(("crps_log", "mse")):
            et = exi.get(("chain", r["f_rung"], r["f_g_version"], r.get("true_h"),
                          r["mark_class"])) or {}
            ef = exi.get(("chain", r["f_rung"], r["f_g_version"], r.get("fnb_h"),
                          r["mark_class"])) or {}
            ex = _ex_text(et.get("n_gt_20"), ef.get("n_gt_20"))
        body.append([_fcell(r), r["mark_class"], r["variant"], f(r["rung"]),
                     "own noise" if r["noise"] else "push", METRIC_LABEL.get(r["metric"],
                                                                             r["metric"]),
                     pm(r["true_mean"], r["true_wobble"]), pm(r["fnb_mean"], None), ex,
                     f(r["delta"]), f(r["bar"]), yn(r["abs_delta_exceeds_bar"]),
                     yn(r["gain_exceeds_bar"]),
                     f"{f(p.get('n_used'))} used, {f(p.get('n_dropped_not_finite'))} dropped"])
    return out + table(header, body)


def part2_vs_old(h: dict) -> list[str]:
    out = ["The chain h (on seed 0 of each real f) against the old grid's p-space real f. The "
           "chain's f is seed 0 only, so the old grid's seed-0 value and its 3-seed mean are both "
           "given. delta = old − chain h, positive = the chain h lower. Push-only and formula h "
           "only (an own-noise h's native number is a point CRPS). The CRPS split is in the rows "
           "of the two split parts.", ""]
    header = ["f", "class", "h", "metric", "chain h", "old grid, seed 0 alone", "old grid",
              "records with CRPS > 20 (chain h / old grid cell)",
              "delta (seed 0)", "delta (mean)", "bar (2 × wobble)",
              EXCEEDS + " (seed 0)", EXCEEDS + " (mean)", "pair-matched", "note"]
    body = []
    for r in _readings(h, "vs_oldgrid"):
        body.append([_fcell(r), r["mark_class"], r["h_name"], METRIC_LABEL.get(r["metric"],
                                                                               r["metric"]),
                     pm(r["h_mean"], r["h_wobble"]), f(r["old_seed0"]),
                     pm(r["old_mean"], r["old_wobble"]),
                     _ex_text(r.get("h_n_crps_gt_20"), r.get("old_n_crps_gt_20")),
                     f(r["delta_seed0"]), f(r["delta"]), f(r["bar"]),
                     yn(r["gain_seed0_exceeds_bar"]), yn(r["gain_exceeds_bar"]), yn(r["matched"]),
                     clean(r.get("note"))])
    return out + table(header, body)


def part2_explosions(h: dict) -> list[str]:
    ex = h.get("explosions", [])
    tot = _explode_totals(ex, lambda e: (e["scope"], e["kind"]), "n_gt_20", "n_native_nonfinite",
                          "n_log_nonfinite")
    n_gt = sum(t[1] for t in tot.values())
    n_nf = sum(t[2] for t in tot.values())
    n_lf = sum(t[3] for t in tot.values())
    out = [f"Part 2 explosion counts: {n_gt} (product or pair, seed) records with CRPS > 20 (push "
           f"only: the proper CRPS; own noise: the point CRPS of the mean), {n_nf} with a "
           f"non-finite native CRPS bin and {n_lf} with a non-finite log-scale CRPS bin, counted "
           "apart, by scope and kind.", ""]
    out += table(["scope", "kind", "records", "CRPS > 20", "non-finite (native)",
                  "non-finite (log scale)"],
                 [[k[0], k[1], f(v[0]), f(v[1]), f(v[2]), f(v[3])] for k, v in sorted(tot.items())])
    nz = [e for e in ex if e.get("n_gt_20") or e.get("n_native_nonfinite") or
          e.get("n_log_nonfinite")]
    out += ["Cells with a nonzero count:", ""]
    out += table(["scope / f", "h", "class", "kind", "records", "CRPS > 20", "non-finite (native)",
                  "non-finite (log scale)"],
                 [[_fcell(e), e["h_name"], e["mark_class"], e["kind"], f(e["n_records"]),
                   f(e["n_gt_20"]), f(e["n_native_nonfinite"]), f(e["n_log_nonfinite"])]
                  for e in sorted(nz, key=_hkey)])
    old = h.get("oldgrid_explosions") or []
    if old:
        out += ["The old grid (p space, real model), per f cell and class:", ""]
        out += table(["rung", "g", "class", "records", "CRPS > 20", "non-finite"],
                     [[e["rung"], gv(e["g_version"]), e["mark_class"], f(e["n_records"]),
                       f(e["n_crps_gt_20"]), f(e["n_crps_nonfinite"])] for e in old])
    cap = h.get("capped", [])
    n_rows = sum(int(c.get("n_capped_rows") or 0) for c in cap)
    out += [f"Capped NB rows in the chain scoring (NB support wider than K_MAX = 4096; every h of "
            f"one f shares them): {n_rows} rows in total over {len(cap)} f runs. A nonzero count "
            "goes to the PI. The capped rows of the fnb fits are under \"Learned h fits\" below.",
            ""]
    out += table(["f run", "pairs", "pairs with a capped row", "capped rows",
                  "the same for every h of this f"],
                 [[c["f_run"], f(c.get("n_pairs")), f(c.get("n_pairs_capped")),
                   f(c.get("n_capped_rows")), yn(c.get("consistent_across_h"))] for c in cap])
    out += part2_merge(h) + part2_fits(h)
    return out


def part2_merge(h: dict) -> list[str]:
    me = h.get("merge_errors")
    if me is None:
        return []
    gs = sorted({m.get("mix_groups") for m in me if m.get("mix_groups") is not None})
    out = ["### Merged mixture", "",
           f"Own-noise h in the chain are scored with a merged mixture: G = "
           f"{', '.join(str(g) for g in gs) or '—'} moment-matched Gaussians at most (PI "
           "2026-10-16, G chosen 2026-10-17). Error = relative error of crps_log against the exact "
           "mixture, on the widest-support bins with more than G components; per bin (the bar: "
           "below 1e-3) and averaged over those bins. Largest over each f's own-noise h.", ""]
    out += table(["f run", "G", "own-noise h", "h with bins checked", "bins checked (max)",
                  "largest per-bin error", "its h", "largest averaged error", "its h",
                  "h at or above 1e-3 per bin", "note"],
                 [[m["f_run"], f(m.get("mix_groups")), f(m.get("n_h")), f(m.get("n_h_checked")),
                   f(m.get("n_bins_max")), f(m.get("rel_err_max")),
                   clean(m.get("rel_err_max_h")) or "—", f(m.get("avg_rel_err_max")),
                   clean(m.get("avg_rel_err_max_h")) or "—", f(m.get("n_h_rel_err_max_ge_bar")),
                   clean(m.get("note"))] for m in me])
    return out


def part2_fits(h: dict) -> list[str]:
    fits = h.get("fits")
    if not fits:
        return []
    out = ["### Learned h fits at the step cap, and their capped NB rows", ""]
    for t in fits.get("totals", []):
        cap = "/".join(str(x) for x in t.get("max_steps", [])) or "—"
        out.append(f"- {t['source']} fits: {t['n_fits']}; {t['n_at_step_cap']} ran to the "
                   f"{cap}-step cap (no early stop), {t['n_best_at_step_cap']} of them with the "
                   f"best step at the cap; {t['n_fits_with_capped_rows']} fits have capped NB rows, "
                   f"{t['n_capped_rows']} capped NB rows in all (training and validation draws).")
    out.append("")
    at = [r for r in fits.get("runs", []) if r.get("at_step_cap")]
    out += table(["h fit", "source", "steps run", "best step", "capped NB rows"],
                 [[clean(r["h_name"]), r["source"], f(r.get("steps_run")), f(r.get("best_step")),
                   f(r.get("n_capped"))] for r in sorted(at, key=lambda r: (r["source"],
                                                                            r["h_name"]))])
    return out


# ---------------------------------------------------------------------------------------------
# what was run, notes
# ---------------------------------------------------------------------------------------------


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_devices(path) -> list[dict] | None:
    if path is None or not Path(path).is_file():
        return None
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def device_note(rows: list[dict] | None, path=None) -> list[str]:
    """GPU / CPU counts of the part-1 training runs, per rung (PI 2026-10-17)."""
    if rows is None:
        return [f"- Part 1 devices: no run_device.tsv at `{path}`." if path else
                "- Part 1 devices: no run_device.tsv given."]
    tot: dict = defaultdict(int)
    by: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        d = (r.get("train_device") or "unknown").lower()
        tot[d] += 1
        by[r["run_name"].split("_")[0]][d] += 1
    order = ["A", "B", "C", "D", "A2", "B2", "C2", "D2", "Aold"]
    rungs = [x for x in order if x in by] + sorted(set(by) - set(order))
    per = "; ".join(f"{x} " + ", ".join(f"{n} {d.upper()}" for d, n in sorted(by[x].items()))
                    for x in rungs)
    totals = ", ".join(f"{n} on {d.upper()}" for d, n in sorted(tot.items()))
    return [f"- Part 1 devices (`{path}`): {len(rows)} runs trained, {totals}. By rung: {per}. "
            "Accepted by the PI (2026-10-17). A comparison across rows of designs (A, B, C, D "
            "against A2, B2, C2, D2, or against Aold) also crosses devices."]


def what_was_run(eta, h, eta_path, h_path, old_tsv, devices=None, dev_path=None) -> list[str]:
    out = []
    if eta is None:
        out.append(f"- Part 1: no aggregate at `{eta_path}`.")
    else:
        out.append(f"- Part 1 (the g → f count grid with the depth offset): "
                   f"{eta.get('runs_present')} of {eta.get('n_expected_runs', N_TASKS)} runs "
                   f"present, {eta.get('runs_missing')} missing; aggregate written "
                   f"{eta.get('created_utc')} (`{eta_path}`).")
    if h is None:
        out.append(f"- Part 2: no aggregate at `{h_path}`.")
    else:
        out.append(f"- Part 2 (the map h to P = −log10 p): {h.get('alone_present')} of "
                   f"{h.get('alone_expected')} h scored alone, {h.get('chain_f_present')} of "
                   f"{h.get('chain_f_expected')} f runs scored in the chain (finished); aggregate "
                   f"written {h.get('created_utc')} (`{h_path}`).")
        if h.get("chain_f_runs_missing"):
            out.append("- " + part2_banner(h)[0])
    if eta is not None:
        out += device_note(devices, dev_path)
    if old_tsv is not None:
        p = Path(old_tsv)
        out.append(f"- Old grid TSV: `{p}`" + (f" (md5 {_md5(p)})." if p.is_file()
                                                else " (not found)."))
    out.append("")
    return out


def drafted_readings() -> list[str]:
    return ["_Placeholder: the readings are drafted here from the tables below, one per line, "
            "each quoted with its seed spread, its CRPS split where it is a CRPS, and \""
            + EXCEEDS + ": yes / no\". A reading reports; it does not conclude._", "",
            "- (part 1, against the old grid)", "- (part 1, against the twin)",
            "- (part 1, mean or spread)", "- (part 1, depth law)",
            "- (part 2, complexity × inputs)", "- (part 2, push only against own noise)",
            "- (part 2, repair)", "- (part 2, the chain against the old grid)", ""]


def _skipped_table(rows: list[dict]) -> list[str]:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r.get("reading"), clean(r.get("reason")))].append(r)
    body = []
    for (reading, reason), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        ex = []
        for r in rs[:3]:
            ex.append(", ".join(clean(f"{k} {r[k]}") for k in ("scope", "rung", "f_rung",
                                                                "g_version",
                                                                "f_g_version", "mark_class",
                                                                "variant", "h_name")
                                if r.get(k) is not None))
        body.append([reading, reason, f(len(rs)), "; ".join(ex)])
    return table(["reading", "reason", "count", "first examples"], body)


def notes(eta, h) -> list[str]:
    out = ["- Every metric is lower-is-better. A value is `mean ± seed wobble` over the seeds "
           "(the largest pairwise |delta| of the per-seed values), or `(one seed)`.",
           "- A gain counts only when it is larger than 2 × the larger seed wobble of the two "
           "arms (handoff Phase 7); the tables say yes or no and never conclude.",
           "- Two arms are compared on common ground only: the pairs or products present in "
           "both and finite in both (PI 2026-10-11); the dropped counts are in the tables.",
           "- Spearman values are reported in the aggregates' JSON only and are no headline here.",
           "- Part 1 law scoring: depth→depth pairs only (PI decision (i) b).",
           "- Part 2: one fnb h seed per f (PI decision (ii) b); own-noise loss on f's NB = the "
           "expected Gaussian NLL (decision (iii) a); own-noise scoring = the exact mixture CRPS "
           "on log(1 + P) and the point CRPS of the P-scale mean (decision (iv) a)."]
    if eta is not None and eta.get("twin_note"):
        out.append(f"- Twin: {clean(eta['twin_note'])}.")
    for k, v in sorted(((h or {}).get("notes") or {}).items()):
        out.append(f"- {clean(v)}.")
    out.append("")
    out += ["### Readings skipped, part 1", ""]
    out += (_skipped_table(eta.get("readings_skipped", [])) if eta is not None
            else ["_No part-1 aggregate._", ""])
    out += ["### Readings skipped, part 2", ""]
    out += (_skipped_table(h.get("readings_skipped", [])) if h is not None
            else ["_No part-2 aggregate._", ""])
    return out


# ---------------------------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------------------------


def build(eta_agg, h_agg, old_grid_tsv=None, run_device=None) -> str:
    eta_path, h_path = Path(eta_agg) / ETA_FILE, Path(h_agg) / H_FILE
    eta, h = _load(eta_path), _load(h_path)
    dev_path = Path(run_device) if run_device is not None else Path(eta_agg) / "run_device.tsv"
    devices = read_devices(dev_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = {
        "What was run": what_was_run(eta, h, eta_path, h_path, old_grid_tsv, devices, dev_path),
        "Drafted readings": drafted_readings(),
        "Notes on how it was run": notes(eta, h),
    }
    p1 = (("Part 1: each design against the old grid", part1_oldgrid),
          ("Part 1: each model against its no-covariates twin", part1_twin),
          ("Part 1: mean or spread", part1_mean_spread),
          ("Part 1: depth law", part1_depth),
          ("Part 1: exploding pairs (CRPS > 20)", part1_explosions))
    p2 = (("Part 2: complexity × inputs", part2_grid),
          ("Part 2: push only against push + own noise", part2_push_noise),
          ("Part 2: repair (h versions on f's NB)", part2_repair),
          ("Part 2: the chain against the old grid's p results", part2_vs_old),
          ("Part 2: exploding pairs (CRPS > 20)", part2_explosions))
    for title, fn in p1:
        body[title] = fn(eta) if eta is not None else [f"_No part-1 aggregate at `{eta_path}`._",
                                                       ""]
    if eta is not None:
        body[p1[0][0]] = device_note(devices, dev_path) + [""] + body[p1[0][0]]
    banner = part2_banner(h)
    for title, fn in p2:
        body[title] = (banner + fn(h)) if h is not None else [
            f"_No part-2 aggregate at `{h_path}`._", ""]
    lines = ["# t125 grid summary", "",
             f"Written by `tools/t125/summary.py` at {now}. Every value is quoted with its seed "
             "spread; every CRPS with its split; each comparison says whether its gain exceeds "
             "2 × the larger seed wobble. Exploratory: no row here is a tick, a close or a "
             "conclusion.", ""] + banner
    for title in SECTIONS:
        lines += [f"## {title}", ""] + body[title]
    text = "\n".join(lines).rstrip("\n") + "\n"
    bad = sorted({m.group(0) for m in VERDICT_RE.finditer(text)})
    if bad:
        raise ValueError(f"verdict words in the summary: {bad}")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eta_agg_dir", type=Path)
    ap.add_argument("h_agg_dir", type=Path)
    ap.add_argument("out_md", type=Path)
    ap.add_argument("--old-grid-tsv", type=Path, default=None,
                    help="recorded (path, md5) under 'What was run'")
    ap.add_argument("--run-device", type=Path, default=None,
                    help="part-1 run_device.tsv (index, run_name, train_device, ...); default "
                         "<eta_agg_dir>/run_device.tsv")
    a = ap.parse_args(argv)
    text = build(a.eta_agg_dir, a.h_agg_dir, a.old_grid_tsv, a.run_device)
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out_md.with_name(a.out_md.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(a.out_md)
    print(f"wrote {a.out_md} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
