"""t118 ladder — the per-rung markdown report, drawn from the pinned `results.json` + checks JSON.

    python tools/t118/ladder/report.py <agg_dir> <rung>

Writes `<agg_dir>/<rung>/report.md` with the sections Summary, Checks, Law test, Depth law,
Shuffle and swap, Figures, Runs, Choices. Runs under the candii env: numpy + stdlib, no
matplotlib (it reuses the labels and loaders of `figures.py`, whose plotting imports are lazy).

Rules the text follows (PI and AGENTS.md section 7.2):
- each check is a value against its bar with "met" / "unmet": a drafted reading for the PI,
  never ticked, never a verdict;
- the seed wobble stands beside every model number (references have no seed and say so);
- the CRPS split (oracle-scaled CRPS + scale error) stands beside every CRPS on all bins, and a
  CRPS whose split was not computed says so;
- hypotheses are named by their content, never by notebook ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:          # the pinned convention: tools/t118 on sys.path
    sys.path.insert(0, str(_T118))
from ladder import figures as F  # noqa: E402  (numpy-only at import; plotting imports are lazy)

SUMMARY_METRICS = ("crps_all", "crps_top1", "spearman_all")
NO_SPLIT = "split not computed on this subset"


def _mw(row):
    """'mean (seed wobble w)' for a per_class row."""
    if row is None or row.get("mean") is None:
        return "n/a"
    n = sum(v is not None for v in row.get("per_seed", []))
    extra = "" if n == len(F.SEEDS) else f", {n} of {len(F.SEEDS)} seeds"
    return f"{F.fmt(row['mean'])} (seed wobble {F.fmt(row['seed_wobble'])}{extra})"


def _split_index(res, rung):
    """(g_version, space, model, mark_class) -> (oracle_scaled, scale_error, wobble of each).

    From the trained/score per_pair records: mean over pairs per track, macro over the class's
    tracks per seed, then mean over seeds (the aggregation rule of the per-class table)."""
    acc: dict = {}
    for r in res.get("per_pair", []):
        if r.get("rung") != rung or r.get("kind") != "trained" or r.get("eval") != "score":
            continue
        if r.get("crps_oracle_scaled_all") is None or r.get("scale_error_all") is None:
            continue
        k = (r["g_version"], r["space"], r["model"], r["mark_class"], r["seed"], r["track"])
        acc.setdefault(k, []).append((r["crps_oracle_scaled_all"], r["scale_error_all"]))
    by_seed: dict = {}
    for (gv, space, model, cls, seed, track), vs in acc.items():
        by_seed.setdefault((gv, space, model, cls, seed), []).append(np.mean(vs, axis=0))
    by_class: dict = {}
    for (gv, space, model, cls, seed), vs in by_seed.items():
        by_class.setdefault((gv, space, model, cls), []).append(np.mean(vs, axis=0))
    out = {}
    for k, vs in by_class.items():
        a = np.asarray(vs)
        out[k] = (float(a[:, 0].mean()), float(a[:, 1].mean()),
                  float(np.ptp(a[:, 0])) if len(a) > 1 else 0.0,
                  float(np.ptp(a[:, 1])) if len(a) > 1 else 0.0)
    return out


def _split_text(split):
    if split is None:
        return "split: n/a"
    o, e, wo, we = split
    return (f"split: oracle-scaled {F.fmt(o)} (wobble {F.fmt(wo)}) + scale error {F.fmt(e)} "
            f"(wobble {F.fmt(we)})")


def _ref_cell(res, cls, space, ref, metric, tbc):
    v = F.ref_class(res, cls, space, ref, metric, tbc)
    if v is None:
        return "n/a"
    s = f"{F.fmt(v)} (reference, no seed)"
    if metric.startswith("crps_"):
        subset = metric.split("_", 1)[1]      # the baseline table carries a split per subset
        o = F.ref_class(res, cls, space, ref, f"crps_oracle_scaled_{subset}", tbc)
        e = F.ref_class(res, cls, space, ref, f"scale_error_{subset}", tbc)
        s += (f"; split: oracle-scaled {F.fmt(o)} + scale error {F.fmt(e)}" if o is not None
              and e is not None else "; split: not in the reference table")
    return s


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def section_summary(res, rung, idx, tbc, split):
    L = ["## Summary", "",
         f"Held-out chromosomes chr19 + chr21 (blacklist removed), trained pairs, all pairs kept "
         f"(variant `all`). Model values are the mean over the 3 seeds of the macro mean over "
         f"the class's tracks; the seed wobble is the largest pairwise difference over the "
         f"seeds. The twins are rung {rung}'s own. noSolution (X' = X) and QuantileMatching are "
         f"references only, never pass/fail. CRPS is the NB CRPS in counts space and the "
         f"log-normal CRPS in -log10 p space; lower is better. Spearman: higher is better.", ""]
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            L += [f"### {F.GV_NAME[gv]}, {F.SPACE_NAME[space]}", ""]
            rows = []
            for cls in F.CLASSES:
                for metric in SUMMARY_METRICS:
                    cells = [cls, F.METRIC_NAME[metric]]
                    cells += [_ref_cell(res, cls, space, ref, metric, tbc)
                              for ref in ("noSolution", "QuantileMatching")]
                    for model in ("nocov", "ids", "real"):
                        row = idx.get((rung, gv, space, model, "trained", cls, metric, "all"))
                        c = _mw(row)
                        if metric == "crps_all" and row is not None:
                            c += "; " + _split_text(split.get((gv, space, model, cls)))
                        elif metric == "crps_top1" and row is not None:
                            c += f"; {NO_SPLIT}"
                        cells.append(c)
                    rows.append(cells)
            L += _table(["class", "metric", "noSolution", "QuantileMatching",
                         "no-covariates twin", "labels-as-ids twin", f"real g, rung {rung}"],
                        rows)
            L.append("")
    L += ["### Without the pairs whose target equals the source", "",
          "Count-space pairs whose two products are byte-identical in that space are kept in the "
          "headline; this table drops them (variant `nonidentical`). "
          f"CRPS on all bins; {NO_SPLIT} (the split is scored on all pairs only).", ""]
    rows = []
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            for cls in F.CLASSES:
                cells = [F.GV_NAME[gv], F.SPACE_NAME[space], cls]
                for model in ("nocov", "real"):
                    cells.append(_mw(idx.get((rung, gv, space, model, "trained", cls, "crps_all",
                                              "nonidentical"))))
                rows.append(cells)
    L += _table(["g", "space", "class", "no-covariates twin", f"real g, rung {rung}"], rows)
    L.append("")
    return L


def section_checks(res, rung, checks):
    L = ["## Checks", "",
         "**Drafted reading for the PI — not ticked.** Each line is the value computed by the "
         "aggregator against its pre-registered bar; \"met\" / \"unmet\" is that comparison and "
         "nothing more. It is not a verdict, and no box in the notebook has been ticked.", "",
         "Gain checks: value = D_other - D_real (CRPS; positive favours the real g), bar = 2 x "
         "seed wobble of the real g (for \"beats the rung below\": 2 x the larger of the two "
         "wobbles). Depth law: value = the largest |predicted scale / depth ratio - 1| over the "
         "depth pairs, bar 0.10. Shuffle and swap gate only the main claim (one transformation f, "
         "chosen by g from the source and target covariates); they are reported for every rung.",
         ""]
    rows = []
    for c in F.sort_checks(checks):
        name = F.CHECK_NAME.get(c["check"], c["check"])
        below = (c.get("components") or {}).get("rung_below")
        if c["check"] == "beatsbelow" and below:
            name += f" ({F.RUNG_NAME.get(below, below)})"
        rows.append([name, F.GV_NAME.get(c["g_version"], c["g_version"]),
                     F.SPACE_NAME.get(c["space"], c["space"]), c["mark_class"],
                     F.METRIC_NAME.get(c["metric"], c["metric"]), F.fmt(c["value"]),
                     f"{F.CHECK_RULE.get(c['check'], '?')} {F.fmt(c['bar'])}",
                     F.fmt(c["seed_wobble"]), "met" if c["met"] else "unmet"])
    if rows:
        L += _table(["check", "g", "space", "class", "metric", "value", "bar", "seed wobble",
                     "drafted reading"], rows)
        n_met = sum(c["met"] for c in checks)
        L += ["", f"{n_met} of {len(checks)} checks met their bar (drafted, not ticked)."]
    else:
        L.append(f"No checks were found for rung {rung}.")
    L += ["", "### Which rung the main claim judges", "",
          "Rule: the lowest rung whose chr22 validation CRPS (all bins, real g, mean over seeds, "
          "macro over all tracks) is within the best rung's seed wobble of the best.", ""]
    rows = []
    for key, v in sorted(res.get("rung_choice", {}).items()):
        gv, _, space = key.partition("|")
        vals = "; ".join(f"{r}: {F.fmt(v['val_by_rung'].get(r))} (seed wobble "
                         f"{F.fmt(v.get('wobble_by_rung', {}).get(r))})"
                         for r in F.RUNGS if r in v.get("val_by_rung", {}))
        rows.append([F.GV_NAME.get(gv, gv), F.SPACE_NAME.get(space, space), v["chosen"],
                     "yes" if v["chosen"] == rung else "no", vals])
    if rows:
        L += _table(["g", "space", "chosen rung", f"is rung {rung}", "validation CRPS by rung"],
                    rows)
        L += ["", f"Validation CRPS is on chr22; {NO_SPLIT} (validation)."]
    else:
        L.append("No rung choice in results.json.")
    L.append("")
    return L


def section_law(res, rung, idx):
    L = ["## Law test", "",
         "Every never-trained arm -> arm pair within a track, scored on chr19 + chr21. A model "
         "that keeps one map per trained pair has no answer for these; the labels-as-ids twin "
         f"can match the real g on trained pairs, and only this test separates them. {NO_SPLIT} "
         "(law pairs).", ""]
    rows = []
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            for cls in F.CLASSES:
                for metric in ("crps_all", "crps_top1"):
                    cells = [F.GV_NAME[gv], F.SPACE_NAME[space], cls, F.METRIC_NAME[metric]]
                    cells += [_mw(idx.get((rung, gv, space, m, "law", cls, metric, "all")))
                              for m in ("nocov", "ids", "real")]
                    rows.append(cells)
    L += _table(["g", "space", "class", "metric", "no-covariates twin", "labels-as-ids twin",
                 f"real g, rung {rung}"], rows)
    L += ["", "### By knob combination", "",
          "Mean CRPS gain (all bins) of the real g over each twin, D_twin - D_real, averaged over "
          "the tracks and levels of each (source arm, target arm) combination; positive favours "
          "the real g. Per-combination seed wobble is not in results.json; the class-level "
          "law-test wobble is in the table above. Pairs whose target equals the source in that "
          "space are counted in the last column.", ""]
    grid = [r for r in res.get("law_grid", []) if r["rung"] == rung and r["metric"] == "crps_all"]
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            sub = [r for r in grid if r["g_version"] == gv and r["space"] == space]
            if not sub:
                continue
            acc: dict = {}
            for r in sub:
                a = acc.setdefault((r["arm_src"], r["arm_tgt"]), {"n": [], "i": [], "id": 0})
                a["n"].append(r["d_nocov"] - r["d_real"])
                a["i"].append(r["d_ids"] - r["d_real"])
                a["id"] += bool(r.get("identical_target"))
            rows = [[s, t, len(v["n"]), F.fmt(float(np.mean(v["n"]))),
                     F.fmt(float(np.mean(v["i"]))), v["id"]]
                    for (s, t), v in sorted(acc.items())]
            L += [f"#### {F.GV_NAME[gv]}, {F.SPACE_NAME[space]}", ""]
            L += _table(["source arm", "target arm", "pairs", "gain over no-covariates twin",
                         "gain over labels-as-ids twin", "identical-target pairs"], rows)
            L.append("")
    return L


def section_depth(res, rung, checks):
    L = ["## Depth law", "",
         "Counts space, pairs whose two products differ only in depth (trained base <-> depth "
         "pairs and never-trained depth <-> depth pairs). Closed-form expectation: the predicted "
         "total count scale log2(sum of predicted means / sum of X) equals the true log2 depth "
         "ratio. Error = |2^(predicted - true) - 1|, computed on the mean over seeds of the "
         "predicted log2 scale; the seed wobble is the largest over the pairs of the per-pair "
         "spread over the seeds.", ""]
    rows = [r for r in res.get("depth_law", []) if r["rung"] == rung]
    tr_cls = {}
    for r in res.get("per_track", []):
        tr_cls[r["track"]] = r["mark_class"]
    out = []
    for gv in F.G_VERSIONS:
        for model in F.MODELS:
            for kind in ("trained", "law"):
                for cls in F.CLASSES:
                    by: dict = {}
                    for r in rows:
                        if r["g_version"] == gv and r["model"] == model and r["kind"] == kind \
                                and tr_cls.get(r["track"]) == cls:
                            by.setdefault((r["track"], r["source_pid"], r["target_pid"]),
                                          []).append(r)
                    if not by:
                        continue
                    errs, wob = [], []
                    for rs in by.values():
                        t = rs[0]["log2_true"]
                        errs.append(abs(2 ** (np.mean([x["log2_pred"] for x in rs]) - t) - 1))
                        e = [abs(2 ** (x["log2_pred"] - t) - 1) for x in rs]
                        wob.append(max(e) - min(e))
                    out.append([F.GV_NAME[gv], F.MODEL_NAME[model], kind, cls, len(by),
                                F.fmt(float(np.median(errs))), F.fmt(float(np.max(errs))),
                                F.fmt(float(np.max(wob)))])
    if out:
        L += _table(["g", "model", "pairs", "class", "n pairs", "median error", "max error",
                     "seed wobble"], out)
    else:
        L.append(f"No depth-law rows for rung {rung}.")
    dl = [c for c in checks if c["check"] == "depthlaw"]
    if dl:
        L += ["", "Drafted reading (not ticked): " + "; ".join(
            f"{F.GV_NAME[c['g_version']]}, {c['mark_class']}: {F.fmt(c['value'])} vs bar "
            f"{F.fmt(c['bar'])} (seed wobble {F.fmt(c['seed_wobble'])}) — "
            f"{'met' if c['met'] else 'unmet'}" for c in F.sort_checks(dl)) + "."]
    L.append("")
    return L


def section_shuffle_swap(res, rung, idx, checks):
    L = ["## Shuffle and swap", "",
         "**Shuffle**: C' is replaced by the C' of another arm of the same track whose target "
         "differs in that space. If g reads the covariates, the real g's advantage over the "
         "no-covariates twin should fall to within 2 x seed wobble. "
         f"CRPS on held-out chromosomes; {NO_SPLIT} (shuffled pairs).", ""]
    rows = []
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            for cls in F.CLASSES:
                for metric in ("crps_all", "crps_top1"):
                    rows.append([F.GV_NAME[gv], F.SPACE_NAME[space], cls, F.METRIC_NAME[metric],
                                 _mw(idx.get((rung, gv, space, "real", "trained", cls, metric,
                                              "all"))),
                                 _mw(idx.get((rung, gv, space, "real", "shuffle", cls, metric,
                                              "all"))),
                                 _mw(idx.get((rung, gv, space, "nocov", "trained", cls, metric,
                                              "all")))])
    L += _table(["g", "space", "class", "metric", "real g, true C'", "real g, wrong C'",
                 "no-covariates twin"], rows)
    L += ["", "**Swap**: C' = C. A g that reads the covariates should leave X unchanged: median "
              "|log(predicted mean / X)| over bins with X > 0, bar 0.1.", ""]
    acc: dict = {}
    for r in res.get("per_pair", []):
        if r.get("rung") == rung and r.get("kind") == "swap" and r.get("model") == "real" \
                and r.get("swap_median_abs_log_ratio") is not None:
            acc.setdefault((r["g_version"], r["space"], r["mark_class"], r["seed"], r["track"]),
                           []).append(r["swap_median_abs_log_ratio"])
    by_seed: dict = {}
    for (gv, space, cls, seed, track), vs in acc.items():
        by_seed.setdefault((gv, space, cls, seed), []).append(float(np.mean(vs)))
    by_cls: dict = {}
    for (gv, space, cls, seed), vs in by_seed.items():
        by_cls.setdefault((gv, space, cls), []).append(float(np.mean(vs)))
    rows = []
    for gv in F.G_VERSIONS:
        for space in F.SPACES:
            for cls in F.CLASSES:
                v = by_cls.get((gv, space, cls))
                if v:
                    rows.append([F.GV_NAME[gv], F.SPACE_NAME[space], cls,
                                 f"{F.fmt(float(np.mean(v)))} (seed wobble "
                                 f"{F.fmt(max(v) - min(v))})"])
    if rows:
        L += _table(["g", "space", "class", "median |log(mean / X)|, real g"], rows)
    else:
        L.append("No swap records in results.json for this rung.")
    ss = [c for c in checks if c["check"] in ("shuffle", "swap")]
    n_met = sum(c["met"] for c in ss)
    L += ["", f"Shuffle and swap checks: {n_met} of {len(ss)} met their bar (drafted, not "
              "ticked; see the Checks table)."]
    L.append("")
    return L


FIG_CAPTION = {
    "fig1_ladder": "Ladder: noSolution, both twins, A-D and QuantileMatching per mark class and "
                   "space; bars = seed range.",
    "fig2_knob_heatmap": "Relative CRPS gain over the no-covariates twin, per arm and rung.",
    "fig3_law_grid": "Law test grid: gain over each twin on every never-trained arm -> arm pair.",
    "fig4_depth_law": "Depth law: true log2 depth ratio against the predicted log2 count scale.",
    "fig5_learned_f": "The learned f made visible.",
    "fig6_meta_profiles": "Peak meta-profiles, ±2 kb around the top bins of X'.",
    "fig7_snippets": "Track snippets over 10 kb loci picked by the fixed rule.",
    "fig8_calibration": "Calibration (secondary): PIT histograms per mark class.",
    "fig9_checks": "Checks card: value against bar per check, per version of g (not ticked).",
}


def section_figures():
    L = ["## Figures", ""]
    for i, name in enumerate(F.FIG_NAMES, start=1):
        L += [f"**Figure {i}.** {FIG_CAPTION[name]}", "", f"![{name}](figures/{name}.png)", ""]
    return L


def _timing(res, agg_dir, rung):
    """phase -> [seconds] over the rung's present runs (timing.json, scores.json, law.json)."""
    acc: dict = {}
    for name in res.get("runs_present", []):
        if not name.startswith(f"{rung}_"):
            continue
        rd = F.run_dir(res, agg_dir, name)
        if rd is None:
            continue
        for fname, prefix in (("timing.json", "train"), ("scores.json", "score"),
                              ("law.json", "law")):
            p = rd / fname
            if not p.is_file():
                continue
            try:
                with open(p) as fh:
                    obj = json.load(fh)
            except (OSError, ValueError):
                continue
            t = obj if fname == "timing.json" else (obj.get("timing", {})
                                                    if isinstance(obj, dict) else {})
            for k, v in (t or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc.setdefault(f"{prefix}: {k}", []).append(float(v))
    return acc


def section_runs(res, agg_dir, rung):
    present = [n for n in res.get("runs_present", []) if n.startswith(f"{rung}_")]
    missing = [n for n in res.get("runs_missing", []) if n.startswith(f"{rung}_")]
    L = ["## Runs", "",
         f"{len(present)} runs present and {len(missing)} missing for rung {rung} (of "
         f"{len(present) + len(missing)} expected). Runs directory: `{res.get('runs_dir')}`. "
         f"results.json written {res.get('created_utc')}.", ""]
    if missing:
        L += ["Missing: " + ", ".join(f"`{n}`" for n in missing) + ".", ""]
    t = _timing(res, agg_dir, rung)
    if t:
        rows = [[k, len(v), F.fmt(float(np.mean(v))), F.fmt(float(np.max(v)))]
                for k, v in sorted(t.items())]
        L += _table(["phase", "runs", "mean seconds (or MB for memory)", "max"], rows)
    else:
        L.append("No timing files found.")
    L.append("")
    return L


def section_choices(res, agg_dir, rung):
    L = ["## Choices", "",
         "Routine training settings (the agent's call, reported with the run), read from "
         "`config.json` of one run:", ""]
    names = sorted(n for n in res.get("runs_present", []) if n.startswith(f"{rung}_"))
    names = sorted(names, key=lambda n: ("_real_" not in n, n))
    cfg, src = None, None
    for n in names:
        rd = F.run_dir(res, agg_dir, n)
        if rd is not None and (rd / "config.json").is_file():
            with open(rd / "config.json") as fh:
                cfg, src = json.load(fh), n
            break
    if cfg is None:
        L += ["No config.json found for this rung.", ""]
        return L
    wanted = (("steps", ("max_steps", "steps")), ("learning rate", ("lr", "learning_rate")),
              ("window (bins)", ("window", "window_bins")), ("batch", ("batch", "batch_size")))
    rows = []
    for label, keys in wanted:
        v = next((cfg[k] for k in keys if k in cfg), "not recorded in config.json")
        rows.append([label, v])
    shown = {k for _, ks in wanted for k in ks}
    rows += [[k, v] for k, v in sorted(cfg.items())
             if k not in shown and isinstance(v, (int, float, str, bool)) and len(str(v)) < 80]
    L += [f"Source: `{src}/config.json`.", ""]
    L += _table(["setting", "value"], rows)
    L.append("")
    return L


def build_report(agg_dir, rung) -> str:
    agg_dir = Path(agg_dir)
    res = F.load_results(agg_dir)
    idx, tbc = F.class_index(res), F.class_tracks(res)
    checks = F.load_checks(agg_dir, rung, res)
    split = _split_index(res, rung)
    L = [f"# Rung {rung}: {F.RUNG_NAME[rung]}", "",
         f"Question: can one transformation f, which a small network g chooses from the "
         f"covariates of the source track (C) and of the wanted track (C'), turn X into X' — "
         f"here with f in the form of {F.RUNG_NAME[rung]}. Each result is compared with two "
         f"twins trained the same way: the no-covariates twin (C, C' redrawn at every step, so "
         f"they carry no information) and the labels-as-ids twin (one fixed permutation of C, "
         f"C', so a label still names its pair). Split: train on every chromosome except chr19, "
         f"chr21, chr22; chr22 validates; chr19 + chr21 score.", "",
         "Nothing in this report is a verdict. The checks are drafted readings for the PI and "
         "are not ticked in the notebook.", ""]
    L += section_summary(res, rung, idx, tbc, split)
    L += section_checks(res, rung, checks)
    L += section_law(res, rung, idx)
    L += section_depth(res, rung, checks)
    L += section_shuffle_swap(res, rung, idx, checks)
    L += section_figures()
    L += section_runs(res, agg_dir, rung)
    L += section_choices(res, agg_dir, rung)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agg_dir")
    ap.add_argument("rung", choices=F.RUNGS)
    a = ap.parse_args(argv)
    out = Path(a.agg_dir) / a.rung / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(a.agg_dir, a.rung))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
