"""Rebuild index.html from deck_src.html: the close-out deck of the counterfactual in-vitro
covariate testbed (anchor q1, task t143; PI request 2026-10-30).

Paper arc (prezit references/paper-arc.md). Every number on a slide is read here from a vault
file and embedded as DATA; the page types none. The build writes the deck-sources manifest
(vault-relative path -> sha256 of every file it read) so `crux deck --verify` can prove the
deck is current.

Two kinds of source:
  results/<tid>/...   result tables (gitignored, so they are synced into this vault first;
                      see SYNC below)
  <node>.md           tracked vault nodes (q1's subtree) whose "Answer so far" / "Findings"
                      carry the earlier readings; a regex reads each number, and the build
                      fails if a sentence it reads has changed shape

Refresh after new readings (for example when the arm-to-arm grid's second arm finishes):

    CLOSEOUT_SYNC_FROM=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results \
      /Users/mforooz/miniforge3/envs/candii/bin/python build_deck.py [VAULT]

With CLOSEOUT_SYNC_FROM set, the build first copies every SYNC file from that results dir into
VAULT/results/ (untracked; never commit results/). VAULT defaults to the vault this file sits in.
Stdlib only.
"""
import csv, glob, hashlib, json, os, pathlib, re, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
SYNC = ["t112/MANIFEST.tsv",
        "t130/blind_table.tsv", "t130/runs.tsv",
        "t136/checks_readings.tsv", "t136/m5_depthlaw_table.tsv", "t136/m1_heldout_table.tsv",
        "t136/n_table.tsv",
        "t139/knob_vs_floor.tsv", "t139/pairs.tsv",
        "t142/floor_at_30M.tsv", "t142/compare_real_class.tsv", "t142/map_classes.tsv",
        "t142/t141_readthin/knob_classes.tsv"]
src_dir = os.environ.get("CLOSEOUT_SYNC_FROM")
if src_dir:
    for rel in SYNC:
        dst = VAULT / "results" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pathlib.Path(src_dir) / rel, dst)

read = []


def path(rel):
    p = VAULT / rel
    read.append(p)
    return p


def tsv(rel):
    with open(path("results/" + rel), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def node(stem):
    """A tracked vault node by its id prefix (q7 -> q7_why_do_...md)."""
    hits = [p for p in glob.glob(str(VAULT / f"{stem}_*.md"))]
    assert len(hits) == 1, (stem, hits)
    return path(os.path.relpath(hits[0], VAULT)).read_text(encoding="utf-8")


def grab(text, pat, label):
    """The capture groups of the first match, as strings exactly as the vault writes them."""
    m = re.search(pat, text)
    assert m, f"{label}: the sentence this deck reads has changed; re-read and fix the regex"
    return list(m.groups())


def f(v):
    return None if v in (None, "", "nan", "NA") else float(v)


R4 = lambda v: None if v is None else round(v, 4)

# ---------------- design: the corpus ----------------
man = tsv("t112/MANIFEST.tsv")
arms = [r for r in man if r["arm"] != "base"]
knobs = {}
for r in arms:
    k = knobs.setdefault(r["arm"], {"levels": [], "route": r["route"], "knob": r["knob"].split(".")[-1]})
    if r["level"] not in k["levels"]:
        k["levels"].append(r["level"])
design = {"n_products": len(man), "n_bases": len(man) - len(arms), "n_arms": len(arms),
          "n_tracks": len({r["track"] for r in man}), "n_knobs": len(knobs),
          "assays": sorted({r["assay"] for r in man}),
          "knobs": knobs,
          "n_ladder_runs": len(tsv("t130/runs.tsv")),
          "n_knob_pairs": len(tsv("t139/pairs.tsv"))}

# ---------------- (a) the main claim, from h15 / h1 ----------------
h15 = node("h15")
g = grab(h15, r"median of (\d+) configurations, trained / never-trained one knob / two knobs: "
              r"counts (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?); p (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?)", "h15 gains")
tw = grab(h15, r"Beats the no-covariates twin: (\d)–(\d) of (\d) cells", "h15 twin")
sw = grab(h15, r"median \|log\(mean / X\)\| (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) in counts, (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) in p \(bar (\d+(?:\.\d+)?)\)",
          "h15 swap")
h1n = node("h1")
h1g = grab(h1n, r"(\d+(?:\.\d+)?) on trained pairs \((\d+) of (\d+) clear 2 × seed \|Δ\|\) and (\d+(?:\.\d+)?) on never-trained "
                r"depth → depth pairs \((\d+) of (\d+)\)", "h1 twin gain")
q2 = node("q2")
q2g = grab(q2, r"median gain (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) against a seed wobble of (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)\)", "q2 gain")
q3 = node("q3")
q3g = grab(q3, r"by a median (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) on trained pairs, and by (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) on never-trained pairs.*?"
               r"seed wobbles of (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)", "q3 gain")
main = {"n_cfg": g[0], "counts": g[1:4], "p": g[4:7], "twin_cells": tw, "swap": sw,
        "h1": h1g, "q2": q2g, "q3": q3g}

# ---------------- (b, c) the blind check set (h1 depth, h2 run type) ----------------
bt = tsv("t130/blind_table.tsv")


def blind(knob, direction):
    out = []
    for r in bt:
        if r["space"] == "counts" and r["knob"] == knob and r["direction"] == direction and r["subset"] == "all" \
                and r["mark_class"] in ("DNase", "narrow", "broad"):
            interval = f(r["d_onewarp"]) - f(r["d_oracle"]) if knob == "depth" else None
            out.append({"cls": r["mark_class"], "rung": r["rung"], "g": r["g_version"],
                        "gc": R4(f(r["gapclosed_onewarp"])) if knob == "depth" else None,
                        "noise_gc": R4(2 * f(r["seed_abs_delta"]) / interval) if interval else None,
                        "gain": R4(f(r["gain_over_onewarp"])), "seed2": R4(2 * f(r["seed_abs_delta"])),
                        "twin_gain": R4(f(r["gain_over_nocov"])),
                        "d_model": R4(f(r["d_model"])), "os": R4(f(r["crps_oracle_scaled_all"])),
                        "se": R4(f(r["scale_error_all"])),
                        "d_onewarp": R4(f(r["d_onewarp"])), "ow_os": R4(f(r["onewarp_oracle_scaled_all"])),
                        "ow_se": R4(f(r["onewarp_scale_error_all"])),
                        "d_oracle": R4(f(r["d_oracle"])), "or_os": R4(f(r["oracle_crps_oracle_scaled_all"])),
                        "or_se": R4(f(r["oracle_scale_error_all"])), "oracle": r["oracle_kind"],
                        "d_floor": R4(f(r["d_floor"]))})
    return out


h1b = blind("depth", "base_to_arm")
h2b = blind("pe", "both")

# ---------------- (d) capacity and the depth offset, from q4 / q5 ----------------
q5 = node("q5")
q5a = grab(q5, r"Per-track A, DNase: (\d+(?:\.\d+)?) ± (\d+(?:\.\d+)?) \((\d+(?:\.\d+)?) \+ (\d+(?:\.\d+)?)\) against (\d+(?:\.\d+)?) ± (\d+(?:\.\d+)?) "
               r"\((\d+(?:\.\d+)?) \+ (\d+(?:\.\d+)?)\); gain (\d+(?:\.\d+)?), bar (\d+(?:\.\d+)?)", "q5 A")
q5c = grab(q5, r"(\d+) of (\d+) CRPS cells clear, all A-type", "q5 cells")
q5bd = grab(q5, r"B–D2 clear (\d+) of (\d+)", "q5 B-D")
q5law = grab(q5, r"bar (\d+(?:\.\d+)?)\): (\d+) of (\d+) cells within", "q5 law")
q4 = node("q4")
q4n = grab(q4, r"relative count gain over the twin falls from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) on same-knob pairs", "q4 never")
q4x = grab(q4, r"\(CRPS > (\d+);", "q4 explode")
cap = {"q5": q5a, "cells": q5c, "bd": q5bd, "law": q5law, "q4never": q4n, "explode": q4x}

# ---------------- (e) the depth-law miss, from q7 ----------------
q7 = node("q7")
law = {
    "data": grab(q7, r"The data obey the law within (\d+(?:\.\d+)?)% over (\d+) ordered pairs", "q7 data"),
    "trained": grab(q7, r"Trained pairs fit \((\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)\); never-trained arm → arm pairs miss "
                        r"\((\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)\)", "q7 trained"),
    "a0": grab(q7, r"misses \((\d+(?:\.\d+)?)–(\d+(?:\.\d+)?), trained (\d+(?:\.\d+)?)\)", "q7 A0"),
    "pois": grab(q7, r"\(A (\d+(?:\.\d+)?), A0 (\d+(?:\.\d+)?), D (\d+(?:\.\d+)?); seed wobble (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)\)", "q7 Poisson"),
    "encd": grab(q7, r"Encoder-decoder D gives (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?), old D (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)", "q7 encdec"),
    "sym": grab(q7, r"gives (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) in its two complete cross-loss settings", "q7 sym"),
    "via": grab(q7, r"meets the law: (\d+(?:\.\d+)?) \(across-track g\) and (\d+(?:\.\d+)?) \(per-track g, seed spread (\d+(?:\.\d+)?)\) "
                    r"against direct (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)", "q7 via"),
    "size": grab(q7, r"wider g misses on never-trained depth pairs \((\d+(?:\.\d+)?) per-track, (\d+(?:\.\d+)?) across\), and so "
                     r"does a 2× wider f \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?);", "q7 size"),
}

# ---------------- (f) arm-to-arm training (h28), from t136 ----------------
m5 = tsv("t136/m5_depthlaw_table.tsv")
m1 = tsv("t136/m1_heldout_table.tsv")
nt = {r["arm"]: r for r in tsv("t136/n_table.tsv")}
ck = tsv("t136/checks_readings.tsv")
h28 = {"law": {}, "crps": {}, "n": {}, "checks": [], "rule": None}
for r in m5:
    if r["scope"] == "all7" and r["class"] == "all" and r["group"] in ("heldout_r2", "trained_r4", "trained_base"):
        h28["law"].setdefault(r["group"], {})[r["arm"]] = {
            "v": R4(f(r["median_abs_err"])), "sr": R4(f(r["seed_spread"])), "complete": r["complete"] == "True"}
for r in m1:
    if r["scope"] == "all7" and r["subgroup"] == "heldout75" and r["knob"] == "all":
        h28["crps"].setdefault(r["class"], {})[r["arm"]] = {
            "v": R4(f(r["crps_all"])), "os": R4(f(r["crps_oracle_scaled_all"])), "se": R4(f(r["scale_error_all"])),
            "sr": R4(f(r["seed_spread"])), "pairs": int(r["n_pairs"]), "complete": r["complete"] == "True"}
for a, r in nt.items():
    h28["n"][a] = {"assigned": int(r["assigned"]), "law_done": int(r["LAW_DONE"] or 0),
                   "train_done": int(r["TRAIN_DONE"] or 0)}
for r in ck:
    if r["check"] == "rule":
        h28["rule"] = r["reading"]
    elif r["item"] in ("arm reading", "check reading") or r["check"] in ("c1", "c3") or \
            (r["check"] == "c4" and r["item"] != "check reading"):
        h28["checks"].append({"c": r["check"], "name": r["name"], "kind": r["kind"], "arm": r["arm"],
                              "item": r["item"], "value": R4(f(r["value"])), "thr": r["threshold"],
                              "reading": r["reading"]})

# ---------------- (g) counts -> p, from q6 ----------------
q6 = node("q6")
hmap = {"repair": grab(q6, r"in (\d+) of (\d+) rows \((\d+) clear; D2, MLP rung 5: (\d+(?:\.\d+)?) ± (\d+(?:\.\d+)?) "
                           r"\((\d+(?:\.\d+)?) \+ (\d+(?:\.\d+)?)\) against (\d+(?:\.\d+)?) \((\d+(?:\.\d+)?) \+ (\d+(?:\.\d+)?)\)", "q6 repair"),
        "best": grab(q6, r"formula \+ MLP (\d+(?:\.\d+)?) ± (\d+(?:\.\d+)?)", "q6 best"),
        "noexplode": grab(q6, r"Push and formula h have (\d+) records with CRPS > (\d+)", "q6 explode")}

# ---------------- (h) knob ranking against the floor, from t139 ----------------
kv = tsv("t139/knob_vs_floor.tsv")
knob = [{"cls": r["mark_class"], "arm": r["arm"], "level": r["level"], "metric": r["metric"],
         "v": R4(f(r["value"])), "floor": f(r["floor"]), "fsr": f(r["floor_seed_spread"]),
         "above": r["above_floor"] == "True", "ratio": R4(f(r["ratio_to_floor"]))}
        for r in kv if r["space"] == "counts" and r["metric"] in ("abs_log2fc_top1", "shape_1k")]

# ---------------- (i) the resampling floor and where it sits, from t142 ----------------
fl = {}
for r in tsv("t142/floor_at_30M.tsv"):
    if r["metric"] in ("shape_1k", "abs_log2fc_top1"):
        fl.setdefault(r["metric"], {})[r["mark_class"]] = {
            "n": int(r["n_products"]), "30M": R4(f(r["value_30M_extrap"])), "15M": R4(f(r["value_15M"])),
            "7.5M": R4(f(r["value_7.5M"])), "3.75M": R4(f(r["value_3.75M"])), "1.875M": R4(f(r["value_1.875M"])),
            "sr05": f(r["seed_spread_r0.5"]), "sr0125": f(r["seed_spread_r0.125"])}
cmp_ = {r["mark_class"]: {"read": R4(f(r["median_real_over_read"])), "unit": R4(f(r["median_real_over_unit"])),
                          "n": int(r["n"])}
        for r in tsv("t142/compare_real_class.tsv") if r["metric"] == "shape_1k"}
mp = {}
for r in tsv("t142/map_classes.tsv"):
    if r["reading"] in ("R2_level", "delta_R2", "delta_R2_shuffle"):
        mp.setdefault(r["mark_class"], {}).setdefault(r["depth"], {})[r["reading"]] = {
            "v": f(r["median"]), "lo": f(r["min"]), "hi": f(r["max"])}
floor = {"curve": fl, "real": cmp_, "map": mp}

# ---------------- (j) where each knob acts, from t142/t141_readthin ----------------
kc = tsv("t142/t141_readthin/knob_classes.tsv")
where = [{"arm": r["arm"], "level": r["level"], "cls": r["mark_class"], "space": r["space"], "series": r["series"],
          "v": f(r["median"]), "lo": f(r["min"]), "hi": f(r["max"])}
         for r in kc if r["reading"] in ("delta_R2",) and r["series"] in ("arm", "null_s0")]
shuf = max(f(r["max"]) for r in kc if r["reading"] == "delta_R2_shuffle")

# ---------------- (k) the trained CANDI probe, from q8 / h26 ----------------
q8 = node("q8")
h26 = node("h26")
probe = {"slope": grab(q8, r"told-depth slope (\d+(?:\.\d+)?)", "q8 slope"),
         "dir": grab(q8, r"run_type direction score is (\d+(?:\.\d+)?)", "q8 dir"),
         "swap": grab(h26, r"`swapwired`: (\d+) of (\d+) rows", "h26 swap"),
         "depth": grab(h26, r"`depthcontrol`: part \(2\) holds on (\d+) of (\d+) rows", "h26 depth"),
         "effect": grab(h26, r"`effectfloor`: (\d+) of (\d+)\.", "h26 effect")}

data = {"design": design, "main": main, "h1": h1b, "h2": h2b, "cap": cap, "law": law, "h28": h28,
        "hmap": hmap, "knob": knob, "floor": floor, "where": where, "shuf": shuf, "probe": probe}

# ---------------- manifest and page ----------------
src_path = HERE / "deck_src.html"
read += [pathlib.Path(__file__).resolve(), src_path]
sources = {}
for p in read:
    sources[p.relative_to(VAULT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
src = src_path.read_bytes().decode("utf-8")
assert src.count("__DATA__") == 1 and src.count("__SOURCES__") == 1
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"wrote {HERE / 'index.html'}: {len(sources)} source file(s); h28 rule: {h28['rule']}; "
      f"Darm2 law done {h28['n'].get('Darm2', {}).get('law_done')} of {h28['n'].get('Darm2', {}).get('assigned')}")
