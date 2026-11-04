"""Rebuild index.html from deck_src.html: the close-out deck of the whole covariate artifact
correction (CAC) testbed, final experiments included (anchor q1, tasks t143 and t146; PI requests
2026-10-30 and 2026-11-03).

Paper arc (prezit references/paper-arc.md). Every number on a slide is read here from a vault
file and embedded as DATA; the page types none. The build writes the deck-sources manifest
(vault-relative path -> sha256 of every file it read) so `crux deck --verify` can prove the
deck is current.

Two kinds of source:
  results/<tid>/...   result tables (gitignored, so they are synced into this vault first;
                      see SYNC below)
  <node>.md           tracked vault nodes (q1's subtree and its findings f1-f32) whose readings
                      carry the earlier numbers; a regex reads each number, and the build
                      fails if a sentence it reads has changed shape
  wiki/, notes/       the ENCODE Imputation Challenge page (motivation) and
                      notes/mini_candi_design_inputs.md (reprocessing cost)

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
        "t142/t141_readthin/knob_classes.tsv",
        "t146/analyze_out.txt", "t146/checks_h29.tsv", "t146/checks_h30_cells.tsv", "t146/checks_h30_pooled.tsv",
        "t146/checks_h31.tsv", "t146/checks_h31_per_track_seed.tsv", "t146/depthlaw.tsv", "t146/pooled.tsv",
        "t146/sacct_per_arm.tsv"]
src_dir = os.environ.get("CLOSEOUT_SYNC_FROM")
if src_dir:
    for rel in SYNC:
        dst = VAULT / "results" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (pathlib.Path(src_dir) / rel).exists():     # a missing source fails later, at its read
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
          "track_assay": {r["track"]: f'{r["assay"]}, cell type {r["cell"]}' for r in man},
          "knobs": knobs,
          "n_ladder_runs": len(tsv("t130/runs.tsv")),
          "n_knob_pairs": len(tsv("t139/pairs.tsv"))}

# ---------------- (a) the main claim, from h15 / h1 ----------------
h15 = node("h15")
g = grab(h15, r"Median gain of (\d+) configurations, trained / one knob / two knobs: "
              r"counts (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?); p (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?)", "h15 gains")
tw = grab(node("f8"), r"beat the scrambled twin in (\d)–(\d) of (\d) mark-class cells", "f8 twin")
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
NUM = r"(\d+(?:\.\d+)?)"
law = {
    "data": grab(q7, r"The data obey the law within (\d+(?:\.\d+)?)% over (\d+) ordered pairs", "q7 data"),
    "trained": grab(node("f19"), r"trained pairs fit the law \(" + NUM + "–" + NUM + r"\) while never-trained arm → arm pairs miss it \("
                    + NUM + "–" + NUM + r"\)", "f19 trained"),
    "a0": grab(node("f1"), r"rung A0 reads " + NUM + r" \(NB loss\) and " + NUM + r" \(Poisson-mean loss\) against " + NUM, "f1 A0"),
    "pois": grab(node("f3"), r"never-trained depth-law miss \(A0 " + NUM + ", D " + NUM + "; trained " + NUM + "–" + NUM + r"\)", "f3 Poisson"),
    "wob": grab(q7, r"the loss\]\] \(seed wobble " + NUM + "–" + NUM + r"\)", "q7 loss wobble"),
    "encd": grab(node("f20"), r"stays at " + NUM + "–" + NUM + r" on never-trained pairs, within noise of old D's " + NUM + "–" + NUM, "f20 encdec"),
    "sym": grab(node("f21"), r"settings read " + NUM + "–" + NUM + " on never-trained pairs", "f21 sym"),
    "via": grab(q7, r"gives " + NUM + r" \(across-track g\) and " + NUM + r" \(per-track g, seed spread " + NUM + r"\) against direct "
                    + NUM + " and " + NUM, "q7 via"),
    "size": grab(node("f2"), r"misses the law on never-trained depth pairs \(" + NUM + " per-track, " + NUM + " across-track", "f2 wide g")
            + grab(node("f4"), r"misses the law on never-trained depth pairs \(" + NUM + " per-track, " + NUM + " across-track", "f4 wide f"),
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
        "formula": grab(q6, r"fragment formula (\d+(?:\.\d+)?)", "q6 formula"),
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
         "swap": grab(h26, r"`swapwired` is met \((\d+) of (\d+) rows", "h26 swap"),
         "depth": grab(h26, r"part \(2\): (\d+) of (\d+)", "h26 depth"),
         "effect": grab(h26, r"`effectfloor` (\d+) of (\d+): 36 bp", "h26 effect"),
         "level": grab(h26, r"told 36 bp, the chr21 level moves (-?\d\.\d+) to (-?\d\.\d+) log2 \(real (-?\d\.\d+) to (-?\d\.\d+), "
                            r"mostly the bin-count rule\); told PE, (-?\d\.\d+) to (-?\d\.\d+) for ChIP and \+(\d\.\d+) to \+(\d\.\d+) "
                            r"for DNase \(real (-?\d\.\d+) to \+(\d\.\d+)\)", "h26 told level")}

# ---------------- motivation: the ENCODE Imputation Challenge, from the wiki ----------------
def text(rel):
    return path(rel).read_text(encoding="utf-8")


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
eicw = text("wiki/encode-imputation-challenge.md")
eic = {"n": int(grab(eicw, r"(\d+) models were submitted", "EIC entrants")[0]),
       "beat": WORDS[grab(eicw, r"outperformed all but (\w+) submissions", "EIC baseline")[0]],
       "cells": grab(eicw, r"only (\d+) of the (\d+) test cell types had more than two", "EIC cell types")}


def fm(stem):
    """direction and confidence of a finding node, from its frontmatter"""
    t = node(stem)
    return {k: (re.search(rf"^{k}: (.*)$", t, re.M) or [None, ""])[1].strip() for k in ("direction", "confidence")}


signed = {k: fm(k) for k in ("f5", "f6", "f7", "f30", "f31", "f32")}

# ---------------- (m)-(o) the final experiments, results/t146 ----------------
ao = text("results/t146/analyze_out.txt")
AT = {}
tbl = ao[ao.index("arm table"):ao.index("primary + checks")]
for m in re.finditer(r"^  (\w+)\s+w(-?[\d.]+)\s+(.*)$", tbl, re.M):
    arm, w, rest = m.groups()
    for g_ in re.finditer(r"(\w+) (\d\.\d+) \[(\d\.\d+)/(\d\.\d+)\] sp (\d\.\d+)", rest):
        AT.setdefault(f"{arm}@{w}", {})[g_[1]] = {"v": float(g_[2]), "os": float(g_[3]), "se": float(g_[4]), "sr": float(g_[5])}
assert "alone@-1" in AT and "xassay@1" in AT, "t146 arm table changed shape"
ck29 = {}
for r in tsv("t146/checks_h29.tsv"):
    ck29[f"{r['arm']}|{r['check']}|{r['group']}"] = {"x": f(r["x"]), "y": f(r["y"]), "d": f(r["delta"]), "bar": f(r["bar"]),
                                                      "pass": r["passes"] == "True", "dos": f(r["d_oracle_scaled"]),
                                                      "dse": f(r["d_scale_error"])}
dl = {f"{r['arm']}|{r['route']}|{r['dgroup']}": {"v": f(r["median_abs_err"]), "sr": f(r["seed_spread"])}
      for r in tsv("t146/depthlaw.tsv")}
cells30 = [{"src": r["src"], "tgt": r["tgt"], "cross": r["cross"] == "True", "gain": f(r["gain"]), "bar": f(r["bar"]),
            "win": r["win"] == "True", "go": f(r["gain_oracle"]), "bo": f(r["bar_oracle"]), "winmix": r["win_mix"] == "True"}
           for r in tsv("t146/checks_h30_cells.tsv")]
p30 = {}
for r in tsv("t146/checks_h30_pooled.tsv"):
    if r["group"] == "never":
        p30[f"{r['arm']}|{r['route']}"] = {"v": f(r["crps_all"]), "os": f(r["crps_oracle_scaled_all"]),
                                            "se": f(r["scale_error_all"]), "sr": f(r["crps_all_spread"]), "n": int(r["n_seeds"])}
seeds30 = [float(x) for x in grab(ao, r"paired direct-chain alone never: per seed \[([^\]]*)\]", "t146 per-seed gain")[0].split(",")]
agree = grab(ao, r"reading consist direct - alone direct never: (-?\d\.\d+) bar (\d\.\d+)", "t146 agreement reading")
n1 = {"s3": grab(ao, r"n1 trained base<->arm alone s0-2: (\d\.\d+) \(oracle_scaled (\d\.\d+), scale_error (\d\.\d+); spread (\d\.\d+)\); "
                     r"old D (\d\.\d+) -> gap ([+-]\d\.\d+) \(bar (\d\.\d+)\)", "t146 n1 three seeds"),
      "s6": grab(ao, r"n1 6-seed exact gap (-?\d\.\d+)", "t146 n1 six seeds")}
# h31: lowest-mappability decile, per-track table, comparator seeds 0-2
pts = tsv("t146/checks_h31_per_track_seed.tsv")


def dec0(arm, kl):
    rs = [r for r in pts if r["arm"] == arm and r["kl"] == kl and (arm != "alone" or int(r["seed"]) < 3)]
    seeds = sorted({r["seed"] for r in rs})
    per_seed = [sum(f(r["crps_dec0"]) for r in rs if r["seed"] == s_) / sum(1 for r in rs if r["seed"] == s_) for s_ in seeds]
    mean = lambda k: sum(f(r[k]) for r in rs) / len(rs)
    mu = sum(per_seed) / len(per_seed)
    sd = (sum((x - mu) ** 2 for x in per_seed) / (len(per_seed) - 1)) ** 0.5
    return {"v": R4(mu), "os": R4(mean("crps_oracle_scaled_dec0")), "se": R4(mean("scale_error_dec0")), "sd": R4(sd),
            "sr": R4(max(per_seed) - min(per_seed)), "n_tracks": len({r["track"] for r in rs})}


def track_gain(arm, kl):
    out = []
    for t in sorted({r["track"] for r in pts if r["kl"] == kl}):
        a = [f(r["crps_dec0"]) for r in pts if r["arm"] == "alone" and r["kl"] == kl and r["track"] == t and int(r["seed"]) < 3]
        b = [f(r["crps_dec0"]) for r in pts if r["arm"] == arm and r["kl"] == kl and r["track"] == t]
        ma = sum(a) / len(a)
        sd = (sum((x - ma) ** 2 for x in a) / (len(a) - 1)) ** 0.5
        out.append({"t": t, "gain": R4(ma - sum(b) / len(b)), "bar": R4(2 * sd)})
    return out


h31 = {"dec0": {kl: {a: dec0(a, kl) for a in ("alone", "map", "mapshift", "dna")} for kl in ("crop:36", "mapq:0")},
       "track": {kl: {a: track_gain(a, kl) for a in ("map", "mapshift", "dna")} for kl in ("crop:36", "mapq:0")},
       "checks": [{"arm": r["arm"], "check": r["check"], "kl": r["kl"], "pass": f(r["n_pass"]), "n": f(r["n_tracks"]),
                   "need": f(r["need"]), "gain": f(r["mean_gain"]), "met": r["met"] == "True", "bar": f(r["bar"])}
                  for r in tsv("t146/checks_h31.tsv")]}
gw = {}
for r in tsv("t146/pooled.tsv"):
    if r["route"] == "direct" and r["group"] == "never" and r["knob"] == "all" and r["arm"] in ("alone", "map", "dna", "mapshift"):
        gw[r["arm"]] = {"v": f(r["crps_all"]), "os": f(r["crps_oracle_scaled_all"]), "se": f(r["scale_error_all"]),
                        "sr": f(r["seed_spread"]), "n": int(r["n_seeds"])}
h31["gw"] = gw
sac = {}
for r in tsv("t146/sacct_per_arm.tsv"):
    sac.setdefault(r["arm"], {})[r["kind"]] = {"n": int(r["n"]), "mean": f(r["wall_min_mean"]), "max": f(r["wall_min_max"]),
                                               "cpus": int(r["cpus"]), "gb": f(r["peak_rss_gb"])}
power = {h: int(grab(node(h), r"Joint power ~(\d+)%", f"{h} power")[0]) for h in ("h29", "h30", "h31")}
note = text("notes/mini_candi_design_inputs.md")
cost = {"fastq": grab(note, r"median (\d+(?:\.\d+)?) h, max (\d+(?:\.\d+)?) h per product", "note FASTQ cost"),
        "bam": grab(note, r"~(\d+) h per product", "note BAM cost"),
        "tb": grab(note, r"~(\d+) TB FASTQ transient", "note FASTQ storage")}
twinlaw = grab(node("f17"), r"nearly meet it \((\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)\)", "f17 twin law")
final = {"AT": AT, "ck29": ck29, "dl": dl, "cells30": cells30, "p30": p30, "seeds30": seeds30, "agree": agree, "n1": n1,
         "h31": h31, "sac": sac, "power": power, "cost": cost, "twinlaw": twinlaw}

data = {"design": design, "main": main, "h1": h1b, "h2": h2b, "cap": cap, "law": law, "h28": h28,
        "hmap": hmap, "knob": knob, "floor": floor, "where": where, "shuf": shuf, "probe": probe,
        "eic": eic, "signed": signed, "final": final}

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
      f"h30 wins {sum(c['win'] for c in cells30)} of {len(cells30)}; t146 runs {sum(a['train']['n'] for a in sac.values())}")
