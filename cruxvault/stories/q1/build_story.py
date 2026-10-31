"""Rebuild story.tex from story_src.tex: every number becomes a LaTeX macro.

The storify build pattern. Every number in the paper is read here from a result file,
never typed into the prose. The build writes each number as a \\newcommand macro and each
chart table as a \\pgfplotstableread table into the `% __DATA__` slot, and it records the
sha256 of every file it read in the `% crux-sources:` manifest line, so that
`crux story --verify story.tex` can prove the paper is current. Formatting (sign,
decimals, thousands separator) is done here, never in the prose. The build does not
compile: run `latexmk -pdf story.tex` after it.

    python3 build_story.py [VAULT]   # default: the folder above this one that holds META.md

Stdlib only. UTF-8 and "\\n" everywhere, so the build is byte-stable.
"""
import hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC, OUT = HERE / "story_src.tex", HERE / "story.tex"


def find_vault():
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).resolve()
    for d in (HERE, *HERE.parents):
        if (d / "META.md").is_file():
            return d
    sys.exit("build_story: no vault found; pass its path as the first argument")


VAULT = find_vault()
read = []                                   # every file the build reads, in order


def load(rel):
    """Read a vault-relative file and record it in the manifest. JSON is parsed."""
    p = VAULT / rel
    read.append(p)
    text = p.read_bytes().decode("utf-8")
    return json.loads(text) if p.suffix == ".json" else text


def num(v, d=1, sign=False):
    """A number as LaTeX text: fixed decimals, a real minus sign, an optional plus sign,
    and a thin-space thousands separator. Use the result as a macro body."""
    v = round(float(v), d)
    s = f"{abs(v):,.{d}f}".replace(",", "{,}")
    if v < 0:
        return "\\ensuremath{-}" + s
    return ("\\ensuremath{+}" if sign and v > 0 else "") + s


def raw(v, d=1):
    """A number as a plain table entry for pgfplots (no LaTeX markup)."""
    return f"{round(float(v), d):.{d}f}"


macros = {}    # macro name (letters only) -> LaTeX text
tables = {}    # table name (letters only) -> (column names, rows)

# ==== REDUCE ===========================================================================
# Close-out of the counterfactual covariate testbed (anchor: the covariate-map question).
# Inputs, all vault-relative under results/ (git-ignored; copied from the main checkout):
#   t130/blind_table.tsv, t130/nested_dispersion.tsv   depth and run-type blind readings
#   t136/checks_readings.tsv, t136/m5_depthlaw_table.tsv arm-to-arm training grid
#   h19/knob_generalization/{verify.txt,group_table.tsv,knob_table.tsv}  trained vs unseen
#   h15/report.md                                      main claim, swap and depth-law checks
#   t122/AUDIT.md                                      real data obey the depth law
#   t125/part1_digest.md, t125/part2_digest.md         depth offset; counts-to-p map
#   t139/knob_vs_floor.tsv                             knob effects against the floor
#   t137/floor_at_30M.tsv, t142/floor_at_30M.tsv       floor, unit and whole-read thinning
#   t142/compare_real_class.tsv, t142/map_classes.tsv  real-halves check; regional map
#   t138/map_classes.tsv                               first-pass map (15M rows)
#   t142/t141_readthin/knob_classes.tsv                where each knob acts
import csv, io, statistics as st


def tsv(rel):
    return list(csv.DictReader(io.StringIO(load(rel)), delimiter="\t"))


def f(x):
    return float(x)


def rng(vals, d=2, sign=False):
    """'lo to hi' pair of macros' bodies."""
    return num(min(vals), d, sign), num(max(vals), d, sign)


def put(name, vals, d=2, sign=False):
    macros[name + "Lo"], macros[name + "Hi"] = rng(vals, d, sign)


# ---- depth claim: blind readings with the nested-read oracle -------------------------
blind = tsv("results/t130/blind_table.tsv")
CLS = ("DNase", "narrow", "broad")


def rows(**kw):
    return [r for r in blind if all(r[k] == v for k, v in kw.items())]


dep = dict(space="counts", knob="depth", subset="all")
b2a = [r for r in rows(direction="base_to_arm", **dep) if r["mark_class"] in CLS]
bd = [r for r in b2a if r["rung"] in "BCD"]
noise = lambda r: 2 * f(r["seed_abs_delta"]) / (f(r["d_onewarp"]) - f(r["d_oracle"]))
put("gcBD", [f(r["gapclosed_onewarp"]) for r in bd])
put("gcBDnoise", [noise(r) for r in bd])
macros["nBDcells"] = num(len(bd), 0)
macros["nBDgain"] = num(sum(f(r["gain_over_onewarp"]) > 2 * f(r["seed_abs_delta"]) for r in bd), 0)
ra = [r for r in b2a if r["rung"] == "A" and r["mark_class"] in ("DNase", "narrow")]
put("gcA", [f(r["gapclosed_onewarp"]) for r in ra])
put("gcAnoise", [noise(r) for r in ra], 1)
put("ivB", [f(r["d_onewarp"]) - f(r["d_oracle"]) for r in b2a], 3)
put("floorCtx", [f(r["d_floor"]) for r in b2a])
a2b = [r for r in rows(direction="arm_to_base", **dep) if r["mark_class"] in CLS]
put("gcCDab", [f(r["gapclosed_onewarp"]) for r in a2b if r["rung"] in "CD"])
# one worked cell with its CRPS split: narrow, design B, one g per track, base to arm
w = [r for r in b2a if r["mark_class"] == "narrow" and r["rung"] == "B" and r["g_version"] == "per_track"][0]
macros.update(
    wModel=num(w["d_model"], 3), wModelOS=num(w["crps_oracle_scaled_all"], 3),
    wModelSE=num(w["scale_error_all"], 4), wWarp=num(w["d_onewarp"], 3),
    wWarpOS=num(w["onewarp_oracle_scaled_all"], 3), wWarpSE=num(w["onewarp_scale_error_all"], 4),
    wOra=num(w["d_oracle"], 3), wOraOS=num(w["oracle_crps_oracle_scaled_all"], 3),
    wOraSE=num(abs(f(w["oracle_scale_error_all"])), 4), wSeed=num(2 * f(w["seed_abs_delta"]), 4))
# panel (a): per-track g, base to arm, each design per class, with the seed bar in gap units
tables["tblGap"] = (["x", "gc", "err", "cls"], [])
xpos = {"DNase": 0, "narrow": 1, "broad": 2}
for r in b2a:
    if r["g_version"] != "per_track":
        continue
    x = xpos[r["mark_class"]] + {"A": -0.27, "B": -0.09, "C": 0.09, "D": 0.27}[r["rung"]]
    tables["tblGap"][1].append([raw(x, 2), raw(r["gapclosed_onewarp"], 3), raw(noise(r), 3), r["rung"]])
for k in "ABCD":
    tables["tblGap" + k] = (["x", "gc", "err"], [row[:3] for row in tables["tblGap"][1] if row[3] == k])
del tables["tblGap"]

# arm dispersion: Var(arm - rho*base) / binomial variance, bins with base >= 1, per arm
o, b = {}, {}
for r in tsv("results/t130/nested_dispersion.tsv"):
    if int(r["base_lo"]) < 1 or r["var_obs_manifest"] == "NA":
        continue
    n = int(r["n_bins"])
    o[r["pid"]] = o.get(r["pid"], 0) + f(r["var_obs_manifest"]) * n
    b[r["pid"]] = b.get(r["pid"], 0) + f(r["var_binom_manifest"]) * n
put("disp", [o[k] / b[k] for k in o], 3)
macros["nArmsDisp"] = num(len(o), 0)

# ---- run-type claim ------------------------------------------------------------------
pe = [r for r in rows(space="counts", knob="pe", subset="all", direction="both")]
hist = [r for r in pe if r["mark_class"] in ("narrow", "broad")]
put("peGain", [f(r["gain_over_onewarp"]) for r in hist], 3)
macros["nPeHist"] = num(len(hist), 0)
macros["nPeHistSig"] = num(sum(f(r["gain_over_onewarp"]) > 2 * f(r["seed_abs_delta"]) for r in hist), 0)
put("peSeed", [2 * f(r["seed_abs_delta"]) for r in hist], 3)
dn = [r for r in pe if r["mark_class"] == "DNase"]
macros["nPeDn"] = num(len(dn), 0)
macros["nPeDnSig"] = num(sum(f(r["gain_over_onewarp"]) > 2 * f(r["seed_abs_delta"]) for r in dn), 0)
nr = [r for r in pe if r["mark_class"] == "narrow"][0]
macros.update(peFloor=num(nr["d_floor"], 3), peFloorOS=num(nr["floor_oracle_scaled_all"], 3),
              peFloorSE=num(nr["floor_scale_error_all"], 3), peWarp=num(nr["d_onewarp"], 3),
              peWarpOS=num(nr["onewarp_oracle_scaled_all"], 3),
              peWarpSE=num(nr["onewarp_scale_error_all"], 3))

# ---- trained versus never-trained pairs, every knob ------------------------------------
ver = load("results/h19/knob_generalization/verify.txt")
for sp, key in (("counts", "Cnt"), ("pval", "P")):
    for grp, g in (("trained", "Tr"), ("same", "Same"), ("cross", "Cross")):
        pat = (r"HEADLINE %s median-over-16-configs %s %s: raw ([0-9.]+)"
               % (sp, "gain_rel_nocov" if sp == "counts" else "typical-pair gain", grp))
        macros["rel" + key + g] = num(re.search(pat, ver).group(1), 2 if grp != "cross" or sp == "counts" else 3)
grp = tsv("results/h19/knob_generalization/group_table.tsv")
for grp_, g in (("trained", "Tr"), ("same", "Same")):
    rr = [r for r in grp if r["space"] == "counts" and r["group"] == grp_]
    macros["nMet" + g] = num(sum(r["met_nocov"] == "True" for r in rr), 0)
    macros["nCfg"] = num(len(rr), 0)
kt = tsv("results/h19/knob_generalization/knob_table.tsv")


def kmed(knob, space, group, col="gain_rel_nocov"):
    v = [f(r[col]) for r in kt if r["knob"] == knob and r["space"] == space and r["group"] == group]
    return st.median(v) if v else None


put("readTr", [kmed(k, "counts", "trained") for k in ("crop", "dedup", "mapq", "abproxy")])
put("readWob", [kmed(k, "counts", "trained", "wobble_real") for k in ("crop", "dedup", "mapq", "abproxy")], 3)
put("ctlPTr", [kmed(k, "pval", "trained") for k in ("ratio", "ctlid", "ctldepth")])
put("ctlPSame", [kmed(k, "pval", "same") for k in ("ratio", "ctlid", "ctldepth")])
put("ctlWob", [kmed(k, "pval", g, "wobble_real") for k in ("ratio", "ctlid", "ctldepth")
               for g in ("trained", "same")], 3)
put("readPSame", [kmed(k, "pval", "same") for k in ("crop", "mapq", "abproxy")])

rep = load("results/h15/report.md")
m = re.search(r"unmet everywhere\s*\n?\s*\(≈([0-9.]+)–([0-9.]+) for C in counts\)", rep)
macros["swapLo"], macros["swapHi"] = num(m.group(1), 2), num(m.group(2), 2)

aud = load("results/t122/AUDIT.md")
m = re.search(r"Over all (\d+) ordered pairs.*?within ([0-9.]+)%", aud, re.S)
macros["nLawPairs"], macros["lawData"] = num(m.group(1), 0), num(m.group(2), 2)

# ---- depth offset and the counts-to-p map ----------------------------------------------
p1 = load("results/t125/part1_digest.md")
m = re.search(r"DNase: ([0-9.]+) ± ([0-9.]+) \(([0-9.]+) \+ ([0-9.]+)\) against ([0-9.]+) ± ([0-9.]+) "
              r"\(([0-9.]+) \+ ([0-9.]+)\); gain ([0-9.]+), bar ([0-9.]+)", p1)
(macros["offA"], macros["offAw"], macros["offAOS"], macros["offASE"], macros["oldA"], macros["oldAw"],
 macros["oldAOS"], macros["oldASE"], macros["offGain"], macros["offBar"]) = (
    num(m.group(1), 3), num(m.group(2), 3), num(m.group(3), 3), num(m.group(4), 3), num(m.group(5), 2),
    num(m.group(6), 3), num(m.group(7), 2), num(m.group(8), 2), num(m.group(9), 3), num(m.group(10), 3))
m = re.search(r"B to D2 clear in (\d+) of (\d+) cells", p1)
macros["offBDclear"], macros["offBDn"] = num(m.group(1), 0), num(m.group(2), 0)
m = re.search(r"within 0\.10 \((\d+) of (\d+) cells\)", p1)
macros["offLawIn"], macros["offLawN"] = num(m.group(1), 0), num(m.group(2), 0)
p2 = load("results/t125/part2_digest.md")
m = re.search(r"formula \+ MLP r5, own noise ([0-9.]+) ± ([0-9.]+) \(([0-9.]+) \+ ([0-9.]+)\)", p2)
macros.update(hBest=num(m.group(1), 4), hBestW=num(m.group(2), 4), hBestOS=num(m.group(3), 4),
              hBestSE=num(m.group(4), 4))
m = re.search(r"affine r1 push ([0-9.]+) ± ([0-9.]+) \(([0-9.]+) \+ ([0-9.]+)\)", p2)
macros.update(hAff=num(m.group(1), 4), hAffW=num(m.group(2), 4), hAffOS=num(m.group(3), 4),
              hAffSE=num(m.group(4), 4))

# ---- arm-to-arm training grid (depth law on unseen arm-to-arm pairs) ---------------------
ck = tsv("results/t136/checks_readings.tsv")
c = lambda chk, arm, item: [r for r in ck if r["check"] == chk and r["arm"] == arm and r["item"] == item][0]
c1 = c("c1", "Darm1", "gap closed on heldout_r2")
macros["cOneGap"] = num(c1["value"], 3)
m = re.search(r"Dref ([0-9.]+) \(spread ([0-9.]+)\), Dviabase ([0-9.]+) \(spread ([0-9.]+)\), "
              r"Darm1 ([0-9.]+) \(spread ([0-9.]+); per seed ([0-9.]+), ([0-9.]+), ([0-9.]+)\)", c1["detail"])
(macros["lawRef"], macros["lawRefS"], macros["lawVia"], macros["lawViaS"], macros["lawArm"],
 macros["lawArmS"]) = (num(m.group(i), 3) for i in range(1, 7))
macros["lawSeedLo"], macros["lawSeedHi"] = num(m.group(7), 3), num(m.group(9), 3)
c3 = c("c3", "Darm1", "heldout_r2 vs trained_r4 + 2 x spread")
m = re.search(r"trained_r4 ([0-9.]+) \(spread ([0-9.]+)\); bar = .*? = ([0-9.]+)", c3["detail"])
macros.update(cThreeHeld=num(c3["value"], 3), cThreeTr=num(m.group(1), 3), cThreeTrS=num(m.group(2), 3),
              cThreeBar=num(m.group(3), 3))
c4 = c("c4", "Darm1", "crps_all gap closed")
macros["cFourGap"] = num(c4["value"], 2)
m = re.search(r"Dref ([0-9.]+) \(oracle_scaled ([0-9.]+), scale_error ([0-9.]+); spread ([0-9.]+)\), "
              r"Dviabase ([0-9.]+) \(([0-9.]+), ([0-9.]+); spread ([0-9.]+)\), "
              r"Darm1 ([0-9.]+) \(([0-9.]+), ([0-9.]+); spread ([0-9.]+)\)", c4["detail"])
names = ("crRef", "crRefOS", "crRefSE", "crRefS", "crVia", "crViaOS", "crViaSE", "crViaS",
         "crArm", "crArmOS", "crArmSE", "crArmS")
for i, nm in enumerate(names, 1):
    macros[nm] = num(m.group(i), 3)
c2 = c("c2", "Darm1", "trained_base median |log2 err|")
m = re.search(r"Darm1 ([0-9.]+) \(spread ([0-9.]+)\) vs Dref ([0-9.]+) \(spread ([0-9.]+)\)", c2["detail"])
macros.update(fitArm=num(m.group(1), 3), fitArmS=num(m.group(2), 3), fitRef=num(m.group(3), 3),
              fitRefS=num(m.group(4), 3))
# second arm: runs finished (complete since 2026-10-31)
d2 = c("c5", "Darm2", "runs finished")
macros["armTwoDone"] = num(d2["value"], 0)
m = re.search(r"(\d+) of (\d+)", d2["threshold"])
macros["armTwoAll"] = num(m.group(1), 0)
macros["ruleReading"] = [r for r in ck if r["check"] == "rule"][0]["reading"].split(" (")[0].replace("-", "~").lower()
# both controls over both arms (the rule is final only when both pass)
assert all(c(k, "both", "check reading")["reading"] == "PASS" for k in ("c2", "c5")), "a control fails: rewrite the arm-to-arm paragraph"
# second arm on held-out pairs: read, not checked (the rule reads the first arm only)
h2 = [r for r in tsv("results/t136/m1_heldout_table.tsv") if r["scope"] == "all7" and r["class"] == "all"
      and r["arm"] == "Darm2" and r["subgroup"] == "heldout75" and r["knob"] == "all"][0]
assert h2["complete"] == "True", "second arm incomplete"
macros.update(crArmTwo=num(h2["crps_all"], 3), crArmTwoOS=num(h2["crps_oracle_scaled_all"], 3),
              crArmTwoSE=num(h2["scale_error_all"], 3), crArmTwoS=num(h2["seed_spread"], 3),
              crArmTwoGap=num(h2["closed_crps_all"], 2))

# ---- knob effects against the floor -------------------------------------------------------
kv = [r for r in tsv("results/t139/knob_vs_floor.tsv") if r["space"] == "counts"]
ph = [r for r in kv if r["metric"] == "abs_log2fc_top1"]
sh = [r for r in kv if r["metric"] == "shape_1k"]
macros.update(nPh=num(len(ph), 0), nPhAbove=num(sum(r["above_floor"] == "True" for r in ph), 0),
              nSh=num(len(sh), 0), nShAbove=num(sum(r["above_floor"] == "True" for r in sh), 0))
put("phRatio", [f(r["ratio_to_floor"]) for r in ph], 0)
put("phFloorSpread", [f(r["floor_seed_spread"]) for r in ph], 3)

# ---- the resampling floor: unit versus whole-read thinning ------------------------------------
fr = {(r["mark_class"], r["metric"]): r for r in tsv("results/t142/floor_at_30M.tsv")}
fu = {(r["mark_class"], r["metric"]): r for r in tsv("results/t137/floor_at_30M.tsv")}
cap = {"DNase": "Dn", "narrow": "Nar", "broad": "Br"}
for k, c_ in cap.items():
    r, u = fr[(k, "shape_1k")], fu[(k, "shape_1k")]
    macros["shThirty" + c_] = num(r["value_30M_extrap"], 3)
    macros["shThirtyS" + c_] = num(r["seed_spread_r0.5"], 3)
    macros["shLow" + c_] = num(r["value_3.75M"], 2)
    macros["shLowUnit" + c_] = num(u["value_3.75M"], 2)
put("shSpread", [f(fr[(k, "shape_1k")][s]) for k in cap for s in
                 ("seed_spread_r0.5", "seed_spread_r0.25", "seed_spread_r0.125")], 3)
x2 = {"value_30M_extrap": 30, "value_15M": 15, "value_7.5M": 7.5, "value_3.75M": 3.75, "value_1.875M": 1.875}
import math
for k, c_ in cap.items():
    for src, tag in ((fr, "Read"), (fu, "Unit")):
        r = src[(k, "shape_1k")]
        tables["tblSh" + tag + c_] = (["x", "y"], [[raw(math.log2(v), 3), raw(r[c], 3)] for c, v in x2.items()])
cr = {(r["mark_class"], r["metric"]): r for r in tsv("results/t142/compare_real_class.tsv")}
a = cr[("all", "shape_1k")]
macros.update(realRead=num(a["median_real_over_read"], 3), realUnit=num(a["median_real_over_unit"], 3),
              realN=num(a["n"], 0), realSpread=num(a["median_read_seed_spread_over_read"], 3))
macros["realCrps"] = num(cr[("all", "crps_all")]["median_real_over_read"], 3)

# regional map of the floor: level null first, then four features
mp = tsv("results/t142/map_classes.tsv")
mv = lambda dpt, rd: [f(r["median"]) for r in mp if r["depth"] == dpt and r["reading"] == rd]
put("lvlFifteen", mv("15", "R2_level"))
put("lvlLow", mv("375", "R2_level"))
mvx = lambda dpt, rd: [f(r["max"]) for r in mp if r["depth"] == dpt and r["reading"] == rd]
macros["featMax"] = num(max(mvx("15", "delta_R2") + mvx("375", "delta_R2")), 4)
macros["shufMax"] = num(max(mvx("15", "delta_R2_shuffle") + mvx("375", "delta_R2_shuffle")), 5)
mp1 = tsv("results/t138/map_classes.tsv")
macros["featLowUnit"] = num(max(f(r["max"]) for r in mp1 if r["depth"] == "375" and r["reading"] == "delta_R2"), 3)

# ---- where each knob acts (one-warp residual, whole-read depth null) --------------------------
kc = tsv("results/t142/t141_readthin/knob_classes.tsv")


def kc_(arm, level, series, reading, cls=("narrow", "broad", "DNase"), space="counts", col="median"):
    return [f(r[col]) for r in kc if r["arm"] == arm and r["level"] == level and r["series"] == series
            and r["reading"] == reading and r["space"] == space and r["mark_class"] in cls]


dA = kc_("depth", "15M", "arm", "mean_d", ("DNase",))[0]
dN = kc_("depth", "15M", "null_s0", "mean_d", ("DNase",))[0]
macros.update(depArm=num(dA, 3), depNull=num(dN, 3))
put("depArmFeat", [v for l in ("15M", "7.5M", "3.75M") for v in kc_("depth", l, "arm", "delta_R2")], 4)
put("depNullFeat", [v for l in ("15M", "7.5M", "3.75M") for s in ("null_s0", "null_s1")
                    for v in kc_("depth", l, s, "delta_R2")], 4)
put("mapqZero", kc_("mapq", "0", "arm", "delta_R2"), 3)
put("mapqZeroDec", kc_("mapq", "0", "arm", "mean_r_map_dec0"), 2)
put("cropFeat", kc_("crop", "36", "arm", "delta_R2") + kc_("crop", "50", "arm", "delta_R2"), 3)
put("dedupFeat", kc_("dedup", "off", "arm", "delta_R2", ("narrow", "broad")) +
    kc_("pe", "pe", "arm", "delta_R2", ("narrow", "broad")), 4)
put("knobShuf", [f(r["median"]) for r in kc if r["reading"] == "delta_R2_shuffle" and r["space"] == "counts"], 5)
put("floorFeat", [f(r["floor_median"]) for r in kc if r["reading"] == "delta_R2" and r["space"] == "counts"
                  and r["series"] == "arm" and r["floor_depth"] == "15"], 5)
put("ctlNone", kc_("ctlid", "none", "arm", "delta_R2", space="pval"), 3)
# panel (c): feature gain per knob level, counts, narrow and broad medians
lv = [("depth", "15M", "{depth 15M}"), ("depth", "3.75M", "{depth 3.75M}"), ("dedup", "off", "{dedup off}"),
      ("pe", "pe", "{paired-end}"), ("abproxy", "f0.9", "{antibody f0.9}"), ("crop", "50", "{length 50}"),
      ("crop", "36", "{length 36}"), ("mapq", "10", "{MAPQ 10}"), ("mapq", "0", "{MAPQ 0}")]
tables["tblKnob"] = (["x", "nar", "br", "lab"],
                     [[i, raw(kc_(a_, l, "arm", "delta_R2", ("narrow",))[0], 6),
                       raw(kc_(a_, l, "arm", "delta_R2", ("broad",))[0], 6), lab]
                      for i, (a_, l, lab) in enumerate(lv)])
macros["nKnobLevels"] = num(len(lv), 0)
# =======================================================================================

for name in (*macros, *tables):
    assert re.fullmatch(r"[A-Za-z]+", name), f"macro/table name {name!r}: letters only"


def in_vault(p):
    """p's vault-relative POSIX path, or None when p lies outside the vault."""
    try:
        return p.relative_to(VAULT).as_posix()
    except ValueError:
        return None


lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
for k, (cols, rows) in sorted(tables.items()):
    body = "\n".join(" ".join(str(c) for c in r) for r in rows)
    lines.append(f"\\pgfplotstableread{{\n{' '.join(cols)}\n{body}\n}}\\{k}")
data = "\n".join(lines) or "% (no numbers)"

# the manifest: every file read, plus this script and the source when they live in the
# vault (they do in <vault>/stories/<anchor-id>/; a shipped example lives outside it)
read += [p for p in (pathlib.Path(__file__).resolve(), SRC) if in_vault(p)]
sources = {in_vault(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
assert all(sources), "every file the build reads must lie inside the vault"

src = SRC.read_bytes().decode("utf-8")
assert src.count("% crux-sources: __SOURCES__") == 1 and src.count("% __DATA__") == 1
out = src.replace("% crux-sources: __SOURCES__",
                  "% crux-sources: " + json.dumps(sources, sort_keys=True))
out = out.replace("% __DATA__", data)
OUT.write_bytes(out.encode("utf-8"))
print(f"wrote {OUT}: {len(macros)} macro(s), {len(tables)} table(s), "
      f"{len(sources)} source file(s)")
