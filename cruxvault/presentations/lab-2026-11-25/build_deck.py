"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Paper-arc build (prezit). Every number on a slide is read here from a result file in
data/, never typed into the deck. The build records the sha256 of every file it read in
the deck-sources manifest, so `crux deck --verify index.html` can prove the deck is current.

Inputs (all in data/, copied verbatim from the branches named below):
  t390_*   origin/implementation/t390-eic-pvalue-entry @ 6714498a, cruxvault/results/t390/
           (val_boards_final45, val_boards_side_by_side, val_scale_diag, depth_rule,
           val_recal_pair_s0, val_scores_rcp_s0_8, postpile, validation_dryrun)
  t392_*, t394_*, t395_*, t370_recal_study_*, presentations_t370-minicandi-recipe_*
           exp/t370-minicandi-recipe @ 4f4663f5, cruxvault/results/... and presentations/...

    python build_deck.py            # vault = the cruxvault/ that holds this folder
"""
import csv, hashlib, io, json, pathlib, re, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
VAULT = pathlib.Path(ARGS[0]).resolve() if ARGS else HERE.parent.parent
DATA_DIR = HERE / "data"
read = []                                       # every file the build reads, in order


def text(name):
    p = DATA_DIR / name
    if p not in read:
        read.append(p)
    return p.read_bytes().decode("utf-8")


def tsv(name):
    return list(csv.DictReader(io.StringIO(text(name)), delimiter="\t"))


def md_rows(txt, ncol=None):
    """Rows of every markdown table in txt, as lists of stripped cells (header rows kept)."""
    out = []
    for line in txt.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if ncol is None or len(cells) == ncol:
                out.append(cells)
    return out


def num(s):
    return float(re.sub(r"[*\s]", "", s))


def position(summary):
    m = re.search(r"plus this one: (\d+) of (\d+)", summary)
    s = re.search(r"(\d+) of 45 validation experiments scored", summary)
    return {"pos": int(m.group(1)), "of": int(m.group(2)), "scored": int(s.group(1))}


data = {}

# ---- EIC validation boards ------------------------------------------------------------
F45 = "t390_val_boards_final45_"
B30 = "t390_val_boards_side_by_side_"
final = {}
for key, folder in (("raw", "rcp_combo_w2x_unet_s0_raw"), ("cap", "rcp_s0_depthcap"),
                    ("avg", "rcp_s0_avgbase_cap")):
    final[key] = position(text(f"{F45}{folder}_SUMMARY.md"))
rows45 = md_rows(text(F45 + "BOARDS.md"), 7)[1:]
for key, r in zip(("raw", "cap", "avg"), rows45):
    assert int(r[1]) == final[key]["pos"], (key, r)
    final[key].update(mean_rank=num(r[2]), med_nmse=num(r[3]),
                      nmse_gt1=int(r[4].split()[0]), top3=int(r[5]), bottom=int(r[6]))
b45 = text(F45 + "BOARDS.md")
m = re.search(r"depth cap was better on (\d+) experiments, worse on (\d+), and changed the mean combined rank by (-?\d+\.\d+)", b45)
final["cap_vs_raw"] = {"better": int(m.group(1)), "worse": int(m.group(2)), "delta": float(m.group(3))}
final["avg_only_final"] = len(re.findall(r"C\d\dM\d\d", re.search(r"three experiments are final-model-only.*", b45).group(0)))

rows30 = md_rows(text(B30 + "BOARDS.md"), 8)[1:]
b30 = {}
for key, r in zip(("raw", "slope_in", "slope_loo", "cap", "avg"), rows30):
    b30[key] = {"pos": int(num(r[2])), "mean_rank": num(r[3]), "med_nmse": num(r[4]),
                "nmse_gt1": int(re.sub(r"\*", "", r[5]).split("/")[0])}
b30txt = text(B30 + "BOARDS.md")
b30["n"] = int(re.search(r"Same (\d+) validation experiments", b30txt).group(1))
b30["unscored"] = int(re.search(r"Unscored experiments \((\d+)\)", b30txt).group(1))
b30["loo_none"] = int(re.search(r"(\d+) of 30 tracks have none", b30txt).group(1))
partial = {"raw": position(text(B30 + "rcp_combo_w2x_unet_s0_raw_SUMMARY.md")),
           "cap": position(text(B30 + "rcp_s0_depthcap_SUMMARY.md"))}
for k in ("rcp_s0_avgbase_cap", "rcp_s0_slope_insample", "rcp_s0_slope_loeo"):
    text(B30 + k + "_SUMMARY.md")          # read so the manifest pins the board the table used
data["boards"] = {"final": final, "b30": b30, "partial": partial}

# per-measure ranks on the final depth-capped board
MEAS = ["gwcorr", "gwspear", "mse", "msegene", "mseprom", "mseenh", "mse1obs", "mse1imp"]
bv = tsv(F45 + "rcp_s0_depthcap_board_validation.tsv")
per = {mm: [0, 0, 0, 0] for mm in MEAS}
exps = set()
for r in bv:
    if r["measure"] in per and r["rank_among_entrants"]:
        k = int(r["rank_among_entrants"])
        per[r["measure"]][0 if k == 1 else 1 if k <= 3 else 2 if k <= 6 else 3] += 1
        exps.add(r["experiment"])
data["measures"] = {"order": MEAS, "bins": per, "n": len(exps)}
data["boards"]["n_entries"] = int(next(r["n_entrants"] for r in bv if r["n_entrants"]))

# ---- depth rule ----------------------------------------------------------------------
dr = text("t390_depth_rule_DEPTH_RULE.md")
chip = dr.split("## dnase")[0]
caps = [(int(c), float(a), float(b)) for c, a, b in
        re.findall(r"cap\s+(\d+)M: training mean nMSE ([\d.]+), median ([\d.]+)", chip)]
vf = re.search(r"validation floor \(chr9\+chr21\): mean nMSE ([\d.]+) -> ([\d.]+); above 1 on (\d+) -> (\d+) of (\d+); better on (\d+)", chip)
pertrack = [(t, float(d), float(a), float(b)) for t, d, a, b in
            re.findall(r"val (C\d\dM\d\d) ([\d.]+)M: floor nMSE ([\d.]+) -> ([\d.]+)", chip)]
data["depth"] = {
    "n_train": int(re.search(r"## chip: (\d+) training tracks", chip).group(1)),
    "eff_med": float(re.search(r"median ([\d.]+)M, quartiles", chip).group(1)),
    "eff_q": [float(x) for x in re.search(r"quartiles ([\d.]+)-([\d.]+)M", chip).groups()],
    "best": int(re.search(r"best cap on training: (\d+)M", chip).group(1)),
    "caps": [c for c in caps if c[0] < 1000], "nocap": [c for c in caps if c[0] >= 1000][0],
    "val": {"before": float(vf.group(1)), "after": float(vf.group(2)), "gt1_before": int(vf.group(3)),
            "gt1_after": int(vf.group(4)), "n": int(vf.group(5)), "better": int(vf.group(6))},
    "tracks": pertrack,
}

# ---- scale diagnosis -----------------------------------------------------------------
dg = tsv("t390_val_scale_diag_DIAG_TABLE.tsv")
text("t390_val_scale_diag_SCALE_DIAG.md")
pts = [{"t": r["track"], "a": r["assay"], "obs": float(r["nmse_obs"]), "pred": float(r["nmse_pred"]),
        "cs": float(r["slope_counts"])} for r in dg]
bad = [p for p in pts if p["pred"] > 1]
data["diag"] = {"pts": pts, "n": len(pts), "bad": len(bad),
                "slope_hi": sum(p["cs"] > 1.3 for p in pts),
                "bad_slope_hi": sum(p["cs"] > 1.3 for p in bad),
                "floor_bad": sum(p["obs"] > 1 for p in pts),
                "floor_only": sum(p["cs"] <= 1 and p["obs"] > 1 for p in bad)}

# ---- recalibration on the board (expected-p pair) -------------------------------------
rec = tsv("t390_val_recal_pair_s0_recalwide_exp_board_validation.tsv")
raw8 = tsv("t390_val_scores_rcp_s0_8_raw_exp_board_validation.tsv")
text("t390_val_recal_pair_s0_recalwide_exp_SUMMARY.md"); text("t390_val_scores_rcp_s0_8_raw_exp_SUMMARY.md")
g = lambda rows, e, mm: float(next(r["value"] for r in rows if r["experiment"] == e and r["measure"] == mm))
e8 = sorted({r["experiment"] for r in rec if r["measure"] == "nmse"})
data["recal_board"] = [{"e": e, "raw": g(raw8, e, "nmse"), "rec": g(rec, e, "nmse"),
                        "rraw": g(raw8, e, "combined_rank"), "rrec": g(rec, e, "combined_rank")} for e in e8]

# ---- disclosures (validation dry run, final boards) ------------------------------------
vd = text("t390_validation_dryrun_VALIDATION_DRYRUN.md")
m = re.search(r"for (\d+) of the (\d+) ChIP-seq tracks, the store's MACS2 `signal.h5` pooled", vd)
data["disc"] = {"ctrl_gap": int(m.group(1)), "chip_n": int(m.group(2)),
                "teams": int(re.search(r"The 10 entries are (\d+) teams", vd).group(1)),
                "frag227": int(re.search(r"fragment length (\d+) bp", b45).group(1)),
                "meas_used": len(MEAS)}
pp = text("t390_postpile_POSTPILE.md")
shareA = [int(x) for x in re.findall(r"= (\d+)% \(C\d\dM\d\d\)", pp.split("- A:")[1].split("\n")[0])]
cl = pp.split("- C:")[1].split("\n")[0]
shareC = [int(x) for x in re.findall(r"= (\d+)% \(C\d\dM\d\d\)", cl)] + [int(re.search(r"= B, (\d+)%", cl).group(1))]
data["postpile"] = {"exps": re.findall(r"\((C\d\dM\d\d)\)", pp.split("- A:")[1].split("\n")[0]), "A": shareA, "C": shareC}

# ---- read-start vs read-count ----------------------------------------------------------
ps = tsv("t392_per_seed.tsv"); tab = tsv("t392_table.tsv"); text("t392_SIDE_BY_SIDE.md")
COLS = ["imputation", "correction", "calibration", "dev_imputation_skill",
        "dev_correction_gap_closed", "dev_calibration_ratio"]
rs = {}
for name, key in (("rcp_halves_mix_avg_pearson_unet", "count"), ("rs_halves_mix_avg_pearson_unet", "start")):
    rs[key] = {c: [float(r[c]) for r in ps if r["model"] == name] for c in COLS}
    t = next(r for r in tab if r["model"] == name)
    rs[key]["mean"] = {c: float(t[c + ".mean"]) for c in COLS}
    rs[key]["range"] = {c: float(t[c + ".range"]) for c in COLS}
data["rs"] = rs

inv = text("t394_INVESTIGATION.md")
sp = tsv("t394_split_per_assay_deviance_split.tsv")
assays = sorted({r["assay"] for r in sp})
skill = {a: {("count" if r["model"].startswith("rcp") else "start"): float(r["dev_imputation_skill"])
             for r in sp if r["assay"] == a} for a in assays}
calW = {w: float(v) for v, w in re.findall(r"from 1\.3081 to ([\d.]+) \(W = (\d+)\)", inv)}
calW.update({w: float(v) for v, w in re.findall(r"([\d.]+) \(W = (\d+)\)", inv.split("moves the calibration ratio")[1].split("\n")[0])})
sm = text("t395_SMOOTHING.md")
k36 = re.search(r"H3K36me3 falls from (−?[\d.]+) \(W = 1\) to (−?[\d.]+) \(W = 9\) and (−?[\d.]+) \(W = 41\)", sm)
data["why"] = {"skill": skill, "calW": calW,
               "smooth36": [float(x.replace("−", "-")) for x in k36.groups()],
               "r_conv": float(re.search(r"correlate r = ([\d.]+) with the observed overlap counts", inv).group(1)),
               "n_assays": len(assays)}

# ---- recalibration study, peak coverage, covariate prompts ------------------------------
rsd = text("t370_recal_study_recal_study.md")
def pit(section):
    out = {}
    for r in md_rows(section, 6):
        if r[0] in ("none", "same_cells", "cross_cell", "shipped"):
            out[r[0]] = {"mean": float(r[1]), "range": float(r[2]), "seeds": [float(x) for x in r[3:6]]}
    return out
imp_sec, pair_sec = rsd.split("## imputation")[1].split("## pair")
data["pit"] = {"imp": pit(imp_sec), "pair": pit(pair_sec)}

pc = text("t370_recal_study_peak_coverage_ref.md").split("## pair")[0]
hdr = next(r for r in md_rows(pc) if r[0] == "stratum")
cov = {}
for r in md_rows(pc, len(hdr)):
    if r[0] in ("all", "obs_top10", "obs_top1", "obs_top0.1", "pred_top10", "pred_top1", "pred_top0.1"):
        cell = lambda col: [float(x) for x in re.match(r"([\d.]+) \(([\d.]+)/([\d.]+)\)", r[hdr.index(col)]).groups()]
        cov[r[0]] = {"ours": cell("none 0.90"), "ref": cell("reference 0.90"), "dep": cell("cross_cell 0.90")}
data["cov"] = cov
text("t370_recal_study_peak_coverage.md")

arms = {}
ntg = 0
for f in ("t370_recal_study_covariate_prompt_acetyl.md", "t370_recal_study_covariate_prompt_methyl.md"):
    tt = text(f)
    ntg += int(re.search(r"(\d+) imputed targets", tt).group(1))
    for c in md_rows(tt, 7):
        if c[6].endswith("/3"):
            a = arms.setdefault(c[2], {"met": 0, "n": 0, "tot": [], "nr": []})
            a["n"] += 1; a["met"] += c[6] == "3/3"
            a["tot"].append(float(c[3].split()[0])); a["nr"].append(float(c[5].split()[0]))
for a in arms.values():
    a["tot"] = [min(a["tot"]), max(a["tot"])]
    a["nr_med"] = statistics.median(a["nr"]); del a["nr"]
data["prompt"] = {"arms": arms, "targets": ntg}
text("presentations_t370-minicandi-recipe_recalibration.md")
text("presentations_t370-minicandi-recipe_recipe_components.md")

# ---- final recipe vs base models, 256 regions (exp/t370 presentations data) -------------
T3 = ("imputation_raw", "correction_v3_raw", "uncertainty_raw")
fin = next(r for r in tsv("presentations_t370-minicandi-recipe_pareto_p256_sv.tsv") if r["arm"] == "rcp_combo_w2x_unet")
bnames = {"eic_v1_2_rung2_base_heldout": "Conv autoencoder", "eic_v1_2_rung3_base_heldout": "Conv + attention",
          "eic_v1_2_rung4_base_heldout": "U-Net"}
bases = [{"name": bnames[r["arm"]], "m": [float(r[t + ".mean"]) for t in T3], "r": [float(r[t + ".range"]) for t in T3]}
         for r in tsv("presentations_t370-minicandi-recipe_deck_base_p256_sv.tsv")]
data["recipe"] = {"final": {"m": [float(fin[t + ".mean"]) for t in T3], "r": [float(fin[t + ".range"]) for t in T3],
                            "seeds": int(fin["n_seeds"])}, "bases": bases}

# ---- deviance metric: simulation checks and rank agreement (results/t371) ---------------
ck = json.loads(text("t371_sim_checks.json"))
ra = tsv("t371_compare_rcp_rank_agreement.tsv")
data["dev"] = {"floor_err": ck["C1_floor_max_abs_error"], "drift": ck["C2_worst_drift"],
               "drift_log1p": ck["contrast_log1p_worst_drift"], "cal": ck["C4_summary"],
               "depth_only": ck["C5_depth_only_max_abs"], "truth": [ck["C5_truth_min"], ck["C5_truth_max"]],
               "all_pass": all(ck[k] for k in ("C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass")),
               "rank": [{"goal": r["goal"], "old": r["old"], "new": r["new"], "rho": float(r["spearman"]),
                         "p": float(r["p"]), "n": int(r["n_arms"])} for r in ra]}

# ---- write ----------------------------------------------------------------------------
read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {p.relative_to(VAULT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
assert src.count("__DATA__") == 1 and src.count("__SOURCES__") == 1
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"index.html written; {len(sources)} sources")
if "-v" in sys.argv:
    print(json.dumps({k: v for k, v in data.items() if k not in ("diag", "depth")}, indent=1)[:6000])
