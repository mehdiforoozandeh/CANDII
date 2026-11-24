"""Rebuild index.html from deck_src.html by embedding every number as DATA (prezit paper arc).

Inputs (vault-relative, all under presentations/t370-minicandi-recipe/data/; refresh them with
refresh_data.sh, which pulls the tables from Fir and runs tools/onion_eval/pareto_select.py):
  deck_base_p16_sv.tsv, deck_arms_p16_sv.tsv     arm summaries, eic_v1_2, correction on 16 regions
  deck_base_p256_sv.tsv, deck_arms_p256_sv.tsv   the same, correction on 256 regions
  pareto_p16_sv.tsv, pareto_p256_sv.tsv          pareto_select.py output (class per arm)
  copy_gain_p16_sv.tsv                           copy_leak.same_view_copy_gain_relative per seed
  eic_v1_arm_summary_p16_sv.tsv                  eic_v1 arm summary (skip-path dissection)
  scores_ladder.md                               six-model ladder, eic_v1
  ROUND2_READ.md                                 160k-window runs, eic_v1
  ens_p256_sv.tsv                                seed ensemble and dispersion recalibration, 256 regions
  table_unet_vs_average.tsv (optional)           EIC validation, U-Net vs Average

    python3 build_deck.py
"""
import csv
import hashlib
import io
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
VAULT = HERE.parents[1]
read = []

IMP, COR, CAL = "imputation_raw", "correction_v3_raw", "uncertainty_raw"


def raw(rel):
    p = HERE / "data" / rel
    read.append(p)
    return p.read_bytes().decode("utf-8")


def tsv(rel):
    txt = raw(rel)
    rows = csv.DictReader(io.StringIO("\n".join(l for l in txt.splitlines() if not l.startswith("#"))),
                          delimiter="\t")
    return {r["arm"] if "arm" in r else r["run"]: r for r in rows}


def f(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def terms(r):
    return {k: {"m": f(r[f"{t}.mean"]), "r": f(r[f"{t}.range"])}
            for k, t in (("imp", IMP), ("cor", COR), ("cal", CAL))} | {"n": int(r["n_seeds"])}


base16, base256 = tsv("deck_base_p16_sv.tsv"), tsv("deck_base_p256_sv.tsv")
arms16, arms256 = tsv("deck_arms_p16_sv.tsv"), tsv("deck_arms_p256_sv.tsv")
par16, par256 = tsv("pareto_p16_sv.tsv"), tsv("pareto_p256_sv.tsv")
all16, all256 = {**base16, **arms16}, {**base256, **arms256}

BASES = {"convae": "eic_v1_2_rung2_base_heldout", "convattn": "eic_v1_2_rung3_base_heldout",
         "unet": "eic_v1_2_rung4_base_heldout"}


def floors(tab):
    out = {}
    for k, t in (("imp", IMP), ("cor", COR), ("cal", CAL)):
        best = max(BASES.values(), key=lambda a: float(tab[a][f"{t}.mean"]))
        m, r = float(tab[best][f"{t}.mean"]), float(tab[best][f"{t}.range"])
        out[k] = {"best": round(m, 4), "range": round(r, 4), "floor": round(m - r, 4)}
    return out


def arm(name, tab16=all16, tab256=all256):
    o = {"p16": terms(tab16[name])}
    if name in tab256:
        o["p256"] = terms(tab256[name])
    for s, par in (("p16", par16), ("p256", par256)):
        if name in par:
            o[f"class_{s}"] = par[name]["class"]
            o[f"score_{s}"] = f(par[name]["score"])
    return o


STEPS = [("eic_v1_2_rung2_base_heldout", "Conv autoencoder"),
         ("eic_v1_2_rung3_base_heldout", "Conv + attention"),
         ("eic_v1_2_rung4_base_heldout", "U-Net"),
         ("rcp_headskip_off_unet", "U-Net, head skip removed"),
         ("rcp_halves_unet", "U-Net + read-disjoint pairs"),
         ("rcp_halves_mix_pearson_unet", "+ Pearson-residual input"),
         ("rcp_halves_mix_avg_unet", "+ average-signal input"),
         ("rcp_halves_headskip_off_unet", "pairs + head skip removed")]
FIXES = {"rcp_cac_cnn_unet": "covariate correction layers (CNN)",
         "rcp_cac_cnn_corefilm_off_unet": "correction layers (CNN), core FiLM off",
         "rcp_cac_curve_unet": "covariate correction layers (curve)",
         "rcp_cac_curve_corefilm_off_unet": "correction layers (curve), core FiLM off",
         "rcp_film_all_unet": "FiLM everywhere",
         "rcp_film_bottleneck_unet": "FiLM at the bottleneck only",
         "rcp_film_decoder_unet": "FiLM in the decoder only",
         "rcp_film_encoder_unet": "FiLM in the encoder only",
         "rcp_headskip_film_unet": "head skip conditioned by FiLM",
         "rcp_skipfilm_ccp_unet": "level skips conditioned on input + output covariates",
         "rcp_skipfilm_ccp_headskip_film_unet": "both skip conditionings",
         "rcp_mix_avg_unet": "average-signal input",
         "rcp_mix_pearson_unet": "Pearson-residual input",
         "rcp_mix_fixdepth_unet": "fixed input depth",
         "rcp_nocov_unet": "no covariates"}
CONFIRM = ["rcp_halves_unet", "rcp_halves_headskip_film_unet", "rcp_halves_mix_pearson_unet",
           "rcp_halves_mix_avg_unet", "rcp_halves_headskip_off_unet",
           "rcp_halves_mix_avg_pearson_unet", "rcp_combo_w2x_unet"]

# copy gain per run: mean and range over seeds
cg = {}
for line in raw("copy_gain_p16_sv.tsv").splitlines()[1:]:
    run, _, _, v = line.split("\t")
    if v not in ("None", ""):
        cg.setdefault(run, []).append(float(v))
copy = {k: {"m": round(sum(v) / len(v), 4), "r": round(max(v) - min(v), 4), "n": len(v)}
        for k, v in cg.items()}

# eic_v1 skip-path dissection (arm summary, 16-region correction)
v1 = tsv("eic_v1_arm_summary_p16_sv.tsv")
SKIP = [("rung4_base_heldout", "U-Net, all skips"),
        ("final_rung4_mech_skip_counts", "count skips only"),
        ("final_rung4_mech_skip_dna", "DNA skips only (no count path)")]
skip = [{"lab": lab, **terms(v1[a])} for a, lab in SKIP]

# six-model ladder, eic_v1 (scores_ladder.md seed-mean table)
LAD = {"rung1_base": "MLP, no context", "rung2_base": "Conv autoencoder",
       "rung3_base": "Conv + attention", "rung4_base": "U-Net", "rung5_base": "Axial attention",
       "rung6_base": "Perceiver IO"}
ladder = []
for line in raw("scores_ladder.md").splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if c and c[0] in LAD and not any(r["key"] == c[0] for r in ladder):
        num = lambda s: (float(s.split()[0]), float(s.split("(")[1].rstrip(")")) if "(" in s else 0.0)
        (i, ir), (co, cr) = num(c[2]), num(c[4])
        ladder.append({"key": c[0], "lab": LAD[c[0]], "imp": i, "imp_r": ir, "cor": co, "cor_r": cr})

# 160k windows vs 80k windows (ROUND2_READ.md: run | composite | imp | cor | base comp | base imp | base cor)
win = []
for line in raw("ROUND2_READ.md").splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if c and c[0].endswith("160k windows") and len(c) == 7 and not any(w["lab"] == c[0] for w in win):
        win.append({"lab": c[0].replace(", 160k windows", ""), "imp2": float(c[2]), "cor2": float(c[3]),
                    "imp1": float(c[5]), "cor1": float(c[6])})

# EIC validation table (optional until the scoring job lands)
eic = None
p = HERE / "data" / "table_unet_vs_average.tsv"
if p.exists():
    txt = raw("table_unet_vs_average.tsv")
    eic = list(csv.DictReader(io.StringIO(txt), delimiter="\t"))

# recalibration: the same three-seed ensemble before and after the per-assay dispersion factors
ens = tsv("ens_p256_sv.tsv")
recal = {k: terms(ens[k]) for k in ("rcp_halves_mix_pearson_unet__ens3", "rcp_halves_mix_pearson_unet__ens3_recal")}

# what each component adds, 256 regions: the ladder from the U-Net base to the final architecture
LADDER256 = [("eic_v1_2_rung4_base_heldout", "U-Net (base)"),
             ("rcp_halves_unet", "+ read-disjoint pairs"),
             ("rcp_halves_mix_pearson_unet", "pairs + Pearson-residual input"),
             ("rcp_halves_mix_avg_unet", "pairs + average-signal input"),
             ("rcp_halves_mix_avg_pearson_unet", "pairs + average + Pearson"),
             ("rcp_combo_w2x_unet", "+ 2× windows (final)")]

data = {
    "floor16": floors(base16), "floor256": floors(base256),
    "steps": [{"key": a, "lab": lab, **arm(a)} for a, lab in STEPS if a in all16],
    "fixes": [{"key": a, "lab": lab, **terms(all16[a])} for a, lab in FIXES.items() if a in all16],
    "confirm": [{"key": a, **arm(a)} for a in CONFIRM if a in all16],
    "halves_conv": {m: terms(all16[f"rcp_halves_{m}"]) for m in ("convae", "convattn")},
    "comp": [{"key": a, "lab": lab, **terms(all256[a])} for a, lab in LADDER256],
    "recal": recal,
    "copy": copy, "skip": skip, "ladder": ladder, "win": win, "eic": eic,
}

read += [pathlib.Path(__file__).resolve(), HERE / "deck_src.html"]
sources = {p.relative_to(VAULT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
assert src.count("__DATA__") == 1 and src.count("__SOURCES__") == 1
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"wrote {HERE / 'index.html'}: {len(sources)} source file(s); eic={'yes' if eic else 'pending'}")
