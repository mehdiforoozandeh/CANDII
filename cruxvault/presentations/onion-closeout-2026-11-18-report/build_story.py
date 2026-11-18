"""Rebuild report.tex from report_src.tex: every number becomes a LaTeX macro.

The storify build pattern (skill `storify`), adapted to the close-out report of the
"peeling the onion" run. Every number in the report is read here from a result file,
never typed into the prose. The build writes each number as a \\newcommand macro, each
generated table body as a macro, and each chart table as a \\pgfplotstableread table into
the `% __DATA__` slot. It records the sha256 of every file it read in the
`% crux-sources:` line, so that `crux story --verify report.tex` can prove the report is
current. Compile with `latexmk -pdf report.tex` after it.

    python3 build_story.py [VAULT]   # default: the folder above this one that holds META.md

Evidence folders under results/ are git-ignored. They exist in the foreman worktree
(.claude/worktrees/onion/cruxvault/results/); copy them to the same paths in the vault you
build from, or pass that vault as VAULT.

Stdlib only. UTF-8 and "\\n" everywhere, so the build is byte-stable.

VERSION 2 (2026-11-18) also reads results/closeout_E4/ladder_2026-11-18/ (ladder.tsv, ladder.md),
results/closeout_D2/layers_2026-11-18/layers.tsv, results/closeout_A4/qc_tables_2026-11-18/coverage.txt,
INSTRUCTIONS/peeling_the_onion/run_state/plan.md (the run-log snapshot; rebuild after each foreman
snapshot commit) and closeout/PI decisions 2026-11-17.md.

RESCORING HOOK. When the 256-region rescoring of the 7 decisive arms lands, copy the
arm-summary table (select_rule.py arm-summary --runs <runs_fir> --sub p256_sv) to
RESCORE_REL below and rebuild. The build then fills the 256-region columns of the
rescoring table; the paragraph marked RESCORING-PARAGRAPH in report_src.tex is the one
piece of prose to rewrite.
"""
import hashlib, json, math, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC, OUT = HERE / "report_src.tex", HERE / "report.tex"


def find_vault():
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).resolve()
    for d in (HERE, *HERE.parents):
        if (d / "META.md").is_file():
            return d
    sys.exit("build_story: no vault found; pass its path as the first argument")


VAULT = find_vault()
read = []


def load(rel):
    p = VAULT / rel
    read.append(p)
    text = p.read_bytes().decode("utf-8")
    return json.loads(text) if p.suffix == ".json" else text


def num(v, d=1, sign=False):
    v = round(float(v), d)
    if v == 0:
        v = 0.0
    s = f"{abs(v):,.{d}f}".replace(",", "{,}")
    if v < 0:
        return "\\ensuremath{-}" + s
    return ("\\ensuremath{+}" if sign and v > 0 else "") + s


def raw(v, d=3):
    return f"{round(float(v), d):.{d}f}"


def grab(text, pattern, group=1, flags=re.S):
    """The first match of `pattern` in `text`; the build fails when it is absent."""
    m = re.search(pattern, text, flags)
    assert m, f"pattern not found: {pattern!r}"
    return m.group(group)


def fnum(s):
    return float(s.replace(",", ""))


macros, tables = {}, {}

# ==== REDUCE ===========================================================================
INS = "INSTRUCTIONS/peeling_the_onion/"
R = "results/"

# ---- data: eic_v1 and its views ------------------------------------------------------
ds = load(R + "t183/DATASET_EIC_V1.md")
macros["nRows"] = num(fnum(grab(ds, r"([\d,]+) rows \(train")), 0)
macros["nRowsTrain"] = num(fnum(grab(ds, r"rows \(train ([\d,]+)")), 0)
macros["nRowsVal"] = num(fnum(grab(ds, r"rows \(train [\d,]+, val ([\d,]+)")), 0)
macros["nRowsBlind"] = num(fnum(grab(ds, r"val [\d,]+, blind ([\d,]+)\)")), 0)
macros["nPairsAll"] = num(fnum(grab(ds, r"; ([\d,]+) pairs;")), 0)
macros["nNeverTrained"] = num(fnum(grab(ds, r"([\d,]+) never-trained designed validation pairs")), 0)

wd = load(R + "t170/withheld_design.md")
macros["nWithheldCombos"] = num(fnum(grab(wd, r"eic_v1 withholds (\d+)")), 0)

sub = load(R + "t185/SUBMITTED.md")
macros["nRegions"] = num(fnum(grab(sub, r"Content: (\d+) regions of")), 0)
macros["nRegionBins"] = num(fnum(grab(sub, r"regions of ([\d,]+) bins")), 0)
macros["nItems"] = num(fnum(grab(sub, r"(\d+) imputation items")), 0)
macros["nItemsLoao"] = num(fnum(grab(sub, r"\((\d+) leave-one-assay-out")), 0)
macros["nItemsAcc"] = num(fnum(grab(sub, r"(\d+) accessibility-only\)")), 0)
macros["nCorrPairs"] = num(fnum(grab(sub, r"([\d,]+) correction pairs")), 0)

of = load(R + "oracle_floor/ORACLE_FLOOR.md")
macros["nCleanPairs"] = num(fnum(grab(of, r"\): (\d+) pairs \(knob `crop`")), 0)
macros["nCleanCells"] = num(fnum(grab(of, r"`crop\+se` \d+\), (\d+) cells")), 0)
macros["nDupPairs"] = num(fnum(grab(of, r"The other (\d+) headline pairs")), 0)

d1 = load(R + "closeout_D1/D1_RESULT.md")
macros["abTrainPct"] = num(fnum(grab(d1, r"antibody ([\d.]+)% \(1,739")), 1)
macros["blindNewAb"] = num(fnum(grab(d1, r"\| antibody \| \d+ of \d+ \(\d+ accessions\) \| (\d+) of \d+")), 0)
macros["blindAbRows"] = num(fnum(grab(d1, r"\| antibody \| \d+ of \d+ \(\d+ accessions\) \| \d+ of (\d+)")), 0)

lr = load(R + "t305/LAYERS_REAL.md")
macros["sampLo"] = grab(lr, r"it is (\d+)% \(DNase-seq\)")
macros["sampHi"] = grab(lr, r"to (\d+)% \(ChIP-seq control\)")
macros["repLo"] = grab(lr, r"is small: ([\d.]+)% to")
macros["repHi"] = grab(lr, r"is small: [\d.]+% to (\d+)%")
macros["nHalfPairs"] = num(fnum(grab(lr, r"Added: ([\d,]+) pseudoreplicate pairs")), 0)
macros["nIsoPairs"] = num(fnum(grab(lr, r"pseudoreplicate, ([\d,]+) isogenic;")), 0)
macros["nCtPairs"] = num(fnum(grab(lr, r"has [\d,]+ pairs: ([\d,]+) other_cell_type")), 0)

ag = load(R + "t308/AG2_VS_BASELINES.md")
macros["agFair"] = grab(ag, r"\| ag2_matched \| ([\d.]+) \|")
macros["agPosAvg"] = grab(ag, r"\| position_average \| ([\d.]+) \|")
macros["agFloor"] = grab(ag, r"\| pseudoreplicate_half \(noise floor\) \| ([\d.]+) \[")

# ---- the arm summary (16-region screening set, scorer sub-directory p16_sv) -----------
ARM_REL = R + "ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv"


def read_summary(rel):
    lines = load(rel).strip("\n").split("\n")
    head = lines[0].split("\t")
    out = {}
    for ln in lines[1:]:
        c = dict(zip(head, ln.split("\t")))
        out[c["arm"]] = c
    return out


S = read_summary(ARM_REL)
KEYS = ["imputation_raw", "uncertainty_raw", "correction_v3_raw", "correction_only", "composite_v3"]
MODELS = ["convae", "convattn", "unet"]
REF = {"convae": "rung2_base_heldout", "convattn": "rung3_base_heldout", "unet": "rung4_base_heldout"}


def val(table, arm, key):
    c = table.get(arm)
    if c is None:
        return None
    m, r = c.get(key + ".mean", ""), c.get(key + ".range", "")
    try:
        m, r = float(m), float(r)
    except ValueError:
        return None
    if math.isnan(m):
        return None
    return m, (r if not math.isnan(r) else float("nan")), int(c["n_seeds"])


def effect(table, arm, ref, key):
    a, b = val(table, arm, key), val(table, ref, key)
    if a is None or b is None:
        return None
    d = a[0] - b[0]
    bar = max(a[1], b[1])
    flag = 2 if abs(d) > 2 * bar else (1 if abs(d) > bar else 0)
    return d, bar, flag, a[2]


MARK = {0: "", 1: "$^{\\dagger}$", 2: "$^{\\ddagger}$"}


def cell(e):
    if e is None:
        return "--"
    d, bar, flag, n = e
    s = num(d, 3, sign=True) + MARK[flag]
    return s + ("$^{a}$" if n < 3 else "")


# reference rows: mean (range), 3 seeds, p16_sv
NAMES = {"convae": "conv autoencoder", "convattn": "conv + attention", "unet": "U-Net"}
rows = []
for m in MODELS:
    vals = [val(S, REF[m], k) for k in ["imputation_raw", "correction_v3_raw", "uncertainty_raw",
                                        "composite_v3", "correction_only"]]
    rows.append(NAMES[m] + " & " + " & ".join(f"{num(v[0], 3)} ({num(v[1], 3)})" for v in vals) + " \\\\")
macros["TabRefs"] = "\n".join(rows)
for m, tag in zip(MODELS, ["AE", "AT", "UN"]):
    for k, kt in zip(["imputation_raw", "correction_v3_raw", "uncertainty_raw", "composite_v3", "correction_only"],
                     ["Imp", "Cor", "Cal", "Comp", "CorOnly"]):
        v = val(S, REF[m], k)
        macros[f"ref{tag}{kt}"] = num(v[0], 3)
        macros[f"ref{tag}{kt}R"] = num(v[1], 3)

fr = load(R + "final_round/FINAL_ROUND_EQUAL.md")
pa = re.search(r"\| Position average \(baseline\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", fr)
macros["paImp"], macros["paCor"], macros["paCal"], macros["paComp"] = pa.groups()
mn = re.search(r"\| Marginal NB \(baseline\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", fr)
macros["mnCor"], macros["mnComp"] = mn.group(2), mn.group(4)
for arm, tag in [("rung5_base_heldout", "Axial"), ("rung6_base_heldout", "Perc"),
                 ("final_rung4_mech_skip_dna", "SkipDna")]:
    v = val(S, arm, "composite_v3")
    macros[f"comp{tag}"], macros[f"comp{tag}R"] = num(v[0], 3), num(v[1], 3)
for arm, tag in [("final_rung2_n_levels3_equal", "TunAE"), ("final_rung3_attn_at_128_positions_equal", "TunAT"),
                 ("final_rung4_lr3e-3_equal", "TunUN")]:
    v = val(S, arm, "composite_v3")
    macros[f"comp{tag}"], macros[f"comp{tag}R"] = num(v[0], 3), num(v[1], 3)

# ---- the ablation tables -------------------------------------------------------------
def fam(stem, ms=MODELS):
    return {m: f"{stem}_{m}" for m in ms}


ABL = [
    ("Inputs, outputs and depth", None),
    ("ChIP control removed as input", fam("abl_control_input_control_output_only")),
    ("ChIP control not predicted", fam("abl_predict_control_predict_control_off")),
    ("average baseline as input", fam("abl_average_input_average_input")),
    ("two-step, native base depth", fam("abl_two_step_two_step_native")),
    ("two-step, fixed base depth", fam("abl_two_step_two_step_fixed")),
    ("fixed input depth", fam("abl_depth_input_input_depth_fixed")),
    ("DNA", None),
    ("DNA off", fam("abl_dna_off_dna_off")),
    ("language-model DNA embeddings", fam("abl_dna_source_dna_gnlm")),
    ("DNA shifted by 5 Mb (control)", fam("abl_dna_source_dna_shift5mb")),
    ("DNA fusion by FiLM", fam("abl_dna_fusion_fusion_film")),
    ("DNA late join", fam("abl_dna_fusion_fusion_late", ["convae", "convattn"])),
    ("DNA cross-attention", fam("abl_dna_fusion_fusion_xattn", ["convattn", "unet"])),
    ("DNA early join at the stem", fam("abl_dna_fusion_fusion_early", ["unet"])),
    ("Covariates and masking", None),
    ("scrambled covariates (control)", fam("abl_scrambled_covariates_scrambled_covariates")),
    ("numeric covariates only", fam("abl_cov_embedding_cov_numeric_only")),
    ("covariate dropout", fam("abl_cov_embedding_cov_dropout")),
    ("per-value numeric codes", fam("abl_cov_embedding_cov_numeric_per_value")),
    ("injection by adaLN-zero", fam("abl_cov_injection_inject_adaln_zero")),
    ("injection by concatenation", fam("abl_cov_injection_form_inject_concat")),
    ("injection as tokens", fam("abl_cov_injection_form_inject_tokens", ["convattn", "unet"])),
    ("hidden bins dropped, no token", fam("abl_mask_mode_mask_drop")),
    ("Losses", None),
    ("imputation weight 4", fam("abl_losses_loss_impute_x4")),
    ("bin-equal weighting", fam("abl_losses_loss_bin_equal")),
    ("Poisson mean, detached size", fam("abl_losses_loss_poisson_mean")),
    ("NB CRPS loss", fam("abl_losses_loss_nb_crps")),
    ("Training views (reference: all views)", None),
    ("base view only", {"convae": "abl_views_base_only_convae", "convattn": "abl_views_base_only_convattn",
                        "unet": "rung4_views_base_only"}),
    ("base + depth thinning", {"convae": "abl_views_base_depth_convae", "convattn": "abl_views_base_depth_convattn",
                               "unet": "rung4_views_base_depth"}),
    ("base + read-level views", {"convae": "abl_views_base_fastq_convae", "convattn": "abl_views_base_fastq_convattn",
                                 "unet": "rung4_views_base_fastq"}),
    ("base + alignment-level views", {"convae": "abl_views_base_alignment_convae",
                                      "convattn": "abl_views_base_alignment_convattn",
                                      "unet": "rung4_views_base_alignment"}),
    ("all views, same-view pairs only", {"convae": "abl_views_all_same_view_only_convae",
                                         "convattn": "abl_views_all_same_view_only_convattn",
                                         "unet": "rung4_views_all_same_view_only"}),
    ("Count transforms (reference: offset, pseudocount 1)", None),
    ("log1p", fam("abl_count_transforms_tf_log1p")),
    ("asinh", fam("abl_count_transforms_tf_asinh")),
    ("Pearson residuals", fam("abl_count_transforms_tf_pearson")),
    ("local ratio", fam("abl_count_transforms_tf_local_ratio")),
    ("per-window quantile", fam("abl_count_transforms_tf_quantile")),
    ("raw counts", fam("abl_count_transforms_tf_raw")),
    ("soft clip", fam("abl_count_transforms_tf_soft_clip")),
    ("offset, pseudocount 0.1", fam("abl_count_transforms_offset_eps0.1")),
    ("offset, pseudocount 0.01", fam("abl_count_transforms_offset_eps0.01")),
]


def abl_table(keys):
    out = []
    for label, arms in ABL:
        if arms is None:
            out.append(f"\\multicolumn{{{1 + 3 * len(keys)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
            continue
        cells = []
        for k in keys:
            for m in MODELS:
                a = arms.get(m)
                cells.append(cell(effect(S, a, REF[m], k)) if a else "--")
        out.append(label + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


# column order: per key, the three models
macros["TabAblFull"] = abl_table(["imputation_raw", "uncertainty_raw"])
macros["TabAblPairs"] = abl_table(["correction_v3_raw", "correction_only", "composite_v3"])

# counts over the arm-model cells
cnt = {k: {"n": 0, "up1": 0, "dn1": 0, "up2": 0, "dn2": 0} for k in KEYS}
for label, arms in ABL:
    if arms is None:
        continue
    for m, a in arms.items():
        for k in KEYS:
            e = effect(S, a, REF[m], k)
            if e is None:
                continue
            c = cnt[k]
            c["n"] += 1
            if e[2] >= 1:
                c["up1" if e[0] > 0 else "dn1"] += 1
            if e[2] == 2:
                c["up2" if e[0] > 0 else "dn2"] += 1
macros["nCells"] = num(cnt["composite_v3"]["n"], 0)
macros["nArmRows"] = num(sum(1 for _, a in ABL if a), 0)
for k, t in zip(KEYS, ["Imp", "Cal", "Cor", "CorOnly", "Comp"]):
    macros[f"n{t}Up"] = num(cnt[k]["up1"], 0)
    macros[f"n{t}Dn"] = num(cnt[k]["dn1"], 0)
    macros[f"n{t}UpTwo"] = num(cnt[k]["up2"], 0)
    macros[f"n{t}DnTwo"] = num(cnt[k]["dn2"], 0)
macros["nCellsCal"] = num(cnt["uncertainty_raw"]["n"], 0)

# named effects for the prose
def emac(tag, arm, m, key, d=3):
    e = effect(S, arm, REF[m], key)
    macros[tag] = num(e[0], d, sign=True)
    macros[tag + "Bar"] = num(e[1], d)
    a = val(S, arm, key)
    macros[tag + "Val"] = num(a[0], d)
    macros[tag + "ValR"] = num(a[1], d)


emac("eLrImpAE", "abl_count_transforms_tf_local_ratio_convae", "convae", "imputation_raw")
emac("eLrImpAT", "abl_count_transforms_tf_local_ratio_convattn", "convattn", "imputation_raw")
emac("eLrImpUN", "abl_count_transforms_tf_local_ratio_unet", "unet", "imputation_raw")
emac("eQuImpAE", "abl_count_transforms_tf_quantile_convae", "convae", "imputation_raw")
emac("eQuImpAT", "abl_count_transforms_tf_quantile_convattn", "convattn", "imputation_raw")
emac("eQuImpUN", "abl_count_transforms_tf_quantile_unet", "unet", "imputation_raw")
emac("eTokImpUN", "abl_cov_injection_form_inject_tokens_unet", "unet", "imputation_raw")
emac("eAvgImpUN", "abl_average_input_average_input_unet", "unet", "imputation_raw")
emac("eDnaOffImpAT", "abl_dna_off_dna_off_convattn", "convattn", "imputation_raw")
emac("eShiftImpAT", "abl_dna_source_dna_shift5mb_convattn", "convattn", "imputation_raw")
emac("eWfourImpAT", "abl_losses_loss_impute_x4_convattn", "convattn", "imputation_raw")
emac("eFixCorUN", "abl_depth_input_input_depth_fixed_unet", "unet", "correction_v3_raw")
emac("eFixImpUN", "abl_depth_input_input_depth_fixed_unet", "unet", "imputation_raw")
emac("eFixCompUN", "abl_depth_input_input_depth_fixed_unet", "unet", "composite_v3")
emac("eScrCoAE", "abl_scrambled_covariates_scrambled_covariates_convae", "convae", "correction_only")
emac("eScrCoAT", "abl_scrambled_covariates_scrambled_covariates_convattn", "convattn", "correction_only")
emac("eScrCoUN", "abl_scrambled_covariates_scrambled_covariates_unet", "unet", "correction_only")
emac("eScrImpAE", "abl_scrambled_covariates_scrambled_covariates_convae", "convae", "imputation_raw")
emac("eBoCoAE", "abl_views_base_only_convae", "convae", "correction_only")
emac("eBoCoAT", "abl_views_base_only_convattn", "convattn", "correction_only")
emac("eBoCoUN", "rung4_views_base_only", "unet", "correction_only")
emac("eBoCompAE", "abl_views_base_only_convae", "convae", "composite_v3")
emac("eBoCompAT", "abl_views_base_only_convattn", "convattn", "composite_v3")
emac("eBoCompUN", "rung4_views_base_only", "unet", "composite_v3")
emac("eBdCoAE", "abl_views_base_depth_convae", "convae", "correction_only")
emac("eBdCoUN", "rung4_views_base_depth", "unet", "correction_only")
emac("eBaImpAE", "abl_views_base_alignment_convae", "convae", "imputation_raw")
emac("eBaImpAT", "abl_views_base_alignment_convattn", "convattn", "imputation_raw")
emac("eBaCorAE", "abl_views_base_alignment_convae", "convae", "correction_v3_raw")
emac("eBaCorAT", "abl_views_base_alignment_convattn", "convattn", "correction_v3_raw")
emac("eSvImpAE", "abl_views_all_same_view_only_convae", "convae", "imputation_raw")
emac("eSvImpAT", "abl_views_all_same_view_only_convattn", "convattn", "imputation_raw")
emac("eSvCompUN", "rung4_views_all_same_view_only", "unet", "composite_v3")
emac("eNatCorAE", "abl_two_step_two_step_native_convae", "convae", "correction_v3_raw")
emac("eNatCorAT", "abl_two_step_two_step_native_convattn", "convattn", "correction_v3_raw")
emac("ePearCompUN", "abl_count_transforms_tf_pearson_unet", "unet", "composite_v3")
emac("eConcCorAT", "abl_cov_injection_form_inject_concat_convattn", "convattn", "correction_v3_raw")
emac("eCrpsCorAT", "abl_losses_loss_nb_crps_convattn", "convattn", "correction_v3_raw")
emac("eNumCorUN", "abl_cov_embedding_cov_numeric_only_unet", "unet", "correction_v3_raw")
emac("eCtlImpAT", "abl_control_input_control_output_only_convattn", "convattn", "imputation_raw")
v = val(S, "abl_count_transforms_tf_anscombe_convae", "imputation_raw")
macros["anscImp"] = num(v[0], 3)
v = val(S, "abl_count_transforms_tf_anscombe_convae", "correction_v3_raw")
macros["anscCor"] = num(v[0], 3)

# views figure: arm minus all views, with the comparison bar as the error bar
VIEWS = [("base only", "base_only"), ("base + depth", "base_depth"), ("base + read-level", "base_fastq"),
         ("base + alignment", "base_alignment"), ("same-view pairs only", "all_same_view_only")]


def varm(m, s):
    return f"rung4_views_{s}" if m == "unet" else f"abl_views_{s}_{m}"


for key, tname in [("correction_only", "tblViewsCo"), ("imputation_raw", "tblViewsImp")]:
    trow = []
    for i, (lab, s) in enumerate(VIEWS):
        r = [i, "{" + lab + "}"]
        for m in MODELS:
            e = effect(S, varm(m, s), REF[m], key)
            if e is None or abs(e[0]) > 5:
                r += ["nan", "nan"]
            else:
                r += [raw(e[0]), raw(e[1])]
        trow.append(r)
    tables[tname] = (["y", "label", "dAE", "bAE", "dAT", "bAT", "dUN", "bUN"], trow)

wv = load(R + "closeout_A3/WHERE_VIEWS.md")
wv2 = wv[wv.index("## 2. Per pair type"):]
macros["thinAll"] = grab(wv2, r"\| all views \| [^|]+\| [^|]+\| [^|]+\| ([\d.]+) \(")
macros["thinBase"] = grab(wv2, r"\| base view only \| [^|]+\| [^|]+\| [^|]+\| ([\d.]+) \(")
macros["thinWithLo"] = grab(wv, r"with depth views \(([\d.]+) to")
macros["thinWithHi"] = grab(wv, r"with depth views \([\d.]+ to ([\d.]+)\)")
macros["thinWithoutHi"] = grab(wv, r"at most ([\d.]+); file")
macros["mnAll"] = grab(wv2, r"\| all views \| [^|]+\| [^|]+\| ([\d.]+) \(")
macros["mnAlign"] = grab(wv2, r"\| base \+ alignment-level \| [^|]+\| [^|]+\| ([\d.]+) \(")
macros["mnAlignR"] = grab(wv2, r"\| base \+ alignment-level \| [^|]+\| [^|]+\| [\d.]+ \(([\d.]+)\)")
macros["nMnPairs"] = grab(wv, r"MAPQ \+ no-dedup \((\d+) pairs\)")

# ---- the 16-versus-256 check ------------------------------------------------------------
pr = load(R + "region_sampling_checks/PAIR_REGIONS_V3.md")
chk = []
for p in ("8", "16", "32"):
    cells = grab(pr, r"\| " + p + r" \| ([a-z |]+)\|\n").split("|")
    cells = [c.strip() for c in cells if c.strip()]
    chk.append(p + " & " + " & ".join(cells) + " \\\\")
macros["TabCheck"] = "\n".join(chk)
macros["nBoot"] = num(fnum(grab(pr, r"with (\d+) region-bootstrap draws")), 0)
macros["nClusters"] = num(fnum(grab(pr, r"over (\d+) clusters")), 0)
rc = load(R + "region_sampling_checks/REGION_CHECKS.md")
macros["pairPct"] = grab(rc, r"at the (\d+)th percentile of all")
macros["pairLowPct"] = grab(rc, r"lowest at the (\d+)th\)")
macros["nCandBlocks"] = num(fnum(grab(rc, r"all ([\d,]+) candidate blocks")), 0)

cv = load(R + "region_sampling_checks/check_v3.tsv").strip("\n").split("\n")
ch = cv[0].split("\t")
crow = [dict(zip(ch, ln.split("\t"))) for ln in cv[1:]]


def crit(p, cond, q):
    r = [c for c in crow if c["p"] == p and c["condition"] == cond and c["scope"] == "ALL"]
    return r[0][q]


macros["condOneP"] = grab(crit("16", "1", "quantity"), r"(\d+) of")
macros["condOneN"] = grab(crit("16", "1", "quantity"), r"of (\d+)")
macros["condTwoShare"] = num(float(crit("16", "2", "value")) * 100, 0)
macros["condFourP"] = grab(crit("16", "4", "quantity"), r"(\d+) of")
macros["condFourN"] = grab(crit("16", "4", "quantity"), r"of (\d+)")
macros["condSixP"] = grab(crit("16", "6", "quantity"), r"(\d+) of")
macros["condSixN"] = grab(crit("16", "6", "quantity"), r"of (\d+)")

# per run-seed values on 16 regions against the 256-region bootstrap interval
ARMS4 = [("rung2_base_heldout", "conv autoencoder"), ("rung3_base_heldout", "conv + attention"),
         ("rung4_base_heldout", "U-Net"), ("final_rung4_mech_skip_dna", "U-Net, DNA-only skips")]
fig = []
y = 0
inside = {"correction_v3": [0, 0], "correction_only": [0, 0]}
for arm, lab in ARMS4:
    for s in ("0", "1", "2"):
        r = {c["quantity"]: c for c in crow if c["p"] == "16" and c["condition"] == "1"
             and c["scope"] == f"{arm}/seed{s}"}
        a, b = r["correction_v3"], r["correction_only"]
        for q, c in (("correction_v3", a), ("correction_only", b)):
            inside[q][0] += c["pass"] == "pass"
            inside[q][1] += 1
        fig.append([y, "{" + lab + ", seed " + s + "}", raw(a["value"]), raw(a["value_all_regions"]),
                    raw(a["lo"]), raw(a["hi"]), raw(b["value"]), raw(b["value_all_regions"]),
                    raw(b["lo"]), raw(b["hi"])])
        y += 1
tables["tblRegion"] = (["y", "label", "cs", "ca", "clo", "chi", "os", "oa", "olo", "ohi"], fig)
macros["inCorP"], macros["inCorN"] = num(inside["correction_v3"][0], 0), num(inside["correction_v3"][1], 0)
macros["inCoP"], macros["inCoN"] = num(inside["correction_only"][0], 0), num(inside["correction_only"][1], 0)

# 256-region means of the four arms (correction terms), mean and range over 3 seeds
def mean_range(arm, q):
    xs = [float(c["value_all_regions"]) for c in crow if c["p"] == "16" and c["condition"] == "1"
          and c["scope"].startswith(arm + "/") and c["quantity"] == q]
    assert len(xs) == 3
    return sum(xs) / 3, max(xs) - min(xs)


rows256 = []
for arm, lab in ARMS4:
    a, b = mean_range(arm, "correction_v3"), mean_range(arm, "correction_only")
    rows256.append(f"{lab} & {num(a[0], 3)} ({num(a[1], 3)}) & {num(b[0], 3)} ({num(b[1], 3)}) \\\\")
macros["TabTwoFiftySix"] = "\n".join(rows256)
uc, ac = mean_range("rung4_base_heldout", "correction_v3"), mean_range("rung2_base_heldout", "correction_v3")
macros["dUnAeCorTwoFS"] = num(uc[0] - ac[0], 3, sign=True)
d3 = [c for c in crow if c["p"] == "16" and c["condition"] == "3" and c["scope"] == "rung2_base_heldout vs rung4_base_heldout"][0]
macros["compAeUnAll"] = num(abs(fnum(grab(d3["note"], r"all regions: composite delta (-?[\d.]+)"))), 3)
macros["compAeUnBar"] = num(fnum(grab(d3["note"], r"all regions: composite delta -?[\d.]+ within 2 x spread ([\d.]+)")), 3)

macros["nResample"] = num(fnum(grab(rc, r"at least 95% of ([\d,]+) paired resamples")), 0)
macros["regNinetySix"] = grab(rc, r"falls from \+([\d.]+) to")
macros["regNinetySixTo"] = grab(rc, r"falls from \+[\d.]+ to \+([\d.]+)")

# ---- the rescoring table (7 decisive arms; 256-region columns filled by the hook) -------
RESCORE_REL = R + "rescoring_p256/arm_summary_p256_sv.tsv"
S256 = read_summary(RESCORE_REL) if (VAULT / RESCORE_REL).is_file() else None
DECISIVE = [
    ("scrambled covariates", "abl_scrambled_covariates_scrambled_covariates", "correction-only use of covariates"),
    ("base view only", None, "do the views help"),
    ("two-step, native", "abl_two_step_two_step_native", "two-step off"),
    ("two-step, fixed", "abl_two_step_two_step_fixed", "two-step off"),
    ("fixed input depth", "abl_depth_input_input_depth_fixed", "input depth"),
    ("average input", "abl_average_input_average_input", "average input"),
    ("imputation weight 4", "abl_losses_loss_impute_x4", "loss weights"),
]
rs = []
for lab, stem, why in DECISIVE:
    for m in MODELS:
        arm = varm(m, "base_only") if stem is None else f"{stem}_{m}"
        c16 = [cell(effect(S, arm, REF[m], k)) for k in ("correction_v3_raw", "correction_only")]
        if S256 is not None:
            c256 = [cell(effect(S256, arm, REF[m], k)) for k in ("correction_v3_raw", "correction_only")]
        else:
            c256 = ["pending", "pending"]
        rs.append(f"{lab} & {NAMES[m]} & " + " & ".join(c16 + c256) + " \\\\")
macros["TabRescore"] = "\n".join(rs)

# ---- settled design document ---------------------------------------------------------------
sd = load(INS + "closeout/settled miniCANDI design.md")
macros["nDesignRows"] = num(len(re.findall(r"^\| \d+ \| ", sd, re.M)), 0)
macros["testsPassed"] = grab(sd, r"Result: (\d+) passed")
macros["testsSkipped"] = grab(sd, r"passed, (\d+) skipped")
macros["testsFailed"] = grab(sd, r"skipped, (\d+) failed")
macros["rescoreSeeds"] = grab(sd, r"\*\*The run\.\*\* (\d+) run-seeds")
macros["rescoreCost"] = num(fnum(grab(sd, r"about ([\d,]+) allocated CPU-hours: 20 conv")), 0)
macros["costAE"] = grab(sd, r"20 conv autoencoder run-seeds x ([\d.]+)")
macros["costAT"] = grab(sd, r"21 conv \+ attention x ([\d.]+)")
macros["costUN"] = grab(sd, r"21 U-Net x (\d+) =")
macros["nRunsAE"] = grab(sd, r"CPU-hours: (\d+) conv autoencoder run-seeds")
macros["nRunsAT"] = grab(sd, r"run-seeds x [\d.]+ \+ (\d+) conv \+ attention")
macros["nRunsUN"] = grab(sd, r"conv \+ attention x [\d.]+ \+ (\d+) U-Net")
macros["dTwoMarginUN"] = grab(sd, r"\(0\.507 - 0\.383 = ([\d.]+) against")
macros["dTwoBarUN"] = grab(sd, r"against 2 x 0\.071 = ([\d.]+)\)")
macros["corrOnlyAEsixteen"] = grab(sd, r"correction-only U-Net [\d.]+ \([\d.]+\) against ([\d.]+) \(")
macros["skipCor"] = grab(sd, r"U-Net skips with DNA only, 256 regions: correction ([\d.]+) \(")
macros["skipCorR"] = grab(sd, r"U-Net skips with DNA only, 256 regions: correction [\d.]+ \(([\d.]+)\)")
macros["unCorTwoFS"] = grab(sd, r"against ([\d.]+) \([\d.]+\), \+0\.211")
macros["skipImp"] = grab(sd, r"imputation ([\d.]+) \([\d.]+\) against 0\.432")

# ---- signal p-value and peaks ---------------------------------------------------------------
e2 = load(R + "closeout_E2/E2_RESULTS.md")
fx = load(R + "closeout_E2/FORMULA_EXTENSION.md")
ib = re.search(r"\| inbin \(today\) \| none \| ([\d.]+) / ([\d.]+) / ([\d.]+) \| ([\d.]+) \| ([\d.]+) / ([\d.]+) / ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", fx).groups()
bx = re.search(r"\| box_d \| centred box of d bp \| \*\*([\d.]+)\*\* / ([\d.]+) / ([\d.]+) \| \*\*([\d.]+)\*\* \| ([\d.]+) / \*\*([\d.]+)\*\* / \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \|", fx).groups()
labels = ["ChIP-seq Pearson, all targets", "ChIP-seq Pearson, table fragment length",
          "ChIP-seq Pearson, estimated fragment length", "ChIP-seq slope", "ChIP-seq peak-tile recall",
          "ChIP-seq peak-tile precision", "ChIP-seq peak-tile F1", "DNase-seq Pearson", "DNase-seq peak-tile F1"]
macros["TabFormula"] = "\n".join(f"{l} & {a} & {b} \\\\" for l, a, b in zip(labels, ib, bx))
macros["nFormTargets"] = grab(fx, r"There were (\d+) validation targets")
macros["nFormChip"] = grab(fx, r"with a pipeline track: (\d+) ChIP-seq")
macros["nVariants"] = grab(fx, r"chosen among (\d+) variants")
macros["eTwoCounts"] = grab(e2, r"counts ([\d.]+), signal \(expectation\)")
macros["eTwoSigExp"] = grab(e2, r"signal \(expectation\) ([\d.]+)")
macros["eTwoSigPlug"] = grab(e2, r"signal \(plug-in\) ([\d.]+)")
macros["eTwoSigPc"] = grab(e2, r"signal with the predicted control (\d\.\d+)")
macros["eTwoRspecCounts"] = grab(e2, r"r_spec on the same 8 assays: counts ([\d.]+)")
macros["eTwoRspecPc"] = grab(e2, r"signal with the predicted control ([\d.]+) \(other")
macros["eTwoNAssays"] = grab(e2, r"mean over the (\d+) scorable assays")
macros["lamHidden"] = grab(e2, r"Control hidden \(regime accessibility_only, \d+ targets, \d+ cells\): median of the per-target median ratio ([\d.]+)")
macros["lamHiddenK"] = grab(e2, r"; (\d+) of 20 targets have a median ratio within 10%")
macros["lamHiddenN"] = grab(e2, r"Control hidden \(regime accessibility_only, (\d+) targets")
macros["lamVisible"] = grab(e2, r"Control visible \(leave_one_assay_out, \d+ targets, \d+ cells\): median ratio ([\d.]+)")
macros["lamVisibleK"] = grab(e2, r"median ratio [\d.]+; (\d+) of 41 within 10%")
macros["lamVisibleN"] = grab(e2, r"Control visible \(leave_one_assay_out, (\d+) targets")

# ---- second experiments, lab and donor ---------------------------------------------------------
d2 = load(R + "closeout_D2/D2_PREP.md")
macros["nSecond"] = grab(d2, r"second-experiment rows: (\d+) \(other lab")
macros["nSecondOther"] = grab(d2, r"\(other lab (\d+), same lab")
macros["nSecondSame"] = grab(d2, r"same lab (\d+)\)`")
macros["nSecondVal"] = grab(d2, r"`: (\d+) validation targets with a second experiment")
macros["smokeOut"] = grab(d2, r"All pairs: (\d+) of \d+ outside")
macros["smokeOtherOut"] = grab(d2, r"Other-lab pairs: (\d+) of \d+ outside")
macros["smokeOtherN"] = grab(d2, r"Other-lab pairs: \d+ of (\d+) outside")
macros["smokeSameOut"] = grab(d2, r"same-lab pairs: (\d+) of \d+")
macros["smokeSameN"] = grab(d2, r"same-lab pairs: \d+ of (\d+)")
macros["smokeValOut"] = grab(d2, r"Validation: (\d+) of \d+ outside")
macros["smokeValN"] = grab(d2, r"Validation: \d+ of (\d+) outside")
d2i = load(INS + "closeout/closeout D2 second experiments and cross-lab pairs.md")
macros["labDonorBroad"] = grab(d2i, r"`experiment;lab;donor` in broad histone \((\d+) pairs")
macros["labDonorBroadCells"] = grab(d2i, r"in broad histone \(\d+ pairs, (\d+) cells\)")
macros["labDonorNarrow"] = grab(d2i, r"narrow histone \((\d+) pairs, \d+ cells\)\. Pooled")
macros["pureLabPooled"] = grab(d2i, r"has (\d+) pairs from 3 cells \(C17")
macros["minPairs"] = grab(d2i, r"`MIN_PAIRS = (\d+)`")

# ---- replicate ladder ---------------------------------------------------------------------------
e4 = load(R + "closeout_E4/E4_SMOKE.md")
e4d = load(R + "closeout_E4/E4_MIXED_DIAG.md")
macros["ladSamp"] = grab(e4, r"sampling ([\d.]+) \[")
macros["ladSampT"] = grab(e4, r"against ([\d.]+) \[[\d., ]+\] there")
macros["ladPairs"] = grab(e4, r"Pair list: (\d+) H3K36me3 pairs")
macros["ladTele"] = grab(e4, r"Telescoping \(`total_variance.telescoping_check`\): (\d+) of \d+ groups pass")
macros["ladTeleN"] = grab(e4, r"\): \d+ of (\d+) groups pass")
macros["ladOneSided"] = grab(e4d, r"\((\d+)% lower\)")
macros["ladLabDonor"] = num(float(grab(e4, r"the lab-with-donor increment is (-?\d\.\d+)")), 4)

# ---- data build, blind build, compute, clusters -------------------------------------------------
d4 = load(R + "closeout_D4/summary.txt")
macros["blindRows"] = grab(d4, r"batch all: rows (\d+)")
macros["blindAlloc"] = num(fnum(grab(d4, r"batch all: rows \d+, M reads [\d]+, sum clean wall \d+ h, alloc CPU-h (\d+)")), 0)
nb = load(INS + "long-horizon notepad.md")
macros["tallyTotal"] = num(fnum(grab(nb, r"\| Both \| [\d,.]+ \| [\d,.]+ \| ([\d,.]+) \|")), 0)
macros["tallyFir"] = num(fnum(grab(nb, r"\| Fir \| [\d,.]+ \| [\d,.]+ \| ([\d,.]+) \|")), 0)
macros["tallyNibi"] = num(fnum(grab(nb, r"\| Nibi \| [\d,.]+ \| [\d,.]+ \| ([\d,.]+) \|")), 0)
macros["oldScoreCost"] = grab(nb, r"First: about (\d+) CPU-h \(overnight brief")
macros["oldCheckCost"] = num(fnum(grab(nb, r"the correction line calls the old figure ([\d,]+)")), 0)
macros["blindOldCost"] = num(fnum(grab(nb, r"First: ([\d,]+) CPU-h for priority 4")), 0)
macros["blindNewCost"] = num(fnum(grab(nb, r"Corrected: about ([\d,]+) allocated CPU-h on Nibi")), 0)
macros["mixOldCost"] = num(fnum(grab(nb, r"Cost of the mix-and-match round\. First: about ([\d,]+) CPU-h")), 0)
fp = load(INS + "closeout/final 24 hours plan.md")
macros["nibiUsed"] = num(fnum(grab(fp, r"read-only \(([\d,]+) GiB, quota")), 0)
macros["nibiQuota"] = num(fnum(grab(fp, r"GiB, quota ([\d,]+) GiB\)")), 0)
macros["nibiWork"] = num(fnum(grab(fp, r"`onion/work`, ([\d,]+) GiB\)")), 0)
macros["firQueued"] = grab(fp, r"The (\d+) queued Fir jobs stay queued")

# ---- ready queue table ------------------------------------------------------------------------
rq = load(INS + "closeout/ready queue for scaling.md")
qrows = re.findall(r"^\| (\d+) \| ([^|]+) \| ([^|]+) \| [^|]+ \| ([^|]+) \|$", rq, re.M)
assert len(qrows) == 19, len(qrows)


def tex_text(s):
    s = s.strip().replace("`", "")
    s = s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
    return s


macros["TabQueue"] = "\n".join(f"{o} & {tex_text(e)} & {tex_text(c)} & {tex_text(h)} \\\\"
                               for o, e, c, h in qrows)
macros["mixCost"] = num(fnum(grab(rq, r"\| 474 \+ ([\d,]+) \(Nibi\)")), 0)
macros["mixGate"] = grab(rq, r"\| (\d+) \+ [\d,]+ \(Nibi\)")

# ---- results that landed after version 1 (2026-11-18) ------------------------------------------
def tsv_rows(rel):
    lines = load(rel).strip("\n").split("\n")
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def gnum(v):
    """A small pair value in LaTeX: four decimals, or three significant digits times a power of ten."""
    v = float(v)
    if abs(v) >= 1e-3:
        return num(v, 4)
    m, e = f"{v:.2e}".split("e")
    m = float(m)
    s = f"{abs(m):.2f}\\times10^{{{int(e)}}}"
    return "\\ensuremath{" + ("-" if m < 0 else "") + s + "}"


def pct(v, d=1):
    return num(100 * float(v), d) + "\\%"


# replicate ladder: two-sided, assay-matched (results/closeout_E4/ladder_2026-11-18)
LADR = R + "closeout_E4/ladder_2026-11-18/"
lad = {(r["assay_class"], r["rung"], r["pair_class"]): r for r in tsv_rows(LADR + "ladder.tsv")
       if r["stratum"] == "all" and r["band"] == "all"}
ladmd = load(LADR + "ladder.md")
macros["ladChainOk"] = grab(ladmd, r"\n.*?: (\d+) of \d+ \(class, stratum, band\) groups telescope exactly")
macros["ladChainN"] = grab(ladmd, r"\n.*?: \d+ of (\d+) \(class, stratum, band\) groups telescope exactly")
CLASSES = [("accessibility_dnase", "DNase-seq"), ("broad_histone", "broad marks"), ("narrow_histone", "narrow marks")]
LAYERS = [("pseudoreplicate", "", "read sampling"), ("isogenic", "", "biological replicate"),
          ("other_experiment", "experiment;lab;donor", "lab with donor"),
          ("other_experiment", "experiment;donor", "repeat with donor"),
          ("other_cell_type", "", "cell type")]
lrows = []
for c, cl in CLASSES:
    for rg, pc, nm in LAYERS:
        r = lad[(c, rg, pc)]
        if r["layer_increment"] in ("", "nan"):
            continue
        lrows.append(f"{cl} & {nm} & {r['n_pairs']} ({r['n_cells']}) & {gnum(r['layer_increment'])} "
                     f"[{gnum(r['ci_low'])}, {gnum(r['ci_high'])}] & {pct(r['share'])} ({r['share_units']}) \\\\")
macros["TabLadder"] = "\n".join(lrows)
for c, t in (("accessibility_dnase", "Dn"), ("broad_histone", "Br"), ("narrow_histone", "Nr")):
    for rg, pc, k in (("pseudoreplicate", "", "Samp"), ("isogenic", "", "Rep"), ("other_cell_type", "", "Ct")):
        macros[f"lad{t}{k}"] = pct(lad[(c, rg, pc)]["share"])
macros["ladBrLd"] = gnum(lad[("broad_histone", "other_experiment", "experiment;lab;donor")]["layer_increment"])
macros["ladBrLdSh"] = pct(lad[("broad_histone", "other_experiment", "experiment;lab;donor")]["share"])
macros["ladDnRepInc"] = gnum(lad[("accessibility_dnase", "isogenic", "")]["layer_increment"])
macros["ladAtacN"] = lad[("accessibility_atac", "pseudoreplicate", "")]["n_pairs_all"]
macros["ladAtacC"] = lad[("accessibility_atac", "pseudoreplicate", "")]["n_cells_all"]
macros["ladBrLabN"] = lad[("broad_histone", "other_experiment", "experiment;lab")]["n_pairs"]
macros["ladBrLabC"] = lad[("broad_histone", "other_experiment", "experiment;lab")]["n_cells"]

# second-experiment layers, rows pooled by assay class (results/closeout_D2/layers_2026-11-18)
dl = {(r["assay_or_class"], r["pair_class"]): r for r in tsv_rows(R + "closeout_D2/layers_2026-11-18/layers.tsv")
      if r["stratum"] == "all" and r["band"] == "all" and r["pooled_by_class"] == "1"
      and r["rung"] == "other_experiment"}
for c, t in (("broad_histone", "Br"), ("narrow_histone", "Nr")):
    for pc, k in (("", "All"), ("experiment;lab;donor", "Ld")):
        r = dl[(c, pc)]
        macros[f"dTwo{t}{k}"] = num(float(r["layer_increment"]), 4, sign=True)
        macros[f"dTwo{t}{k}Lo"] = num(float(r["ci_low"]), 4)
        macros[f"dTwo{t}{k}Hi"] = num(float(r["ci_high"]), 4)
        macros[f"dTwo{t}{k}N"] = r["n_pairs"]
        macros[f"dTwo{t}{k}C"] = r["n_cells"]
        macros[f"dTwo{t}{k}Tech"] = r["tech_diff_pairs"]
macros["dTwoSmall"] = str(max(int(r["n_pairs"]) for (c, _), r in dl.items() if c.startswith("accessibility")))
pid = load(INS + "closeout/PI decisions 2026-11-17.md")
macros["dTwoDropped"] = grab(pid, r"\| 4 \| The (\d+) unbuilt DNase-seq second-experiment rows")
macros["dTwoKept"] = grab(pid, r"The add-on stays at (\d+) experiments")

# QC side tables, the 1 bp store, the archive and the scratch (coverage.txt; the run-log snapshot)
cov = {}
for ln in load(R + "closeout_A4/qc_tables_2026-11-18/coverage.txt").splitlines():
    c = ln.split("\t")
    if len(c) == 3 and "/" in c[2]:
        cov[c[0]] = [int(x) for x in c[2].split("/")]
macros["qcRows"] = num(cov["nsc"][1], 0)
for k, t in (("nsc", "Nsc"), ("frag_len", "Frag"), ("nrf", "Nrf"), ("frip", "Frip")):
    macros[f"qc{t}"] = num(cov[k][0], 0)
    macros[f"qc{t}Pct"] = num(100 * cov[k][0] / cov[k][1], 1)
assert cov["rsc"] == cov["nsc"] and cov["pbc1"] == cov["nrf"] == cov["pbc2"]
pl = load(INS + "run_state/plan.md")
q = re.search(r"records whole ([\d,]+) \(= rows\), pr1 ([\d,]+) \(= rows\); row states in the whole pass: "
              r"([\d,]+) ok, ([\d,]+) no read set, ([\d,]+) refused", pl).groups()
macros["qcWhole"], macros["qcPrOne"], macros["qcOk"], macros["qcNoRead"], macros["qcRefused"] = \
    (num(fnum(x), 0) for x in q)
macros["bpViews"] = grab(pl, r"RECOUNT DONE: \d+ of \d+ tasks COMPLETED \([^)]*\), (\d+) views in CO-A7/res_store")
macros["bpPts"] = num(fnum(grab(pl, r"independent (\d+) points, store \d+, bins differing 0 -> PASS")), 0)
macros["bpPtsStore"] = num(fnum(grab(pl, r"independent \d+ points, store (\d+), bins differing 0 -> PASS")), 0)
macros["bpMeas"] = grab(pl, r"tables were built \(CO-A7/tables/eic_res_v1: (\d+) measurements")
macros["bpPairs"] = grab(pl, r"tables were built \(CO-A7/tables/eic_res_v1: \d+ measurements, (\d+) pairs")
grab(pl, r"(1 bp verify \d+ and variance \d+ running)")
a = re.search(r"(\d+) of (\d+) tars OK \(tar -d, 0 non-empty diff files\), 0 failed, ([\d,]+) GiB, ([\d,]+) entries",
              pl).groups()
macros["arcTars"], macros["arcGib"], macros["arcEntries"] = a[0], num(fnum(a[2]), 0), num(fnum(a[3]), 0)
s1 = re.search(r"Scratch ([\d,]+) -> ([\d,]+) GiB \(soft quota ([\d,]+): still read-only\)", pl).groups()
s2 = re.search(r"Scratch ([\d,]+) -> ([\d,]+) GiB of ([\d,]+); a write test passed", pl).groups()
macros["scBefore"], macros["scMid"], macros["scAfter"] = (num(fnum(x), 0) for x in (s1[0], s1[1], s2[1]))
# rescoring status (run log)
rsub = re.search(r"- (2026-11-17)T([\d:]+)Z RESCORING SUBMITTED on Nibi.*?: (\d+) run-seeds", pl).groups()
macros["rsDay"], macros["rsStart"], macros["rsN"] = rsub
macros["rsHours"] = grab(pl, r"so about ([\d.]+) h for the predict stage")
macros["rsCap"] = grab(pl, r"raise the cap to (\d+) active run-seeds")
macros["rsNow"] = grab(pl, r"Rescoring ([\d:]+)Z: \d+ active, \d+ scored")
macros["rsScored"] = grab(pl, r"Rescoring [\d:]+Z: \d+ active, (\d+) scored")
macros["rsNowDay"] = grab(pl, r"- (2026-11-1\d)T[\d:]+Z REPLICATE LADDER REPORT")
rounds = -(-int(macros["rsN"]) // int(macros["rsCap"]))
macros["rsRounds"] = str(rounds)
macros["rsTotal"] = num(rounds * float(macros["rsHours"]), 0)

# PI decisions of 2026-11-17: the nine taken, and the two moved to the next session
drows = []
body = grab(pid, r"\| # \| Decision \| PI choice \| Consequence \|\n\|[-|]+\|\n(.*?)\n\n")
for ln in body.splitlines():
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) != 4 or not c[0].isdigit():
        continue
    k = re.sub(r"\s*\(jobs? [\d to]+\)", "", c[3])
    k = re.sub(r"Submitted 2026-11-17: layers \d+; baselines [\d, ]+", "Submitted on Nibi; the layers are done", k)
    k = re.sub(r"\s*\(ready queue entry (\d+), `DESIGN=full`\)", r" (ready queue entry \1)", k)
    drows.append(f"{c[0]} & {tex_text(c[1])} & {tex_text(c[2])} & {tex_text(k)} \\\\")
assert len(drows) == 9, len(drows)
macros["TabDecisions"] = "\n".join(drows)
macros["nDecisions"] = str(len(drows))
mv = re.search(r'PI ruling of 2026-11-17: "(scaling is next session[^"]*)"\. On the data version: "([^"]*)"', sd)
macros["piQuoteScale"], macros["piQuoteData"] = (tex_text(x) for x in mv.groups())
# =======================================================================================

for name in (*macros, *tables):
    assert re.fullmatch(r"[A-Za-z]+", name), f"macro/table name {name!r}: letters only"


def in_vault(p):
    try:
        return p.relative_to(VAULT).as_posix()
    except ValueError:
        return None


lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
for k, (cols, rows_) in sorted(tables.items()):
    body = "\n".join(" ".join(str(c) for c in r) for r in rows_)
    lines.append(f"\\pgfplotstableread{{\n{' '.join(cols)}\n{body}\n}}\\{k}")
data = "\n".join(lines) or "% (no numbers)"

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
