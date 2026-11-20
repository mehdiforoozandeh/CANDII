"""Rebuild index.html from deck_src.html: shipment 1 of the peeling-the-onion run (2026-11-15).

Shipment 1 = Methods (data, evaluation) and the model-free Results: the variance layers on real data,
the training-free baselines, the AlphaGenome 2 audit, the region-selection interim finding, and the
screening-against-full look on the baselines. One closing preview slide shows the first screening
score of one model run (one seed), with no conclusion.

Paper arc (prezit references/paper-arc.md). Every number on a slide is read here from a vault file
and embedded as DATA; the page types none. The build writes the deck-sources manifest
(vault-relative path -> sha256 of every file it read) so that `crux deck --verify` can prove the
deck is current. `results/` is gitignored: the build reads this worktree's copies.

    /Users/mforooz/miniforge3/envs/candii/bin/python build_deck.py [VAULT]
Stdlib only. A regex that no longer matches stops the build and names the file.
"""
import csv, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
read = []


def path(rel):
    p = VAULT / rel
    if p not in read:
        read.append(p)
    return p


def text(rel):
    return path(rel).read_bytes().decode("utf-8")


def grab(rel, pattern, n=1, src=None):
    s = text(rel) if src is None else src
    m = re.search(pattern, s, re.S)
    if not m:
        sys.exit(f"build_deck: pattern not found in {rel}: {pattern}")
    return [m.group(i + 1) for i in range(n)] if n > 1 else m.group(1)


def num(s):
    return float(str(s).replace(",", ""))


def inum(s):
    return int(str(s).replace(",", ""))


def tsv(rel):
    return list(csv.DictReader([l for l in text(rel).splitlines() if not l.startswith("#")], delimiter="\t"))


def jload(rel):
    return json.loads(text(rel))


SRC = {}   # slide key -> vault paths read for it (shown in the speaker notes)


def cite(key, *rels):
    SRC.setdefault(key, [])
    for r in rels:
        if r not in SRC[key]:
            SRC[key].append(r)


WHERE = "INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md"

# --- Methods, data ---------------------------------------------------------------------------
DS = "results/t183/DATASET_EIC_V1.md"
ds = {k: [int(x) for x in grab(DS, r"\| %s \| ([0-9]+) \| ([0-9]+) \|" % k, 2)]
      for k in ("training", "validation", "control", "blind")}
b = grab(DS, r"([0-9,]+) rows \(train ([0-9,]+), val ([0-9,]+), blind ([0-9,]+);.*?([0-9,]+) pairs; .*?"
             r"([0-9,]+) never-trained designed validation pairs", 6)
cost = grab(DS, r"median ([0-9.]+) CPU-h per experiment; total ([0-9,]+) CPU-h used and ([0-9,]+) CPU-h allocated", 3)
cpm = grab(DS, r"CPU-h per million reads ([0-9.]+) \(chip-seq-pipeline2\), ([0-9.]+) \(dnase-seq-pipeline\), "
               r"([0-9.]+) \(atac-seq-pipeline\)", 3)
n_exp = grab(DS, r"([0-9]+) of the ([0-9]+) experiments of priorities 1 to 3 have an `ALIGN_DONE`", 2)
no_acc = grab(DS, r"Validation targets with no accessibility-only item .*?: ([0-9]+) of ([0-9]+)", 2)
dnase_val = grab(DS, r"Not scorable: DNase-seq \(([0-9]+) validation track with halves of ([0-9]+) needed", 2)
DT = "results/t309/dataset_table.tsv"
dt = [r for r in tsv(DT) if r["assay_class"]]
classes = {}
for r in dt:
    c = classes.setdefault(r["assay_class"], dict(assays=0, train=0, train_plan=0, val=0, val_plan=0, imp=0, cor=0))
    c["assays"] += 1
    c["train"] += int(r["train_eic_v1"]); c["train_plan"] += int(r["train_ledger"])
    c["val"] += int(r["val_eic_v1"]); c["val_plan"] += int(r["val_ledger"])
    c["imp"] += r["imputation_scorable"].startswith("yes"); c["cor"] += r["correction_scorable"] == "yes"
tot = next(r for r in tsv(DT) if r["assay"] == "all non-control tracks")
if int(tot["train_eic_v1"]) != ds["training"][0] or int(tot["val_eic_v1"]) != ds["validation"][0]:
    sys.exit("build_deck: dataset_table.tsv and DATASET_EIC_V1.md disagree on tracks present")
pipes = sorted({r["pipeline"] for r in dt})
data_d = dict(ds=ds, rows=inum(b[0]), rows_split=[inum(x) for x in b[1:4]], pairs=inum(b[4]), never=inum(b[5]),
              cpu_med=num(cost[0]), cpu_used=inum(cost[1]), cpu_alloc=inum(cost[2]), cpm=[num(x) for x in cpm],
              n_exp=[int(x) for x in n_exp], no_acc=[int(x) for x in no_acc], dnase_val=[int(x) for x in dnase_val],
              classes=classes, n_assays=len(dt), pipes=pipes)
cite("data", DS, DT)

# --- Methods, evaluation -----------------------------------------------------------------------
EVS = "results/t185/SUBMITTED.md"
ev = grab(EVS, r"Content: ([0-9,]+) regions of ([0-9,]+) bins, ([0-9]+) fit regions; ([0-9]+) imputation items "
               r"\(([0-9]+) leave-one-assay-out, ([0-9]+) accessibility-only\).*?([0-9,]+) correction pairs "
               r"\(0 dropped\); ([0-9,]+) replicate pairs", 8)
evalset = dict(regions=inum(ev[0]), region_bins=inum(ev[1]), fit=inum(ev[2]), items=inum(ev[3]),
               loao=inum(ev[4]), acc=inum(ev[5]), corr=inum(ev[6]), rep=inum(ev[7]))
imp_assays = grab(EVS, r"Imputation-scorable by the builder's rule \(([0-9]+)\)")
cor_assays = grab(EVS, r"Correction-scorable \(([0-9]+)\)")
evalset.update(imp_assays=int(imp_assays), cor_assays=int(cor_assays))
RG = "results/t283/REGIONS.md"
rg_bins, bp_unit, rg_bp = grab(RG, r"Region size: ([0-9,]+) bins of ([0-9]+) bp \(([0-9,]+) bp\)", 3)
if inum(rg_bins) != evalset["region_bins"]:
    sys.exit("build_deck: REGIONS.md and the evaluation set disagree")
regs = jload("results/t283/regions.json")
CS = "presentations/onion-status-2026-11-14/chrom_sizes.json"
cs_sha = grab(RG, r"`chrom_sizes_sha256` \(sha256 of `[^`]+`\): `([0-9a-f]{64})`")
if hashlib.sha256(path(CS).read_bytes()).hexdigest() != cs_sha:
    sys.exit("build_deck: chrom_sizes.json is not the file whose sha256 REGIONS.md records")
SC = "results/t311/SUBMITTED.md"
p = grab(SC, r"`pair_regions` = 0, ([0-9]+), ([0-9]+), \.\.\., ([0-9]+) \(([0-9]+) positions of the region order\)", 4)
p_n = int(p[3])
positions = [i * (evalset["regions"] // p_n) for i in range(p_n)]
if positions[1] != int(p[0]) or positions[-1] != int(p[2]):
    sys.exit("build_deck: the screening pair-region positions are not every (256/16)-th region")
SRCS = "results/t309/sources.md"
wts = [num(x) for x in grab(SRCS, r"\| composite weights \| ([0-9.]+), ([0-9.]+), ([0-9.]+) \|", 3)]
never_n = grab(SRCS, r"\| withheld combinations; never-trained pairs per validation track \| ([0-9]+);")
data_e = dict(ev=evalset, bp=int(bp_unit), region_bp=inum(rg_bp), p16=p_n, positions=positions, weights=wts,
              withheld=int(never_n), list=regs["regions"], fit_list=regs["fit_regions"],
              chrom=jload(CS)["chrom_sizes"])
cite("eval", EVS, RG, "results/t283/regions.json", CS, SC, SRCS)
cite("metrics", SRCS, EVS)

# --- Result 1: variance layers -----------------------------------------------------------------
LA = "results/t305/layers_by_assay.tsv"
LS = "results/t305/layers_by_stratum.tsv"
LN = "results/t305/LAYERS_REAL.md"
ASSAYS = ["DNase-seq", "H3K27ac", "H3K4me3", "H3K9ac", "H3K4me1", "H3K27me3", "H3K9me3", "H3K36me3", "chipseq-control"]
LAYER_OF = {"pseudoreplicate": "samp", "isogenic": "rep", "other_cell_type": "ct"}
by = {(r["assay_or_class"], r["rung"]): r for r in tsv(LA) if r["pooled_by_class"] == "0"}
srows = tsv(LS)
sby = {(r["assay_or_class"], r["stratum"], r["band"], r["rung"]): num(r["layer_increment"])
       for r in srows if r["pooled_by_class"] == "0" and r["layer_increment"] != ""}
STRATA = ["all", "promoters", "enhancers", "gene_bodies", "heterochromatin"]
BANDS = ["fine", "peak", "broad", "domain"]


def ct_share(a, st, bd):
    v = [sby[(a, st, bd, r)] for r in LAYER_OF]
    return v[2] / sum(v)


layers = []
for a in ASSAYS:
    e = dict(assay=a)
    for rung, k in LAYER_OF.items():
        r = by[(a, rung)]
        e[k] = dict(v=num(r["layer_increment"]), lo=num(r["ci_low"]), hi=num(r["ci_high"]),
                    pairs=int(r["n_pairs"]), share=num(r["share_of_layers"]))
    if a != "chipseq-control":
        e["strata"] = [ct_share(a, s, "all") for s in STRATA]
        e["bands"] = [ct_share(a, "all", bd) for bd in BANDS]
    layers.append(e)
ln = text(LN)
for must in ("The common depth is now the depth of a pseudoreplicate half", "It is not a pure cell-type layer"):
    if must not in ln:
        sys.exit(f"build_deck: LAYERS_REAL.md no longer says: {must}")
lay_pairs = [inum(x) for x in grab(LN, r"The pair list has ([0-9,]+) pairs: ([0-9,]+) other_cell_type, "
                                        r"([0-9,]+) pseudoreplicate, ([0-9,]+) isogenic", 4, ln)]
tele = [inum(x) for x in grab(LN, r"Of ([0-9,]+) groups \(assay x stratum x band\): ([0-9,]+) hold exactly", 2, ln)]
data_l = dict(layers=layers, pairs=lay_pairs, strata=STRATA, bands=BANDS, tele=tele)
cite("layers", LA, LN)
cite("strata", LS, LN)

# --- Result 2: baselines ---------------------------------------------------------------------------
BD = "results/t307/"
BM = BD + "BASELINES.md"
bm = text(BM)
for must in ("All checks passed on both tiers, with exact equality", "The floor is therefore not a lower bound"):
    if must not in bm:
        sys.exit(f"build_deck: BASELINES.md no longer says: {must}")
BL = ["marginal_nb", "position_average", "depth_only_scaling", "pseudoreplicate_floor"]
H = {}
for r in tsv(BD + "baselines_headline.tsv"):
    if r["full_256"] == "":
        continue
    H.setdefault(r["key"], {})[r["baseline"]] = dict(
        v=num(r["full_256"]), lo=num(r["ci95_lo"]) if r["ci95_lo"] else None,
        hi=num(r["ci95_hi"]) if r["ci95_hi"] else None, s=num(r["screening_16"]),
        slo=num(r["screening_ci95_lo"]) if r["screening_ci95_lo"] else None,
        shi=num(r["screening_ci95_hi"]) if r["screening_ci95_hi"] else None,
        out=r["screening_outside_full_ci95"])
need = ["specific_variance_recovered_above_noise.counts_log1p", "count_crps", "count_crps_oracle_scaled",
        "count_crps_scale_error", "count_crps_skill_vs_position_average", "correction_crps_model",
        "correction_crps_model_oracle_scaled", "correction_crps_model_scale_error",
        "correction_gap_closed_vs_depth_only_never_trained", "depth_law_abs_log2_error_never_trained",
        "pit_deviation_pooled", "pit_deviation_worst_stratum", "predicted_over_replicate_variance",
        "selection.composite", "r_spec.counts_log1p"]
for k in need:
    if k not in H:
        sys.exit(f"build_deck: baselines_headline.tsv has no row {k}")
HEAD = {k: H[k] for k in need}
nf = tsv(BD + "noise_floor_by_assay.tsv")
noise = [dict(assay=r["assay"], N=num(r["noise_floor_N_mean"]), V=num(r["specific_variance_V_sp_mean"]),
              share=num(r["N_over_V_sp"])) for r in nf if r["imputation_scorable"] == "1"]
kn = tsv(BD + "correction_headroom.tsv")
knobs = [dict(knob=r["knob"], set=r["pair_set"], n=int(r["n_pairs"]), cells=int(r["n_cells"]),
              d=num(r["crps_depth_only"]), f=num(r["crps_floor"]), g=num(r["gap_depth_only_minus_floor"]),
              glo=num(r["gap_depth_only_minus_floor_lo"]) if r["gap_depth_only_minus_floor_lo"] else None,
              ghi=num(r["gap_depth_only_minus_floor_hi"]) if r["gap_depth_only_minus_floor_hi"] else None,
              pa=num(r["crps_position_average"])) for r in kn]
cal = tsv(BD + "calibration_baselines.tsv")
calib = [dict(b=r["baseline"], assay=r["assay"], pooled=num(r["pit_deviation_pooled"]),
              vr=num(r["variance_ratio_pseudoreplicate"])) for r in cal
         if r["imputation_scorable"] == "1" and not r["assay"].startswith("__")]
pit = tsv(BD + "pit_curves.tsv")
scor = {r["assay"] for r in cal if r["imputation_scorable"] == "1" and not r["assay"].startswith("__")}
us = [k for k in pit[0] if k.startswith("u=")]
pitc = [dict(b=r["baseline"], assay=r["assay"], y=[num(r[k]) for k in us]) for r in pit if r["assay"] in scor]
val = [dict(tier=r["tier"], check=r["check"], ok=r["pass"] == "pass") for r in tsv(BD + "validity_checks.tsv")]
repro = grab(BM, r"A second method reproduced (three) scorer macros")
per = {}
for bl in BL:
    sj = jload(BD + f"src/baselines/{bl}/seed0/scores.json")

    def find(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                x = find(v, key)
                if x is not None:
                    return x
        return None
    per[bl] = dict(crps=find(sj, "correction_crps_model")["per_assay"],
                   gap=find(sj, "correction_gap_closed_vs_depth_only_never_trained")["per_assay"],
                   macro_gap=find(sj, "correction_gap_closed_vs_depth_only_never_trained")["macro"])
if abs(per["position_average"]["macro_gap"] - HEAD["correction_gap_closed_vs_depth_only_never_trained"]["position_average"]["v"]) > 1e-4:
    sys.exit("build_deck: scores.json and baselines_headline.tsv disagree on the gap closed")
times = tsv(BD + "scoring_time_memory.tsv")
data_b = dict(head=HEAD, noise=noise, knobs=knobs, calib=calib, pit=pitc, us=[num(k[2:]) for k in us], valid=val,
              repro=repro, per=per, n_time=len(times))
cite("baselines", BM, BD + "validity_checks.tsv", BD + "baselines_headline.tsv")
cite("reach", BM, BD + "baselines_headline.tsv", BD + "noise_floor_by_assay.tsv")
cite("headroom", BM, BD + "correction_headroom.tsv")
cite("calib", BM, BD + "calibration_baselines.tsv", BD + "pit_curves.tsv")
cite("room", BM, BD + "baselines_headline.tsv", BD + "correction_headroom.tsv")
cite("open", BM, BD + "baselines_headline.tsv", BD + "correction_headroom.tsv",
     *[BD + f"src/baselines/{bl}/seed0/scores.json" for bl in BL])
cite("screenfull", BM, BD + "baselines_headline.tsv")

# --- Result 3: AlphaGenome 2 ------------------------------------------------------------------------
AL = "results/t273/AG2_LEAKAGE.md"
STAT = ["observed_matched", "observed_other_read_type", "sibling_condition", "unseen", "no_channel", "no_condition"]
st = {s: [int(x) for x in grab(AL, r"\| `%s` \| ([0-9]+) \| ([0-9]+) \| ([0-9]+) \|" % s, 3)] for s in STAT}
aud = [int(x) for x in grab(AL, r"The audit has ([0-9]+) rows: ([0-9]+) training, ([0-9]+) validation and ([0-9]+) blind", 4)]
for i in range(3):
    if sum(v[i] for v in st.values()) != aud[i + 1]:
        sys.exit("build_deck: the statuses of AG2_LEAKAGE.md do not add up")
PK = "results/t308/SUBMITTED.md"
pk_u = [int(x) for x in grab(PK, r"\| unseen \| ([0-9]+) \| ([0-9]+) \| `unseen_by_both` \|", 2)]
pk_t = [int(x) for x in grab(PK, r"\| observed_matched \| ([0-9]+) \| ([0-9]+) \| `ag2_trained` \|", 2)]
pk_s = int(grab(PK, r"\| sibling_condition \| ([0-9]+) \| [0-9]+ \| set aside \|"))
dump = [int(x) for x in grab(PK, r"The dump holds only ([0-9]+) of the ([0-9]+) headline targets", 2)]
fl = [int(x) for x in grab(PK, r"the column `frag_len` is empty for ([0-9]+) of ([0-9]+) validation base tracks", 2)]
if "FAILED" not in text(PK):
    sys.exit("build_deck: t308 SUBMITTED.md no longer records the failed oracle-baseline job; re-read it")
board_md = VAULT / "results/t308/AG2_VS_BASELINES.md"
board_tsv = VAULT / "results/t308/ag2_board_val.tsv"
board = board_md.exists() or board_tsv.exists()
AF = "results/ag2/AG2_FACTS.md"
af = text(AF)
seq = grab(AF, r"Input: a ([0-9]+)-Mb DNA sequence", src=af)
chip_bp = grab(AF, r"Histone ChIP-seq and TF ChIP-seq: ([0-9]+) bp\. Unit \"Fold-change over control\"", src=af)
acc_bp = grab(AF, r"DNase-seq, ATAC-seq, scATAC-seq: ([0-9]+) bp\.", src=af)
data_a = dict(st=st, aud=aud, unseen=pk_u, trained=pk_t, sib=pk_s, dump=dump, fl=fl, board=board,
              seq=int(seq), chip_bp=int(chip_bp), acc_bp=int(acc_bp))
cite("ag2", AL, PK, AF)

# --- Result 4: region selection ---------------------------------------------------------------------
ST = "results/t292/STRATA_REAL.md"
ess = {}
for name in ("peakfreq_w", "ccre_like_w", "variance_w", "kcenter"):
    ess[name] = num(grab(ST, r"\| %s \| [0-9]+ \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \| ([0-9.]+) \|" % name))
for name in ("ccredens_e50_w", "ccredens_e25_w", "variance_e50_w"):
    ess[name] = num(grab(ST, r"\| %s \| [a-z_]+ \| [0-9.]+ \| ([0-9.]+) \|" % name))
empty = [inum(x) for x in grab(ST, r"([0-9]+) of ([0-9,]+) tiles hold no training peak", 2,
                               text("INSTRUCTIONS/peeling_the_onion/onion region budget and selection.md"))]
tile_bins = grab(ST, r"(At [0-9.]+ kb tiles)")  # presence check only
rho = num(grab(ST, r"\(\|rho\| at most ([0-9.]+)\)"))
rcls = [int(x) for x in grab(ST, r"pf_0-0\.05 has ([0-9]+) region, pf_0\.05-0\.20 has ([0-9]+), pf_0\.20-0\.50 has ([0-9]+), "
                                  r"and pf_gt0\.50 has ([0-9]+)", 4)]
RB = "INSTRUCTIONS/peeling_the_onion/onion region budget and selection.md"
win = grab(RB, r"A training window is ([0-9,]+) bins \(([0-9.]+) kb\), and the sampling tiles are ([0-9,]+) bins \(([0-9.]+) kb\)", 4)
data_r = dict(ess=ess, empty=empty, rho=rho, rcls=rcls, win=[inum(win[0]), num(win[1]), inum(win[2]), num(win[3])])
cite("select", ST, RB)

# --- Preview: first screening score ------------------------------------------------------------------
FS = "results/metrics_screening/first_scores.md"
fs = text(FS)
if "**1 seed**" not in fs or "It is not a result." not in fs:
    sys.exit("build_deck: first_scores.md lost its one-seed / not-a-result labels")


def fsrow(label):
    m = re.search(r"\| " + re.escape(label) + r" " + r"\|\s*([^|]*?)\s*" * 5 + r"\|", fs)
    if not m:
        sys.exit(f"build_deck: first_scores.md has no row {label}")
    return [num(x) if x.strip() else None for x in m.groups()]


prev = dict(
    composite=fsrow("composite (select_rule)"),
    imp=fsrow("imputation: specific variance recovered above noise (counts_log1p)"),
    crps=fsrow("imputation: count CRPS"),
    crps_os=fsrow("imputation: count CRPS, oracle_scaled"),
    crps_se=fsrow("imputation: count CRPS, scale_error"),
    corr=fsrow("correction: CRPS"),
    pit=fsrow("uncertainty: PIT deviation, pooled"))
cite("preview", FS)

# --- schedule --------------------------------------------------------------------------------------
wh = text(WHERE)
sch = {}
for name in ("Ship 1", "Ship 2", "Iteration 2"):
    sch[name] = grab(WHERE, r"\| %s \| [^|]+ \| ([^|]+) \|" % name, src=wh).strip()
s2 = re.match(r"(2026-[0-9-]+) ([0-9:]+) \(hard limit (2026-[0-9-]+) ([0-9:]+)\)", sch["Ship 2"])
s1 = re.match(r"(2026-[0-9-]+) ([0-9:]+) \(deck cut ([0-9:]+)\)", sch["Ship 1"])
if not (s1 and s2):
    sys.exit("build_deck: the schedule table of WHERE_WE_ARE.md changed shape")
schedule = dict(ship1=[s1.group(1), s1.group(2)], ship2=[s2.group(1), s2.group(2)], hard2=[s2.group(3), s2.group(4)],
                it2=sch["Iteration 2"])
cite("title", WHERE)
cite("next", WHERE, FS)

data = dict(d=data_d, e=data_e, l=data_l, b=data_b, a=data_a, r=data_r, prev=prev, sch=schedule, src=SRC)

read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {q.relative_to(VAULT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
if src.count("__SOURCES__") != 1 or src.count("__DATA__") != 1:
    sys.exit("build_deck: deck_src.html must hold exactly one __SOURCES__ and one __DATA__")
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"build_deck: wrote {HERE / 'index.html'} from {len(sources)} files; AlphaGenome 2 board present: {board}")
