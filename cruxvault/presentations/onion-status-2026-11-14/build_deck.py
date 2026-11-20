"""Rebuild index.html from deck_src.html: the status deck of the peeling-the-onion run, 2026-11-14.

PI request 2026-11-14: one deck of the status so far: the story, what is done, what is left; what
the 256 regions and the 16 regions are, why there are two sets, why not more regions, and what the
iterative-deployment iterations are.

Paper arc (prezit references/paper-arc.md), used as a status report. Every number on a slide is
read here from a vault file and embedded as DATA; the page types none. The build writes the
deck-sources manifest (vault-relative path -> sha256 of every file it read) so that
`crux deck --verify` can prove the deck is current. No anchor question: study slug
onion-status-2026-11-14.

Inputs (vault-relative). `results/` is gitignored: the build reads this worktree's copies.
  results/t183/DATASET_EIC_V1.md                    the frozen dataset (tracks present / planned)
  results/t185/SUBMITTED.md                         the evaluation set on that dataset
  results/t283/REGIONS.md, results/t283/regions.json  the frozen evaluation regions
  presentations/onion-status-2026-11-14/chrom_sizes.json  copy of Nibi ONION/genome/chrom_sizes.json;
                                                    its sha256 must equal chrom_sizes_sha256 of REGIONS.md
  results/t305/layers_by_assay.tsv, layers_by_stratum.tsv, LAYERS_REAL.md   real-data variance layers
  results/t273/AG2_LEAKAGE.md, results/t308/SUBMITTED.md, results/t284/calls_summary.tsv,
  results/t276/MAX_REPRO.md                         AlphaGenome 2 audit, pack, fetch, scorer gate
  results/t213/SUBMITTED.md, results/t291/SUBMITTED.md   trained rung-1 runs and budget-curve runs
  results/predict_store_speed/FORWARD_SPEED.md      measured forward speed, windows per scoring
  results/scorer_memory/NOTES.md                    measured baseline scoring time on all 256 regions
  results/t311/SUBMITTED.md                         the screening evaluation set (16 pair regions)
  results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md   the check that chooses the pair-region count
  results/region_budget/REGION_RESEARCH.md          prior evidence on the region count
  results/trainings_iteration1/SUBMITTED.md, results/arms_d10_d15/SUBMITTED.md   iteration-1 trainings
  INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md   status page and schedule
  INSTRUCTIONS/peeling_the_onion/run_state/plan.md  plan snapshot (Amendments 8 to 10, Log)
  INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md     PI rulings 41 to 46, file limit

A regex that no longer matches stops the build and names the file.

    /Users/mforooz/miniforge3/envs/candii/bin/python build_deck.py [VAULT]
Stdlib only.
"""
import csv, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
SLUG = HERE.name
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
    return list(csv.DictReader(text(rel).splitlines(), delimiter="\t"))


SRC = {}   # slide key -> vault paths read for it (shown in the speaker notes)


def cite(key, *rels):
    SRC.setdefault(key, [])
    for r in rels:
        if r not in SRC[key]:
            SRC[key].append(r)


PLAN = "INSTRUCTIONS/peeling_the_onion/run_state/plan.md"
WHERE = "INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md"
NOTEPAD = "INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md"

# --- dataset and evaluation set ------------------------------------------------------------
DS = "results/t183/DATASET_EIC_V1.md"
ds = {k: [int(x) for x in grab(DS, r"\| %s \| ([0-9]+) \| ([0-9]+) \|" % k, 2)]
      for k in ("training", "validation", "control")}
EVS = "results/t185/SUBMITTED.md"
ev = grab(EVS, r"Content: ([0-9,]+) regions of ([0-9,]+) bins, ([0-9]+) fit regions; ([0-9]+) imputation items "
               r"\(([0-9]+) leave-one-assay-out, ([0-9]+) accessibility-only\).*?([0-9,]+) correction pairs "
               r"\(0 dropped\); ([0-9,]+) replicate pairs", 8)
evalset = dict(regions=inum(ev[0]), region_bins=inum(ev[1]), fit=inum(ev[2]), items=inum(ev[3]),
               loao=inum(ev[4]), acc=inum(ev[5]), corr=inum(ev[6]), rep=inum(ev[7]))
cc = grab(PLAN, r"so ([0-9]+) of the ([0-9]+) validation replicate pairs raise", 2)
rjobs = grab(PLAN, r"1,480 correction pairs and ([0-9]+) replicate pair jobs")
evalset.update(cross_cell=inum(cc[0]), val_rep=inum(cc[1]), rep_jobs=inum(rjobs))
cite("data", DS, EVS)
cite("scored", EVS, PLAN)

# --- the regions ---------------------------------------------------------------------------
RG = "results/t283/REGIONS.md"
rg_bins, rg_bp_unit, rg_bp = grab(RG, r"Region size: ([0-9,]+) bins of ([0-9]+) bp \(([0-9,]+) bp\)", 3)
n_cand = grab(RG, r"Candidate blocks: ([0-9,]+)\.")
n_eval, n_chrom = grab(RG, r"Eval regions: ([0-9,]+), on ([0-9]+) chromosomes", 2)
n_fit = grab(RG, r"Fit regions: ([0-9,]+), disjoint")
seed = grab(RG, r"Seed: ([0-9]+)\.")
cs_sha = grab(RG, r"`chrom_sizes_sha256` \(sha256 of `[^`]+`\): `([0-9a-f]{64})`")
regs = json.loads(text("results/t283/regions.json"))
CS = f"presentations/{SLUG}/chrom_sizes.json"
if hashlib.sha256(path(CS).read_bytes()).hexdigest() != cs_sha:
    sys.exit("build_deck: chrom_sizes.json is not the file whose sha256 REGIONS.md records")
chrom = json.loads(text(CS))["chrom_sizes"]
if (len(regs["regions"]) != inum(n_eval) or regs["region_bins"] != inum(rg_bins)
        or regs["n_candidates"] != inum(n_cand) or len(regs["fit_regions"]) != inum(n_fit)):
    sys.exit("build_deck: regions.json disagrees with REGIONS.md")
if len({c for c, _ in regs["regions"]}) != inum(n_chrom):
    sys.exit("build_deck: chromosome count of regions.json disagrees with REGIONS.md")
if inum(n_eval) != evalset["regions"] or inum(rg_bins) != evalset["region_bins"]:
    sys.exit("build_deck: REGIONS.md and the evaluation set disagree")
regions = dict(bins=inum(rg_bins), bp_unit=inum(rg_bp_unit), bp=inum(rg_bp), cand=inum(n_cand),
               n=inum(n_eval), chroms=inum(n_chrom), fit=inum(n_fit), seed=int(seed),
               list=regs["regions"], fit_list=regs["fit_regions"], chrom=chrom)
cite("region", RG, "results/t283/regions.json", CS)

# --- the screening set: 16 pair regions ----------------------------------------------------
SC = "results/t311/SUBMITTED.md"
p = grab(SC, r"`pair_regions` = 0, ([0-9]+), ([0-9]+), \.\.\., ([0-9]+) \(([0-9]+) positions of the region order\)", 4)
p_n, step = int(p[3]), int(p[0])
positions = [i * (regions["n"] // p_n) for i in range(p_n)]
if positions[1] != step or positions[2] != int(p[1]) or positions[-1] != int(p[2]):
    sys.exit("build_deck: the pair-region positions of t311 are not every (256/16)-th region")
d = grab(SC, r"The derived file has ([0-9]+) regions, ([0-9]+) imputation targets, ([0-9,]+) correction pairs "
             r"and ([0-9,]+) replicate pair ids, as the base has", 4)
if [inum(x) for x in d] != [evalset["regions"], evalset["items"], evalset["corr"], evalset["rep"]]:
    sys.exit("build_deck: the screening set's counts differ from the base evaluation set")
screen = dict(n=p_n, positions=positions)
cite("screen", SC, RG)

# --- the pre-registered check ---------------------------------------------------------------
PR = "results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md"
const = json.loads(grab(PR, r"Constants: `(\{.*?\})`"))
check_runs = int(grab(PR, r"of each of ([0-9]+) runs"))
n_cond = grab(PR, r"A size p passes when all (five) conditions hold")
check = dict(sizes=const["sizes"], n_boot=const["n_boot"], spread_max=const["spread_ratio_max"],
             per_assay=const["per_assay_share_min"], pct=const["interval_percentiles"], runs=check_runs,
             n_cond=n_cond)
r4 = grab(PLAN, r"SRa: the ([0-9]+) runs are `rung4_base` and `rung4_scrambled_covariates`, seeds 0 to ([0-9]+)", 2)
r4c = grab(PLAN, r"Compute: 6 x 2\.56 = ([0-9.]+) GPU-slice-h and 6 x 19 = ([0-9]+) CPU-h", 2)
check.update(r4_runs=int(r4[0]), r4_gpu=num(r4c[0]), r4_cpu=num(r4c[1]))
NP = text(NOTEPAD)
r41 = grab(NOTEPAD, r"\*\*Ruling 41\.\*\* The evaluation-region count may go to ([0-9,]+) on evidence\. "
                    r"A step to ([0-9,]+) needs the PI", 2, NP)
RR = "results/region_budget/REGION_RESEARCH.md"
prior = grab(RR, r"covering ([0-9.]+)% of the held-out bins reproduced the full checkpoint ordering in ([0-9]+) "
                 r"of ([0-9]+) draws.*?256 regions is ([0-9.]+)%", 4)
more = dict(r512=inum(r41[0]), r1024=inum(r41[1]), prior_pct=num(prior[0]), prior_k=int(prior[1]),
            prior_n=int(prior[2]), now_pct=num(prior[3]))
cite("more", NOTEPAD, PR, PLAN, RR)

# --- cost of the two tiers -------------------------------------------------------------------
FS = "results/predict_store_speed/FORWARD_SPEED.md"
f = grab(FS, r"Rung 1, GPU slice, ([0-9.]+) windows per second: ([0-9,]+) / [0-9.]+ = [0-9,]+ s = ([0-9.]+) h of one "
             r"slice at p = 16; ([0-9,]+) / [0-9.]+ = [0-9,]+ s = ([0-9.]+) h at 256 pair regions", 5)
w = grab(PLAN, r"A scoring with pair jobs on 16 regions: ([0-9,]+) \+ ([0-9,]+) x ([0-9]+) = ([0-9,]+) windows; "
               r"at 256 pair regions ([0-9,]+)", 5)
if inum(w[3]) != inum(f[1]) or inum(w[4]) != inum(f[3]):
    sys.exit("build_deck: window counts of the plan and FORWARD_SPEED.md differ")
met16 = grab(PLAN, r"Estimate, not measured: a model's metrics at 16 pair regions ([0-9.]+) CPU-h")
met256 = grab(PLAN, r"about 69,000 CPU-s = ([0-9]+) CPU-h")
st16 = grab(PLAN, r"([0-9.]+) GB per run, removed by its metrics job")
st256 = grab(PLAN, r"a store at 256 pair regions is ([0-9]+) GB")
SM = "results/scorer_memory/NOTES.md"
base_t = grab(SM, r"job 27256987: ([0-9]+) CPUs, [0-9]+M, [^;]*; COMPLETED in ([0-9:]+)", 2)
cost = dict(speed=num(f[0]), win16=inum(f[1]), win256=inum(f[3]), gpu16=num(f[2]), gpu256=num(f[4]),
            item_win=inum(w[0]), pairs=inum(w[1]), met16=num(met16), met256=num(met256),
            st16=num(st16), st256=num(st256), base_cpus=int(base_t[0]), base_wall=base_t[1])
cite("cost", FS, PLAN, SM)
cite("scored", FS, PLAN)

# --- variance layers on real data -----------------------------------------------------------
LA = "results/t305/layers_by_assay.tsv"
LS = "results/t305/layers_by_stratum.tsv"
LN = "results/t305/LAYERS_REAL.md"
ASSAYS = ["DNase-seq", "H3K27ac", "H3K4me3", "H3K9ac", "H3K4me1", "H3K27me3", "H3K9me3", "H3K36me3", "chipseq-control"]
LAYER_OF = {"pseudoreplicate": "samp", "isogenic": "rep", "other_cell_type": "ct"}
rows = [r for r in tsv(LA) if r["pooled_by_class"] == "0"]
by = {(r["assay_or_class"], r["rung"]): r for r in rows}
srows = [r for r in tsv(LS) if r["pooled_by_class"] == "0" and r["stratum"] == "all" and r["band"] == "broad"]
sby = {(r["assay_or_class"], r["rung"]): r for r in srows}
layers = []
for a in ASSAYS:
    e = dict(assay=a)
    for rung, k in LAYER_OF.items():
        r = by[(a, rung)]
        e[k] = dict(v=num(r["layer_increment"]), lo=num(r["ci_low"]), hi=num(r["ci_high"]),
                    pairs=int(r["n_pairs"]), share=num(r["share_of_layers"]))
        s = sby[(a, rung)]
        e[k + "_broad"] = num(s["layer_increment"])
    layers.append(e)
ln = text(LN)
if "The common depth is now the depth of a pseudoreplicate half" not in ln:
    sys.exit("build_deck: LAYERS_REAL.md no longer states the half-depth caveat")
lay_pairs = grab(LN, r"The pair list has ([0-9,]+) pairs: ([0-9,]+) other_cell_type, ([0-9,]+) pseudoreplicate, "
                     r"([0-9,]+) isogenic", 4, ln)
cite("layers", LA, LS, LN)

# --- AlphaGenome 2 ----------------------------------------------------------------------------
AL = "results/t273/AG2_LEAKAGE.md"
st = {}
for s in ("observed_matched", "observed_other_read_type", "sibling_condition", "unseen", "no_channel", "no_condition"):
    st[s] = [int(x) for x in grab(AL, r"\| `%s` \| ([0-9]+) \| ([0-9]+) \| ([0-9]+) \|" % s, 3)]
aud_rows = grab(AL, r"The audit has ([0-9]+) rows: ([0-9]+) training, ([0-9]+) validation and ([0-9]+) blind", 4)
PK = "results/t308/SUBMITTED.md"
pk_u = grab(PK, r"\| unseen \| ([0-9]+) \| ([0-9]+) \| `unseen_by_both` \|", 2)
pk_t = grab(PK, r"\| observed_matched \| ([0-9]+) \| ([0-9]+) \| `ag2_trained` \|", 2)
calls = {r["quantity"]: r["value"] for r in tsv("results/t284/calls_summary.tsv")}
GT = "results/t276/MAX_REPRO.md"
gate = grab(GT, r"\ngate: (PASS|FAIL)\n")
g_t = grab(GT, r"for the ([0-9]+) drawn targets")
g_c = grab(GT, r"All ([0-9]+) comparisons were within")
ag2 = dict(val={k: v[1] for k, v in st.items()}, rows=[int(x) for x in aud_rows],
           unseen=[int(x) for x in pk_u], trained=[int(x) for x in pk_t],
           calls=int(calls["total_calls"]), ok=int(calls["successes"]), gate=gate, gate_t=int(g_t), gate_c=int(g_c))
if sum(ag2["val"].values()) != ag2["rows"][2]:
    sys.exit("build_deck: the validation statuses of AG2_LEAKAGE.md do not add up")
cite("ag2", AL, PK, "results/t284/calls_summary.tsv", GT)

# --- trainings and the fast scorer ----------------------------------------------------------
n_r1 = int(grab("results/t213/SUBMITTED.md", r"lists the ([0-9]+) runs"))
n_bc = int(grab("results/t291/SUBMITTED.md", r"seeds 0, 1, 2\. ([0-9]+) runs\."))
wh = text(WHERE)
n_tr = int(grab(WHERE, r"all ([0-9]+) queued or running", src=wh))
n_tr_held = int(grab(WHERE, r"([0-9]+) held trainings, first two released", src=wh))
n_base = int(grab(WHERE, r"Baselines: ([0-9]+) scorings run", src=wh))
files = grab(WHERE, r"Now ([0-9]+)K of ([0-9]+)K", 2, wh)
state_time = grab(WHERE, r"## State on (2026-[0-9-]+ [0-9:]+)", src=wh)
if "The metric phase is not merged" not in wh:
    sys.exit("build_deck: WHERE_WE_ARE.md no longer says the metric phase is not merged")
TI = "results/trainings_iteration1/SUBMITTED.md"
n_ti = int(grab(TI, r"all ([0-9]+) jobs pending, none held"))
if n_ti != n_tr:
    sys.exit("build_deck: trainings count of the status page and the training note differ")
A1 = "results/arms_d10_d15/SUBMITTED.md"
n_a1 = int(grab(A1, r"All ([0-9]+) trainings are submitted and HELD"))
if n_a1 != n_tr_held:
    sys.exit("build_deck: the rung-1 arm trainings of the status page and the note differ")
lim = grab(NOTEPAD, r"has a limit of ([0-9,]+) files, shared by the whole lab\. It was full at about ([0-9:]+) UTC", 2, NP)
kit_files = grab(NOTEPAD, r"Each code snapshot \(\"kit\"\) that a builder pushes holds about ([0-9,]+) files", src=NP)
rm = grab(NOTEPAD, r"removal of ([0-9]+) kits \(([0-9,]+) files\)", 2, NP)
fair = grab(PLAN, r"fair-share is low \(CPU about ([0-9.]+), GPU about ([0-9.]+)\)", 2)
slices = grab(PLAN, r"\(([0-9,]+) queued jobs; ([0-9]+) to ([0-9]+) GPU slices", 3)
fwa = grab(PLAN, r"Compute: 30 x 2\.25 = ([0-9.]+) GPU-slice-h; ([0-9]+) jobs", 2)
status = dict(r1=n_r1, bc=n_bc, tr=n_tr, a1=n_a1, base=n_base, files_now=inum(files[0]) * 1000,
              files_max=inum(files[1]) * 1000, limit=inum(lim[0]), full_at=lim[1], kit_files=inum(kit_files),
              rm_kits=int(rm[0]), rm_files=inum(rm[1]), fair_cpu=num(fair[0]), fair_gpu=num(fair[1]),
              queue_max=inum(slices[0]), sl_lo=int(slices[1]), sl_hi=int(slices[2]), fwa_gpu=num(fwa[0]),
              fwa_jobs=int(fwa[1]), when=state_time)
if status["files_max"] != status["limit"]:
    sys.exit("build_deck: the file limit of the status page and the notepad differ")
cite("trained", "results/t213/SUBMITTED.md", "results/t291/SUBMITTED.md", FS, WHERE)
cite("now", WHERE, TI, A1, SC, LN)
cite("risks", WHERE, NOTEPAD, PLAN)

# --- schedule ----------------------------------------------------------------------------------
sch = {}
for name in ("Ship 1", "Ship 2", "Iteration 2"):
    sch[name] = grab(WHERE, r"\| %s \| [^|]+ \| ([^|]+) \|" % name, src=wh).strip()
s1 = re.match(r"(2026-[0-9-]+) ([0-9:]+) \(deck cut ([0-9:]+)\)", sch["Ship 1"])
s2 = re.match(r"(2026-[0-9-]+) ([0-9:]+) \(hard limit (2026-[0-9-]+) ([0-9:]+)\)", sch["Ship 2"])
i2 = re.match(r"(2026-[0-9-]+) ([0-9:]+)$", sch["Iteration 2"])
if not (s1 and s2 and i2):
    sys.exit("build_deck: the schedule table of WHERE_WE_ARE.md changed shape")
t0 = grab(PLAN, r"new: `T0` = (2026-[0-9-]+) ([0-9:]+) UTC", 2)
schedule = dict(t0=list(t0), ship1=[s1.group(1), s1.group(2)], cut1=s1.group(3),
                ship2=[s2.group(1), s2.group(2)], hard2=[s2.group(3), s2.group(4)], it2=[i2.group(1), i2.group(2)])
# the same dates must stand in the adopted plan (ruling 43)
for dd, hh in (schedule["ship1"], schedule["ship2"], schedule["hard2"]):
    if f"{dd} {hh} UTC" not in text(PLAN):
        sys.exit(f"build_deck: {dd} {hh} UTC is in the status page but not in the plan")
cite("schedule", WHERE, PLAN)
cite("principle", PLAN)
cite("close", WHERE, PLAN)
cite("story", PLAN, LN)

data = dict(ds=ds, ev=evalset, reg=regions, screen=screen, check=check, more=more, cost=cost,
            layers=layers, lay_pairs=[inum(x) for x in lay_pairs], ag2=ag2, st=status, sch=schedule, src=SRC)

read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {q.relative_to(VAULT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
if src.count("__SOURCES__") != 1 or src.count("__DATA__") != 1:
    sys.exit("build_deck: deck_src.html must hold exactly one __SOURCES__ and one __DATA__")
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"build_deck: wrote {HERE / 'index.html'} from {len(sources)} files")
