"""Rebuild index.html from deck_src.html: shipment 2 of the peeling-the-onion run (2026-11-15).

Shipment 2 = the model Results on screening scores (the ladder of six architectures, the rung-1 arms,
the rung-4 arms with the hard offset, training-region selection), the AlphaGenome 2 board, what the
eta view changed in the variance layers, and a first Discussion.

Paper arc (prezit references/paper-arc.md). Every number on a slide is read here from a vault file
and embedded as DATA; the page types none. The build writes the deck-sources manifest
(vault-relative path -> sha256 of every file it read) so that `crux deck --verify` can prove the
deck is current. `results/` is gitignored: the build reads this worktree's copies.

    /Users/mforooz/miniforge3/bin/python build_deck.py [VAULT]
Stdlib only. A pattern or row that is missing stops the build and names the file.
"""
import csv, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
read = []
SRC = {}


def path(rel):
    p = VAULT / rel
    if p not in read:
        read.append(p)
    return p


def text(rel):
    return path(rel).read_bytes().decode("utf-8")


def grab(rel, pattern, n=1):
    m = re.search(pattern, text(rel), re.S)
    if not m:
        sys.exit(f"build_deck: pattern not found in {rel}: {pattern}")
    return [m.group(i + 1) for i in range(n)] if n > 1 else m.group(1)


def tsv(rel):
    return list(csv.DictReader([l for l in text(rel).splitlines() if not l.startswith("#")], delimiter="\t"))


def num(s):
    s = str(s).strip()
    return None if s in ("", "None", "nan") else float(s)


def cite(key, *rels):
    SRC.setdefault(key, [])
    for r in rels:
        if r not in SRC[key]:
            SRC[key].append(r)


MS = "results/metrics_screening/"
R = MS + "sh2_refresh/"
WHERE = "INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md"
METRICS = ["specific_variance_recovered", "gap_closed_oracle", "gap_closed_oracle_no_nodedup_side",
           "gap_closed_halfread_superseded", "composite_v2"]

# --- run state of the tables ----------------------------------------------------------------------
built = text(R + "run_time.txt").strip()
n_scored = len([l for l in text(R + "scored_list.txt").splitlines() if l.strip()])
checks = {r["check"]: r for r in tsv(R + "checks_tb2.tsv")}
if checks["composite versions present"]["max_abs_difference"] != "2":
    sys.exit("build_deck: checks_tb2.tsv shows a composite version other than 2")
if int(checks["scored runs (SCORED files)"]["n_compared"]) != n_scored:
    sys.exit("build_deck: scored_list.txt and checks_tb2.tsv disagree on the number of scored runs")

# --- per-seed terms, including the uncertainty term (select_rule table) ----------------------------
unc = {}
pending = []
for rel in (R + "table_runs.tsv", R + "table_region.tsv"):
    for r in tsv(rel):
        if r["state"] != "done":
            pending.append([r["arm"], int(r["seed"])])
            continue
        unc[(r["arm"], int(r["seed"]))] = dict(u=num(r["uncertainty_raw"]), i=num(r["imputation_raw"]),
                                               c=num(r["correction_raw"]), comp=num(r["composite"]))


def seeds_of(arm):
    return sorted(s for (a, s) in unc if a == arm)


def uterm(arm):
    v = [unc[(arm, s)]["u"] for s in seeds_of(arm)]
    return dict(seeds=seeds_of(arm), v=v, mean=sum(v) / len(v), spread=max(v) - min(v))


# --- ladder ----------------------------------------------------------------------------------------
lad = {}
for r in tsv(R + "ladder_tb2.tsv"):
    lad.setdefault(r["row"], {})[r["metric"]] = dict(
        seeds=[num(r[f"seed{k}"]) for k in range(3) if f"seed{k}" in r and num(r[f"seed{k}"]) is not None],
        mean=num(r["mean"]), spread=num(r["spread"]))
RUNGS = [f"rung{k}_base" for k in range(1, 7)]
for k in RUNGS + ["baseline position_average", "baseline marginal_nb", "baseline depth_only_scaling",
                  "baseline pseudoreplicate_floor"]:
    if k not in lad:
        sys.exit(f"build_deck: ladder_tb2.tsv has no row {k}")
ladder = dict(rows={k: lad[k] for k in lad}, unc={k: uterm(k) for k in RUNGS})
# check: the uncertainty seed values reproduce the composite with the weights
WS = "results/t309/sources.md"
wts = [num(x) for x in grab(WS, r"\| composite weights \| ([0-9.]+), ([0-9.]+), ([0-9.]+) \|", 3)]
clip = lambda x: min(1.0, max(0.0, x))
for (a, s), d in unc.items():
    comp = wts[0] * clip(d["i"]) + wts[1] * clip(d["c"]) + wts[2] * clip(d["u"])
    if abs(comp - d["comp"]) > 2e-6:
        sys.exit(f"build_deck: composite of {a} seed {s} does not follow from its terms")
cite("ladder", MS + "scores_ladder.md", R + "ladder_tb2.tsv", R + "table_runs.tsv", WS)

# compute per run
rj = tsv(R + "runjson_rungs.tsv")
compute = {}
for r in rj:
    c = compute.setdefault(r["run"], dict(params=int(r["param_count"]), flops=[], device=r["device"]))
    c["flops"].append(num(r["training_flops"]))
cite("compute", R + "runjson_rungs.tsv", MS + "scores_ladder.md")

# --- arm tables (paired differences) ---------------------------------------------------------------


def arms(rel):
    out = {}
    for r in tsv(rel):
        if r["metric"] not in METRICS:
            continue
        out.setdefault(r["arm"], dict(ref=r["reference"], n=int(r["arm_n"])))[r["metric"]] = dict(
            ref_mean=num(r["ref_mean"]), ref_spread=num(r["ref_spread"]), arm_mean=num(r["arm_mean"]),
            arm_spread=num(r["arm_spread"]), diffs=[num(r[f"diff_seed{k}"]) for k in range(3)],
            mean=num(r["mean_diff"]), larger=num(r["larger_seed_spread"]), exceeds=r["exceeds_seed_spread"])
    return out


def unc_pair(arm, ref):
    s = [k for k in seeds_of(arm) if k in seeds_of(ref)]
    d = [unc[(arm, k)]["u"] - unc[(ref, k)]["u"] for k in s]
    a, b = uterm(arm), uterm(ref)
    return dict(seeds=s, diffs=d, mean=sum(d) / len(d), larger=max(a["spread"], b["spread"]) if len(a["v"]) > 1 else None,
                arm=a, ref=b)


r1 = arms(R + "rung1_arms_tb2.tsv")
r4 = arms(R + "rung4_arms_tb2.tsv")
sel = arms(R + "selection_arms_tb2.tsv")
if "rung4_hard_offset" not in r4:
    sys.exit("build_deck: rung4_arms_tb2.tsv has no hard-offset arm")
for d in (r1, r4, sel):
    for a, v in d.items():
        v["unc"] = unc_pair(a, v["ref"])
cite("r1", MS + "scores_rung1_arms.md", R + "rung1_arms_tb2.tsv", R + "table_runs.tsv")
cite("r4", MS + "scores_rung4_arms.md", R + "rung4_arms_tb2.tsv", R + "table_runs.tsv", R + "table_region.tsv")
cite("ho", MS + "scores_rung4_arms.md", R + "rung4_arms_tb2.tsv", R + "table_runs.tsv", WS)
cite("sel", MS + "scores_selection_arms.md", R + "selection_arms_tb2.tsv", R + "table_region.tsv", WS)

# --- oracle floor scale --------------------------------------------------------------------------
OF = "results/oracle_floor/ORACLE_FLOOR.md"
nn = [int(x) for x in grab(OF, r"\(a\) All headline pairs: never trained, not a subset pair, ([0-9]+) pairs from ([0-9]+) cells\. "
                               r"\(b\) Headline pairs where no side keeps duplicates .*?: ([0-9]+) pairs .*?, ([0-9]+) cells, ([0-9]+) of the ([0-9]+) assays", 6)]
cite("scale", OF, MS + "scores_ladder.md", R + "ladder_tb2.tsv")

# --- AlphaGenome 2 board -------------------------------------------------------------------------
AG = "results/t308/AG2_VS_BASELINES.md"
board = {}
for m in ("ag2_matched", "position_average", "marginal_nb", "closest5_oracle"):
    board[m] = num(grab(AG, r"\n\| %s \| ([0-9.]+) \| [0-9.]+ \| [0-9.]+ \|" % m))
nf = [num(x) for x in grab(AG, r"\| pseudoreplicate_half \(noise floor\) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \|", 3)]
fair = [int(x) for x in grab(AG, r"## Board, fair set \(`unseen_by_both`, ([0-9]+) tracks, ([0-9]+) cells", 2)]
con = [num(x) for x in grab(AG, r"\| position_average \| (-?[0-9.]+) \[(-?[0-9.]+), (-?[0-9.]+)\], ([0-9]+) of ([0-9]+) ahead \|", 5)]
cite("ag2", AG)

# --- eta view of the layers --------------------------------------------------------------------
DI = "results/eta_space/depth_invariance.tsv"
eta = {}
for r in tsv(DI):
    if r["assay"] in ("chipseq-control",):
        continue
    e = eta.setdefault(r["assay"], [])
    e.append(dict(f=num(r["depth_factor"]), log1p=num(r["log1p_share_celltype"]), eta=num(r["eta_celltype_cv2"]),
                  share=num(r["eta_celltype_share_of_V"])))
LE = "results/eta_space/LAYERS_ETA.md"
inv = [num(x) for x in grab(LE, r"changed by at most ([0-9]+)% .*?the log1p cell-type share of the 10 target assays fell by ([0-9]+)% to ([0-9]+)%", 3)]
cite("eta", DI, LE, "results/eta_space/THEORY.md")
text("results/eta_space/THEORY.md")

# --- schedule ------------------------------------------------------------------------------------
sch = {}
for name in ("Ship 2", "Iteration 2"):
    sch[name] = grab(WHERE, r"\| %s \| ([^|]+) \| ([^|]+) \|" % name, 2)
cite("next", WHERE, R + "table_runs.tsv", R + "table_region.tsv", AG)
cite("title", WHERE, R + "run_time.txt", R + "scored_list.txt", R + "checks_tb2.tsv")
cite("recap", WS, MS + "scores_ladder.md")
cite("disc", MS + "scores_ladder.md", MS + "scores_rung4_arms.md", MS + "scores_selection_arms.md", LE)

data = dict(built=built, n_scored=n_scored, w=wts, ladder=ladder, compute=compute, r1=r1, r4=r4, sel=sel,
            nn=nn, ag=dict(board=board, nf=nf, fair=fair, con=con), eta=eta, inv=inv, sch=sch,
            pending=pending, src=SRC)

read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {q.relative_to(VAULT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
if src.count("__SOURCES__") != 1 or src.count("__DATA__") != 1:
    sys.exit("build_deck: deck_src.html must hold exactly one __SOURCES__ and one __DATA__")
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"build_deck: wrote {HERE / 'index.html'} from {len(sources)} files; {n_scored} scored runs; pending {len(pending)}")
