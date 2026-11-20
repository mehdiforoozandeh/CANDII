"""Rebuild index.html from deck_src.html: the close-out deck of the peeling-the-onion run (2026-11-18).

Paper arc (prezit references/paper-arc.md), one part per success criterion of the PI, then what did not
run, the ready queue and the open PI decisions. Every number on a slide is read here from a vault file
and embedded as DATA (prose numbers as preformatted strings in DATA.n, figures as matplotlib PNGs in
DATA.fig). The page types no number. The build writes the deck-sources manifest (vault-relative path ->
sha256 of every file it read) so that `crux deck --verify` can prove the deck is current.

`results/` is git-ignored: the build reads this worktree's copies, which are byte copies of the
foreman's worktree files of the same path.

    /Users/mforooz/miniforge3/bin/python build_deck.py [VAULT]
Needs matplotlib and Pillow (the `candii` env has neither; the base miniforge python has both).
A pattern or row that is missing stops the build and names the file.
"""
import base64, csv, hashlib, io, json, math, pathlib, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
read, SRC, N, FIG = [], {}, {}, {}


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
    return list(csv.DictReader(text(rel).splitlines(), delimiter="\t"))


def cite(key, *rels):
    SRC.setdefault(key, [])
    for r in rels:
        path(r)
        if r not in SRC[key]:
            SRC[key].append(r)


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def s3(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


def comma(x):
    return f"{x:,}"


def put(key, val):
    if key in N:
        sys.exit(f"build_deck: key {key} set twice")
    N[key] = val


INS = "INSTRUCTIONS/peeling_the_onion/"
OBJ = INS + "onion training objective.md"
PAPER = INS + "onion paper.md"
DES = INS + "closeout/settled miniCANDI design.md"
RQ = INS + "closeout/ready queue for scaling.md"
TODO = INS + "closeout/closeout todo.md"
PLAN24 = INS + "closeout/final 24 hours plan.md"
ARMS = "results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv"
READ = "results/ablations_closeout/ABLATIONS_READ.md"
CHK = "results/region_sampling_checks/check_v3.tsv"
CHKMD = "results/region_sampling_checks/PAIR_REGIONS_V3.md"
WV = "results/closeout_A3/where_vs_allviews.tsv"
WVMD = "results/closeout_A3/WHERE_VIEWS.md"
# results that landed after version 1 (2026-11-18), and the PI decisions of 2026-11-17
# version 3: the two-sided ladder (PI decision of 2026-11-17, row 5) and the one-sided form beside it once
LAD = "results/closeout_E4/ladder_two_sided_2026-11-18/two_sided/ladder.tsv"
LAD1 = "results/closeout_E4/ladder_two_sided_2026-11-18/one_sided/ladder.tsv"
LADCC = "results/closeout_E4/ladder_two_sided_2026-11-18/two_sided/common_cells.tsv"
LADCMP = "results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md"
LADPV = "results/closeout_E4/ladder_2026-11-18/pair_values.tsv.gz"
D2L = "results/closeout_D2/layers_2026-11-18/layers.tsv"
QCC = "results/closeout_A4/qc_tables_2026-11-18/coverage.txt"
RSTAT = "presentations/onion-closeout-2026-11-18/run_status.json"   # replaces the run log (it changes per snapshot)
PID = INS + "closeout/PI decisions 2026-11-17.md"

# ---- colors (light, printed on a white card in both themes) -------------------------------------
INK, INK2, MUTED, HAIR = "#1b1d1f", "#36393d", "#6c6f73", "#d9dad6"
MC = {"convae": "#2f6f8f", "convattn": "#4f9a8a", "unet": "#b5543a"}
MODEL = {"convae": "conv autoencoder", "convattn": "conv + attention", "unet": "U-Net"}
MODELS = ["convae", "convattn", "unet"]
REF = {"convae": "rung2_base_heldout", "convattn": "rung3_base_heldout", "unet": "rung4_base_heldout"}
OK, WAIT, BAD, GREY = "#3f8a5c", "#c99a2e", "#b5543a", "#a7a9ad"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def png(key, fig):
    """Save a figure as a palette PNG (small, under 1 MB in all) and embed it as base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    FIG[key] = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def panel_label(ax, s):
    ax.text(-0.01, 1.04, s, transform=ax.transAxes, fontsize=16, fontweight="bold", color="#2f6f8f",
            ha="right", va="bottom")


# ==================================================================================================
# Arm summary (16-region screening set; imputation and calibration use all 256 regions)
# ==================================================================================================
ARM = {}
for r in tsv(ARMS):
    ARM[r["arm"]] = r
KEYS = {"imp": "imputation_raw", "cor": "correction_v3_raw", "cal": "uncertainty_raw",
        "comp": "composite_v3", "conly": "correction_only"}


def mr(arm, k):
    r = ARM.get(arm)
    if r is None:
        sys.exit(f"build_deck: {ARMS} has no row {arm}")
    m, g = r[KEYS[k] + ".mean"], r[KEYS[k] + ".range"]
    return (float(m) if m not in ("", "None") else None, float(g) if g not in ("", "None") else None)


def diff(arm, ref, k):
    (a, ra), (b, rb) = mr(arm, k), mr(ref, k)
    if a is None or b is None:
        return None
    bar = max(ra or 0, rb or 0)
    return dict(d=a - b, bar=bar, arm=a, ref=b, ra=ra, rb=rb, z=(a - b) / bar if bar else 0.0,
                beyond=abs(a - b) > bar, x2=abs(a - b) > 2 * bar, n=int(ARM[arm]["n_seeds"]))


for m in MODELS:
    for k in KEYS:
        mean, rng = mr(REF[m], k)
        put(f"ref.{m}.{k}", f3(mean))
        put(f"ref.{m}.{k}.r", f3(rng))
cite("arms", ARMS, READ)

# ==================================================================================================
# Criterion 1: data
# ==================================================================================================
fr = grab(PAPER, r"It read ([0-9,]+) measurement directories and wrote [0-9,]+ rows: ([0-9,]+) training, "
                 r"([0-9,]+) validation and ([0-9,]+) blind rows", 4)
for k, v in zip(("rows", "train", "val", "blindrows"), fr):
    put(f"data.{k}", v)
VT = grab(OBJ, r"\*\*Views in training\*\*.*?: base ([0-9]+); depth 0\.50, 0\.25, 0\.125 \(([0-9]+) each\); "
               r"no dedup ([0-9]+); MAPQ 0 ([0-9]+); single-end ([0-9]+); crop 36 ([0-9]+); crop 50 ([0-9]+)\.", 7)
VT = [int(x) for x in VT]
views = [("base view", VT[0], "base"), ("depth 0.50", VT[1], "align"), ("depth 0.25", VT[1], "align"),
         ("depth 0.125", VT[1], "align"), ("no duplicate removal", VT[2], "align"),
         ("no mapping-quality filter", VT[3], "align"), ("single-end", VT[4], "read"),
         ("reads cut to 36 bp", VT[5], "read"), ("reads cut to 50 bp", VT[6], "read")]
put("data.views_total", comma(sum(v for _, v, _ in views)))
put("data.base", comma(VT[0]))
put("data.cf", comma(sum(v for _, v, _ in views[1:])))
put("data.readlevel", comma(VT[4] + VT[5] + VT[6]))
cite("data", PAPER, OBJ)

fig, ax = plt.subplots(figsize=(12.6, 4.6))
VC = {"base": "#36393d", "align": "#2f6f8f", "read": "#d9822b"}
ys = list(range(len(views)))[::-1]
for y, (n, v, kind) in zip(ys, views):
    ax.barh(y, v, color=VC[kind], height=0.68)
    ax.text(v + 8, y, comma(v), va="center", fontsize=12, color=INK)
ax.set_yticks(ys, [n for n, _, _ in views])
ax.set_xlabel("training measurements of this view (table version with filled covariates)")
ax.set_title("Training views per kind: one base view per measurement, eight counterfactual views", loc="left")
ax.set_xlim(0, VT[0] * 1.12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=VC["base"], label="base view (all reads)"),
                   Patch(color=VC["align"], label="alignment-level view (from the base alignment)"),
                   Patch(color=VC["read"], label="read-level view (from the FASTQ files)")],
          loc="lower right", frameon=False, fontsize=12)
fig.tight_layout()
png("views", fig)

# data deliverables: parsed from the close-out todo list (statuses written by the foreman)
d2 = grab(TODO, r"add-on table eic_v1_2x \(([0-9]+) second experiments, ([0-9]+) other-experiment pairs\)", 2)
d3 = grab(TODO, r"candidate table \(([0-9,]+) rows; DNase-seq correction-scorable\)")
d3c = grab(TODO, r"the 4 row rebuilds \(about ([0-9,]+) CPU-hours\)")
d4 = grab(TODO, r"blind-only scope ruled 2026-11-17 \(([0-9]+) blind rows \+ ([0-9]+) controls \+ ([0-9]+) reruns, "
                r"about ([0-9,]+) allocated CPU-hours\)", 4)
qcn = grab(DES, r"The QC table holds ([0-9,]+) of ([0-9,]+) records", 2)
put("data.d2", d2[0])
put("data.d2pairs", d2[1])
put("data.d3rows", d3)
put("data.d3cost", d3c)
put("data.blind", d4[0])
put("data.blindctl", d4[1])
put("data.blindre", d4[2])
put("data.blindcost", d4[3])
put("data.qc", qcn[0])
put("data.qcall", qcn[1])
cite("data2", TODO, DES, PAPER, RSTAT, QCC, PID)

# ---- run-log facts of the last night (the run log snapshot in the vault) ----------------------------
RS = json.loads(text(RSTAT))       # run-log facts, copied by the foreman from the run log (DECK.md)
_a, _s, _q, _b, _r = RS["archive"], RS["scratch"], RS["qc_fill"], RS["bp_store"], RS["rescoring"]
pl_arch = [_a["tars_ok"], _a["tars_all"], _a["gib"], _a["entries"]]
pl_sc1 = [_s["before_gib"], _s["after_archive_gib"], _s["soft_quota_gib"]]
pl_sc2 = [_s["after_archive_gib"], _s["after_work_folder_gib"], _s["soft_quota_gib"]]
pl_qc = [_q["records_whole"], _q["records_pr1"], _q["ok"], _q["no_read_set"], _q["refused"]]
pl_rc = _b["views_recounted"]
pl_pt = [_b["points_independent"], _b["points_store"], _b["bins_differing"]]
pl_tb = [_b["measurements"], _b["pairs"]]
pl_rs = [_r["submitted_day"], _r["submitted_utc"], _r["run_seeds"], _r["jobs"]]
pl_rsh = _r["predict_hours_per_round"]
pl_cap = _r["cap_active_run_seeds"]
pl_now = [_r["status_utc"], _r["active"], _r["scored"]]
pl_now_day = _r["status_day"]
if _b["verify_state"] != "running":   # slide 5 says the 1 bp verify job is running (the variance job has completed)
    sys.exit(f"build_deck: {RSTAT} bp_store.verify_state is {_b['verify_state']!r}; reword slide 5 first")
put("arc.tars", pl_arch[0])
put("arc.gib", pl_arch[2])
put("arc.entries", pl_arch[3])
put("sc.before", pl_sc1[0])
put("sc.mid", pl_sc1[1])
put("sc.quota", pl_sc1[2])
put("sc.after", pl_sc2[1])
put("qc.whole", pl_qc[0])
put("qc.pr1", pl_qc[1])
put("qc.ok", pl_qc[2])
put("qc.noread", pl_qc[3])
put("qc.refused", pl_qc[4])
put("bp.views", pl_rc)
put("bp.pts", comma(int(pl_pt[0])))
put("bp.ptstore", comma(int(pl_pt[1])))
put("bp.meas", pl_tb[0])
put("bp.pairs", pl_tb[1])
put("rs.day", pl_rs[0])
put("rs.start", pl_rs[1])
put("rs.n", pl_rs[2])
put("rs.hours", pl_rsh)
put("rs.cap", pl_cap)
put("rs.now", pl_now[0])
put("rs.nowday", pl_now_day)
put("rs.scored", pl_now[2])
# predict time if the cap holds: ceil(run-seeds / cap) rounds of the measured hours each (the agent's arithmetic)
rounds = -(-int(pl_rs[2]) // int(pl_cap))
put("rs.rounds", str(rounds))
put("rs.total", f"{rounds * float(pl_rsh):.0f}")

QCF = {}
for line in text(QCC).splitlines():
    c = line.split("\t")
    if len(c) == 3 and "/" in c[2]:
        k, n = c[2].split("/")
        QCF[c[0]] = (int(k), int(n))
for k in ("nsc", "rsc", "nrf", "pbc1", "pbc2", "frip", "frag_len"):
    if k not in QCF:
        sys.exit(f"build_deck: {QCC} has no row {k}")
put("qcf.rows", comma(QCF["nsc"][1]))
put("qcf.nsc", comma(QCF["nsc"][0]))
put("qcf.frag", comma(QCF["frag_len"][0]))
put("qcf.nrf", comma(QCF["nrf"][0]))
put("qcf.frip", comma(QCF["frip"][0]))
cite("side", QCC, RSTAT, INS + "closeout/closeout A4 QC metrics on the input side.md")

fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
QN = [("NSC", "nsc"), ("RSC", "rsc"), ("fragment length", "frag_len"), ("NRF", "nrf"), ("PBC1", "pbc1"),
      ("PBC2", "pbc2"), ("FRiP", "frip")]
ys = list(range(len(QN)))[::-1]
for y, (lab, k) in zip(ys, QN):
    kk, nn = QCF[k]
    ax.barh(y, 100 * kk / nn, color="#2f6f8f" if kk else GREY, height=0.66)
    ax.text(100 * kk / nn + 1.5, y, f"{comma(kk)} of {comma(nn)}" + ("  (no peaks called in this pass)" if not kk else ""),
            va="center", fontsize=11, color=INK)
ax.set_yticks(ys, [l for l, _ in QN])
ax.set_xlim(0, 135)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("rows with a value (% of training and validation rows)")
ax.set_title("(a) QC side tables: fill rate per metric", loc="left")
ax = axs[1]
st = [("before the removal", pl_sc1[0]), ("archived folders removed", pl_sc1[1]), ("work folder removed", pl_sc2[1])]
for i, (lab, v) in enumerate(st):
    vv = float(v.replace(",", ""))
    q = float(pl_sc1[2].replace(",", ""))
    ax.bar(i, vv, color=BAD if vv > q else OK, width=0.6)
    ax.text(i, vv + 60, f"{v} GiB", ha="center", fontsize=12, color=INK)
ax.axhline(float(pl_sc1[2].replace(",", "")), color=INK2, lw=2, ls="--")
ax.text(2.45, float(pl_sc1[2].replace(",", "")) + 60, f"soft quota {pl_sc1[2]} GiB", ha="right", fontsize=11, color=INK2)
ax.set_xticks(range(3), [l.replace(" ", "\n", 1) for l, _ in st], fontsize=11)
ax.set_ylabel("Nibi /scratch use (GiB)")
ax.set_ylim(0, float(pl_sc1[0].replace(",", "")) * 1.15)
ax.set_title("(b) Nibi scratch after the tape archive", loc="left")
fig.tight_layout()
png("side", fig)

# ==================================================================================================
# Variance by bin size on the 1 bp region store (version 4; scale_variance.tsv, VARIANCE_READ.md)
# ==================================================================================================
SV = "results/closeout_A7/variance_2026-11-18/scale_variance.tsv"
SVMD = "results/closeout_A7/variance_2026-11-18/VARIANCE_READ.md"
if _b.get("variance_state") != "completed":
    sys.exit(f"build_deck: {RSTAT} bp_store.variance_state is not 'completed'; the variance slide reports a result")
put("bp.verifyjob", _b["verify_job"])
SVR = {(r["assay"], r["layer_or_all"], r["chain"], r["from_bp"], r["to_bp"], r["scale"]): r for r in tsv(SV)}
# assay classes: the assay_class column of the store tables (Nibi CO-A7/tables/eic_res_v1/measurements.tsv)
SV_BROAD = {"H3K27me3", "H3K36me3", "H3K4me1", "H3K79me1", "H3K79me2", "H3K9me3", "H4K20me1"}
SV_CLS = [("accessibility", "accessibility (DNase-seq)"), ("broad", "broad histone marks"),
          ("narrow", "narrow histone marks"), ("control", "control (no antibody)")]
SV_COL = {"accessibility": "#d9822b", "broad": "#2f6f8f", "narrow": "#b5543a", "control": "#6c6f73"}


def sv_cls(a):
    return ("accessibility" if a == "DNase-seq" else "control" if a == "chipseq-control"
            else "broad" if a in SV_BROAD else "narrow")


SV_STEP = {5: ("1,5,25,125", "1", "5"), 25: ("1,5,25,125", "5", "25"), 50: ("1,5,25,50", "25", "50"),
           125: ("1,5,25,125", "25", "125")}     # counts rows are cumulative: the share below to_bp


def sv_below(c, layer, b):
    """Track-weighted mean over the assays of class c of the counts share below b; (value, n, assays, lo, hi)."""
    if b == 1:
        return 0.0, 0, 0, 0.0, 0.0
    fld = "share_raw" if layer == "all" else "share_reproducible"
    num = den = 0.0
    per = []
    for (a, l, ch, f, t, s), r in SVR.items():
        if sv_cls(a) != c or l != layer or s != "counts" or (ch, f, t) != SV_STEP[b]:
            continue
        if r["n_tracks"] == "0" or r[fld] == "":
            continue
        num += float(r[fld]) * int(r["n_tracks"])
        den += int(r["n_tracks"])
        per.append(float(r[fld]))
    if not den:
        return None
    return num / den, int(den), len(per), min(per), max(per)


def f4(x):
    return f"{x:.4f}".replace("-", "−")


SVB = {(c, l, b): sv_below(c, l, b) for c, _ in SV_CLS for l in ("all", "isogenic", "pseudoreplicate")
       for b in (1, 5, 25, 50, 125)}
for c, k in (("broad", "br"), ("narrow", "nr")):
    raw, rep = SVB[(c, "all", 25)], SVB[(c, "isogenic", 25)]
    put(f"bv.{k}.raw25", f3(raw[0]))
    put(f"bv.{k}.n", str(raw[1]))
    put(f"bv.{k}.na", str(raw[2]))
    put(f"bv.{k}.rep25", f4(rep[0]))
    put(f"bv.{k}.rep25.lo", f4(rep[3]))
    put(f"bv.{k}.rep25.hi", f4(rep[4]))
    put(f"bv.{k}.np", str(rep[1]))
    dr = SVB[(c, "all", 125)][0] - raw[0]
    dp = SVB[(c, "isogenic", 125)][0] - rep[0]
    put(f"bv.{k}.raw25_125", f3(dr))
    put(f"bv.{k}.rep25_125", f4(dp))
    put(f"bv.{k}.pct25_125", f"{100 * dp / dr:.0f}")
    put(f"bv.{k}.pct25", f"{100 * rep[0] / raw[0]:.1f}")
put("bv.noise25", f3(1 - 1 / 25))
put("bv.dn.raw25", f3(SVB[("accessibility", "all", 25)][0]))
psall = [float(r["share_reproducible"]) for (a, l, ch, f, t, s), r in SVR.items()
         if l == "pseudoreplicate" and s == "counts" and r["n_tracks"] != "0"]
put("bv.ps.n", str(len(psall)))
put("bv.ps.neg", str(sum(v < 0 for v in psall)))
put("bv.ctl.ps5", f4(SVB[("control", "pseudoreplicate", 5)][0]))
ctl_reads = int(grab(SVMD, r"its provenance file holds ([0-9,]+) reads in the 768 regions").replace(",", ""))
put("bv.ctl.pred5", f4(-ctl_reads / (768 * 409600) * 4 / 5))      # exclusion term -m (b - 1) / b at b = 5
put("bv.dn.id", grab(SVMD, r"The one DNase-seq track is (ENCSR[0-9A-Z]+)"))
cite("binvar", SV, SVMD, RSTAT)

fig, axs = plt.subplots(1, 2, figsize=(13.4, 4.7))
BS = [1, 5, 25, 50, 125]
for ax, layer, title in ((axs[0], "all", "(a) all variance (raw)"),
                         (axs[1], "isogenic", "(b) variance shared by biological replicates")):
    for c, lab in SV_CLS:
        pts = [(b, SVB[(c, layer, b)]) for b in BS if SVB[(c, layer, b)] is not None]
        if layer != "all" and len(pts) <= 1:
            continue
        x = [b for b, _ in pts]
        y = [v[0] for _, v in pts]
        n = max(v[1] for _, v in pts)
        unit = "track" if layer == "all" else "replicate pair"
        ax.plot(x, y, "-o", color=SV_COL[c], lw=2.2, ms=5,
                label=f"{lab}, {n} {unit}{'' if n == 1 else 's'}")
    ax.axvline(25, color=MUTED, lw=1.2, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(BS, [str(b) for b in BS])
    ax.minorticks_off()
    ax.set_xlabel("bin size (base pairs, log scale)")
    ax.set_title(title, loc="left")
axs[0].plot(BS, [1 - 1 / b for b in BS], "--", color=INK2, lw=1.4, label="pure read-sampling noise")
axs[0].set_ylim(0, 1.02)
axs[0].set_ylabel("share of the 1 base pair count variance\nthat lies below the bin size")
axs[0].text(26.5, 0.66, "model bin", fontsize=11, color=MUTED)
axs[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=1, fontsize=10.5)
axs[1].axhline(0, color=HAIR, lw=1)
axs[1].set_ylim(-0.0005, 0.0075)
axs[1].set_ylabel("share of the 1 base pair count variance\nthat lies below the bin size")
axs[1].text(26.5, 0.0068, "model bin", fontsize=11, color=MUTED)
axs[1].legend(loc="upper left", frameon=False, fontsize=10.5)
fig.tight_layout()
png("binvar", fig)

# ==================================================================================================
# Replicate ladder (two-sided, assay-matched; ladder.tsv, common_cells.tsv) and second-experiment layers
# ==================================================================================================
LR, LR1 = {}, {}
for d_, f_ in ((LR, LAD), (LR1, LAD1)):
    for r in tsv(f_):
        if r["stratum"] == "all" and r["band"] == "all":
            d_[(r["assay_class"], r["rung"], r["pair_class"])] = r
CLS = [("accessibility_dnase", "DNase-seq"), ("broad_histone", "broad histone marks"),
       ("narrow_histone", "narrow histone marks")]
RUNGS = [("pseudoreplicate", "", "sampling (read halves)"), ("sequencing", "", "sequencing run"),
         ("technical", "", "library"), ("isogenic", "", "biological replicate"), ("anisogenic", "", "donor"),
         ("other_experiment", "experiment", "experiment repeat"), ("other_experiment", "experiment;lab", "lab alone"),
         ("other_experiment", "experiment;lab;donor", "lab with donor"),
         ("other_experiment", "experiment;donor", "repeat with donor"),
         ("other_cell_type", "", "cell type")]
for c, _ in CLS + [("accessibility_atac", "")]:
    for rg, pc, _ in RUNGS:
        if (c, rg, pc) not in LR or (c, rg, pc) not in LR1:
            sys.exit(f"build_deck: {LAD} or {LAD1} has no all/all row {c} {rg} {pc}")


def est(r):
    return r["layer_increment"] not in ("", "nan")


def why(r):
    f = r["flag"]
    if f.startswith("no pair: the per-library"):
        return "units not built"
    if f.startswith("no pair in EIC"):
        return "no pair in EIC"
    if int(r["n_pairs"]) == 0 and int(r["n_pairs_all"]) > 0 and "pairs whose b-side" in f:
        return f"a side has no replicate pair ({r['n_pairs_all']} pairs)"
    m = re.match(r"not estimable: ([0-9]+) pairs from ([0-9]+) cells", f)
    return f"too few matched: {m.group(1)} pairs, {m.group(2)} cells" if m else f[:30]


# common-cell check (replaces telescoping): groups that pass, all strata and bands
CC = {}
for r in tsv(LADCC):
    CC.setdefault((r["assay_class"], r["stratum"], r["band"]), r["ok"])
put("lad.cc.pass", str(sum(v == "1" for v in CC.values())))
put("lad.cc.n", str(len(CC)))
put("lad.cc.hpass", str(sum(v == "1" for k, v in CC.items() if k[0] != "accessibility_dnase")))
put("lad.cc.hn", str(sum(1 for k in CC if k[0] != "accessibility_dnase")))
# DNase-seq common depth of the ladder's pair values (one value for the assay; see COMPARE.md)
import gzip
with gzip.open(path(LADPV), "rt", newline="") as fh_:
    dn_depth = {r["depth"] for r in csv.DictReader(fh_, delimiter="\t") if r["assay"] == "DNase-seq"}
if len(dn_depth) != 1:
    sys.exit(f"build_deck: {LADPV}: DNase-seq has {len(dn_depth)} common depths")
put("lad.dn.depth", comma(int(float(dn_depth.pop()))))
put("lad.dn.next", grab(LADCMP, r"The next shallowest DNase-seq side has ([0-9,]+) reads"))
put("lad.dn.cell", grab(LADCMP, r"The shallowest side is the second experiment [A-Z0-9]+, pooled track, cell (C[0-9]+)"))

fig = plt.figure(figsize=(13.4, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.2], hspace=0.62, wspace=0.08)
ax = fig.add_subplot(gs[0, :])
sc_ = 1e-3
yy = 0
ticks = []
for c, lab in CLS[1:]:
    for form, d_, col in (("one-sided (version 2)", LR1, GREY), ("two-sided (now)", LR, "#b5543a")):
        r = d_[(c, "other_cell_type", "")]
        v, lo, hi = (float(r[k]) / sc_ for k in ("layer_increment", "ci_low", "ci_high"))
        ax.plot([lo, hi], [yy, yy], color=col, lw=3)
        ax.plot(v, yy, "o", color=col, ms=8)
        ax.text(7.2, yy, f"{v:.2f} [{lo:.2f}, {hi:.2f}]; {r['n_pairs']} pairs, {r['n_cells']} cells", va="center",
                fontsize=10, color=INK)
        ticks.append(f"{lab.replace(' histone marks', '')}: {form.split(' ')[0]}")
        yy += 1
    yy += 0.4
ax.set_yticks([0, 1, 2.4, 3.4], ticks, fontsize=10)
ax.set_ylim(3.9, -0.5)
ax.set_xlim(0, 12.5)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("cell-type increment over the biological replicate, in units of $10^{-3}$ (pair value; 95% interval)",
              fontsize=10.5)
ax.set_title("(a) Cell-type rung: the one-sided form of version 2 against the two-sided form", loc="left",
             fontsize=13.5)
for j, (c, lab) in enumerate(CLS):
    ax = fig.add_subplot(gs[1, j])
    rows_ = [rr for rr in RUNGS if rr[0] != "pseudoreplicate"]
    xs_ = [abs(float(LR[(c, rg, pc)][k])) for rg, pc, _ in rows_ for k in ("ci_low", "ci_high")
           if est(LR[(c, rg, pc)])]
    ex = int(math.floor(math.log10(max(xs_))))
    sc_ = 10.0 ** ex
    lim = max(xs_) / sc_ * 1.15
    for i, (rg, pc, nm) in enumerate(rows_):
        r = LR[(c, rg, pc)]
        if est(r):
            v, lo, hi = (float(r[k]) / sc_ for k in ("layer_increment", "ci_low", "ci_high"))
            col = "#4f9a8a" if rg == "isogenic" else ("#b5543a" if rg == "other_cell_type" else "#6b4fa0")
            ax.plot([lo, hi], [i, i], color=col, lw=3)
            ax.plot(v, i, "o", color=col, ms=8)
            ax.text(lim * 0.98, i - 0.33, f"{r['n_pairs']} pairs, {r['n_cells']} cells", ha="right", fontsize=8.8,
                    color=MUTED)
        else:
            ax.text(-lim * 0.95, i, why(r), va="center", fontsize=8.6, color=GREY, style="italic", zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(rows_) - 0.5, -0.6)
    ax.set_yticks(range(len(rows_)), [nm for _, _, nm in rows_] if j == 0 else [], fontsize=10.5)
    ttl = f"({'bcd'[j]}) {lab}" + (" (depth defect)" if c == "accessibility_dnase" else "")
    ax.set_title(ttl, loc="left", fontsize=12.5, color=GREY if c == "accessibility_dnase" else INK)
    ax.set_xlabel(f"two-sided increment over the rung below,\nin units of $10^{{{ex}}}$ (pair value; 95% interval)",
                  fontsize=10.5)
fig.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.12)
png("ladder", fig)
def g3(x):
    s = f"{x:.3g}" if abs(x) < 1e-3 else f"{x:.4f}"
    return s.replace("-", "−")


for c, tag in (("accessibility_dnase", "dn"), ("broad_histone", "br"), ("narrow_histone", "nr")):
    for pre, d_ in (("lad", LR), ("lad1", LR1)):
        for rg, pc, key in (("pseudoreplicate", "", "samp"), ("isogenic", "", "rep"), ("other_cell_type", "", "ct"),
                            ("other_experiment", "experiment;lab;donor", "ld"),
                            ("other_experiment", "experiment;donor", "rd")):
            r = d_[(c, rg, pc)]
            if not est(r) or (pre == "lad1" and rg in ("pseudoreplicate", "isogenic")):
                continue
            put(f"{pre}.{tag}.{key}.sh", f"{100 * float(r['share']):.1f}%".replace("-", "−"))
            put(f"{pre}.{tag}.{key}.n", r["n_pairs"])
            put(f"{pre}.{tag}.{key}.c", r["n_cells"])
            put(f"{pre}.{tag}.{key}.u", r["share_units"])
            put(f"{pre}.{tag}.{key}.v", g3(float(r["layer_increment"])))
            put(f"{pre}.{tag}.{key}.lo", g3(float(r["ci_low"])))
            put(f"{pre}.{tag}.{key}.hi", g3(float(r["ci_high"])))
    r1, r2 = LR1[(c, "other_cell_type", "")], LR[(c, "other_cell_type", "")]
    if est(r1) and est(r2):
        ch = 100 * (float(r2["layer_increment"]) - float(r1["layer_increment"])) / abs(float(r1["layer_increment"]))
        put(f"lad.{tag}.ct.chg", f"{ch:+.0f}%".replace("-", "−"))
        put(f"lad.{tag}.ct.left", r2["n_pairs_left_b"])
for c, tag in (("broad_histone", "br"), ("narrow_histone", "nr")):
    put(f"lad.{tag}.oe.left", str(sum(int(LR[(c, "other_experiment", pc)]["n_pairs_left_b"] or 0)
                                      for rg, pc, _n in RUNGS if rg == "other_experiment")))
    put(f"lad.{tag}.oe.all", str(sum(int(LR[(c, "other_experiment", pc)]["n_pairs_all"] or 0)
                                     for rg, pc, _n in RUNGS if rg == "other_experiment")))
r = LR[("accessibility_atac", "pseudoreplicate", "")]
put("lad.atac.n", r["n_pairs_all"])
put("lad.atac.c", r["n_cells_all"])
cite("ladder", LAD, LAD1, LADCC, LADCMP, LADPV, PID, INS + "closeout/closeout E4 complete the replicate ladder.md")

# second-experiment layers (pooled by assay class), lab reported together with donor (PI decision 3)
DL = {}
for r in tsv(D2L):
    if r["stratum"] == "all" and r["band"] == "all" and r["pooled_by_class"] == "1" and r["rung"] == "other_experiment":
        DL[(r["assay_or_class"], r["pair_class"])] = r
fig, axs = plt.subplots(1, 2, figsize=(13.2, 3.6), sharey=True)
D2R = [("", "all second-experiment pairs"), ("experiment;lab;donor", "lab with donor")]
for ax, (c, lab) in zip(axs, CLS[1:]):
    for i, (pc, nm) in enumerate(D2R):
        r = DL.get((c, pc))
        if r is None:
            sys.exit(f"build_deck: {D2L} has no pooled row {c} {pc}")
        v, lo, hi = float(r["layer_increment"]), float(r["ci_low"]), float(r["ci_high"])
        ax.plot([lo, hi], [i, i], color="#6b4fa0", lw=3)
        ax.plot(v, i, "o", color="#6b4fa0", ms=9)
        ax.text(0.01, i - 0.33, f"{r['n_pairs']} pairs, {r['n_cells']} cells; {r['tech_diff_pairs']} also differ "
                f"in run type, read length or platform", fontsize=9.6, color=MUTED, ha="left",
                transform=ax.get_yaxis_transform(), zorder=4, bbox=dict(fc="white", ec="none", pad=0.5))
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_ylim(len(D2R) - 0.4, -0.7)
    ax.set_yticks(range(len(D2R)), [nm for _, nm in D2R], fontsize=11.5)
    ax.set_title(f"({'ab'[CLS.index((c, lab)) - 1]}) {lab}", loc="left", fontsize=13)
    ax.set_xlabel("increment over the biological replicate (pair value; 95% interval)", fontsize=10.5)
fig.tight_layout()
png("second", fig)
for c, tag in (("broad_histone", "br"), ("narrow_histone", "nr")):
    for pc, k in (("", "all"), ("experiment;lab;donor", "ld")):
        r = DL[(c, pc)]
        put(f"d2.{tag}.{k}.n", r["n_pairs"])
        put(f"d2.{tag}.{k}.c", r["n_cells"])
        put(f"d2.{tag}.{k}.v", f"{float(r['layer_increment']):+.4f}".replace("-", "−"))
        put(f"d2.{tag}.{k}.lo", f"{float(r['ci_low']):.4f}".replace("-", "−"))
        put(f"d2.{tag}.{k}.hi", f"{float(r['ci_high']):.4f}".replace("-", "−"))
put("d2.small", str(max(int(DL[(c, pc)]["n_pairs"]) for (c, pc) in DL if c.startswith("accessibility"))))
put("d2.dropped", grab(PID, r"\| 4 \| The ([0-9]+) unbuilt DNase-seq second-experiment rows"))
put("d2.kept", grab(PID, r"The add-on stays at ([0-9]+) experiments"))
cite("second", D2L, PID, LAD, LADCMP, INS + "closeout/closeout D2 second experiments and cross-lab pairs.md")

fig, ax = plt.subplots(figsize=(12.6, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")
boxes = [
    (2, 24, "Tables, first freeze", f"{fr[0]} rows\n{fr[1]} training, {fr[2]} validation", OK, "delivered"),
    (27, 24, "Covariates filled", "lab, antibody, donor\nand platform filled", OK, "delivered (default)"),
    (52, 24, "Second experiments", f"add-on table: {d2[0]} second\nexperiments, {d2[1]} pairs", WAIT,
     "layers done; models not run"),
    (77, 24, "DNase-seq update", f"candidate table,\n{d3} rows", WAIT, "built; 4 rebuilds owed"),
    (2, 2, "QC side tables", f"{pl_qc[0]} + {pl_qc[1]} records filled;\nFRiP empty (no peaks called)", OK,
     "delivered"),
    (27, 2, "1 bp region store", f"{pl_rc} views recounted;\ntables: {pl_tb[0]} measurements", WAIT,
     "built; check running"),
    (52, 2, "Blind tracks rebuilt", f"{d4[0]} blind rows, {d4[1]} controls,\n{d4[2]} reruns", BAD,
     "not submitted"),
    (77, 2, "Final model, board row", "needs the settled design\nand the blind build", BAD, "not built"),
]
for x, y, t, b, c, s in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 21, 17.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc=c, ec=c,
                                alpha=0.13))
    ax.add_patch(FancyBboxPatch((x, y), 21, 17.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc="none", ec=c,
                                lw=2))
    ax.text(x + 10.5, y + 14.6, t, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(x + 10.5, y + 8.6, b, ha="center", va="center", fontsize=11.5, color=INK2, linespacing=1.3)
    ax.text(x + 10.5, y + 2.2, s, ha="center", va="center", fontsize=11.5, fontweight="bold", color=c)
for x in (23.5, 48.5, 73.5):
    ax.annotate("", xy=(x + 3, 32.8), xytext=(x, 32.8), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
fig.tight_layout()
png("data_state", fig)

# ==================================================================================================
# Where the counterfactual views help
# ==================================================================================================
VARMS = [("base view only", "abl_views_base_only_{m}", "rung4_views_base_only"),
         ("base + depth thinning", "abl_views_base_depth_{m}", "rung4_views_base_depth"),
         ("base + read-level views", "abl_views_base_fastq_{m}", "rung4_views_base_fastq"),
         ("base + alignment-level views", "abl_views_base_alignment_{m}", "rung4_views_base_alignment")]
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.9), sharey=True)
for ax, m in zip(axs, MODELS):
    rows = [("all views (reference)", REF[m])] + [(n, (t.format(m=m) if m != "unet" else u)) for n, t, u in VARMS]
    for i, (n, arm) in enumerate(rows):
        mean, rng = mr(arm, "conly")
        nseed = int(ARM[arm]["n_seeds"])
        c = MC[m] if i == 0 else "#8a8d92"
        ax.plot([i, i], [mean - rng / 2, mean + rng / 2], color=c, lw=6, alpha=0.35, solid_capstyle="butt")
        ax.plot(i, mean, "o", color=c, ms=10)
        ax.text(i, mean + rng / 2 + 0.03, f3(mean) + ("\n2 seeds" if nseed == 2 else ""), ha="center",
                va="bottom", fontsize=10.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(rows)), ["all\nviews", "base\nonly", "base +\ndepth", "base +\nread-level",
                                     "base +\nalignment"], fontsize=11)
    ax.set_title(f"{['(a)', '(b)', '(c)'][MODELS.index(m)]} {MODEL[m]}", loc="left")
    ax.set_ylim(-0.2, 1.0)
axs[0].set_ylabel("correction-only term\n(share of the covariate cost closed)")
fig.suptitle("Correction-only term by training views, three models, 16 pair regions (screening), 3 seeds "
             "(2 where marked); bar = seed range", fontsize=12.5, color=MUTED, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ax in axs[1:]:
    ax.spines["left"].set_visible(True)
png("views_conly", fig)
for m in MODELS:
    d = diff("abl_views_base_only_" + m if m != "unet" else "rung4_views_base_only", REF[m], "conly")
    put(f"vw.bo.{m}", f3(d["arm"]))
    put(f"vw.bo.{m}.d", s3(d["d"]))
    put(f"vw.bo.{m}.bar", f3(d["bar"]))
    dc = diff("abl_views_base_only_" + m if m != "unet" else "rung4_views_base_only", REF[m], "comp")
    put(f"vw.bocomp.{m}.d", s3(dc["d"]))
    put(f"vw.bocomp.{m}.bar", f3(dc["bar"]))
for m in ("convae", "convattn"):
    d = diff(f"abl_views_base_alignment_{m}", REF[m], "imp")
    put(f"vw.al.{m}.d", s3(d["d"]))
    put(f"vw.al.{m}.bar", f3(d["bar"]))
cite("views", ARMS, READ)

# U-Net per pair type, per seed, from where_vs_allviews.tsv
W = {}
for r in tsv(WV):
    W[(r["arm"], r["quantity"], r["group"])] = r
WARMS = [("all views", "rung4_base_heldout", MC["unet"]), ("base only", "rung4_views_base_only", "#8a8d92"),
         ("base + depth", "rung4_views_base_depth", "#2f6f8f"),
         ("base + read-level", "rung4_views_base_fastq", "#d9822b"),
         ("base + alignment", "rung4_views_base_alignment", "#6b4fa0")]
WQ = [("(a) depth-thinning pairs", "thin_assay", "__macro__"),
      ("(b) no-dedup + MAPQ pairs", "corr_knob_all", "mapq+nodedup"),
      ("(c) crop pairs (scored term)", "corr_knob_all", "crop")]
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.7))
for ax, (title, q, g) in zip(axs, WQ):
    for i, (n, arm, c) in enumerate(WARMS):
        r = W.get((arm, q, g))
        if r is None:
            sys.exit(f"build_deck: {WV} has no row {arm} {q} {g}")
        vals = [float(x) for x in r["values"].split(",")]
        ax.scatter([i] * len(vals), vals, color=c, s=46, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [float(r["mean"])] * 2, color=c, lw=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(WARMS)), [n.replace(" + ", " +\n").replace("all views", "all\nviews")
                                      .replace("base only", "base\nonly") for n, _, _ in WARMS], fontsize=9.6)
    ax.set_title(title, loc="left", fontsize=14)
axs[0].set_ylabel("gap closed (0 = depth-only scaling,\n1 = oracle reference)")
fig.suptitle("U-Net: gap closed by pair type and training views, 16 pair regions (screening); dot = one seed, "
             "line = mean", fontsize=12.5, color=MUTED, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
png("views_pairs", fig)


def wrow(arm, q, g):
    return W[(arm, q, g)]


r0, r1 = wrow("rung4_base_heldout", "thin_assay", "__macro__"), wrow("rung4_views_base_depth", "thin_assay", "__macro__")
put("wv.thin.all", f3(float(r0["mean"])))
put("wv.thin.all.r", f3(float(r0["range"])))
put("wv.thin.depth", f3(float(r1["mean"])))
put("wv.thin.depth.r", f3(float(r1["range"])))
nodepth = []
for arm in ("rung4_views_base_only", "rung4_views_base_fastq", "rung4_views_base_alignment"):
    nodepth += [float(x) for x in wrow(arm, "thin_assay", "__macro__")["values"].split(",")]
withdepth = []
for arm in ("rung4_base_heldout", "rung4_views_base_depth"):
    withdepth += [float(x) for x in wrow(arm, "thin_assay", "__macro__")["values"].split(",")]
put("wv.thin.with.n", str(len(withdepth)))
put("wv.thin.with.min", f3(min(withdepth)))
put("wv.thin.without.n", str(len(nodepth)))
put("wv.thin.without.max", f3(max(nodepth)))
ra, rb = wrow("rung4_views_base_alignment", "corr_knob_all", "mapq+nodedup"), wrow("rung4_base_heldout", "corr_knob_all", "mapq+nodedup")
put("wv.mq.al", f3(float(ra["mean"])))
put("wv.mq.al.r", f3(float(ra["range"])))
put("wv.mq.all", f3(float(rb["mean"])))
put("wv.mq.all.r", f3(float(rb["range"])))
cite("views2", WV, WVMD)

# ==================================================================================================
# Criterion 2: the training objective (schematic) and the loss ablations
# ==================================================================================================
ob = {}
ob["bins"] = grab(OBJ, r"One training example is a window of ([0-9,]+) bins of 25 bp")
ob["tracks"] = grab(OBJ, r"Each of the ([0-9]+) tracks \(35 assays plus the ChIP control\)")
ob["p"] = grab(OBJ, r"draw p from U\(([0-9.]+, [0-9.]+)\)")
ob["svp"] = grab(OBJ, r"With probability `same_view_prob` = ([0-9.]+)")
ob["bmf"] = grab(OBJ, r"each bin hidden with probability `bin_mask_frac` = ([0-9.]+)")
ob["lr"] = grab(OBJ, r"\| learning rate \| ([0-9e.-]+) \(`lr`\)")
for k, v in ob.items():
    put("obj." + k, v)
cite("obj", OBJ)

fig, ax = plt.subplots(figsize=(13.2, 5.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")


def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=c, ec=c, alpha=0.12))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc="none", ec=c, lw=1.8))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + (h - 5) / 2, body, ha="center", va="center", fontsize=10.8, color=INK2, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))


box(0.5, 16, 17, 20, "One window", f"{ob['bins']} bins of 25 bp,\none cell type,\n{ob['tracks']} tracks\n"
    "+ DNA sequence\n+ covariates C, C'", "#36393d")
box(23, 35, 23, 14, "(1) imputation", "track hidden whole\n(probability p ~ U(" + ob["p"] + "))\ntarget: a view of it",
    "#2f6f8f")
box(23, 18, 23, 14, "(2) correction", "input: one view\ntarget: a different view\nscored on every bin", "#d9822b")
box(23, 1, 23, 14, "(3) same-view reconstruction", f"{ob['bmf']} of bins hidden\ntarget: the same view,\n"
    "scored on hidden bins only", "#4f9a8a")
box(52, 16, 20, 20, "Model", "conv autoencoder,\nconv + attention\nor U-Net;\nabout one million\nparameters",
    "#6b4fa0")
box(78, 16, 21, 20, "Negative Binomial\nper bin and track", "mean = target depth x 2^eta\nsize n per bin\n\n"
    "loss: mean NB negative\nlog-likelihood over\nthe scored bins", "#b5543a")
for y in (42, 25, 8):
    arrow(17.8, 26, 22.6, y)
    arrow(46.3, y, 51.6, 26)
arrow(72.3, 26, 77.6, 26)
ax.text(34.5, 50.2, f"each visible track: same view with probability {ob['svp']}, else a different view",
        ha="center", fontsize=10.5, color=MUTED)
fig.tight_layout()
png("objective", fig)

LOSSES = [("imputation weight 4", "abl_losses_loss_impute_x4_{m}"), ("bin-equal weighting", "abl_losses_loss_bin_equal_{m}"),
          ("Poisson mean, NB size", "abl_losses_loss_poisson_mean_{m}"), ("NB CRPS", "abl_losses_loss_nb_crps_{m}")]
fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.6), sharey=True)
for ax, (k, title) in zip(axs, (("comp", "(a) composite (screening)"), ("imp", "(b) imputation (256 regions)"))):
    for j, (n, t) in enumerate(LOSSES):
        for i, m in enumerate(MODELS):
            d = diff(t.format(m=m), REF[m], k)
            y = j + (i - 1) * 0.24
            ax.plot([-d["bar"], d["bar"]], [y, y], color=MC[m], alpha=0.25, lw=7, solid_capstyle="butt")
            ax.plot(d["d"], y, "o", color=MC[m], ms=8, mfc=MC[m] if d["beyond"] else "white", mew=2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title(title, loc="left")
    ax.set_xlabel("arm minus base")
axs[0].set_yticks(range(len(LOSSES)), [n for n, _ in LOSSES])
axs[0].invert_yaxis()
axs[1].legend(handles=[plt.Line2D([], [], marker="o", color=MC[m], ls="", ms=8, label=MODEL[m]) for m in MODELS]
              + [plt.Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", ms=8, label="within the range (open)")],
              loc="lower right", frameon=False, fontsize=11)
fig.suptitle("Loss arms minus base, 3 seeds; shaded band = plus or minus the larger seed range; filled dot = beyond it",
             fontsize=12.5, color=MUTED, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
png("losses", fig)
cite("losses", ARMS, READ, OBJ)
d = diff("abl_losses_loss_impute_x4_convattn", REF["convattn"], "imp")
put("loss.x4.ca.d", s3(d["d"]))
put("loss.x4.ca.bar", f3(d["bar"]))

# ==================================================================================================
# Criterion 3: evaluation; the 16-versus-256 check
# ==================================================================================================
CK = {}
for r in tsv(CHK):
    if r["scope"] == "ALL":
        CK[(r["p"], r["condition"])] = r
CN = {"1": "(1) each run's value inside the\n95% interval of its 256-region value",
      "2": "(2) at least 95% of the run x assay\ncells inside their interval",
      "3": "(3) same selection verdict\n(4 architecture comparisons)",
      "4": "(4) seed range at most 1.5 times\nthe 256-region seed range",
      "5": "(5) replicate-variance metric\nunchanged",
      "6": "(6) sign of every clear\n256-region difference kept"}
fig, ax = plt.subplots(figsize=(12.6, 4.9))
ps = ["8", "16", "32"]
for j, p in enumerate(ps):
    for i, c in enumerate("123456"):
        r = CK.get((p, c))
        if r is None:
            sys.exit(f"build_deck: {CHK} has no ALL row p={p} condition={c}")
        ok = r["pass"] == "pass"
        lab = r["quantity"]
        if c == "2":
            lab = f"{float(r['value']) * 100:.0f}% inside"
        ax.add_patch(Rectangle((j, 5 - i), 0.94, 0.9, fc=OK if ok else BAD, alpha=0.18, ec=OK if ok else BAD, lw=1.5))
        ax.text(j + 0.47, 5 - i + 0.45, ("pass: " if ok else "fail: ") + lab, ha="center", va="center", fontsize=11.5,
                color=INK, fontweight="bold" if not ok else "normal")
ax.set_xlim(-2.05, 3)
ax.set_ylim(-0.1, 6)
ax.axis("off")
for i, c in enumerate("123456"):
    ax.text(-0.06, 5 - i + 0.45, CN[c], ha="right", va="center", fontsize=11, color=INK2)
for j, p in enumerate(ps):
    ax.text(j + 0.47, 6.0, f"{p} pair regions", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
fig.tight_layout()
png("check", fig)
v3n = grab(CHKMD, r"ran on ([0-9]+) runs of ([0-9]+) arms .*? with ([0-9,]+) region-bootstrap draws", 3)
put("chk.runs", v3n[0])
put("chk.arms", v3n[1])
put("chk.boot", v3n[2])
c16 = CK[("16", "1")]["quantity"]
put("chk.c1.16", c16)
put("chk.c2.16", f"{float(CK[('16', '2')]['value']) * 100:.0f}%")
put("chk.c4.16", CK[("16", "4")]["quantity"])
put("chk.c6.16", CK[("16", "6")]["quantity"])
cite("check", CHK, CHKMD)

tests = grab(DES, r"Result: ([0-9,]+) passed, ([0-9]+) skipped, ([0-9]+) failed", 3)
put("test.pass", tests[0])
put("test.skip", tests[1])
put("test.fail", tests[2])
cite("evaltest", DES, TODO, PAPER)
ev = grab(PAPER, r"It scores ([0-9]+) regions of ([0-9,]+) bins each")
put("ev.regions", ev)
put("ev.items", grab(PAPER, r"The set has ([0-9]+) imputation items"))
put("ev.pairs", grab(PAPER, r"The set has ([0-9,]+) correction pairs over"))
put("ev.crop", grab(PAPER, r"this value on the ([0-9]+) never-trained crop and crop \+ single-end pairs"))
put("ev.p16", ps[1])
put("seeds", ARM[REF["convae"]]["n_seeds"])
put("des.windows", grab(DES, r"\| 28 \| Training length \| ([0-9,]+) windows"))
cw = grab(PAPER, r"The composite is ([0-9.]+) imputation \+ ([0-9.]+) correction \+ ([0-9.]+) uncertainty", 3)
put("ev.w1", cw[0])
put("ev.w2", cw[1])
put("ev.w3", cw[2])

# ==================================================================================================
# Criterion 4: architectures on 256 regions
# ==================================================================================================
S256 = {}
for r in tsv(CHK):
    if r["p"] == "16" and r["condition"] == "1" and r["quantity"] in ("correction_v3", "correction_only"):
        arm, seed = r["scope"].split("/")
        S256.setdefault((arm, r["quantity"]), []).append(float(r["value_all_regions"]))
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
for ax, (title, k) in zip(axs, (("(a) imputation", "imp"), ("(b) correction, crop pairs", "correction_v3"),
                                ("(c) correction-only term", "correction_only"))):
    for i, m in enumerate(MODELS):
        if k == "imp":
            mean, rng = mr(REF[m], "imp")
            ax.plot([i, i], [mean - rng / 2, mean + rng / 2], color=MC[m], lw=8, alpha=0.35, solid_capstyle="butt")
            ax.plot(i, mean, "o", color=MC[m], ms=11)
            ax.text(i + 0.12, mean, f3(mean), va="center", fontsize=11)
        else:
            v = S256[(REF[m], k)]
            mean = sum(v) / len(v)
            ax.scatter([i] * len(v), v, color=MC[m], s=46, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [mean] * 2, color=MC[m], lw=3)
            ax.text(i + 0.3, mean, f3(mean), va="center", fontsize=11)
            put(f"a256.{m}.{k}", f3(mean))
            put(f"a256.{m}.{k}.r", f3(max(v) - min(v)))
    ax.set_xticks(range(3), ["conv\nautoencoder", "conv +\nattention", "U-Net"], fontsize=11.5)
    ax.set_xlim(-0.5, 2.8)
    ax.set_title(title, loc="left")
axs[0].set_ylabel("score (higher is better)")
fig.suptitle("Base recipes of the three models on all 256 regions, 3 seeds; (a) mean with seed range, "
             "(b) and (c) dot = one seed", fontsize=12.5, color=MUTED, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
png("arch", fig)
for (a, b) in (("convae", "unet"), ("convattn", "unet")):
    for r in tsv(CHK):
        if r["p"] == "16" and r["condition"] == "3" and r["scope"] == f"{REF[a]} vs {REF[b]}":
            mm = re.search(r"all regions: composite delta (-?[0-9.]+) within 2 x spread ([0-9.]+)", r["note"])
            put(f"tie.{a}", s3(float(mm.group(1))))
            put(f"tie.{a}.bar", f3(float(mm.group(2))))
cite("arch", CHK, CHKMD, ARMS, DES)

# ==================================================================================================
# Ablation overview heatmaps (two slides)
# ==================================================================================================
G1 = [("ChIP control off as input", "abl_control_input_control_output_only_{m}"),
      ("ChIP control off as output", "abl_predict_control_predict_control_off_{m}"),
      ("average baseline as input", "abl_average_input_average_input_{m}"),
      ("two-step, native base depth", "abl_two_step_two_step_native_{m}"),
      ("two-step, fixed base depth", "abl_two_step_two_step_fixed_{m}"),
      ("fixed input depth", "abl_depth_input_input_depth_fixed_{m}"),
      ("DNA off", "abl_dna_off_dna_off_{m}"),
      ("DNA from a language model", "abl_dna_source_dna_gnlm_{m}"),
      ("DNA shifted by 5 Mb (control)", "abl_dna_source_dna_shift5mb_{m}"),
      ("DNA joined by FiLM", "abl_dna_fusion_fusion_film_{m}"),
      ("DNA joined late", "abl_dna_fusion_fusion_late_{m}"),
      ("DNA by cross-attention", "abl_dna_fusion_fusion_xattn_{m}"),
      ("DNA joined at the stem", "abl_dna_fusion_fusion_early_{m}"),
      ("scrambled covariates (control)", "abl_scrambled_covariates_scrambled_covariates_{m}"),
      ("covariates: numeric only", "abl_cov_embedding_cov_numeric_only_{m}"),
      ("covariate dropout", "abl_cov_embedding_cov_dropout_{m}"),
      ("numeric covariates as codes", "abl_cov_embedding_cov_numeric_per_value_{m}"),
      ("injection: adaLN-zero", "abl_cov_injection_inject_adaln_zero_{m}"),
      ("injection: concatenation", "abl_cov_injection_form_inject_concat_{m}"),
      ("injection: tokens", "abl_cov_injection_form_inject_tokens_{m}"),
      ("mask by dropping (no token)", "abl_mask_mode_mask_drop_{m}")]
G2 = [("loss: imputation weight 4", "abl_losses_loss_impute_x4_{m}"),
      ("loss: bin-equal weighting", "abl_losses_loss_bin_equal_{m}"),
      ("loss: Poisson mean", "abl_losses_loss_poisson_mean_{m}"),
      ("loss: NB CRPS", "abl_losses_loss_nb_crps_{m}"),
      ("views: base only", "abl_views_base_only_{m}|rung4_views_base_only"),
      ("views: base + depth", "abl_views_base_depth_{m}|rung4_views_base_depth"),
      ("views: base + read-level", "abl_views_base_fastq_{m}|rung4_views_base_fastq"),
      ("views: base + alignment", "abl_views_base_alignment_{m}|rung4_views_base_alignment"),
      ("views: same-view pairs only", "abl_views_all_same_view_only_{m}|rung4_views_all_same_view_only"),
      ("transform: log(1 + x)", "abl_count_transforms_tf_log1p_{m}"),
      ("transform: asinh", "abl_count_transforms_tf_asinh_{m}"),
      ("transform: Pearson residual", "abl_count_transforms_tf_pearson_{m}"),
      ("transform: local ratio", "abl_count_transforms_tf_local_ratio_{m}"),
      ("transform: per-window quantile", "abl_count_transforms_tf_quantile_{m}"),
      ("transform: raw counts", "abl_count_transforms_tf_raw_{m}"),
      ("transform: soft clip", "abl_count_transforms_tf_soft_clip_{m}"),
      ("offset, pseudocount 0.1", "abl_count_transforms_offset_eps0.1_{m}"),
      ("offset, pseudocount 0.01", "abl_count_transforms_offset_eps0.01_{m}")]
cmap = plt.get_cmap("RdBu")
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)


def armname(t, m):
    if "|" in t:
        a, u = t.split("|")
        return u if m == "unet" else a.format(m=m)
    return t.format(m=m)


def heat(key, rows, title):
    mets = [("imp", "(a) imputation\n256 regions"), ("cor", "(b) correction, crop pairs\n16 regions (screening)"),
            ("conly", "(c) correction-only\n16 regions (screening)")]
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 0.30 * len(rows) + 1.5), sharey=True)
    nsig = 0
    for ax, (k, t) in zip(axs, mets):
        for i, (n, tpl) in enumerate(rows):
            for j, m in enumerate(MODELS):
                arm = armname(tpl, m)
                if arm not in ARM:
                    ax.add_patch(Rectangle((j, i), 1, 1, fc="#f1f1ee", ec="white", lw=1.5))
                    ax.text(j + 0.5, i + 0.5, "not run", ha="center", va="center", fontsize=10, color=GREY)
                    continue
                dd = diff(arm, REF[m], k)
                if dd is None:
                    ax.add_patch(Rectangle((j, i), 1, 1, fc="#f1f1ee", ec="white", lw=1.5))
                    continue
                z = max(-4, min(4, dd["z"]))
                ax.add_patch(Rectangle((j, i), 1, 1, fc=cmap(norm(z)), ec="white", lw=1.5))
                dag = " \u2020" if dd["n"] < 3 else ""
                if dd["beyond"]:
                    nsig += 1
                    lab = s3(dd["d"]) + (" 2x" if dd["x2"] else "") + dag
                    ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=11,
                            color="white" if abs(z) > 2.6 else INK, fontweight="bold" if dd["x2"] else "normal")
                elif dag:
                    ax.text(j + 0.5, i + 0.5, dag.strip(), ha="center", va="center", fontsize=11, color=INK2)
        ax.set_xlim(0, 3)
        ax.set_ylim(len(rows), 0)
        ax.set_xticks([0.5, 1.5, 2.5], ["conv\nautoencoder", "conv +\nattention", "U-Net"], fontsize=12)
        ax.xaxis.tick_top()
        ax.set_title(t, fontsize=14, pad=40)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_yticks([i + 0.5 for i in range(len(rows))], [n for n, _ in rows], fontsize=13)
    fig.tight_layout()
    png(key, fig)
    return nsig


heat("heat1", G1, "Color: arm minus base in units of the larger seed range (blue up, red down, clipped at 4). "
                  "Number shown when the difference is beyond the range; 2x = beyond twice the range. 3 seeds each.")
heat("heat2", G2, "Color: arm minus base in units of the larger seed range (blue up, red down, clipped at 4). "
                  "Number shown when beyond the range; 2x = beyond twice the range. Anscombe arm void (code defect).")
for t, key in (("abl_count_transforms_tf_local_ratio_{m}", "lr"), ("abl_count_transforms_tf_quantile_{m}", "q")):
    for m in MODELS:
        dd = diff(t.format(m=m), REF[m], "imp")
        put(f"tf.{key}.{m}", s3(dd["d"]))
        put(f"tf.{key}.{m}.bar", f3(dd["bar"]))
for m in MODELS:
    dd = diff(f"abl_scrambled_covariates_scrambled_covariates_{m}", REF[m], "conly")
    put(f"sc.{m}", s3(dd["d"]))
    put(f"sc.{m}.bar", f3(dd["bar"]))
dd = diff("abl_depth_input_input_depth_fixed_unet", REF["unet"], "cor")
put("fd.unet.cor", s3(dd["d"]))
put("fd.unet.cor.bar", f3(dd["bar"]))
dd = diff("abl_depth_input_input_depth_fixed_unet", REF["unet"], "imp")
put("fd.unet.imp", s3(dd["d"]))
put("fd.unet.imp.bar", f3(dd["bar"]))
for m in ("convae", "convattn"):
    dd = diff(f"abl_two_step_two_step_native_{m}", REF[m], "cor")
    put(f"ts.{m}.cor", s3(dd["d"]))
    put(f"ts.{m}.cor.bar", f3(dd["bar"]))
dd = diff("abl_average_input_average_input_unet", REF["unet"], "imp")
put("ai.unet.imp", s3(dd["d"]))
put("ai.unet.imp.bar", f3(dd["bar"]))
cite("heat", ARMS, READ)

# ==================================================================================================
# The settled design table: status of each row
# ==================================================================================================
sec = grab(DES, r"## 1\. The design table\n(.*?)\n\nEvery ablated switch")
rows = []
for line in sec.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 6 or not cells[0].isdigit():
        continue
    rows.append(dict(n=int(cells[0]), name=cells[1], conf=cells[4], status=cells[5]))
if len(rows) < 30:
    sys.exit(f"build_deck: {DES} design table has {len(rows)} rows")


def stclass(s, conf=""):
    s = s.lower()
    if "ruling" in conf.lower() and not s.startswith("provisional") and "provisional" not in s:
        return "ruling"
    if s.startswith("provisional") or ("settled" in s and "provisional" in s):
        return "provisional" if s.startswith("provisional") else "mixed"
    if "ruling" in s:
        return "ruling"
    return "settled"


def cfclass(c):
    c = c.lower()
    for k in ("strong", "moderate", "weak"):
        if c.startswith(k):
            return k
    return "other"


SC = {"settled": OK, "ruling": "#2f6f8f", "mixed": WAIT, "provisional": BAD}
SL = {"settled": "settled by evidence", "ruling": "settled by a PI ruling", "mixed": "settled for some models, provisional for others",
      "provisional": "provisional: a named run can change it"}
fig, ax = plt.subplots(figsize=(13.4, 5.6))
ncol = 4
nrow = (len(rows) + ncol - 1) // ncol
for i, r in enumerate(rows):
    c, rr = i // nrow, i % nrow
    st = stclass(r["status"], r["conf"])
    x, y = c * 3.35, nrow - 1 - rr
    ax.add_patch(FancyBboxPatch((x, y + 0.08), 3.2, 0.84, boxstyle="round,pad=0,rounding_size=0.12", fc=SC[st],
                                alpha=0.30, ec=SC[st], lw=1.6))
    nm = re.sub(r"\s*\(.*?\)", "", r["name"])
    nm = {"Pearson residual transform on the U-Net": "Pearson residuals, U-Net",
          "Two-step routing through the base view": "Two-step routing",
          "Hidden-position representation": "Hidden-position token"}.get(nm, nm)
    nm = nm if len(nm) <= 34 else nm[:33].rstrip() + "."
    ax.text(x + 0.1, y + 0.5, nm, va="center", fontsize=10.2, color=INK)
    ax.text(x + 3.12, y + 0.5, cfclass(r["conf"]) if cfclass(r["conf"]) != "other" else "", va="center", ha="right",
            fontsize=9, color=MUTED, style="italic")
ax.set_xlim(-0.05, ncol * 3.35)
ax.set_ylim(-0.1, nrow + 0.05)
ax.axis("off")
ax.legend(handles=[Patch(fc=SC[k], alpha=0.30, ec=SC[k], label=SL[k]) for k in SC], loc="upper center",
          bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
png("design", fig)
cnt = {k: sum(1 for r in rows if stclass(r["status"], r["conf"]) == k) for k in SC}
put("des.rows", str(len(rows)))
for k, v in cnt.items():
    put(f"des.{k}", str(v))
put("des.weak", str(sum(1 for r in rows if cfclass(r["conf"]) == "weak")))
cite("design", DES)

# ==================================================================================================
# What did not run, and the ready queue
# ==================================================================================================
sq = grab(RQ, r"\(`quota -w`\): [0-9,]+ KB used \(([0-9,]+) GiB\) against a soft quota of [0-9,]+ KB \(([0-9,]+) GiB\)", 2)
work = grab(RQ, r"The folder `onion/work` \(([0-9,]+) GiB\) leaves scratch")
fj = grab(RQ, r"([0-9]+) Fir jobs wait in the queue")
put("nr.used", sq[0])
put("nr.quota", sq[1])
put("nr.work", work)
put("nr.firjobs", fj)
put("nr.firend", grab(RQ, r"Planned outage from 2026-11-17 07:00 UTC to (about 2026-11-20)"))
put("nr.deadline", grab(PLAN24, r"deadline: (2026-11-18T20:00Z)"))
cite("notrun", RQ, PLAN24, RSTAT)

fig, axs = plt.subplots(1, 2, figsize=(13.2, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
used, quota, wk = [float(x.replace(",", "")) for x in (sq[0], sq[1], work)]
ax.barh(1, used - wk, color="#8a8d92", height=0.5)
ax.barh(1, wk, left=used - wk, color=WAIT, height=0.5)
ax.text(used + 25, 1, f"{sq[0]} GiB used", va="center", fontsize=12)
ax.axvline(quota, color=BAD, lw=2.5)
ax.text(quota + 20, 1.75, f"soft quota {sq[1]} GiB: above it, every write is refused", color=BAD, fontsize=11.5, ha="left")
ax.text(quota + 40, 0.62, f"pipeline work folder, {work} GiB:\nleaves after the QC fill and the recount", ha="left",
        va="top", fontsize=10.5, color=INK2)
ax.set_xlim(0, used * 1.25)
ax.set_ylim(-0.3, 2.1)
ax.set_yticks([])
ax.set_xlabel("Nibi /scratch use (GiB)")
ax.set_title("(a) Nibi scratch on 2026-11-17, 20:12 UTC", loc="left")
ax = axs[1]
import datetime as dt
t0 = dt.datetime(2026, 11, 16, 12)
H = lambda d: (d - t0).total_seconds() / 3600
ax.barh(2, H(dt.datetime(2026, 11, 20, 12)) - H(dt.datetime(2026, 11, 17, 7)), left=H(dt.datetime(2026, 11, 17, 7)),
        color=BAD, alpha=0.75, height=0.5)
ax.text(H(dt.datetime(2026, 11, 17, 9)), 2, "Fir planned outage", va="center", color="white",
        fontsize=11, fontweight="bold")
ax.barh(1, H(dt.datetime(2026, 11, 18, 20)) - H(dt.datetime(2026, 11, 17, 20, 15)), left=H(dt.datetime(2026, 11, 17, 20, 15)),
        color="#2f6f8f", alpha=0.75, height=0.5)
ax.text(H(dt.datetime(2026, 11, 17, 19)), 1, "the 24-hour close-out window", va="center", ha="right", color=INK, fontsize=11,
        fontweight="bold")
ax.axvline(H(dt.datetime(2026, 11, 18, 20)), color=INK, lw=1.5, ls="--")
days = [dt.datetime(2026, 11, d) for d in (17, 18, 19, 20, 21)]
ax.set_xticks([H(d) for d in days], [d.strftime("%b %d") for d in days])
ax.set_yticks([])
ax.set_ylim(0.4, 2.6)
ax.set_xlim(0, H(dt.datetime(2026, 11, 21)))
ax.set_xlabel("date (UTC); dashed line = end of the run")
ax.set_title("(b) Fir outage and the close-out window", loc="left")
fig.tight_layout()
png("notrun", fig)

# ready queue: parse section 1
q = grab(RQ, r"## 1\. Order of sending\n\n(.*?)\n\n(?:Entries 1 to 4|State on )")
QE = []
for line in q.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 5 or not cells[0].isdigit():
        continue
    QE.append(dict(order=int(cells[0]), entry=cells[1].replace("`", "").replace("eic_v1_2", "Filled-covariate tables:"), cluster=cells[2], waits=cells[3], cost=cells[4]))
if len(QE) < 19:   # entries 20 to 22 were added on 2026-11-18
    sys.exit(f"build_deck: {RQ} order table has {len(QE)} entries")


def cost_of(s):
    """The CPU-hours used for the bar: the largest 'about'/plain figure before any '+', summing '+' parts;
    a range 'a to b' takes a; 'not built' gives None."""
    if "not built" in s:
        return None
    s2 = re.sub(r"\(.*?\)", "", s)
    s2 = s2.split(" or ")[0]
    tot = 0.0
    for part in s2.split("+"):
        mm = re.findall(r"[0-9][0-9,]*\.?[0-9]*", part)
        if mm:
            tot += float(mm[0].replace(",", ""))
    return tot or None


def gclass(e):
    w, o = e["waits"].lower(), e["order"]
    if o <= 5:   # entries 1 to 5 were submitted on Nibi in the close-out window (run log, 2026-11-17)
        return "submitted in the close-out window"
    if "pi" in w:
        return "waits for a PI decision"
    if "fir return" in w or "queued" in w or e["cluster"].startswith("Fir"):
        return "waits for Fir"
    return "waits for an earlier entry"


GC = {"submitted in the close-out window": OK, "waits for Fir": "#2f6f8f", "waits for a PI decision": WAIT, "waits for an earlier entry": "#8a8d92"}
fig, ax = plt.subplots(figsize=(13.4, 6.0))
for i, e in enumerate(QE):
    c = cost_of(e["cost"])
    g = gclass(e)
    y = len(QE) - 1 - i
    nm = e["entry"]
    nm = nm if len(nm) <= 52 else nm[:51] + "."
    ax.text(0.8, y, f"{e['order']}. {nm}", ha="right", va="center", fontsize=10.2, color=INK)
    if c is None:
        ax.text(1.05, y, "no submit script yet", va="center", fontsize=9.5, color=MUTED, style="italic")
        continue
    ax.barh(y, c, left=1, color=GC[g], height=0.66, log=True)
    ax.text(c * 1.12 + 1, y, e["cost"] if len(e["cost"]) <= 40 else e["cost"][:39] + ".", va="center",
            fontsize=8.8, color=INK2)
ax.set_xscale("log")
ax.set_xlim(1, 4e5)
ax.set_ylim(-0.7, len(QE) - 0.3)
ax.set_yticks([])
ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ["0", "10", "100", "1,000", "10,000", "100,000"])
ax.set_xlabel("allocated CPU-hours (log scale; the bar starts at zero)")
ax.spines["left"].set_visible(False)
ax.legend(handles=[Patch(color=GC[k], label=k) for k in GC], loc="upper right", frameon=False, fontsize=11)
fig.tight_layout()
png("queue", fig)
tot = sum(cost_of(e["cost"]) or 0 for e in QE)
put("rq.n", str(len(QE)))
put("rq.now", str(sum(1 for e in QE if gclass(e) == "submitted in the close-out window")))
put("rq.noscript", str(sum(1 for e in QE if cost_of(e["cost"]) is None)))
put("rq.total", comma(int(round(tot, -2))))
cite("queue", RQ, RSTAT)

# PI decisions of 2026-11-17: the nine taken (decision record) and the two moved to the next session
pid = grab(PID, r"\| # \| Decision \| PI choice \| Consequence \|\n\|[-|]+\|\n(.*?)\n\n")
DEC = []
for line in pid.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 4 or not cells[0].isdigit():
        continue
    cons = cells[3].replace("`", "")
    cons = re.sub(r"\s*\(jobs? [0-9 to]+\)", "", cons)
    cons = re.sub(r"Submitted 2026-11-17: layers [0-9]+; baselines [0-9, ]+", "Submitted on Nibi on 2026-11-17; "
                  "the layers are done", cons)
    cons = re.sub(r"\s*\(ready queue entry ([0-9]+), DESIGN=full\)", r" (ready queue entry \1)", cons)
    DEC.append(dict(n=cells[0], d=cells[1], c=cells[2], k=cons))
if len(DEC) != 9:
    sys.exit(f"build_deck: {PID} decision table has {len(DEC)} rows")
mv = grab(DES, r'PI ruling of 2026-11-17: "(scaling is next session[^"]*)"\. On the data version: "([^"]*)"', 2)
MOVED = []
msec = grab(DES, r"## Decisions that the PI makes in the next session\n(.*?)(?:\n## |\Z)")
for line in msec.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 4 and cells[0] not in ("Decision", "") and not cells[0].startswith("---"):
        MOVED.append(dict(d=cells[0], r=cells[1].replace("`", "")))
if len(MOVED) != 2:
    sys.exit(f"build_deck: {DES} next-session table has {len(MOVED)} rows")
put("dec.n", str(len(DEC)))
put("dec.moved", str(len(MOVED)))
cite("dec", PID, DES, RQ)

# rescoring (pending slide)
rs = grab(DES, r"\*\*The run\.\*\* ([0-9]+) run-seeds of ([0-9]+) decisive arms, scored into `p256_sv` on Nibi "
               r"\(PI approved, about ([0-9,]+) allocated CPU-hours", 3)
put("rs.seeds", rs[0])
put("rs.arms", rs[1])
put("rs.cost", rs[2])
cite("pending", DES, RQ, CHKMD, RSTAT, PID)

# ---- the 256-region rescoring read (version 5) ---------------------------------------------------
# tools/onion_eval/apply_rescoring_rules.py applies rules 2.1 to 2.6 to the p256_sv arm summary and
# writes results/rescoring_p256/read_<stamp>/verdicts.json; run_status.json rescoring.read_dir names
# that folder. Every rescoring sentence of the deck is a string of verdicts.json (no hand edits).
if "read_dir" not in _r:
    sys.exit(f"build_deck: {RSTAT} has no rescoring.read_dir; run tools/onion_eval/apply_rescoring_rules.py")
VREL = _r["read_dir"] + "/verdicts.json"
V = json.loads(text(VREL))
P256 = _r["read_table"]
path(P256)
VT_ = V["text"]


def dk(s):
    """A generated sentence for the page: a minus before a digit becomes the minus sign."""
    return re.sub(r"(?<![\w.])-(?=[0-9])", "−", s)


put("rs.banner", dk(VT_["banner"]))
put("rs.h2", dk(VT_["h2"]))
put("rs.status", dk(VT_["status"]))
RULE_IDS = list(V["rules"])
for rid in RULE_IDS:
    r = V["rules"][rid]
    put(f"rs.r{rid}.head", dk(VT_[f"rule{rid}.head"]))
    none = all(not c["seeds_ok"] for c in r["cells"])
    put(f"rs.r{rid}", dk(r["verdict"].split(". ")[0].rstrip(".") + "." + (" " + r["pending_text"] if r.get("pending_text") else "") if r["decided"] else
                         "Not decided: no arm has finished." if none else
                         "Not decided: it waits for " + "; ".join(r["waiting"]) + "."))
put("vw.p256", dk(VT_["views"]))
put("loss.p256", dk(VT_["loss"]))
put("heat1.p256", dk(VT_["heat1"]))
put("heat2.p256", dk(VT_["heat2"]))
put("des.p256", dk(VT_["design"]))
put("test.p256", dk(VT_["eval_card"]))
put("sum.views", dk(VT_["sum_views"]))
put("sum.eval", dk(VT_["sum_eval"]))
put("sum.design", dk(VT_["sum_design"]))

# figure: per rule and model, arm minus base on the rule's deciding metric, with the bar
from matplotlib.transforms import blended_transform_factory
PANELS = [(rid, g) for rid in RULE_IDS for g in V["rules"][rid]["groups"]]
SHORT = {"two_step_native": "two-step, native base depth", "two_step_fixed": "two-step, fixed base depth"}
fig, axs = plt.subplots(4, 2, figsize=(13.4, 7.0))
fig.subplots_adjust(left=0.115, right=0.985, top=0.95, bottom=0.07, wspace=0.08, hspace=0.95)
for ax, (rid, g) in zip(axs.flat, PANELS):
    r = V["rules"][rid]
    k = r["figure_key"]
    cs = {c["model"]: c for c in r["cells"] if c["group"] == g and c["key"] == k}
    lim = max([0.1] + [max(abs(c["d"]), 2 * c["bar"]) * 1.15 for c in cs.values() if c["complete"]])
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, m in enumerate(MODELS):
        y = 2 - i
        c = cs[m]
        if not c["complete"]:
            ax.add_patch(Rectangle((-lim, y - 0.36), 3 * lim, 0.72, fc="#f1f1ee", ec="none"))
            ax.text(0.5 * lim, y, "not finished (" + c["cls"].replace("incomplete (", ""), ha="center",
                    va="center", fontsize=10, color=MUTED, style="italic")
            continue
        b = c["bar"]
        ax.add_patch(Rectangle((-2 * b, y - 0.36), 4 * b, 0.72, fc="#e9e9e5", ec="none"))
        ax.add_patch(Rectangle((-b, y - 0.36), 2 * b, 0.72, fc="#cfd0cb", ec="none"))
        col = {"within": GREY, "beyond": WAIT, "2x": BAD}[c["cls"]] if c["d"] < 0 else \
            {"within": GREY, "beyond": "#5c8fb0", "2x": "#2f6f8f"}[c["cls"]]
        ax.plot(c["d"], y, "o", color=col, ms=9, zorder=3)
        lab = s3(c["d"]) + {"within": " within", "beyond": " beyond", "2x": " beyond twice"}[c["cls"]]
        ax.text(1.08 * lim, y, lab, ha="left", va="center", fontsize=10, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-lim, 2 * lim)
    ax.set_xticks([t for t in ax.get_xticks() if -lim <= t <= lim])
    ax.axvline(lim, color=HAIR, lw=1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([2, 1, 0], [MODEL[m] for m in MODELS] if ax in axs[:, 0] else ["", "", ""], fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(f"Rule {rid}: {SHORT.get(g, V['groups'][g])}" + (" (decided)" if r["decided"] else " (not decided)"),
                 loc="left", fontsize=11.5, color=INK if r["decided"] else MUTED)
    ax.set_xlabel(f"arm minus base: {V['keys'][k]}", fontsize=9.5)
ax = axs.flat[len(PANELS)]
ax.axis("off")
ax.add_patch(Rectangle((0.0, 0.78), 0.07, 0.16, fc="#cfd0cb", transform=ax.transAxes, clip_on=False))
ax.text(0.10, 0.86, "within the seed range (larger of arm and base)", va="center", fontsize=10, transform=ax.transAxes)
ax.add_patch(Rectangle((0.0, 0.52), 0.07, 0.16, fc="#e9e9e5", transform=ax.transAxes, clip_on=False))
ax.text(0.10, 0.60, "within twice the seed range", va="center", fontsize=10, transform=ax.transAxes)
ax.text(0.0, 0.30, "dot: mean of the seeds; red or orange lower, blue higher, gray within", va="center",
        fontsize=10, transform=ax.transAxes, color=INK2)
ax.text(0.0, 0.02, f"256 regions, scores as of {V['as_of']}\nnot finished: fewer seeds than planned; no value shown", va="center",
        fontsize=10, transform=ax.transAxes, color=MUTED)
png("rescore", fig)
cite("rescore", VREL, P256, DES, RSTAT)
cite("views", VREL)
cite("losses", VREL)
cite("heat", VREL)
cite("design", VREL)
cite("evaltest", VREL)
cite("status", VREL)

# title
cite("title", TODO, PLAN24, DES)
cite("status", TODO, PLAN24, DES, OBJ, CHKMD)

# ==================================================================================================
data = dict(n=N, fig=FIG, dec=DEC, moved=MOVED, quote=list(mv), src=SRC)
read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {p.relative_to(VAULT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
if src.count("__SOURCES__") != 1 or src.count("__DATA__") != 1:
    sys.exit("build_deck: deck_src.html must hold exactly one __SOURCES__ and one __DATA__")
used_keys = set(re.findall(r'data-k="([^"]+)"', src))
missing = sorted(k for k in used_keys if k not in N)
if missing:
    sys.exit(f"build_deck: deck_src.html uses keys not in DATA.n: {missing}")
used_figs = set(re.findall(r'data-fig="([^"]+)"', src))
if used_figs - set(FIG):
    sys.exit(f"build_deck: deck_src.html uses figures not built: {sorted(used_figs - set(FIG))}")
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
size = len(out.encode("utf-8"))
print(f"build_deck: wrote index.html ({size / 1e6:.2f} MB) from {len(sources)} files; {len(FIG)} figures; "
      f"{len(N)} numbers; unused figures {sorted(set(FIG) - used_figs)}")
if size > 1_000_000:
    sys.exit("build_deck: index.html is over 1 MB")
