#!/usr/bin/env python3
"""Score a miniCANDI blind prediction on the two EIC boards (crux task t390, entry pipeline).

Input: a folder that `tools/minicandi/predict.py` wrote (`<pred>/counts/<eic_track>.h5`, groups
`mu` and `n`, one float16 dataset per chromosome on the floor(bp / 25) grid), and a covariates TSV
with one row per track (the target experiment's own metadata, see COV_COLUMNS). Per track:

  1. counts   the NB -> one 25 bp count track: the mean `mu` (default, `--mode plug_in`), or the
              NB itself for the expected -log10 p (`--mode expectation`, reads `n` too).
  2. raw      the count route of rung L6: `signal_from_counts.signal_track`, pileup `box_d`,
              ChIP-seq MACS2 with a control, DNase-seq `macs2_no_control`. The control is the
              covariates row's `control`: a `.h5` with group `mu` (a predicted control; labelled
              `predicted_control`) or a `.npz` of 25 bp overlap counts (`<chrom>_cnt`, a
              `ladder_probe.py starts` file; labelled `observed_control`), or `none` (DNase-seq).
              -> <out>/raw/<track>/<track>.npy (challenge .npy, raw ceil(bp / 25) grid)
  3. MOESM2   the raw track scored against the original 2019 truth (eic_score.ORIGINAL_TRUTH),
              the PI's headline board. -> <out>/scores/moesm2_<track>.tsv
  4. MOESM4   the raw track quantile normalized (QN) with peaks called on itself (rung L7:
              `ladder_probe.py callpeaks` thresh 2, max gap = read length, min length = d, cap
              500K ChIP-seq / 300K DNase-seq; then `qnorm` with Max's per-assay reference), scored
              against the qnorm truth (Zenodo 14736610). The pipeline's half-read overlap rule
              (rung L7o) needs reads, so a mean prediction cannot use it.
              -> <out>/peaks/<track>.bed, <out>/qn/<track>/<track>.npy, <out>/scores/moesm4_<track>.tsv
  5. placement (subcommand `collect`): per measure, the rank among the 25 published entrants on
              the experiment (chromosome set all23), and the organizers' combined rank of the entry
              added alone to the published field, over bootstraps 0-9.
              -> <out>/board_moesm2.tsv, <out>/board_moesm4.tsv, <out>/SUMMARY.md

Subcommands:
  run        steps 1-4 for one track (one Slurm job per track; `run_entry.sh submit`)
  collect    step 5 for every scored track
  make-test  (test) observed 25 bp overlap counts of the ladder probe -> predict.py's layout + a
             covariates TSV, so the pipeline can be checked against rungs L6 and L7
  check      (test) the pipeline's tracks and scores vs rungs L6 / L7 and their scores

Entrant scores: MOESM4 = our MOESM4-reproducing cache ENTRANTS_MOESM4 (`run_probe.sh` ONION_CQN);
MOESM2 = Max Libbrecht's sprint 04 pe scores ENTRANTS_MOESM2 (`run_moesm2.sh`, `moesm2.py`), which
reproduce MOESM2. Paths are Nibi paths.

Validation board (`--board validation` on `run` and `collect`): the 45 EIC validation experiments.
Step 2's raw track is scored against the organizers' validation bigWigs (VAL_TRUTH, the original
2019 processing, like MOESM2's truth); there is no QN truth for them, so step 4 is skipped. Placement
is against MOESM6 (the paper's Additional file 6, the round-1 validation leaderboard: 10 entries per
experiment, bootstraps 0-9, no all23 row): per measure the rank of the mean over bootstraps, and the
combined rank per bootstrap, over VAL_MEASURES (msevar is reported but not ranked: its round-1
weights are not reproduced). `collect` also writes nmse = mse(all23) / var(truth).
  val-cov    covariates of every validation track from the ONION tables (TABLES, FRAGLEN); the
             control is the control measurement's observed counts.h5 (store overlap rule)
  prep-val   bins the validation truth (and the validation-truth msevar weights) into the cache
  average    (test) the round-1 Average baseline of one validation track, from the training
             bigWigs; scored like an entry and checked against MOESM6's Average rows
  check-val  (test) the raw track vs the store's own MACS2 signal.h5 (same reads) and vs the truth
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_probe as LP  # noqa: E402  (sets up the candi stubs and tools/ on sys.path)
from onion_eval import eic_score as E  # noqa: E402

# ---------------------------------------------------------------------------- Nibi paths
SIZES = "/scratch/mforooz/onion/eic/truth/se_unnormalized/C19M22.bigwig"   # GRCh38 chrom sizes
QN_REF = "/scratch/maxwl/epi_imputation/s04/qnorm/ref"                      # ref_M##.npz (Max s04)
BOARDS = {
    # board: truth bigWigs, cache whose truth/ and var/ we link, entrant scores
    "moesm2": dict(truth=E.ORIGINAL_TRUTH,
                   cache=Path("/scratch/mforooz/eicx/moesm2/score_cache/pe"),
                   published=E.DEFAULT_PUBLISHED_MOESM2, kind="raw"),
    "moesm4": dict(truth="/scratch/mforooz/onion/eic/truth/qnorm",
                   cache=Path("/scratch/mforooz/onion/eic/score_cache/qnorm"),
                   published=E.DEFAULT_PUBLISHED, kind="qn"),
}
SYN = "/scratch/maxwl/epi_imputation/data/synapse"                          # Max's Synapse copy
VAL_TRUTH = f"{SYN}/validation_data"                                        # 45 C##M##.bigwig + metadata
VAL_TRAIN = f"{SYN}/training_data"                                          # round 1's training set
MOESM6 = "/project/def-maxwl/mforooz/ONION/eic/published/13059_2023_2915_MOESM6_ESM.csv"
MOESM6_SHA256 = "151a5c3c5f18d371bc50d33cf6af1ba4b0b795602c52d10d6936c235dc53c1cf"
BOARDS["validation"] = dict(truth=VAL_TRUTH, cache=Path("/scratch/mforooz/eicx/val/score_cache"),
                            published=MOESM6, kind="raw")
# msevar on the validation board: no weight vector we tried reproduces MOESM6's msevar for the
# round-1 Average (validation truth, training, training + validation: 7-67% off; the other eight
# measures agree to <= 1.3e-6, results/t390/validation_dryrun). It is reported, never ranked.
VAL_MEASURES = [m for m in E.MEASURES9 if m != "msevar"]
BOARD_SETS = {"blind": ("moesm2", "moesm4"), "validation": ("validation",)}
TABLES = "/project/def-maxwl/mforooz/ONION/tables/eic_v1_2/measurements.tsv"
FRAGLEN = "/project/def-maxwl/mforooz/ONION/tables/eic_v1_fraglen.tsv"
ENTRANTS_MOESM4 = Path("/scratch/mforooz/onion/eic/score_cache/qnorm/scores")      # team_<tid>/<exp>.tsv
ENTRANTS_MOESM2 = Path("/project/def-maxwl/maxwl/epi_imputation/s04/scores/pe")    # <slug>/<exp>.csv
SLUG = {tid: re.sub(r"[^A-Za-z0-9_.-]", "_", name) for tid, name, _ in E.TEAMS}

COV_COLUMNS = ["track", "route", "depth_reads", "read_length", "frag_len", "control",
               "control_depth_reads", "control_read_length"]
ROUTE_CLASS = {"chip": "narrow_histone", "dnase": "DNase-seq"}   # one ChIP route for both classes
PEAK_CAP = {"chip": 500000, "dnase": 300000}                      # the pipelines' caps

# ---------------------------------------------------------------------------- test fixture
LADDER = Path("/scratch/mforooz/eicx/ladder")
TEST = {   # experiment: treatment starts, control starts or none, d, route (run_probe.sh conf)
    "C19M22": ("ENCFF748TKZ.merged.srt.nodup.30M", "ENCFF433TZR.merged.srt.nodup", 205, "chip"),
    "C40M17": ("ENCFF581NOL.merged.srt.nodup.30M", "ENCFF337JNL.merged.srt.nodup", 200, "chip"),
    "C12M02": ("ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M", "none", 150, "dnase"),
}
REF_L6 = LADDER / "rungs/L6"                       # <exp>.npy
REF_L7 = LADDER / "rungs/L7"
REF_SCORES = {"moesm2": Path("/scratch/mforooz/eicx/moesm2/scores/L6.tsv"),
              "moesm4": LADDER / "scores/qn/L7.tsv"}

log = LP.log


def read_cov(path, track):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["track"] == track]
    if len(rows) != 1:
        raise SystemExit(f"{path}: {len(rows)} rows for track {track} (need 1)")
    r = rows[0]
    if r["route"] not in ROUTE_CLASS:
        raise SystemExit(f"{track}: route {r['route']!r} not in {sorted(ROUTE_CLASS)}")
    return r


def _control(spec, chrom):
    """(25 bp control counts float64, lambda label) from a .npz (observed), an .h5 with group `mu`
    (predicted) or a store counts.h5 with one dataset per chromosome at the root (observed)."""
    if spec.endswith(".npz"):
        with np.load(spec) as z:
            return z[f"{chrom}_cnt"].astype(np.float64), "observed_control"
    import h5py
    with h5py.File(spec, "r") as f:
        if "mu" in f:
            return f["mu"][chrom][:].astype(np.float64), "predicted_control"
        return f[chrom][:].astype(np.float64), "observed_control"


# ---------------------------------------------------------------------------- steps 1-2
def convert(pred, cov, out, mode, eps):
    """NB -> one -log10 p track on the challenge grid (steps 1 and 2)."""
    import h5py
    from onion_eval import signal_from_counts as SFC

    t0 = time.time()
    sizes = LP.chrom_sizes(SIZES)
    route = cov["route"]
    c = {"depth_reads": cov["depth_reads"], "read_length": cov["read_length"]}
    if route == "chip":
        c.update(frag_len=cov["frag_len"], control_depth_reads=cov["control_depth_reads"],
                 control_read_length=cov["control_read_length"],
                 control_is_predicted=_control(cov["control"], "chr21")[1] == "predicted_control")
    elif cov["control"] not in ("", "none"):
        raise SystemExit(f"{cov['track']}: a {route} track has no control; got {cov['control']}")
    h5 = Path(pred) / "counts" / f"{cov['track']}.h5"
    res, src, dtypes = {}, None, set()
    with h5py.File(h5, "r") as f:
        for ch in LP.CHROMS:
            dtypes.add(str(f["mu"][ch].dtype))
            mu = f["mu"][ch][:].astype(np.float64) + eps      # plug_in asks mu > 0 (as rung L6)
            n = None
            if mode == "expectation":
                n = np.maximum(f["n"][ch][:].astype(np.float64), 1e-6)
            ctl = None
            if route == "chip":
                ctl, _ = _control(cov["control"], ch)
            sig, src = SFC.signal_track(mu, n, ROUTE_CLASS[route], ctl, c, mode=mode,
                                        pileup="box_d", dnase_route="macs2_no_control")
            y = np.zeros(LP.ngrid(sizes[ch]), np.float32)
            y[: sig.shape[0]] = sig[: y.shape[0]]
            res[ch] = y
    LP.save_track(out, res)
    LP.write_json(str(out) + ".json", dict(pred=str(h5), counts_dtype=sorted(dtypes), covariates=c,
                                           route=route, control=cov["control"], lambda_source=src,
                                           mode=mode, pileup="box_d", eps=eps,
                                           seconds=time.time() - t0))
    log(f"raw {cov['track']}: lambda source {src}, {time.time() - t0:.0f}s")


def qn(raw, cov, peaks, out):
    """Rung L7: peaks called on the track itself, then Max's quantile normalization (step 4)."""
    track = cov["track"]
    LP.cmd_callpeaks(argparse.Namespace(x=str(raw), out=str(peaks), thresh=2.0,
                                        max_gap=float(cov["read_length"]),
                                        min_len=float(cov["frag_len"]),
                                        cap=PEAK_CAP[cov["route"]]))
    LP.cmd_qnorm(argparse.Namespace(x=str(raw), mask=str(peaks), ref=f"{QN_REF}/ref_{track[3:]}.npz",
                                    out=str(out), verify="", sizes_from=SIZES))


def board_cache(out, board):
    """<out>/score_cache/<board>, truth/ and var/ linked to the board's existing cache."""
    c = Path(out) / "score_cache" / board
    c.mkdir(parents=True, exist_ok=True)
    for sub in ("truth", "var"):
        if not (c / sub).exists():
            try:
                os.symlink(BOARDS[board]["cache"] / sub, c / sub)
            except FileExistsError:                                   # a parallel job made it
                pass
    return c


def score(out, board, track, path):
    """Nine measures of one track on one board -> <out>/scores/<board>_<track>.tsv."""
    cache = board_cache(out, board)
    key = f"entry_{board}_{track}"
    shutil.rmtree(cache / "scores" / key, ignore_errors=True)      # never reuse an old score
    E.prepare_truth(BOARDS[board]["truth"], cache, 1)
    E._run_scoring(BOARDS[board]["truth"], cache, {track: [(key, f"entry_{board}", "", str(path))]},
                   1, (0, 1))
    rows = E.read_score_rows(cache / "scores" / key / f"{track}.tsv")
    dst = Path(out) / "scores" / f"{board}_{track}.tsv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(E.ROW_KEYS)
        for r in rows:
            w.writerow([f"entry_{board}", "", r["experiment"], r["cell"], r["assay"], r["chrom_set"]]
                       + [repr(r[m]) for m in E.MEASURES9])
    log(f"{board} {track}: wrote {dst}")


def cmd_run(a):
    cov = read_cov(a.cov, a.track)
    out = Path(a.out)
    raw = out / "raw" / a.track / f"{a.track}.npy"
    qnp = out / "qn" / a.track / f"{a.track}.npy"
    convert(a.pred, cov, raw, a.mode, a.eps)
    if a.board == "validation":                     # no QN truth exists for the validation set
        score(out, "validation", a.track, raw)
        return 0
    score(out, "moesm2", a.track, raw)
    (out / "peaks").mkdir(parents=True, exist_ok=True)
    qn(raw, cov, out / "peaks" / f"{a.track}.bed", qnp)
    score(out, "moesm4", a.track, qnp)
    return 0


# ---------------------------------------------------------------------------- step 5
def _entrants_all23(board, exp):
    """{team_id: {measure: value}} of the published entrants on one experiment, all23."""
    out = {}
    for tid, _, _ in E.TEAMS:
        if board == "moesm4":
            p = ENTRANTS_MOESM4 / f"team_{tid}" / f"{exp}.tsv"
            if p.exists():
                out[tid] = [r for r in E.read_score_rows(p) if r["chrom_set"] == "all23"][0]
        else:
            p = ENTRANTS_MOESM2 / SLUG[tid] / f"{exp}.csv"
            if p.exists():
                with open(p) as fh:
                    for r in csv.DictReader(fh):
                        if r["chrom_set"] == "all23":
                            out[tid] = {m: float(r[m]) for m in E.MEASURES9}
    return out


def _g(x):
    return f"{x:.10g}"


def cmd_collect(a):
    out = Path(a.out)
    if a.board == "validation":
        return collect_validation(out)
    summary = ["# EIC entry scores", "",
               f"Folder `{out}`. Cell = value (rank among the published entrants on that experiment, "
               "chromosome set all23; 1 = better than or equal to every entrant). Combined = the "
               "organizers' per-experiment rank of the entry added alone to the published field, "
               "mean over bootstraps 0-9 (min-max).", ""]
    for board in ("moesm2", "moesm4"):
        pub = E.read_published(BOARDS[board]["published"])
        pmap = {}
        for r in pub:
            pmap.setdefault((r["cell"] + r["assay"], r["bootstrap_id"]), []).append(r)
        files = sorted((out / "scores").glob(f"{board}_C*M*.tsv"))
        rows, md = [], []
        for f in files:
            sr = E.read_score_rows(f)
            exp = sr[0]["experiment"]
            mine = [r for r in sr if r["chrom_set"] == "all23"][0]
            ent = _entrants_all23(board, exp)
            cells = []
            for m in E.MEASURES9:
                ev = np.array([x[m] for x in ent.values()])
                hib = m in E.HIGHER_IS_BETTER
                better = int(np.sum(ev > mine[m])) if hib else int(np.sum(ev < mine[m]))
                rows.append([board, exp, BOARDS[board]["kind"], m, _g(mine[m]), better + 1, len(ev),
                             _g(ev.max() if hib else ev.min()), _g(np.median(ev)),
                             _g(ev.min() if hib else ev.max()), ""])
                cells.append(f"{mine[m]:.4g} ({better + 1})")
            boots = {int(r["chrom_set"][4:]): r for r in sr if r["chrom_set"].startswith("boot")}
            cr = [float(E.combined_ranks(pmap[(exp, b)] + [boots[b]])[-1]) for b in range(10)]
            n_field = len(pmap[(exp, 0)]) + 1
            rows.append([board, exp, BOARDS[board]["kind"], "combined_rank", _g(np.mean(cr)), "",
                         n_field, "", "", "", f"min {min(cr):g} max {max(cr):g} over bootstraps 0-9"])
            md.append(f"| {exp} | " + " | ".join(cells)
                      + f" | {np.mean(cr):.3g} ({min(cr):g}-{max(cr):g}) of {n_field} |")
        with open(out / f"board_{board}.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["board", "experiment", "track", "measure", "value", "rank_among_entrants",
                        "n_entrants", "best_entrant", "median_entrant", "worst_entrant", "note"])
            w.writerows(rows)
        title = {"moesm2": "MOESM2 (raw -log10 p vs the original 2019 truth; headline)",
                 "moesm4": "MOESM4 (QN with own peaks vs the qnorm truth)"}[board]
        summary += [f"## {title}", "", "| experiment | " + " | ".join(E.MEASURES9) + " | combined |",
                    "|---" * (len(E.MEASURES9) + 2) + "|"] + md + [""]
    srcs = sorted({json.loads(p.read_text())["lambda_source"]
                   for p in (out / "raw").glob("*/*.npy.json")})
    summary += [f"Lambda sources: {', '.join(srcs)}. QN peaks: rung L7 threshold caller on the entry "
                "itself (the half-read overlap rule needs reads and is not available for a mean "
                "prediction).", ""]
    (out / "SUMMARY.md").write_text("\n".join(summary))
    log(f"wrote {out}/board_moesm2.tsv, board_moesm4.tsv, SUMMARY.md")
    return 0


# ---------------------------------------------------------------------------- validation board
def read_moesm6():
    """MOESM6 bootstrap rows (sha256-pinned): {experiment: {team: {bootstrap: row}}}."""
    if E.sha256_file(MOESM6) != MOESM6_SHA256:
        raise SystemExit(f"{MOESM6}: sha256 is not MOESM6's {MOESM6_SHA256}")
    by = {}
    with open(MOESM6) as fh:
        for r in csv.DictReader(fh):
            d = dict(team_id=r["team"], cell=r["cell_id"], assay=r["assay_id"],
                     bootstrap_id=int(r["bootstraip_id"]))        # sic, the column's own spelling
            d.update({m: float(r[m]) for m in E.MEASURES9})
            by.setdefault(r["cell_id"] + r["assay_id"], {}).setdefault(r["team"], {})[d["bootstrap_id"]] = d
    return by


def collect_validation(out):
    """Placement among MOESM6's 10 round-1 entries, nmse, SUMMARY.md (the validation board)."""
    pub = read_moesm6()
    cache = BOARDS["validation"]["cache"]
    rows, md, mine_rows = [], [], []
    for f in sorted((out / "scores").glob("validation_C*M*.tsv")):
        sr = E.read_score_rows(f)
        exp = sr[0]["experiment"]
        a23 = [r for r in sr if r["chrom_set"] == "all23"][0]
        boots = {int(r["chrom_set"][4:]): r for r in sr if r["chrom_set"].startswith("boot")}
        y = np.concatenate(list(E.load_truth(cache, exp).values())).astype(np.float64)
        nmse = a23["mse"] / float(y.var())
        rows.append(["validation", exp, "raw", "nmse", _g(nmse), "", "", "", "", "",
                     "mse(all23) / var(truth), blacklist bins deleted"])
        ent = pub[exp]
        cells = [f"{nmse:.4g}"]
        for m in E.MEASURES9:
            mv = float(np.mean([boots[b][m] for b in range(10)]))
            ev = np.array([np.mean([t[b][m] for b in range(10)]) for t in ent.values()])
            hib = m in E.HIGHER_IS_BETTER
            better = int(np.sum(ev > mv)) if hib else int(np.sum(ev < mv))
            if m not in VAL_MEASURES:
                rows.append(["validation", exp, "raw", m, _g(mv), "", len(ev), "", _g(np.median(ev)), "",
                             "not ranked: MOESM6's msevar weights are not reproduced"])
                cells.append(f"{mv:.4g} (-)")
                continue
            rows.append(["validation", exp, "raw", m, _g(mv), better + 1, len(ev),
                         _g(ev.max() if hib else ev.min()), _g(np.median(ev)),
                         _g(ev.min() if hib else ev.max()), "mean over bootstraps 0-9 (MOESM6 has no all23)"])
            cells.append(f"{mv:.4g} ({better + 1})")
        cr = [float(E.combined_ranks([t[b] for t in ent.values()] + [boots[b]], VAL_MEASURES)[-1])
              for b in range(10)]
        rows.append(["validation", exp, "raw", "combined_rank", _g(np.mean(cr)), "", len(ent) + 1, "", "",
                     "", f"min {min(cr):g} max {max(cr):g} over bootstraps 0-9"])
        md.append(f"| {exp} | " + " | ".join(cells) + f" | {np.mean(cr):.3g} ({min(cr):g}-{max(cr):g}) of {len(ent) + 1} |")
        mine_rows += [dict(boots[b], team_id="entry", cell=exp[:3], assay=exp[3:], bootstrap_id=b)
                      for b in range(10)]
    done = {r["cell"] + r["assay"] for r in mine_rows}
    if done == set(pub):                     # the organizers' leaderboard needs every experiment
        allrows = [r for e in pub.values() for t in e.values() for r in t.values()] + mine_rows
        pos = [d for d in E.global_ranks(allrows, VAL_MEASURES) if d["team_id"] == "entry"][0]
        rows.append(["validation", "all", "raw", "leaderboard_position", pos["position"], "", 11, "", "",
                     "", f"rank_stat {pos['rank_stat']:g} score_mean {pos['score_mean']:.6g}"])
        lead = (f"Leaderboard position among the 10 round-1 entries plus this one: {pos['position']} of 11 "
                f"(rank_stat {pos['rank_stat']:g}).")
    else:
        lead = (f"No leaderboard position: {len(done)} of {len(pub)} validation experiments scored "
                "(the organizers' statistic counts a missing experiment as the worst).")
    with open(out / "board_validation.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["board", "experiment", "track", "measure", "value", "rank_among_entrants",
                    "n_entrants", "best_entrant", "median_entrant", "worst_entrant", "note"])
        w.writerows(rows)
    srcs = sorted({json.loads(p.read_text())["lambda_source"] for p in (out / "raw").glob("*/*.npy.json")})
    summary = ["# EIC validation scores", "",
               f"Folder `{out}`. Truth: `{VAL_TRUTH}` (original 2019 processing). Cell = mean over "
               "bootstraps 0-9 (rank among MOESM6's 10 round-1 entries on that experiment; 1 = better than "
               "or equal to every entry). nmse = mse(all23) / var(truth). Combined = the organizers' "
               "per-experiment rank of the entry added alone to MOESM6's field, mean over bootstraps "
               "(min-max), over the eight measures other than msevar (its weights are not reproduced).", "", lead, "",
               "| experiment | nmse | " + " | ".join(E.MEASURES9) + " | combined |",
               "|---" * (len(E.MEASURES9) + 3) + "|"] + md + ["", f"Lambda sources: {', '.join(srcs)}.", ""]
    (out / "SUMMARY.md").write_text("\n".join(summary))
    log(f"wrote {out}/board_validation.tsv, SUMMARY.md")
    return 0


def cmd_val_cov(a):
    """Covariates of every validation track (pooled base measurement; uniform layer, else native)."""
    from onion_eval import signal_from_counts as SFC
    with open(f"{VAL_TRUTH}/metadata_validation_data.tsv") as fh:
        tracks = sorted(r["cell_type_id"] + r["mark_id"] for r in csv.DictReader(fh, delimiter="\t"))
    with open(TABLES) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    byid = {r["measurement_id"]: r for r in rows}
    fl = SFC.load_fraglen_table(FRAGLEN)
    base = {}
    for r in rows:
        if r["unit"] == "pooled" and not r["arm_knob"] and r["measurement_id"].endswith(".base"):
            base.setdefault(r["eic_track"], {})[r["layer"]] = r
    cols = COV_COLUMNS + ["measurement_id", "layer", "assay_class", "frag_len_source", "observed_counts",
                          "observed_signal"]
    good, miss = [], []
    for t in tracks:
        r = base.get(t, {}).get("uniform") or base.get(t, {}).get("native")
        if r is None:
            miss.append((t, "no pooled base measurement in the tables"))
            continue
        cls = r["assay_class"]
        route = "dnase" if "dnase" in cls else ("chip" if cls.endswith("_histone") else "")
        mdir = Path(r["counts_path"][3:]).parent
        row = dict(track=t, route=route, depth_reads=r["depth_reads"], read_length=r["read_length"],
                   frag_len=r["frag_len"], control="none", control_depth_reads="", control_read_length="",
                   measurement_id=r["measurement_id"], layer=r["layer"], assay_class=cls,
                   frag_len_source="", observed_counts=r["counts_path"][3:],
                   observed_signal=str(mdir / "signal.h5") if r["has_signal"] == "1" else "")
        why = ""
        if not route:
            why = f"assay class {cls} has no route"
        elif not r["counts_path"].startswith("h5:"):
            why = f"counts_path {r['counts_path']} is not an h5: path"
        elif route == "chip":
            rr, src, note = SFC.with_fraglen(r, fl)
            ctl = byid.get(r["control_measurement_id"])
            if src == "none":
                why = note
            elif ctl is None or not ctl["counts_path"].startswith("h5:"):
                why = f"control {r['control_measurement_id'] or 'none'} has no counts.h5"
            else:
                row.update(frag_len=rr["frag_len"], frag_len_source=f"{src} {note}".strip(),
                           control=ctl["counts_path"][3:], control_depth_reads=ctl["depth_reads"],
                           control_read_length=ctl["read_length"])
        if why:
            miss.append((t, why))
        else:
            good.append(row)
    with open(a.cov, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(good)
    with open(str(a.cov) + ".missing.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["track", "reason"])
        w.writerows(miss)
    log(f"{len(good)} of {len(tracks)} validation tracks -> {a.cov}; {len(miss)} missing -> {a.cov}.missing.tsv")
    return 0


def cmd_prep_val(a):
    """Bin the 45 validation truth bigWigs into the board's cache (msevar weights: validation truth)."""
    E.prepare_truth(VAL_TRUTH, BOARDS["validation"]["cache"], a.workers)
    return 0


def cmd_average(a):
    """Round-1 Average baseline of one validation track: the per-bin mean over the training bigWigs
    of its assay. Scored like an entry (<out>/scores/validation_<track>.tsv), then compared with
    MOESM6's Average rows; msevar is also computed with three candidate weight vectors."""
    exp, out = a.track, Path(a.out)
    srcs = sorted(Path(VAL_TRAIN).glob(f"C*{exp[3:]}.bigwig"))
    bl = E.load_blacklist_bins(E.asset_dir())
    tr = {c: [] for c in E.CHROMS}
    for p in srcs:
        y = E.bin_bigwig(p, E.CHROMS)
        for c in E.CHROMS:
            tr[c].append(y[c].astype(np.float64))
        log(f"binned {p.name}")
    npy = out / "average" / f"{exp}.npy"
    L = {c: min(x.shape[0] for x in tr[c]) for c in E.CHROMS}
    LP.save_track(npy, {c: np.mean([x[: L[c]] for x in tr[c]], axis=0) for c in E.CHROMS})
    score(out, "validation", exp, npy)
    cache = BOARDS["validation"]["cache"]
    truth = E.load_truth(cache, exp)
    pred, T = E.align(E.load_prediction(npy, E.CHROMS, bl), truth, E.CHROMS)
    trbl = {c: [E.apply_blacklist(x, bl.get(c)) for x in tr[c]] for c in E.CHROMS}
    vals = sorted(p.stem for p in Path(VAL_TRUTH).glob(f"C*{exp[3:]}.bigwig"))
    vtr = {c: [] for c in E.CHROMS}
    for v in vals:
        for c, x in E.load_truth(cache, v).items():
            vtr[c].append(x.astype(np.float64))

    def var_of(members):
        if len(members[E.CHROMS[0]]) < 2:
            return None
        n = {c: min(len(T[c]), *(x.shape[0] for x in members[c])) for c in E.CHROMS}
        return {c: E.two_pass_var([x[: n[c]] for x in members[c]]) for c in E.CHROMS}

    cands = {"validation_truth": var_of(vtr), "training": var_of(trbl),
             "training_plus_validation": var_of({c: trbl[c] + vtr[c] for c in E.CHROMS})}
    mine = {int(r["chrom_set"][4:]): r for r in E.read_score_rows(out / "scores" / f"validation_{exp}.tsv")
            if r["chrom_set"].startswith("boot")}
    theirs = read_moesm6()[exp]["Average"]
    res = []
    for m in E.MEASURES9:
        rel = [abs(mine[b][m] - theirs[b][m]) / max(abs(theirs[b][m]), 1e-12) for b in range(10)]
        res.append([exp, m, "board scorer", f"max rel {max(rel):.3g}", f"median rel {np.median(rel):.3g}"])
    for name, v in cands.items():
        if v is None:
            res.append([exp, "msevar", f"weights {name}", "fewer than 2 members", ""])
            continue
        rel = []
        for b in range(10):
            cs = sorted(E.BOOTSTRAP_CHROM[b])
            num = sum(float(((T[c][: len(v[c])] - pred[c][: len(v[c])]) ** 2) @ v[c]) for c in cs)
            den = sum(float(v[c].sum()) for c in cs)
            rel.append(abs(num / den - theirs[b]["msevar"]) / abs(theirs[b]["msevar"]))
        res.append([exp, "msevar", f"weights {name}", f"max rel {max(rel):.3g}", f"median rel {np.median(rel):.3g}"])
    with open(out / f"check_average_{exp}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["experiment", "measure", "against MOESM6 Average with", "result", "detail"])
        w.writerows(res)
    for r in res:
        log("\t".join(r))
    LP.write_json(out / f"average_{exp}.json", dict(sources=[p.name for p in srcs], validation_members=vals))
    return 0


def cmd_check_val(a):
    """Raw track vs the store's MACS2 signal.h5 of the same reads, and both vs the truth (nMSE)."""
    import h5py
    out, t = Path(a.out), a.track
    cov = read_cov(a.cov, t)
    sizes = LP.chrom_sizes(SIZES)
    sig, res = out / "observed_signal" / f"{t}.npy", {}
    with h5py.File(cov["observed_signal"], "r") as f:
        for c in LP.CHROMS:
            y = np.zeros(LP.ngrid(sizes[c]), np.float32)
            s = f[c][:]
            y[: min(len(s), len(y))] = s[: len(y)]
            res[c] = y
    LP.save_track(sig, res)
    raw, truth = out / "raw" / t / f"{t}.npy", f"{VAL_TRUTH}/{t}.bigwig"
    rows = []
    for name, x, y in (("raw_vs_observed_signal", raw, sig), ("raw_vs_truth", raw, truth),
                       ("observed_signal_vs_truth", sig, truth)):
        j = out / "compare" / f"{t}_{name}.json"
        LP.cmd_compare(argparse.Namespace(a=str(x), b=str(y), name=name, out=str(j)))
        d = json.loads(j.read_text())
        rows.append([t, name, _g(d["nmse"]), _g(d["nmse_asinh"]), _g(d["pearson"]), _g(d["max_abs_diff"]),
                     d["n_bins"]])
    with open(out / f"check_{t}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["experiment", "pair", "nmse", "nmse_asinh", "pearson", "max_abs_diff", "n_bins"])
        w.writerows(rows)
    return 0


# ---------------------------------------------------------------------------- test
def cmd_make_test(a):
    """Observed 25 bp overlap counts in predict.py's layout, and the covariates row (rung L6's)."""
    import h5py

    pred = Path(a.pred)
    (pred / "counts").mkdir(parents=True, exist_ok=True)
    if a.from_cov:                    # a validation track: its store counts.h5 and its val-cov row
        row = read_cov(a.from_cov, a.track)
        with h5py.File(row["observed_counts"], "r") as f:
            cnt = {c: f[c][:].astype(np.float64) for c in LP.CHROMS}
        exact16 = all(np.array_equal(cnt[c].astype(np.float16).astype(np.float64), cnt[c]) for c in LP.CHROMS)
        dt = np.float16 if exact16 else np.float32
        with h5py.File(pred / "counts" / f"{a.track}.h5", "w") as f:
            f.attrs.update({"eic_track": a.track, "note": "observed store counts (t390 validation test)"})
            for c in LP.CHROMS:
                f.require_group("mu").create_dataset(c, data=cnt[c].astype(dt))
                f.require_group("n").create_dataset(c, data=np.full(cnt[c].shape, 1e4, dt))
        with open(a.cov, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(row), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerow(row)
        log(f"test input {a.track} ({np.dtype(dt).name}) from {row['observed_counts']}, covariates {a.cov}")
        return 0
    if a.track not in TEST:
        raise SystemExit(f"{a.track}: not a blind test track ({sorted(TEST)}); give --from-cov")
    trt, ctl, d, route = TEST[a.track]
    it = json.load(open(LADDER / "starts" / f"{trt}.json"))
    h5 = pred / "counts" / f"{a.track}.h5"
    exact16 = True
    with np.load(LADDER / "starts" / f"{trt}.npz") as z:
        cnt = {c: z[f"{c}_cnt"].astype(np.float64) for c in LP.CHROMS}
    for c in LP.CHROMS:
        exact16 &= bool(np.array_equal(cnt[c].astype(np.float16).astype(np.float64), cnt[c]))
    dt = np.float16 if exact16 else np.float32       # predict.py writes float16
    with h5py.File(h5, "w") as f:
        f.attrs.update({"eic_track": a.track, "note": "observed overlap counts (t390 entry test)"})
        for c in LP.CHROMS:
            f.require_group("mu").create_dataset(c, data=cnt[c].astype(dt))
            f.require_group("n").create_dataset(c, data=np.full(cnt[c].shape, 1e4, dt))
    row = dict(track=a.track, route=route, depth_reads=repr(it["n_lines"]),
               read_length=repr(it["mean_read_length"]), frag_len=repr(float(d)), control="none",
               control_depth_reads="", control_read_length="")
    if ctl != "none":
        ic = json.load(open(LADDER / "starts" / f"{ctl}.json"))
        row.update(control=str(LADDER / "starts" / f"{ctl}.npz"),
                   control_depth_reads=repr(ic["n_lines"]),
                   control_read_length=repr(ic["mean_read_length"]))
    cov = Path(a.cov)
    with open(cov, "w", newline="") as fh:
        w = csv.DictWriter(fh, COV_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerow(row)
    log(f"test input {h5} ({np.dtype(dt).name}), covariates {cov}")
    return 0


def _arrdiff(x, y):
    A, B = LP.load_track_raw(x), LP.load_track_raw(y)
    nd = tot = 0
    mx = 0.0
    for c in LP.CHROMS:
        dd = np.abs(A[c].astype(np.float64) - B[c].astype(np.float64))
        mx, nd, tot = max(mx, float(dd.max())), nd + int((dd > 0).sum()), tot + dd.size
    return nd, tot, mx


def cmd_check(a):
    """Pipeline vs the ladder: raw = rung L6, qn = rung L7 (arrays), and their board scores."""
    out, t = Path(a.out), a.track
    ref_l6, ref_l7 = Path(a.ref_l6), Path(a.ref_l7)
    ref_scores = {b: Path(p) for b, p in (("moesm2", a.ref_moesm2), ("moesm4", a.ref_moesm4)) if p != "none"}
    rows = []
    for name, mine, ref in (("raw vs rung L6 (array)", out / "raw" / t / f"{t}.npy", ref_l6 / f"{t}.npy"),
                            ("qn vs rung L7 (array)", out / "qn" / t / f"{t}.npy", ref_l7 / f"{t}.npy")):
        nd, tot, mx = _arrdiff(mine, ref)
        rows.append([t, name, f"{tot} bins", f"{nd} differ", f"max abs {mx:.3g}", nd == 0])
    for board, ref in ref_scores.items():
        mine = {r["chrom_set"]: r for r in E.read_score_rows(out / "scores" / f"{board}_{t}.tsv")}
        theirs = {r["chrom_set"]: r for r in E.read_score_rows(ref) if r["experiment"] == t}
        rel = [abs(mine[s][m] - theirs[s][m]) / max(abs(theirs[s][m]), 1e-12)
               for s in theirs for m in E.MEASURES9]
        what = {"moesm2": "raw MOESM2 scores vs rung L6 MOESM2 scores",
                "moesm4": "qn MOESM4 scores vs probe-1 rung L7 qn scores"}[board]
        rows.append([t, what, f"{len(rel)} values", f"max rel {max(rel):.3g}",
                     f"median rel {float(np.median(rel)):.3g}", max(rel) <= 1e-9])
    with open(out / f"check_{t}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["experiment", "check", "n", "result", "detail", "pass"])
        w.writerows(rows)
    for r in rows:
        log("\t".join(map(str, r)))
    return 0 if all(r[-1] for r in rows) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run", help="steps 1-4 for one track")
    p.add_argument("--pred", required=True, help="predict.py --out folder (holds counts/)")
    p.add_argument("--cov", required=True, help=f"covariates TSV: {', '.join(COV_COLUMNS)}")
    p.add_argument("--track", required=True, help="C##M##")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="plug_in", choices=("plug_in", "expectation"),
                   help="plug_in: score of the mean (default); expectation: expected -log10 p")
    p.add_argument("--eps", type=float, default=1e-6, help="added to mu (rung L6's eps)")
    p.add_argument("--board", default="blind", choices=sorted(BOARD_SETS),
                   help="blind: MOESM2 + MOESM4 (default); validation: the validation truth only")
    p.set_defaults(fn=cmd_run)
    p = sp.add_parser("collect", help="placement on the board set, SUMMARY.md")
    p.add_argument("--out", required=True)
    p.add_argument("--board", default="blind", choices=sorted(BOARD_SETS))
    p.set_defaults(fn=cmd_collect)
    p = sp.add_parser("make-test", help="observed counts as a prediction (test)")
    p.add_argument("--track", required=True)
    p.add_argument("--pred", required=True); p.add_argument("--cov", required=True)
    p.add_argument("--from-cov", default="", help="a val-cov TSV: take the track's store counts.h5")
    p.set_defaults(fn=cmd_make_test)
    p = sp.add_parser("check", help="pipeline vs rungs L6 / L7 (test)")
    p.add_argument("--track", required=True, choices=sorted(TEST)); p.add_argument("--out", required=True)
    # references (t390 span_fix): default the probe-1 rungs; a folder of <exp>.npy, a score TSV or none
    p.add_argument("--ref-l6", default=str(REF_L6)); p.add_argument("--ref-l7", default=str(REF_L7))
    p.add_argument("--ref-moesm2", default=str(REF_SCORES["moesm2"]), help="score TSV, or none")
    p.add_argument("--ref-moesm4", default=str(REF_SCORES["moesm4"]), help="score TSV, or none")
    p.set_defaults(fn=cmd_check)
    p = sp.add_parser("val-cov", help="covariates TSV of the validation tracks")
    p.add_argument("--cov", required=True)
    p.set_defaults(fn=cmd_val_cov)
    p = sp.add_parser("prep-val", help="bin the validation truth, msevar weights")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(fn=cmd_prep_val)
    p = sp.add_parser("average", help="round-1 Average baseline of one validation track (test)")
    p.add_argument("--track", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_average)
    p = sp.add_parser("check-val", help="raw track vs observed MACS2 signal and truth (test)")
    p.add_argument("--track", required=True); p.add_argument("--out", required=True)
    p.add_argument("--cov", required=True)
    p.set_defaults(fn=cmd_check_val)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
