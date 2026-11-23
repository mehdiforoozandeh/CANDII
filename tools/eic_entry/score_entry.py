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
    """(25 bp control counts float64, lambda label) from a .npz (observed) or .h5 (predicted)."""
    if spec.endswith(".npz"):
        with np.load(spec) as z:
            return z[f"{chrom}_cnt"].astype(np.float64), "observed_control"
    import h5py
    with h5py.File(spec, "r") as f:
        return f["mu"][chrom][:].astype(np.float64), "predicted_control"


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
                 control_is_predicted=not cov["control"].endswith(".npz"))
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


# ---------------------------------------------------------------------------- test
def cmd_make_test(a):
    """Observed 25 bp overlap counts in predict.py's layout, and the covariates row (rung L6's)."""
    import h5py

    pred = Path(a.pred)
    (pred / "counts").mkdir(parents=True, exist_ok=True)
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
    rows = []
    for name, mine, ref in (("raw vs rung L6 (array)", out / "raw" / t / f"{t}.npy", REF_L6 / f"{t}.npy"),
                            ("qn vs rung L7 (array)", out / "qn" / t / f"{t}.npy", REF_L7 / f"{t}.npy")):
        nd, tot, mx = _arrdiff(mine, ref)
        rows.append([t, name, f"{tot} bins", f"{nd} differ", f"max abs {mx:.3g}", nd == 0])
    for board, ref in REF_SCORES.items():
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
    p.set_defaults(fn=cmd_run)
    p = sp.add_parser("collect", help="placement on both boards, SUMMARY.md")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_collect)
    p = sp.add_parser("make-test", help="observed counts as a prediction (test)")
    p.add_argument("--track", required=True, choices=sorted(TEST))
    p.add_argument("--pred", required=True); p.add_argument("--cov", required=True)
    p.set_defaults(fn=cmd_make_test)
    p = sp.add_parser("check", help="pipeline vs rungs L6 / L7 (test)")
    p.add_argument("--track", required=True, choices=sorted(TEST)); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
