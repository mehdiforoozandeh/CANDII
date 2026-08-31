#!/usr/bin/env python3
"""Score one Dataset-3 prediction track: the nine challenge measures plus the P-block.

An I/O adapter around `vendor/`, which is byte-identical to experiment 001's scorer and is never
edited. The vendored `fir_score.py` remains the authority for the nine measures and is what the
Average gate runs unmodified; this driver exists for two reasons `fir_score.py` cannot serve:

1. **One bigwig load, two blocks.** A whole-genome track is ~124 M bins over chr1-22+chrX; loading
   it twice to get the nine measures and then the P-block roughly doubles the wall clock of a
   23-entrant x 51-experiment grid. This driver loads truth and prediction once and computes both.

2. **The two blocks need different grids.** The nine measures run on the blacklist-DELETED arrays,
   because deletion is what produced the published numbers. The P-block is positional -- its
   promoter windows are bin coordinates -- and deletion shifts every downstream index, so it runs on
   the intact grid. `pblock_bigwig` documents this; `src/candi/bench/annotations.py` refuses the
   deletion on the store path for the same reason. Computing both from one load means the two grids
   are derived from identical bytes rather than from two reads that could drift.

`--check-against-fir-score` re-runs the vendored `fir_score.py` as a subprocess on the same inputs
and asserts this driver's nine measures match it exactly. That is the standing proof that adapting
the I/O did not move a number; the Average gate runs it.

DNase-seq is refused outright rather than scored and dropped later (plan §2/P3, decision B3): in
Dataset 2 it is read-depth normalized signal and in Dataset 3 a -log10 p-value, so a DNase row here
would be a number with no defensible meaning. Pass `--allow-dnase` only to measure that gap
deliberately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
sys.path.insert(0, HERE)

import eic_metrics as EM          # noqa: E402  vendored, byte-identical
import fir_tracks as FT           # noqa: E402  vendored, byte-identical
import pblock_bigwig as PB        # noqa: E402  our port

# Reported separately in every table, never pooled (decision B3, plan §6.4).
BROAD_MARKS = {"H3K27me3", "H3K36me3", "H3K9me3"}
EXCLUDED_ASSAYS = {"DNase-seq", "DNase"}

# 001 could not reconstruct the published variance vector: no candidate got closer than median ratio
# 0.19. Both 001 and 005 exclude it from every comparison and so does this driver, unconditionally.
MSEVAR_EXCLUDED_REASON = (
    "msevar excluded: experiment 001 could not reconstruct the published variance vector "
    "(median ratio 0.19 training-only, 0.61 pooled). Eight of nine measures reproduce.")
SCORED_MEASURES = [m for m in EM.MEASURES if m != "msevar"]


def load_bridge(path: str) -> Dict[str, dict]:
    return {r["filename"]: r for r in csv.DictReader(open(path))}


def nine_measures(truth, pred, prom, body, enh) -> List[dict]:
    """One row per bootstrap chromosome group, on the blacklist-deleted grid.

    `var_d` is left None on purpose, so `eic_metrics.score` writes `msevar = 0.0`; the key is then
    dropped rather than carried as a fake zero. The ten groups are the challenge's own fixed
    chromosome subsets, not resamples of genomic positions -- the paper's Methods text says
    otherwise and the code is right.
    """
    rows = []
    for b, chroms in enumerate(EM.BOOTSTRAP_CHROM):
        s = EM.score(truth, pred, chroms, prom, body, enh, None)
        rows.append({"bootstrap_id": b, **{m: s[m] for m in SCORED_MEASURES}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="C##M##")
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--repo", required=True, help="imputation_challenge checkout")
    ap.add_argument("--truth-dir", required=True, help="blind_truth bigwigs, C##M##.bigwig")
    ap.add_argument("--pred-dir", required=True, help="one method's bigwigs, C##M##.bigwig")
    ap.add_argument("--label", required=True, help="method name written into every row")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True, help="P-block and provenance")
    ap.add_argument("--allow-dnase", action="store_true")
    ap.add_argument("--skip-pblock", action="store_true")
    ap.add_argument("--check-against-fir-score", action="store_true",
                    help="re-run vendor/fir_score.py on the same inputs and require an exact match "
                         "on the eight scored measures")
    args = ap.parse_args()

    row = load_bridge(args.bridge)[args.experiment]
    assay = row["assay_name"]
    if assay in EXCLUDED_ASSAYS and not args.allow_dnase:
        print(f"[{args.experiment}] {assay} excluded from every Dataset-3 row (B3); "
              f"pass --allow-dnase to override", flush=True)
        return 0

    t0 = time.time()
    prom, body, enh = EM.load_annotations(args.repo)
    blacklist = EM.load_blacklist_bins(args.repo)

    truth_path = os.path.join(args.truth_dir, f"{args.experiment}.bigwig")
    pred_path = os.path.join(args.pred_dir, f"{args.experiment}.bigwig")
    for p in (truth_path, pred_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing bigwig: {p}")

    truth = FT.load_official_bigwig(truth_path, EM.CHROMS)
    pred = FT.load_official_bigwig(pred_path, EM.CHROMS)
    truth, pred = FT.align([truth, pred])
    n_bins = sum(len(v) for v in truth.values())
    print(f"[{args.experiment}] {args.label} {assay}: loaded {n_bins} bins "
          f"in {time.time()-t0:.0f}s", flush=True)

    # --- P-block first, on the INTACT grid -----------------------------------
    pblock = None
    if not args.skip_pblock:
        t1 = time.time()
        genes = PB.load_gene_annotations(args.repo) if assay == "H3K4me3" else None
        pblock = PB.partition_suite_multichrom(truth, pred, EM.CHROMS, assay=assay,
                                               gene_annotations=genes)
        print(f"  p-block in {time.time()-t1:.0f}s: "
              f"macro_acc_obs={pblock['acc_by_obs_strength']['macro_accuracy']:.4f}", flush=True)

    # --- the nine measures, on the DELETED grid ------------------------------
    truth_bl = EM.apply_blacklist(truth, blacklist)
    pred_bl = EM.apply_blacklist(pred, blacklist)
    rows = nine_measures(truth_bl, pred_bl, prom, body, enh)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "experiment", "cell", "assay", "assay_name", "mark_class",
                    "bootstrap_id", "truth_source", "blacklist_filtered"] + SCORED_MEASURES)
        for r in rows:
            w.writerow([args.label, args.experiment, row["cell_id"], row["assay_id"], assay,
                        "broad" if assay in BROAD_MARKS else "punctate",
                        r["bootstrap_id"], "official", 1] + [r[m] for m in SCORED_MEASURES])

    prov = {
        "method": args.label,
        "experiment": args.experiment,
        "assay": assay,
        "mark_class": "broad" if assay in BROAD_MARKS else "punctate",
        "cell": row["cell_id"],
        "dataset": "Dataset 3 (challenge tracks, syn17083203)",
        "truth": truth_path,
        "prediction": pred_path,
        "n_bins": n_bins,
        "chroms": EM.CHROMS,
        "scored_measures": SCORED_MEASURES,
        "msevar_excluded": MSEVAR_EXCLUDED_REASON,
        "nine_measures_blacklist_deleted": True,
        "pblock_blacklist_deleted": False,
        "scorer_vendor_md5": vendor_md5(),
        "pblock": pblock,
        "bootstraps": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(prov, fh, indent=1)

    if args.check_against_fir_score:
        check_against_fir_score(args, rows)

    print(f"[{args.experiment}] done in {time.time()-t0:.0f}s -> {args.out_csv}", flush=True)
    return 0


def vendor_md5() -> Dict[str, str]:
    """md5 of every vendored file, stamped into each result so a score traces to its scorer."""
    import hashlib
    out = {}
    vdir = os.path.join(HERE, "vendor")
    for name in sorted(os.listdir(vdir)):
        if not name.endswith(".py"):
            continue
        h = hashlib.md5()
        with open(os.path.join(vdir, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def check_against_fir_score(args, rows) -> None:
    """Run the vendored scorer unmodified and require an exact match on the eight measures.

    Exact, not approximate: both paths call the same `eic_metrics.score` on arrays built by the same
    `fir_tracks.load_official_bigwig`, so any difference at all means the adaptation changed an
    input, which is a defect rather than rounding.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ref_csv = os.path.join(td, "ref.csv")
        cmd = [sys.executable, os.path.join(HERE, "vendor", "fir_score.py"),
               "--experiment", args.experiment, "--bridge", args.bridge, "--repo", args.repo,
               "--truth", "official", "--truth-dir", args.truth_dir,
               "--pred", "bigwig", "--pred-dir", args.pred_dir,
               "--var-source", "none", "--label", args.label, "--out", ref_csv]
        print(f"  cross-check: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        ref = list(csv.DictReader(open(ref_csv)))

    assert len(ref) == len(rows), f"bootstrap count {len(ref)} != {len(rows)}"
    worst = 0.0
    for r_ref, r_ours in zip(ref, rows):
        for m in SCORED_MEASURES:
            a, b = float(r_ref[m]), float(r_ours[m])
            if a != b:
                worst = max(worst, abs(a - b) / max(abs(a), 1e-30))
                raise SystemExit(
                    f"MISMATCH vs vendored fir_score.py on {m} bootstrap {r_ref['bootstrap_id']}: "
                    f"vendored={a!r} driver={b!r} (rel {abs(a-b)/max(abs(a),1e-30):.3e})")
    print(f"  cross-check PASSED: exact match vs vendored fir_score.py on "
          f"{len(SCORED_MEASURES)} measures x {len(rows)} bootstraps", flush=True)


if __name__ == "__main__":
    sys.exit(main())
