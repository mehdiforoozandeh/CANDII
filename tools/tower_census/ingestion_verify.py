"""Second method for the ingestion census (t78).

For each picked `biosample:assay:regime:split`, re-derive the reported largest ingested
value through a DIFFERENT path:

* the counts come out of `candi.store.reader.CorpusStore` (the loader's own reader), not
  raw h5py whole-chromosome blocks;
* the window is read at its coordinates as a single `[start, start+768)` slice, and its
  max and argmax are recomputed;
* eligibility is re-derived from the D12 DEFINITION — `mask[s : s+768].mean() >= 0.9`
  on the raw mask slice — instead of the cumulative-sum `eligible_window_mask`;
* the window is confirmed to be the tile the census says it is, and for `eic.pilot`
  train it is confirmed to lie wholly inside one Pilot Region.

Read-only.

    python ingestion_verify.py <per_track.json> <src> <pilot_bed> bios:assay:regime:split …
"""
import json
import sys

import h5py
import numpy as np

PER_TRACK = sys.argv[1]
SRC = sys.argv[2]
BED = sys.argv[3]
PICK = sys.argv[4:]

sys.path.insert(0, SRC)
from candi.store.reader import CorpusStore
from candi.store.regime import RegionSet

STORE = "/project/def-maxwl/mforooz/CANDI_STORE/eic"
GENOME = "/project/def-maxwl/mforooz/CANDI_STORE/genome"
CB, RES, MVF = 768, 25, 0.9
SHA = "13e11a198fdee08edb7797d1e402b5d985846b5a7d973ade91e8511462acb7a3"

rows = {(r["biosample"], r["assay"]): r for r in json.load(open(PER_TRACK))}
tw = {(r["biosample"], r["assay"]): r["_top_windows"] for r in json.load(open(PER_TRACK))}
pilot = RegionSet.from_obj({"bed": BED, "sha256": SHA})
corpus = CorpusStore(STORE)
mask = h5py.File(f"{GENOME}/mask.h5", "r")

PFX = {("eic.19", "train"): "e19_train", ("eic.19", "eval"): "e19_eval",
       ("eic.pilot", "train"): "pilot_train", ("eic.pilot", "eval"): "pilot_eval"}

bad = 0
for key in PICK:
    b, a, reg, sp = key.split(":")
    r = rows[(b, a)]
    p = PFX[(reg, sp)]
    want_val = int(r[f"{p}_max"])
    chrom, bp = r[f"{p}_max_locus"].split(":")
    bin_idx = int(bp) // RES
    start = (bin_idx // CB) * CB

    v = corpus[b][a].counts(chrom, start, start + CB).astype(np.int64)
    got = int(v.max())
    got_bin = start + int(np.argmax(v))

    m = mask[chrom][start : start + CB]
    frac = float(np.asarray(m, dtype=np.float64).mean())
    elig = frac >= MVF - 1e-9

    contained = True
    if reg == "eic.pilot" and sp == "train":
        contained = any(c == chrom and s <= start * RES and (start + CB) * RES <= e
                        for c, s, e, _ in pilot.intervals)

    ok = (got == want_val) and (got_bin == bin_idx) and elig and contained
    bad += 0 if ok else 1
    print(f"{'OK ' if ok else 'BAD'} {key:<52} window {chrom}:{start*RES}-{(start+CB)*RES} "
          f"max={got} (census {want_val}) at bin {got_bin} (census {bin_idx}) "
          f"mask_mean={frac:.4f} eligible={elig} contained={contained}")

    tops = tw[(b, a)].get(f"{reg}/{sp}", [])
    for t in tops[:5]:
        s2 = t["window_start_bin"]
        vv = corpus[b][a].counts(t["chrom"], s2, s2 + CB).astype(np.int64)
        mm = float(np.asarray(mask[t["chrom"]][s2 : s2 + CB], dtype=np.float64).mean())
        agree = int(vv.max()) == t["count"] and (s2 + int(np.argmax(vv))) == t["max_bin"]
        if not agree or mm < MVF - 1e-9:
            bad += 1
            print(f"   TOPWINDOW MISMATCH {t['chrom']}:{t['window_start_bp']} "
                  f"reader={int(vv.max())} census={t['count']} mask_mean={mm:.4f}")

mask.close()
corpus.close()
print(f"\n{len(PICK)} window(s) checked, {bad} disagreement(s)")
