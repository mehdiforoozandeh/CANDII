"""Second method for the tower census.

The census reads counts.h5 with raw h5py, whole-chromosome blocks, two passes.
This re-derives the same numbers for a sample of tracks through a DIFFERENT code path:
`candi.store.reader.CorpusStore`, chunked 2 M bins at a time, int32 out of the reader.
It also re-reads each reported top bin at its coordinate and checks the value.

Read-only.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/project/def-maxwl/mforooz/CANDII_t77/src")
from candi.store.reader import CorpusStore

CENSUS = sys.argv[1]
PICK = sys.argv[2:]          # "biosample:assay" pairs
CHUNK = 2_000_000

rows = {(r["biosample"], r["assay"]): r for r in json.load(open(CENSUS))}
corpus = CorpusStore("/project/def-maxwl/mforooz/CANDI_STORE/eic")

bad = 0
for key in PICK:
    b, a = key.split(":")
    r = rows[(b, a)]
    bs = corpus[b]
    total = 0
    gmax = 0
    n_over = 0
    excess = 0
    nbins = 0
    ceiling = r["keepdup1_ceiling"]
    for c in corpus.chroms():
        if not bs.has(a, "counts"):
            continue
        n = bs.n_bins(c)
        nbins += n
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            v = bs[a].counts(c, s, e).astype(np.int64)
            total += int(v.sum())
            gmax = max(gmax, int(v.max()))
            m = v > ceiling
            k = int(m.sum())
            n_over += k
            if k:
                excess += int(v[m].sum()) - k * ceiling
    # re-read the reported top bins at their coordinates
    tb_ok = True
    for tb in r["top_bins"][:10]:
        got = int(bs[a].counts(tb["chrom"], tb["bin"], tb["bin"] + 1)[0])
        if got != tb["count"]:
            tb_ok = False
            print(f"   TOPBIN MISMATCH {key} {tb['chrom']}:{tb['start_bp']} "
                  f"census={tb['count']} reader={got}")
    checks = [
        ("n_bins", nbins, r["n_bins_genome"]),
        ("total_counts", total, r["total_counts"]),
        ("global_max_count", gmax, r["global_max_count"]),
        ("n_bins_over_ceiling", n_over, r["n_bins_over_ceiling"]),
        ("excess_mass_over_ceiling", excess, r["excess_mass_over_ceiling"]),
    ]
    ok = all(abs(x - y) <= 1e-6 * max(1.0, abs(y)) for _, x, y in checks) and tb_ok
    bad += 0 if ok else 1
    print(f"{'OK ' if ok else 'BAD'} {key:<38} ceil={ceiling:<6} max={gmax:<12} "
          f"over={n_over:<10} total={total}")
    if not ok:
        for nm, x, y in checks:
            if abs(x - y) > 1e-6 * max(1.0, abs(y)):
                print(f"   {nm}: reader={x} census={y}")

corpus.close()
print(f"\n{len(PICK)} tracks checked, {bad} disagreements")
