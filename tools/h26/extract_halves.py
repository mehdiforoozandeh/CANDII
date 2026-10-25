"""h26 step 1 (Nibi): pull the chr21 25 bp counts of every pseudoreplicate half the probe needs.

The t117 halves (pr1, pr2) live only on Nibi; the checkpoint, the CANDI corpus prompt and the
full t112 products are on Fir. This writes ONE compressed npz holding chr21 only, keyed
`<pid>|<half>`, so the probe on Fir can read the halves without the whole-genome files.

Products: per track, base + crop 36/50 (ChIP only) + pe + depth 15M/7.5M/3.75M.
Also records, per half, the line counts from the t117 MANIFEST so the probe can check that
each half holds about half the reads (the shared half-depth shift assumes it).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/def-maxwl/mforooz/t112_cf/pseudoreps")
TRACKS = ["C12M02", "C19M16", "C40M17", "C40M18", "C07M20", "C19M22", "C07M29"]
ARMS = ["base__base", "crop__36", "crop__50", "pe__pe", "depth__15M", "depth__7.5M", "depth__3.75M"]


def main(out: str) -> None:
    man = {(r["pid"], r["half"]): r for r in csv.DictReader(open(ROOT / "MANIFEST.tsv"), delimiter="\t")}
    arrays, meta = {}, {}
    for t in TRACKS:
        for a in ARMS:
            pid = f"{t}__{a}"
            for half in ("pr1", "pr2"):
                f = ROOT / pid / half / "counts25.npz"
                if not f.exists():
                    meta[f"{pid}|{half}"] = {"missing": str(f)}
                    continue
                md5 = hashlib.md5(f.read_bytes()).hexdigest()
                row = man.get((pid, half), {})
                with np.load(f) as z:
                    x = z["chr21"]
                arrays[f"{pid}|{half}"] = x
                meta[f"{pid}|{half}"] = {
                    "file": str(f), "md5": md5, "manifest_md5": row.get("counts25_md5"),
                    "md5_ok": md5 == row.get("counts25_md5"),
                    "n_lines": int(row["n_lines"]) if row.get("n_lines") else None,
                    "product_lines": int(row["product_lines"]) if row.get("product_lines") else None,
                    "chr21_sum": int(x.sum()), "dtype": str(x.dtype), "n_bins": int(x.shape[0]),
                }
                print(pid, half, meta[f"{pid}|{half}"]["chr21_sum"], meta[f"{pid}|{half}"]["md5_ok"], flush=True)
    np.savez_compressed(out, **arrays)
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, len(arrays), "arrays")


if __name__ == "__main__":
    main(sys.argv[1])
