"""Deterministic generator for the t139 knob-scorer fixture (tests/fixtures/t139_knob/).

Two synthetic tracks on chr19 + chr21, 2 000 x 25 bp bins each:
  C07M20 (H3K4me1, narrow): base, depth 15M (an independent draw at half the mean), ratio k2
                            (counts bit-identical to base; -log10 p rescaled -- a p-only knob)
  C12M02 (DNase-seq):       base, dedup off (extra duplicate reads, mostly in peaks)

    python tests/fixtures/t139_knob/make_fixture.py [out_dir]   # default: this directory

`arrays()` returns every array keyed by (pid, file, chrom); the test hashes them against the
files on disk (npz zip headers carry a timestamp, so the file md5 is not reproducible).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEED = 20261029
N_BINS = 2000
CHROMS = ("chr19", "chr21")
HEADER = ("pid biosample track cell assay arm level route knob knob_value depth read_length "
          "run_type fraglen control_accession control_source pipeline_repo pipeline_release "
          "sif_md5 counts25_md5 pval25_md5 control_counts25_md5 slurm_job_ids product_dir").split()
PRODUCTS = [  # pid, track, cell, assay, arm, level, knob, knob_value, depth
    ("C07M20__base__base", "C07M20", "C07", "H3K4me1", "base", "base", "none", "", "30000000"),
    ("C07M20__depth__15M", "C07M20", "C07", "H3K4me1", "depth", "15M", "treatment_reads",
     "15000000", "15000000"),
    ("C07M20__ratio__k2", "C07M20", "C07", "H3K4me1", "ratio", "k2", "macs2_ratio", "0.375",
     "30000000"),
    ("C12M02__base__base", "C12M02", "C12", "DNase-seq", "base", "base", "none", "", "50000000"),
    ("C12M02__dedup__off", "C12M02", "C12", "DNase-seq", "dedup", "off", "atac.no_dup_removal",
     "true", "50000000"),
]
BLACKLIST = "chr19\t10000\t12010\nchr21\t30010\t31000\nchr21\t49000\t49980\n"


def _pval(counts: np.ndarray, scale: float, rng) -> np.ndarray:
    return (scale * np.log1p(counts) + rng.uniform(0, 0.05, counts.size)).astype(np.float32)


def arrays() -> dict:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    for track in ("C07M20", "C12M02"):
        for c in CHROMS:
            lam = rng.gamma(2.0, 1.5, N_BINS)
            peaks = rng.choice(N_BINS, 30, replace=False)
            lam[peaks] += rng.uniform(60, 300, peaks.size)
            base = rng.poisson(lam).astype(np.uint32)
            out[(f"{track}__base__base", "counts", c)] = base
            out[(f"{track}__base__base", "pval", c)] = _pval(base, 1.0, rng)
            if track == "C07M20":
                d = rng.poisson(0.5 * lam).astype(np.uint32)
                out[("C07M20__depth__15M", "counts", c)] = d
                out[("C07M20__depth__15M", "pval", c)] = _pval(d, 1.0, rng)
                out[("C07M20__ratio__k2", "counts", c)] = base.copy()
                out[("C07M20__ratio__k2", "pval", c)] = _pval(base, 1.3, rng)
            else:
                dup = np.zeros(N_BINS)
                dup[peaks] = 0.2 * lam[peaks]
                x = (base + rng.poisson(dup + 0.1)).astype(np.uint32)
                out[("C12M02__dedup__off", "counts", c)] = x
                out[("C12M02__dedup__off", "pval", c)] = _pval(x, 1.0, rng)
    return out


def manifest_text() -> str:
    lines = ["\t".join(HEADER)]
    for pid, track, cell, assay, arm, level, knob, kv, depth in PRODUCTS:
        row = dict.fromkeys(HEADER, "")
        row.update(pid=pid, biosample=f"CF_{cell}__{arm}__{level}", track=track, cell=cell,
                   assay=assay, arm=arm, level=level, route="bam", knob=knob, knob_value=kv,
                   depth=depth, product_dir=f"synthetic/{pid}")
        lines.append("\t".join(row[h] for h in HEADER))
    return "\n".join(lines) + "\n"


def write(out_dir: Path) -> None:
    prod = out_dir / "products"
    prod.mkdir(parents=True, exist_ok=True)
    arr = arrays()
    for pid, *_ in PRODUCTS:
        (prod / pid).mkdir(exist_ok=True)
        for space, fname in (("counts", "counts25.npz"), ("pval", "pval25.npz")):
            np.savez_compressed(prod / pid / fname, **{c: arr[(pid, space, c)] for c in CHROMS})
    (prod / "MANIFEST.tsv").write_text(manifest_text())
    (out_dir / "blacklist.bed").write_text(BLACKLIST)


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
