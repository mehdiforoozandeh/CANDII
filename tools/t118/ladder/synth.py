"""t118 ladder — synthetic t112-layout products, a MANIFEST.tsv and a covariates.tsv, for tests.

Two tracks by default: `T1` (assay H3K27ac, narrow, a matched control) and `T2` (DNase-seq, no
control). Each has a base and four arms:
  depth:15M    counts = binomial thinning of the base at ratio 0.5,  p = 0.5  x base p
  depth:7.5M   counts = binomial thinning of the base at ratio 0.25, p = 0.25 x base p
  extsize:k2   counts byte-identical to the base (the npz is copied), p = 5-bin mean of base p
  pe:pe        counts = 3-bin rounded mean of the base, p recomputed from those counts
(`depth_ratio_arm=False` leaves out the two depth arms.) The base counts are Poisson on a latent
gamma background with peaks; -log10 p is the Poisson upper tail of the count against the
background mean. npz files are written with `tools/t112/bin25.py::write_npz`; the manifest has the
t112 columns (`tools/t112/records.py::MANIFEST_COLUMNS`) with the real npz file md5s; the
covariates table has the pinned t118-C0 columns. Pids are `<track>__<arm>__<level>`.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import poisson

_TOOLS = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bin25 = _load("t112_bin25_for_ladder", _TOOLS / "t112" / "bin25.py")

MANIFEST_COLUMNS = ("pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
                    "knob_value", "depth", "read_length", "run_type", "fraglen",
                    "control_accession", "control_source", "pipeline_repo", "pipeline_release",
                    "sif_md5", "counts25_md5", "pval25_md5", "control_counts25_md5",
                    "slurm_job_ids", "product_dir")
#: the t118-C0 covariates.tsv columns, in order
COV_COLUMNS = ("pid", "track", "assay", "arm", "level", "usable", "depth_reads", "depth_log2",
               "read_length", "run_type_pe", "dedup_on", "mapq_thresh", "has_control",
               "control_fraction", "ratio_k", "ctl_identity", "control_reads",
               "control_depth_log2", "extsize_k", "fraglen_bp", "src_depth", "src_read_length",
               "src_run_type", "src_dedup", "src_mapq", "src_control_fraction", "src_ratio_k",
               "src_ctl_identity", "src_control_depth", "src_extsize_k", "src_fraglen")
DEFAULT_N_BINS = {"chr1": 4000, "chr2": 3000, "chr19": 1500, "chr21": 1200, "chr22": 800,
                  "chrX": 900}

#: per synthetic track: assay, cell, pipeline, base depth, read length, fraglen, control reads.
#: Both bases hold 30 M reads so every depth arm's depth = ratio x base depth, as its level says.
TRACK_SPECS = {
    "T1": {"assay": "H3K27ac", "cell": "S01", "pipeline": "chip", "depth": 30_000_000,
           "read_length": 101, "fraglen": 180, "control_reads": 40_000_000},
    "T2": {"assay": "DNase-seq", "cell": "S02", "pipeline": "atac", "depth": 30_000_000,
           "read_length": 76, "fraglen": 150, "control_reads": 0},
}
PIPELINES = {"chip": ("ENCODE-DCC/chip-seq-pipeline2", "v2.2.2", "f6e408f7e77bafde4556883f33552191"),
             "atac": ("ENCODE-DCC/atac-seq-pipeline", "v2.2.3", "04d9fa482d67cf633845653750f58cef")}
BG_MEAN = 2.0


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _pval(counts: np.ndarray, bg: float) -> np.ndarray:
    """-log10 P(Poisson(bg) >= k), >= 0 and finite; 0 at k = 0."""
    k = np.asarray(counts, dtype=np.float64)
    lsf = poisson.logsf(k - 1.0, bg)
    return np.clip(-lsf / math.log(10.0), 0.0, 300.0).astype(np.float32)


def _base_counts(rng, n: int) -> np.ndarray:
    lam = rng.gamma(0.6, BG_MEAN / 0.6, n)
    n_peaks = max(1, n // 200)
    for s in rng.integers(0, max(1, n - 20), n_peaks):
        w = int(rng.integers(3, 20))
        lam[s:s + w] += rng.gamma(2.0, 10.0)
    return rng.poisson(lam).astype(np.uint32)


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(np.asarray(a, dtype=np.float64), np.ones(w) / w, mode="same")


def _cov_row(track: str, spec: dict, arm: str, level: str, depth: int, read_length: int,
             run_type_pe: int, fraglen: int, extsize_k: float) -> dict:
    has_ctl = int(spec["control_reads"] > 0)
    ctl_reads = spec["control_reads"] if has_ctl else 0
    syn = "synth.py:"
    return {
        "pid": f"{track}__{arm}__{level}", "track": track, "assay": spec["assay"], "arm": arm,
        "level": level, "usable": 1, "depth_reads": depth, "depth_log2": repr(math.log2(depth)),
        "read_length": read_length, "run_type_pe": run_type_pe, "dedup_on": 1, "mapq_thresh": 30,
        "has_control": has_ctl, "control_fraction": repr(0.0),
        "ratio_k": repr(1.0 if has_ctl else 0.0),
        "ctl_identity": "matched" if has_ctl else "none", "control_reads": ctl_reads,
        "control_depth_log2": repr(math.log2(ctl_reads) if ctl_reads else 0.0),
        "extsize_k": repr(float(extsize_k)), "fraglen_bp": fraglen,
        "src_depth": "manifest:depth", "src_read_length": "manifest:read_length",
        "src_run_type": "manifest:run_type", "src_dedup": syn + "default",
        "src_mapq": syn + "default", "src_control_fraction": syn + "default",
        "src_ratio_k": syn + "default" if has_ctl else "none: no control",
        "src_ctl_identity": syn + "TRACK_SPECS" if has_ctl else "none: no control",
        "src_control_depth": syn + "TRACK_SPECS" if has_ctl else "none: no control",
        "src_extsize_k": syn + "arm level", "src_fraglen": "manifest:fraglen",
    }


def make_products(root, seed, n_bins=DEFAULT_N_BINS, tracks=("T1", "T2"),
                  depth_ratio_arm=True) -> tuple[Path, Path]:
    """Write products under `root`; returns (`root/MANIFEST.tsv`, `root/covariates.tsv`)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    man_rows, cov_rows = [], []
    for ti, track in enumerate(tracks):
        spec = TRACK_SPECS[track]
        rng = np.random.default_rng([int(seed), 7200, ti])
        base_c = {c: _base_counts(rng, n) for c, n in n_bins.items()}
        base_p = {c: _pval(v, BG_MEAN) for c, v in base_c.items()}
        repo, release, sif = PIPELINES[spec["pipeline"]]
        pre = spec["pipeline"]
        # (arm, level, counts, pval, knob, knob_value, depth, run_type, fraglen, extsize_k, route)
        arms = [("base", "base", base_c, base_p, "none", "", spec["depth"], "single-ended",
                 spec["fraglen"], 1.0, "bam")]
        if depth_ratio_arm:
            for level, ratio in (("15M", 0.5), ("7.5M", 0.25)):
                d = int(spec["depth"] * ratio)
                cc = {c: rng.binomial(v.astype(np.int64), ratio).astype(np.uint32)
                      for c, v in base_c.items()}
                pp = {c: (ratio * v).astype(np.float32) for c, v in base_p.items()}
                arms.append(("depth", level, cc, pp, "treatment_reads", str(d), d,
                             "single-ended", spec["fraglen"], 1.0, "bam"))
        ext_knob = "fraglen" if pre == "chip" else "atac.smooth_win"
        pp = {c: _smooth(v, 5).astype(np.float32) for c, v in base_p.items()}
        arms.append(("extsize", "k2", None, pp, ext_knob, str(2 * spec["fraglen"]),
                     spec["depth"], "single-ended", 2 * spec["fraglen"], 2.0, "bam"))
        cc = {c: np.rint(_smooth(v, 3)).astype(np.uint32) for c, v in base_c.items()}
        pp = {c: _pval(v, BG_MEAN) for c, v in cc.items()}
        arms.append(("pe", "pe", cc, pp, f"{pre}.paired_end", "true", spec["depth"],
                     "paired-ended", spec["fraglen"], 1.0, "fastq"))

        base_dir = root / f"{track}__base__base"
        for arm, level, cc, pp, knob, kval, depth, run_type, fraglen, ek, route in arms:
            pid = f"{track}__{arm}__{level}"
            d = root / pid
            if cc is None:                     # p-only arm: counts byte-identical to the base
                d.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base_dir / "counts25.npz", d / "counts25.npz")
            else:
                _bin25.write_npz(d / "counts25.npz", cc)
            _bin25.write_npz(d / "pval25.npz", pp)
            has_ctl = spec["control_reads"] > 0
            man_rows.append({
                "pid": pid, "biosample": f"SYN_{spec['cell']}__{arm}__{level}", "track": track,
                "cell": spec["cell"], "assay": spec["assay"], "arm": arm, "level": level,
                "route": route, "knob": knob, "knob_value": kval, "depth": depth,
                "read_length": spec["read_length"], "run_type": run_type, "fraglen": fraglen,
                "control_accession": f"SYNCTL{ti + 1}" if has_ctl else "",
                "control_source": "matched" if has_ctl else "", "pipeline_repo": repo,
                "pipeline_release": release, "sif_md5": sif,
                "counts25_md5": _md5(d / "counts25.npz"), "pval25_md5": _md5(d / "pval25.npz"),
                "control_counts25_md5": "", "slurm_job_ids": "synthetic", "product_dir": str(d),
            })
            cov_rows.append(_cov_row(track, spec, arm, level, depth, spec["read_length"],
                                     int(run_type == "paired-ended"), fraglen, ek))

    manifest = root / "MANIFEST.tsv"
    covariates = root / "covariates.tsv"
    for path, cols, rows in ((manifest, MANIFEST_COLUMNS, man_rows),
                             (covariates, COV_COLUMNS, cov_rows)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["pid"]):
                w.writerow(r)
    return manifest, covariates
