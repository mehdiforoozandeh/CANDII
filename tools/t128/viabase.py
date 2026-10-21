"""t128 reference — old D via the base: t126's trained rung-D maps chained arm -> base -> arm.

Old D (t126 arm D, `/project/def-maxwl/mforooz/t126/D/runs/D_<g>_counts_real_s<seed>`, depth
7.5M held out) learned only base<->arm maps. On a never-trained arm->arm pair (a, b) of one
track this predictor composes the two trained maps through the track's base product:

  hop 1  (loc1, _) = old D on x_a under g(c_a, c_base)
  x_mid  = the NB mean of hop 1 (float64, not rounded; plan_t128 Decision (iv))
  hop 2  (loc, disp) = old D on x_mid under g(c_base, c_b)

and `score.score_law(..., split=True)` scores every never-trained pair of the run's g, so the
CRPS carries its oracle_scaled / scale_error split. The old run dir is read only; everything is
written to `<out_dir>` (t128's own tree): law.json, LAW_DONE, viabase.json.

**Tasks.** 24 = G_IDS x SEEDS, g slowest: `index = g_i*3 + seed`,
`run_name = f"D_{g}_counts_real_s{seed}"` (index 9 = D_C19M16_counts_real_s0).

    python tools/t128/viabase.py tasks
    python tools/t128/viabase.py run <manifest> <covariates.tsv> <data_dir> <blacklist.bed> \\
        <old_run_dir> <out_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:          # import convention: tools/t118 holds `ladder`
    sys.path.insert(0, str(_T118))

from ladder import model, pairs, score  # noqa: E402
from ladder.train import git_sha  # noqa: E402

G_IDS = pairs.G_IDS
SEEDS = (0, 1, 2)
TASK_KEYS = ("index", "run_name", "g", "seed")
VIABASE_RUNS: list[dict] = [
    {"index": gi * len(SEEDS) + si, "run_name": f"D_{g}_counts_real_s{seed}", "g": g, "seed": seed}
    for gi, g in enumerate(G_IDS) for si, seed in enumerate(SEEDS)]


def base_pid_of(rows: list[dict], pid: str) -> str:
    """The `arm == "base"` product of pid's track."""
    by_pid = {r["pid"]: r for r in pairs.usable_products(rows)}
    if pid not in by_pid:
        raise ValueError(f"{pid}: not a usable product")
    base, _ = pairs._by_track(rows, by_pid[pid]["track"])
    return base["pid"]


class ViaBasePredictor(model.Predictor):
    """An old rung-D run whose arm->arm prediction is the two trained hops through the base."""

    def __init__(self, pred: model.Predictor, rows: list[dict]):
        super().__init__(pred.ladder, pred.encoder, pred.config, pred.corpus, pred.device,
                         pred.step, pred.val_nll)
        if self.per_bin or self.model != "real":
            raise ValueError(f"{self.run_name}: via the base needs a row-1 real run")
        self.rows = rows

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        base = base_pid_of(self.rows, x_src_pid)
        if base in (x_src_pid, cov_src_pid, cov_tgt_pid):
            raise ValueError(f"({cov_src_pid}, {cov_tgt_pid}): the base {base} is an end of the "
                             f"pair; via the base scores never-trained arm->arm pairs only")
        if base_pid_of(self.rows, cov_tgt_pid) != base:
            raise ValueError(f"{x_src_pid} and {cov_tgt_pid} are on different tracks")
        loc1, _ = super().predict(x_src_pid, cov_src_pid, base, chrom)
        x_mid = self.mean(loc1)                       # NB mean, float64, not rounded
        return model.predict_chrom(self.ladder, x_mid.astype(np.float32),
                                   self._theta_t(base, cov_tgt_pid), self.device)


def load_viabase(old_run_dir, data_dir, covariates, manifest, rows, device="cpu") -> ViaBasePredictor:
    """`model.load_run` of the OLD run (read only), wrapped as a ViaBasePredictor."""
    pred = model.load_run(old_run_dir, data_dir, covariates, manifest, device=device)
    return ViaBasePredictor(pred, rows)


def _refuse_t126(out_dir: Path) -> None:
    if any(part.startswith("t126") for part in Path(out_dir).resolve().parts):
        raise ValueError(f"out dir {out_dir} is under a t126 directory; t128 never writes there")


def run(manifest, covariates, data_dir, blacklist, old_run_dir, out_dir, workers=1) -> Path:
    """Score old D via the base on every never-trained pair -> <out_dir>/law.json, LAW_DONE,
    viabase.json. The old run dir is only read."""
    out_dir, old_run_dir = Path(out_dir), Path(old_run_dir)
    _refuse_t126(out_dir)
    if out_dir.resolve() == old_run_dir.resolve():
        raise ValueError("out dir equals the old run dir")
    rows = pairs.read_manifest(manifest)
    pred = load_viabase(old_run_dir, data_dir, covariates, manifest, rows, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"old_run_dir": str(old_run_dir),
            "old_ckpt_md5": model.md5_path(old_run_dir / "ckpt.pt"),
            "intermediate": "nb_mean", "git_sha": git_sha()}
    score.write_json(meta, out_dir / "viabase.json")
    return score.score_law(pred, rows, blacklist, out_dir, workers, split=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    a = sub.add_parser("run")
    for pos in ("manifest", "covariates", "data_dir", "blacklist", "old_run_dir", "out_dir"):
        a.add_argument(pos, type=Path)
    a.add_argument("--workers", type=int, default=score.default_workers())
    args = ap.parse_args(argv)
    if args.cmd == "tasks":
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + pairs._tsv(VIABASE_RUNS, TASK_KEYS))
        return 0
    done = args.out_dir / "LAW_DONE"
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    t0 = time.time()
    path = run(args.manifest, args.covariates, args.data_dir, args.blacklist, args.old_run_dir,
               args.out_dir, workers=args.workers)
    print(f"viabase {args.out_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
