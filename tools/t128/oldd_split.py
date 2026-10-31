"""t128 reference — old D's never-trained (law) pairs re-scored WITH the CRPS split.

Old D (t126 arm D, `/project/def-maxwl/mforooz/t126/D/runs/D_<g>_counts_<model>_s<seed>`, depth
7.5M held out) was law-scored by `score.py law` without the oracle_scaled / scale_error split.
This re-runs the same scoring, `score.score_law(..., split=True)`, on the same checkpoint (the
direct arm->arm map, not via the base) and writes to `<out_dir>` (t128's own tree): law.json,
LAW_DONE, oldd_split.json. The old run dir is read only. law.json's "run" block is the old run's
own `run_info` (steps, best_step, val_nll from its train_log), so the records join old D's.

**Tasks.** 48 = MODELS x G_IDS x SEEDS, model slowest, then g: `index = m_i*24 + g_i*3 + seed`,
`run_name = f"D_{g}_counts_{model}_s{seed}"` (index 9 = D_C19M16_counts_real_s0, 33 = its twin).

    python tools/t128/oldd_split.py tasks
    python tools/t128/oldd_split.py run <manifest> <covariates.tsv> <data_dir> <blacklist.bed> \\
        <old_run_dir> <out_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:          # import convention: tools/t118 holds `ladder`
    sys.path.insert(0, str(_T118))

from ladder import model, pairs, score  # noqa: E402
from ladder.train import git_sha  # noqa: E402

G_IDS = pairs.G_IDS
MODELS = ("real", "nocov")
SEEDS = (0, 1, 2)
TASK_KEYS = ("index", "run_name", "g", "model", "seed")
OLDD_RUNS: list[dict] = [
    {"index": (mi * len(G_IDS) + gi) * len(SEEDS) + si,
     "run_name": f"D_{g}_counts_{m}_s{seed}", "g": g, "model": m, "seed": seed}
    for mi, m in enumerate(MODELS) for gi, g in enumerate(G_IDS) for si, seed in enumerate(SEEDS)]


def _refuse_t126(out_dir: Path) -> None:
    if any(part.startswith("t126") for part in Path(out_dir).resolve().parts):
        raise ValueError(f"out dir {out_dir} is under a t126 directory; t128 never writes there")


def run(manifest, covariates, data_dir, blacklist, old_run_dir, out_dir, workers=1) -> Path:
    """Re-score old D's law pairs with the split -> <out_dir>/law.json, LAW_DONE, oldd_split.json.
    The old run dir is only read."""
    out_dir, old_run_dir = Path(out_dir), Path(old_run_dir)
    _refuse_t126(out_dir)
    if out_dir.resolve() == old_run_dir.resolve():
        raise ValueError("out dir equals the old run dir")
    if out_dir.name != old_run_dir.name:
        raise ValueError(f"out dir {out_dir.name} is not named after the old run {old_run_dir.name}")
    rows = pairs.read_manifest(manifest)
    pred = model.load_run(old_run_dir, data_dir, covariates, manifest, device="cpu")
    if pred.rung != "D" or pred.space != "counts" or pred.model not in MODELS:
        raise ValueError(f"{old_run_dir.name}: not an old rung-D counts real/nocov run")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"old_run_dir": str(old_run_dir),
            "old_ckpt_md5": model.md5_path(old_run_dir / "ckpt.pt"),
            "old_law_md5": (model.md5_path(old_run_dir / "law.json")
                            if (old_run_dir / "law.json").is_file() else None),
            "git_sha": git_sha()}
    score.write_json(meta, out_dir / "oldd_split.json")
    done = out_dir / "LAW_DONE"
    path = score.score_law(pred, rows, blacklist, out_dir, workers, split=True)
    done.unlink()                       # LAW_DONE only after the run block is the old run's
    doc = json.loads(path.read_text())
    doc["run"] = score.run_info(pred, old_run_dir)
    score.write_json(doc, path)
    done.write_text(doc["created_utc"] + "\n")
    return path


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
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + pairs._tsv(OLDD_RUNS, TASK_KEYS))
        return 0
    done = args.out_dir / "LAW_DONE"
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    t0 = time.time()
    path = run(args.manifest, args.covariates, args.data_dir, args.blacklist, args.old_run_dir,
               args.out_dir, workers=args.workers)
    print(f"oldd_split {args.out_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
