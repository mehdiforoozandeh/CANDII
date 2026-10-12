"""t125 — the local end-to-end smokes: every CLI of part 1 and part 2, on synthetic products, on CPU.

    python tools/t125/smoke.py <work_dir> eta --rung <rung> [--steps 40] [--seeds 3] [--jobs N]
    python tools/t125/smoke.py <work_dir> h   [--steps 40] [--jobs N]
    python tools/t125/smoke.py <work_dir> all [--rung A] [--steps 40] [--seeds 3] [--jobs N]

Every command is a subprocess with the argument list the SLURM scripts (`slurm/t125/*.sh`) use,
except that `train-index` becomes `train` with the explicit (rung, g, model, seed): the synthetic
products have tracks T1 (H3K27ac, a matched control) and T2 (DNase-seq), not the t112 tracks the
432-row table names. Each python process runs one thread (OMP/MKL = 1) and `--workers 1` (on macOS
a forked torch worker can segfault); the parallelism is across commands, `--jobs` at a time.

**eta** (in `<work_dir>`): `ladder.synth.make_products` -> `products/`, a small `blacklist.bed`, an
empty `old_grid_empty.tsv`; then for g in (T1, all) x (real, nocov) x seeds 0..seeds-1:
    eta/train.py train <manifest> <covariates> <products> <runs> <rung> <g> <model> <seed>
        --max-steps <steps> --device cpu
    eta/score.py trained <manifest> <covariates> <products> <blacklist> <runs>/<run> --workers 1
    eta/score.py law     <manifest> <covariates> <products> <blacklist> <runs>/<run> --workers 1
        --law-scope depth
then `eta/aggregate.py <manifest> <covariates> <runs> <agg> --old-grid-tsv <empty>` (no vs_oldgrid
rows, not an error). Asserts `agg/{results.json, grid.md, results_summary.tsv, depth_law.tsv}`,
every run's TRAIN_DONE, ckpt.pt, scores.json, SCORE_DONE, law.json, LAW_DONE, and that
results.json lists every run present and none without law.json. Prints `SMOKE OK eta`.

**h** (in `<work_dir>`): products + `hmap.synth.add_controls` (T1 controls; the measured-read-length
table `control_read_length.tsv`), `hmap/controls.py cache` for every product into `ctl_cache/`, a
3-step eta run `A_T1_counts_real_s0` (the f) into `eta_runs/`, then under `h/` (h_train.sh layout):
    hmap/train.py train ... runs_true affine 1 0 true 0, ... runs_true mlp 5 1 true 0,
    hmap/train.py train ... runs_fnb formula_mlp 5 0 fnb 0 --fnb-run eta_runs/A_T1_counts_real_s0
        (each --max-steps <steps> --device cpu --control-rl-tsv control_read_length.tsv)
    hmap/score.py alone ... <h_spec> h/alone --workers 1   (formula:bin, formula:frag, the 2 true h)
    hmap/score.py chain ... eta_runs/A_T1_counts_real_s0 h/chain --h x5 --workers 1
    hmap/aggregate.py <manifest> <covariates> h h/agg --old-grid-tsv <empty> --eta-runs eta_runs
    summary.py eta_agg_absent h/agg GRID_SUMMARY.md   (no part-1 aggregate in this smoke)
Asserts the alone ALONE_DONE files, the chain CHAIN_DONE, `h/agg/{results_h.json, grid_h.md,
h_alone_summary.tsv, h_chain_summary.tsv, h_repair.tsv, h_vs_oldgrid.tsv}` and GRID_SUMMARY.md.
Prints `SMOKE OK h`.

**all**: the eta smoke in `<work_dir>/eta`, the h smoke in `<work_dir>/h`, then
`summary.py eta/agg h/h/agg GRID_SUMMARY.md`. Prints `SMOKE OK all`.

`<work_dir>` is recreated when it holds this script's marker file; a non-empty directory without
the marker is refused rather than deleted. `smoke_timing.json` holds the wall seconds per phase.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS / "t125"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import synth  # noqa: E402
from eta import pairs as ep  # noqa: E402
from hmap import synth as hsynth  # noqa: E402
from hmap import tasks as htasks  # noqa: E402

REPO = _TOOLS.parent
T125 = _TOOLS / "t125"
ETA = T125 / "eta"
HMAP = T125 / "hmap"
SUMMARY = T125 / "summary.py"
MARKER = ".t125_smoke"
G_SMOKE = ("T1", "all")
#: a few blacklist intervals on the score and validation chromosomes (bp, half-open)
BLACKLIST = (("chr19", 10_000, 12_500), ("chr21", 5_000, 5_100), ("chr22", 0, 1_000))
F_RUN = "A_T1_counts_real_s0"
F_STEPS = 3
#: (variant, rung, noise, source) of the learned h the h smoke trains
H_TRAIN = (("affine", 1, 0, "true"), ("mlp", 5, 1, "true"), ("formula_mlp", 5, 0, "fnb"))
ETA_AGG_FILES = ("results.json", "grid.md", "results_summary.tsv", "depth_law.tsv")
ETA_RUN_FILES = ("TRAIN_DONE", "ckpt.pt", "scores.json", "SCORE_DONE", "law.json", "LAW_DONE")
H_AGG_FILES = ("results_h.json", "grid_h.md", "h_alone_summary.tsv", "h_chain_summary.tsv",
               "h_repair.tsv", "h_vs_oldgrid.tsv")


def _env() -> dict:
    env = dict(os.environ)
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list, log: Path) -> float:
    """Run `cmd`, output to `log`; raise with the log's tail on a nonzero exit. Wall seconds."""
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, env=_env())
    if r.returncode != 0:
        tail = "".join(Path(log).read_text("utf-8").splitlines(keepends=True)[-30:])
        raise RuntimeError(f"exit {r.returncode}: {' '.join(str(c) for c in cmd)}\n{tail}")
    return time.time() - t0


def _parallel(fn, items: list, jobs: int) -> list:
    with ThreadPoolExecutor(max(1, min(jobs, len(items) or 1))) as ex:
        return list(ex.map(fn, items))


def _prepare(work: Path) -> None:
    if work.exists() and any(work.iterdir()):
        if not (work / MARKER).is_file():
            raise SystemExit(f"{work} is not empty and has no {MARKER} marker; refusing to clear it")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / MARKER).write_text("written by tools/t125/smoke.py; the directory is scratch\n")


def _products(work: Path) -> dict:
    """Synthetic products, the blacklist and the empty old-grid TSV under `work`."""
    paths = {"products": work / "products", "blacklist": work / "blacklist.bed",
             "old_empty": work / "old_grid_empty.tsv", "logs": work / "logs"}
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["manifest"], paths["covariates"] = synth.make_products(paths["products"], seed=0)
    paths["blacklist"].write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in BLACKLIST))
    paths["old_empty"].write_text("")
    return paths


def _missing(paths) -> list[str]:
    return [str(p) for p in paths if not Path(p).is_file()]


# ---------------------------------------------------------------------------------------------
# part 1
# ---------------------------------------------------------------------------------------------


def eta_argvs(rung: str, g: str, model: str, seed: int, steps: int, paths: dict) -> dict:
    """The three CLI commands of one eta run: {"run", "train", "score", "law"}."""
    py = sys.executable
    name = ep.run_name(rung, g, model, seed)
    run_dir = paths["runs"] / name
    common = [paths["manifest"], paths["covariates"], paths["products"]]
    cmds = {"train": [py, ETA / "train.py", "train", *common, paths["runs"], rung, g, model, seed,
                      "--max-steps", steps, "--device", "cpu"],
            "score": [py, ETA / "score.py", "trained", *common, paths["blacklist"], run_dir,
                      "--workers", 1],
            "law": [py, ETA / "score.py", "law", *common, paths["blacklist"], run_dir,
                    "--workers", 1, "--law-scope", "depth"]}
    return {"run": name, "cost": 2 if g == "all" else 1,
            **{k: [str(v) for v in c] for k, c in cmds.items()}}


def eta_agg_argv(paths: dict) -> list[str]:
    """`eta/aggregate.py` on the smoke's runs, the old grid an empty TSV (eta_agg.sh's form)."""
    return [str(c) for c in (sys.executable, ETA / "aggregate.py", paths["manifest"],
                             paths["covariates"], paths["runs"], paths["agg"],
                             "--old-grid-tsv", paths["old_empty"])]


def smoke_eta(work: Path, rung: str, steps: int = 40, seeds: int = 3, jobs: int = 4) -> dict:
    """The part-1 smoke in an already prepared `work`; returns the timing dict."""
    if rung not in ep.RUNGS:
        raise ValueError(f"rung {rung!r} not in {ep.RUNGS}")
    t_start = time.time()
    t0 = time.time()
    paths = _products(work)
    paths["runs"], paths["agg"] = work / "runs", work / "agg"
    paths["runs"].mkdir(parents=True, exist_ok=True)
    timing: dict = {"part": "eta", "rung": rung, "steps": steps, "seeds": seeds, "jobs": jobs,
                    "products": time.time() - t0}

    runs = [(g, model, seed) for g in G_SMOKE for model in ep.MODELS for seed in range(seeds)]
    # the across-track g first, so the workers finish together
    specs = sorted((eta_argvs(rung, *r, steps, paths) for r in runs), key=lambda s: -s["cost"])

    def one(spec):
        log = paths["logs"] / f"{spec['run']}.log"
        return {"run": spec["run"], **{ph: _run(spec[ph], log) for ph in ("train", "score", "law")}}

    t0 = time.time()
    per_run = _parallel(one, specs, jobs)
    timing["runs_wall"] = time.time() - t0
    timing["per_run"] = per_run

    timing["aggregate"] = _run(eta_agg_argv(paths), paths["logs"] / "agg.log")
    names = [s["run"] for s in specs]
    need = [paths["agg"] / f for f in ETA_AGG_FILES]
    need += [paths["runs"] / n / f for n in names for f in ETA_RUN_FILES]
    missing = _missing(need)
    if missing:
        raise AssertionError("smoke eta: missing outputs:\n  " + "\n  ".join(missing))
    res = json.loads((paths["agg"] / "results.json").read_text("utf-8"))
    absent = sorted(set(names) - set(res["runs_present"]))
    if absent:
        raise AssertionError(f"smoke eta: runs not in results.json runs_present: {absent}")
    no_law = sorted(set(names) & set(res.get("runs_without_law", [])))
    if no_law:
        raise AssertionError(f"smoke eta: runs without law.json: {no_law}")
    if any(r["reading"] == "vs_oldgrid" for r in res["readings"]):
        raise AssertionError("smoke eta: vs_oldgrid rows from an empty old-grid TSV")
    timing["total"] = time.time() - t_start
    (work / "smoke_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing


# ---------------------------------------------------------------------------------------------
# part 2
# ---------------------------------------------------------------------------------------------


def h_commands(paths: dict, steps: int) -> dict:
    """Every part-2 command of the h smoke, by phase (the lists the smoke runs)."""
    py = sys.executable
    man, cov, prod = paths["manifest"], paths["covariates"], paths["products"]
    ctl, rl, bl = paths["ctl_cache"], paths["rl_tsv"], paths["blacklist"]
    h_root, eta_runs = paths["h_root"], paths["eta_runs"]
    f_dir = eta_runs / F_RUN
    n_products = sum(1 for _ in open(man, encoding="utf-8")) - 1
    cmds: dict = {"ctl_cache": [[py, HMAP / "controls.py", "cache", man, prod, ctl, i]
                                for i in range(n_products)],
                  "f_train": [[py, ETA / "train.py", "train", man, cov, prod, eta_runs, "A", "T1",
                               "real", 0, "--max-steps", F_STEPS, "--device", "cpu"]]}
    train_true, train_fnb, specs = [], [], {}
    for variant, rung, noise, source in H_TRAIN:
        runs = h_root / f"runs_{source}"
        tag = "true" if source == "true" else F_RUN
        name = htasks.h_name(variant, rung, noise, tag, 0)
        specs[name] = runs / name
        c = [py, HMAP / "train.py", "train", man, cov, prod, ctl, runs, variant, rung, noise,
             source, 0, "--max-steps", steps, "--device", "cpu", "--control-rl-tsv", rl]
        if source == "fnb":
            train_fnb.append(c + ["--fnb-run", f_dir])
        else:
            train_true.append(c)
    cmds["h_train_true"], cmds["h_train_fnb"] = train_true, train_fnb
    alone = ["formula:bin", "formula:frag"] + [str(p) for n, p in specs.items()
                                               if n.endswith("_true_s0")]
    common = [man, cov, prod, ctl, bl]
    cmds["alone"] = [[py, HMAP / "score.py", "alone", *common, s, h_root / "alone",
                      "--workers", 1, "--control-rl-tsv", rl, "--device", "cpu"] for s in alone]
    chain = ["formula:bin", "formula:frag"] + [str(p) for p in specs.values()]
    cmds["chain"] = [[py, HMAP / "score.py", "chain", *common, f_dir, h_root / "chain",
                      *[x for s in chain for x in ("--h", s)], "--workers", 1,
                      "--control-rl-tsv", rl, "--device", "cpu"]]
    cmds["aggregate"] = [[py, HMAP / "aggregate.py", man, cov, h_root, h_root / "agg",
                          "--old-grid-tsv", paths["old_empty"], "--eta-runs", eta_runs]]
    cmds["h_names"] = {"alone": [htasks.formula_name("bin"), htasks.formula_name("frag")]
                       + [n for n in specs if n.endswith("_true_s0")],
                       "chain": [htasks.formula_name("bin"), htasks.formula_name("frag")]
                       + list(specs)}
    return {k: ([[str(x) for x in c] for c in v] if k != "h_names" else v)
            for k, v in cmds.items()}


def smoke_h(work: Path, steps: int = 40, jobs: int = 4, eta_agg: Path | None = None) -> dict:
    """The part-2 smoke in an already prepared `work`; returns the timing dict."""
    t_start = time.time()
    t0 = time.time()
    paths = _products(work)
    paths["rl_tsv"] = work / "control_read_length.tsv"
    hsynth.add_controls(paths["products"], paths["manifest"], seed=0, rl_tsv=paths["rl_tsv"])
    paths.update(ctl_cache=work / "ctl_cache", eta_runs=work / "eta_runs", h_root=work / "h")
    for k in ("ctl_cache", "eta_runs", "h_root"):
        paths[k].mkdir(parents=True, exist_ok=True)
    timing: dict = {"part": "h", "steps": steps, "jobs": jobs, "products": time.time() - t0}
    cmds = h_commands(paths, steps)
    logs = paths["logs"]

    def run_all(phase: str, items: list) -> float:
        t = time.time()
        _parallel(lambda ic: _run(ic[1], logs / f"{phase}_{ic[0]}.log"), list(enumerate(items)),
                  jobs)
        return time.time() - t

    timing["ctl_cache"] = run_all("ctl_cache", cmds["ctl_cache"])
    # the f and the true-X' h do not depend on each other
    timing["train_f_and_true_h"] = run_all("train1", cmds["f_train"] + cmds["h_train_true"])
    # the fnb h needs the f; alone needs the true h
    timing["train_fnb_and_alone"] = run_all("train2", cmds["h_train_fnb"] + cmds["alone"])
    timing["chain"] = run_all("chain", cmds["chain"])
    timing["aggregate"] = run_all("agg", cmds["aggregate"])

    h_root, agg = paths["h_root"], paths["h_root"] / "agg"
    out_md = work / "GRID_SUMMARY.md"
    eta_agg = Path(eta_agg) if eta_agg is not None else work / "eta_agg_absent"
    timing["summary"] = _run([sys.executable, SUMMARY, eta_agg, agg, out_md], logs / "summary.log")

    names = cmds["h_names"]
    need = [h_root / "alone" / n / f for n in names["alone"] for f in ("scores.json", "ALONE_DONE")]
    need += [h_root / "chain" / F_RUN / n / "scores.json" for n in names["chain"]]
    need += [h_root / "chain" / F_RUN / "CHAIN_DONE"]
    need += [agg / f for f in H_AGG_FILES] + [out_md]
    need += [paths["ctl_cache"] / f"{pid}__counts.json" for pid in _control_pids(paths["manifest"])]
    missing = _missing(need)
    if missing:
        raise AssertionError("smoke h: missing outputs:\n  " + "\n  ".join(missing))
    res = json.loads((agg / "results_h.json").read_text("utf-8"))
    absent = sorted(set(names["alone"]) - set(res["alone_present"]))
    if absent:
        raise AssertionError(f"smoke h: alone h not in results_h.json: {absent}")
    chain_absent = sorted(set(names["chain"]) - set(res["chain_present"].get(F_RUN, [])))
    if chain_absent:
        raise AssertionError(f"smoke h: chain h not in results_h.json: {chain_absent}")
    timing["total"] = time.time() - t_start
    (work / "smoke_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing


def _control_pids(manifest) -> list[str]:
    with open(manifest, newline="", encoding="utf-8") as fh:
        return sorted(r["pid"] for r in csv.DictReader(fh, delimiter="\t")
                      if r.get("control_counts25_md5"))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def smoke(work_dir, part: str, rung: str = "A", steps: int = 40, seeds: int = 3,
          jobs: int | None = None) -> dict:
    work = Path(work_dir).resolve()
    jobs = max(1, int(jobs or min(8, os.cpu_count() or 1)))
    if part == "eta" and rung not in ep.RUNGS:
        raise ValueError(f"rung {rung!r} not in {ep.RUNGS}")
    _prepare(work)
    if part == "eta":
        return smoke_eta(work, rung, steps, seeds, jobs)
    if part == "h":
        return smoke_h(work, steps, jobs)
    if part != "all":
        raise ValueError(f"part {part!r} not in (eta, h, all)")
    t0 = time.time()
    (work / "eta").mkdir()
    (work / "h").mkdir()
    t_eta = smoke_eta(work / "eta", rung, steps, seeds, jobs)
    t_h = smoke_h(work / "h", steps, jobs, eta_agg=work / "eta" / "agg")
    out_md = work / "GRID_SUMMARY.md"
    t_sum = _run([sys.executable, SUMMARY, work / "eta" / "agg", work / "h" / "h" / "agg", out_md],
                 work / "summary.log")
    if not out_md.is_file():
        raise AssertionError(f"smoke all: no {out_md}")
    timing = {"part": "all", "eta": t_eta, "h": t_h, "summary": t_sum, "total": time.time() - t0}
    (work / "smoke_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work_dir")
    ap.add_argument("part", choices=("eta", "h", "all"))
    ap.add_argument("--rung", default="A", choices=ep.RUNGS, help="eta and all (default A)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3, help="eta seeds 0..N-1 (default 3)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="commands at a time, one thread each (default min(8, cpu count))")
    a = ap.parse_args(argv)
    t = smoke(a.work_dir, a.part, a.rung, a.steps, a.seeds, a.jobs)
    print(f"wall s: total {t['total']:.1f}")
    print(f"SMOKE OK {a.part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
