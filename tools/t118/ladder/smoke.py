"""t118 ladder — the end-to-end smoke: every CLI of the chain, on synthetic products, on CPU.

    python tools/t118/ladder/smoke.py <work_dir> <rung> [--mpl-python PATH] [--steps 40]
                                      [--seeds 3] [--jobs N]

In `<work_dir>`: `synth.make_products` -> `products/` (tracks T1 H3K27ac, T2 DNase-seq), a small
`blacklist.bed` and an empty `refs_empty.tsv`; then for g in (T1, all) x both spaces x the three
models x seeds 0..seeds-1, each run through the CLIs' own `main(argv)` with the argument lists
    train.py train <manifest> <covariates> <products> <runs> <rung> <g> <space> <model> <seed>
        --max-steps <steps> --device cpu
    score.py trained <manifest> <covariates> <products> <blacklist> <runs>/<run> --workers 1
    score.py law     <manifest> <covariates> <products> <blacklist> <runs>/<run> --workers 1
inside `--jobs` persistent worker processes (one thread each; a python start per command would
cost ~5 s of torch + candi imports, a third of the smoke), then — as subprocesses — `aggregate.py`
(the empty refs), `aggregate.py qm-curves`, `figures.py --check-schema`, `report.py`, and
`figures.py <agg> <rung> --refs-qm` under `--mpl-python` (skipped with a note when not given). It asserts the files the acceptance
criteria name, writes `smoke_timing.json` (wall seconds per phase and per run) and prints
`SMOKE OK <rung>`.

`<work_dir>` is recreated when it holds this script's marker file; a non-empty directory without
the marker is refused rather than deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:
    sys.path.insert(0, str(_T118))

from ladder import pairs, synth  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LADDER = REPO / "tools" / "t118" / "ladder"
MARKER = ".t118_smoke"
G_SMOKE = ("T1", "all")
FIG_FILES = ("fig1_ladder", "fig2_knob_heatmap", "fig3_law_grid", "fig4_depth_law",
             "fig5_learned_f", "fig6_meta_profiles", "fig7_snippets", "fig8_calibration",
             "fig9_checks")
#: a few blacklist intervals on the score and validation chromosomes (bp, half-open)
BLACKLIST = (("chr19", 10_000, 12_500), ("chr21", 5_000, 5_100), ("chr22", 0, 1_000))


def _env() -> dict:
    env = dict(os.environ)
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list, log: Path) -> float:
    """Run `cmd`, output to `log`; raise with the log's tail on failure. Returns wall seconds."""
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, env=_env())
    if r.returncode != 0:
        tail = "".join(Path(log).read_text("utf-8").splitlines(keepends=True)[-30:])
        raise RuntimeError(f"exit {r.returncode}: {' '.join(str(c) for c in cmd)}\n{tail}")
    return time.time() - t0


def _prepare(work: Path) -> None:
    if work.exists() and any(work.iterdir()):
        if not (work / MARKER).is_file():
            raise SystemExit(f"{work} is not empty and has no {MARKER} marker; refusing to clear it")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / MARKER).write_text("written by tools/t118/ladder/smoke.py; the directory is scratch\n")


def run_argvs(rung: str, g: str, space: str, model: str, seed: int, steps: int,
              paths: dict) -> dict:
    """The three CLI argument lists of one run: {"run", "train", "score", "law"}."""
    name = f"{rung}_{g}_{space}_{model}_s{seed}"
    run_dir = paths["runs"] / name
    common = [paths["manifest"], paths["covariates"], paths["products"]]
    argv = {"train": ["train", *common, paths["runs"], rung, g, space, model, seed,
                      "--max-steps", steps, "--device", "cpu"],
            "score": ["trained", *common, paths["blacklist"], run_dir, "--workers", 1],
            "law": ["law", *common, paths["blacklist"], run_dir, "--workers", 1]}
    return {"run": name, "cost": 2 if g == "all" else 1,
            **{k: [str(v) for v in a] for k, a in argv.items()}}


def worker(spec_path) -> int:
    """One persistent worker: claims runs from the spec in order (an O_EXCL claim file per run, so
    the workers share the list without a coordinator) and runs each through `train.main` then
    `score.main` twice — the CLIs' own argument parsers, without a python start (torch + candi
    imports cost ~5 s) per command. Appends one timing line per run to the spec's result file."""
    spec = json.loads(Path(spec_path).read_text("utf-8"))
    from ladder import score, train
    claims = Path(spec["claims"])
    for r in spec["runs"]:
        try:
            os.close(os.open(claims / r["run"], os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        t = {"run": r["run"], "worker": spec["worker"]}
        for phase, fn in (("train", train.main), ("score", score.main), ("law", score.main)):
            print(f"$ {phase}.main {' '.join(r[phase])}", flush=True)
            t0 = time.time()
            rc = fn(list(r[phase]))
            if rc not in (0, None):
                raise SystemExit(f"{r['run']}: {phase} returned {rc}")
            t[phase] = time.time() - t0
        with open(spec["results"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(t) + "\n")
    return 0


def smoke(work_dir, rung: str, mpl_python=None, steps: int = 40, seeds: int = 3,
          jobs: int | None = None) -> dict:
    if rung not in pairs.RUNGS:
        raise ValueError(f"rung {rung!r} not in {pairs.RUNGS}")
    t_start = time.time()
    py = sys.executable
    work = Path(work_dir).resolve()
    _prepare(work)
    paths = {"products": work / "products", "runs": work / "runs", "agg": work / "agg",
             "logs": work / "logs", "blacklist": work / "blacklist.bed",
             "refs": work / "refs_empty.tsv"}
    for k in ("runs", "agg", "logs"):
        paths[k].mkdir(parents=True, exist_ok=True)
    timing: dict = {"rung": rung, "steps": steps, "seeds": seeds}

    t0 = time.time()
    paths["manifest"], paths["covariates"] = synth.make_products(paths["products"], seed=0)
    paths["blacklist"].write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in BLACKLIST))
    paths["refs"].write_text("")
    timing["products"] = time.time() - t0

    runs = [(g, space, model, seed) for g in G_SMOKE for space in pairs.SPACES
            for model in pairs.MODELS for seed in range(seeds)]
    n_jobs = max(1, int(jobs or min(8, os.cpu_count() or 1)))
    t0 = time.time()
    # the most expensive runs (the across-track g) first, so the workers finish together
    specs = sorted((run_argvs(rung, *r, steps, paths) for r in runs), key=lambda s: -s["cost"])
    claims, results = work / "claims", work / "run_timing.jsonl"
    claims.mkdir()
    spec_files = []
    for w in range(min(n_jobs, len(specs))):
        f = work / f"worker_{w}.json"
        f.write_text(json.dumps({"worker": w, "runs": specs, "claims": str(claims),
                                 "results": str(results)}))
        spec_files.append(f)
    with ThreadPoolExecutor(len(spec_files)) as ex:
        list(ex.map(lambda f: _run([py, Path(__file__).resolve(), "--worker", f],
                                   paths["logs"] / f"{f.stem}.log"), spec_files))
    per_run = [json.loads(line) for line in results.read_text("utf-8").splitlines()]
    if len(per_run) != len(specs):
        raise AssertionError(f"smoke: {len(per_run)} of {len(specs)} runs reported")
    timing["runs_wall"] = time.time() - t0
    timing["per_run"] = per_run
    timing["jobs"] = n_jobs
    for phase in ("train", "score", "law"):
        timing[f"{phase}_sum"] = sum(r[phase] for r in per_run)

    log = paths["logs"] / "agg.log"
    agg = paths["agg"]
    qm = agg / "qm_curves.json"
    timing["aggregate"] = _run([py, LADDER / "aggregate.py", paths["manifest"],
                                paths["covariates"], paths["runs"], paths["refs"], agg], log)
    timing["qm_curves"] = _run([py, LADDER / "aggregate.py", "qm-curves", paths["manifest"],
                                paths["products"], qm], log)
    timing["check_schema"] = _run([py, LADDER / "figures.py", "--check-schema", agg], log)
    timing["report"] = _run([py, LADDER / "report.py", agg, rung], log)
    if mpl_python:
        timing["figures"] = _run([mpl_python, LADDER / "figures.py", agg, rung, "--refs-qm", qm],
                                 log)
    else:
        print("note: no --mpl-python given; figures skipped (the candii env has no matplotlib)")

    # the files the acceptance criteria name
    rd = agg / rung
    need = [rd / "report.md", rd / f"checks_{rung}.json", agg / "results.json"]
    run0 = paths["runs"] / f"{rung}_T1_counts_real_s0"
    need += [run0 / f for f in ("ckpt.pt", "scores.json", "figdata.npz", "law.json")]
    if mpl_python:
        need += [rd / "figures" / f"{f}.png" for f in FIG_FILES]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise AssertionError("smoke: missing outputs:\n  " + "\n  ".join(missing))
    if mpl_python:
        pngs = sorted((rd / "figures").glob("fig*_*.png"))
        if len(pngs) != 9:
            raise AssertionError(f"smoke: {len(pngs)} figure PNGs in {rd / 'figures'}, expected 9")
    res = json.loads((agg / "results.json").read_text("utf-8"))
    want = {f"{rung}_{g}_{s}_{m}_s{sd}" for g, s, m, sd in runs}
    absent = sorted(want - set(res["runs_present"]))
    if absent:
        raise AssertionError(f"smoke: runs not in results.json runs_present: {absent}")
    no_law = sorted(want & set(res.get("runs_without_law", [])))
    if no_law:
        raise AssertionError(f"smoke: runs without law.json: {no_law}")
    timing["total"] = time.time() - t_start
    (work / "smoke_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker"] and len(argv) == 2:
        return worker(argv[1])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work_dir")
    ap.add_argument("rung", choices=pairs.RUNGS)
    ap.add_argument("--mpl-python", default=None,
                    help="a python with numpy + matplotlib for figures.py (no torch needed)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..N-1 (default 3)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="runs at a time, one thread each (default min(8, cpu count))")
    a = ap.parse_args(argv)
    t = smoke(a.work_dir, a.rung, a.mpl_python, a.steps, a.seeds, a.jobs)
    print(f"wall s: total {t['total']:.1f}, runs {t['runs_wall']:.1f} ({len(t['per_run'])} runs, "
          f"{t['jobs']} at a time; sums train {t['train_sum']:.1f} score {t['score_sum']:.1f} "
          f"law {t['law_sum']:.1f}), aggregate {t['aggregate']:.1f}, report {t['report']:.1f}"
          + (f", figures {t['figures']:.1f}" if "figures" in t else ""))
    print(f"SMOKE OK {a.rung}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
