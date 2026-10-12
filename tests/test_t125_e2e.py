"""t125-E1 end to end: the two local smokes, the GRID_SUMMARY writer, and the pin of the task table.

Pinned against the plan (`.orchestrate/plan_t125.md`, chunk t125-E1, acceptance criteria 4–6 and
12) and handoff Phase 7, not against the implementation:

- `tools/t125/smoke.py <work> eta --rung A --steps 10 --seeds 1 --jobs 2` ends `SMOKE OK eta`
  within 4 min; `... h --steps 10 --jobs 2` ends `SMOKE OK h`; both write the files the plan names.
- the smokes call the CLIs in the SLURM scripts' forms (same subcommand, same positionals, the
  script's flags kept).
- `tools/t125/summary.py <eta_agg> <h_agg> <out_md>` writes every Phase-7 section header in order,
  "CRPS > 20" for part 1 and part 2, every value with its seed spread, the CRPS split, the
  "gain exceeds 2 × the larger seed wobble" flag, capped NB rows, the skipped readings with their
  reasons, no verdict word and no correlation in a table.
- the 432-row task table (`eta.pairs.tasks()` and the `pairs.py tasks` CLI the SLURM scripts read)
  has the md5 of the formula written out here.

The smokes run once per module (both at the same time); a second fixture adds, through the CLIs,
the arms that give every reading kind a row (an old grid made from the smoke's own outputs, two
more h), so each summary table is checked with rows in it.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
T125 = REPO / "tools" / "t125"
for _p in (REPO / "tools" / "t118", T125):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eta import pairs as ep  # noqa: E402

PY = sys.executable
SMOKE = T125 / "smoke.py"
SUMMARY = T125 / "summary.py"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
SECTIONS = [
    "What was run",
    "Drafted readings",
    "Part 1: each design against the old grid",
    "Part 1: each model against its no-covariates twin",
    "Part 1: mean or spread",
    "Part 1: depth law",
    "Part 1: exploding pairs (CRPS > 20)",
    "Part 2: complexity × inputs",
    "Part 2: push only against push + own noise",
    "Part 2: repair (h versions on f's NB)",
    "Part 2: the chain against the old grid's p results",
    "Part 2: exploding pairs (CRPS > 20)",
    "Notes on how it was run",
]
VERDICT = re.compile(r"\b(supported|refuted|pass|fail|confirmed)\b", re.IGNORECASE)
EXCEEDS = "gain exceeds 2 × the larger seed wobble"
F_RUN = "A_T1_counts_real_s0"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env() -> dict:
    env = dict(os.environ)
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    return env


def _cli(args: list, timeout: int = 600) -> subprocess.CompletedProcess:
    r = subprocess.run([PY, *[str(a) for a in args]], capture_output=True, text=True, env=_env(),
                       cwd=REPO, timeout=timeout)
    assert r.returncode == 0, f"{args}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------------------------
# the task-table pin (the table the Nibi arrays index)
# ---------------------------------------------------------------------------------------------

#: written out from the plan: 8 rungs x 8 g x 2 models x 3 seeds, rung slowest, seed fastest,
#: index = rung_i*48 + g_i*6 + model_i*3 + seed; then Aold at 384 + g_i*6 + model_i*3 + seed
PIN_RUNGS = ("A", "B", "C", "D", "A2", "B2", "C2", "D2")
PIN_G = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18", "all")
PIN_MODELS = ("real", "nocov")
PIN_MD5 = "2f7e1072765b2bf0d0019890a141a1fe"


def _formula_table() -> str:
    rows = []
    for ri, rung in enumerate(PIN_RUNGS):
        for gi, g in enumerate(PIN_G):
            for mi, m in enumerate(PIN_MODELS):
                for s in range(3):
                    rows.append((ri * 48 + gi * 6 + mi * 3 + s, rung, g, m, s))
    for gi, g in enumerate(PIN_G):
        for mi, m in enumerate(PIN_MODELS):
            for s in range(3):
                rows.append((384 + gi * 6 + mi * 3 + s, "Aold", g, m, s))
    lines = ["index\trung\tg\tmodel\tseed\trun_name"]
    lines += [f"{i}\t{r}\t{g}\t{m}\t{s}\t{r}_{g}_counts_{m}_s{s}" for i, r, g, m, s in rows]
    return "\n".join(lines) + "\n"


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def test_task_table_md5_equals_the_formula():
    want = _formula_table()
    assert len(want.splitlines()) == 433
    assert _md5(want) == PIN_MD5
    keys = ("index", "rung", "g", "model", "seed", "run_name")
    got = "\t".join(keys) + "\n" + "".join(
        "\t".join(str(t[k]) for k in keys) + "\n" for t in ep.tasks())
    assert _md5(got) == PIN_MD5
    cli = _cli([T125 / "eta" / "pairs.py", "tasks", FIXTURE_MANIFEST]).stdout
    assert _md5(cli) == PIN_MD5
    by_index = {int(ln.split("\t")[0]): ln.split("\t")[-1] for ln in cli.splitlines()[1:]}
    assert by_index[18] == "A_C19M16_counts_real_s0"
    assert by_index[402] == "Aold_C19M16_counts_real_s0"


# ---------------------------------------------------------------------------------------------
# the smokes
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smokes(tmp_path_factory):
    """Both smokes at once (subprocesses); {part: {work, out, err, rc, seconds}}."""
    root = tmp_path_factory.mktemp("t125_e2e")
    cmds = {"eta": [PY, SMOKE, root / "eta", "eta", "--rung", "A", "--steps", "10", "--seeds", "1",
                    "--jobs", "2"],
            "h": [PY, SMOKE, root / "h", "h", "--steps", "10", "--jobs", "2"]}
    procs, t0 = {}, {}
    for part, c in cmds.items():
        t0[part] = time.time()
        procs[part] = subprocess.Popen([str(x) for x in c], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=_env(), cwd=REPO)
    out = {}
    for part, p in procs.items():
        so, se = p.communicate(timeout=900)
        out[part] = {"work": root / part, "out": so, "err": se, "rc": p.returncode,
                     "seconds": time.time() - t0[part]}
    out["root"] = root
    return out


def test_eta_smoke_ok_within_4_min(smokes):
    s = smokes["eta"]
    assert s["rc"] == 0, s["err"][-3000:]
    assert _last_line(s["out"]) == "SMOKE OK eta"
    assert s["seconds"] <= 240, f"eta smoke took {s['seconds']:.0f} s"


def test_eta_smoke_files(smokes):
    w = smokes["eta"]["work"]
    for f in ("results.json", "grid.md", "results_summary.tsv", "depth_law.tsv"):
        assert (w / "agg" / f).is_file(), f
    runs = [f"A_{g}_counts_{m}_s0" for g in ("T1", "all") for m in ("real", "nocov")]
    for r in runs:
        for f in ("TRAIN_DONE", "ckpt.pt", "scores.json", "SCORE_DONE", "law.json", "LAW_DONE"):
            assert (w / "runs" / r / f).is_file(), (r, f)
    res = json.loads((w / "agg" / "results.json").read_text())
    assert set(runs) <= set(res["runs_present"])
    # an empty old-grid TSV gives no vs_oldgrid row and no error
    assert not [r for r in res["readings"] if r["reading"] == "vs_oldgrid"]
    law = json.loads((w / "runs" / runs[0] / "law.json").read_text())
    assert law["t125"]["law_scope"] == "depth"


def test_h_smoke_ok(smokes):
    s = smokes["h"]
    assert s["rc"] == 0, s["err"][-3000:]
    assert _last_line(s["out"]) == "SMOKE OK h"


def test_h_smoke_files(smokes):
    w = smokes["h"]["work"]
    h = w / "h"
    alone = ["h_formula_bin", "h_formula_frag", "h_affine_r1_push_true_s0",
             "h_mlp_r5_noise_true_s0"]
    for n in alone:
        assert (h / "alone" / n / "ALONE_DONE").is_file(), n
    chain = alone + [f"h_formula_mlp_r5_push_{F_RUN}_s0"]
    for n in chain:
        assert (h / "chain" / F_RUN / n / "scores.json").is_file(), n
    assert sorted((h / "chain" / F_RUN / "CHAIN_DONE").read_text().split()) == sorted(chain)
    for f in ("results_h.json", "grid_h.md", "h_alone_summary.tsv", "h_chain_summary.tsv",
              "h_repair.tsv", "h_vs_oldgrid.tsv"):
        assert (h / "agg" / f).is_file(), f
    assert (w / "eta_runs" / F_RUN / "TRAIN_DONE").is_file()
    assert (w / "GRID_SUMMARY.md").is_file()
    # one control cache entry per product with a control (synthetic track T1), none for T2
    with open(w / "products" / "MANIFEST.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        assert (w / "ctl_cache" / f"{r['pid']}__counts.json").is_file() == bool(
            r["control_counts25_md5"]), r["pid"]
    assert any(r["control_counts25_md5"] for r in rows)


def test_smoke_refuses_a_dir_without_its_marker(tmp_path):
    smoke = _load("t125_smoke_mod", SMOKE)
    d = tmp_path / "keep"
    d.mkdir()
    (d / "precious.txt").write_text("x")
    with pytest.raises(SystemExit):
        smoke.smoke(d, "eta", "A", steps=1, seeds=1, jobs=1)
    assert (d / "precious.txt").read_text() == "x"


# ---------------------------------------------------------------------------------------------
# the smokes call the CLIs as the SLURM scripts do
# ---------------------------------------------------------------------------------------------


def _shape(tokens: list[str]) -> tuple:
    """(script name, subcommand or None, number of positionals, flag names) of one command."""
    i = next(k for k, t in enumerate(tokens) if t.endswith(".py"))
    script = Path(tokens[i]).name
    rest = tokens[i + 1:]
    sub = rest[0] if rest and not rest[0].startswith("-") and script in (
        "train.py", "score.py", "controls.py", "pairs.py", "tasks.py") else None
    pos, flags, k = 0, set(), 1 if sub else 0
    while k < len(rest):
        if rest[k].startswith("--"):
            flags.add(rest[k])
            k += 2
        else:
            pos += 1
            k += 1
    return script, sub, pos, flags


def _dry(script: str, args: list, idx=None) -> list[str]:
    env = {**_env(), "DRY_RUN": "1", "PYTHON": PY}
    if idx is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(idx)
    r = subprocess.run(["bash", str(REPO / "slurm" / "t125" / script), *[str(a) for a in args]],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr
    return [ln.split() for ln in r.stdout.splitlines() if ln.strip()]


def _same_form(ours: list[str], script_line: list[str]) -> None:
    so, ss = _shape(ours), _shape(script_line)
    assert so[:3] == ss[:3], (so, ss)
    # the script's flags are kept (values may differ: workers, paths); extra local flags allowed
    assert ss[3] <= so[3], (so, ss)


def test_smoke_commands_have_the_slurm_forms(tmp_path):
    smoke = _load("t125_smoke_mod2", SMOKE)
    man = tmp_path / "MANIFEST.tsv"
    shutil.copyfile(FIXTURE_MANIFEST, man)
    paths = {k: tmp_path / k for k in ("products", "covariates", "blacklist", "runs", "ctl_cache",
                                       "rl_tsv", "h_root", "eta_runs", "old_empty")}
    paths["manifest"] = man
    eta = smoke.eta_argvs("A", "T1", "real", 0, 10, paths)
    train_cpu = _dry("eta_train_cpu.sh", [REPO, "/nonexistent", "/cache", "/out"], idx=18)
    _same_form(eta["score"], train_cpu[1])
    law = _dry("eta_law.sh", [REPO, "/nonexistent", "/cache", "/bl.bed", "/out"], idx=18)
    _same_form(eta["law"], law[0])
    assert eta["law"][eta["law"].index("--law-scope") + 1] == "depth"
    assert eta["train"][eta["train"].index("--device") + 1] == "cpu"
    h = smoke.h_commands(paths, 10)
    alone = _dry("h_alone.sh", [REPO, "/nonexistent", "/cache", "/ctl", "/bl.bed", "/out"], idx=0)
    for c in h["alone"]:
        _same_form(c, alone[0])
    ctl = _dry("h_ctl_cache.sh", [REPO, "/nonexistent", "/ctl"], idx=0)
    _same_form(h["ctl_cache"][0], ctl[0])
    agg = _dry("h_agg.sh", [REPO, "/nonexistent", "/out", "/old.tsv", "/eta_runs"])
    _same_form(h["aggregate"][0], [t for t in agg[0] if t != "--old-grid-json"
                                   and not t.endswith("results.json")])
    eagg = _dry("eta_agg.sh", [REPO, "/nonexistent", "/out", "/old.tsv"])
    paths["agg"] = tmp_path / "agg"
    _same_form(smoke.eta_agg_argv(paths), [t for t in eagg[0] if t != "--old-grid-json"
                                           and not t.endswith("results.json")])
    # every learned-h score runs one worker (macOS: a forked torch forward can segfault)
    for c in h["alone"] + h["chain"] + [eta["score"], eta["law"]]:
        assert c[c.index("--workers") + 1] == "1"


# ---------------------------------------------------------------------------------------------
# the summary writer
# ---------------------------------------------------------------------------------------------


def _check_summary(text: str) -> None:
    heads = [ln[3:] for ln in text.splitlines() if ln.startswith("## ")]
    assert heads == SECTIONS
    assert len(re.findall(r"CRPS > 20", text)) >= 2
    p1 = text.split("## Part 1: exploding pairs (CRPS > 20)")[1].split("## Part 2")[0]
    p2 = text.split("## Part 2: exploding pairs (CRPS > 20)")[1].split("## Notes")[0]
    assert "CRPS > 20" in p1 and "non-finite" in p1
    assert "CRPS > 20" in p2 and "non-finite" in p2
    assert "Capped NB rows" in p2
    assert not VERDICT.search(text), VERDICT.findall(text)
    for ln in text.splitlines():
        if ln.startswith("|"):
            assert "spearman" not in ln.lower() and "correlation" not in ln.lower(), ln


def _numbers_have_spread(section: str, cols: list[str]) -> int:
    """Every non-empty cell of the named value columns is `x ± w` or `x (one seed)`."""
    lines = [ln for ln in section.splitlines() if ln.startswith("|")]
    if not lines:
        return 0
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    idx = [head.index(c) for c in cols]
    n = 0
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        for i in idx:
            if cells[i] != "—":
                assert " ± " in cells[i] or cells[i].endswith("(one seed)"), (cols, cells[i])
                n += 1
    return n


def _section(text: str, title: str) -> str:
    return text.split(f"## {title}\n")[1].split("\n## ")[0]


def test_summary_on_the_smoke_outputs(smokes, tmp_path):
    out = tmp_path / "GRID_SUMMARY.md"
    r = _cli([SUMMARY, smokes["eta"]["work"] / "agg", smokes["h"]["work"] / "h" / "agg", out])
    assert out.is_file() and "wrote" in r.stdout
    text = out.read_text()
    _check_summary(text)
    twin = _section(text, "Part 1: each model against its no-covariates twin")
    assert _numbers_have_spread(twin, ["real", "no-covariates twin"]) > 0
    assert "oracle-scaled" in twin and "scale error" in twin       # CRPS with its split
    assert EXCEEDS in twin
    assert _numbers_have_spread(_section(text, "Part 1: depth law"), ["value"]) > 0
    grid = _section(text, "Part 2: complexity × inputs")
    assert _numbers_have_spread(grid, ["CRPS on P, all bins", "CRPS on log(1 + P), all bins"]) > 0
    # the brief's h set gives no push/noise pair and no repair pair: skipped, with reasons
    notes = _section(text, "Notes on how it was run")
    assert "only the push arm was scored" in notes
    assert "no fnb-trained h of this version was scored in this f cell" in notes
    assert F_RUN in _section(text, "Part 2: exploding pairs (CRPS > 20)")
    # the h smoke's own summary (no part-1 aggregate) has every header too
    own = (smokes["h"]["work"] / "GRID_SUMMARY.md").read_text()
    assert [ln[3:] for ln in own.splitlines() if ln.startswith("## ")] == SECTIONS
    assert "No part-1 aggregate" in own and "Capped NB rows" in own
    assert not VERDICT.search(own)


@pytest.fixture(scope="module")
def extended(smokes):
    """Every reading kind with rows, through the CLIs: the eta runs re-aggregated with their own
    outputs as the old grid; two more h (affine r1 own noise and formula_mlp r5 push, true X′)
    scored alone and in the chain; the h re-aggregated against a p-space old grid made from the
    eta smoke's outputs (space relabelled pval)."""
    e, w = smokes["eta"]["work"], smokes["h"]["work"]
    _cli([T125 / "eta" / "aggregate.py", e / "products" / "MANIFEST.tsv",
          e / "products" / "covariates.tsv", e / "runs", e / "agg_self",
          "--old-grid-tsv", e / "agg" / "results_summary.tsv",
          "--old-grid-json", e / "agg" / "results.json"])
    man, cov, prod = w / "products" / "MANIFEST.tsv", w / "products" / "covariates.tsv", w / "products"
    ctl, rl, bl, h = w / "ctl_cache", w / "control_read_length.tsv", w / "blacklist.bed", w / "h"
    new = [("affine", 1, 1), ("formula_mlp", 5, 0)]
    for variant, rung, noise in new:
        _cli([T125 / "hmap" / "train.py", "train", man, cov, prod, ctl, h / "runs_true", variant,
              rung, noise, "true", 0, "--max-steps", 10, "--device", "cpu", "--control-rl-tsv", rl])
    _cli([T125 / "hmap" / "score.py", "alone", man, cov, prod, ctl, bl,
          h / "runs_true" / "h_affine_r1_noise_true_s0", h / "alone", "--workers", 1,
          "--control-rl-tsv", rl, "--device", "cpu"])
    specs = ["formula:bin", "formula:frag"] + [
        h / "runs_true" / n for n in ("h_affine_r1_push_true_s0", "h_mlp_r5_noise_true_s0",
                                      "h_affine_r1_noise_true_s0", "h_formula_mlp_r5_push_true_s0")
    ] + [h / "runs_fnb" / f"h_formula_mlp_r5_push_{F_RUN}_s0"]
    _cli([T125 / "hmap" / "score.py", "chain", man, cov, prod, ctl, bl, w / "eta_runs" / F_RUN,
          h / "chain", *[x for s in specs for x in ("--h", s)], "--workers", 1,
          "--control-rl-tsv", rl, "--device", "cpu"])
    old = smokes["root"] / "old_pval"
    old.mkdir(exist_ok=True)
    with open(e / "agg" / "results_summary.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(old / "results_summary.tsv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        wr.writeheader()
        wr.writerows({**r, "space": "pval"} for r in rows)
    per_pair = json.loads((e / "agg" / "results.json").read_text())["per_pair"]
    (old / "results.json").write_text(json.dumps(
        {"per_pair": [{**r, "space": "pval"} for r in per_pair]}))
    _cli([T125 / "hmap" / "aggregate.py", man, cov, h, h / "agg_x",
          "--old-grid-tsv", old / "results_summary.tsv", "--old-grid-json", old / "results.json",
          "--eta-runs", w / "eta_runs"])
    out = smokes["root"] / "GRID_SUMMARY_x.md"
    _cli([SUMMARY, e / "agg_self", h / "agg_x", out, "--old-grid-tsv",
          old / "results_summary.tsv"])
    return out.read_text()


def test_summary_every_table_has_rows(extended):
    text = extended
    _check_summary(text)
    design = _section(text, "Part 1: each design against the old grid")
    assert _numbers_have_spread(design, ["t125", "old grid"]) > 0
    assert "oracle-scaled" in design and EXCEEDS in design
    ms = _section(text, "Part 1: mean or spread")
    assert "vs_oldgrid" in ms and "beatstwin" in ms
    pvn = _section(text, "Part 2: push only against push + own noise")
    assert _numbers_have_spread(pvn, ["push only", "push + own noise"]) > 0
    assert "| alone |" in pvn and "CRPS on log(1 + P), all bins" in pvn
    rep = _section(text, "Part 2: repair (h versions on f's NB)")
    assert _numbers_have_spread(rep, ["true-X′ h", "fnb h"]) > 0
    assert "formula_mlp" in rep
    vs = _section(text, "Part 2: the chain against the old grid's p results")
    assert _numbers_have_spread(vs, ["chain h", "old grid"]) > 0
    assert "CRPS split: oracle-scaled part" in vs and "CRPS split: scale-error part" in vs
    assert "h_mlp_r5_noise_true" not in vs                     # own noise: never vs the old grid
    assert "md5" in _section(text, "What was run")


def test_summary_without_aggregates_writes_every_header(tmp_path):
    out = tmp_path / "s.md"
    _cli([SUMMARY, tmp_path / "no_eta", tmp_path / "no_h", out])
    text = out.read_text()
    assert [ln[3:] for ln in text.splitlines() if ln.startswith("## ")] == SECTIONS
    assert "No part-1 aggregate" in text and "No part-2 aggregate" in text
    assert not VERDICT.search(text)


def test_summary_cleans_verdict_words_from_input_text(smokes, tmp_path):
    src = smokes["h"]["work"] / "h" / "agg" / "summary_data_h.json"
    d = json.loads(src.read_text())
    d["readings_skipped"].append({"reading": "vs_oldgrid", "scope": "chain", "f_rung": "A",
                                  "f_g_version": "per_track", "mark_class": "narrow",
                                  "h_name": "h_x", "reason": "pair matching failed; pass the "
                                                              "JSON | it was not confirmed"})
    agg = tmp_path / "h_agg"
    agg.mkdir()
    (agg / "summary_data_h.json").write_text(json.dumps(d))
    out = tmp_path / "s.md"
    _cli([SUMMARY, smokes["eta"]["work"] / "agg", agg, out])
    text = out.read_text()
    assert not VERDICT.search(text)
    assert "pair matching did not work" in text


def test_summary_flags_and_spread_formatting():
    s = _load("t125_summary_mod", SUMMARY)
    assert s.pm(1.23456, 0.0123) == "1.235 ± 0.0123"
    assert s.pm(1.0, None) == "1 (one seed)"
    assert s.yn(True) == "yes" and s.yn(False, "no bar") == "no (no bar)"
    assert s.SECTIONS == tuple(SECTIONS)
