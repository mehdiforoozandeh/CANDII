#!/usr/bin/env python3
"""Checks of task t392, read over `hpc run <cluster> '<cmd>'`. Nothing is ever written on a cluster.

    check.py probe  --cluster nibi --store <store root>
    check.py store  --cluster fir  --store <store root> --tables <eic_v1_2>
    check.py train  --cluster fir  --runs <arm dir> [--seeds 0 1 2]
    check.py scores --cluster fir  --runs <arm dir> --sub p16_sv [--seeds 0 1 2]
    check.py spend  <jobs.tsv> [<jobs.tsv> ...]          (local files; no cluster)

Each cluster check sends one read-only python3 program (standard library only) to the cluster, which
prints one JSON summary between two marker lines; the pass or fail is decided here. Exit codes: 0 pass,
1 fail, 69 when the tunnel is down (the message says to run `hpc up <cluster>`).

Store layout read (plan, C3): `<store>/<measurement id>/{starts.h5, starts_pr1.h5, starts_pr2.h5,
provenance.json}`. In `provenance.json` a gate record is any object with the keys `reference` and
`n_differing`; its `reference` file name (`counts.h5`, `pr1.h5`, `pr2.h5`) names the store file it gates.
The conversion floor of a file is the `conversion_floor_nmse` of the nearest object that holds its gate.
`wall_s` and `peak_rss_kb` are read wherever they first occur.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import subprocess
import sys
from pathlib import Path

TUNNEL_DOWN = 69
BEGIN, END = "@@T392_CHECK_BEGIN@@", "@@T392_CHECK_END@@"
STORE_FILES = {"counts.h5": "starts.h5", "pr1.h5": "starts_pr1.h5", "pr2.h5": "starts_pr2.h5"}
WINDOWS_SEEN = 80000
PARAM_TARGET, PARAM_TOL = 1_000_000, 0.02
TABLES_SHA256_PREFIX = "7a1a0325d00ddff3"     # eic_v1_2, plan C12c


class TunnelDown(Exception):
    pass


# --------------------------------------------------------------------------- remote side
# Each program reads sys.argv[1] (a JSON argument object) and prints one JSON object between the markers.

_COMMON = r'''
import json, os, sys
A = json.loads(sys.argv[1])
FILES = {"counts.h5": "starts.h5", "pr1.h5": "starts_pr1.h5", "pr2.h5": "starts_pr2.h5"}

def walk(obj, chain, found, first):
    if isinstance(obj, dict):
        chain = chain + [obj]
        for k in ("wall_s", "peak_rss_kb"):
            if k in obj and k not in first:
                first[k] = obj[k]
        if "reference" in obj and "n_differing" in obj:
            floor = None
            for anc in reversed(chain):
                if "conversion_floor_nmse" in anc:
                    floor = anc["conversion_floor_nmse"]
                    break
            found.append({"reference": os.path.basename(str(obj["reference"])),
                          "n_differing": obj["n_differing"], "result": obj.get("result"),
                          "floor": floor})
        for v in obj.values():
            walk(v, chain, found, first)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, chain, found, first)

def measurement(root, mid):
    d = os.path.join(root, mid)
    rec = {"id": mid, "present": sorted(f for f in FILES.values() if os.path.isfile(os.path.join(d, f)))}
    p = os.path.join(d, "provenance.json")
    if not os.path.isfile(p):
        rec["error"] = "no provenance.json"
        return rec
    try:
        prov = json.load(open(p))
    except Exception as e:
        rec["error"] = "provenance.json: %s" % e
        return rec
    gates, first = [], {}
    walk(prov, [], gates, first)
    rec["gates"] = gates
    rec.update(first)
    return rec
'''

_PROGRAMS = {
    "probe": _COMMON + r'''
root = A["store"]
ids = sorted(m for m in os.listdir(root) if os.path.isdir(os.path.join(root, m))) if os.path.isdir(root) else None
out = {"store_exists": ids is not None, "measurements": [measurement(root, m) for m in (ids or [])]}
''',
    "store": _COMMON + r'''
import csv
root = A["store"]
with open(os.path.join(A["tables"], "measurements.tsv"), newline="") as fh:
    ids = [r["measurement_id"] for r in csv.DictReader(fh, delimiter="\t")]
out = {"store_exists": os.path.isdir(root), "measurements": [measurement(root, m) for m in ids]}
''',
    "train": r'''
import json, os, sys
A = json.loads(sys.argv[1])
seeds = {}
for s in A["seeds"]:
    d = os.path.join(A["runs"], "seed%s" % s)
    rec = {"dir": os.path.isdir(d), "DONE": os.path.isfile(os.path.join(d, "DONE")),
           "ckpt": os.path.isfile(os.path.join(d, "ckpt.pt"))}
    p = os.path.join(d, "run.json")
    if os.path.isfile(p):
        r = json.load(open(p))
        rec.update({k: r.get(k) for k in ("done", "windows_seen", "param_count", "tables_sha256",
                                           "last_train_loss", "last_heldout_loss", "device")})
    seeds[str(s)] = rec
out = {"seeds": seeds}
''',
    "scores": r'''
import json, os, sys
A = json.loads(sys.argv[1])
seeds = {}
for s in A["seeds"]:
    d = os.path.join(A["runs"], "seed%s" % s, A["sub"])
    seeds[str(s)] = {"SCORED": os.path.isfile(os.path.join(d, "SCORED")),
                     "scores_json": os.path.isfile(os.path.join(d, "scores.json"))}
out = {"seeds": seeds}
''',
}


def remote_command(kind: str, args: dict) -> str:
    prog = _PROGRAMS[kind] + f'\nprint({BEGIN!r}); print(json.dumps(out)); print({END!r})\n'
    b64 = base64.b64encode(prog.encode()).decode()
    arg = json.dumps(args).replace("'", "'\"'\"'")
    return f"python3 -c \"$(echo {b64} | base64 -d)\" '{arg}'"


def hpc_run(cluster: str, cmd: str) -> subprocess.CompletedProcess:
    """The one place that reaches a cluster. Tests replace this."""
    return subprocess.run(["hpc", "run", cluster, cmd], capture_output=True, text=True)


def remote(cluster: str, kind: str, args: dict) -> dict:
    p = hpc_run(cluster, remote_command(kind, args))
    if p.returncode == TUNNEL_DOWN:
        raise TunnelDown(cluster)
    text = p.stdout or ""
    if BEGIN not in text or END not in text:
        raise RuntimeError(f"hpc run {cluster} gave no result (exit {p.returncode}): "
                           f"{(p.stderr or '').strip()[-500:]}")
    return json.loads(text.split(BEGIN, 1)[1].split(END, 1)[0])


# --------------------------------------------------------------------------- decisions

def _gate_ok(g: dict) -> bool:
    return g.get("n_differing") == 0 and not str(g.get("result", "")).lower().startswith("fail")


def judge_measurement(rec: dict) -> dict:
    """{file: gate} for the three store files, plus the problems of this measurement."""
    gates = {}
    for g in rec.get("gates", []):
        f = STORE_FILES.get(g["reference"])
        if f is not None:
            gates[f] = g
    problems = []
    if "error" in rec:
        problems.append(rec["error"])
    for f in STORE_FILES.values():
        if f not in rec.get("present", []):
            problems.append(f"{f} missing")
        if f not in gates:
            problems.append(f"{f}: no gate record")
        elif not _gate_ok(gates[f]):
            problems.append(f"{f}: gate n_differing {gates[f].get('n_differing')}")
    return {"gates": gates, "problems": problems}


def cmd_probe(a) -> int:
    res = remote(a.cluster, "probe", {"store": a.store})
    ms = res["measurements"]
    if not res["store_exists"] or not ms:
        print(f"no measurements under {a.store}")
        return 1
    bad = 0
    for rec in ms:
        j = judge_measurement(rec)
        print(f"{rec['id']}  wall_s {rec.get('wall_s')}  peak_rss_kb {rec.get('peak_rss_kb')}")
        for f in STORE_FILES.values():
            g = j["gates"].get(f, {})
            print(f"  {f}: differing bins: {g.get('n_differing', 'NA')}  "
                  f"conversion_floor_nmse {g.get('floor', 'NA')}")
        for p in j["problems"]:
            print(f"  FAIL {p}")
        bad += bool(j["problems"])
    print(f"measurements {len(ms) - bad} of {len(ms)} pass")
    return 1 if bad else 0


def cmd_store(a) -> int:
    res = remote(a.cluster, "store", {"store": a.store, "tables": a.tables})
    ms = res["measurements"]
    n = len(ms)
    good = files = differing = 0
    failures = []
    for rec in ms:
        j = judge_measurement(rec)
        files += len(rec.get("present", []))
        differing += sum(int(g.get("n_differing") or 0) for g in j["gates"].values())
        if j["problems"]:
            failures.append(f"{rec['id']}: {'; '.join(j['problems'])}")
        else:
            good += 1
    print(f"measurements {good} of {n}")
    print(f"files {files} of {3 * n}")
    print(f"differing bins: {differing}")
    for f in failures[:20]:
        print(f"  FAIL {f}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 0 if n and good == n and files == 3 * n and differing == 0 else 1


def cmd_train(a) -> int:
    seeds = remote(a.cluster, "train", {"runs": a.runs, "seeds": a.seeds})["seeds"]
    done, ok = [], True
    for s in a.seeds:
        r = seeds[str(s)]
        is_done = bool(r.get("DONE") and r.get("ckpt") and r.get("done"))
        if is_done:
            done.append(str(s))
        pc = r.get("param_count")
        dev = (pc - PARAM_TARGET) / PARAM_TARGET if isinstance(pc, (int, float)) else None
        sha = str(r.get("tables_sha256") or "")
        print(f"seed{s}: done {is_done}  windows_seen {r.get('windows_seen')}  param_count {pc}"
              f" ({'NA' if dev is None else f'{100 * dev:+.2f}%'})  tables_sha256 {sha[:16] or 'NA'}"
              f"  train_loss {r.get('last_train_loss')}  heldout_loss {r.get('last_heldout_loss')}")
        ok &= (is_done and r.get("windows_seen") == WINDOWS_SEEN and dev is not None
               and abs(dev) <= PARAM_TOL and sha.startswith(TABLES_SHA256_PREFIX))
    print(f"seeds done: {' '.join(done)}")
    return 0 if ok else 1


def cmd_scores(a) -> int:
    seeds = remote(a.cluster, "scores", {"runs": a.runs, "sub": a.sub, "seeds": a.seeds})["seeds"]
    n = sum(1 for s in a.seeds if seeds[str(s)]["SCORED"] and seeds[str(s)]["scores_json"])
    for s in a.seeds:
        r = seeds[str(s)]
        print(f"seed{s}/{a.sub}: SCORED {r['SCORED']}  scores.json {r['scores_json']}")
    print(f"SCORED {n} of {len(a.seeds)}")
    return 0 if a.seeds and n == len(a.seeds) else 1


def cmd_spend(a) -> int:
    total, missing = 0.0, 0
    for path in a.jobs:
        sub = 0.0
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                v = (r.get("cputime_raw_s") or "").strip()
                if v:
                    sub += float(v) / 3600
                else:
                    missing += 1
        print(f"{path}\t{sub:.2f} CPU-h")
        total += sub
    print(f"total\t{total:.2f} CPU-h")
    if missing:
        print(f"rows without cputime_raw_s: {missing} (not collected; the total is a lower bound)")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--cluster", required=True, choices=("fir", "nibi"))
    p.add_argument("--store", required=True)
    p = sub.add_parser("store")
    p.add_argument("--cluster", required=True, choices=("fir", "nibi"))
    p.add_argument("--store", required=True)
    p.add_argument("--tables", required=True)
    for name in ("train", "scores"):
        p = sub.add_parser(name)
        p.add_argument("--cluster", required=True, choices=("fir", "nibi"))
        p.add_argument("--runs", required=True)
        p.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
        if name == "scores":
            p.add_argument("--sub", required=True)
    p = sub.add_parser("spend")
    p.add_argument("jobs", nargs="+")
    a = ap.parse_args(argv)
    fn = {"probe": cmd_probe, "store": cmd_store, "train": cmd_train,
          "scores": cmd_scores, "spend": cmd_spend}[a.cmd]
    try:
        return fn(a)
    except TunnelDown as e:
        print(f"tunnel to {e} is down: run `hpc up {e}`", file=sys.stderr)
        return TUNNEL_DOWN


if __name__ == "__main__":
    sys.exit(main())
