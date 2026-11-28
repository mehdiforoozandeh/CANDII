"""t396 C8: score the read-count reference and the read-start model in three spaces (h57).

    python tools/t396/score_spaces.py records --truth <dir> --model <name> --seed <k>
        (--arrays <dir> | --pred-store <dir>) --scores-json <p16_dev scores.json>
        --regions all|pair|even|odd [...] --out <records.jsonl> [--shard i/N]
        [--validate-truth] [--evalset <json>] [--n-iter 10,25,50,100,200]
    python tools/t396/score_spaces.py table --records <files...> --out <dir>
    python tools/t396/score_spaces.py check --records <files...> --out <check.tsv>

Inputs (the arrays contract of tools/t396/dump_native.py): `<truth>/<target id>.npz` (truth) and,
for the read-start model, `<arrays>/<target id>.npz` (`mu_sub`, `n_sub` [R, rb, 2] native, `mu`,
`n` [R, rb] converted, read from its prediction store, `region_indices`). The read-count reference
is read from its prediction store, key `item/<target>/leave_one_assay_out` (numpy only; the file
name is the sha1 of the key, as `predict_store.key_file`).

Spaces (tools/t396/spaces.py):
  overlap_count  y vs the stored converted (read start) or stored (reference) mu, n; ref pa_counts;
                 halves y_pr1, y_pr2 with `shares`.
  start_sum      strand-summed starts s vs the read-start model's native NB pooled over strands
                 (`read_start.strand_output`: `pool_nb` over the two channels for a stranded track,
                 channel 0 alone for an unstranded one), or vs the reference's
                 `deconv.richardson_lucy(mu, mix_kernel, n_iter)` with `deconv.match_size`, one row
                 per n_iter; ref pa_starts; halves s_pr1, s_pr2 (strand sums) with `shares_starts`.
  sum_1kb_count  `to_1kb` of the overlap arrays; size `pool_size_1kb(mu, n)`.
The read-start overlap and 1 kb rows use the stored converted mu, n (the arrays the scorer read),
so no conversion of mu_sub is redone here.

Regions: `all` = every region of the truth file (the evaluation set's order); `pair` = the
evaluation set's `pair_regions` (all regions when it has none); `even` / `odd` = region index mod 2.
RL acts along the bin axis of each region, so one RL on the union of the needed regions, sliced
afterwards, equals RL on each set.

`--validate-truth` adds rows with model "observed" (kind "validate"), per target, region set and
n_iter: `roundtrip_nmse` = nmse(richardson_lucy(forward(pa_starts)), pa_starts);
`conversion_floor_nmse` = nmse(forward(s strand sum, mix kernel), y); `rl_pr1_vs_s_pr2_nmse` =
nmse(richardson_lucy(y_pr1) * f2s / f1, s_pr2) and `raw_pr1_vs_s_pr2_nmse` = the same with
y_pr1 / w.sum() (reported, not a bar). nmse(a, b) = mean((a - b)^2) / var(b), the definition of
`onion_data.read_starts.conversion_floor`.

Gate (c), evaluated by `table` on the validate rows of region set `all` at n_iter 50: (1) on every
scorable assay the median over targets of `roundtrip_nmse` <= 0.05; (2) the median over all targets
of `conversion_floor_nmse` <= 0.10. If either fails, or no validate row covers a scorable assay,
the reference's start_sum rows carry the flag `not_interpreted`.

Records: JSON lines; every row carries kind, model, seed, path (`reference` | `read_start` |
`observed`), target, assay, cell, read_length, stranded, regions, space, n_iter (null where none),
scores_json, expected_skill (`scores.json["deviance"]["dev_imputation_skill"]["macro"]`),
expected_n_targets, scorable (`scores.json["scorable"]["imputation"]`), and the terms of
`spaces.score_target` (score rows) or the nmse values (validate rows).

table -> `table.tsv` (one row per space, model, regions, n_iter, macro): skill, r2,
  calibration_ratio, ratio_to_halves are means over seeds of the macro; floor = the macro of
  mean(floor / D0) (the Poisson plug-in floor in D0 units, the attainable D_model / D0); seed_range
  = max - min of the skill over seeds; per_seed = `seed:skill` list; flag (last column).
  `per_assay.tsv`, `per_read_length.tsv` (read_length 36 = `36bp`, else `longer`, -1 = `unknown`),
  `truth_validation.tsv` (per assay and `ALL`, per n_iter; last column the gate verdict at 50).
check -> one row per run (model, seed): the overlap_count, regions all, scorable-macro skill
  against expected_skill; `pass` when |diff| <= 0.002 (gate b), last column; exit 1 if any fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from t396 import deconv as DC  # noqa: E402
from t396 import spaces as SP  # noqa: E402

REGIME = "leave_one_assay_out"
REGION_SETS = ("all", "pair", "even", "odd")
GATE_B = 0.002
GATE_C_ROUNDTRIP = 0.05
GATE_C_FLOOR = 0.10
GATE_C_N_ITER = DC.N_ITER_PRIMARY
MACROS = ("scorable", "all_assays")
FLAG_ARTIFACT = "conversion_artifact"
FLAG_NOT_INTERPRETED = "not_interpreted"


# ------------------------------------------------------------------ small helpers
def parse_shard(spec: str) -> tuple:
    try:
        i, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard {spec!r}: want i/N")
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {spec!r}: need 0 <= i < N")
    return i, n


def nmse(a, b) -> float:
    """mean((a - b)^2) / var(b) (onion_data.read_starts.conversion_floor)."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    v = float(b.var())
    return float(np.mean((a - b) ** 2) / v) if v > 0 else float("nan")


def key_file(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz"


def read_store(store_dir, target: str, shape) -> dict:
    """mu, n float64 [R, rb] of one imputation item of a prediction store."""
    key = f"item/{target}/{REGIME}"
    p = Path(store_dir) / key_file(key)
    with np.load(p) as z:
        if str(z["key"]) != key:
            raise ValueError(f"{p}: holds {z['key']!s}, not {key}")
        return {k: np.asarray(z[k], np.float64).reshape(shape) for k in ("mu", "n")}


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pair_indices(evalset: dict, chrom, start_bin) -> list:
    """Indices of the evaluation set's `pair_regions` among the truth regions (the rule of
    `predict_store.pair_region_indices`)."""
    n = len(chrom)
    pr = evalset.get("pair_regions")
    if pr is None:
        return list(range(n))
    pos = {(str(c), int(s)): i for i, (c, s) in enumerate(zip(chrom, start_bin))}
    out = []
    for x in pr:
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            i = int(x)
        else:
            i = pos.get((str(x[0]), int(x[1])), -1)
        if not 0 <= i < n:
            raise ValueError(f"pair_regions entry {x!r} is not a region of the truth file")
        out.append(i)
    return out


def region_sets(names, n_regions: int, pair: list | None) -> dict:
    out = {}
    for nm in names:
        if nm == "all":
            out[nm] = list(range(n_regions))
        elif nm == "pair":
            out[nm] = list(pair)
        elif nm == "even":
            out[nm] = list(range(0, n_regions, 2))
        elif nm == "odd":
            out[nm] = list(range(1, n_regions, 2))
        else:
            raise SystemExit(f"--regions {nm!r}: want one of {REGION_SETS}")
    return out


def pool_strands(mu_sub, n_sub, stranded: bool) -> tuple:
    """The start-space NB of `read_start.strand_output`: pool_nb over the two channels for a
    stranded track (clamps of `contract.nb_nll`), channel 0 alone for an unstranded one."""
    mu_sub = np.asarray(mu_sub, np.float64)
    n_sub = np.asarray(n_sub, np.float64)
    if not stranded:
        return mu_sub[..., 0], n_sub[..., 0]
    m = np.maximum(mu_sub, SP.POOL_MU_MIN)
    n = np.maximum(n_sub, SP.POOL_N_MIN)
    s = m.sum(-1)
    return s, s * s / (m * m / n).sum(-1)


def rl_grid(c, w, n_iters) -> dict:
    """{n_iter: richardson_lucy(c, w, n_iter)} for every n_iter, by continuing one run (each RL
    step depends on the current s only, so continuing from n1 steps to n2 equals n2 steps)."""
    out, s, done = {}, None, 0
    for k in sorted(set(int(x) for x in n_iters)):
        s = DC.richardson_lucy(c, w, k - done, s0=s)
        done = k
        out[k] = s
    return out


def scores_meta(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    dev = d.get("deviance") or {}
    mac = (dev.get("dev_imputation_skill") or {}).get("macro")
    sc = (d.get("scorable") or {}).get("imputation") or []
    return {"scores_json": str(path),
            "expected_skill": float(mac) if isinstance(mac, (int, float)) else None,
            "expected_n_targets": d.get("n_targets"),
            "scorable": sorted(sc),
            "evalset": d.get("evalset")}


# ------------------------------------------------------------------ records
def _row(base: dict, regions: str, space: str, n_iter, terms: dict) -> dict:
    r = dict(base)
    r.update({"regions": regions, "space": space, "n_iter": n_iter})
    r.update({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in terms.items()})
    return r


def target_rows(tr: dict, base: dict, sets: dict, *, mu_c, n_c, start_nb=None, n_iters=None) -> list:
    """Score rows of one target. `mu_c`, `n_c` [R, rb] overlap mean and size (R = the truth
    regions). Read start: `start_nb` = (mu_s, n_s) [R, rb]. Reference: `n_iters` (RL grid)."""
    y = tr["y"].astype(np.float64)
    y1, y2 = tr["y_pr1"].astype(np.float64), tr["y_pr2"].astype(np.float64)
    s = tr["s"].astype(np.float64).sum(-1)
    s1, s2 = tr["s_pr1"].astype(np.float64).sum(-1), tr["s_pr2"].astype(np.float64).sum(-1)
    pa_c = tr["pa_counts"].astype(np.float64)
    pa_s = tr["pa_starts"].astype(np.float64)
    sh, shs = tuple(tr["shares"]), tuple(tr["shares_starts"])
    rows = []
    starts = {}
    if start_nb is None:
        union = sorted(set().union(*sets.values()))
        w = DC.mix_kernel(tr["kernel"], tr["n_reads_strand"])
        for k, sk in rl_grid(mu_c[union], w, n_iters).items():
            full = np.zeros_like(mu_c)
            nfull = np.ones_like(mu_c)
            full[union] = sk
            nfull[union] = DC.match_size(sk, w, mu_c[union], n_c[union])
            starts[k] = (full, nfull)
    else:
        starts[None] = start_nb
    for nm, idx in sets.items():
        rows.append(_row(base, nm, "overlap_count", None,
                         SP.score_target(y[idx], mu_c[idx], n_c[idx], pa_c[idx], y1[idx], y2[idx], sh)))
        rows.append(_row(base, nm, "sum_1kb_count", None,
                         SP.score_target(SP.to_1kb(y[idx]), SP.to_1kb(mu_c[idx]),
                                         SP.pool_size_1kb(mu_c[idx], n_c[idx]), SP.to_1kb(pa_c[idx]),
                                         SP.to_1kb(y1[idx]), SP.to_1kb(y2[idx]), sh)))
        for k, (ms, ns) in starts.items():
            rows.append(_row(base, nm, "start_sum", k,
                             SP.score_target(s[idx], ms[idx], ns[idx], pa_s[idx], s1[idx], s2[idx], shs)))
    return rows


def validate_rows(tr: dict, base: dict, sets: dict, n_iters) -> list:
    """Truth validation rows of one target (module docstring)."""
    union = sorted(set().union(*sets.values()))
    w = DC.mix_kernel(tr["kernel"], tr["n_reads_strand"])
    y = tr["y"].astype(np.float64)
    s = tr["s"].astype(np.float64).sum(-1)
    pa = tr["pa_starts"].astype(np.float64)
    y1 = tr["y_pr1"].astype(np.float64)
    s2 = tr["s_pr2"].astype(np.float64).sum(-1)
    f1 = float(tr["shares"][0])
    f2s = float(tr["shares_starts"][1])
    scale = f2s / f1
    conv = DC.forward(s, w)
    rt = rl_grid(DC.forward(pa[union], w), w, n_iters)
    hp = rl_grid(y1[union], w, n_iters)
    pos = {r: i for i, r in enumerate(union)}
    rows = []
    for nm, idx in sets.items():
        j = [pos[r] for r in idx]
        common = {"conversion_floor_nmse": nmse(conv[idx], y[idx]),
                  "raw_pr1_vs_s_pr2_nmse": nmse(y1[idx] / w.sum() * scale, s2[idx])}
        for k in sorted(rt):
            t = dict(common)
            t["roundtrip_nmse"] = nmse(rt[k][j], pa[idx])
            t["rl_pr1_vs_s_pr2_nmse"] = nmse(hp[k][j] * scale, s2[idx])
            rows.append(_row(base, nm, "start_sum", k, t))
    return rows


def cmd_records(a) -> int:
    out = Path(a.out)
    if out.is_file():
        print(f"records: {out} exists, skipped", flush=True)
        return 0
    if (a.arrays is None) == (a.pred_store is None):
        raise SystemExit("give exactly one of --arrays (read-start model) or --pred-store (reference)")
    meta = scores_meta(a.scores_json)
    names = [x for v in a.regions for x in v.split(",") if x]
    n_iters = [int(x) for x in str(a.n_iter).split(",") if x]
    truth_files = sorted(Path(a.truth).glob("*.npz"))
    if not truth_files:
        raise SystemExit(f"--truth {a.truth}: no .npz files")
    i, nsh = parse_shard(a.shard)
    mine = [p for k, p in enumerate(truth_files) if k % nsh == i]
    evalset = None
    if "pair" in names:
        ev_path = a.evalset or meta["evalset"]
        if not ev_path:
            raise SystemExit("--regions pair needs --evalset (scores.json names none)")
        evalset = json.loads(Path(ev_path).read_text(encoding="utf-8"))
    path = "read_start" if a.arrays is not None else "reference"
    rows = []
    for p in mine:
        tr = load_npz(p)
        target = p.stem
        R, rb = tr["y"].shape
        pair = (pair_indices(evalset, [str(c) for c in tr["chrom"]], tr["start_bin"])
                if evalset is not None else None)
        sets = region_sets(names, R, pair)
        rl = int(tr["read_length"])
        stranded = bool(tr["stranded"])
        base = {"kind": "score", "model": a.model, "seed": int(a.seed), "path": path,
                "target": target, "assay": str(tr["assay"]), "cell": str(tr["cell"]),
                "read_length": rl, "stranded": stranded,
                **{k: meta[k] for k in ("scores_json", "expected_skill", "expected_n_targets",
                                        "scorable")}}
        if path == "reference":
            st = read_store(a.pred_store, target, (R, rb))
            rows += target_rows(tr, base, sets, mu_c=st["mu"], n_c=st["n"], n_iters=n_iters)
        else:
            m = load_npz(Path(a.arrays) / f"{target}.npz")
            ri = [int(x) for x in m["region_indices"]]
            need = sorted(set().union(*sets.values()))
            missing = sorted(set(need) - set(ri))
            if missing:
                raise SystemExit(f"{target}: model arrays lack regions {missing[:5]}...")
            mu_c, n_c = _place(m["mu"], ri, R, 1.0), _place(m["n"], ri, R, 1.0)
            ms, ns = pool_strands(m["mu_sub"], m["n_sub"], stranded)
            rows += target_rows(tr, base, sets, mu_c=mu_c, n_c=n_c,
                                start_nb=(_place(ms, ri, R, 1.0), _place(ns, ri, R, 1.0)))
        if a.validate_truth:
            vb = dict(base, kind="validate", model="observed", path="observed")
            rows += validate_rows(tr, vb, sets, n_iters)
        print(f"records: {target} done ({len(rows)} rows so far)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    print(f"records: wrote {len(rows)} rows to {out}", flush=True)
    return 0


def _place(x, ri, R, fill) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.full((R,) + x.shape[1:], fill, np.float64)
    out[ri] = x
    return out


# ------------------------------------------------------------------ reading records
def read_records(files) -> list:
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def _scorable(rows) -> list:
    out = set()
    for r in rows:
        out |= set(r.get("scorable") or [])
    return sorted(out)


def _fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NA" if not np.isfinite(v) else f"{float(v):.6g}"
    return str(v)


def write_tsv(path, cols, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(c)) for c in cols) + "\n")


def _floor_macro(per_recs: dict, assays) -> float:
    v = [float(np.mean([r["floor"] / r["D0"] for r in per_recs[a]])) for a in sorted(assays)
         if a in per_recs]
    return float(np.mean(v)) if v and all(np.isfinite(v)) else float("nan")


def _key(r) -> tuple:
    return (r["space"], r["model"], r["regions"], r["n_iter"])


# ------------------------------------------------------------------ gate (c)
def gate_c(val_rows, scorable) -> dict:
    """{"pass": bool | None, "rows": [...]} from validate rows (one per target, region set, n_iter;
    duplicates of one target are dropped). `pass` is None when a scorable assay has no row."""
    seen = {}
    for r in val_rows:
        seen[(r["target"], r["regions"], r["n_iter"])] = r
    rows_all = [r for r in seen.values() if r["regions"] == "all"]
    by = defaultdict(list)
    for r in rows_all:
        by[(r["assay"], r["n_iter"])].append(r)
        by[("ALL", r["n_iter"])].append(r)
    med = lambda rs, k: float(np.median([x[k] for x in rs])) if rs else float("nan")  # noqa: E731
    out_rows = []
    rt_ok = {}
    for (assay, k), rs in sorted(by.items(), key=lambda t: (t[0][0] == "ALL", t[0][0], t[0][1])):
        o = {"assay": assay, "n_iter": k, "n_targets": len(rs),
             "scorable": (assay in scorable) if assay != "ALL" else None,
             "roundtrip_nmse_median": med(rs, "roundtrip_nmse"),
             "conversion_floor_nmse_median": med(rs, "conversion_floor_nmse"),
             "rl_pr1_vs_s_pr2_nmse_median": med(rs, "rl_pr1_vs_s_pr2_nmse"),
             "raw_pr1_vs_s_pr2_nmse_median": med(rs, "raw_pr1_vs_s_pr2_nmse"),
             "roundtrip_pass": None, "conversion_floor_pass": None, "gate_c": "NA"}
        if k == GATE_C_N_ITER:
            o["roundtrip_pass"] = bool(o["roundtrip_nmse_median"] <= GATE_C_ROUNDTRIP)
            o["conversion_floor_pass"] = bool(o["conversion_floor_nmse_median"] <= GATE_C_FLOOR)
            if assay != "ALL":
                rt_ok[assay] = o["roundtrip_pass"]
                o["gate_c"] = "pass" if o["roundtrip_pass"] else "fail"
        out_rows.append(o)
    missing = [a for a in scorable if a not in rt_ok]
    allrow = [o for o in out_rows if o["assay"] == "ALL" and o["n_iter"] == GATE_C_N_ITER]
    if missing or not allrow:
        verdict = None
    else:
        verdict = all(rt_ok[a] for a in scorable) and bool(allrow[0]["conversion_floor_pass"])
    for o in allrow:
        o["gate_c"] = {None: "not_evaluated", True: "pass", False: "fail"}[verdict]
    if missing:
        for a in missing:
            out_rows.append({"assay": a, "n_iter": GATE_C_N_ITER, "n_targets": 0, "scorable": True,
                             "gate_c": "not_evaluated"})
    return {"pass": verdict, "rows": out_rows, "missing_scorable": missing}


TV_COLS = ["assay", "n_iter", "n_targets", "scorable", "roundtrip_nmse_median",
           "conversion_floor_nmse_median", "rl_pr1_vs_s_pr2_nmse_median",
           "raw_pr1_vs_s_pr2_nmse_median", "roundtrip_pass", "conversion_floor_pass", "gate_c"]


# ------------------------------------------------------------------ table
TABLE_COLS = ["space", "model", "regions", "n_iter", "macro", "skill", "r2", "calibration_ratio",
              "ratio_to_halves", "floor", "seed_range", "per_seed", "n_seeds", "n_assays", "flag"]
ASSAY_COLS = ["space", "model", "regions", "n_iter", "assay", "scorable", "n_targets", "skill",
              "r2", "calibration_ratio", "ratio_to_halves", "seed_range", "per_seed", "flag"]
RL_COLS = ["space", "model", "regions", "n_iter", "read_length_group", "macro", "n_targets",
           "n_assays", "skill", "r2", "calibration_ratio", "ratio_to_halves", "seed_range",
           "per_seed", "flag"]


def rl_group(rl) -> str:
    rl = int(rl)
    return "36bp" if rl == 36 else ("unknown" if rl < 0 else "longer")


def _seed_summary(per_seed: dict, metric="skill") -> dict:
    """Means over seeds of each metric; seed_range and per_seed of `metric`."""
    out = {}
    for m in SP.METRICS:
        v = [d[m] for d in per_seed.values()]
        out[m] = float(np.mean(v)) if v else float("nan")
    v = {s: d[metric] for s, d in sorted(per_seed.items())}
    fin = [x for x in v.values() if np.isfinite(x)]
    out["seed_range"] = (max(fin) - min(fin)) if len(fin) == len(v) and fin else float("nan")
    out["per_seed"] = ",".join(f"{s}:{_fmt(x)}" for s, x in v.items())
    out["n_seeds"] = len(v)
    return out


def _flags(space, path, n_iter, best, gate) -> str:
    if not (space == "start_sum" and path == "reference"):
        return ""
    f = [FLAG_ARTIFACT]
    if n_iter == DC.N_ITER_PRIMARY:
        f.append("primary")
    if best is not None and n_iter == best:
        f.append("best_over_grid")
    if gate is not True:
        f.append(FLAG_NOT_INTERPRETED)
    return ",".join(f)


def cmd_table(a) -> int:
    rows = read_records(a.records)
    score = [r for r in rows if r.get("kind") == "score"]
    val = [r for r in rows if r.get("kind") == "validate"]
    if not score:
        raise SystemExit("no score rows in --records")
    scorable = _scorable(score)
    gc = gate_c(val, scorable)
    out = Path(a.out)
    path_of = {r["model"]: r["path"] for r in score}

    # macro rows per (space, model, regions, n_iter) x seed
    groups = defaultdict(lambda: defaultdict(list))
    for r in score:
        groups[_key(r)][r["seed"]].append(r)
    agg = {}
    for key, by_seed in groups.items():
        agg[key] = {s: SP.aggregate(rs, scorable) for s, rs in by_seed.items()}
        for s, rs in by_seed.items():
            per = defaultdict(list)
            for x in rs:
                per[x["assay"]].append(x)
            agg[key][s]["floor"] = {"scorable": _floor_macro(per, [p for p in per if p in scorable]),
                                    "all_assays": _floor_macro(per, list(per))}
    # best_over_grid per (model, regions, macro) for the reference's start space
    best = {}
    for (space, model, regions, n_iter), d in agg.items():
        if space != "start_sum" or path_of.get(model) != "reference":
            continue
        for mac in MACROS:
            v = np.mean([x[mac]["skill"] for x in d.values()])
            k = (model, regions, mac)
            if np.isfinite(v) and (k not in best or v > best[k][1]):
                best[k] = (n_iter, v)
    order = lambda k: (SP.SPACES.index(k[0]), k[1], REGION_SETS.index(k[2]) if k[2] in REGION_SETS
                       else 9, -1 if k[3] is None else k[3])
    trows = []
    for key in sorted(agg, key=order):
        space, model, regions, n_iter = key
        for mac in MACROS:
            ps = {s: x[mac] for s, x in agg[key].items()}
            sm = _seed_summary(ps)
            b = best.get((model, regions, mac), (None,))[0]
            trows.append({"space": space, "model": model, "regions": regions, "n_iter": n_iter,
                          "macro": mac, **sm,
                          "floor": float(np.mean([x["floor"][mac] for x in agg[key].values()])),
                          "n_assays": int(np.mean([x[mac]["n_assays"] for x in agg[key].values()])),
                          "flag": _flags(space, path_of.get(model), n_iter, b, gc["pass"])})
    write_tsv(out / "table.tsv", TABLE_COLS, trows)

    arows = []
    for key in sorted(agg, key=order):
        space, model, regions, n_iter = key
        b = best.get((model, regions, "scorable"), (None,))[0]
        assays = sorted(set().union(*(x["per_assay"] for x in agg[key].values())))
        for assay in assays:
            ps = {s: x["per_assay"][assay] for s, x in agg[key].items() if assay in x["per_assay"]}
            sm = _seed_summary(ps)
            arows.append({"space": space, "model": model, "regions": regions, "n_iter": n_iter,
                          "assay": assay, "scorable": assay in scorable,
                          "n_targets": int(np.mean([p["n_targets"] for p in ps.values()])), **sm,
                          "flag": _flags(space, path_of.get(model), n_iter, b, gc["pass"])})
    write_tsv(out / "per_assay.tsv", ASSAY_COLS, arows)

    rgroups = defaultdict(lambda: defaultdict(list))
    for r in score:
        rgroups[_key(r) + (rl_group(r["read_length"]),)][r["seed"]].append(r)
    rrows = []
    for key in sorted(rgroups, key=lambda k: order(k[:4]) + (k[4],)):
        space, model, regions, n_iter, grp = key
        b = best.get((model, regions, "scorable"), (None,))[0]
        ag = {s: SP.aggregate(rs, scorable) for s, rs in rgroups[key].items()}
        n_t = int(np.mean([len(rs) for rs in rgroups[key].values()]))
        for mac in MACROS:
            sm = _seed_summary({s: x[mac] for s, x in ag.items()})
            rrows.append({"space": space, "model": model, "regions": regions, "n_iter": n_iter,
                          "read_length_group": grp, "macro": mac, "n_targets": n_t,
                          "n_assays": int(np.mean([x[mac]["n_assays"] for x in ag.values()])), **sm,
                          "flag": _flags(space, path_of.get(model), n_iter, b, gc["pass"])})
    write_tsv(out / "per_read_length.tsv", RL_COLS, rrows)
    write_tsv(out / "truth_validation.tsv", TV_COLS, gc["rows"])
    print(f"table: gate (c) {gc['pass']}; wrote {len(trows)} table rows to {out}", flush=True)
    return 0


# ------------------------------------------------------------------ check (gate b)
CHECK_COLS = ["model", "seed", "scores_json", "n_targets", "expected_n_targets", "expected",
              "computed", "diff", "status"]


def cmd_check(a) -> int:
    rows = [r for r in read_records(a.records) if r.get("kind") == "score"
            and r["space"] == "overlap_count" and r["regions"] == "all"]
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["seed"])].append(r)
    out = []
    for (model, seed), rs in sorted(by.items()):
        exp = {r["expected_skill"] for r in rs}
        sj = {r["scores_json"] for r in rs}
        en = {r.get("expected_n_targets") for r in rs}
        targets = {r["target"] for r in rs}
        comp = SP.aggregate(rs, _scorable(rs))["scorable"]["skill"]
        e = exp.pop() if len(exp) == 1 else None
        d = (comp - e) if (e is not None and np.isfinite(comp)) else float("nan")
        ok = bool(np.isfinite(d) and abs(d) <= GATE_B and len(rs) == len(targets))
        out.append({"model": model, "seed": seed, "scores_json": ";".join(sorted(sj)),
                    "n_targets": len(targets), "expected_n_targets": en.pop() if len(en) == 1 else None,
                    "expected": e, "computed": comp, "diff": d, "status": "pass" if ok else "fail"})
    write_tsv(a.out, CHECK_COLS, out)
    n_pass = sum(o["status"] == "pass" for o in out)
    print(f"check: {n_pass} of {len(out)} runs within {GATE_B} of scores.json", flush=True)
    return 0 if out and n_pass == len(out) else 1


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("--truth", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--seed", required=True, type=int)
    r.add_argument("--arrays")
    r.add_argument("--pred-store")
    r.add_argument("--scores-json", required=True)
    r.add_argument("--regions", nargs="+", default=["all"],
                   help="one or more of all, pair, even, odd (space or comma separated)")
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--validate-truth", action="store_true")
    r.add_argument("--evalset", help="evaluation set json (default: scores.json's `evalset`)")
    r.add_argument("--n-iter", default=",".join(str(x) for x in DC.N_ITER_GRID))
    t = sub.add_parser("table")
    t.add_argument("--records", nargs="+", required=True)
    t.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--records", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"records": cmd_records, "table": cmd_table, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
