"""t152 chunk D6t -- build a tables directory (pinned interface B) from verified measurement dirs.

    python build_tables.py build --build-list <tsv> --priorities 1,2,3 --measurements <dir> \\
        --work <dir> --testbed-rows <tsv> --withheld <tsv> --ledger <tsv> --out <tables dir> \\
        [--superset-of <tables dir>] [--dry-run] [--workers N] [--recount-chrom none|chr21]

ROWS. For every `expected_measurements` token of the build-list rows of the chosen priorities, the
measurement directory `<measurements>/<measurement_id>/` gives its row: `meta.json` (the
`measurements.tsv` row that `align_views.py` writes). The row is kept only when
  * its experiment is a ledger experiment with `match = 1`, or a control that such an experiment
    names (`build_list.ledger_sets`, the D4 rule), and its `in_eic` equals that ledger role;
  * `align_views.verify_measurement` passes on the directory, and `align_views.experiment_checks`
    passes across the directories of its experiment (pooled = sum of the replicates). The chr21
    recount from the half tagAligns is off by default (`--recount-chrom none`): chunk D6b runs the
    full `align_views.py verify` with it on the same directories, and it decompresses every half.
A token with no verified directory goes to `missing.tsv` with its reason; nothing is fabricated.
A kept row whose expected products are not all on disk (e.g. no `signal.h5`) is kept and also
listed in `missing.tsv` with `products_missing`.

TESTBED. The rows of `import_testbed.py`'s `testbed_rows.tsv` enter through
`import_testbed.rows_for_tables(path, allow_blind=True)`: they keep `in_eic = blind` and are listed
in `evaluation_only.tsv` (`measurement_id, use`, the testbed's own `use` text). `OnionDataset`
refuses `in_eic = blind` rows in train and val modes. `layer = testbed` directories under
`--measurements` are never read as build-list rows.

QC. `nsc, rsc, nrf, pbc1, pbc2, frag_len` come from the FASTQ run directory
`<work>/<experiment>/<run arm>/qc.json` (`lib_complexity`, `align_enrich.xcor_score`), else
`rep<i>.xcor.qc`, for a measurement that is one replicate's own read set of that run: unit
`br<i>` (rep i), or unit `pooled` of a single-replicate experiment (rep 1), with arm `base`, `se`,
`crop36` or `crop50`. Every other row keeps its QC cells empty (derived arms and multi-replicate
pools differ in reads from any one replicate; chunk QCt fills them). `frip` stays empty (the build
calls no peaks). A value already in `meta.json` is kept.

PAIRS (`pairs.tsv`, pinned interface B; `pair_id` = `p` + 12 hex of sha1(rung|a|b), so ids are
stable across a superset build):
  pseudoreplicate  a = b = one measurement with halves, arm base (its pr1 against its pr2)
  isogenic         br<i> against br<j> of one experiment, arm base, one layer; `anisogenic` when
                   the ledger's `replication_type` of the experiment says so
  technical        br<i>tr<j> against br<i>tr<k>;  sequencing: br<i>tr<j>ln<k> against ...ln<m>
  other_cell_type  the pooled base views (uniform preferred over native; never testbed) of one
                   assay in two cells with no cell in common
  designed         every ordered (source a, target b) pair of two arms of one track (experiment,
                   unit pooled). Written only when both sides' halves derive from ONE read-level
                   split of the parent (`split_signature`, read from each half tagAlign's
                   `READSET.json`); a pair with a `layer = testbed` side, or whose halves were split
                   apart, goes to `pairs_shared_reads.tsv` (same columns, read by no scorer).
                   `subset_pair = 1` for two depth-only views (`build_list.is_subset_pair`).
                   A combination of `withheld.tsv` on a training track is not written at all; on a
                   validation or blind track it is written with `never_trained = 1`.
`split` is the more restricted EIC role of the two sides (train < val < blind).

OUTPUT. `--out` must not exist or hold nothing but `withheld.tsv` (frozen tables are never
overwritten); a `withheld.tsv` there must be byte-identical to `--withheld`. Written:
measurements.tsv pairs.tsv pairs_shared_reads.tsv withheld.tsv covariates.json missing.tsv
evaluation_only.tsv build_info.json. With `--superset-of`, every line of the older
`measurements.tsv` and `pairs.tsv` must reappear byte-identical, `withheld.tsv` must be identical
and each covariates vocabulary must extend the older one (ids stay); else exit 1 and nothing is
written. `--dry-run` prints the same counts and writes nothing.

ADD-ON (Amendment 4, chunk OXt). `build` is unchanged: an experiment with no ledger role, such as
a second experiment of PI ruling 33, still goes to `missing.tsv`.

    python build_tables.py addon --base <frozen tables> --other-build-list $BX \\
        --other-experiments $OXL --measurements <dir> --work <dir> --out <add-on tables>

copies every base file byte for byte, and appends to `measurements.tsv`, `pairs.tsv`,
`missing.tsv` and `evaluation_only.tsv` (the base file stays a byte prefix). A build-list
measurement becomes a row only when its experiment is a selected or fallback experiment of the
reviewed list `$OXL`, its directory verifies as in `build`, its `meta.json` says
`in_eic = evalonly`, and it is the pooled base view of its allowed layer: `uniform`, or `native`
for a redacted-alignment build (`$OXL` source `redacted_bam`, build-list pipeline
`build_list.NATIVE_PIPELINE`; both must agree). The row takes `eic_track`, `cell`, `assay` of
its EIC track, `in_eic = evalonly` (`blind` when the EIC side is blind), and no control; it is
listed in `evaluation_only.tsv` with `USE_OTHER_EXPERIMENT`, and it gets exactly one pair:
`other_experiment` against the EIC track's pooled base view (uniform before native, never
testbed), `knobs_differing` = the `$OXL` pair class then its `tech_diff` (a fallback's technical
knobs are read from the two rows), split `val` (or `blind`). Everything else goes to
`missing.tsv`, e.g. a second experiment whose EIC partner is not in the base. `build_info.json`
holds `addon_of`, `addon_of_tables_sha256`, `other_experiments(_sha256)` and the base's own
`build_info.json` under `base_build_info`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import (MEASUREMENT_COLUMNS, PAIR_COLUMNS, WITHHELD_COLUMNS,  # noqa: E402
                               Withheld, read_tsv, view_of)
from onion_data import build_list as BL  # noqa: E402

ROLE_ORDER = {"train": 0, "val": 1, "blind": 2}
HALVES = ("pr1", "pr2")
QC_RUN_ARMS = ("base", "se", "crop36", "crop50")
QC_FROM_JSON = {"NSC": "nsc", "RSC": "rsc", "NRF": "nrf", "PBC1": "pbc1", "PBC2": "pbc2",
                "estimated_fragment_len": "frag_len"}
VOCAB_FIELDS = ("pipeline", "pipeline_version", "lab", "platform", "antibody", "donor")
MISSING_COLUMNS = ("row", "priority", "experiment", "measurement_id", "products_missing", "reason")
EVAL_ONLY_COLUMNS = ("measurement_id", "use")
MEASUREMENTS, PAIRS, SHARED = "measurements.tsv", "pairs.tsv", "pairs_shared_reads.tsv"
WITHHELD, COVARIATES, MISSING = "withheld.tsv", "covariates.json", "missing.tsv"
EVAL_ONLY, BUILD_INFO = "evaluation_only.tsv", "build_info.json"
LAYER_PREF = {"uniform": 0, "native": 1}


# --------------------------------------------------------------------------------------------
# small io


def tsv_text(columns, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in columns})
    return buf.getvalue()


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def body_lines(path) -> list:
    """The data lines of a TSV (header dropped), exactly as on disk."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    return lines[1:]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_sha() -> str:
    stamp = TOOLS.parent / ".kit_stamp"
    if stamp.is_file():
        return stamp.read_text().strip().splitlines()[0]
    import subprocess
    out = subprocess.run(["git", "-C", str(TOOLS.parent), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def arm_of(mid: str) -> str:
    return mid.split(".", 3)[3] if mid.count(".") >= 3 else ""


def unit_of(mid: str) -> str:
    return mid.split(".", 3)[1] if mid.count(".") >= 3 else ""


def mdir_of(row: dict) -> Path | None:
    p = row.get("counts_path", "")
    return Path(p[3:]).parent if p.startswith("h5:") else None


# --------------------------------------------------------------------------------------------
# ledger


def ledger_index(ledger_rows: list) -> dict:
    """{"role": experiment -> EIC role, "replication": experiment -> replication_type}.

    EIC experiments: ledger rows with `match = 1` (not controls). Controls: every ENCODE possible
    control such an experiment names (role by `build_list.control_role`), and every ledger control
    row with `match = 1`."""
    S = BL.ledger_sets(ledger_rows)
    role = {x: r["role"] for x, r in S["eic"].items()}
    rep = {x: r.get("replication_type", "") for x, r in S["eic"].items()}
    for c, roles in S["controls"].items():
        role.setdefault(c, BL.control_role(roles))
    for r in ledger_rows:
        if r.get("assay_class") == BL.CONTROL_CLASS and r.get("match") == "1" and r["experiment"]:
            role.setdefault(r["experiment"], r["role"])
            rep.setdefault(r["experiment"], r.get("replication_type", ""))
    return {"role": role, "replication": rep, "eic": S["eic"], "controls": S["controls"]}


# --------------------------------------------------------------------------------------------
# read-level split of the halves


class SplitSignature:
    """Which read-level split made a measurement's halves, from the `READSET.json` beside each
    half tagAlign (`align_views.build_readsets`). A thinned read set inherits its parent's split.
    Two measurements whose signatures are equal and known were cut from one split of the parent's
    reads, so the pr1 of one shares no read with the pr2 of the other."""

    def __init__(self):
        self._rs: dict = {}
        self._m: dict = {}

    def _readset(self, d: Path):
        if d not in self._rs:
            f = d / "READSET.json"
            try:
                self._rs[d] = json.loads(f.read_text()) if f.is_file() else None
            except (OSError, ValueError):
                self._rs[d] = None
        return self._rs[d]

    def half(self, ta: str, h: str, depth: int = 0) -> tuple:
        d = Path(ta).parent
        rec = self._readset(d)
        if rec is None or depth > 8:
            return ("unknown", str(ta))
        tf = rec.get("thinned_from") or {}
        if tf.get(h):
            return self.half(tf[h], h, depth + 1)
        sp = rec.get("split") or {}
        m = str(sp.get("method", ""))
        if m.startswith("blake2b"):
            return ("name", str(sp.get("salt", "")), bool(sp.get("name_tokens")))
        if m:
            return ("split", str(d))
        return ("unknown", str(ta))

    def of_dir(self, mdir: Path) -> frozenset:
        mdir = Path(mdir)
        if mdir not in self._m:
            try:
                prov = json.loads((mdir / "provenance.json").read_text())
                sig = {self.half(ta, h) for h in HALVES for ta in prov["halves"][h]["tagalign"]}
            except (OSError, ValueError, KeyError, TypeError):
                sig = {("unknown", str(mdir))}
            self._m[mdir] = frozenset(sig)
        return self._m[mdir]

    def of_row(self, row: dict) -> frozenset:
        d = mdir_of(row)
        return self.of_dir(d) if d is not None else frozenset({("unknown", row["measurement_id"])})

    @staticmethod
    def known(sig: frozenset) -> bool:
        return bool(sig) and all(s[0] != "unknown" for s in sig)

    def one_split(self, a: dict, b: dict) -> bool:
        sa, sb = self.of_row(a), self.of_row(b)
        return sa == sb and self.known(sa)


# --------------------------------------------------------------------------------------------
# QC from the run directory


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    f = float(v)
    return str(int(f)) if f.is_integer() and abs(f) < 1e12 else repr(f)


def qc_values(work, experiment: str, unit: str, arm: str, prov: dict) -> dict:
    """QC columns of one measurement from `<work>/<experiment>/<arm>/` (see the module doc)."""
    if arm not in QC_RUN_ARMS:
        return {}
    m = re.fullmatch(r"br(\d+)", unit)
    if m:
        rep = int(m.group(1))
    elif unit == "pooled" and not prov.get("pooled_from"):
        rep = 1
    else:
        return {}
    run = Path(work) / experiment / arm
    out = {}
    qj = run / "qc.json"
    if qj.is_file():
        try:
            q = json.loads(qj.read_text())
        except ValueError:
            q = {}
        lib = ((q.get("lib_complexity") or {}).get("lib_complexity") or {}).get(f"rep{rep}") or {}
        xc = ((q.get("align_enrich") or {}).get("xcor_score") or {}).get(f"rep{rep}") or {}
        for src in (lib, xc):
            for k, col in QC_FROM_JSON.items():
                if src.get(k) not in (None, ""):
                    out[col] = _fmt(src[k])
    xf = run / f"rep{rep}.xcor.qc"
    if xf.is_file() and not {"nsc", "rsc", "frag_len"} <= set(out):
        f = xf.read_text().split("\n")[0].split("\t")
        if len(f) >= 10:
            out.setdefault("frag_len", _fmt(f[2].split(",")[0]))
            out.setdefault("nsc", _fmt(f[8]))
            out.setdefault("rsc", _fmt(f[9]))
    return out


# --------------------------------------------------------------------------------------------
# verification (align_views-style), one experiment per task


def verify_experiment(args) -> dict:
    """(measurements root, experiment, [measurement ids], recount chrom) -> {mid: [problems]}."""
    from onion_data import align_views as AV
    root, exp, mids, recount = args
    root = Path(root)
    all_meta, probs = {}, {}
    for m in mids:
        try:
            all_meta[m] = json.loads((root / m / "meta.json").read_text())
        except (OSError, ValueError) as e:
            probs[m] = [f"meta.json unreadable: {e}"]
    for m in mids:
        if m in probs:
            continue
        try:
            probs[m] = AV.verify_measurement(root / m, all_meta, recount)
        except Exception as e:  # noqa: BLE001 - a broken dir is a failed dir, never a crash
            probs[m] = [f"verify raised {type(e).__name__}: {e}"]
    ok = [m for m in mids if not probs[m]]
    if len(ok) > 1:
        try:
            xp = AV.experiment_checks(root, ok, all_meta, recount)
        except Exception as e:  # noqa: BLE001
            xp = [f"experiment check raised {type(e).__name__}: {e}"]
        for x in xp:
            head = x.split(" ", 1)[0]
            for m in ([head] if head in probs else ok):
                probs[m].append(f"experiment check: {x}")
    return probs


# --------------------------------------------------------------------------------------------
# pairs


def pair_id(rung: str, a: str, b: str) -> str:
    return "p" + hashlib.sha1(f"{rung}|{a}|{b}".encode()).hexdigest()[:12]


def pair_split(a: dict, b: dict) -> str:
    return max(a["in_eic"], b["in_eic"], key=lambda r: ROLE_ORDER.get(r, 9))


def cells_of(row: dict) -> set:
    return {c for c in row.get("cell", "").split(";") if c}


def make_pairs(rows: list, replication: dict, withheld: Withheld, sig: SplitSignature) -> tuple:
    """(pairs, shared-read pairs, stats) from the kept rows (see the module doc)."""
    pairs, shared, stats = [], [], Counter()

    def add(lst, a, b, rung, knobs="", subset=0, nt=0):
        same = a["cell"] == b["cell"]
        lst.append({"pair_id": pair_id(rung, a["measurement_id"], b["measurement_id"]),
                    "a": a["measurement_id"], "b": b["measurement_id"], "rung": rung,
                    "assay": a["assay"], "cell": a["cell"] if same else "",
                    "knobs_differing": knobs, "subset_pair": str(int(subset)),
                    "split": pair_split(a, b), "never_trained": str(int(nt))})

    base = [r for r in rows if view_of(r)[0] == "base"]
    for r in base:
        if r.get("has_pseudoreps") == "1":
            add(pairs, r, r, "pseudoreplicate")

    units = defaultdict(dict)                       # (experiment, layer) -> unit -> base row
    for r in base:
        units[(r["experiment"], r["layer"])][r["unit"]] = r
    for (exp, _layer), us in sorted(units.items()):
        brs = sorted(u for u in us if re.fullmatch(r"br\d+", u))
        rung = "anisogenic" if "anisogenic" in replication.get(exp, "") else "isogenic"
        for u, v in combinations(brs, 2):
            add(pairs, us[u], us[v], rung)
        libs = defaultdict(list)
        lanes = defaultdict(list)
        for u in us:
            m = re.fullmatch(r"(br\d+)(tr\d+)(ln\d+)?", u)
            if m and m.group(3):
                lanes[m.group(1) + m.group(2)].append(u)
            elif m:
                libs[m.group(1)].append(u)
        for group, rung_name in ((libs, "technical"), (lanes, "sequencing")):
            for _, members in sorted(group.items()):
                for u, v in combinations(sorted(members), 2):
                    add(pairs, us[u], us[v], rung_name)

    best = {}                                       # (cell, assay) -> pooled base view
    for r in base:
        if r["unit"] != "pooled" or r["layer"] not in LAYER_PREF:
            continue
        k = (r["cell"], r["assay"])
        if k not in best or (LAYER_PREF[r["layer"]], r["measurement_id"]) < \
                (LAYER_PREF[best[k]["layer"]], best[k]["measurement_id"]):
            best[k] = r
    by_assay = defaultdict(list)
    for (_, assay), r in sorted(best.items()):
        by_assay[assay].append(r)
    for assay in sorted(by_assay):
        for a, b in combinations(by_assay[assay], 2):
            if cells_of(a) & cells_of(b):
                continue
            add(pairs, a, b, "other_cell_type", "cell")

    tracks = defaultdict(list)                      # (experiment, pooled) -> every arm's row
    for r in rows:
        if r["unit"] == "pooled":
            tracks[r["experiment"]].append(r)
    for exp in sorted(tracks):
        views = sorted(tracks[exp], key=lambda r: r["measurement_id"])
        for a in views:
            for b in views:
                if a is b:
                    continue
                ka, kb = view_of(a)[0], view_of(b)[0]
                knobs = "+".join(sorted({ka, kb} - {"base"}))
                subset = BL.is_subset_pair(arm_of(a["measurement_id"]), arm_of(b["measurement_id"]))
                blocked = withheld.blocks(a, b)
                split = pair_split(a, b)
                if blocked and split == "train":
                    stats["withheld combinations on training tracks (not written)"] += 1
                    continue
                nt = int(blocked and split in ("val", "blind"))
                if a["layer"] == "testbed" and b["layer"] == "testbed":
                    stats["designed pairs between two testbed arms (to pairs_shared_reads.tsv)"] += 1
                    add(shared, a, b, "designed", knobs, subset, nt)
                elif not sig.one_split(a, b):
                    stats["designed pairs whose halves come from different splits "
                          "(to pairs_shared_reads.tsv)"] += 1
                    add(shared, a, b, "designed", knobs, subset, nt)
                else:
                    add(pairs, a, b, "designed", knobs, subset, nt)
    for lst in (pairs, shared):
        lst.sort(key=lambda p: (BL_RUNG_INDEX.get(p["rung"], 99), p["a"], p["b"]))
    return pairs, shared, stats


BL_RUNG_INDEX = {r: i for i, r in enumerate(("pseudoreplicate", "sequencing", "technical",
                                              "isogenic", "anisogenic", "other_experiment",
                                              "other_cell_type", "designed"))}


# --------------------------------------------------------------------------------------------
# covariates


def covariates_spec(rows, older: dict | None = None) -> dict:
    vocab = {}
    for f in VOCAB_FIELDS:
        old = list((older or {}).get("vocab", {}).get(f, []))
        new = sorted({r[f] for r in rows if r.get(f)} - set(old))
        vocab[f] = old + new
    return {"schema": 1, "meta_columns": list(C.META_COLUMNS), "qc_columns": list(C.QC_COLUMNS),
            "vocab": vocab}


# --------------------------------------------------------------------------------------------
# build


def check_out(out: Path, withheld_bytes: bytes) -> str | None:
    """None when `out` may be written, else the reason."""
    if not out.exists():
        return None
    if not out.is_dir():
        return f"{out} exists and is not a directory"
    extra = sorted(p.name for p in out.iterdir() if p.name != WITHHELD)
    if extra:
        return f"{out} is not empty ({', '.join(extra[:5])}): frozen tables are never overwritten"
    w = out / WITHHELD
    if w.is_file() and w.read_bytes() != withheld_bytes:
        return f"{w} differs from --withheld"
    return None


def collect(a) -> dict:
    """Everything `build` would write, in memory, with the counts it prints."""
    t0 = time.time()
    M, W = Path(a.measurements), Path(a.work)
    ledger_rows = BL.read_tsv(a.ledger)
    L = ledger_index(ledger_rows)
    prios = {p.strip() for p in str(a.priorities).split(",") if p.strip()}
    bl = [r for r in BL.read_tsv(a.build_list) if r["priority"] in prios]
    wh_bytes = Path(a.withheld).read_bytes()
    withheld = Withheld(read_tsv(a.withheld, WITHHELD_COLUMNS))

    expected = {}                                   # mid -> (build-list row, products, state)
    for r in bl:
        state, _ = BL.row_state(r, M, W)
        for mid, prods in BL.parse_tokens(r.get("expected_measurements", "")):
            expected.setdefault(mid, (r, prods, state))

    missing, metas, by_exp = [], {}, defaultdict(list)
    n_read = 0

    def miss(r, mid, reason, products=""):
        missing.append({"row": r.get("row", ""), "priority": r.get("priority", ""),
                        "experiment": mid.split(".", 1)[0], "measurement_id": mid,
                        "products_missing": products, "reason": reason})

    for mid in sorted(expected):
        r, prods, state = expected[mid]
        d = M / mid
        if not (d / "provenance.json").is_file() or not (d / "meta.json").is_file():
            miss(r, mid, "not applicable (NOT_APPLICABLE in the run directory)"
                 if state == "not_applicable" else "not built (no provenance.json)")
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except ValueError as e:
            miss(r, mid, f"meta.json unreadable: {e}")
            continue
        n_read += 1
        exp = meta.get("experiment", "")
        role = L["role"].get(exp)
        if meta.get("measurement_id") != mid:
            miss(r, mid, f"meta measurement_id {meta.get('measurement_id')!r} != directory")
        elif role is None:
            miss(r, mid, "experiment is not a ledger experiment with match = 1 nor its control")
        elif meta.get("in_eic") != role:
            miss(r, mid, f"in_eic {meta.get('in_eic')!r} != ledger role {role!r}")
        elif meta.get("layer") == "testbed":
            miss(r, mid, "layer testbed in the build list (testbed rows enter via --testbed-rows)")
        else:
            metas[mid] = meta
            by_exp[exp].append(mid)

    tasks = [(str(M), exp, sorted(mids), a.recount_chrom) for exp, mids in sorted(by_exp.items())]
    probs = {}
    if a.workers > 1 and len(tasks) > 1:
        with Pool(a.workers) as pool:
            for res in pool.imap_unordered(verify_experiment, tasks):
                probs.update(res)
    else:
        for t in tasks:
            probs.update(verify_experiment(t))

    rows, n_qc = [], Counter()
    for mid in sorted(metas):
        r, prods, _ = expected[mid]
        if probs.get(mid):
            miss(r, mid, "verify failed: " + "; ".join(probs[mid][:3]))
            continue
        row = {k: str(metas[mid].get(k, "")) for k in MEASUREMENT_COLUMNS}
        prov = json.loads((M / mid / "provenance.json").read_text())
        for col, v in qc_values(W, row["experiment"], row["unit"], arm_of(mid), prov).items():
            if row[col] == "":
                row[col] = v
                n_qc[col] += 1
        row["frip"] = ""
        lack = [p for p in prods if p not in ("counts",) and not (M / mid / f"{p}.h5").is_file()]
        if lack:
            miss(r, mid, "kept; expected products missing", "+".join(lack))
        rows.append(row)

    # testbed rows: evaluation only, in_eic = blind kept
    from onion_data import import_testbed as IT
    tb_rows, eval_only, n_tb_read = [], [], 0
    if a.testbed_rows:
        uses = {r["measurement_id"]: r.get("use", "") for r in IT.read_tsv(a.testbed_rows)}
        for row in IT.rows_for_tables(a.testbed_rows, allow_blind=True):
            mid = row["measurement_id"]
            d = mdir_of(row)
            tag = {"row": "", "priority": "testbed"}
            if d is None or not (d / "meta.json").is_file() or not (d / "provenance.json").is_file():
                miss(tag, mid, "testbed row: measurement directory missing")
                continue
            n_tb_read += 1
            meta = json.loads((d / "meta.json").read_text())
            role = L["role"].get(row["experiment"])
            if {k: str(meta.get(k, "")) for k in MEASUREMENT_COLUMNS} != row:
                miss(tag, mid, "testbed row != its meta.json")
            elif role is None or row["in_eic"] != role:
                miss(tag, mid, f"testbed row: experiment role {role!r} != in_eic {row['in_eic']!r}")
            elif row["layer"] != "testbed":
                miss(tag, mid, f"testbed row with layer {row['layer']!r}")
            else:
                tb_rows.append(row)
                eval_only.append({"measurement_id": mid, "use": uses.get(mid) or
                                  "testbed: evaluation use only, never training"})
    dup = {r["measurement_id"] for r in rows} & {r["measurement_id"] for r in tb_rows}
    if dup:
        raise SystemExit(f"measurement ids both in the build and the testbed rows: {sorted(dup)[:5]}")
    rows = sorted(rows + tb_rows, key=lambda r: r["measurement_id"])

    sig = SplitSignature()
    pairs, shared, stats = make_pairs(rows, L["replication"], withheld, sig)

    older = None
    if a.superset_of:
        older = json.loads((Path(a.superset_of) / COVARIATES).read_text())
    cov = covariates_spec(rows, older)
    texts = {MEASUREMENTS: tsv_text(MEASUREMENT_COLUMNS, rows),
             PAIRS: tsv_text(PAIR_COLUMNS, pairs), SHARED: tsv_text(PAIR_COLUMNS, shared),
             MISSING: tsv_text(MISSING_COLUMNS, sorted(missing, key=lambda m: m["measurement_id"])),
             EVAL_ONLY: tsv_text(EVAL_ONLY_COLUMNS, sorted(eval_only,
                                                           key=lambda m: m["measurement_id"])),
             COVARIATES: json.dumps(cov, indent=1) + "\n"}
    superset_bad = []
    if a.superset_of:
        superset_bad = superset_problems(Path(a.superset_of), texts, wh_bytes)
    n_dirs_other = 0
    if M.is_dir():
        want = set(expected)
        n_dirs_other = sum(1 for p in M.iterdir() if p.is_dir() and not p.name.startswith(".")
                           and p.name not in want and p.name.count(".") >= 3
                           and p.name.split(".")[2] != "testbed")
    return {"rows": rows, "pairs": pairs, "shared": shared, "missing": missing, "stats": stats,
            "texts": texts, "wh_bytes": wh_bytes, "n_read": n_read, "n_tb_read": n_tb_read,
            "n_expected": len(expected), "n_qc": n_qc, "superset_bad": superset_bad,
            "n_eval_only": len(eval_only), "n_dirs_other": n_dirs_other,
            "n_verify_failed": sum(1 for m in metas if probs.get(m)), "wall_s": time.time() - t0,
            "ledger_sha256": sha256_bytes(Path(a.ledger).read_bytes())}


def superset_problems(old: Path, texts: dict, wh_bytes: bytes) -> list:
    out = []
    for name in (MEASUREMENTS, PAIRS):
        new = set(texts[name].splitlines(keepends=True)[1:])
        gone = [ln for ln in body_lines(old / name) if ln not in new]
        if gone:
            out.append(f"{name}: {len(gone)} rows of the older tables changed or missing, e.g. "
                       f"{gone[0].split(chr(9))[0]}")
    if (old / WITHHELD).read_bytes() != wh_bytes:
        out.append("withheld.tsv differs from the older tables")
    o = json.loads((old / COVARIATES).read_text()).get("vocab", {})
    n = json.loads(texts[COVARIATES])["vocab"]
    for f, vals in o.items():
        if n.get(f, [])[:len(vals)] != list(vals):
            out.append(f"covariates.json vocab {f}: the older ids do not stay")
    return out


def print_summary(res: dict, a) -> None:
    rows = res["rows"]
    print(f"measurement directories read: {res['n_read'] + res['n_tb_read']} "
          f"({res['n_read']} from {res['n_expected']} build-list measurements of priorities "
          f"{a.priorities}, {res['n_tb_read']} testbed)")
    print(f"rows that would be written: {len(rows)}")
    sp = Counter(r["in_eic"] for r in rows)
    print("rows per EIC split: " + ", ".join(f"{s} {sp.get(s, 0)}" for s in ROLE_ORDER))
    print("rows per layer: " + ", ".join(f"{k} {v}" for k, v in
                                         sorted(Counter(r["layer"] for r in rows).items())))
    print(f"evaluation-only rows (testbed, in_eic = blind): {res['n_eval_only']}")
    print(f"verify failed: {res['n_verify_failed']}")
    reasons = Counter(m["reason"].split(":")[0] for m in res["missing"])
    print(f"missing.tsv rows: {len(res['missing'])}"
          + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ")"
             if reasons else ""))
    pc = Counter((p["rung"], p["split"]) for p in res["pairs"])
    for (rung, split), n in sorted(pc.items(), key=lambda kv: (BL_RUNG_INDEX[kv[0][0]], kv[0][1])):
        print(f"pairs {rung} {split}: {n}")
    nt = Counter(p["split"] for p in res["pairs"] if p["never_trained"] == "1")
    print("never-trained designed pairs: " + ", ".join(f"{s} {nt.get(s, 0)}" for s in ("val", "blind"))
          + f"; of them subset_pair = 0: "
          + str(sum(1 for p in res["pairs"] if p["never_trained"] == "1" and p["subset_pair"] == "0")))
    print(f"pairs_shared_reads.tsv rows: {len(res['shared'])}")
    for k, v in sorted(res["stats"].items()):
        print(f"  {k}: {v}")
    print("QC cells filled from the run directories: "
          + (", ".join(f"{k} {v}" for k, v in sorted(res["n_qc"].items())) or "none"))
    print(f"directories under --measurements not in the selected priorities (testbed excluded): "
          f"{res['n_dirs_other']}")
    if a.superset_of:
        print(f"rows of the older tables changed or missing: "
              f"{sum(1 for x in res['superset_bad'] if 'rows of the older' in x)} problem(s)")
        for x in res["superset_bad"]:
            print(f"  FAIL {x}")
    print(f"wall {res['wall_s']:.0f} s")


def cmd_build(a) -> int:
    out = Path(a.out)
    if not a.dry_run:
        why = check_out(out, Path(a.withheld).read_bytes())
        if why:
            print(f"refused: {why}")
            return 2
    res = collect(a)
    print_summary(res, a)
    if res["superset_bad"]:
        print("refused: the new tables are not a superset of the older tables; nothing written")
        return 1
    if a.dry_run:
        print("dry run: no table written")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name, text in res["texts"].items():
        write_text(out / name, text)
    if not (out / WITHHELD).is_file():
        tmp = out / (WITHHELD + ".part")
        tmp.write_bytes(res["wh_bytes"])
        os.replace(tmp, out / WITHHELD)
    info = {"schema": 1, "tool": "tools/onion_data/build_tables.py", "git_sha": git_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_list": str(Path(a.build_list).resolve()), "priorities": a.priorities,
            "measurements": str(Path(a.measurements).resolve()),
            "work": str(Path(a.work).resolve()),
            "testbed_rows": str(Path(a.testbed_rows).resolve()) if a.testbed_rows else "",
            "withheld": str(Path(a.withheld).resolve()), "ledger": str(Path(a.ledger).resolve()),
            "ledger_sha256": res["ledger_sha256"], "superset_of": a.superset_of or "",
            "recount_chrom": a.recount_chrom,
            "files_sha256": {n: sha256_bytes((out / n).read_bytes())
                             for n in sorted(res["texts"]) + [WITHHELD]},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
    write_text(out / BUILD_INFO, json.dumps(info, indent=1) + "\n")
    print(f"wrote {out}: {len(res['rows'])} measurements, {len(res['pairs'])} pairs, "
          f"{len(res['shared'])} shared-read pairs, {len(res['missing'])} missing")
    return 0


# --------------------------------------------------------------------------------------------
# add-on of the evaluation-only second experiments (Amendment 4, chunk OXt)

#: the `use` text of a second experiment in `evaluation_only.tsv` (Amendment 4, decision 2)
USE_OTHER_EXPERIMENT = "other_experiment: evaluation use only, never training"
#: base files to which the add-on appends rows; every other base file is copied byte for byte
ADDON_APPENDED = (MEASUREMENTS, PAIRS, MISSING, EVAL_ONLY)
ADDON_COLUMNS = {MEASUREMENTS: MEASUREMENT_COLUMNS, PAIRS: PAIR_COLUMNS,
                 MISSING: MISSING_COLUMNS, EVAL_ONLY: EVAL_ONLY_COLUMNS}
#: `$OXL` source value of a second experiment built from ENCODE's redacted alignment
REDACTED_SOURCE = "redacted_bam"
TECH_KNOBS = ("run_type", "read_length", "platform")


def reviewed_list(rows: list) -> dict:
    """experiment -> {"track", "split", "pair_class", "tech_diff", "source", "choice"} for every
    selected (`experiment`) and fallback (`fallback`) experiment of the reviewed list `$OXL`.
    A fallback's `tech_diff` is not in `$OXL` (None: computed from the two table rows)."""
    out = {}
    for r in rows:
        if r.get("experiment"):
            out[r["experiment"]] = {"track": r["eic_track"], "split": r["split"],
                                    "pair_class": r["pair_class"],
                                    "tech_diff": r.get("tech_diff", ""),
                                    "source": r.get("source") or "fastq", "choice": "selected"}
        if r.get("fallback"):
            out.setdefault(r["fallback"], {"track": r["eic_track"], "split": r["split"],
                                           "pair_class": r.get("fallback_pair_class", ""),
                                           "tech_diff": None,
                                           "source": r.get("fallback_source") or "fastq",
                                           "choice": "fallback"})
    return out


def second_view_layer(source: str) -> str:
    """The layer of a second experiment's one allowed view: `native` for a redacted-alignment
    build (`build_list.NATIVE_PIPELINE`, whose rows write `<x>.pooled.native.base`), else
    `uniform`."""
    return "native" if source == REDACTED_SOURCE else "uniform"


def is_second_view(row: dict, layer: str) -> bool:
    """The pooled base view of the second experiment in its allowed layer, and nothing else."""
    return (row["measurement_id"] == f"{row['experiment']}.pooled.{layer}.base"
            and row["unit"] == "pooled" and row["layer"] == layer
            and view_of(row)[0] == "base")


def eic_partner(rows: list, track: str) -> dict | None:
    """The EIC track's pooled base view in the base tables: uniform before native, never
    testbed and never a second experiment (the rule of `other_cell_type`)."""
    c = [r for r in rows if r["eic_track"] == track and r["unit"] == "pooled"
         and r["layer"] in LAYER_PREF and view_of(r)[0] == "base"
         and r["in_eic"] in ROLE_ORDER]
    return min(c, key=lambda r: (LAYER_PREF[r["layer"]], r["measurement_id"])) if c else None


def other_experiment_split(eic_row: dict) -> str:
    """`split` of an `other_experiment` pair: `blind` for a blind EIC side, else `val`."""
    return "blind" if eic_row["in_eic"] == "blind" else "val"


def knobs_of(entry: dict, a: dict, b: dict) -> str:
    tech = entry["tech_diff"]
    if tech is None:
        tech = ";".join(k for k in TECH_KNOBS if a.get(k, "") != b.get(k, ""))
    return ";".join(x for x in (entry["pair_class"], tech) if x)


def collect_addon(a) -> dict:
    t0 = time.time()
    base, M, W = Path(a.base), Path(a.measurements), Path(a.work)
    oxl_bytes = Path(a.other_experiments).read_bytes()
    oxl = reviewed_list(_tsv_rows_of(oxl_bytes))
    base_rows = read_tsv(base / MEASUREMENTS, MEASUREMENT_COLUMNS)
    base_ids = {r["measurement_id"] for r in base_rows}
    by_track = defaultdict(list)
    for r in base_rows:
        by_track[r["eic_track"]].append(r)

    expected = {}                                   # mid -> (build-list row, products)
    for r in BL.read_tsv(a.other_build_list):
        for mid, prods in BL.parse_tokens(r.get("expected_measurements", "")):
            expected.setdefault(mid, (r, prods))
    missing, cand = [], {}

    def miss(r, mid, reason, products=""):
        missing.append({"row": r.get("row", ""), "priority": r.get("priority", ""),
                        "experiment": mid.split(".", 1)[0], "measurement_id": mid,
                        "products_missing": products, "reason": reason})

    for mid in sorted(expected):
        r, prods = expected[mid]
        d = M / mid
        exp = mid.split(".", 1)[0]
        entry = oxl.get(exp)
        if entry is None:
            miss(r, mid, "second experiment: not in the reviewed list (other_experiments.tsv)")
            continue
        layer = second_view_layer(entry["source"])
        native_row = r.get("pipeline", "") == BL.NATIVE_PIPELINE
        if native_row != (layer == "native"):
            miss(r, mid, f"second experiment: build-list pipeline {r.get('pipeline')!r} "
                         f"disagrees with the reviewed source {entry['source']!r}")
            continue
        if mid in base_ids:
            miss(r, mid, "second experiment: measurement id already in the base tables")
            continue
        if not (d / "provenance.json").is_file() or not (d / "meta.json").is_file():
            miss(r, mid, "second experiment: not built (no provenance.json)")
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except ValueError as e:
            miss(r, mid, f"second experiment: meta.json unreadable: {e}")
            continue
        row = {k: str(meta.get(k, "")) for k in MEASUREMENT_COLUMNS}
        if meta.get("measurement_id") != mid or row["experiment"] != exp:
            miss(r, mid, "second experiment: meta measurement_id or experiment != directory")
        elif meta.get("in_eic") != "evalonly":
            miss(r, mid, f"second experiment: meta in_eic {meta.get('in_eic')!r} != 'evalonly'")
        elif not is_second_view(row, layer):
            miss(r, mid, f"second experiment: not the pooled {layer} base view")
        else:
            cand[mid] = (r, prods, row, entry)

    tasks = [(str(M), mid.split(".", 1)[0], [mid], a.recount_chrom) for mid in sorted(cand)]
    probs = {}
    for t in tasks:
        probs.update(verify_experiment(t))

    rows, pairs, eval_only = [], [], []
    for mid in sorted(cand):
        r, prods, row, entry = cand[mid]
        if probs.get(mid):
            miss(r, mid, "second experiment: verify failed: " + "; ".join(probs[mid][:3]))
            continue
        partner = eic_partner(by_track.get(entry["track"], []), entry["track"])
        if partner is None:
            miss(r, mid, f"second experiment: EIC partner of track {entry['track']} absent "
                         f"from the base tables")
            continue
        prov = json.loads((M / mid / "provenance.json").read_text())
        for col, v in qc_values(W, row["experiment"], row["unit"], arm_of(mid), prov).items():
            if row[col] == "":
                row[col] = v
        row.update({"in_eic": "blind" if partner["in_eic"] == "blind" else "evalonly",
                    "eic_track": partner["eic_track"], "cell": partner["cell"],
                    "assay": partner["assay"], "assay_class": partner["assay_class"],
                    "control_measurement_id": "", "frip": ""})
        lack = [p for p in prods if p != "counts" and not (M / mid / f"{p}.h5").is_file()]
        if lack:
            miss(r, mid, "kept; expected products missing", "+".join(lack))
        rows.append(row)
        eval_only.append({"measurement_id": mid, "use": USE_OTHER_EXPERIMENT})
        pairs.append({"pair_id": pair_id("other_experiment", partner["measurement_id"], mid),
                      "a": partner["measurement_id"], "b": mid, "rung": "other_experiment",
                      "assay": partner["assay"], "cell": partner["cell"],
                      "knobs_differing": knobs_of(entry, partner, row), "subset_pair": "0",
                      "split": other_experiment_split(partner), "never_trained": "0"})
    pairs.sort(key=lambda p: (p["a"], p["b"]))
    missing.sort(key=lambda m: m["measurement_id"])
    return {"rows": rows, "pairs": pairs, "missing": missing, "eval_only": eval_only,
            "oxl": oxl, "oxl_sha256": sha256_bytes(oxl_bytes), "n_expected": len(expected),
            "wall_s": time.time() - t0}


def _tsv_rows_of(b: bytes) -> list:
    return [dict(r) for r in csv.DictReader(io.StringIO(b.decode("utf-8")), delimiter="\t")]


def cmd_addon(a) -> int:
    from minicandi.dataset import tables_sha256
    base, out = Path(a.base), Path(a.out)
    if not (base / MEASUREMENTS).is_file():
        print(f"refused: {base} holds no measurements.tsv")
        return 2
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print(f"refused: {out} exists and is not empty: frozen tables are never overwritten")
        return 2
    res = collect_addon(a)
    new = {MEASUREMENTS: res["rows"], PAIRS: res["pairs"], MISSING: res["missing"],
           EVAL_ONLY: res["eval_only"]}
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(base.iterdir()):
        if p.is_file() and p.name != BUILD_INFO and p.name not in ADDON_APPENDED:
            tmp = out / (p.name + ".part")
            tmp.write_bytes(p.read_bytes())
            os.replace(tmp, out / p.name)
    for name in ADDON_APPENDED:
        old = (base / name).read_text(encoding="utf-8") if (base / name).is_file() else \
            tsv_text(ADDON_COLUMNS[name], [])
        if old and not old.endswith("\n"):
            old += "\n"
        body = tsv_text(ADDON_COLUMNS[name], new[name]).split("\n", 1)[1]
        write_text(out / name, old + body)
    base_info = (json.loads((base / BUILD_INFO).read_text())
                 if (base / BUILD_INFO).is_file() else {})
    info = {"schema": 1, "tool": "tools/onion_data/build_tables.py addon", "git_sha": git_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "addon_of": str(base.resolve()), "addon_of_tables_sha256": tables_sha256(base),
            "other_experiments": str(Path(a.other_experiments).resolve()),
            "other_experiments_sha256": res["oxl_sha256"],
            "other_build_list": str(Path(a.other_build_list).resolve()),
            "measurements": str(Path(a.measurements).resolve()),
            "work": str(Path(a.work).resolve()), "recount_chrom": a.recount_chrom,
            "ledger": base_info.get("ledger", ""), "base_build_info": base_info,
            "files_sha256": {p.name: sha256_bytes(p.read_bytes())
                             for p in sorted(out.iterdir()) if p.is_file()},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
    write_text(out / BUILD_INFO, json.dumps(info, indent=1) + "\n")
    reasons = Counter(m["reason"].split(":")[1].strip() if m["reason"].startswith("second")
                      else m["reason"] for m in res["missing"])
    print(f"second-experiment measurements in the build list: {res['n_expected']}")
    print(f"rows appended: {len(res['rows'])} ("
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['in_eic'] for r in res['rows'])
                                                    .items())) + ")")
    print(f"other_experiment pairs appended: {len(res['pairs'])} ("
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(p['split'] for p in res['pairs'])
                                                    .items())) + ")")
    print(f"missing.tsv rows appended: {len(res['missing'])}"
          + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ")"
             if reasons else ""))
    print(f"wrote {out} (add-on of {base}); wall {res['wall_s']:.0f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a tables directory (or --dry-run)")
    b.add_argument("--build-list", required=True)
    b.add_argument("--priorities", required=True, help="comma list, e.g. 1,2,3")
    b.add_argument("--measurements", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--testbed-rows", default="")
    b.add_argument("--withheld", required=True)
    b.add_argument("--ledger", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--superset-of", default="")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    b.add_argument("--recount-chrom", default="none",
                   help="chromosome recounted from the half tagAligns in verify (none = off)")
    b.set_defaults(func=cmd_build)
    x = sub.add_parser("addon", help="add-on tables: a frozen base plus the second experiments")
    x.add_argument("--base", required=True, help="the frozen tables directory")
    x.add_argument("--other-build-list", required=True, help="$BX")
    x.add_argument("--other-experiments", required=True, help="$OXL, the reviewed list")
    x.add_argument("--measurements", required=True)
    x.add_argument("--work", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--recount-chrom", default="none",
                   help="chromosome recounted from the half tagAligns in verify (none = off)")
    x.set_defaults(func=cmd_addon)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
