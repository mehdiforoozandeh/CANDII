"""t118 ladder — constants, the pair tables and the SLURM task table, all pure functions of the manifest.

**Products.** `usable_products` = every manifest row except `EXCLUDED_PIDS` (the two DNase MAPQ
arms, byte-identical to the DNase base: a t112 defect), sorted by pid — 128 on the t112 manifest.

**Pairs.** Each pair is a dict with the keys `PAIR_KEYS`.
  train_pairs(rows, g)  g = a track id: base->arm and arm->base for every usable non-base product
                        of the track, in (arm pid, direction) order, `base_to_arm` first — 38 per
                        histone track, 14 for the DNase track. g = "all": the concatenation over
                        the tracks in sorted order (242).
  law_pairs(rows, g)    every ordered pair (a, b), a != b, of usable non-base products within a
                        track, sorted by (a pid, b pid), `direction = "arm_to_arm"` — 342 per
                        histone track, 42 for DNase, 2094 for "all". Never trained.
  `knob_combo = f"{arm_src}:{level_src}->{arm_tgt}:{level_tgt}"`; `counts_identical` /
  `pval_identical` = the two products' manifest npz md5s in that space are equal.

**Tasks.** 576 = RUNGS x G_IDS x SPACES x MODELS x SEEDS, rung slowest and seed fastest:
`index = rung_i*144 + g_i*18 + space_i*9 + model_i*3 + seed`,
`run_name = f"{rung}_{g}_{space}_{model}_s{seed}"`.

    python tools/t118/ladder/pairs.py count <MANIFEST.tsv>
    python tools/t118/ladder/pairs.py tasks <MANIFEST.tsv>
    python tools/t118/ladder/pairs.py train-pairs <MANIFEST.tsv> <g>
    python tools/t118/ladder/pairs.py law-pairs <MANIFEST.tsv> <g>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:          # script mode: make `baseline_rungs` and `ladder` importable
    sys.path.insert(0, str(_T118))

import baseline_rungs as _br  # noqa: E402  (tools/t118/baseline_rungs.py)

SCORE_CHROMS = ("chr19", "chr21")
VAL_CHROMS = ("chr22",)
DROP_CHROMS = {"chrY", "chrM"}
#: restated identically from `tools/t112/bin25.py::MAIN_CHROMS` (a test asserts equality)
MAIN_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)
MARK_CLASS = dict(_br.MARK_CLASS)
EXCLUDED_PIDS = ("C12M02__mapq__0", "C12M02__mapq__10")
RUNGS = ("A", "B", "C", "D")
MODELS = ("real", "nocov", "ids")
SPACES = ("counts", "pval")
SEEDS = (0, 1, 2)
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")
G_IDS = TRACKS + ("all",)
#: space -> the manifest column holding that space's npz file md5
MD5_COLUMN = {"counts": "counts25_md5", "pval": "pval25_md5"}
DIRECTIONS = ("base_to_arm", "arm_to_base")
PAIR_KEYS = ("track", "cell", "assay", "mark_class", "arm_src", "level_src", "arm_tgt",
             "level_tgt", "source_pid", "target_pid", "direction", "knob_combo",
             "counts_identical", "pval_identical")
TASK_KEYS = ("index", "rung", "g", "space", "model", "seed", "run_name")


def read_manifest(path) -> list[dict]:
    """The t112 MANIFEST.tsv as a list of str dicts (`baseline_rungs.read_manifest`)."""
    return _br.read_manifest(path)


def usable_products(rows: list[dict]) -> list[dict]:
    """Every row not in `EXCLUDED_PIDS`, sorted by pid."""
    pids = [r["pid"] for r in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("duplicate pid in the manifest")
    return sorted((r for r in rows if r["pid"] not in EXCLUDED_PIDS), key=lambda r: r["pid"])


def track_ids(rows: list[dict]) -> list[str]:
    """The sorted distinct tracks of the usable products (== TRACKS on the t112 manifest)."""
    return sorted({r["track"] for r in usable_products(rows)})


def _by_track(rows: list[dict], track: str) -> tuple[dict, list[dict]]:
    """(base row, usable non-base rows sorted by pid) of one track."""
    prods = [r for r in usable_products(rows) if r["track"] == track]
    if not prods:
        raise ValueError(f"no usable products for track {track!r}")
    base = [r for r in prods if r["arm"] == "base"]
    if len(base) != 1:
        raise ValueError(f"track {track}: {len(base)} base rows, need exactly 1")
    base = base[0]
    arms = [r for r in prods if r["arm"] != "base"]
    for r in arms:
        if r["assay"] != base["assay"]:
            raise ValueError(f"{r['pid']}: assay {r['assay']} != base assay {base['assay']}")
    return base, arms


def _pair(src: dict, tgt: dict, direction: str) -> dict:
    return {
        "track": src["track"], "cell": src["cell"], "assay": src["assay"],
        "mark_class": MARK_CLASS[src["assay"]],
        "arm_src": src["arm"], "level_src": src["level"],
        "arm_tgt": tgt["arm"], "level_tgt": tgt["level"],
        "source_pid": src["pid"], "target_pid": tgt["pid"], "direction": direction,
        "knob_combo": f"{src['arm']}:{src['level']}->{tgt['arm']}:{tgt['level']}",
        "counts_identical": src[MD5_COLUMN["counts"]] == tgt[MD5_COLUMN["counts"]],
        "pval_identical": src[MD5_COLUMN["pval"]] == tgt[MD5_COLUMN["pval"]],
    }


def _tracks_of(rows: list[dict], g: str) -> list[str]:
    return track_ids(rows) if g == "all" else [g]


def train_pairs(rows: list[dict], g: str) -> list[dict]:
    """base->arm and arm->base per usable arm product, (pid, direction) order; "all" = concat."""
    out = []
    for track in _tracks_of(rows, g):
        base, arms = _by_track(rows, track)
        for r in arms:
            out.append(_pair(base, r, "base_to_arm"))
            out.append(_pair(r, base, "arm_to_base"))
    return out


def law_pairs(rows: list[dict], g: str) -> list[dict]:
    """Every ordered (a, b), a != b, of usable non-base products within a track; never trained."""
    out = []
    for track in _tracks_of(rows, g):
        _, arms = _by_track(rows, track)
        for a in arms:
            for b in arms:
                if a["pid"] != b["pid"]:
                    out.append(_pair(a, b, "arm_to_arm"))
    return out


def fit_pids(rows: list[dict], g: str) -> list[str]:
    """Sorted set of the source and target pids of `train_pairs(rows, g)`."""
    return sorted({p[k] for p in train_pairs(rows, g) for k in ("source_pid", "target_pid")})


def _pair_index(rows: list[dict], pair: dict) -> int:
    """The pair's position in its own TRACK's list (train_pairs, or law_pairs for arm_to_arm).

    Defined per track, not per g, so a pair gets the same shuffle under its per-track g and under
    the across-track g.
    """
    table = law_pairs if pair["direction"] == "arm_to_arm" else train_pairs
    key = (pair["source_pid"], pair["target_pid"])
    for i, p in enumerate(table(rows, pair["track"])):
        if (p["source_pid"], p["target_pid"]) == key:
            return i
    raise ValueError(f"pair {key} is not a {pair['direction']} pair of track {pair['track']}")


def shuffle_target(rows: list[dict], pair: dict, space: str, seed: int) -> str:
    """A wrong target covariate pid for `pair`, uniform over the track's usable products minus the
    target, the source and every product whose `space` md5 equals the target's.

    rng = `np.random.default_rng([seed, 7119, 0 if space == "counts" else 1, pair_index])`, where
    `pair_index` is the pair's position in its track's pair list (`_pair_index`).
    """
    if space not in SPACES:
        raise ValueError(f"space {space!r} not in {SPACES}")
    col = MD5_COLUMN[space]
    prods = {r["pid"]: r for r in usable_products(rows) if r["track"] == pair["track"]}
    tgt_md5 = prods[pair["target_pid"]][col]
    cand = [pid for pid, r in sorted(prods.items())
            if pid not in (pair["target_pid"], pair["source_pid"]) and r[col] != tgt_md5]
    if not cand:
        raise ValueError(f"no shuffle candidate for {pair['source_pid']}->{pair['target_pid']}")
    rng = np.random.default_rng([int(seed), 7119, 0 if space == "counts" else 1,
                                 _pair_index(rows, pair)])
    return cand[int(rng.integers(len(cand)))]


def ids_permutation(n_pairs: int, seed: int) -> np.ndarray:
    """A derangement of range(n_pairs) from `default_rng([seed, 7118])`, redrawn until no fixed point."""
    n = int(n_pairs)
    if n == 1:
        raise ValueError("no derangement of a single pair")
    rng = np.random.default_rng([int(seed), 7118])
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 576-row task table; a pure function of the pinned constants (rows kept for the signature)."""
    out = []
    for ri, rung in enumerate(RUNGS):
        for gi, g in enumerate(G_IDS):
            for si, space in enumerate(SPACES):
                for mi, model in enumerate(MODELS):
                    for seed in SEEDS:
                        index = ri * 144 + gi * 18 + si * 9 + mi * 3 + seed
                        out.append({"index": index, "rung": rung, "g": g, "space": space,
                                    "model": model, "seed": seed,
                                    "run_name": f"{rung}_{g}_{space}_{model}_s{seed}"})
    assert [t["index"] for t in out] == list(range(len(out)))
    return out


def count_lines(rows: list[dict]) -> list[str]:
    lines = [f"usable_products {len(usable_products(rows))}"]
    lines += [f"train_pairs {t} {len(train_pairs(rows, t))}" for t in track_ids(rows)]
    lines.append(f"train_pairs all {len(train_pairs(rows, 'all'))}")
    lines.append(f"law_pairs all {len(law_pairs(rows, 'all'))}")
    lines.append(f"tasks {len(tasks(rows))}")
    return lines


def _tsv(records: list[dict], keys) -> str:
    def cell(v):
        return str(int(v)) if isinstance(v, bool) else str(v)
    return "".join("\t".join(cell(r[k]) for k in keys) + "\n" for r in records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "tasks", "train-pairs", "law-pairs"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        if name in ("train-pairs", "law-pairs"):
            p.add_argument("g")
    args = ap.parse_args(argv)
    rows = read_manifest(args.manifest)
    if args.cmd == "count":
        sys.stdout.write("".join(line + "\n" for line in count_lines(rows)))
    elif args.cmd == "tasks":
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + _tsv(tasks(rows), TASK_KEYS))
    else:
        fn = train_pairs if args.cmd == "train-pairs" else law_pairs
        sys.stdout.write("\t".join(PAIR_KEYS) + "\n" + _tsv(fn(rows, args.g), PAIR_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
