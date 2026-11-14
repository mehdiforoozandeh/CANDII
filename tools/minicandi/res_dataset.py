"""miniCANDI at another bin size (Amendment 3, section C, chunk RM): the dataset and the train entry.

    python tools/minicandi/res_dataset.py train --config <json> --tables <eic_res_v1> --seed <k> \\
        --out <run dir> [the other options of train.py]

`ResDataset(tables_dir, mode, config, seed)` is `OnionDataset` on the region store of pinned
interface E. A token is `res_bp` bp (config `res_bp` in 1, 5, 25, 50, 125; default 25): its counts are
`recount.pool(point1bp, res_bp)` of the 1 bp point-rule files, `x_dna` is the window's `L * res_bp`
bases, and every other key (covariates, kinds, masks, `meta`) is built by `OnionDataset.window`
unchanged. The batch is the contract batch at L tokens of `res_bp`
(`contract.validate_batch(batch, res_bp=res_bp)`).

Positions. A window starts at a base pair, not at a bin: `read`, `dna` and `window` take
`start` in bp (a token grid that starts at each region's first base, so a 125 bp arm tiles a
409,600 bp region from its start). `meta` holds `start_bp`, `res_bp`, and `start_bin` =
start_bp // 25 (the 25 bp bin of the first base).

Files. A row's 1 bp files are `<res_store>/<measurement id>/point1bp.h5` (`pr1`, `pr2`:
`point1bp_pr1.h5`, `point1bp_pr2.h5`) when the config sets `res_store`, else beside the row's
`counts_path` (`h5:<dir>/counts.h5`). Each holds one dataset `/r<region index>` per region of
`regions.tsv` (the tables directory's copy, else `<res_store>/regions.tsv`); a position outside
every region, or a region the file does not hold, reads 0.

Training windows come from the regions with role `train` only: a region uniformly, then a start
uniformly over the bases that keep the whole window inside it (the regions were drawn with a mask
valid fraction >= 0.9, so the genome mask is not read again). Validation and blind windows tile
each evalset region from its first base in windows of `L * res_bp` bp (`region_bins` of the
evalset are 25 bp bins).

Decoupled arms (`res_bp` = 25). `res_in_sub = 5` adds `x_counts_sub` int32 [B, L, A, 5]: the five
5 bp counts of every input token (0 where the token is hidden or the track missing; their sum is
`x_counts`). `res_out_sub = 5` adds `y_counts_sub` int32 [B, L, A, 5] of every target (their sum
is `y_counts`); the loss is then at 5 bp (`res_total_loss`).

Refused (ValueError), because the region store has no layer for them: `two_step`, `rna_track`,
`dna_embedding_h5`, `dna = precomputed`, `dna_shift_bp`, `average_h5`, a `store:` row; and with a
sub key, `aug_revcomp` and `input_depth = fixed`.

Training. `train(config, tables, seed, out, ...)` runs `train.train` unchanged, with two names of
that module swapped for the call: its dataset class by `make_dataset` (a `ResDataset` when the
config sets `res_store` or a resolution key off its default, else `OnionDataset`), and its loss by
`res_total_loss` (the loss of `losses.total_loss`, taken at the sub-bins when the output holds
`mu_sub`). `train.py` writes the measured FLOPs per window into `run.json` as for every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import fine_to_sub, res_bp_of, res_subs_of, sub_to_fine  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.losses import total_loss  # noqa: E402
from onion_data.recount import REGION_BP, pool, read_regions  # noqa: E402

POINT_FILES = {"counts": "point1bp.h5", "pr1": "point1bp_pr1.h5", "pr2": "point1bp_pr2.h5"}
#: config keys that the region store cannot serve
REFUSED = ("two_step", "rna_track", "dna_embedding_h5", "dna_shift_bp", "average_h5")


def is_res_config(cfg: dict) -> bool:
    """True when the config trains from the region store (`res_store` set, or a resolution key
    off its default)."""
    return bool(cfg.get("res_store")) or res_bp_of(cfg) != C.RES or max(res_subs_of(cfg)) > 1


def make_dataset(tables_dir, mode: str, config: dict | None = None, seed: int = 0):
    """`ResDataset` for a resolution config (`is_res_config`), else `OnionDataset`."""
    cfg = dict(config or {})
    cls = ResDataset if is_res_config(cfg) else OnionDataset
    return cls(tables_dir, mode, cfg, seed)


class ResDataset(OnionDataset):
    """See the module docstring."""

    def __init__(self, tables_dir, mode: str, config: dict | None = None, seed: int = 0):
        cfg = dict(config or {})
        self.res = res_bp_of(cfg)
        self.in_sub, self.out_sub = res_subs_of(cfg)
        bad = [k for k in REFUSED if cfg.get(k)]
        if cfg.get("dna") == "precomputed":
            bad.append("dna = precomputed")
        if max(self.in_sub, self.out_sub) > 1:
            if cfg.get("aug_revcomp"):
                bad.append("aug_revcomp with a sub-bin key")
            if cfg.get("input_depth", "native") != "native":
                bad.append("input_depth = fixed with a sub-bin key")
        if bad:
            raise ValueError(f"ResDataset (region store) does not serve {bad}")
        super().__init__(tables_dir, mode, cfg, seed)
        sizes = self.genome.chrom_sizes
        self.n_bins = {c: int(sizes[c]) // self.res for c in self.n_bins}
        store = cfg.get("res_store")
        self.res_store = Path(store) if store else None
        reg = self.tables_dir / "regions.tsv"
        if not reg.is_file() and self.res_store is not None:
            reg = self.res_store / "regions.tsv"
        if not reg.is_file():
            raise FileNotFoundError(f"{self.tables_dir}/regions.tsv: missing (the region store's "
                                    f"regions)")
        self.regions = read_regions(reg)
        self.region_bp = REGION_BP
        self._by_chrom: dict = {}
        for r in self.regions:
            self._by_chrom.setdefault(r["chrom"], []).append(r)
        self.train_regions = [r for r in self.regions if r["role"] == "train"]
        self.window_bp = self.L * self.res
        if self.window_bp > self.region_bp:
            raise ValueError(f"a window of {self.L} tokens of {self.res} bp ({self.window_bp} bp) "
                             f"does not fit in a {self.region_bp} bp region")
        if mode == "train" and not self.train_regions:
            raise ValueError(f"{reg}: no region with role train")

    # ---------------------------------------------------------------- reading

    def point_path(self, mid: str, which: str = "counts") -> Path:
        row = self.check_row(mid)
        if which not in POINT_FILES:
            raise ValueError(f"which {which!r} not in {tuple(POINT_FILES)}")
        if self.res_store is not None:
            return self.res_store / mid / POINT_FILES[which]
        p = row["counts_path"]
        if not p.startswith("h5:"):
            raise ValueError(f"{mid}: counts_path {p!r} is not h5: (the region store is)")
        return Path(p[3:]).parent / POINT_FILES[which]

    def read_bp(self, mid: str, chrom: str, start_bp: int, n_bp: int,
                which: str = "counts") -> np.ndarray:
        """int64 [n_bp] of the 1 bp point-rule counts from base `start_bp`; 0 outside the
        regions the file holds."""
        self._fresh()
        out = np.zeros(int(n_bp), np.int64)
        s, e = int(start_bp), int(start_bp) + int(n_bp)
        f = None
        for r in self._by_chrom.get(chrom, ()):
            a, b = max(s, r["start_bp"]), min(e, r["start_bp"] + self.region_bp)
            if a >= b:
                continue
            if f is None:
                f = self._h5file(self.point_path(mid, which))
            key = f"r{r['region']}"
            if key in f:
                out[a - s:b - s] = f[key][a - r["start_bp"]:b - r["start_bp"]]
        return out

    def read(self, mid: str, chrom: str, start: int, L: int | None = None,
             which: str = "counts") -> np.ndarray:
        """int32 [L] counts of L tokens of `res_bp` from base `start`."""
        L = self.L if L is None else int(L)
        return pool(self.read_bp(mid, chrom, start, L * self.res, which), self.res).astype(np.int32)

    def read_sub(self, mid: str, chrom: str, start: int, L: int, s: int,
                 which: str = "counts") -> np.ndarray:
        """int32 [L, s]: the s sub-bins of each of L tokens from base `start`."""
        x = self.read_bp(mid, chrom, start, L * self.res, which)
        return pool(x, self.res // s).reshape(L, s).astype(np.int32)

    def dna(self, chrom: str, start: int, L: int | None = None) -> np.ndarray:
        L = self.L if L is None else int(L)
        n = L * self.res
        out = np.full(n, 4, dtype=np.uint8)
        end = min(int(start) + n, self.genome.chrom_sizes[chrom]) if self.genome.has_dna else 0
        if end > int(start):
            self._fresh()
            v = self.genome.dna(chrom, int(start), end)
            out[: v.shape[0]] = v
        return out

    # ---------------------------------------------------------------- windows

    def _draw_locus(self, rng) -> tuple[str, int]:
        r = self.train_regions[int(rng.integers(len(self.train_regions)))]
        off = int(rng.integers(0, self.region_bp - self.window_bp + 1))
        return r["chrom"], r["start_bp"] + off

    def window(self, chrom: str, start_bin: int, inputs: list, targets: list, **kw) -> dict:
        """`OnionDataset.window` at base `start_bin` (bp, see the module docstring), plus the
        sub-bin keys of the decoupled arms."""
        start = int(start_bin)
        win = super().window(chrom, start, inputs, targets, **kw)
        meta = win["meta"]
        meta.update({"start_bp": start, "start_bin": start // C.RES, "res_bp": self.res})
        L = self.L
        if self.in_sub > 1:
            s = self.in_sub
            sub = np.zeros((L, C.A, s), np.int32)
            hidden = win["x_bin_mask"].numpy()
            for t, m in enumerate(meta["measurements_in"]):
                if m:
                    v = self.read_sub(m, chrom, start, L, s, meta["input_file"])
                    v[hidden[:, t]] = 0
                    sub[:, t] = v
            if not np.array_equal(sub.sum(-1), win["x_counts"].numpy()):
                raise ValueError(f"{chrom}:{start}: the 5 bp input sub-bins do not sum to x_counts")
            win["x_counts_sub"] = torch.from_numpy(sub)
        if self.out_sub > 1:
            s = self.out_sub
            sub = np.zeros((L, C.A, s), np.int32)
            for t, m in enumerate(meta["measurements_out"]):
                if m:
                    sub[:, t] = self.read_sub(m, chrom, start, L, s, meta["target_file"])
            if not np.array_equal(sub.sum(-1), win["y_counts"].numpy()):
                raise ValueError(f"{chrom}:{start}: the 5 bp target sub-bins do not sum to "
                                 f"y_counts")
            win["y_counts_sub"] = torch.from_numpy(sub)
        return win

    def evalset_items(self) -> list[dict]:
        """`OnionDataset.evalset_items` with each evalset region (`[chrom, start bin of 25 bp]`)
        tiled from its first base in whole windows of `L * res_bp` bp; an item's `start_bin` is a
        base pair here. Items of one task keep the parent's order (job, then region, then tile)
        and the tasks are interleaved as in the parent."""
        if self.mode == "train":
            raise RuntimeError("evalset_items() is for val and blind modes")
        path = self.config.get("evalset") or (self.tables_dir.parent.parent / "eval" / self.tables_dir.name
                         / "evalset.json")
        ev = json.loads(Path(path).read_text(encoding="utf-8"))
        jobs = _eval_jobs(ev, self.mode)
        rb_bp = int(self.config["region_bins"]) * C.RES
        per_task: dict = {}
        for j in jobs:
            for chrom, start in ev.get("regions", []):
                s0 = int(start) * C.RES
                for k in range(max(rb_bp // self.window_bp, 1)):
                    b = s0 + k * self.window_bp
                    if b + self.window_bp > self.genome.chrom_sizes.get(chrom, 0):
                        break
                    per_task.setdefault(j["task"], []).append(dict(j, chrom=chrom, start_bin=b))
        lists = [per_task[t] for t in ("imputation", "correction", "reconstruction")
                 if t in per_task]
        items = [lst[i] for i in range(max((len(x) for x in lists), default=0))
                 for lst in lists if i < len(lst)]
        for i, it in enumerate(items):
            it["item"] = i
        return items


def _eval_jobs(ev: dict, split: str) -> list:
    """The job dicts of an eval set's split, as `OnionDataset.evalset_items` builds them (the same
    three tasks and keys; a copy, since the parent builds them inside its tiling loop)."""
    jobs = []
    for t in ev.get("imputation_targets", []):
        if t.get("split") == split:
            jobs.append({"inputs": list(t["inputs"]), "targets": [t["target"]],
                         "task": "imputation", "regime": t.get("regime")})
    for p in ev.get("correction_pairs", []):
        if p.get("split") == split:
            jobs.append({"inputs": [p["source"]], "targets": [p["target"]],
                         "task": "correction", "input_file": "pr1", "target_file": "pr2",
                         "never_trained": p.get("never_trained"),
                         "subset_pair": p.get("subset_pair")})
    seen = set()
    for t in ev.get("imputation_targets", []):
        if t.get("split") == split and t["target"] not in seen:
            seen.add(t["target"])
            ctx = [m for m in t["inputs"] if m != t["target"]]
            jobs.append({"inputs": [t["target"]] + ctx, "targets": [t["target"]],
                         "task": "reconstruction"})
    return jobs


def res_total_loss(out: dict, batch: dict, config: dict | None) -> tuple[torch.Tensor, dict]:
    """`losses.total_loss`, at the sub-bins when the output holds `mu_sub` (`res_out_sub`): the
    output, `y_counts_sub` and the bin mask are laid out as [B, L*s, A] first."""
    if "mu_sub" not in out:
        return total_loss(out, batch, config)
    if "y_counts_sub" not in batch:
        raise KeyError("an output with mu_sub needs batch['y_counts_sub'] (res_out_sub)")
    s = out["mu_sub"].shape[-1]
    fine_out = {k: v for k, v in out.items() if k not in ("n", "mu", "n_sub", "mu_sub")}
    fine_out.update(n=sub_to_fine(out["n_sub"]), mu=sub_to_fine(out["mu_sub"]))
    fine_batch = dict(batch, y_counts=sub_to_fine(batch["y_counts_sub"]),
                      x_bin_mask=batch["x_bin_mask"].repeat_interleave(s, dim=1))
    return total_loss(fine_out, fine_batch, config)


def train(config: dict, tables, seed: int, out, *args, **kw) -> dict:
    """`train.train` with the region-store dataset and the sub-bin loss (module docstring)."""
    from minicandi import train as T
    saved = (T.OnionDataset, T.total_loss)
    T.OnionDataset, T.total_loss = make_dataset, res_total_loss
    try:
        return T.train(config, tables, seed, out, *args, **kw)
    finally:
        T.OnionDataset, T.total_loss = saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="train.py's options; the dataset reads the region store")
    p.add_argument("--config", required=True)
    p.add_argument("--tables", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-minutes", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    info = train(config, a.tables, a.seed, a.out, a.device, a.max_windows, a.val_every,
                 a.val_batches, a.log_every, a.ckpt_minutes, a.workers)
    print(json.dumps({k: info[k] for k in ("windows_seen", "steps", "param_count",
                                           "flops_per_window", "training_flops", "done")}))
    return 0


__all__ = ["POINT_FILES", "ResDataset", "is_res_config", "make_dataset", "res_total_loss",
           "train"]

if __name__ == "__main__":
    sys.exit(main())
