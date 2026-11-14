"""PG -- genome-wide prediction of a miniCANDI run and its export in the EIC challenge format.

    python tools/minicandi/predict.py --run <run dir> --tables <dir> --evalset <json>
        --split val|blind [--allow-blind --freeze <FREEZE.md>] --out <dir>
        [--chroms chr21[,chr22]] [--stride-bins 512] [--device auto] [--infer-batch N]
        [--cell-index i] [--repeat-blind-reason "<text>"]
    -> <out>/counts/<eic_track>.h5          groups `mu` and `n`, one float16 dataset per chromosome
       <out>/signal/<eic_track>.npy         pickled {chrom: float32} on the raw 25 bp grid (the
       <out>/signal_plug_in/<eic_track>.npy challenge's own .npy, read by eic_score.load_prediction)
       <out>/lambda_source.tsv, <out>/DONE  (both after the last target of the split)

Targets: the eval set's `imputation_targets` of the split with regime `leave_one_assay_out` (its
inputs are the leave-one-assay-out inputs). The target's own counts are never read: the target's
track (every view of that cell and assay that is not an input) reads as zeros while the windows are
built, so a prediction consumes no truth. Each window is decoded with the target's own C'.

Tiling: windows of the run's `context_bins` L start every `--stride-bins` s bins (s <= L, L - s
even). Window k keeps its centre s bins [w + m, w + m + s), m = (L - s) / 2; the first window also
keeps [0, m) and the last keeps everything up to the chromosome end (a last window starts at
n - L when the stride does not reach the end). The dataset grid is floor(bp / 25) bins; the
challenge grid is ceil(bp / 25): the partial last bin (when there is one) gets signal 0.

Signal: `signal_from_counts.signal_from_nb` on each whole chromosome, `expectation` (headline) and
`plug_in` beside it (PI ruling 20); DNase-seq by `macs2_no_control` (ruling 19). A ChIP-seq target
with no `control_measurement_id` uses the model's predicted control (a second forward pass with the
control channel asked for at the target's own C', as the scorer does), labelled
`predicted_control`; a run without `predict_control` is refused for such a target.

Resume: per target and chromosome, `<out>/parts/<eic_track>/<chrom>.npz` is written atomically after
that chromosome's counts; a target is finished when both .npy files exist, and its parts are then
removed. A rerun skips finished targets and finished chromosomes. `--chroms` restricts the job to
some chromosomes (the cost probe: chr21); the .npy files and DONE need every challenge chromosome
(chr1 to chr22 and chrX that the genome holds). `--cell-index i` (Slurm arrays over cells) keeps the
targets of the i-th cell of the split, cells sorted; an index past the last cell exits 0.

Blind discipline (gate G4). `--split blind` needs `--allow-blind`, an eval set built with
`--include-blind`, and `--freeze <FREEZE.md>`: the file must exist, be non-empty and name this run
directory (the `--run` path as given or resolved), else the job is refused before any blind read.
Every blind prediction is ledgered in `BLIND_PASSES.tsv` beside the eval set with the scorer's own
columns and writer (`score.LEDGER_COLUMNS`, `score.ledger_append`), under the key
`predict:<resolved run dir>` (never the bare run dir, which is the scorer's key), status `started`
before the first target and `predicted` after the last. A job that finds its own open pass (a
`started` row of this key, eval set and out with no `predicted` row after it) continues without a
new row; any other earlier row of the key and eval set refuses the job unless
`--repeat-blind-reason "<text>"` is given (recorded in `repeat_reason`). `<out>/DONE` present: the
job prints skip and exits 0 before it reads anything.

Size: 51 tracks x 123.5 M bins x 2 arrays x 2 bytes = about 25 GB of counts per model, beside
about 50 GB of signal (two float32 tracks per target).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import OnionDataset, tables_sha256  # noqa: E402
from onion_eval import eic_score as EIC  # noqa: E402
from onion_eval import score as S  # noqa: E402
from onion_eval import signal_from_counts as SIG  # noqa: E402

KEY_PREFIX = "predict:"
STARTED, PREDICTED = "started", "predicted"
REGIME = "leave_one_assay_out"
DNASE_ROUTE = "macs2_no_control"                     # PI ruling 19
SIGNAL_DIRS = (("signal", "expectation"), ("signal_plug_in", "plug_in"))   # PI ruling 20
F16_MAX = float(np.finfo(np.float16).max)
DONE = "DONE"
LAMBDA_COLUMNS = ("eic_track", "measurement_id", "cell", "assay", "lambda_source")


class PredictRefused(Exception):
    """The job may not run (blind discipline, tables or ledger); nothing was read."""


# ------------------------------------------------------------------------------------- pieces

def ledger_key(run) -> str:
    return KEY_PREFIX + str(Path(run).resolve())


def check_freeze(freeze, run) -> str:
    """sha256 of the freeze file, which must name `run`; else PredictRefused."""
    if not freeze:
        raise PredictRefused("--split blind needs --freeze <FREEZE.md> (the frozen list of runs "
                             "that may read blind data)")
    p = Path(freeze)
    if not p.is_file():
        raise PredictRefused(f"freeze file {p} does not exist")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise PredictRefused(f"freeze file {p} is empty")
    names = {str(run).rstrip("/"), str(Path(run).absolute()), str(Path(run).resolve())}
    if not any(n and n in text for n in names):
        raise PredictRefused(f"run {run} is not named in the freeze file {p}")
    return S.sha256_file(p)


def tiles(n: int, L: int, stride: int) -> list[tuple[int, int, int]]:
    """(window start, keep lo, keep hi) covering [0, n) once, centre crops of `stride` bins."""
    if not 0 < stride <= L or (L - stride) % 2:
        raise ValueError(f"stride {stride} must be in (0, {L}] with {L} - stride even")
    if n < L:
        raise ValueError(f"chromosome of {n} bins is shorter than the {L}-bin window")
    m = (L - stride) // 2
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] + L < n:
        starts.append(n - L)
    out, lo = [], 0
    for k, w in enumerate(starts):
        hi = n if k == len(starts) - 1 else w + m + stride
        out.append((w, lo, hi))
        lo = hi
    return out


def raw_grid_bins(chrom_bp: int) -> int:
    return (int(chrom_bp) - 1) // EIC.WINDOW + 1


def _atomic_npy(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, obj, allow_pickle=True)
    os.replace(tmp, path)


@contextlib.contextmanager
def _lock(path: Path):
    """An exclusive lock on <ledger>.lock (best effort: a filesystem without flock goes on)."""
    fh = open(str(path) + ".lock", "a")
    try:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):              # pragma: no cover - no flock on this fs
            pass
        yield
    finally:
        fh.close()


def open_pass(rows: list[dict], key: str, esha: str, out: str):
    """The last `started` row of (key, eval set, out) with no `predicted` row after it, or None."""
    last = None
    for r in rows:
        if r["run"] != key or r["evalset_sha256"] != esha or r["out"] != out:
            continue
        last = r if r["status"] == STARTED else (None if r["status"] == PREDICTED else last)
    return last


# ------------------------------------------------------------------------------------- predictor

class Predictor:
    def __init__(self, run, tables, evalset, *, split="val", allow_blind=False, freeze=None,
                 out, model=None, config=None, device="cpu", stride_bins=512, chroms=None,
                 infer_batch=None, cell_index=None, repeat_blind_reason=None, log=print):
        if split not in ("val", "blind"):
            raise ValueError("split must be val or blind")
        if split == "blind" and not allow_blind:
            raise PermissionError("predicting blind targets needs --allow-blind (gate G4)")
        self.log, self.split, self.run = log, split, Path(run)
        self.out = Path(out)
        self.evalset_path = Path(evalset).resolve()
        self.ev = json.loads(self.evalset_path.read_text(encoding="utf-8"))
        if split == "blind":
            if not self.ev.get("include_blind"):
                raise PredictRefused(f"{self.evalset_path}: not a blind eval set (built without "
                                     f"--include-blind)")
            self.freeze_sha256 = check_freeze(freeze, run)
        self.tables = Path(tables)
        self.tables_sha256 = tables_sha256(self.tables)
        if not self.ev.get("tables_sha256"):
            raise PredictRefused(f"{self.evalset_path}: no tables_sha256 (rebuild the eval set)")
        if self.ev["tables_sha256"] != self.tables_sha256:
            raise PredictRefused(f"tables {self.tables} (sha256 {self.tables_sha256[:12]}) are not "
                                 f"the eval set's ({self.ev['tables_sha256'][:12]})")
        self.repeat_blind_reason = (repeat_blind_reason or "").strip()
        if model is None:
            if split == "val":
                rs = S.run_tables_sha256(run)
                if rs != self.tables_sha256:
                    raise PredictRefused(f"the run was trained on tables {rs[:12]}, the eval "
                                         f"set's are {self.tables_sha256[:12]}")
            from minicandi.registry import load_run
            model, config = load_run(run, map_location=device)
        self.model = model.to(device).eval()
        self.device = device
        cfg = dict(config or {})
        cfg.pop("chroms", None)                       # genome-wide, whatever training used
        cfg["allow_blind"] = split == "blind"
        cfg.setdefault("batch_size", 8)
        self.ds = OnionDataset(self.tables, split, cfg, seed=0)
        self.L = self.ds.L
        self.stride = int(stride_bins)
        tiles(self.L, self.L, self.stride)            # validates the stride early
        self.bs = int(infer_batch or self.ds.config["batch_size"])
        self.predict_control = bool(self.ds.config.get("predict_control"))
        self.full = [c for c in EIC.CHROMS if c in self.ds.n_bins]
        missing = [c for c in EIC.CHROMS if c not in self.ds.n_bins]
        if missing:
            log(f"warning: the genome lacks {missing}; the .npy files hold {len(self.full)} chroms")
        self.chroms = list(chroms) if chroms else list(self.full)
        bad = [c for c in self.chroms if c not in self.full]
        if bad:
            raise ValueError(f"--chroms {bad} not among the challenge chromosomes of the genome")
        self.jobs = self._jobs()
        cells = sorted({j["cell"] for j in self.jobs})
        self.cells = cells
        if cell_index is None:
            self.mine = list(self.jobs)
        else:
            ci = int(cell_index)
            self.mine = [j for j in self.jobs if ci < len(cells) and j["cell"] == cells[ci]]

    # ------------------------------------------------------------------ targets
    def _jobs(self) -> list[dict]:
        jobs, seen = [], set()
        for t in self.ev["imputation_targets"]:
            if t["split"] != self.split or t.get("regime", REGIME) != REGIME:
                continue
            row = self.ds.check_row(t["target"])
            if row["assay"] == C.CONTROL:
                raise ValueError(f"control track {t['target']} listed as an imputation target")
            for m in t["inputs"]:
                self.ds.check_row(m)
            track = (row.get("eic_track") or "").strip() or t["target"]
            if track in seen:
                raise ValueError(f"two targets write the track {track}")
            seen.add(track)
            jobs.append({"target": t["target"], "inputs": list(t["inputs"]), "track": track,
                         "cell": row["cell"], "assay": row["assay"]})
        return jobs

    def finished(self, job) -> bool:
        return all((self.out / d / f"{job['track']}.npy").is_file() for d, _ in SIGNAL_DIRS)

    def _part(self, job, chrom) -> Path:
        return self.out / "parts" / job["track"] / f"{chrom}.npz"

    # ------------------------------------------------------------------ one chromosome
    @contextlib.contextmanager
    def _hidden_target(self, job):
        """While windows are built, the target's track reads as zeros (no truth is consumed)."""
        row = self.ds.rows[job["target"]]
        keep = set(job["inputs"]) | set(self.ds.train_base.values())
        hide = {m for m, r in self.ds.rows.items()
                if r["cell"] == row["cell"] and r["assay"] == row["assay"]} - keep
        orig = self.ds.read

        def read(mid, chrom, start, L=None, which="counts"):
            if mid in hide:
                self.ds.check_row(mid)
                return np.zeros(self.ds.L if L is None else int(L), np.int32)
            return orig(mid, chrom, start, L, which)
        self.ds.read = read
        try:
            yield
        finally:
            self.ds.read = orig

    def _forward(self, batch):
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            o = self.model(b)
        return {k: o[k].detach().cpu().double().numpy() for k in ("n", "mu")}

    def _needs_predicted_control(self, row) -> bool:
        return (SIG.route_of(row["assay_class"] or row["assay"]) == "chip"
                and not (row.get("control_measurement_id") or "").strip())

    def predict_chrom(self, job, chrom):
        """(n, mu, predicted control mean or None), float64 [n_bins(chrom)] of the target track."""
        row = self.ds.rows[job["target"]]
        t = C.TRACK_INDEX[row["assay"]]
        ci = C.TRACK_INDEX[C.CONTROL]
        want_ctl = self._needs_predicted_control(row)
        if want_ctl and not self.predict_control:
            raise PredictRefused(f"{job['target']}: a ChIP-seq target with no observed control "
                                 f"needs the model's predicted control (predict_control)")
        nb = self.ds.n_bins[chrom]
        n = np.empty(nb, np.float64)
        mu = np.empty(nb, np.float64)
        ctl = np.empty(nb, np.float64) if want_ctl else None
        tl = tiles(nb, self.L, self.stride)
        for b0 in range(0, len(tl), self.bs):
            part = tl[b0:b0 + self.bs]
            with self._hidden_target(job):
                batch = C.collate([self.ds.window(chrom, w, job["inputs"], [job["target"]])
                                   for w, _, _ in part])
            p = self._forward(batch)
            if want_ctl:
                b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                b2["c_out"][:, ci] = b2["c_out"][:, t]
                b2["y_avail"][:, ci] = True
                b2["y_kind"][:, ci] = C.KIND_IMPUTE
                pc = self._forward(b2)["mu"][:, :, ci]
            for i, (w, lo, hi) in enumerate(part):
                n[lo:hi] = p["n"][i, lo - w:hi - w, t]
                mu[lo:hi] = p["mu"][i, lo - w:hi - w, t]
                if want_ctl:
                    ctl[lo:hi] = pc[i, lo - w:hi - w]
        return n, mu, ctl

    def _write_counts(self, job, chrom, n, mu) -> None:
        d = self.out / "counts"
        d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / f"{job['track']}.h5", "a") as f:
            f.attrs.update({"measurement_id": job["target"], "eic_track": job["track"],
                            "dtype_note": "float16, clipped to 65504"})
            for name, arr in (("mu", mu), ("n", n)):
                g = f.require_group(name)
                if chrom in g:
                    del g[chrom]
                g.create_dataset(chrom, data=np.clip(arr, 0.0, F16_MAX).astype(np.float16))
            f.flush()

    def do_chrom(self, job, chrom) -> None:
        t0 = time.time()
        n, mu, ctl = self.predict_chrom(job, chrom)
        self._write_counts(job, chrom, n, mu)
        row = self.ds.rows[job["target"]]
        nb = self.ds.n_bins[chrom]
        sig, src = {}, None
        for _, mode in SIGNAL_DIRS:
            s, src = SIG.signal_from_nb(n, mu, row, ds=self.ds, regions=[(chrom, 0)], rb=nb,
                                        predicted_control=ctl, dnase_route=DNASE_ROUTE, mode=mode)
            sig[mode] = np.asarray(s, np.float32)
        p = self._part(job, chrom)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez(tmp, lambda_source=np.array(src), **sig)
        os.replace(tmp, p)
        self.log(f"{job['track']} {chrom}: {nb} bins in {time.time() - t0:.1f} s ({src})")

    def assemble(self, job) -> None:
        """The two challenge .npy files of a target from its chromosome parts; parts removed."""
        sizes = self.ds.genome.chrom_sizes
        tracks = {mode: {} for _, mode in SIGNAL_DIRS}
        src = None
        for c in self.full:
            with np.load(self._part(job, c)) as z:
                src = str(z["lambda_source"])
                for mode in tracks:
                    v = np.zeros(raw_grid_bins(sizes[c]), np.float32)
                    v[: z[mode].shape[0]] = z[mode]
                    tracks[mode][c] = v
        ld = self.out / "lambda"
        ld.mkdir(parents=True, exist_ok=True)
        (ld / f"{job['track']}.tsv").write_text(
            "\t".join([job["track"], job["target"], job["cell"], job["assay"], src]) + "\n",
            encoding="utf-8")
        for d, mode in SIGNAL_DIRS:
            (self.out / d).mkdir(parents=True, exist_ok=True)
            _atomic_npy(self.out / d / f"{job['track']}.npy", tracks[mode])
        for c in self.full:
            self._part(job, c).unlink()
        with contextlib.suppress(OSError):
            self._part(job, self.full[0]).parent.rmdir()

    # ------------------------------------------------------------------ ledger
    def _ledger_start(self):
        ledger = self.evalset_path.parent / S.BLIND_LEDGER
        key, esha = ledger_key(self.run), S.sha256_file(self.evalset_path)
        out = str(self.out.resolve())
        with _lock(ledger):
            rows = S.ledger_rows(ledger)
            own = open_pass(rows, key, esha, out)
            if own is not None:
                self.log(f"resume: open blind pass of {key} (started {own['time']}); no new row")
                return ledger, dict(own)
            prior = [r for r in rows if r["run"] == key and r["evalset_sha256"] == esha]
            if prior and not self.repeat_blind_reason:
                raise PredictRefused(f"{key} already had {len(prior)} blind-pass ledger rows for "
                                     f"this eval set ({ledger}); a repeat needs "
                                     f"--repeat-blind-reason \"<text>\"")
            entry = {"run": key, "evalset_sha256": esha, "tables_sha256": self.tables_sha256,
                     "git_sha": S.git_sha(), "out": out,
                     "repeat_reason": self.repeat_blind_reason if prior else ""}
            S.ledger_append(ledger, dict(entry, time=S._utc(), status=STARTED))
        return ledger, entry

    def _ledger_finish(self, ledger, entry) -> None:
        with _lock(ledger):
            if open_pass(S.ledger_rows(ledger), entry["run"], entry["evalset_sha256"],
                         entry["out"]) is not None:
                S.ledger_append(ledger, dict(entry, time=S._utc(), status=PREDICTED))

    # ------------------------------------------------------------------ run
    def run_all(self) -> dict:
        if not self.jobs:
            raise PredictRefused(f"the eval set has no {REGIME} imputation target on split "
                                 f"{self.split}")
        ledger = entry = None
        if self.split == "blind":
            ledger, entry = self._ledger_start()
        self.out.mkdir(parents=True, exist_ok=True)
        for job in self.mine:
            if self.finished(job):
                self.log(f"skip: {job['track']} finished")
                continue
            for c in self.chroms:
                if not self._part(job, c).is_file():
                    self.do_chrom(job, c)
            if all(self._part(job, c).is_file() for c in self.full):
                self.assemble(job)
        done = all(self.finished(j) for j in self.jobs)
        if done:
            with open(self.out / "lambda_source.tsv", "w", encoding="utf-8") as fh:
                fh.write("\t".join(LAMBDA_COLUMNS) + "\n")
                for j in self.jobs:
                    fh.write((self.out / "lambda" / f"{j['track']}.tsv").read_text(encoding="utf-8"))
            if ledger is not None:
                self._ledger_finish(ledger, entry)
            (self.out / DONE).write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n",
                                         encoding="utf-8")
        return {"targets": len(self.jobs), "mine": len(self.mine), "done": done,
                "ledger": str(ledger) if ledger else None}


def predict(run, tables, evalset, out, **kw) -> dict:
    return Predictor(run, tables, evalset, out=out, **kw).run_all()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--split", required=True, choices=("val", "blind"))
    ap.add_argument("--allow-blind", action="store_true")
    ap.add_argument("--freeze", default=None,
                    help="FREEZE.md that names this run (required with --split blind)")
    ap.add_argument("--repeat-blind-reason", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chroms", default=None, help="comma list, e.g. chr21 (default: all)")
    ap.add_argument("--stride-bins", type=int, default=512)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--infer-batch", type=int, default=None,
                    help="windows per forward pass (default: the run's batch_size)")
    ap.add_argument("--cell-index", type=int, default=None,
                    help="only the targets of the i-th cell of the split (cells sorted)")
    a = ap.parse_args(argv)
    if (Path(a.out) / DONE).is_file():
        print(f"skip: {a.out}/{DONE} exists")
        return 0
    S.set_threads()
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        p = Predictor(a.run, a.tables, a.evalset, out=a.out, split=a.split,
                      allow_blind=a.allow_blind, freeze=a.freeze, device=device,
                      stride_bins=a.stride_bins,
                      chroms=a.chroms.split(",") if a.chroms else None,
                      infer_batch=a.infer_batch, cell_index=a.cell_index,
                      repeat_blind_reason=a.repeat_blind_reason)
        if a.cell_index is not None and a.cell_index >= len(p.cells):
            print(f"skip: cell index {a.cell_index} >= {len(p.cells)} cells")
            return 0
        res = p.run_all()
    except (PermissionError, PredictRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"predicted {res['mine']} of {res['targets']} targets -> {a.out}; done {res['done']}"
          + (f"; ledger {res['ledger']}" if res["ledger"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
