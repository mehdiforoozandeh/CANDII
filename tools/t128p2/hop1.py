"""t128 phase 2 — hop-1 scoring of the anchored arm's encoder (metric 6 of plan/T126_ENCDEC_D2.md
§ "Phase 2: PI decisions, 2026-10-25", arm c = DSanchor, rung DSANC).

The encoder half of rung DSANC is old D's NB head predicting X_base (`form.encode_nb`). This pass
scores that NB prediction against the REAL base counts, with the CRPS split, for every arm product
of the run's g tracks:
  trained   the arm products of `pairs.fit_pids` (the bases removed);
  heldout   the depth 7.5M products the run held out.
Each record is the pair `pairs._pair(arm_row, base_row, "arm_to_base")`, scored by
`score.score_job` (kind "hop1", eval "score": chr19 + chr21, blacklist bins removed) through a
`Hop1Predictor`: `predict(arm, arm, base, chrom)` = `encode_nb(log1p(X_arm), theta_e)` over the
whole chromosome, theta_e = g(c_arm, C_clean) = `_theta_t(arm, arm)[:256]`, chunked as
`latent.encode_chrom` with a 30-bin zero-padded halo (one f).

    python tools/t128p2/hop1.py run <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N]
        -> <run_dir>/hop1.json, <run_dir>/HOP1_DONE   (exit 2 when the run's rung is not DSANC)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # `ladder` from tools/t118, `t128` from tools
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, score  # noqa: E402
from ladder.fforms import form_ds  # noqa: E402
from t128 import latent  # noqa: E402

RUNG = "DSANC"
#: the encoder's receptive field each side: ONE f (as `latent.HALO`)
HALO = form_ds.CTX1
RULE = ("NB head of the encoder, theta_e = g(c_arm, C_clean), scored against the base's counts "
        "on chr19+chr21 without blacklist bins")


@torch.no_grad()
def encode_nb_chrom(ladder, X: np.ndarray, theta_e: torch.Tensor, device,
                    chunk: int = model.CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """NB (loc, disp) float32[n] of the encoder head over a whole chromosome X (counts).

    As `latent.encode_chrom`, but calling `form.encode_nb`: chunks of `chunk` bins, each with a
    zero-padded halo of `HALO` (30) bins, outside the chromosome X = 0."""
    h = HALO
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * h, dtype=np.float64)
    Xp[h:h + n] = X
    xp = model.transform_x(Xp, ladder.space)
    theta = theta_e.reshape(1, -1).to(device=device, dtype=torch.float32)
    locs, disps = [], []
    for b0 in range(0, n_chunks, model.BATCH_CHUNKS):
        idx = range(b0, min(b0 + model.BATCH_CHUNKS, n_chunks))
        xb = np.stack([xp[i * chunk: i * chunk + chunk + 2 * h] for i in idx])
        xt = torch.from_numpy(xb).to(device)
        loc, disp = ladder.form.encode_nb(xt, theta.expand(xt.shape[0], -1))
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return (np.concatenate(locs)[:n].astype(np.float32),
            np.concatenate(disps)[:n].astype(np.float32))


class Hop1Predictor(model.Predictor):
    """A DSANC run whose `predict` is the encoder's NB head (X_arm -> X_base). Built from the
    `model.load_run` Predictor of the run and the manifest rows (which name each track's base)."""

    def __init__(self, pred: model.Predictor, rows: list[dict], chunk: int = model.CHUNK):
        if pred.rung != RUNG:
            raise ValueError(f"rung {pred.rung!r}: hop-1 needs rung {RUNG}")
        self.__dict__.update(pred.__dict__)
        self.chunk = int(chunk)
        bases = {r["track"]: r["pid"] for r in pairs.usable_products(rows) if r["arm"] == "base"}
        self.base_of = {r["pid"]: bases.get(r["track"]) for r in rows}

    @classmethod
    def load(cls, run_dir, data_dir, covariates_tsv, manifest_tsv, device="cpu"):
        pred = model.load_run(run_dir, data_dir, covariates_tsv, manifest_tsv, device=device)
        return cls(pred, pairs.read_manifest(manifest_tsv))

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        """(loc, disp) float32[n]: the NB head of the encoder over the chromosome of `x_src_pid`,
        theta_e = g(c_src, C_clean); `cov_tgt_pid` must be the base of `x_src_pid`'s track."""
        base = self.base_of.get(x_src_pid)
        if base is None or cov_tgt_pid != base:
            raise ValueError(f"hop 1 predicts the base: target {cov_tgt_pid!r} is not the base "
                             f"of {x_src_pid!r} ({base!r})")
        X = self.corpus.get(x_src_pid, self.space, chrom)
        theta_e = self._theta_t(cov_src_pid, cov_src_pid)[:form_ds.FILM]
        return encode_nb_chrom(self.ladder, X, theta_e, self.device, chunk=self.chunk)


def jobs_of(pred, rows: list[dict]) -> list[tuple[str, dict]]:
    """(group, pair) per arm product of the run's g: trained (fit pids minus the bases), then
    heldout (the products of the run's hold-out), each `_pair(arm, base, "arm_to_base")`."""
    holdout = [tuple(h) for h in (pred.config or {}).get("holdout", ())]
    fit = set(pairs.fit_pids(rows, pred.g_id, holdout))
    held = {(str(a), str(lv)) for a, lv in holdout}
    out = {"trained": [], "heldout": []}
    for track in pairs._tracks_of(rows, pred.g_id):
        base, arms = pairs._by_track(rows, track)
        for r in arms:
            grp = ("trained" if r["pid"] in fit else
                   "heldout" if (str(r["arm"]), str(r["level"])) in held else None)
            if grp is not None:
                out[grp].append(pairs._pair(r, base, "arm_to_base"))
    return [(g, p) for g in ("trained", "heldout") for p in out[g]]


def run(manifest, covariates, data_dir, blacklist, run_dir, workers: int = 1,
        device: str = "cpu", chunk: int = model.CHUNK) -> Path:
    """hop1 records -> `<run_dir>/hop1.json`, then `HOP1_DONE`."""
    t0 = time.time()
    run_dir = Path(run_dir)
    rows = pairs.read_manifest(manifest)
    pred = Hop1Predictor(model.load_run(run_dir, data_dir, covariates, manifest, device=device),
                         rows, chunk=chunk)
    jobs = jobs_of(pred, rows)
    score._setup(pred, rows, blacklist)
    specs = [{"kind": "hop1", "eval": "score", "pair": p, "x_pid": p["source_pid"],
              "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": p["target_pid"],
              "split": True, "pit": True, "describe": False} for _, p in jobs]
    t1 = time.time()
    results = score._run_jobs(specs, workers)
    t2 = time.time()
    records = []
    for (grp, _), (recs, _, _) in zip(jobs, results):
        for r in recs:
            records.append({**r, "group": grp})
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": score.run_info(pred, run_dir),
           "config": {"chroms": list(pairs.SCORE_CHROMS), "blacklist": str(blacklist),
                      "blacklist_sha256": hashlib.sha256(Path(blacklist).read_bytes()).hexdigest(),
                      "rule": RULE, "halo": HALO},
           "records": records,
           "timing": {"setup": round(t1 - t0, 3), "score": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(specs),
                      "workers": int(workers)}}
    score.write_json(out, run_dir / "hop1.json")
    (run_dir / "HOP1_DONE").write_text(out["created_utc"] + "\n")
    score._CTX.clear()
    return run_dir / "hop1.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    for pos in ("manifest", "covariates", "data_dir", "blacklist", "run_dir"):
        a.add_argument(pos, type=Path)
    a.add_argument("--workers", type=int, default=score.default_workers())
    args = ap.parse_args(argv)
    done = args.run_dir / "HOP1_DONE"
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    rung = latent._rung_of(args.run_dir)
    if rung != RUNG:
        print(f"{args.run_dir.name}: rung {rung!r} has no NB encoder head (want {RUNG})",
              file=sys.stderr)
        return 2
    t0 = time.time()
    path = run(args.manifest, args.covariates, args.data_dir, args.blacklist, args.run_dir,
               workers=args.workers)
    print(f"hop1 {args.run_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
