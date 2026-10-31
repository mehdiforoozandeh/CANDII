"""h26 step 3 (Fir, CPU): decode one t112 track on chr21 under every probe prompt.

The covariate-swap probe of h26. The ENCODER input is the cell type's `T_` assays in the CANDI
`eic` corpus (the target column is held out by `_apply_loo_mask`, a no-op when the `T_` cell lacks
the assay).

RERUN 1 (PI ruling (iii), 2026-10-26): the 2019 challenge's blind-panel `B_` arrays are not used in
any form, not even loaded. The first run built the batch as the `T_ -> B_` pair, which read the `B_`
cell's counts into `y_data_imp` (never fed to the model, but loaded). Here the regime is rewritten
with no `eval_pairs` and no `B_`/`V_` biosample, the batch is the `T_` cell self-paired (so no
imputation keys are built), and `CorpusStore.__getitem__` refuses any `B_` name as a tripwire. A
track whose `T_` cell has no available assay has no usable prompt and is dropped (recorded). The latent is
computed once per window; the DECODER is then run once per prompt, and the prompts differ only in
the target column of `y_meta` (`[log2 depth, assay_id, read_length, run_type_id]`, the units
`store/dataset.py` uses: raw read length in bp, run type SE=0 / PE=1).

Prompts (all from the t112 cf MANIFEST row of each product):
  base        the base product's covariates
  crop36/50   base, with read_length replaced (ChIP tracks only)
  pe          base, with run_type replaced
  d15M/d7.5M/d3.75M   base, with log2 depth replaced (the positive control)

Writes `<out>/<track>_s<seed>.npz` (mu, n, eta per prompt, float32, chr21 length) and a json with
the swapwired evidence: the per-prompt y_meta vectors, the exact tensor elements that differ from
the base prompt, the training-corpus read-length range, and the checkpoint md5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from candi._vendored import CLOZE, MISSING
from candi.batch import make_masker, prepare_masked_batch
from candi.bench.harness import Pair, StoreSource, _apply_loo_mask
from candi.model import build_model_from_arch

CELL = {"C07": "Caco-2", "C12": "DND-41", "C19": "HAP-1", "C40": "SJSA1"}
EXPECTED_MD5 = {0: "26aeefc87534006dd987d480a1963935", 1: "c8d3b3c2996eeae8889c76df208fe204"}
RUN_TYPE_ID = {"single-ended": 0, "paired-ended": 1}
ROWS = ("depth", "assay_id", "read_length", "run_type")
CHROM = "chr21"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompts_for(track: str, rows: dict, assay_id: int) -> dict:
    base = rows[f"{track}__base__base"]
    vec = lambda d, rl, rt: [math.log2(float(d)), float(assay_id), float(rl), float(RUN_TYPE_ID[rt])]
    b = vec(base["depth"], base["read_length"], base["run_type"])
    out = {"base": b}
    for lvl in ("36", "50"):
        r = rows.get(f"{track}__crop__{lvl}")
        if r is not None:
            out[f"crop{lvl}"] = [b[0], b[1], float(r["read_length"]), b[3]]
    r = rows.get(f"{track}__pe__pe")
    if r is not None:
        out["pe"] = [b[0], b[1], b[2], float(RUN_TYPE_ID[r["run_type"]])]
    for lvl in ("15M", "7.5M", "3.75M"):
        r = rows.get(f"{track}__depth__{lvl}")
        if r is not None:
            out[f"d{lvl}"] = [math.log2(float(r["depth"])), b[1], b[2], b[3]]
    return out


def noblind_regime(regime_path: str, cell_bios: str, out_dir: Path) -> Path:
    """The regime with every B_ (and V_) biosample and every eval pair removed; eval = the T_ cell."""
    r = json.loads(Path(regime_path).read_text())
    r["eval_pairs"] = []
    r["biosamples"] = {"train": [b for b in r["biosamples"]["train"] if b.startswith("T_")],
                       "eval": [cell_bios]}
    r["_comment"] = f"h26 rerun 1: {regime_path} with no B_/V_ biosample and no eval_pairs (PI ruling (iii))"
    txt = json.dumps(r, indent=1)
    assert '"B_' not in txt, "a B_ name survived in the derived regime"
    out = out_dir / f"regime.noblind.{cell_bios}.json"
    out.write_text(txt)
    return out


def forbid_blind() -> list:
    """Make any attempt to open a B_ biosample in the store raise. Returns the opened-name log."""
    from candi.store.reader import CorpusStore
    opened, orig = [], CorpusStore.__getitem__

    def guarded(self, name):
        if str(name).startswith("B_"):
            raise RuntimeError(f"h26 rerun 1: refusing to open blind-panel biosample {name} (ruling iii)")
        opened.append(str(name))
        return orig(self, name)
    CorpusStore.__getitem__ = guarded
    return opened


def train_read_lengths(regime: dict) -> dict:
    man = json.loads((Path(regime["store"]) / "manifest.json").read_text())
    rls, rts = [], []
    for b in regime["biosamples"]["train"]:
        for t in man["biosamples"][b]["tracks"]:
            if t["assay"] in regime["assays"]:
                rls.append(int(t["read_length"]))
                rts.append(t["run_type"])
    vals, cnt = np.unique(rls, return_counts=True)
    return {"min": int(min(rls)), "max": int(max(rls)),
            "counts": {int(v): int(c) for v, c in zip(vals, cnt)},
            "run_types": sorted(set(rts))}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-dir", default="/project/def-maxwl/mforooz/t81_checkpoints")
    ap.add_argument("--regime", default="configs/regime.eic_19.json")
    ap.add_argument("--cf-manifest", default="/project/def-maxwl/mforooz/t112_cf/products/MANIFEST.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-windows", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Path(a.ckpt_dir) / f"t81_eic_19_s{a.seed}.best.ckpt"
    arch_json = Path(a.ckpt_dir) / f"t81_eic_19_s{a.seed}.json"
    got_md5 = md5(ckpt)
    if got_md5 != EXPECTED_MD5[a.seed]:
        raise SystemExit(f"checkpoint md5 {got_md5} != recorded {EXPECTED_MD5[a.seed]} (TRAIN_CANDI_EIC19.md)")
    arch = json.loads(arch_json.read_text())["config"]["arch"]
    model = build_model_from_arch(arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()

    opened = forbid_blind()
    rows = {r["pid"]: r for r in csv.DictReader(open(a.cf_manifest), delimiter="\t")}
    base = rows[f"{a.track}__base__base"]
    cell = CELL[base["cell"]]
    regime = json.loads(Path(a.regime).read_text())
    t_bios = f"T_{cell}"
    regime_nb = noblind_regime(a.regime, t_bios, out)
    src = StoreSource(regime_nb, chroms=[CHROM], biosamples=[t_bios])
    pair = Pair(t_bios, t_bios)
    assert not src.eval_pairs, src.eval_pairs
    col = src.assays.index(base["assay"])
    t_avail = src.ds._availability[t_bios]
    prompt_assays = [src.assays[i] for i in range(len(src.assays)) if bool(t_avail[i]) and i != col]
    if not prompt_assays:
        (out / f"{a.track}_s{a.seed}.json").write_text(json.dumps(
            {"track": a.track, "seed": a.seed, "dropped": f"{t_bios} has no available assay (ruling iii)"}))
        print(f"[h26] DROPPED {a.track}: {t_bios} has no usable prompt", flush=True)
        return
    prompts = prompts_for(a.track, rows, col)
    names = list(prompts)
    rl_train = train_read_lengths(regime)

    n_bins = src.n_bins(CHROM)
    starts = src.windows(CHROM)
    L = src.context_bins
    buf = {p: {k: np.zeros(n_bins, np.float32) for k in ("mu", "n", "eta")} for p in names}
    written = np.zeros(n_bins, bool)
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    wired = {}
    t0 = time.time()
    for lo in range(0, len(starts), a.batch_windows):
        chunk = starts[lo:lo + a.batch_windows]
        batch = src.batch(pair, CHROM, chunk, "impute")
        _apply_loo_mask(batch, col)
        prep = prepare_masked_batch(batch, noop, torch.device("cpu"), apply_mask=False)
        assert prep is not None and prep["x_data"].shape[0] == len(chunk), "a window was dropped"
        x_meta0 = prep["x_meta"].clone()
        z = model.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
        ym_base = None
        for p in names:
            ym = prep["y_meta"].clone()
            ym[:, :4, col] = torch.tensor(prompts[p], dtype=ym.dtype).unsqueeze(0)
            if ym_base is None:
                ym_base = ym.clone()
            if lo == 0:
                diff = (ym != ym_base).nonzero().tolist()
                wired[p] = {
                    "y_meta_target_col": [float(v) for v in ym[0, :4, col]],
                    "diff_elements_vs_base": sorted({(r, c) for _, r, c in diff}),
                    "diff_fields": sorted({ROWS[r] for _, r, c in diff if r < 4}),
                    "diff_cols": sorted({c for _, r, c in diff}),
                    "x_meta_untouched": bool(torch.equal(prep["x_meta"], x_meta0)),
                    "x_meta_target_col_is_cloze": [float(v) for v in prep["x_meta"][0, :4, col]],
                }
            o = model.decode_latent(z, ym, prep.get("log_ref"))
            for j, s in enumerate(chunk):
                sl = slice(int(s), int(s) + L)
                buf[p]["mu"][sl] = o["mu"][j, :, col].float().numpy()
                buf[p]["n"][sl] = o["n"][j, :, col].float().numpy()
                buf[p]["eta"][sl] = o["eta"][j, :, col].float().numpy()
                written[sl] = True
        if lo % (40 * a.batch_windows) == 0:
            print(f"[h26] {a.track} s{a.seed} window {lo}/{len(starts)} {time.time() - t0:.0f}s", flush=True)
    assert written.all(), f"{(~written).sum()} bins never written"
    assert not any(n.startswith("B_") for n in opened), opened
    assert not any(k.startswith("B_") for k in src.ds.corpus._bios), list(src.ds.corpus._bios)
    payload = {f"{p}|{k}": v for p, d in buf.items() for k, v in d.items()}
    np.savez_compressed(out / f"{a.track}_s{a.seed}.npz", **payload)
    meta = {
        "track": a.track, "seed": a.seed, "cell": cell, "pair": str(pair), "assay": base["assay"],
        "assay_col": col, "ckpt": str(ckpt), "ckpt_md5": got_md5, "arch_json": str(arch_json),
        "depth_center": arch.get("depth_center"), "use_offset": arch.get("use_offset"),
        "chrom": CHROM, "n_bins": n_bins, "n_windows": len(starts), "context_bins": L,
        "prompts": prompts, "swapwired": wired, "train_read_lengths": rl_train,
        "missing": MISSING, "cloze": CLOZE, "wall_s": round(time.time() - t0, 1),
        "ruling_iii": {"regime_noblind": str(regime_nb), "prompt_biosample": t_bios,
                       "prompt_assays": prompt_assays, "target_in_T": bool(t_avail[col]),
                       "store_biosamples_opened": sorted(set(opened)),
                       "store_biosamples_cached": sorted(src.ds.corpus._bios)},
    }
    (out / f"{a.track}_s{a.seed}.json").write_text(json.dumps(meta, indent=1))
    src.close()
    print(f"[h26] wrote {out}/{a.track}_s{a.seed}.npz in {meta['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
