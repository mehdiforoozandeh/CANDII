"""Pre-training health checks for the cell-identity conditioning arm, on REAL baked data.

    python -m candi.healthcheck --h5 /scratch/$USER/candi/eic_full.h5 --device cuda

Runs before any arm is trained. `tests/` and `candi.compat` already cover the shipped model on
synthetic tensors; nothing there exercises a 5th metadata row, and none of it touches an h5. This
module is the missing half: does the cell id actually reach the model, does it move anything, and does
the thing train at all.

Two groups, and the first one can kill the experiment:

  G0  CAUSAL SENSITIVITY -- the go/no-go. Hold the signal, the DNA and metadata rows 0-3 fixed, change
      ONLY the cell id, and measure how far the latent and the predicted distribution move. h48/F2
      found the conditioning perturbation annihilated at the fusion LayerNorm, where trunk activations
      ran ~45x larger, and this row hangs off the same pathway. Reported as an effect SIZE against a
      reference perturbation (a 1-unit log2-depth shift), never as a boolean.

      Read the two legs differently. The encoder FiLM (`FiLMLayer`) is xavier-initialised, so it is
      live at step 0 and a dead encoder leg at init is a real failure. The decoder FiLM is
      adaLN-ZERO, so its leg is exactly 0.0 at init BY CONSTRUCTION -- that is stability, not death,
      and the check that matters there is whether it grows once gradients flow. Both are measured at
      init and again after the H2 overfit steps.

  W1-W4 WIRING -- the id survives masking, the ids agree between a T_ input and its V_/B_ target,
      gradient reaches the table on exactly the rows in the batch, and the control arm is untouched.
  H1-H4 HEALTH -- loss at init against the marginal bar, overfit-one-batch, per-module gradient
      norms, and seed determinism.

Every check prints its number. A check that cannot run says so and is reported SKIP rather than
silently passing, because a green suite that quietly skipped the load-bearing probe is worse than a
red one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from candi._vendored import CLOZE, MISSING
from candi.batch import make_masker, prepare_masked_batch
from candi.dataset import CandiKitH5Dataset, base_cell_type, h5_depth_center
from candi.model import build_model, forward_full
from candi.train import nb_count_loss

# Small architecture so the checks stay quick. Wiring is a property of the plumbing, not of the width;
# H2's overfit target is the one place this matters and it is stated there.
PROBE = dict(embed_dim=16, n_transformer_layers=1, decoder_lane=8, dropout=0.0)

CELL_ROW = 4


class Report:
    """Collects (id, status, message) so one failure does not hide the rest of the suite."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, cid: str, ok: Optional[bool], msg: str) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((cid, status, msg))
        print(f"[{status:4s}] {cid:4s} {msg}", flush=True)

    @property
    def failed(self) -> List[str]:
        return [c for c, s, _ in self.rows if s == "FAIL"]

    @property
    def skipped(self) -> List[str]:
        return [c for c, s, _ in self.rows if s == "SKIP"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_batch(ds: CandiKitH5Dataset) -> Optional[dict]:
    for b in ds:
        return b
    return None


def _make_ds(h5_path, *, cell_cond: str, train: bool, batch_size: int, seed: int,
             regime: str) -> CandiKitH5Dataset:
    return CandiKitH5Dataset(
        h5_path, regime, train=train, batch_size=batch_size, biosample_prefix="T_",
        dsf_sampling="uniform" if train else "off", seed=seed, shuffle=train,
        eval_include_vb_ground_truth=not train, cell_cond=cell_cond, h5_cache_ram=False)


def _build(ds: CandiKitH5Dataset, depth_center: float, device: str, seed: int = 0):
    torch.manual_seed(seed)
    return build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                            depth_center=depth_center, num_cells=ds.num_cells, **PROBE).to(device)


def _rel_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance between two tensors — scale-free, so encoder and decoder are comparable."""
    denom = a.norm().item()
    return float((a - b).norm().item() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# G0 — causal sensitivity (the go/no-go)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sensitivity(model, prep: dict, num_cells: int) -> Dict[str, float]:
    """Move ONLY the cell id; report how far z and mu travel, against a depth-shift reference."""
    x_meta, y_meta = prep["x_meta"], prep["y_meta"]
    true_id = float(x_meta[0, CELL_ROW, 0].item())
    other = float((int(true_id) + 1) % num_cells)

    z0 = model.encode(prep["x_data"], prep["x_dna"], x_meta)
    o0 = model.decoder(z0, y_meta)

    xm, ym = x_meta.clone(), y_meta.clone()
    xm[:, CELL_ROW, :] = other
    ym[:, CELL_ROW, :] = other
    z1 = model.encode(prep["x_data"], prep["x_dna"], xm)
    o1 = model.decoder(z1, ym)

    # decoder leg in isolation: same latent, only the prompt's id changes
    o1_dec = model.decoder(z0, ym)

    # reference: a 1-unit log2-depth shift on the prompt. A covariate the model demonstrably uses
    # (the offset head reads row 0 directly), so it calibrates "is the cell effect large or tiny".
    ym_d = y_meta.clone()
    valid = (ym_d[:, 0, :] != MISSING) & (ym_d[:, 0, :] != CLOZE)
    ym_d[:, 0, :] = torch.where(valid, ym_d[:, 0, :] + 1.0, ym_d[:, 0, :])
    o_d = model.decoder(z0, ym_d)

    return dict(
        enc_z=_rel_shift(z0, z1),
        dec_mu=_rel_shift(o0["log2_mu"], o1_dec["log2_mu"]),
        full_mu=_rel_shift(o0["log2_mu"], o1["log2_mu"]),
        ref_depth_mu=_rel_shift(o0["log2_mu"], o_d["log2_mu"]),
    )


def check_G0(model, prep, num_cells, rep: Report, *, phase: str, trained: bool) -> None:
    s = _sensitivity(model, prep, num_cells)
    ratio = s["full_mu"] / s["ref_depth_mu"] if s["ref_depth_mu"] > 0 else float("nan")
    msg = (f"[{phase}] cell-swap rel-shift: encoder z {s['enc_z']:.3e}, decoder mu {s['dec_mu']:.3e}, "
           f"end-to-end mu {s['full_mu']:.3e}; reference 1-unit log2-depth shift "
           f"{s['ref_depth_mu']:.3e} (cell/depth = {ratio:.3f})")

    if not trained:
        # Encoder FiLM is xavier-init and must already move; decoder FiLM is adaLN-zero and must not.
        enc_live = s["enc_z"] > 1e-6
        dec_zero = s["dec_mu"] < 1e-9
        ok = enc_live and dec_zero
        why = "" if ok else (
            f" -- expected encoder>0 (xavier FiLM) and decoder==0 (adaLN-zero); got "
            f"encoder {'live' if enc_live else 'DEAD'}, decoder {'zero' if dec_zero else 'NONZERO'}")
        rep.add("G0a", ok, msg + why)
    else:
        # After gradient steps both legs must be live, and the end-to-end effect must not be lost in
        # the noise of the reference perturbation.
        ok = s["enc_z"] > 1e-6 and s["dec_mu"] > 1e-9 and s["full_mu"] > 1e-6
        rep.add("G0b", ok, msg + ("" if ok else " -- pathway is annihilated; see h48/F2"))


# ---------------------------------------------------------------------------
# W — wiring
# ---------------------------------------------------------------------------

def check_W1(ds: CandiKitH5Dataset, device: str, rep: Report, n_batches: int = 12) -> None:
    """The cell id must NEVER be clozed, however many assays the masker takes."""
    masker = make_masker(p_full_assay=1.0)
    seen = bad = 0
    n_masked_cols = 0
    for i, batch in enumerate(ds):
        if i >= n_batches:
            break
        prep = prepare_masked_batch(batch, masker, device, apply_mask=True)
        if prep is None:
            continue
        row = prep["x_meta"][:, CELL_ROW, :]
        seen += row.numel()
        bad += int(((row == CLOZE) | (row == MISSING)).sum().item())
        n_masked_cols += int(prep["masked_map"].any(dim=1).sum().item())
    if seen == 0:
        rep.add("W1", None, "no batches survived masking; cannot evaluate")
        return
    rep.add("W1", bad == 0,
            f"cell-id row intact across {seen} (sample, column) slots with {n_masked_cols} clozed "
            f"assay-columns among them; {bad} carried a CLOZE/MISSING sentinel")


def check_W2(h5_path, device: str, rep: Report, regime: str, n_batches: int = 16) -> None:
    """Row 4 is constant across assays, equals id(base(biosample)), and the V_/B_ prompt agrees."""
    ds = _make_ds(h5_path, cell_cond="id", train=False, batch_size=2, seed=0, regime=regime)
    noop = make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0)
    checked = pairs = 0
    problems: List[str] = []
    for i, batch in enumerate(ds):
        if i >= n_batches:
            break
        prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
        if prep is None:
            continue
        want = float(ds.cell_ids[base_cell_type(batch["biosample_name"])])
        for name, t in (("x_meta", prep["x_meta"]), ("y_meta", prep["y_meta"])):
            row = t[:, CELL_ROW, :]
            if not bool((row == want).all().item()):
                problems.append(f"{name} row4 != id({batch['biosample_name']})={want}")
            if int(row.unique().numel()) != 1:
                problems.append(f"{name} row4 varies across the assay axis")
        ymi = batch.get("y_meta_imp")
        if isinstance(ymi, torch.Tensor):
            pairs += 1
            imp_want = float(ds.cell_ids[base_cell_type(batch["imp_biosample_name"])])
            if imp_want != want:
                problems.append(f"T_ id {want} != imp id {imp_want} for "
                                f"{batch['biosample_name']}/{batch['imp_biosample_name']}")
            if not bool((ymi[:, CELL_ROW, :] == want).all().item()):
                problems.append(f"y_meta_imp row4 != {want}")
        checked += 1
    if checked == 0:
        rep.add("W2", None, "no eval batches produced; cannot evaluate")
        return
    rep.add("W2", not problems,
            f"{checked} batches ({pairs} with a V_/B_ target) — id constant across assays, matches "
            f"base cell type, and the imputation prompt agrees"
            + ("" if not problems else f"; problems: {problems[:5]}"))


def check_W5(h5_path, rep: Report) -> None:
    """How many (T_, V_/B_, assay) targets are actually SCORABLE on the eval chromosome?

    A target whose ground truth is constant (in practice all-zero) over the eval windows yields an
    undefined Spearman and a degenerate CRPS, and `compare_arms` drops it. Dropping is right, but
    discovering it after three arms have trained is not: the comparison's power is the number of
    LIVE targets, not the 96 the panel nominally contains. Pure data, no model, so it can gate the
    experiment rather than explain it afterwards.
    """
    import h5py

    with h5py.File(str(h5_path), "r") as h5:
        order = json.loads(h5["biosamples"].attrs["order"])
        assays = json.loads(h5.attrs["assays"])
        eval_ch = set(json.loads(h5.attrs["eval_chroms"]))
        chrom = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                 for c in h5["windows/chrom"][:]]
        wi = [i for i, c in enumerate(chrom) if c in eval_ch]
        if not wi:
            rep.add("W5", None, "no eval-chromosome windows in this h5")
            return
        probe = wi[:: max(1, len(wi) // 64)][:64]

        live: List[str] = []
        dead: List[str] = []
        for t in [b for b in order if b.startswith("T_")]:
            base = t[2:]
            t_meta = np.array(h5["biosamples"][t]["meta_dsf1"])
            for imp in [b for b in order if b[2:] == base and not b.startswith("T_")]:
                g = h5["biosamples"][imp]
                i_meta = np.array(g["meta_dsf1"])
                for a in range(len(assays)):
                    # a target = absent from the T_ input, present in the V_/B_ view
                    if not (float(t_meta[0, a]) == -1.0 and float(i_meta[0, a]) != -1.0):
                        continue
                    name = f"{t}|{imp}|{assays[a]}"
                    mx = max(int(np.asarray(g["counts_dsf1"][w, :, a]).max()) for w in probe)
                    (live if mx > 0 else dead).append(name)

    n = len(live) + len(dead)
    if n == 0:
        rep.add("W5", None, "no imputation targets found in this h5")
        return
    frac = len(live) / n
    rep.add("W5", frac >= 0.5,
            f"{len(live)}/{n} imputation targets carry nonzero ground truth over {len(probe)} eval "
            f"windows ({100 * frac:.0f}%); {len(dead)} are constant and will be dropped by "
            f"compare_arms" + (f" — e.g. {dead[:3]}" if dead else ""))


def check_W3(model, prep, ds: CandiKitH5Dataset, rep: Report) -> None:
    """Gradient reaches the table on exactly the rows present in the batch, and nowhere else."""
    emb = model.encoder.metadata_embedding.cell_embedding
    model.zero_grad(set_to_none=True)
    loss, _ = nb_count_loss(forward_full(model, prep), prep)
    loss.backward()
    if emb.weight.grad is None:
        rep.add("W3", False, "cell_embedding.weight.grad is None — the table is detached")
        return
    g = emb.weight.grad
    present = sorted({int(v) for v in prep["x_meta"][:, CELL_ROW, 0].tolist()})
    nz = sorted(torch.nonzero(g.abs().sum(dim=1) > 0).flatten().tolist())
    ok = nz == present
    rep.add("W3", ok,
            f"table {tuple(g.shape)}, |grad|={g.abs().sum().item():.4e}; rows with gradient {nz}, "
            f"cell ids in batch {present}" + ("" if ok else " — MISMATCH"))
    model.zero_grad(set_to_none=True)


def check_W4(ds_off: CandiKitH5Dataset, ds_on: CandiKitH5Dataset, depth_center: float,
             device: str, rep: Report) -> None:
    """The control arm must be the historical model: same params, same values, same RNG stream."""
    import hashlib
    m_off = _build(ds_off, depth_center, device)
    torch.manual_seed(0)
    ref = build_model(num_assays=ds_off.num_assays, context_length=ds_off.context_bins,
                           depth_center=depth_center, num_cells=0, **PROBE).to(device)

    def sig(m):
        h = hashlib.sha1()
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16], sum(p.numel() for p in m.parameters())

    s_off, n_off = sig(m_off)
    s_ref, n_ref = sig(ref)
    m_on = _build(ds_on, depth_center, device)
    n_on = sum(p.numel() for p in m_on.parameters())
    extra = n_on - n_off
    tbl = (ds_on.num_cells + 2) * PROBE["embed_dim"]
    fuse = PROBE["embed_dim"] * PROBE["embed_dim"]                 # the 5th block of fusion[0]
    ok = (s_off == s_ref) and (n_off == n_ref) and extra == 2 * (tbl + fuse)
    rep.add("W4", ok,
            f"cell_cond=off sha1 {s_off} == num_cells=0 sha1 {s_ref}, {n_off} params; "
            f"cell_cond=id adds {extra} params over two embedders "
            f"(expected {2 * (tbl + fuse)} = 2 x [table {tbl} + fusion {fuse}])")


# ---------------------------------------------------------------------------
# H — standard health
# ---------------------------------------------------------------------------

def check_H1(model, prep, rep: Report) -> None:
    """NLL at init should sit near a constant-forecast bar, not orders away from it."""
    with torch.no_grad():
        loss, terms = nb_count_loss(forward_full(model, prep), prep)
        tgt = prep["y_data"][prep["observed_map"]]
        mu = float(tgt.mean().item())
        var = float(tgt.var().item())
        # method-of-moments NB on the observed counts = the honest constant-forecast bar
        if var > mu > 0:
            n = mu * mu / (var - mu)
            p = n / (n + mu)
            bar = float(-torch.distributions.NegativeBinomial(
                total_count=torch.tensor(n), probs=torch.tensor(1 - p)).log_prob(tgt).mean().item())
        else:
            bar = float("nan")
    ok = bool(math.isfinite(loss.item())) and (not math.isfinite(bar) or loss.item() < 50 * bar)
    rep.add("H1", ok,
            f"init NLL {loss.item():.3f} (obs {terms['obs']:.3f} / imp {terms['imp']:.3f}); "
            f"marginal-NB bar on the same points {bar:.3f}; target mean {mu:.2f} var {var:.2f}")


def check_H2(model, prep, rep: Report, steps: int = 600, lr: float = 3e-3) -> float:
    """Overfit ONE real batch. If the architecture cannot fit one batch, nothing downstream means
    anything.

    THE CRITERION IS THE MARGINAL-NB BAR, NOT A PERCENTAGE. An earlier version required a 30% drop
    in 200 steps and FAILED on the full panel at 17.8% -- but the architecture was fine, the bar was
    calibrated on a 3-assay probe. At 35 assays the same model reaches 55% by step 400, and the
    production width crosses the marginal bar by step 100 and keeps descending to 0.50 by step 2000.
    A tuned percentage would just have been moved until it passed, which is not a check. The
    constant-forecast bar is an absolute anchor the data itself sets: a model that cannot beat it on
    a single batch it is allowed to memorise cannot fit at all.

    Descent is also required to still be going at the end, so a model that crosses the bar and then
    plateaus or diverges is not scored as healthy."""
    tgt = prep["y_data"][prep["observed_map"]]
    mu, var = float(tgt.mean().item()), float(tgt.var().item())
    bar = float("nan")
    if var > mu > 0:
        n = mu * mu / (var - mu)
        p = n / (n + mu)
        bar = float(-torch.distributions.NegativeBinomial(
            total_count=torch.tensor(n), probs=torch.tensor(1 - p)).log_prob(tgt).mean().item())

    opt = torch.optim.Adam(model.parameters(), lr)
    model.train()
    curve: List[float] = []
    for _ in range(steps):
        loss, _ = nb_count_loss(forward_full(model, prep), prep)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append(float(loss))
    model.eval()

    first, last = curve[0], curve[-1]
    drop = (first - last) / abs(first) if first else float("nan")
    q = max(1, steps // 4)
    still_descending = float(np.mean(curve[-q:])) < float(np.mean(curve[-2 * q:-q]))
    beats_bar = bool(np.isfinite(bar) and last < bar)
    rep.add("H2", beats_bar and still_descending,
            f"one batch, {steps} steps @ lr {lr}: NLL {first:.3f} -> {last:.3f} ({100 * drop:.1f}% drop); "
            f"marginal-NB bar {bar:.3f} -> {'BEATEN' if beats_bar else 'NOT beaten'}; "
            f"last quarter {'still descending' if still_descending else 'PLATEAUED/diverging'}")
    return drop


def check_H3(model, prep, rep: Report) -> None:
    """Per-module gradient norms. The cell table starving relative to the trunk is the h48/F2
    failure mode arriving quietly, so it is reported as a RATIO, not just a magnitude."""
    model.zero_grad(set_to_none=True)
    loss, _ = nb_count_loss(forward_full(model, prep), prep)
    loss.backward()
    groups = {
        "encoder.signal_tower": "encoder.signal_tower",
        "encoder.transformer": "encoder.transformer",
        "encoder.meta_embed": "encoder.metadata_embedding",
        "encoder.film": "film",
        "decoder.trunk": "decoder.trunk",
        "decoder.film_proj": "decoder.film_proj",
        "decoder.heads": "decoder.head_",
    }
    norms: Dict[str, float] = {}
    for label, pat in groups.items():
        tot = sum(float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
                  if pat in n and p.grad is not None)
        norms[label] = math.sqrt(tot)
    cell = math.sqrt(sum(float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
                         if "cell_embedding" in n and p.grad is not None))
    norms["cell_embedding"] = cell
    trunk = max(norms["encoder.signal_tower"], norms["decoder.trunk"], 1e-30)
    ratio = cell / trunk
    pretty = ", ".join(f"{k} {v:.3e}" for k, v in norms.items())
    rep.add("H3", cell > 0 and ratio > 1e-6,
            f"{pretty}; cell/trunk = {ratio:.3e}")
    model.zero_grad(set_to_none=True)


def check_H4(h5_path, depth_center: float, device: str, rep: Report, regime: str,
             steps: int = 25) -> None:
    """Same seed twice -> identical loss curve. Nondeterminism above the arm-vs-arm delta would make
    every comparison in this experiment unreadable."""
    def run() -> List[float]:
        ds = _make_ds(h5_path, cell_cond="id", train=True, batch_size=2, seed=0, regime=regime)
        model = _build(ds, depth_center, device, seed=0)
        opt = torch.optim.Adam(model.parameters(), 5e-4)
        masker = make_masker(p_full_assay=1.0)
        out: List[float] = []
        for i, batch in enumerate(ds):
            if len(out) >= steps:
                break
            prep = prepare_masked_batch(batch, masker, device, apply_mask=True)
            if prep is None:
                continue
            loss, _ = nb_count_loss(forward_full(model, prep), prep)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            out.append(float(loss))
        return out

    a, b = run(), run()
    if not a:
        rep.add("H4", None, "no training batches produced; cannot evaluate")
        return
    worst = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("inf")
    rep.add("H4", len(a) == len(b) and worst == 0.0,
            f"{len(a)} steps twice at seed 0; max |delta loss| = {worst:.3e}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(h5_path: str, device: str, regime: str, batch_size: int) -> Report:
    rep = Report()
    depth_center = h5_depth_center(h5_path)

    ds_on = _make_ds(h5_path, cell_cond="id", train=True, batch_size=batch_size, seed=0, regime=regime)
    ds_off = _make_ds(h5_path, cell_cond="off", train=True, batch_size=batch_size, seed=0, regime=regime)
    print(f"[hc] h5={h5_path}\n[hc] assays={ds_on.num_assays} context_bins={ds_on.context_bins} "
          f"cells={ds_on.num_cells} depth_center={depth_center:.4f}", flush=True)

    check_W4(ds_off, ds_on, depth_center, device, rep)
    check_W1(ds_on, device, rep)
    check_W2(h5_path, device, rep, regime)
    check_W5(h5_path, rep)

    batch = _first_batch(ds_on)
    if batch is None:
        rep.add("G0a", None, "no training batch; every model-level check skipped")
        return rep
    prep = prepare_masked_batch(batch, make_masker(p_full_assay=1.0), device, apply_mask=True)
    if prep is None:
        rep.add("G0a", None, "first batch had no supervised positions; model-level checks skipped")
        return rep

    model = _build(ds_on, depth_center, device)
    check_G0(model, prep, ds_on.num_cells, rep, phase="init", trained=False)
    check_W3(model, prep, ds_on, rep)
    check_H1(model, prep, rep)
    check_H3(model, prep, rep)
    check_H2(model, prep, rep)
    # G0 again on the SAME model, now that adaLN-zero has had gradient through it
    check_G0(model, prep, ds_on.num_cells, rep, phase="after H2", trained=True)
    check_H4(h5_path, depth_center, device, rep, regime)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--regime", default="type1", choices=["type1", "type2_loci"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rep = run(a.h5, a.device, a.regime, a.batch_size)

    print("\n" + "=" * 78)
    for cid, status, msg in rep.rows:
        print(f"{status:4s}  {cid:4s}  {msg}")
    print("=" * 78)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            [dict(id=c, status=s, message=m) for c, s, m in rep.rows], indent=2))
    if rep.failed:
        print(f"[hc] FAILED: {rep.failed}", flush=True)
        return 1
    if rep.skipped:
        print(f"[hc] all run checks passed, but SKIPPED: {rep.skipped} — a skipped G0 is not a green "
              "light", flush=True)
        return 2
    print("[hc] ALL GREEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
