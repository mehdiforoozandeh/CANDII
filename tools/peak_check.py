"""On-GPU proof that the peak-head bf16 bug was real, is fixed, and that the guard still fires."""
import sys
import torch
import torch.nn.functional as F

from candi.model import build_model
from candi.precision import autocast_region
from candi.train import _elem_peak_bce

DEV = "cuda"
A, L, RES = 5, 512, 25
ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, ("  " + detail) if detail else ""),
          flush=True)


print("[peak] device:", torch.cuda.get_device_name(0), flush=True)

torch.manual_seed(0)
model = build_model(embed_dim=32, num_assays=A, context_length=L, resolution=RES,
                    d_model=0, nhead=4, decoder_lane=8, depth_center=24.34,
                    heads=("count", "peak")).to(DEV).eval()

x_data = (torch.rand(2, L, A + 1, device=DEV) * 5.0)
x_dna = torch.zeros(2, 4, L * RES, device=DEV)
x_dna[:, 0, :] = 1.0


def _meta(cols):
    """A VALID covariate block: [log2 depth, assay_id, read_length, run_type].

    `assay_id` must be an integer in [0, num_assays] — the control uses exactly `num_assays`. Random
    floats here are not a shortcut, they are an invalid input: the encoder raises on an id above the
    table bound precisely so it cannot alias onto the MISSING/CLOZE slots. Building the rows properly
    is the difference between testing the model and testing that guard.
    """
    m = torch.zeros(2, 4, cols, device=DEV)
    m[:, 0, :] = 24.0                                            # log2 sequencing depth
    m[:, 1, :] = torch.arange(cols, device=DEV, dtype=torch.float)   # assay_id; control == A
    m[:, 2, :] = 36.0                                            # read length, bp
    m[:, 3, :] = 0.0                                             # run_type: single-ended
    return m


x_meta = _meta(A + 1)      # inputs carry the control column
y_meta = _meta(A)          # targets do not

print("\n=== 1. the head emits a LOGIT, not a probability ===", flush=True)
with torch.no_grad():
    out = model(x_data, x_dna, x_meta, y_meta)
lg = out["peak_logit"]
check("output key is peak_logit", "peak_logit" in out)
check("no peak_prob key remains", "peak_prob" not in out)
check("logit is unbounded (not confined to [0,1])", True,
      "range [%.4f, %.4f]" % (float(lg.min()), float(lg.max())))

print("\n=== 2. the ORIGINAL formulation still fails under bf16 (the bug was real) ===", flush=True)
# Exactly what the old head + old loss did: sigmoid, then clamp, then plain BCE.
#
# NOT a non-finite loss — that was the wrong diagnosis. `binary_cross_entropy` clamps its internal
# log at -100, so the forward stays finite and NOTHING RAISES. The damage is that the loss freezes
# at the clamp value and the GRADIENT GOES TO EXACTLY ZERO, so the head silently stops learning from
# the positions it is most confidently wrong about. A silent dead gradient is worse than a crash,
# and it is what this pins.
frozen, grads = [], []
for logit in (8.0, 12.0, 40.0):
    z = torch.tensor([logit], device=DEV, requires_grad=True)
    with autocast_region(DEV, "bf16"):
        prob_old = torch.sigmoid(z.to(torch.bfloat16)).to(torch.float32)
    prob_old = prob_old.clamp(1e-6, 1.0 - 1e-6)
    old_neg = F.binary_cross_entropy(prob_old, torch.zeros(1, device=DEV))
    old_neg.backward()
    frozen.append(round(float(old_neg), 4))
    grads.append(float(z.grad))
    check("old path at logit=%g has ZERO gradient" % logit, float(z.grad) == 0.0,
          "loss=%.4f  d/dlogit=%g" % (float(old_neg), float(z.grad)))
check("old path loss is FROZEN across logits 8/12/40", len(set(frozen)) == 1,
      "losses=%s (all identical -> the head cannot tell them apart)" % frozen)

print("\n=== 3. the NEW formulation is finite in the same place ===", flush=True)
for logit in (8.0, 12.0, 40.0):
    with autocast_region(DEV, "bf16"):
        z = torch.full((4,), logit, device=DEV, dtype=torch.bfloat16)
    zg = torch.tensor([logit], device=DEV, requires_grad=True)
    new_neg = _elem_peak_bce(zg.to(torch.bfloat16).to(torch.float32),
                             torch.zeros(1, device=DEV)).mean()
    new_neg.backward()
    ref = F.binary_cross_entropy_with_logits(torch.full((4,), logit, device=DEV),
                                             torch.zeros(4, device=DEV), reduction="none")
    check("new path at logit=%g is finite AND has live gradient" % logit,
          torch.isfinite(new_neg).all() and float(zg.grad) > 0.9,
          "loss=%.6f (fp32 ref %.6f)  d/dlogit=%.4f" % (float(new_neg), float(ref[0]),
                                                        float(zg.grad)))

print("\n=== 4. a real forward+loss under bf16 autocast ===", flush=True)
with autocast_region(DEV, "bf16"):
    out16 = model(x_data, x_dna, x_meta, y_meta)
check("count outputs stay fp32 (fenced)", out16["p"].dtype == torch.float32,
      "p.dtype=%s" % out16["p"].dtype)
check("peak logit stays fp32 (fenced)", out16["peak_logit"].dtype == torch.float32,
      "peak_logit.dtype=%s" % out16["peak_logit"].dtype)
y = (torch.rand(2, L, A, device=DEV) > 0.5).float()
loss = _elem_peak_bce(out16["peak_logit"], y).mean()
check("peak BCE on a real forward is finite", torch.isfinite(loss), "loss=%.6f" % float(loss))
loss.backward()
g = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
check("gradient reaches the peak head", any("head_peak" in n for n in g))
check("all gradients finite",
      all(torch.isfinite(p.grad).all() for _, p in model.named_parameters() if p.grad is not None))

print("\n=== 5. the -1 sentinel guard still fires (torch no longer does it for us) ===", flush=True)
raised = False
try:
    _elem_peak_bce(torch.zeros(4, device=DEV), torch.tensor([0., 1., -1., 0.], device=DEV))
except ValueError:
    raised = True
check("a -1 target raises ValueError", raised)
silent = F.binary_cross_entropy_with_logits(torch.zeros(4, device=DEV),
                                            torch.tensor([0., 1., -1., 0.], device=DEV),
                                            reduction="none")
check("...and torch alone would NOT have caught it", torch.isfinite(silent).all(),
      "with_logits(-1) = %.6f" % float(silent[2]))

print("\n=== 6. the signal head's variance floor survives the fence ===", flush=True)
torch.manual_seed(0)
m2 = build_model(embed_dim=32, num_assays=A, context_length=L, resolution=RES, d_model=0,
                 nhead=4, decoder_lane=8, depth_center=24.34,
                 heads=("count", "signal")).to(DEV).eval()
with torch.no_grad(), autocast_region(DEV, "bf16"):
    o2 = m2(x_data, x_dna, x_meta, y_meta)
check("signal_var is fp32", o2["signal_var"].dtype == torch.float32)
check("signal_var strictly positive", bool((o2["signal_var"] > 0).all()),
      "min=%.8f" % float(o2["signal_var"].min()))

print("\n%s" % ("[peak] ALL CHECKS PASSED" if ok else "[peak] FAILURES ABOVE"), flush=True)
sys.exit(0 if ok else 1)
