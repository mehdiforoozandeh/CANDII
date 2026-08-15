#!/usr/bin/env python3
"""Bit-exactness gate for CANDII.

    python tools/golden.py save <ref.pt>     record outputs + digests of the current tree
    python tools/golden.py check <ref.pt>    rebuild and assert 0 ULP against the recording

Every stage of the build must clear this before the next one starts. The model is constructed from
one seed on fixed inputs, so any refactor that only moves code must reproduce the recording exactly;
a nonzero diff, a changed parameter count or a changed state_dict key is a failure, never noise.

The builder is resolved at run time so the same file works before and after the module reshuffle:
it prefers `candi.model.build_model` (the stripped tree) and falls back to `candi.model2.build_arm`
with arch="a1" (the tree as copied from candi_kit2).
"""
import hashlib
import sys

import torch

A, L, RES, B = 35, 768, 25, 2

# a1's recorded config, read off ck2_a1_seed0.json — the run these numbers must stay comparable to.
CFG = dict(embed_dim=32, dropout=0.1, depth_center=24.340200424194336, use_offset=True,
           num_assays=A, context_length=L, d_model=288, nhead=4, num_cells=0,
           meta_embed_layernorm=True, meta_gain=1.0, decoder_lane=8)


def build():
    torch.manual_seed(0)
    try:
        from candi.model import build_model
        return build_model(**CFG).eval()
    except ImportError:
        from candi.model2 import build_arm
        return build_arm("a1", lane_width=32, dna_dim=128, **CFG).eval()


def make_inputs():
    torch.manual_seed(1234)
    x_data = torch.rand(B, L, A + 1) * 5.0
    x_meta = torch.zeros(B, 4, A + 1)
    x_meta[:, 0, :] = 26.0
    x_meta[:, 1, :] = torch.arange(A + 1).float()
    x_meta[:, 2, :] = 100.0
    x_meta[:, 3, :] = 1.0
    y_meta = x_meta[:, :, :A].clone()
    x_dna = torch.zeros(B, 4, L * RES)
    x_dna[torch.arange(B).view(B, 1, 1), torch.randint(0, 4, (B, 1, L * RES)),
          torch.arange(L * RES).view(1, 1, -1)] = 1.0
    for a in (3, 11, 27):                      # exercise the mask-token and sentinel branches
        x_data[:, :, a] = -2.0
        x_meta[:, 0, a] = -2.0
    return x_data, x_dna, x_meta, y_meta


def digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def snapshot():
    model = build()
    with torch.no_grad():
        out = model(*make_inputs())
    rec = dict(
        n_params=sum(p.numel() for p in model.parameters()),
        sd_sha=digest(model),
        keys=sorted(model.state_dict()),
        out={k: v.clone() for k, v in out.items() if torch.is_tensor(v)},
    )
    print(f"  params={rec['n_params']:,}  sd={rec['sd_sha'][:12]}  tensors={sorted(rec['out'])}")
    return rec


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "save":
        print("=== recording ===")
        torch.save(snapshot(), path)
        print(f"saved -> {path}")
        raise SystemExit(0)

    ref = torch.load(path, weights_only=False)
    print("=== rebuilding ===")
    now = snapshot()
    probs = []
    if ref["n_params"] != now["n_params"]:
        probs.append(f"params {ref['n_params']:,} -> {now['n_params']:,}")
    if ref["keys"] != now["keys"]:
        gone = sorted(set(ref["keys"]) - set(now["keys"]))
        new = sorted(set(now["keys"]) - set(ref["keys"]))
        probs.append(f"state_dict keys changed (-{len(gone)} +{len(new)}): {(gone + new)[:6]}")
    if ref["sd_sha"] != now["sd_sha"] and not probs:
        probs.append("state_dict VALUES changed — the RNG stream moved")
    worst = 0.0
    for k in sorted(ref["out"]):
        if k not in now["out"]:
            probs.append(f"output {k!r} disappeared")
            continue
        d = (ref["out"][k] - now["out"][k]).abs().max().item()
        worst = max(worst, d)
        if d != 0.0:
            probs.append(f"{k} max|diff|={d:.3e}")
    print(f"\nworst_out_diff={worst:.1e}")
    if probs:
        for p in probs:
            print(f"  FAIL  {p}")
        raise SystemExit(1)
    print("  OK  0 ULP — the tree is bit-identical to the recording")
