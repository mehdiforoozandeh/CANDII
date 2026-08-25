"""Prove the port reproduces a real Keras checkpoint, numerically. The §7.4 parity gate.

`tests/test_lavawizard.py` checks the loader against an `.h5` this repo builds itself. That proves
the mapping is self-consistent; it cannot prove it matches what Keras 2.2.4 *does* with the same
file. Only Keras can say that, so this script is in two halves that never run in the same process:

```bash
# on Fir. Half A — in the period-correct TF 1.15 env (its ONLY sanctioned use besides debugging):
micromamba run -n lw_period python parity_keras.py reference \
    --h5 smoke_out/model_v4_guacamole6_chr21.h5 --out ref_chr21.npz --n 4096

# Half B — in a modern torch env, no TensorFlow anywhere:
source torch_env/bin/activate
python parity_keras.py compare --h5 smoke_out/model_v4_guacamole6_chr21.h5 --ref ref_chr21.npz
```

Half A writes the inputs it used and the outputs Keras produced. Half B loads the same file
through `keras_weights.load_keras_h5`, runs the same inputs, and reports the worst absolute and
relative disagreement. Nothing is asserted about a tolerance here: the PI asked for the observed
tolerance to be *documented on one chromosome before being rolled out*, so this prints and
`--max-abs` opts in to a hard gate once that number is known.

**Batch norm is why this script exists.** Their statistics were fitted under Keras's `epsilon=1e-3`
and are stored as `moving_variance`. Torch defaults to `1e-5` and calls it `running_var`. Both
mappings are shape-legal and only one is right, and no synthetic fixture can tell them apart —
running the real graph can.

Note the deliberate asymmetry with the true gate: this compares us against *Keras on their
weights*. The gate in §7.4 compares us against *their submitted bigwig tracks*, which adds their
preprocessing and bigwig quantisation on top. This script is the half of that difference we can
settle before the Synapse pull lands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

__all__ = ["make_reference", "compare"]

#: The five factor widths are per-chromosome upstream (`Lavawizard_pipeline.sh`). chr21's row.
CHR21_FACTORS = dict(n_25bp_factors=25, n_250bp_factors=30, n_5kbp_factors=60)


def make_reference(h5_path: Path, out: Path, n: int, seed: int) -> None:
    """Half A. Run their checkpoint under Keras and record `(inputs, outputs)`. Needs TF 1.15."""
    import h5py
    from keras.models import load_model

    with h5py.File(str(h5_path), "r") as f:
        emb = f["model_weights"]["genome_25bp_embedding"]["genome_25bp_embedding/embeddings:0"]
        n_positions = int(emb.shape[0])
        n_celltypes = int(f["model_weights"]["celltype_embedding"]
                          ["celltype_embedding/embeddings:0"].shape[0])
        n_assays = int(f["model_weights"]["assay_embedding"]
                       ["assay_embedding/embeddings:0"].shape[0])
    print(f"{h5_path.name}: {n_positions} positions, {n_celltypes} cell types, {n_assays} assays")

    rng = np.random.default_rng(seed)
    d = {
        "celltype_input": rng.integers(0, n_celltypes, n).astype("int32"),
        "assay_input": rng.integers(0, n_assays, n).astype("int32"),
        "genome_25bp_input": rng.integers(0, n_positions, n).astype("int32"),
        "average_value_input": rng.uniform(0, 4, n).astype("float32"),
        "variance_value_input": rng.uniform(0, 2, n).astype("float32"),
    }
    d["genome_250bp_input"] = (d["genome_25bp_input"] // 10).astype("int32")
    d["genome_5kbp_input"] = (d["genome_25bp_input"] // 200).astype("int32")

    model = load_model(str(h5_path))
    y = model.predict(d, batch_size=1024, verbose=1).squeeze()
    np.savez_compressed(out, y=np.asarray(y, dtype=np.float64),
                        n_positions=n_positions, n_celltypes=n_celltypes, n_assays=n_assays,
                        **{k: v for k, v in d.items()})
    print(f"wrote {out}  y[{y.shape}]  mean={y.mean():.6f}  min={y.min():.6f}  max={y.max():.6f}")


def compare(h5_path: Path, ref_path: Path, max_abs: float | None) -> int:
    """Half B. Load the same file into the port and diff against Keras's answer. No TensorFlow."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lavawizard.keras_weights import load_keras_h5
    from lavawizard.model import Guacamole

    ref = np.load(ref_path)
    model = Guacamole(n_celltypes=int(ref["n_celltypes"]), n_assays=int(ref["n_assays"]),
                      n_positions=int(ref["n_positions"]), **CHR21_FACTORS)
    report = load_keras_h5(model, h5_path)
    print(f"loaded {report['layers']} layers / {report['tensors']} tensors / "
          f"{report['parameters']:,} parameters")

    with torch.no_grad():
        got = model(
            celltype=torch.from_numpy(ref["celltype_input"].astype("int64")),
            assay=torch.from_numpy(ref["assay_input"].astype("int64")),
            pos25=torch.from_numpy(ref["genome_25bp_input"].astype("int64")),
            average=torch.from_numpy(ref["average_value_input"].astype("float32")),
            variance=torch.from_numpy(ref["variance_value_input"].astype("float32")),
        ).double().numpy()

    want = ref["y"]
    abs_err = np.abs(got - want)
    scale = np.maximum(np.abs(want), 1e-9)
    rel_err = abs_err / scale
    print(f"n = {want.size}")
    print(f"keras  mean {want.mean(): .8f}  min {want.min(): .8f}  max {want.max(): .8f}")
    print(f"torch  mean {got.mean(): .8f}  min {got.min(): .8f}  max {got.max(): .8f}")
    print(f"abs err  max {abs_err.max():.3e}  mean {abs_err.mean():.3e}  "
          f"p99 {np.percentile(abs_err, 99):.3e}")
    print(f"rel err  max {rel_err.max():.3e}  mean {rel_err.mean():.3e}")
    corr = float(np.corrcoef(got, want)[0, 1])
    print(f"pearson r = {corr:.12f}")

    if max_abs is None:
        print("\nno --max-abs given: reporting only. Record this number, then gate on it.")
        return 0
    ok = abs_err.max() <= max_abs
    print(f"\nGATE max_abs <= {max_abs:g}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("reference", help="half A — run Keras, dump inputs and outputs (needs TF)")
    a.add_argument("--h5", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--n", type=int, default=4096)
    a.add_argument("--seed", type=int, default=20260825)

    b = sub.add_parser("compare", help="half B — load into the port and diff (no TF)")
    b.add_argument("--h5", type=Path, required=True)
    b.add_argument("--ref", type=Path, required=True)
    b.add_argument("--max-abs", type=float, default=None,
                   help="fail if the worst absolute disagreement exceeds this")

    ns = p.parse_args(argv)
    if ns.cmd == "reference":
        make_reference(ns.h5, ns.out, ns.n, ns.seed)
        return 0
    return compare(ns.h5, ns.ref, ns.max_abs)


if __name__ == "__main__":
    raise SystemExit(main())
