"""The 31 stage-2 arm configs for any rung or hybrid (chunk S2K; plan table "Stage-2 arms").

    python tools/minicandi/stage2_arms.py write --incumbent <settled config json> --out <dir>
        [--average-h5 P] [--qc-table P] [--flashzoi-emb P] [--gnlm-emb P]
        [--dna-embedding-dim 128]

Each arm is one change from the incumbent (the design that the ladder, the mix-and-match step and
the evolution step give), with `simpler_than_base` set from `SIMPLER` (null when not listed).
`write` puts `<arm>.json` in `--out` for every arm that applies, and `arms.tsv` with one row per
arm (`ARMS_TSV_COLUMNS`). `two_step_fixed` is compared with `two_step_native`; every other arm
with the incumbent.

An arm gets no json, and its `status` column gives the reason, when (1) building it through
`registry.build_model` raises `NotApplicable` (e.g. a two-step arm on an incumbent with
`qc_on_input`, or `qc_on_input` on a two-step incumbent: the pair is exclusive), (2) it changes
nothing in the incumbent (the incumbent already holds the arm's values), or (3) its key does not
act on the incumbent (`offset_eps` acts only with `count_transform = offset`).

Input files. `needs` names the file an arm reads: `average_h5` (average_input), `qc_table`
(qc_on_input), `dna_embedding_h5` (dna_flashzoi, dna_gnlm). A path given on the command line is
written into the arm's json; without it the json lacks the key, and the submitter must add it
(the dataset computes the average on the fly and reads the QC columns of `measurements.tsv`
without one; a precomputed-DNA arm cannot run without its embedding).

`joins_design` says whether a winning arm may enter the final design. It is `no` for the leakage
controls `dna_shuffled` and `dna_shift5mb` (they settle with `--simpler incumbent`), for
`control_none` (PI ruling 14: the EIC model keeps a control output), and for `dna_flashzoi` (PI
ruling 26 outcome: Flashzoi's training loci overlap the EIC tracks, so it is an ablation only,
judged on its held-out test fold, fold 3, loci only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

#: dna_embedding_dim after PCA (chunk A5)
DNA_EMBEDDING_DIM = 128
#: the covariate-dropout probability of the `cov_dropout` arm (agent call; not pinned by the plan)
COV_DROPOUT = 0.1

#: ARMS[arm] = (group, {key: value}): the change of each arm from the incumbent, in table order
ARMS = {
    "qc_on_input": ("covariate vector (D6)", {"qc_on_input": True}),
    "two_step_native": ("correction route (D15)", {"two_step": True, "base_depth": "native"}),
    "two_step_fixed": ("correction route (D15)", {"two_step": True, "base_depth": "fixed"}),
    "control_output_only": ("ChIP control", {"control_input": False}),
    "control_none": ("ChIP control", {"control_input": False, "predict_control": False}),
    "control_lambda_head": ("ChIP control", {"control_input": False,
                                             "control_head": "lambda_smooth"}),
    "average_input": ("per-position average (D10)", {"average_input": True}),
    "dna_flashzoi": ("DNA source", {"dna": "precomputed"}),
    "dna_gnlm": ("DNA source", {"dna": "precomputed"}),
    "dna_shuffled": ("DNA source", {"dna": "shuffled"}),
    "dna_shift5mb": ("DNA source", {"dna_shift_bp": 5_000_000}),
    "inject_adaln_zero": ("covariate embedding and injection",
                          {"covariate_injection": "adaln_zero"}),
    "cov_dropout": ("covariate embedding and injection", {"covariate_dropout": COV_DROPOUT}),
    "cov_numeric_only": ("covariate embedding and injection",
                         {"covariate_embedding": "numeric_only"}),
    "input_depth_fixed": ("depth-controlled input", {"input_depth": "fixed"}),
    "tf_log1p": ("count transform", {"count_transform": "log1p"}),
    "tf_asinh": ("count transform", {"count_transform": "asinh"}),
    "tf_anscombe": ("count transform", {"count_transform": "anscombe"}),
    "tf_pearson": ("count transform", {"count_transform": "pearson"}),
    "tf_local_ratio": ("count transform", {"count_transform": "local_ratio"}),
    "tf_quantile": ("count transform", {"count_transform": "quantile"}),
    "tf_raw": ("count transform", {"count_transform": "raw"}),
    "tf_soft_clip": ("count transform", {"count_transform": "soft_clip"}),
    "offset_eps0.1": ("count transform", {"offset_eps": 0.1}),
    "loss_poisson_mean": ("loss and head", {"loss": "poisson_mean_detached_size"}),
    "loss_nb_crps": ("loss and head", {"loss": "nb_crps"}),
    "loss_impute_x4": ("loss and head", {"loss_weights": {"impute": 4, "correct": 1, "same": 1,
                                                          "base": 1}}),
    "head_shared_size": ("loss and head", {"head": "nb_shared_size"}),
    "mask_dirichlet": ("masking", {"mask_scheme": "dirichlet"}),
    "mask_panel": ("masking", {"mask_scheme": "panel"}),
    "bin_mask_span": ("masking", {"bin_mask_shape": "span"}),
}
_TRUE = ("control_none", "control_output_only", "cov_numeric_only", "head_shared_size", "tf_raw")
_FALSE = ("qc_on_input", "two_step_native", "two_step_fixed", "average_input", "dna_flashzoi",
          "dna_gnlm", "control_lambda_head", "inject_adaln_zero")
#: simpler_than_base per arm: True, False or None
SIMPLER = {a: (True if a in _TRUE else False if a in _FALSE else None) for a in ARMS}
#: D14 condition 1 arms (the listed comparisons of D6 and D15)
D14 = ("qc_on_input", "two_step_native", "two_step_fixed")
#: the arm each arm is compared with (default: the incumbent)
COMPARISON = {"two_step_fixed": "two_step_native"}
#: arm -> (the input key, the CLI option that fills it)
NEEDS = {"qc_on_input": "qc_table", "average_input": "average_h5",
         "dna_flashzoi": "dna_embedding_h5", "dna_gnlm": "dna_embedding_h5"}
#: arms that never join a design, with the reason
NEVER_IN_DESIGN = {
    "control_none": "PI ruling 14: the EIC model keeps a control output",
    "dna_flashzoi": ("PI ruling 26: ablation only (Flashzoi training loci overlap EIC); "
                     "judged on fold-3 loci only"),
    "dna_shuffled": "leakage control (--simpler incumbent)",
    "dna_shift5mb": "leakage control (--simpler incumbent)",
}
ARMS_TSV_COLUMNS = ("arm", "keys changed", "comparison incumbent", "d14_condition_1", "needs",
                    "group", "simpler_than_base", "joins_design", "status")
#: arms whose key acts only under an incumbent condition: arm -> (predicate, reason)
_ACTS_ONLY = {"offset_eps0.1": (lambda inc: inc.get("count_transform", "offset") == "offset",
                                "offset_eps acts only with count_transform = offset")}


def _json(v) -> str:
    return json.dumps(v, sort_keys=True)


def _cell(v) -> str:
    """A value as a tsv cell: a string as is, anything else as compact json."""
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, separators=(",", ":"))


def arm_change(arm: str, paths: dict | None = None,
               dna_embedding_dim: int = DNA_EMBEDDING_DIM) -> dict:
    """The declared keys of `arm` (simpler_than_base excluded), with its input path when given.
    `paths` maps `average_h5`, `qc_table`, `flashzoi_emb`, `gnlm_emb` to paths."""
    paths = paths or {}
    change = json.loads(json.dumps(ARMS[arm][1]))
    if arm in ("dna_flashzoi", "dna_gnlm"):
        change["dna_embedding_dim"] = int(dna_embedding_dim)
        p = paths.get("flashzoi_emb" if arm == "dna_flashzoi" else "gnlm_emb")
        if p:
            change["dna_embedding_h5"] = str(p)
    elif arm == "average_input" and paths.get("average_h5"):
        change["average_h5"] = str(paths["average_h5"])
    elif arm == "qc_on_input" and paths.get("qc_table"):
        change["qc_table"] = str(paths["qc_table"])
    return change


def arm_config(incumbent: dict, arm: str, paths: dict | None = None,
               dna_embedding_dim: int = DNA_EMBEDDING_DIM) -> dict:
    """The incumbent with the arm's change and its `simpler_than_base`."""
    return {**incumbent, **arm_change(arm, paths, dna_embedding_dim),
            "simpler_than_base": SIMPLER[arm]}


def changed_keys(incumbent: dict, cfg: dict) -> list[str]:
    """Keys whose value differs from the incumbent (simpler_than_base excluded)."""
    return sorted(k for k in cfg if k != "simpler_than_base"
                  and _json(cfg[k]) != _json(incumbent.get(k)))


def _build_check(cfg: dict) -> str | None:
    """None when the config builds through the registry; else the NotApplicable reason."""
    import torch  # lazy: `ARMS` and the table helpers need no torch

    from minicandi import registry as R
    from minicandi.components import NotApplicable
    try:
        with torch.random.fork_rng(devices=[]):
            R.build_model(cfg)
    except NotApplicable as e:
        return str(e)
    return None


def status(incumbent: dict, arm: str, cfg: dict, build_check=_build_check) -> str:
    """`write`, or `not_applicable: <reason>`, or `no_change: <reason>`."""
    if arm in _ACTS_ONLY and not _ACTS_ONLY[arm][0](incumbent):
        return f"not_applicable: {_ACTS_ONLY[arm][1]}"
    declared = {k for k in ARMS[arm][1]}
    if not (declared & set(changed_keys(incumbent, cfg))):
        return "no_change: the incumbent already holds " + ", ".join(
            f"{k} = {_json(cfg[k])}" for k in sorted(declared))
    why = build_check(cfg)
    if why is not None:
        return f"not_applicable: {why}"
    return "write"


def plan(incumbent: dict, paths: dict | None = None,
         dna_embedding_dim: int = DNA_EMBEDDING_DIM, build_check=_build_check) -> list[dict]:
    """One row per arm (ARMS_TSV_COLUMNS) plus `config` (None when no json is written)."""
    rows, statuses = [], {}
    for arm, (group, _) in ARMS.items():
        cfg = arm_config(incumbent, arm, paths, dna_embedding_dim)
        st = statuses[arm] = status(incumbent, arm, cfg, build_check)
        comp = COMPARISON.get(arm, "incumbent")
        if comp != "incumbent" and statuses.get(comp, "write").startswith("no_change"):
            comp = "incumbent"      # the incumbent already is that arm
        keys = changed_keys(incumbent, cfg)
        need = NEEDS.get(arm, "")
        if need and need in cfg:
            need = f"{need}={cfg[need]}"
        rows.append({
            "arm": arm,
            "keys changed": ";".join(f"{k}={_cell(cfg[k])}" for k in keys),
            "comparison incumbent": comp,
            "d14_condition_1": "true" if arm in D14 else "false",
            "needs": need,
            "group": group,
            "simpler_than_base": _json(SIMPLER[arm]),
            "joins_design": ("no: " + NEVER_IN_DESIGN[arm]) if arm in NEVER_IN_DESIGN else "yes",
            "status": st,
            "config": cfg if st == "write" else None,
        })
    return rows


def write(incumbent: dict, out, paths: dict | None = None,
          dna_embedding_dim: int = DNA_EMBEDDING_DIM, build_check=_build_check) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = plan(incumbent, paths, dna_embedding_dim, build_check)
    for r in rows:
        stale = out / f"{r['arm']}.json"
        if r["config"] is not None:
            stale.write_text(json.dumps(r["config"], indent=1) + "\n", encoding="utf-8")
        elif stale.exists():
            stale.unlink()
    lines = ["\t".join(ARMS_TSV_COLUMNS)]
    lines += ["\t".join(str(r[c]).replace("\t", " ").replace("\n", " ")
                        for c in ARMS_TSV_COLUMNS) for r in rows]
    (out / "arms.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--incumbent", required=True, help="the settled config json")
    w.add_argument("--out", required=True)
    w.add_argument("--average-h5", default=None)
    w.add_argument("--qc-table", default=None)
    w.add_argument("--flashzoi-emb", default=None)
    w.add_argument("--gnlm-emb", default=None)
    w.add_argument("--dna-embedding-dim", type=int, default=DNA_EMBEDDING_DIM)
    args = ap.parse_args(argv)
    inc = json.loads(Path(args.incumbent).read_text(encoding="utf-8"))
    paths = {"average_h5": args.average_h5, "qc_table": args.qc_table,
             "flashzoi_emb": args.flashzoi_emb, "gnlm_emb": args.gnlm_emb}
    rows = write(inc, args.out, paths, args.dna_embedding_dim)
    n = sum(r["config"] is not None for r in rows)
    print(f"wrote {n} arm configs and arms.tsv ({len(rows)} rows) under {args.out}")
    for r in rows:
        if r["config"] is None:
            print(f"  {r['arm']}: {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
