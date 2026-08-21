"""t18 / D7 — build the `msevar` cross-biosample variance pools, and report what went into them.

`msevar` weights each squared error by the variance of that assay at that position across a pool of
experiments. The organizers built theirs with `build_var_npy.py` over the **267 EIC training
experiments**; no such vector was ever published, so ours has to be rebuilt and the difference from
theirs has to stay visible rather than be assumed away. That is the whole job of this script: the
numbers go into `<out>/<corpus>/var_<chrom>.npz`, and the *membership* goes into `membership.json`
and `REPORT.md` beside them, so a reader can check the pool rather than trust it.

Runs **where the store is** (Fir). The pools are far too large for git; only the report and the
membership JSON are small enough to commit.

    python tools/build_varpool.py \
        --store  /…/CANDI_STORE/eic \
        --out    /scratch/$USER/candi_kit/varpool \
        --corpus eic --chroms chr21 --prefix T_

`--prefix T_` is how "which biosamples are training" is answered for the EIC store, and it is a
flag rather than a default buried in the builder because the answer is a property of the regime,
not of the corpus. D16 says store names are opaque and nothing strips a prefix for you; here the
prefix is being used deliberately, by a caller who knows this corpus carries the challenge's own
T_/V_/B_ split, and the resulting list is written out so the choice is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

#: The count the EIC pool is reported against. Not a target — a reference point. Our pool is drawn
#: from the store's training biosamples, which is not the same set as the challenge's 267 training
#: experiments, and the report states the gap rather than hiding it.
EIC_TRAINING_EXPERIMENTS = 267


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True, help="corpus root, e.g. …/CANDI_STORE/eic")
    p.add_argument("--out", required=True, help="varpool root; the corpus dir is made under it")
    p.add_argument("--corpus", default="eic")
    p.add_argument("--chroms", nargs="+", default=["chr21"])
    p.add_argument("--prefix", default="T_",
                   help="biosample-name prefix that marks a training biosample in this corpus")
    p.add_argument("--space", default="pval", choices=["pval", "count"],
                   help="D7: EIC is pval space. A pool is only ever applied to its own arm.")
    p.add_argument("--block-bins", type=int, default=1_000_000)
    args = p.parse_args(argv)

    from candi.bench.annotations import build_variance_pools, varpool_path
    from candi.store.reader import CorpusStore

    store = CorpusStore(args.store)
    train = [b for b in store.biosamples if b.startswith(args.prefix)]
    if not train:
        raise SystemExit(f"no biosample in {args.store} starts with {args.prefix!r}; "
                         f"first few are {store.biosamples[:5]}")
    kind = "pval" if args.space == "pval" else "counts"
    n_tracks = sum(len(store[b].assays(kind)) for b in train)
    print(f"[varpool] {len(train)} training biosamples, {n_tracks} {kind} tracks, "
          f"chroms={args.chroms}")

    t0 = time.time()
    meta_path = build_variance_pools(
        args.store, args.out, corpus=args.corpus, chroms=args.chroms,
        train_biosamples=train, space=args.space, block_bins=args.block_bins)
    elapsed = time.time() - t0
    membership = json.loads(meta_path.read_text())

    lines = [
        f"# msevar variance pools — `{args.corpus}` (D7)",
        "",
        f"- store: `{args.store}`",
        f"- space: **{args.space}** — this pool may only weight the {args.space} arm "
        f"(`EVAL_PLAN.md` §9.7).",
        f"- pool: biosamples whose name starts with `{args.prefix}` — "
        f"**{len(train)}** biosamples, **{n_tracks}** {kind} tracks.",
        f"- against the challenge's **{EIC_TRAINING_EXPERIMENTS}** training experiments: "
        f"{n_tracks - EIC_TRAINING_EXPERIMENTS:+d}. The two sets are not the same set; "
        f"the number is here so the difference is visible, not so it is matched.",
        f"- assays with a usable pool (>= 2 members): **{len(membership)}**. An assay only one "
        f"biosample carries is skipped, because a one-member pool is zero everywhere and "
        f"`msevar` divides by `var.sum()`.",
        f"- built in {elapsed / 60:.1f} min.",
        "",
        "## Files",
        "",
        "| chrom | file | bytes | sha256 |",
        "|---|---|---:|---|",
    ]
    for chrom in args.chroms:
        f = varpool_path(args.out, args.corpus, chrom)
        lines.append(f"| {chrom} | `{f.name}` | {f.stat().st_size:,} | `{_sha256(f)[:16]}…` |")
    lines += ["", "## Pool membership, per assay", "",
              "| assay | n | biosamples |", "|---|---:|---|"]
    for assay in sorted(membership):
        e = membership[assay]
        lines.append(f"| {assay} | {e['n']} | {', '.join(e['biosamples'])} |")
    lines.append("")

    report = meta_path.parent / "REPORT.md"
    report.write_text("\n".join(lines))
    print(f"[varpool] wrote {meta_path}\n[varpool] wrote {report}")
    if not membership:
        # An empty pool is the signature of a wrong --prefix, and it is silent everywhere else:
        # the npz writes fine, the job exits 0, and `msevar` goes missing at scoring time with no
        # trace back to here. Fail at the point where the cause is still visible.
        print(f"[varpool] EMPTY POOL — no assay had 2+ members among the {len(train)} biosamples "
              f"matching --prefix {args.prefix!r}. Check the prefix against the store's own names.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
