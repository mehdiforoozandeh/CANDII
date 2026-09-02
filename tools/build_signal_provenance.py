"""Join the downloader's plan with the ENCODE portal into one signal-provenance table.

`EpiDenoise/data/download_plan_eic.json` (on Fir) already carries `signal_bigwig_accession` for
every track. What it does not carry is what ENCODE calls that file's units, and that omission is
the whole §10 defect: 40 DNase-seq bigWigs are `read-depth normalized signal` sitting in a layer
the docs call `-log10 p`, and the accession travels with no artifact that would let anyone check.
This writes the accession and its `output_type` per (biosample, assay) so
`candi.store.manifest` can stamp both into `manifest.json` and refuse a build that mixes units.

    python tools/build_signal_provenance.py \\
        --download-plan download_plan_eic.json --corpus eic \\
        --out configs/signal_provenance.eic.json

Read-only against `https://www.encodeproject.org` — one batched `search` request per 40
accessions, no authentication, nothing downloaded but JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PORTAL = "https://www.encodeproject.org"
#: What we keep per file. `output_type` is the point; the rest is there so a later reader can see
#: the sweep really did land on a released GRCh38 bigWig rather than on something else.
FIELDS = ("accession", "output_type", "assay_term_name", "file_format", "assembly", "status",
          "dataset", "date_created", "md5sum", "biological_replicates")
BATCH = 40


def fetch(accessions: list) -> dict:
    out: dict = {}
    for i in range(0, len(accessions), BATCH):
        chunk = accessions[i:i + BATCH]
        q = [("type", "File"), ("format", "json"), ("limit", "all")]
        q += [("accession", a) for a in chunk]
        q += [("field", f) for f in FIELDS]
        url = f"{PORTAL}/search/?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as fh:
            payload = json.load(fh)
        for g in payload.get("@graph", []):
            out[g["accession"]] = g
        print(f"  {min(i + BATCH, len(accessions))}/{len(accessions)}", file=sys.stderr, flush=True)
        time.sleep(0.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download-plan", required=True, type=Path)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--declared-output-type", default="signal p-value")
    ap.add_argument(
        "--alias", action="append", default=[], metavar="STORE=PLAN",
        help="repeatable. The store and the download plan do not always spell a biosample the "
             "same way, and a mismatch leaves that track unstamped. Only pass an alias you have "
             "PROVEN — same BAM accession and same bios accession on both sides — because this "
             "is the one place a wrong name would put a real accession on the wrong track. Every "
             "alias used is written into the output.")
    args = ap.parse_args()

    aliases = dict(a.split("=", 1) for a in args.alias)
    plan = json.loads(args.download_plan.read_text(encoding="utf-8"))
    for store_name, plan_name in aliases.items():
        if plan_name not in plan:
            raise SystemExit(f"--alias {store_name}={plan_name}: {plan_name!r} is not in the plan")
        if store_name in plan:
            raise SystemExit(f"--alias {store_name}=…: {store_name!r} is already a plan name")
        plan[store_name] = plan.pop(plan_name)
    wanted = sorted({r["signal_bigwig_accession"]
                     for per_assay in plan.values() for r in per_assay.values()
                     if r.get("signal_bigwig_accession")})
    print(f"{len(wanted)} signal bigWig accessions in {args.download_plan}", file=sys.stderr)
    files = fetch(wanted)
    missing = [a for a in wanted if a not in files]
    if missing:
        raise SystemExit(f"the portal returned nothing for {len(missing)} accession(s): {missing[:10]}")

    tracks: dict = {}
    counts: dict = {}
    for bios in sorted(plan):
        for assay in sorted(plan[bios]):
            acc = plan[bios][assay].get("signal_bigwig_accession")
            if not acc:
                continue
            f = files[acc]
            tracks.setdefault(bios, {})[assay] = {
                "signal_bigwig_accession": acc,
                "output_type": f.get("output_type"),
                "assay_term_name": f.get("assay_term_name"),
                "file_format": f.get("file_format"),
                "assembly": f.get("assembly"),
                "status": f.get("status"),
                "md5sum": f.get("md5sum"),
                "biological_replicates": f.get("biological_replicates"),
            }
            counts[(f.get("assay_term_name"), f.get("output_type"))] = \
                counts.get((f.get("assay_term_name"), f.get("output_type")), 0) + 1

    doc = {
        "corpus": args.corpus,
        "declared_output_type": args.declared_output_type,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "download_plan": str(args.download_plan),
            "portal": f"{PORTAL}/search/?type=File&accession=…&field=" + "&field=".join(FIELDS),
        },
        "biosample_aliases": aliases,
        "summary": {f"{a} / {o}": n for (a, o), n in sorted(counts.items(), key=lambda kv: -kv[1])},
        "tracks": tracks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    for k, n in doc["summary"].items():
        print(f"  {n:4d}  {k}", file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
