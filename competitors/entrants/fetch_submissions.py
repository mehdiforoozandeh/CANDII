#!/usr/bin/env python3
"""Survey and download the 23 entrant submissions from Synapse `syn17083203/submissions_round2/`.

Wraps `vendor/fir_synapse_download.py`, which experiment 001 used to pull the 254 GB of challenge
tracks already on Fir. That module is vendored byte-identical and is not edited; the two things it
does not do are added here:

* **Folder recursion.** Its `list_children` asks for `includeTypes: ["file"]` at one level.
  `submissions_round2/` is a folder of per-team folders, so the tree has to be walked.
* **A size survey that is a gate, not a side effect.** `--survey` alone talks to the API, walks the
  tree, and prints per-team and total bytes without transferring a single track. Run it, report the
  total, and only then download. The budget is ~1 TB scale against a group-charged `/project` quota;
  a pull that turns out to be larger than expected is other people's storage.

The token is a Synapse personal access token with view + download scope at `~/.synapse_pat`,
mode 600. Everything under syn17083203 is otherwise unrestricted -- no data-access committee -- so a
plain registered account suffices. **This script never prints the token**, including in error paths.

Not this script's job: `syn21519009` (the Lavawizard weights) belongs to a different task.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import fir_synapse_download as SYN     # noqa: E402  vendored, byte-identical

SUBMISSIONS_PARENT = "syn17083203"


def read_token(path: str) -> str:
    """Read the PAT, refusing a world- or group-readable file.

    A token that other cluster users can read is a token that has to be revoked, so this fails
    closed rather than warning. The message names the fix and never echoes the contents.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(
            f"no Synapse token at {path}. Create it with view+download scope at "
            f"https://www.synapse.org/#!PersonalAccessTokens: and `chmod 600` it.")
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"{path} is group/world accessible (mode {oct(stat.S_IMODE(mode))}); "
                         f"run `chmod 600 {path}` before using it.")
    token = open(path).read().strip()
    if not token:
        raise SystemExit(f"{path} is empty.")
    return token


def list_entities(parent: str, token: str, types: List[str]) -> List[dict]:
    """Paged `entity/children`, for whichever types are asked for.

    The vendored `list_children` hardcodes files; this is the same call with the type list open, so
    folders can be enumerated too. Same API, same pagination, same retry behaviour.
    """
    out, tok = [], None
    while True:
        body = {"parentId": parent, "includeTypes": types,
                "sortBy": "NAME", "sortDirection": "ASC"}
        if tok:
            body["nextPageToken"] = tok
        d = json.loads(SYN.req(f"{SYN.API}/entity/children", token, "POST", body))
        out.extend(d.get("page", []))
        tok = d.get("nextPageToken")
        if not tok:
            return out


def find_folder(parent: str, name: str, token: str) -> str:
    for f in list_entities(parent, token, ["folder"]):
        if f["name"] == name:
            return f["id"]
    have = [f["name"] for f in list_entities(parent, token, ["folder"])]
    raise SystemExit(f"no folder named {name!r} under {parent}. Present: {have}")


def walk(folder_id: str, token: str, prefix: str = "") -> List[Tuple[str, str, dict]]:
    """Depth-first walk. Returns `(relative_dir, name, entity)` for every file in the tree."""
    out: List[Tuple[str, str, dict]] = []
    for f in list_entities(folder_id, token, ["file"]):
        out.append((prefix, f["name"], f))
    for d in list_entities(folder_id, token, ["folder"]):
        out.extend(walk(d["id"], token, os.path.join(prefix, d["name"])))
    return out


def survey(folder_id: str, token: str) -> Dict[str, object]:
    """Total and per-team bytes, from file handles only -- no track is transferred."""
    files = walk(folder_id, token)
    per_team: Dict[str, Dict[str, float]] = {}
    total = 0
    for rel, name, ent in files:
        size, _ = SYN.file_handle(ent["id"], token)
        total += size
        team = rel.split(os.sep)[0] if rel else "(root)"
        t = per_team.setdefault(team, {"n_files": 0, "bytes": 0})
        t["n_files"] += 1
        t["bytes"] += size
    return {"folder": folder_id, "n_files": len(files), "total_bytes": total,
            "total_gb": total / 1e9, "per_team": per_team}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default="~/.synapse_pat")
    ap.add_argument("--parent", default=SUBMISSIONS_PARENT)
    ap.add_argument("--folder-name", default="submissions_round2")
    ap.add_argument("--folder-id", help="skip the lookup when the synID is already known")
    ap.add_argument("--dest", default="/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/submissions_round2")
    ap.add_argument("--survey", action="store_true", help="size only; transfers nothing")
    ap.add_argument("--survey-out", help="write the survey as json")
    ap.add_argument("--shard", default="0/1", help="'i/n', as in the vendored downloader")
    ap.add_argument("--max-gb", type=float, default=500.0,
                    help="refuse to download a tree larger than this without --yes-i-checked. The "
                         "orchestrator is told the number first; this makes forgetting an error.")
    ap.add_argument("--yes-i-checked", action="store_true",
                    help="the survey total was reported and the pull was approved")
    args = ap.parse_args()

    token = read_token(args.token_file)
    fid = args.folder_id or find_folder(args.parent, args.folder_name, token)
    print(f"[synapse] {args.folder_name} = {fid}", flush=True)

    s = survey(fid, token)
    print(f"[synapse] {s['n_files']} files, {s['total_gb']:.1f} GB total", flush=True)
    for team in sorted(s["per_team"], key=lambda k: -s["per_team"][k]["bytes"]):
        v = s["per_team"][team]
        print(f"   {team:40s} {v['n_files']:4d} files  {v['bytes']/1e9:8.1f} GB", flush=True)
    if args.survey_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.survey_out)), exist_ok=True)
        with open(args.survey_out, "w") as fh:
            json.dump(s, fh, indent=1)
    if args.survey:
        return 0

    if s["total_gb"] > args.max_gb and not args.yes_i_checked:
        raise SystemExit(
            f"tree is {s['total_gb']:.1f} GB, over the {args.max_gb:.0f} GB threshold. Report the "
            f"number, get the pull approved, then re-run with --yes-i-checked.")

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    files = walk(fid, token)[shard_i::shard_n]
    print(f"[synapse] downloading {len(files)} files (shard {args.shard}) -> {args.dest}",
          flush=True)
    for rel, name, ent in files:
        outdir = os.path.join(args.dest, rel)
        os.makedirs(outdir, exist_ok=True)
        dest = os.path.join(outdir, name)
        size, md5 = SYN.file_handle(ent["id"], token)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  have {rel}/{name}", flush=True)
            continue
        # The vendored downloader verifies size and md5 and removes a bad temp file itself.
        SYN.download(ent["id"], dest, token, size, md5)
        print(f"  got  {rel}/{name} {size/1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
