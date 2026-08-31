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
import time
from typing import Dict, List, Optional, Tuple

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


def walk(folder_id: str, token: str, prefix: str = "",
         only: Optional[set] = None) -> List[Tuple[str, str, dict]]:
    """Depth-first walk. Returns `(relative_dir, name, entity)` for every file in the tree.

    `only` filters on the basename stem (`C05M17` from `C05M17.bigwig`). The order is deterministic
    -- names sorted within a folder, folders sorted -- which is what makes `--shard i/n` a stable
    partition across separate job submissions, so a resumed shard resumes its own slice.
    """
    out: List[Tuple[str, str, dict]] = []
    for f in list_entities(folder_id, token, ["file"]):
        if only is not None and f["name"].rsplit(".", 1)[0] not in only:
            continue
        out.append((prefix, f["name"], f))
    for d in list_entities(folder_id, token, ["folder"]):
        out.extend(walk(d["id"], token, os.path.join(prefix, d["name"]), only))
    return out


def read_only_set(path: str) -> set:
    return {l.strip() for l in open(path) if l.strip()}


def survey(folder_id: str, token: str, only: Optional[set] = None) -> Dict[str, object]:
    """Total and per-team bytes, from file handles only -- no track is transferred.

    Per-file sizes and md5s are recorded, not just the totals. That makes the survey json the
    manifest `--verify` checks the download against, and it is what lets a team's MISSING experiment
    be named rather than merely counted -- UIOWA_Michaelson ships 50 files where every other team
    ships 51, and a placement table has to say which one is absent.
    """
    files = walk(folder_id, token, only=only)
    per_team: Dict[str, Dict[str, object]] = {}
    total = 0
    for rel, name, ent in files:
        size, md5 = SYN.file_handle(ent["id"], token)
        total += size
        team = rel.split(os.sep)[0] if rel else "(root)"
        t = per_team.setdefault(team, {"n_files": 0, "bytes": 0, "files": {}})
        t["n_files"] += 1
        t["bytes"] += size
        t["files"][name] = {"size": size, "md5": md5, "syn": ent["id"], "rel": rel}
    return {"folder": folder_id, "n_files": len(files), "total_bytes": total,
            "total_gb": total / 1e9, "per_team": per_team}


def verify(s: Dict[str, object], dest: str) -> int:
    """Per-team on-disk check against the survey manifest. Returns the number of bad teams.

    Size only, not md5: the downloader already verified md5 for every file it wrote, refusing to
    rename the temp file otherwise, so a re-hash of a terabyte would re-prove what is already
    proven. What this catches is the thing md5 cannot -- a file that never arrived at all.
    """
    bad = 0
    print(f"{'team':40s} {'files':>12s} {'bytes':>14s}  status")
    for team in sorted(s["per_team"]):
        t = s["per_team"][team]
        missing, wrong = [], []
        on_disk = 0
        for name, meta in t["files"].items():
            p = os.path.join(dest, meta["rel"], name)
            if not os.path.exists(p):
                missing.append(name)
                continue
            got = os.path.getsize(p)
            on_disk += got
            if got != meta["size"]:
                wrong.append(f"{name} ({got} != {meta['size']})")
        n_have = t["n_files"] - len(missing)
        ok = not missing and not wrong
        bad += 0 if ok else 1
        status = "OK" if ok else f"INCOMPLETE missing={len(missing)} wrong_size={len(wrong)}"
        print(f"{team:40s} {n_have:5d}/{t['n_files']:<6d} {on_disk/1e9:11.1f} GB  {status}")
        for m in missing[:3]:
            print(f"{'':40s}   missing: {m}")
        for w in wrong[:3]:
            print(f"{'':40s}   wrong size: {w}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default="~/.synapse_pat")
    ap.add_argument("--parent", default=SUBMISSIONS_PARENT)
    ap.add_argument("--folder-name", default="submissions_round2")
    ap.add_argument("--folder-id", help="skip the lookup when the synID is already known")
    # User scratch, NOT /project (PI ruling, 2026-08-25). These are one-shot, re-downloadable
    # scoring inputs: read once per team, they produce small score tables and are never needed
    # again. /project quota is charged to the whole group, so a terabyte of regenerable input there
    # is a cost other people pay. Scratch's 60-day purge is the correct lifetime -- only the
    # DERIVED score tables are permanent, and those live in the repo checkout and rsync down to
    # `cruxvault/results/t54/` as small files.
    ap.add_argument("--dest", default=os.path.expanduser("~/scratch/t54_submissions_round2"))
    ap.add_argument("--survey", action="store_true", help="size only; transfers nothing")
    ap.add_argument("--survey-out", help="write the survey as json")
    ap.add_argument("--verify", action="store_true",
                    help="check what is on disk against the survey manifest, per team, and exit "
                         "non-zero if any team is short. Transfers nothing.")
    ap.add_argument("--survey-in", help="reuse a saved survey json instead of re-walking the API")
    ap.add_argument("--shard", default="0/1", help="'i/n', as in the vendored downloader")
    ap.add_argument("--only-from",
                    help="a file of C##M## stems (blind_experiments.txt); restrict the pull to "
                         "files whose basename matches one. The organizers' two baseline folders "
                         "hold 56 tracks -- the 51 blind experiments plus five C14* predictions "
                         "for a cell that is not in the EIC 363 at all -- and scoring the panel "
                         "needs only the 51.")
    ap.add_argument("--max-gb", type=float, default=500.0,
                    help="refuse to download a tree larger than this without --yes-i-checked. The "
                         "orchestrator is told the number first; this makes forgetting an error.")
    ap.add_argument("--yes-i-checked", action="store_true",
                    help="the survey total was reported and the pull was approved")
    args = ap.parse_args()
    only_set = read_only_set(args.only_from) if args.only_from else None
    if only_set:
        print(f"[synapse] restricted to {len(only_set)} named experiments", flush=True)

    if args.survey_in:
        # Verifying a finished pull needs no token and no API round-trip: the saved manifest already
        # carries every name, size and md5.
        s = json.load(open(args.survey_in))
        print(f"[synapse] manifest {args.survey_in}: {s['n_files']} files, "
              f"{s['total_gb']:.1f} GB", flush=True)
    else:
        token = read_token(args.token_file)
        fid = args.folder_id or find_folder(args.parent, args.folder_name, token)
        print(f"[synapse] {args.folder_name} = {fid}", flush=True)
        s = survey(fid, token, only=only_set)
        print(f"[synapse] {s['n_files']} files, {s['total_gb']:.1f} GB total", flush=True)
        for team in sorted(s["per_team"], key=lambda k: -s["per_team"][k]["bytes"]):
            v = s["per_team"][team]
            print(f"   {team:40s} {v['n_files']:4d} files  {v['bytes']/1e9:8.1f} GB", flush=True)
        if args.survey_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.survey_out)), exist_ok=True)
            with open(args.survey_out, "w") as fh:
                json.dump(s, fh, indent=1)

    if args.verify:
        bad = verify(s, args.dest)
        print(f"\n{'ALL TEAMS COMPLETE' if not bad else str(bad) + ' TEAM(S) INCOMPLETE'}")
        return 0 if not bad else 1
    if args.survey:
        return 0

    token = read_token(args.token_file)
    fid = args.folder_id or find_folder(args.parent, args.folder_name, token)

    if s["total_gb"] > args.max_gb and not args.yes_i_checked:
        raise SystemExit(
            f"tree is {s['total_gb']:.1f} GB, over the {args.max_gb:.0f} GB threshold. Report the "
            f"number, get the pull approved, then re-run with --yes-i-checked.")

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    files = walk(fid, token, only=only_set)[shard_i::shard_n]
    print(f"[synapse] downloading {len(files)} files (shard {args.shard}) -> {args.dest}",
          flush=True)
    n_got = n_have = 0
    for rel, name, ent in files:
        outdir = os.path.join(args.dest, rel)
        os.makedirs(outdir, exist_ok=True)
        dest = os.path.join(outdir, name)
        size, md5 = SYN.file_handle(ent["id"], token)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  have {rel}/{name}", flush=True)
            n_have += 1
            continue
        # The vendored downloader verifies size and md5 and removes a bad temp file itself, but it
        # does NOT retry -- its own main() wraps the call in this loop, and dropping that was a real
        # defect: a first run with eight shards lost five of them, four to `HTTP 403 Forbidden` and
        # one to `RemoteDisconnected`, spread over the first hour. The vendored comment says exactly
        # why: presigned URLs expire and Synapse throttles concurrent streams, so a 403 mid-transfer
        # is normal rather than fatal. `download()` requests a FRESH presigned URL on entry, so a
        # plain retry is the correct remedy.
        for attempt in range(6):
            try:
                SYN.download(ent["id"], dest, token, size, md5)
                break
            except Exception as e:                       # noqa: BLE001
                if attempt == 5:
                    print(f"  GIVING UP on {rel}/{name} after 6 attempts: "
                          f"{type(e).__name__}: {e}", flush=True)
                    raise
                wait = 5 * 2 ** attempt
                print(f"  retry {rel}/{name} in {wait}s after {type(e).__name__}: {e}", flush=True)
                time.sleep(wait)
        n_got += 1
        print(f"  got  {rel}/{name} {size/1e6:.0f} MB", flush=True)
    print(f"[synapse] shard {args.shard} done: {n_got} downloaded, {n_have} already present",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
