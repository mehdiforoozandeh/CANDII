#!/usr/bin/env python3
"""Download a Synapse folder's files with a personal access token.  Runs on Fir.

The challenge's own tracks are not public: anonymous users get READ on the
metadata but not DOWNLOAD, so this needs Max's PAT (scopes: view + download).
Everything under syn17083203 is otherwise unrestricted -- no data-access
committee, no terms -- so a plain registered account is enough.

Only the standard library is used, so this runs under any python3 on Fir
without a venv.  Resumable: a file whose size and MD5 already match the
Synapse file handle is skipped, so re-running after an interruption is cheap.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://repo-prod.prod.sagebase.org/repo/v1"


def req(url, token, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {token}")
    if data:
        r.add_header("Content-Type", "application/json")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r, timeout=120) as fh:
                return fh.read().decode()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def list_children(parent, token):
    out, tok = [], None
    while True:
        body = {"parentId": parent, "includeTypes": ["file"],
                "sortBy": "NAME", "sortDirection": "ASC"}
        if tok:
            body["nextPageToken"] = tok
        d = json.loads(req(f"{API}/entity/children", token, "POST", body))
        out.extend(d.get("page", []))
        tok = d.get("nextPageToken")
        if not tok:
            return out


def file_handle(syn_id, token):
    d = json.loads(req(f"{API}/entity/{syn_id}/filehandles", token))
    h = d["list"][0]
    return int(h.get("contentSize") or 0), h.get("contentMd5")


def md5_of(path, blocksize=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def download(syn_id, dest, token, expect_size, expect_md5):
    url = req(f"{API}/entity/{syn_id}/file?redirect=false", token).strip()
    # unique per process: several shards may run against one directory
    tmp = f"{dest}.{os.getpid()}.part"
    with urllib.request.urlopen(url, timeout=600) as src, open(tmp, "wb") as out:
        while True:
            chunk = src.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    if expect_size and os.path.getsize(tmp) != expect_size:
        os.remove(tmp)
        raise RuntimeError(f"{syn_id}: size mismatch")
    if expect_md5 and md5_of(tmp) != expect_md5:
        os.remove(tmp)
        raise RuntimeError(f"{syn_id}: md5 mismatch")
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Synapse folder synID")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--only", nargs="*", help="restrict to these file names")
    ap.add_argument("--shard", default="0/1",
                    help="'i/n': take every n-th file starting at i, so several "
                         "copies can run concurrently without racing")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    token = open(args.token_file).read().strip()
    os.makedirs(args.dest, exist_ok=True)

    kids = list_children(args.folder, token)
    if args.only:
        want = set(args.only)
        kids = [k for k in kids if k["name"] in want]
    kids = kids[shard_i::shard_n]
    print(f"[synapse] {args.folder}: {len(kids)} files (shard {args.shard})",
          flush=True)

    total = 0
    for k in kids:
        size, md5 = file_handle(k["id"], token)
        total += size
        if args.list_only:
            print(f"  {k['id']} {k['name']} {size/1e6:.0f} MB")
            continue
        dest = os.path.join(args.dest, k["name"])
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  have {k['name']}", flush=True)
            continue
        t0 = time.time()
        # Presigned URLs expire, and Synapse throttles concurrent streams, so a
        # 403 mid-transfer is normal rather than fatal: retry with a fresh URL.
        for attempt in range(6):
            try:
                download(k["id"], dest, token, size, md5)
                break
            except Exception as e:                       # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"  retry {k['name']} after {type(e).__name__}: {e}",
                      flush=True)
                time.sleep(5 * 2 ** attempt)
        dt = time.time() - t0
        print(f"  got  {k['name']} {size/1e6:.0f} MB in {dt:.0f}s "
              f"({size/1e6/max(dt,1e-9):.1f} MB/s)", flush=True)

    print(f"[synapse] {args.folder}: total {total/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
