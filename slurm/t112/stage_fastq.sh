#!/bin/bash
# t112, chunk C4: stage and md5-verify the FASTQs every FASTQ arm needs, into ONE shared cache.
#
# The cache uses autouri's loc layout, <cache>/<md5 of the URL string>/<basename>, so a Caper leader
# whose --local-loc-dir is a hardlink copy of it logs "skipped due to name_size_match" and downloads
# nothing. The downloading and the md5 check against the ENCODE portal are done by EIC_REPRO's own
# prestage_fastqs.py, called read-only; this script only sequences it and seals the result.
#
# Usage (from the Nibi login node; two jobs, B after A):
#   A=$(sbatch --parsable stage_fastq.sh C19M16 C40M17 C07M20 C12M02)
#   sbatch --dependency=afterok:$A stage_fastq.sh --finalize C40M18 C19M22 C07M29
#
# WHY --force ON EVERY --go. prestage_fastqs.py was written for one loc dir per track, and one of its
# three idle gates refuses a loc dir holding any file modified in the last 24 h. Here seven tracks
# share one cache, so every track after the first would be refused by the previous track's fresh
# downloads. --force skips only that gate (and stale .lock files); the other two gates (a live
# CAPER_<track> SLURM job, a non-idle run_batches driver state) still run. It also lets the script
# replace a right-size, wrong-md5 file, which is what we want. Nothing else writes to this cache:
# the FASTQ-arm leaders read hardlinks of it and are not submitted until both jobs here end.
#
# WHY --finalize SEALS THE CACHE. autouri's copy path truncates in place through a hardlink, so a
# killed leader could corrupt a cached FASTQ for every other arm (the same reason stage_reference_
# cache.py makes its cache 0444). Finalize re-verifies ALL seven tracks with --verify-only, writes
# VERIFIED.tsv, refuses any file in the cache that no track accounts for, then chmods every file 0444.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t112_stage_fastq
#SBATCH --output=/scratch/mforooz/t112_cf/logs/stage/%x_%j.out
#SBATCH --error=/scratch/mforooz/t112_cf/logs/stage/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -uo pipefail

CF=/scratch/mforooz/t112_cf
CACHE=$CF/fastq_cache
LOGDIR=$CF/logs/stage
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline
PRESTAGE=$EIC/scripts/prestage_fastqs.py
ALL_TRACKS=(C19M16 C40M17 C07M20 C12M02 C40M18 C19M22 C07M29)

FINALIZE=0
if [ "${1:-}" = "--finalize" ]; then FINALIZE=1; shift; fi
[ "$#" -ge 1 ] || { echo "usage: stage_fastq.sh [--finalize] TRACK..."; exit 2; }

mkdir -p "$CACHE" "$LOGDIR"
cd "$LOGDIR" || exit 3
echo "=== job ${SLURM_JOB_ID:-none} on $(hostname) tracks: $* finalize=$FINALIZE $(date -u) ==="
echo "prestage md5: $(md5sum "$PRESTAGE" | cut -d' ' -f1)"

# Every input JSON is a single-end one (--tag se). The pe JSONs name the same URLs (checked
# 2026-09-17 for all six ChIP tracks), and C12M02 has only a .dnase.se.json, whose 8 URLs are the
# 4 PE pairs the DNase pe arm uses.
for T in "$@"; do
  echo "=== $T stage start $(date -u) ==="
  python3 "$PRESTAGE" --track "$T" --tag se --loc-dir "$CACHE" --go --force
  RC=$?
  echo "=== $T stage end $(date -u) rc=$RC ==="
  if [ "$RC" -ne 0 ]; then
    echo "!!! HALT: $T prestage exited $RC. Not md5-verified. Stopping; nothing is sealed."
    exit "$RC"
  fi
done

[ "$FINALIZE" -eq 1 ] || { echo "=== staged, not finalized $(date -u) ==="; exit 0; }

for T in "${ALL_TRACKS[@]}"; do
  echo "=== $T verify-only $(date -u) ==="
  python3 "$PRESTAGE" --track "$T" --tag se --loc-dir "$CACHE" --verify-only >"$LOGDIR/verify_$T.txt" 2>&1
  RC=$?
  tail -n 3 "$LOGDIR/verify_$T.txt"
  if [ "$RC" -ne 0 ]; then
    echo "!!! HALT: $T --verify-only exited $RC (see $LOGDIR/verify_$T.txt). Cache NOT sealed."
    exit "$RC"
  fi
done

python3 - "$CACHE" "$LOGDIR" "$EIC" "${ALL_TRACKS[@]}" <<'PY' || { echo "!!! HALT: VERIFIED.tsv not written. Cache NOT sealed."; exit 4; }
import hashlib, json, os, re, sys

cache, logdir, eic, tracks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
PREFIX = "https://www.encodeproject.org/files/"


def urls(o):
    if isinstance(o, str):
        if o.startswith(PREFIX):
            yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from urls(v)
    elif isinstance(o, list):
        for v in o:
            yield from urls(v)


rows = {}
for t in tracks:
    js = [p for p in (f"{eic}/inputs_bwa/{t}.bwa.se.json", f"{eic}/inputs_dnase/{t}.dnase.se.json")
          if os.path.exists(p)]
    if len(js) != 1:
        sys.exit(f"{t}: expected one se input JSON, found {js}")
    by_path = {os.path.join(cache, hashlib.md5(u.encode()).hexdigest(), u.rsplit("/", 1)[-1]): u
               for u in urls(json.load(open(js[0])))}
    text = open(f"{logdir}/verify_{t}.txt").read()
    m = re.search(r"^verified (\d+)/(\d+)$", text, re.M)
    if not m or m.group(1) != m.group(2) or int(m.group(2)) != len(by_path):
        sys.exit(f"{t}: verify log does not show all {len(by_path)} files verified")
    seen = 0
    for line in text.splitlines():
        f = line.split()
        if len(f) == 4 and f[3].startswith(cache + "/") and re.fullmatch(r"[0-9a-f]{32}", f[2]):
            acc, md5, path = f[0], f[2], f[3]
            if path not in by_path:
                sys.exit(f"{t}: {path} is not named by {js[0]}")
            rows[path] = (acc, by_path[path], md5, str(os.path.getsize(path)), path)
            seen += 1
    if seen != len(by_path):
        sys.exit(f"{t}: parsed {seen} plan rows, expected {len(by_path)}")

on_disk = set()
for root, dirs, files in os.walk(cache):
    for d in dirs:
        if d.startswith(".stage_"):
            sys.exit(f"leftover temp dir {os.path.join(root, d)}")
    for fn in files:
        p = os.path.join(root, fn)
        if p != os.path.join(cache, "VERIFIED.tsv"):
            on_disk.add(p)
stray = sorted(on_disk - set(rows))
if stray or set(rows) - on_disk:
    sys.exit(f"cache and verified set differ: stray={stray[:5]} missing={sorted(set(rows) - on_disk)[:5]}")

tmp = os.path.join(cache, ".VERIFIED.tsv.tmp")
with open(tmp, "w") as fh:
    fh.write("acc\turl\tmd5\tbytes\tpath\n")
    for p in sorted(rows):
        fh.write("\t".join(rows[p]) + "\n")
os.replace(tmp, os.path.join(cache, "VERIFIED.tsv"))
print(f"VERIFIED.tsv: {len(rows)} files, {sum(int(r[3]) for r in rows.values()) / 2**30:.2f} GiB")
PY

find "$CACHE" -type f -exec chmod 0444 {} +
NOT_RO=$(find "$CACHE" -type f ! -perm 0444 | wc -l)
echo "files not 0444 after seal: $NOT_RO"
{ echo "=== after, job ${SLURM_JOB_ID:-none} $(date -u) ==="; diskusage_report 2>&1 || echo "diskusage_report unavailable on $(hostname)"; } >>"$LOGDIR/diskusage.txt"
[ "$NOT_RO" -eq 0 ] || exit 5
echo "=== C4 FASTQ CACHE SEALED $(date -u) ==="
