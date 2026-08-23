# Pin `candi` to $KIT, and then PROVE it. Sourced by every job script, after the venv is active
# and after `cd "$KIT"`. Not executable on its own.
#
# WHY THIS EXISTS. `candi_venv` carries an editable install whose .pth is a HARD PATH to the shared
# checkout's src. So `$KIT` and `cd "$KIT"` choose which SCRIPTS run, and the LIBRARY still comes
# from that one shared clone — parked on whatever branch somebody pinned for some other job. A job
# submitted from a branch therefore runs the branch's scripts against main's `candi`, or worse,
# against a third branch's.
#
# It surfaced when t31's calibration ran this branch's tool against a `candi.eval` three branches
# old and died on an ImportError, because the branch had ADDED a function. Had it merely CHANGED
# one, the job would have produced numbers, and nothing in the output would have said which code
# made them. That is the failure this guard exists to make impossible rather than unlikely.
#
# PYTHONPATH is inserted ahead of site-packages, so setting it here wins. Measured on Fir: without
# it `candi` resolves to CANDII/src, with it to $KIT/src. (The module measured at the time was
# `candi.eval`, since deleted — D15; the import below checks the package itself, which cannot go
# stale that way.)
export PYTHONPATH="$KIT/src"
python - "$KIT" <<'PY' || exit 1
import pathlib, sys
import candi
want = pathlib.Path(sys.argv[1]).resolve() / "src" / "candi"
got = pathlib.Path(candi.__file__).resolve().parent
if got != want:
    sys.exit(f"[kit-pin] FATAL: candi imported from {got}, not {want}. The venv's editable install "
             f"pins the shared kit; PYTHONPATH must point at this checkout's src. Refusing to run "
             f"rather than measure unknown code.")
print(f"[kit-pin] candi from {got}")
PY
