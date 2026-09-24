#!/usr/bin/env python3
"""Resubmit the experiment/fold runs of a framework that did not complete.

Runs on the login node with the system python: it only reads the filesystem and
squeue, so it needs neither the venv nor hydra.

    python3 scripts/resubmit.py seg                 # every experiment
    python3 scripts/resubmit.py seg exp_wd_0 exp_ls_1
    python3 scripts/resubmit.py seg --dry-run
    python3 scripts/resubmit.py seg --exclude ixb7 --yes

A run counts as complete when its run dir holds both markers every framework
writes: .train_complete (touched as the last step of train()) and
.pred_complete (touched once every test-case prediction is written). A job
killed mid-way leaves the marker missing, and predict() overwrites whatever
partial predictions it left behind, so nothing has to be cleaned up first.
See scripts/check_runs.py for the read-only version of this check.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

TRAIN_COMPLETE = ".train_complete"
PRED_COMPLETE = ".pred_complete"
RESUBMIT_STATES = {"NOT_STARTED", "FAILED", "INCOMPLETE", "NO_PREDICTIONS"}


def log_has_error(framework, exp, fold):
    logs = sorted(
        glob.glob(f"slurm_logs/{framework}-{exp}-fold_{fold}/*.out"),
        key=os.path.getmtime,
    )
    if not logs:
        return False
    tail = open(logs[-1], errors="ignore").read()[-2000:]
    return "Error" in tail or "Traceback" in tail


def check(framework, exp, fold):
    """-> (state, detail)"""
    run_dir = os.path.join("results", framework, exp, f"fold_{fold}")
    if not os.path.isdir(run_dir):
        return "NOT_STARTED", ""

    if not os.path.exists(os.path.join(run_dir, TRAIN_COMPLETE)):
        state = "FAILED" if log_has_error(framework, exp, fold) else "INCOMPLETE"
        return state, "training did not finish"

    if not os.path.exists(os.path.join(run_dir, PRED_COMPLETE)):
        return "NO_PREDICTIONS", "predictions did not finish"

    n = len(glob.glob(os.path.join(run_dir, "predictions", "*.npz")))
    return "COMPLETED", f"{n} cases"


def queued_job_names():
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"],
            capture_output=True, text=True, timeout=30,
        )
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework")
    ap.add_argument("experiments", nargs="*", help="default: every experiment with a conf/experiment yaml")
    ap.add_argument("--folds", type=int, default=None, help="number of folds (default: dataset.folds.k)")
    ap.add_argument("--env", default="cluster")
    ap.add_argument("--exclude", default=None, help="node list to exclude, e.g. ixb7")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    fw = args.framework
    if not os.path.isdir(fw):
        raise SystemExit(f"no such framework directory: {fw}/")

    experiments = args.experiments or sorted(
        os.path.basename(p)[:-5] for p in glob.glob(f"{fw}/conf/experiment/*.yaml")
    )
    if not experiments:
        raise SystemExit(f"no experiments found under {fw}/conf/experiment/")

    n_folds = args.folds
    if n_folds is None:
        try:
            cfg_text = open("common/conf/dataset/woodknots.yaml").read()
            n_folds = int(re.search(r"^\s+k:\s*(\d+)", cfg_text, re.M).group(1))
        except Exception:
            n_folds = 5

    queued = queued_job_names()
    todo, counts = [], {}
    print(f"{'experiment':<22} " + " ".join(f"f{i}".ljust(11) for i in range(n_folds)))
    for exp in experiments:
        cells = []
        for fold in range(n_folds):
            if f"{fw}-{exp}-fold_{fold}" in queued:
                state, detail = "QUEUED", ""
            else:
                state, detail = check(fw, exp, fold)
            counts[state] = counts.get(state, 0) + 1
            if state in RESUBMIT_STATES:
                todo.append((exp, fold))
            cells.append(state[:11])
        print(f"{exp:<22} " + " ".join(c.ljust(11) for c in cells))

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not todo:
        print("\nnothing to resubmit.")
        return 0

    print(f"\n{len(todo)} job(s) to resubmit:")
    for exp, fold in todo:
        print(f"   {fw} {exp} fold {fold}")
    if args.dry_run:
        print("\n--dry-run: nothing submitted.")
        return 0
    if not args.yes:
        if input("\nproceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1

    env = dict(os.environ)
    if args.exclude:
        env["SBATCH_EXCLUDE"] = args.exclude  # sbatch reads this as --exclude
    failed = 0
    for exp, fold in todo:
        r = subprocess.run(
            ["./scripts/run.sh", fw, exp, str(fold), f"--env={args.env}"],
            env=env,
        )
        if r.returncode != 0:
            failed += 1
            print(f"submit FAILED: {exp} fold {fold}", file=sys.stderr)
    print(f"\nsubmitted {len(todo) - failed}/{len(todo)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
