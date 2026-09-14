#!/usr/bin/env python3
"""Resubmit the experiment/fold runs of a framework that did not complete.

Runs on the login node with the system python: it only reads the filesystem and
squeue, so it needs neither the venv nor hydra.

    python3 scripts/resubmit.py seg                 # every experiment
    python3 scripts/resubmit.py seg exp_wd_0 exp_ls_1
    python3 scripts/resubmit.py seg --dry-run
    python3 scripts/resubmit.py seg --exclude ixb7 --yes

Completion follows each framework's own check_training.py, with one addition:
predictions/ must hold as many files as the fold has validation cases, not just
be non-empty. predict() writes them one per iteration and skips entirely when
the directory is non-empty, so a job killed mid-loop would otherwise look
finished forever and be scored on an arbitrary subset of cases.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import yaml

MIN_CKPT_BYTES = 1024
RESUBMIT_STATES = {
    "NOT_STARTED",
    "FAILED",
    "INCOMPLETE",
    "CORRUPTED",
    "NO_PREDICTIONS",
    "PARTIAL_PREDICTIONS",
}


def resolve(value, cfg, depth=0):
    """Expand ${a.b.c} and ${oc.env:VAR} against an already-composed config."""
    if not isinstance(value, str) or depth > 10:
        return value
    value = re.sub(r"\$\{oc\.env:(\w+)\}", lambda m: os.environ.get(m.group(1), os.getcwd()), value)

    def lookup(match):
        node = cfg
        for part in match.group(1).split("."):
            if not isinstance(node, dict) or part not in node:
                return match.group(0)
            node = node[part]
        return str(node)

    expanded = re.sub(r"\$\{([\w.]+)\}", lookup, value)
    return expanded if expanded == value else resolve(expanded, cfg, depth + 1)


def load_run_cfg(run_dir):
    path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(path):
        return None
    try:
        return yaml.safe_load(open(path))
    except Exception:
        return None


def expected_predictions(cfg, fold):
    """Validation-case count for this fold, from the config the run actually
    used — the experiment's yaml may have been edited or deleted since."""
    if not cfg:
        return None
    try:
        folds_file = resolve(cfg["dataset"]["folds"]["file_path"], cfg)
        return len(json.loads(open(folds_file).read())[fold]["val"])
    except Exception:
        return None


def log_has_error(framework, exp, fold):
    logs = sorted(
        glob.glob(f"slurm_logs/{framework}-{exp}-fold_{fold}/*.out"),
        key=os.path.getmtime,
    )
    if not logs:
        return False
    tail = open(logs[-1], errors="ignore").read()[-2000:]
    return "Error" in tail or "Traceback" in tail


def big_enough(path):
    return os.path.exists(path) and os.path.getsize(path) >= MIN_CKPT_BYTES


def training_artifact(framework, run_dir, cfg):
    """(trained, marker_present) for the framework's own layout."""
    if framework == "seg":
        ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "best-model-*.ckpt"))
        trained = len(ckpts) == 1 and big_enough(ckpts[0])
        return trained, os.path.exists(os.path.join(run_dir, "checkpoints", ".complete"))
    if framework in ("yolo", "yolo_sem"):
        weights = os.path.join(run_dir, "train", "weights", "best.pt")
        return big_enough(weights), os.path.exists(os.path.join(run_dir, "train", ".complete"))
    if framework == "nnunet":
        # nnU-Net writes outside the hydra run dir; the path is only knowable
        # from the config the run composed.
        if not cfg:
            return False, False
        try:
            fold_dir = os.path.join(
                resolve(cfg["paths"]["nnunet"]["results"], cfg),
                f"Dataset{int(cfg['dataset']['id']):03d}_{cfg['dataset']['name']}",
                f"{cfg['experiment'].get('trainer', 'nnUNetTrainer')}__"
                f"{cfg['experiment'].get('plans', 'nnUNetPlans')}__2d",
                f"fold_{cfg['experiment']['fold']}",
            )
        except Exception:
            return False, False
        trained = big_enough(os.path.join(fold_dir, "checkpoint_final.pth"))
        validation = os.path.join(fold_dir, "validation")
        return trained, bool(trained and os.path.isdir(validation) and os.listdir(validation))
    raise SystemExit(f"unknown framework: {framework}")


def check(framework, exp, fold):
    """-> (state, detail, predictions_dir_to_clear_or_None)"""
    run_dir = os.path.join("results", framework, exp, f"fold_{fold}")
    if not os.path.isdir(run_dir):
        return "NOT_STARTED", "", None

    cfg = load_run_cfg(run_dir)
    trained, marker = training_artifact(framework, run_dir, cfg)
    if not trained:
        state = "FAILED" if log_has_error(framework, exp, fold) else "INCOMPLETE"
        return state, "no usable checkpoint", None

    if framework == "nnunet":
        # `marker` doubles as "validation output present" for nnU-Net.
        return ("COMPLETED", "", None) if marker else ("NO_PREDICTIONS", "", None)

    pred_dir = os.path.join(run_dir, "predictions")
    n = len(glob.glob(os.path.join(pred_dir, "*.png")))
    if n == 0:
        return "NO_PREDICTIONS", "", None
    want = expected_predictions(cfg, fold)
    if want is not None and n != want:
        # predict() skips whenever predictions/ is non-empty, so the directory
        # has to go or the resubmitted job will not regenerate it.
        return "PARTIAL_PREDICTIONS", f"{n}/{want} cases", pred_dir
    if not marker and want is None:
        return "INCOMPLETE", "no completion marker", None
    return "COMPLETED", f"{n} cases", None


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
            n_folds = yaml.safe_load(open("common/conf/dataset/woodknots.yaml"))["folds"]["k"]
        except Exception:
            n_folds = 5

    queued = queued_job_names()
    todo, to_clear, counts = [], [], {}
    print(f"{'experiment':<22} " + " ".join(f"f{i}".ljust(11) for i in range(n_folds)))
    for exp in experiments:
        cells = []
        for fold in range(n_folds):
            if f"{fw}-{exp}-fold_{fold}" in queued:
                state, detail, clear = "QUEUED", "", None
            else:
                state, detail, clear = check(fw, exp, fold)
            counts[state] = counts.get(state, 0) + 1
            if state in RESUBMIT_STATES:
                todo.append((exp, fold))
                if clear:
                    to_clear.append(clear)
            cells.append(state[:11])
        print(f"{exp:<22} " + " ".join(c.ljust(11) for c in cells))

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not todo:
        print("\nnothing to resubmit.")
        return 0

    print(f"\n{len(todo)} job(s) to resubmit:")
    for exp, fold in todo:
        print(f"   {fw} {exp} fold {fold}")
    if to_clear:
        print(f"\n{len(to_clear)} partial predictions dir(s) will be DELETED first")
        print("(predict() skips a non-empty predictions/, so they would not be regenerated):")
        for d in to_clear:
            print(f"   rm -rf {d}")

    if args.dry_run:
        print("\n--dry-run: nothing submitted.")
        return 0
    if not args.yes:
        if input("\nproceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1

    for d in to_clear:
        shutil.rmtree(d, ignore_errors=True)
        print(f"removed {d}")

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
