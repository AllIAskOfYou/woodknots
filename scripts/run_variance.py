#!/usr/bin/env python3
"""Estimate run-to-run variance: train, evaluate and report num_runs repeats of
one reference configuration, across all folds.

Run i is the reference config with seed=i, written under its own name
<reference_config>_<i> (results/<framework>/<reference_config>_<i>/fold_<f>/)
via Hydra overrides — no per-run config files. The spread across runs combines
every source of run-to-run variation: the seed, GPU nondeterminism, and the
hardware each job lands on (e.g. B200 vs B300), which cannot be separated here.

Three stages, each safe to rerun (finished and queued runs are skipped):

    python3 scripts/run_variance.py seg --env=cluster                  # 1. train + predict
    python3 scripts/run_variance.py seg --env=cluster --evaluate       # 2. evaluate
    python3 scripts/run_variance.py seg --report article/var_seg.txt   # 3. report

Standard library only. nnU-Net ignores the seed, so for it the runs differ by
nondeterminism and hardware only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_COMPLETE = ".train_complete"
PRED_COMPLETE = ".pred_complete"


def num_folds() -> int:
    text = (PROJECT_ROOT / "common/conf/dataset/woodknots.yaml").read_text()
    return int(re.search(r"^\s+k:\s*(\d+)", text, re.MULTILINE).group(1))


def queued_job_names() -> set[str]:
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"],
            capture_output=True, text=True, timeout=30,
        )
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def launch(script: str, framework: str, reference: str, run: int, fold: int, env: str, dry_run: bool) -> bool:
    """scripts/run.sh or scripts/evaluate.sh for run `run` of the reference."""
    cmd = [
        f"./scripts/{script}", framework, reference, str(fold), f"--env={env}",
        "--", f"seed={run}", f"experiment.name={reference}_{run}",
    ]
    if dry_run:
        print("   would run:", " ".join(cmd))
        return True
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("--env", default="cluster", choices=["local", "cluster"])
    ap.add_argument("--num_runs", type=int, default=10)
    ap.add_argument("--reference_config", default="exp_reference")
    stage = ap.add_mutually_exclusive_group()
    stage.add_argument("--evaluate", action="store_true", help="evaluate every predicted run")
    stage.add_argument("--report", metavar="OUTPUT", help="write the variance report (.json = full numbers)")
    ap.add_argument("--force", action="store_true",
                    help="with --evaluate: re-evaluate runs that already have eval_results.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fw, ref = args.framework, args.reference_config
    if not (PROJECT_ROOT / fw / "conf" / "experiment" / f"{ref}.yaml").exists():
        raise SystemExit(f"no reference config {fw}/conf/experiment/{ref}.yaml")

    if args.report:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import seed_variance

        runs = seed_variance.load_runs(PROJECT_ROOT / "results", fw, f"{ref}_")
        runs = {seed: folds for seed, folds in runs.items() if seed < args.num_runs}
        if len(runs) < 2:
            raise SystemExit(f"need at least 2 evaluated runs of {ref}_<i>, found {len(runs)}")
        summary = seed_variance.summarize(runs)
        report = seed_variance.render(summary, fw, f"{ref}_")
        print(report, end="")
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".json":
            out.write_text(json.dumps({"framework": fw, "reference_config": ref, **summary}, indent=2) + "\n")
        else:
            out.write_text(report)
        print(f"\nSaved to {out}")
        return 0

    queued = queued_job_names() if args.env == "cluster" else set()
    job_prefix = "eval-" if args.evaluate else ""
    todo, counts = [], {}
    for run in range(args.num_runs):
        for fold in range(num_folds()):
            run_dir = PROJECT_ROOT / "results" / fw / f"{ref}_{run}" / f"fold_{fold}"
            predicted = (run_dir / PRED_COMPLETE).exists()
            if f"{job_prefix}{fw}-{ref}_{run}-fold_{fold}" in queued:
                state = "queued"
            elif args.evaluate:
                if not predicted:
                    state = "not predicted yet"
                elif (run_dir / "eval_results.json").exists() and not args.force:
                    state = "evaluated"
                else:
                    state = "to evaluate"
            else:
                state = "done" if predicted else "to train"
            counts[state] = counts.get(state, 0) + 1
            if state in ("to train", "to evaluate"):
                todo.append((run, fold))

    print(f"{fw} {ref}, {args.num_runs} runs x {num_folds()} folds: "
          + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    if fw == "nnunet" and not args.evaluate:
        print("note: nnU-Net ignores the seed; its runs differ by nondeterminism and hardware only.")

    script = "evaluate.sh" if args.evaluate else "run.sh"
    failed = sum(not launch(script, fw, ref, run, fold, args.env, args.dry_run) for run, fold in todo)
    if todo and not args.dry_run:
        print(f"launched {len(todo) - failed}/{len(todo)} via {script}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
