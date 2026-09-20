#!/usr/bin/env python3
"""Report how often the phase-2 fine-tune beat phase-1 training.

train() logs one of two lines per run and keeps the better weights:
    Fine-tuned weights improved fitness (0.1066 > 0.0908), replacing ...
    Fine-tuning did not improve fitness (phase1=0.22467, phase2=0.18069), ...
This scans results/yolo/<experiment>/fold_<i>/run.log for them. A rerun
appends to the same log, so the last match in a file wins.

Standard library only, so it runs with a plain `python3`:

    python3 yolo/check_finetune.py
    python3 yolo/check_finetune.py exp_reference exp_rot_0
    python3 yolo/check_finetune.py --results-root results_sc
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMPROVED = re.compile(r"weights improved fitness \(([\d.]+) > ([\d.]+)\)")
NOT_IMPROVED = re.compile(r"did not improve fitness \(phase1=([\d.]+), phase2=([\d.]+)\)")


def read_run(log_path: Path):
    """-> (improved, phase1, phase2) for the last fine-tune in this log, or None."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")

    last = None
    for match in IMPROVED.finditer(text):
        last = (match.end(), True, float(match.group(2)), float(match.group(1)))
    for match in NOT_IMPROVED.finditer(text):
        if last is None or match.end() > last[0]:
            last = (match.end(), False, float(match.group(1)), float(match.group(2)))
    return last[1:] if last else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("experiments", nargs="*", help="default: every experiment found in the results dir")
    ap.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results"),
        help="paths.results_root (default: <project>/results)",
    )
    args = ap.parse_args()

    yolo_root = Path(args.results_root) / "yolo"
    experiments = args.experiments or sorted(p.name for p in yolo_root.iterdir() if p.is_dir())

    rows, improved_deltas, kept_deltas = [], [], []
    for exp in experiments:
        for fold_dir in sorted((yolo_root / exp).glob("fold_*")):
            run = read_run(fold_dir / "run.log")
            if run is None:
                continue
            improved, phase1, phase2 = run
            rows.append((exp, fold_dir.name, improved, phase1, phase2))
            (improved_deltas if improved else kept_deltas).append(phase2 - phase1)

    if not rows:
        print(f"No fine-tune lines found under {yolo_root}")
        return 1

    width = max(len(exp) for exp, *_ in rows)
    print(f"{'experiment':<{width}}  {'fold':<6}  {'result':<9}  {'phase1':>8}  {'phase2':>8}  {'delta':>8}")
    print("-" * (width + 48))
    for exp, fold, improved, phase1, phase2 in rows:
        print(
            f"{exp:<{width}}  {fold:<6}  {'improved' if improved else 'kept p1':<9}  "
            f"{phase1:>8.4f}  {phase2:>8.4f}  {phase2 - phase1:>+8.4f}"
        )

    n_improved = len(improved_deltas)
    print(f"\nfine-tuning improved fitness in {n_improved}/{len(rows)} runs "
          f"({100 * n_improved / len(rows):.0f}%)")
    if improved_deltas:
        print(f"  when it improved: mean gain {sum(improved_deltas) / n_improved:+.4f}, "
              f"max {max(improved_deltas):+.4f}")
    if kept_deltas:
        print(f"  when it did not:  mean change {sum(kept_deltas) / len(kept_deltas):+.4f}, "
              f"worst {min(kept_deltas):+.4f}")

    per_exp = {}
    for exp, _, improved, *_ in rows:
        hit, total = per_exp.get(exp, (0, 0))
        per_exp[exp] = (hit + improved, total + 1)
    print("\nper experiment:")
    for exp, (hit, total) in per_exp.items():
        print(f"  {exp:<{width}}  {hit}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
