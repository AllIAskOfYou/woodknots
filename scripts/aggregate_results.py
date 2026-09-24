import json
import statistics
import sys
from pathlib import Path


def find_experiments(framework: str) -> list[str]:
    base = Path("results", framework)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def aggregate(framework: str, experiment: str) -> dict:
    result_files = sorted(Path("results", framework, experiment).glob("fold_*/eval_results.json"))

    if not result_files:
        raise RuntimeError(
            f"No eval_results.json found under results/{framework}/{experiment}/fold_*/"
        )

    per_fold = []
    for path in result_files:
        data = json.loads(path.read_text())
        per_fold.append({"fold": data["fold"], "metrics": data["metrics"]})

    metric_names = list(per_fold[0]["metrics"].keys())
    aggregated = {}
    for name in metric_names:
        values = []
        for entry in per_fold:
            if name not in entry["metrics"]:
                print(f"WARNING: fold {entry['fold']} is missing metric '{name}', skipping it for this metric.")
                continue
            values.append(entry["metrics"][name])
        aggregated[name] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    return {
        "framework": framework,
        "experiment": experiment,
        "folds_found": sorted(entry["fold"] for entry in per_fold),
        "metrics": aggregated,
        "per_fold": sorted(per_fold, key=lambda entry: entry["fold"]),
    }


def print_summary(summary: dict) -> None:
    print(f"{summary['framework']}/{summary['experiment']} — folds found: {summary['folds_found']}")
    width = max([12, *(len(name) for name in summary["metrics"])])
    header = f"{'metric':<{width}} {'mean':>10} {'std':>10} {'n':>4}"
    print(header)
    print("-" * len(header))
    for name, stats in summary["metrics"].items():
        mean_str = f"{stats['mean']:.4f}" if stats["mean"] is not None else "n/a"
        print(f"{name:<{width}} {mean_str:>10} {stats['std']:>10.4f} {stats['n']:>4}")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <framework> [<experiment>]")
        return 1

    framework = sys.argv[1]
    experiments = [sys.argv[2]] if len(sys.argv) == 3 else find_experiments(framework)

    if not experiments:
        print(f"No experiments found under results/{framework}/")
        return 1

    exit_code = 0
    for i, experiment in enumerate(experiments):
        if i > 0:
            print()
        try:
            summary = aggregate(framework, experiment)
        except RuntimeError as e:
            print(f"WARNING: {e}")
            exit_code = 1
            continue

        print_summary(summary)

        out_path = Path("results", framework, experiment, "summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {out_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
