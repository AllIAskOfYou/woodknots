import logging
import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.setup import (
    compose_with_hydra_config,
    experiment_overrides,
    get_experiment_names,
)
from mltools.yolo.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    stream=sys.stdout,
)


def main() -> int:
    framework_dir = Path(__file__).parent
    config_dir = str((framework_dir / "conf").resolve())

    experiment_names = (
        sys.argv[1].split() if len(sys.argv) > 1 and sys.argv[1] else get_experiment_names(framework_dir)
    )

    config_names = get_experiment_names(framework_dir)
    failed = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for exp_name in experiment_names:
            overrides = experiment_overrides(exp_name, config_names)
            if overrides is None:
                print(f"SKIPPED: {exp_name} has no experiment config")
                failed.append(f"{exp_name} (no config)")
                continue
            probe_cfg = compose_with_hydra_config([*overrides, "experiment.fold=0"])
            num_folds = probe_cfg.dataset.folds.k
            for fold in range(num_folds):
                cfg = compose_with_hydra_config([*overrides, f"experiment.fold={fold}"])
                print(f"Evaluating {exp_name} fold={fold}")
                try:
                    evaluate(cfg)
                except Exception as e:
                    print(f"FAILED: {exp_name} fold={fold}: {e}")
                    failed.append(f"{exp_name} fold={fold}")

    if failed:
        print(f"\n{len(failed)} evaluation(s) failed:")
        for entry in failed:
            print(f"  {entry}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
