"""Score an nnU-Net run's final checkpoint against the one it selected.

nnU-Net keeps two checkpoints per fold: checkpoint_best.pth, chosen by the EMA
of the validation pseudo-dice, and checkpoint_final.pth from the last epoch. The
pipeline predicts from whichever mltools.nnunet.train._checkpoint() selects, so
the gap between them is what checkpoint selection is worth here.

Worth measuring because the training curves suggest it should be large — the
EMA pseudo-dice falls ~37% from its peak and validation loss more than doubles
— while the same comparison on YOLO turned out to be worth nothing, its
apparent decline being the maximum of a noisy plateau rather than real decay.
Only held-out test data settles which of those nnU-Net is.

No training is involved: nnUNetv2_predict is re-run over each fold's test
partition with -chk set to the unused checkpoint, written to
<run_dir>/final_predictions (raw output to <run_dir>/nnunet_final_predictions),
then scored with the same evaluate_run() the pipeline uses. The reported side is
read from the existing eval_results.json. The run's own predictions/ and
eval_results.json are left untouched.

    uv run nnunet/last_vs_best.py
    uv run nnunet/last_vs_best.py --experiment exp_reference_0 -o article/nnunet_last_vs_best.json
"""
import argparse
import json
import logging
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
from hydra import initialize_config_dir

from mltools.common.evaluate import evaluate_run
from mltools.common.instances import probs_to_instances, save_instances
from mltools.common.setup import compose_with_hydra_config
from mltools.common.splits import get_split
from mltools.common.utils import run_command, symlink_relative
from mltools.nnunet.train import CHECKPOINTS, _checkpoint, _fold_dir, _split_mode
from mltools.nnunet.utils import build_nnunet_env

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = str((Path(__file__).resolve().parent / "conf").resolve())


def other_checkpoint(cfg) -> str:
    """The checkpoint the run did NOT predict from. Under train_test that is
    checkpoint_best.pth, selected on the test partition — a measure of what such
    a selection would be worth, not a number to report."""
    used = _checkpoint(cfg)
    return next(name for name in CHECKPOINTS.values() if name != used)


def predict_with_checkpoint(cfg, checkpoint: str, raw_dir: Path, out_dir: Path) -> None:
    """mltools.nnunet.train.predict, but from a named checkpoint into explicit
    directories and without the .pred_complete bookkeeping."""
    fold = cfg.experiment.fold
    dataset_name = f"Dataset{cfg.dataset.id:03d}_{cfg.dataset.name}"
    test_cases = get_split(cfg.dataset.folds.file_path, fold, mode=_split_mode(cfg))["test"]

    # nnUNetv2_predict takes a folder, so gather the test images into one.
    images_tr_dir = Path(cfg.paths.nnunet.raw) / dataset_name / "imagesTr"
    input_dir = Path(cfg.paths.run_dir) / "test_images"
    input_dir.mkdir(parents=True, exist_ok=True)
    for case_id in test_cases:
        symlink_relative(input_dir / f"{case_id}_0000.png", images_tr_dir / f"{case_id}_0000.png")

    run_command(
        [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(raw_dir),
            "-d", f"{cfg.dataset.id:03d}",
            "-c", "2d",
            "-f", str(fold),
            "-tr", cfg.experiment.get("trainer", "nnUNetTrainer"),
            "-p", cfg.experiment.get("plans", "nnUNetPlans"),
            "-chk", checkpoint,
            "-device", cfg.experiment.get("device", "cuda"),
            "--save_probabilities",
        ],
        env=build_nnunet_env(
            nnunet_raw=str(cfg.paths.nnunet.raw),
            nnunet_preprocessed=str(cfg.paths.nnunet.preprocessed),
            nnunet_results=str(cfg.paths.nnunet.results),
        ),
        logger=logger,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for case_id in test_cases:
        # (num_classes + 1, [1,] H, W) softmax, background first; 2D natural
        # images carry a singleton through-plane axis.
        with np.load(raw_dir / f"{case_id}.npz") as data:
            probs = data["probabilities"][1:].astype(np.float32)
        if probs.ndim == 4:
            probs = probs[:, 0]
        save_instances(out_dir / f"{case_id}.npz", probs_to_instances(probs))


def selected_epoch(fold_dir: Path) -> tuple[int | None, int | None]:
    """(epoch of the best EMA pseudo-dice, epochs trained), read from nnU-Net's
    own training log — the checkpoint it selects on."""
    logs = sorted(fold_dir.glob("training_log_*.txt"))
    if not logs:
        return None, None
    dices = []
    for line in logs[-1].read_text().splitlines():
        if m := re.search(r"Pseudo dice \[np\.float32\(([\d.]+)\)\]", line):
            dices.append(float(m.group(1)))
    if not dices:
        return None, None
    return max(range(len(dices)), key=lambda i: dices[i]), len(dices)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment", default="exp_reference_0")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument(
        "--force", action="store_true", help="re-predict folds already done in this run dir"
    )
    ap.add_argument("-o", "--output", default=None, help="write the per-fold numbers as JSON")
    args = ap.parse_args()

    per_fold: dict[int, dict] = {}
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        probe = compose_with_hydra_config(
            [f"experiment={args.experiment}", "experiment.fold=0"]
        )
        num_folds = args.folds if args.folds is not None else probe.dataset.folds.k

        for fold in range(num_folds):
            cfg = compose_with_hydra_config(
                [f"experiment={args.experiment}", f"experiment.fold={fold}"]
            )
            run_dir = Path(cfg.paths.run_dir)
            fold_dir = _fold_dir(cfg)
            checkpoint = other_checkpoint(cfg)
            final = fold_dir / checkpoint
            if not final.exists():
                logging.warning(f"fold {fold}: no {final}, skipping")
                continue

            best_results = run_dir / "eval_results.json"
            if not best_results.exists():
                logging.warning(f"fold {fold}: no {best_results}, skipping")
                continue

            out_dir = run_dir / "final_predictions"
            if args.force or not any(out_dir.glob("*.npz")):
                logging.info(f"fold {fold}: predicting from {checkpoint}")
                predict_with_checkpoint(
                    cfg, checkpoint, run_dir / "nnunet_final_predictions", out_dir
                )
            else:
                logging.info(f"fold {fold}: reusing predictions in {out_dir}")

            best_epoch, total_epochs = selected_epoch(fold_dir)
            per_fold[fold] = {
                "best": json.loads(best_results.read_text())["metrics"],
                "final": evaluate_run(cfg, out_dir)["metrics"],
                "selected_epoch": best_epoch,
                "total_epochs": total_epochs,
                "reported_checkpoint": _checkpoint(cfg),
                "other_checkpoint": checkpoint,
            }
            logging.info(f"fold {fold} {checkpoint}: {per_fold[fold]['final']}")

    if not per_fold:
        print("nothing evaluated")
        return 1

    first = next(iter(per_fold.values()))
    metrics = [m for m in first["best"] if m in first["final"]]
    reported = first["reported_checkpoint"].removeprefix("checkpoint_").removesuffix(".pth")
    other = first["other_checkpoint"].removeprefix("checkpoint_").removesuffix(".pth")
    print(f"\nnnunet/{args.experiment}: reported {reported} vs {other} checkpoint, "
          f"{len(per_fold)} folds")
    print("  selected epoch per fold: " + ", ".join(
        f"f{f}:{v['selected_epoch']}/{v['total_epochs']}" for f, v in sorted(per_fold.items())
    ))
    header = f"{'metric':<26}{reported:>10}{other:>10}{'delta':>10}{f'folds {other} better':>20}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        best = [per_fold[f]["best"][metric] for f in per_fold]
        final_ = [per_fold[f]["final"][metric] for f in per_fold]
        deltas = [f_ - b_ for b_, f_ in zip(best, final_)]
        better = sum(1 for d in deltas if d > 0)
        print(
            f"{metric:<26}{st.fmean(best):>10.4f}{st.fmean(final_):>10.4f}"
            f"{st.fmean(deltas):>+10.4f}{f'{better}/{len(deltas)}':>20}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"framework": "nnunet", "experiment": args.experiment,
             "per_fold": {str(f): v for f, v in sorted(per_fold.items())}},
            indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
