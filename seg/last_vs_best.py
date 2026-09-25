"""Score a seg run's FINAL-epoch checkpoint against the one validation selected.

Everything reported so far uses the selected checkpoint, so the gap between the
two is the cost (or benefit) of checkpoint selection itself — which matters
here because val_dice hands a perfect score to every knot-free image and the
selected epoch varies widely between folds.

This needs checkpoints/final.ckpt, which only runs trained with
train.save_final_checkpoint=true have. ModelCheckpoint's last.ckpt is NOT a
substitute: Lightning writes it only on epochs that also wrote the best
checkpoint, so it is byte-identical to it and comparing the two measures
nothing but inference nondeterminism.

Re-predicts each fold's test split from last.ckpt and scores it with the same
evaluate_run() the pipeline uses, then prints both side by side. The run's own
predictions/ and eval_results.json are left untouched; last.ckpt predictions go
to <run_dir>/last_ckpt_predictions/.

Run it inside the container:

    uv run seg/last_vs_best.py
    uv run seg/last_vs_best.py --experiment exp_reference --folds 5
    uv run seg/last_vs_best.py -o article/seg_last_vs_best.json
"""
import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path

import torch
from hydra import initialize_config_dir

from mltools.common.evaluate import evaluate_run
from mltools.common.instances import make_instances, probs_to_instances, save_instances
from mltools.common.setup import compose_with_hydra_config
from mltools.common.utils import hash_cfg
from mltools.seg.dataset import (
    SETUP_IRRELEVANT_DATASET_KEYS,
    get_dataset_dir,
    load_case,
    write_fold_split,
)
from mltools.seg.seg_module import SegModule
from mltools.seg.transforms import INFERENCE_TRANSFORM, get_preprocessing_transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

CONFIG_DIR = str((Path(__file__).resolve().parent / "conf").resolve())


def predict_with_checkpoint(cfg, checkpoint: Path, out_dir: Path) -> None:
    """mltools.seg.train.predict, but from an explicit checkpoint into an
    explicit directory, and without the .pred_complete bookkeeping."""
    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    _, _, test_cases = write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold
    )

    device = torch.device(cfg.experiment.get("device", "cuda:0"))
    module = SegModule.load_from_checkpoint(str(checkpoint), weights_only=False)
    module.eval()
    module.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = get_preprocessing_transform(cfg.train.preprocessing)
    native_instances = hasattr(module.model, "predict_instances")
    for case_id in test_cases:
        image, label = load_case(dataset_dir, case_id)
        image = preprocessing(image=image, mask=label)["image"]
        image_t = INFERENCE_TRANSFORM(image=image)["image"].unsqueeze(0).to(device)
        if native_instances:
            masks, labels, scores = module.model.predict_instances(image_t)[0]
            instances = make_instances(
                masks.numpy(), labels.numpy(), scores.numpy(), shape=image_t.shape[-2:]
            )
        else:
            with torch.no_grad():
                probs = module.forward(image_t)[0].cpu().numpy()
            instances = probs_to_instances(probs)
        save_instances(out_dir / f"{case_id}.npz", instances)


def selected_epoch(checkpoint_dir: Path) -> int | None:
    """The epoch the best checkpoint's filename records, for context in the
    report — the metric name in it follows train.checkpoint_metric, so only
    the epoch is parsed."""
    best = sorted(checkpoint_dir.glob("best-model-*.ckpt"))
    if len(best) != 1:
        return None
    for part in best[0].stem.split("-"):
        if part.startswith("epoch="):
            return int(part.split("=")[1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment", default="exp_reference")
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
            final = run_dir / "checkpoints" / "final.ckpt"
            if not final.exists():
                logging.warning(
                    f"fold {fold}: no {final} — retrain with train.save_final_checkpoint=true "
                    f"(last.ckpt is a copy of the best checkpoint, not the final epoch), skipping"
                )
                continue

            best_results = run_dir / "eval_results.json"
            if not best_results.exists():
                logging.warning(f"fold {fold}: no {best_results}, skipping")
                continue

            out_dir = run_dir / "final_ckpt_predictions"
            if args.force or not any(out_dir.glob("*.npz")):
                logging.info(f"fold {fold}: predicting from {final}")
                predict_with_checkpoint(cfg, final, out_dir)
            else:
                logging.info(f"fold {fold}: reusing predictions in {out_dir}")

            per_fold[fold] = {
                "best": json.loads(best_results.read_text())["metrics"],
                "final": evaluate_run(cfg, out_dir)["metrics"],
                "selected_epoch": selected_epoch(run_dir / "checkpoints"),
                "total_epochs": cfg.train.total_epochs,
            }
            logging.info(f"fold {fold} final.ckpt: {per_fold[fold]['final']}")

    if not per_fold:
        print("nothing evaluated")
        return 1

    metrics = [m for m in next(iter(per_fold.values()))["best"] if m in next(iter(per_fold.values()))["final"]]
    epochs = [(f, v["selected_epoch"], v["total_epochs"]) for f, v in sorted(per_fold.items())]
    print(f"\nseg/{args.experiment}: best (selected) vs final-epoch checkpoint, {len(per_fold)} folds")
    print("  selected epoch per fold: " + ", ".join(
        f"fold {f}: {e}/{t - 1}" for f, e, t in epochs
    ))
    header = f"{'metric':<20}{'best':>10}{'final':>10}{'delta':>10}{'folds final better':>20}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        best = [per_fold[f]["best"][metric] for f in per_fold]
        final_ = [per_fold[f]["final"][metric] for f in per_fold]
        deltas = [f_ - b_ for b_, f_ in zip(best, final_)]
        better = sum(1 for d in deltas if d > 0)
        print(
            f"{metric:<20}{st.fmean(best):>10.4f}{st.fmean(final_):>10.4f}"
            f"{st.fmean(deltas):>+10.4f}{f'{better}/{len(deltas)}':>20}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"framework": "seg", "experiment": args.experiment,
             "per_fold": {str(f): v for f, v in sorted(per_fold.items())}},
            indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
