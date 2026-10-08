"""Score a test-time ensemble of seg runs trained at different resolutions.

Each run re-predicts its fold's test images at its own resolution (sliding
window, from the checkpoint the pipeline uses), its probability map is resized
to the base run's image size, and the maps are averaged. The base run (the
first --experiments entry) sets the resolution the ground truth is scored at.
Each member is also scored alone through the same path, so the comparison is
like for like.

Probabilities are cached at each run's own resolution under
results/ensembles/seg/probs/native/<experiment>/fold_<i>/ and resized when
combined, so one cache serves every ensemble and every scoring resolution, and
re-scoring other combinations or --weights skips the GPU.
Seed runs (<config>_<i>, scripts/run_variance.py) can be members.

Members scored at their own resolution are checked against their own
eval_results.json; a mismatch means their stored predictions don't come from
the checkpoint this script loads.

    uv run seg/ensemble_scales.py
    uv run seg/ensemble_scales.py --experiments exp_reference exp_res_1 --weights 0.5 0.5
    uv run seg/ensemble_scales.py --experiments exp_reference_{0..4} exp_union --weights 1 1 1 1 1 5
"""
import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import initialize_config_dir

from mltools.common.evaluate import evaluate_run
from mltools.common.instances import probs_to_instances, save_instances
from mltools.common.setup import compose_with_hydra_config, experiment_overrides, get_experiment_names
from mltools.common.splits import TRAIN_TEST
from mltools.common.utils import hash_cfg
from mltools.seg.dataset import (
    SETUP_IRRELEVANT_DATASET_KEYS,
    get_dataset_dir,
    load_case,
    write_fold_split,
)
from mltools.seg.seg_module import SegModule
from mltools.seg.train import FINAL_CHECKPOINT_NAME, _checkpoint_dir, _split_mode
from mltools.seg.transforms import INFERENCE_TRANSFORM, get_preprocessing_transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

CONFIG_DIR = str((Path(__file__).resolve().parent / "conf").resolve())
ENSEMBLE_ROOT = Path("results/ensembles/seg")
SANITY_TOLERANCE = 0.005  # mAP50 / dice_fg_pooled difference that counts as a mismatch


def _dataset_dir(cfg) -> str:
    return get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )


def _checkpoint(cfg) -> Path:
    """The checkpoint mltools.seg.train.predict uses for this run."""
    if _split_mode(cfg) == TRAIN_TEST:
        return _checkpoint_dir(cfg) / FINAL_CHECKPOINT_NAME
    best = sorted(_checkpoint_dir(cfg).glob("best-model-*.ckpt"))
    if len(best) != 1:
        raise RuntimeError(f"expected one best checkpoint in {_checkpoint_dir(cfg)}, found {len(best)}")
    return best[0]


def predict_probs(cfg, test_cases, cache_dir: Path, force: bool) -> None:
    """Foreground probabilities (H, W) per test case at the run's own
    resolution, cached as float16."""
    if not force and all((cache_dir / f"{c}.npz").exists() for c in test_cases):
        logging.info(f"reusing cached probabilities in {cache_dir}")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = _dataset_dir(cfg)
    checkpoint = _checkpoint(cfg)
    logging.info(f"predicting {len(test_cases)} cases from {checkpoint}")

    device = torch.device(cfg.experiment.get("device", "cuda:0"))
    module = SegModule.load_from_checkpoint(str(checkpoint), weights_only=False)
    if hasattr(module.model, "predict_instances"):
        raise NotImplementedError("instance-native models (Mask2Former) have no probability map to average")
    module.eval()
    module.to(device)
    preprocessing = get_preprocessing_transform(cfg.train.preprocessing)

    for case_id in test_cases:
        image, label = load_case(dataset_dir, case_id)
        image = preprocessing(image=image, mask=label)["image"]
        image_t = INFERENCE_TRANSFORM(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = module.forward(image_t)[0, 0].float().cpu().numpy()
        np.savez_compressed(cache_dir / f"{case_id}.npz", probs=probs.astype(np.float16))

    del module
    torch.cuda.empty_cache()


def load_probs(prob_dir: Path, case_id: str, shape: tuple[int, int]) -> np.ndarray:
    """Cached probabilities, bilinearly resized to shape (H, W) if needed."""
    probs = np.load(prob_dir / f"{case_id}.npz")["probs"].astype(np.float32)
    if probs.shape != tuple(shape):
        probs = cv2.resize(probs, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return probs


def write_instances(prob_dirs, weights, test_cases, out_dir: Path, shapes: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_id in test_cases:
        fused = sum(
            w * load_probs(d, case_id, shapes[case_id]) for d, w in zip(prob_dirs, weights)
        ) / sum(weights)
        save_instances(out_dir / f"{case_id}.npz", probs_to_instances(fused[None]))


def run_ensembles(
    names: list[str], columns: dict[str, list[tuple[str, float]]], root: Path,
    folds: int | None = None, force: bool = False, score_at: str | None = None,
) -> tuple[dict[int, dict], dict[str, int]]:
    """Score each column — a weighted average of member probabilities — on
    every fold, at the resolution of score_at (default names[0]), against that
    resolution's ground truth. Returns
    ({fold: {column: metrics}}, {member: its own training resolution})."""
    score_at = score_at or names[0]
    config_names = get_experiment_names(Path(__file__).resolve().parent)
    overrides = {}
    for n in names:
        overrides[n] = experiment_overrides(n, config_names)
        if overrides[n] is None:
            raise SystemExit(f"{n}: no experiment config (nor a base config for a seed run)")

    per_fold: dict[int, dict] = {}
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        probe = compose_with_hydra_config([*overrides[names[0]], "experiment.fold=0"])
        num_folds = folds if folds is not None else probe.dataset.folds.k

        for fold in range(num_folds):
            cfgs = {
                n: compose_with_hydra_config([*overrides[n], f"experiment.fold={fold}"]) for n in names
            }
            base = cfgs[score_at]
            resolutions = {n: c.dataset.resolution for n, c in cfgs.items()}
            base_dir = _dataset_dir(base)
            _, _, test_cases = write_fold_split(base_dir, base.dataset.folds.file_path, fold)
            base_shapes = {c: load_case(base_dir, c)[0].shape[:2] for c in test_cases}

            prob_dirs = {}
            for name, cfg in cfgs.items():
                cache = ENSEMBLE_ROOT / "probs" / "native" / name / f"fold_{fold}"
                predict_probs(cfg, test_cases, cache, force)
                prob_dirs[name] = cache

            per_fold[fold] = {}
            for column, members in columns.items():
                pred_dir = root / f"res_{base.dataset.resolution}" / "predictions" / column / f"fold_{fold}"
                write_instances(
                    [prob_dirs[m] for m, _ in members], [w for _, w in members], test_cases, pred_dir,
                    base_shapes,
                )
                per_fold[fold][column] = evaluate_run(base, pred_dir)["metrics"]
            logging.info(f"fold {fold} done")
    return per_fold, resolutions


def check_members(names, per_fold, resolutions, score_at: str | None = None) -> None:
    """Warn about members, scored at their own resolution, that do not
    reproduce their own eval_results.json."""
    score_at = score_at or names[0]
    mismatches = []
    for name in names:
        if resolutions[name] != resolutions[score_at]:
            continue
        for f in sorted(per_fold):
            own = Path("results/seg") / name / f"fold_{f}" / "eval_results.json"
            if not own.exists():
                continue
            stored = json.loads(own.read_text())["metrics"]
            for m in ("mAP50", "dice_fg_pooled"):
                if abs(stored[m] - per_fold[f][name][m]) > SANITY_TOLERANCE:
                    mismatches.append(f"{name} fold {f} {m}: stored {stored[m]:.4f}, here {per_fold[f][name][m]:.4f}")
    if mismatches:
        print("\nWARNING: members that do not reproduce their own eval_results.json:")
        for line in mismatches:
            print("  " + line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["exp_reference", "exp_res_1"],
                    help="first one is the base: its resolution is the scoring resolution")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="default: equal")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument("--output-dir", default=None,
                    help="default: results/ensembles/seg/<experiments joined by +>")
    ap.add_argument("--force", action="store_true", help="re-predict cached probabilities")
    ap.add_argument("-o", "--output", default=None, help="write the per-fold numbers as JSON")
    args = ap.parse_args()

    names = args.experiments
    if len(names) < 2:
        raise SystemExit("need at least two experiments")
    weights = args.weights or [1.0] * len(names)
    if len(weights) != len(names):
        raise SystemExit("--weights needs one value per experiment")
    root = Path(args.output_dir or ENSEMBLE_ROOT / "+".join(names))
    columns = {n: [(n, 1.0)] for n in names} | {"ensemble": list(zip(names, weights))}

    per_fold, resolutions = run_ensembles(names, columns, root, args.folds, args.force)
    check_members(names, per_fold, resolutions)

    metrics = list(per_fold[next(iter(per_fold))]["ensemble"])
    folds = sorted(per_fold)

    print(f"\nseg ensemble of {' + '.join(names)} (weights {weights}), "
          f"scored at {names[0]}'s resolution, {len(folds)} folds")
    width = max(len(m) for m in metrics)
    col = max(10, *(len(c) for c in columns)) + 1
    header = (f"{'metric':<{width}}" + "".join(f"{c:>{col}}" for c in columns)
              + f"{'ens - best':>12}{'beats all':>11}")
    print(header)
    print("-" * len(header))
    for m in metrics:
        means = {c: st.fmean(per_fold[f][c][m] for f in folds) for c in columns}
        best_member = max(names, key=lambda n: means[n])
        beats = sum(all(per_fold[f]["ensemble"][m] > per_fold[f][n][m] for n in names) for f in folds)
        print(f"{m:<{width}}" + "".join(f"{means[c]:>{col}.4f}" for c in columns)
              + f"{means['ensemble'] - means[best_member]:>+12.4f}{f'{beats}/{len(folds)}':>11}")
    print("ens - best: ensemble minus the best single member on that metric")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"experiments": names, "weights": weights,
             "per_fold": {str(f): v for f, v in per_fold.items()}}, indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
