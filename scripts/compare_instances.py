"""Does a method's own instance output matter, compared with connected components?

For each given experiment and fold, scores GMS twice against the same ground
truth as the evaluation (mltools.common.evaluate):
  gms_stored  — on the stored instances (what the evaluation reports as gms);
  gms_cc      — on the 8-connected components of their union, i.e. what the
                same prediction would score as a semantic output.
For semantic methods the two are identical by construction; for
instance-native ones (YOLO instance, Mask2Former) the difference is what
predicting instances adds. Connected components are scored 1.0 — irrelevant
for greedy matching at IoU >= 0.5, where each match is unique.

Needs the venv (run inside the container):

    uv run scripts/compare_instances.py yolo exp_reference exp_rot_0
    uv run scripts/compare_instances.py seg exp_decoder_2 exp_decoder_3 -o article/m2f_instances.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.instances import (
    instances_to_label_map,
    label_map_to_instances,
    load_gt_instances,
    load_instances,
)
from mltools.common.metrics import instance_gms
from mltools.common.setup import compose_with_hydra_config, get_annotations_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compare_run(pred_dir: Path, gt: dict, num_classes: int) -> dict[str, float]:
    stored, cc = [], []
    for path in sorted(pred_dir.glob("*.npz")):
        pred, target = load_instances(path), gt[path.stem]
        stored.append(instance_gms(pred, target, num_classes))
        cc.append(instance_gms(label_map_to_instances(instances_to_label_map(pred)), target, num_classes))
    if not stored:
        raise FileNotFoundError(f"no predictions in {pred_dir}")
    return {"gms_stored": st.fmean(stored), "gms_cc": st.fmean(cc), "cases": len(stored)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("-o", "--output", default=None, help="write the per-fold numbers as JSON")
    args = ap.parse_args()

    config_dir = str((PROJECT_ROOT / args.framework / "conf").resolve())
    results: dict[str, dict[int, dict]] = {}
    gt_cache: dict[str, dict] = {}
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for exp in args.experiments:
            probe = compose_with_hydra_config([f"experiment={exp}", "experiment.fold=0"])
            results[exp] = {}
            for fold in range(probe.dataset.folds.k):
                cfg = compose_with_hydra_config([f"experiment={exp}", f"experiment.fold={fold}"])
                annotations = get_annotations_path(cfg.dataset)
                classes, confidences = list(cfg.dataset.class_names), list(cfg.dataset.eval_confidences)
                key = f"{annotations}|{classes}|{confidences}"
                if key not in gt_cache:
                    gt_cache[key] = load_gt_instances(annotations, classes, confidences)
                pred_dir = Path(cfg.paths.run_dir) / "predictions"
                results[exp][fold] = compare_run(pred_dir, gt_cache[key], len(classes))

    print(f"{'experiment':<22} {'gms_stored':>11} {'gms_cc':>11} {'stored - cc':>12}  (mean over folds)")
    for exp, folds in results.items():
        stored = st.fmean(r["gms_stored"] for r in folds.values())
        cc = st.fmean(r["gms_cc"] for r in folds.values())
        diffs = [r["gms_stored"] - r["gms_cc"] for r in folds.values()]
        spread = f" ±{st.stdev(diffs):.4f}" if len(diffs) > 1 else ""
        print(f"{exp:<22} {stored:>11.4f} {cc:>11.4f} {stored - cc:>+12.4f}{spread}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"framework": args.framework, "per_fold": results}, indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
