import argparse
import json
from mltools.common.data import get_folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data splits for cross-validation"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        help="Method for splitting (random or greedy)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the generated splits (default: data_dir/splits.json)",
    )
    args = parser.parse_args()

    splits = get_folds(args.data_dir, k=args.k, method=args.method, seed=args.seed)
    if args.output_path is None:
        args.output_path = f"{args.data_dir}/splits.json"
    with open(args.output_path, "w") as f:
        json.dump(splits, f, indent=4)
