"""Two-model ensembles of two configs, paired by seed index.

Seed run <base>_<i> is averaged 1:1 with seed run <partner>_<i> (both from
scripts/run_variance.py), giving one ensemble per index. The spread across those
ensembles gives a mean ± sd for the ensemble, like every other row, and the
paired differences (ensemble_i - base_i, ensemble_i - partner_i) test the gain
against seed noise. The pairing by index is arbitrary (the seeds of the two
configs are independent), so the sd covers the variation of both.

Scored twice, at the base's and at the partner's resolution (each against
that resolution's ground truth), through seg/ensemble_scales.py's shared
probability cache, so runs already cached need no GPU.

    uv run seg/ensemble_pairs.py
    uv run seg/ensemble_pairs.py --base exp_reference --partner exp_union --seeds 5
    python3 seg/ensemble_pairs.py --report-only        # tables again from the saved .json

Besides the summary table, writes a per-fold table (<output>_per_fold.txt): for
each metric, one row per model and per difference, one column per fold, each
cell the mean ± sd across the seed runs / pairs.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ENSEMBLE_ROOT = Path("results/ensembles/seg")


def mean_sd(values: list[float]) -> str:
    return f"{st.fmean(values):.4f} ± {st.stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def table(per_fold, args, seeds, partners, columns, resolution) -> str:
    folds = sorted(per_fold)
    metrics = list(per_fold[folds[0]]["pair_0"])
    # 5-fold mean of every column: one number per seed run, per pair and for the partner.
    score = {c: {m: st.fmean(per_fold[f][c][m] for f in folds) for m in metrics} for c in columns}

    def paired(diffs):
        sd = st.stdev(diffs)
        t = st.fmean(diffs) / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
        return [f"{st.fmean(diffs):+.4f} ± {sd:.4f}", f"{sum(d > 0 for d in diffs)}/{len(diffs)}", f"{t:+.1f}"]

    rows = []
    for m in metrics:
        base_scores = [score[s][m] for s in seeds]
        partner_scores = [score[p][m] for p in partners]
        pair_scores = [score[f"pair_{i}"][m] for i in range(len(seeds))]
        rows.append((m, mean_sd(base_scores), mean_sd(partner_scores), mean_sd(pair_scores),
                     *paired([e - b for e, b in zip(pair_scores, base_scores)]),
                     *paired([e - p for e, p in zip(pair_scores, partner_scores)])))

    head = ("metric", args.base, args.partner, "pair ensemble",
            f"pair - {args.base}", "up", "t", f"pair - {args.partner}", "up", "t")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    line = lambda cells: "  ".join(
        f"{c:<{w}}" if i == 0 else f"{c:>{w}}" for i, (c, w) in enumerate(zip(cells, widths))
    )
    text = [
        f"seg: {args.base}_i averaged 1:1 with {args.partner}_i for i = 0..{len(seeds) - 1}, "
        f"{len(folds)} folds, scored at resolution {resolution}",
        f"mean ± sd across the {len(seeds)} seed runs / pairs (each value a {len(folds)}-fold mean)",
        f"differences paired by seed index; t = mean / (sd / sqrt({len(seeds)})), {len(seeds) - 1} df",
        "",
        line(head),
        "-" * (sum(widths) + 2 * (len(widths) - 1)),
        *(line(r) for r in rows),
    ]
    return "\n".join(text) + "\n"


def per_fold_table(per_fold, args, seeds, partners, resolution) -> str:
    folds = sorted(per_fold)
    metrics = list(per_fold[folds[0]]["pair_0"])
    pairs = [f"pair_{i}" for i in range(len(seeds))]

    def cell(values, signed=False):
        mean = f"{st.fmean(values):+.4f}" if signed else f"{st.fmean(values):.4f}"
        return f"{mean} ± {st.stdev(values):.4f}" if len(values) > 1 else mean

    def series(columns, minus=None):
        """Per fold, then the fold mean: values across seeds of columns[i] (minus minus[i])."""
        def at(fold):
            return [per_fold[fold][c][m] - (per_fold[fold][d][m] if minus else 0.0)
                    for c, d in zip(columns, minus or columns)]
        per = [at(f) for f in folds]
        overall = [st.fmean(v[i] for v in per) for i in range(len(columns))]
        return [cell(v, minus is not None) for v in per] + [cell(overall, minus is not None)]

    labels = [args.base, args.partner, "pair ensemble", f"pair - {args.base}", f"pair - {args.partner}"]
    head = ["", *(f"fold {f}" for f in folds), "mean"]
    blocks = []
    for m in metrics:
        rows = [
            [labels[0], *series(seeds)],
            [labels[1], *series(partners)],
            [labels[2], *series(pairs)],
            [labels[3], *series(pairs, seeds)],
            [labels[4], *series(pairs, partners)],
        ]
        blocks.append((m, rows))
    widths = [max(len(head[i]), *(len(r[i]) for _, rows in blocks for r in rows)) for i in range(len(head))]
    line = lambda cells: "  ".join(
        f"{c:<{w}}" if i == 0 else f"{c:>{w}}" for i, (c, w) in enumerate(zip(cells, widths))
    )
    text = [
        f"seg: {args.base}_i + {args.partner}_i per fold, scored at resolution {resolution}",
        f"each cell: mean ± sd across the {len(seeds)} seed runs / pairs; "
        f"the mean column averages the folds first, then takes mean ± sd across seeds",
    ]
    for m, rows in blocks:
        text += ["", f"=== {m}", line(head), *(line(r) for r in rows)]
    return "\n".join(text) + "\n"


def write_reports(per_res, args, seeds, partners, columns, out: Path) -> None:
    summary = "\n\n".join(
        table(pf, args, seeds, partners, columns, r) for r, pf in per_res.items()
    )
    detail = "\n\n".join(per_fold_table(pf, args, seeds, partners, r) for r, pf in per_res.items())
    print("\n" + summary, end="")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary)
    per_fold_out = out.with_name(out.stem + "_per_fold.txt")
    per_fold_out.write_text(detail)
    print(f"\nSaved to {out} and {per_fold_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="exp_reference", help="seed runs are <base>_0 .. <base>_<seeds-1>")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--partner", default="exp_union", help="seed runs are <partner>_0 .. <partner>_<seeds-1>")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument("--force", action="store_true", help="re-predict cached probabilities")
    ap.add_argument("--report-only", action="store_true",
                    help="no evaluation: rebuild the tables from the saved .json")
    ap.add_argument("-o", "--output", default=None,
                    help="default: results/ensembles/seg/pairs_<base>+<partner>.txt (+ .json)")
    args = ap.parse_args()

    seeds = [f"{args.base}_{i}" for i in range(args.seeds)]
    partners = [f"{args.partner}_{i}" for i in range(args.seeds)]
    names = [*seeds, *partners]
    pairs = {f"pair_{i}": [(s, 1.0), (p, 1.0)] for i, (s, p) in enumerate(zip(seeds, partners))}
    columns = {n: [(n, 1.0)] for n in names} | pairs
    root = ENSEMBLE_ROOT / f"pairs_{args.base}+{args.partner}"

    out = Path(args.output or ENSEMBLE_ROOT / f"pairs_{args.base}+{args.partner}.txt")
    if args.report_only:
        saved = json.loads(out.with_suffix(".json").read_text())
        # Older files hold one resolution under "per_fold".
        raw = saved.get("per_resolution") or {"?": saved["per_fold"]}
        per_res = {r: {int(f): v for f, v in pf.items()} for r, pf in raw.items()}
        write_reports(per_res, args, seeds, partners, columns, out)
        return 0

    from ensemble_scales import check_members, run_ensembles

    per_res = {}
    for k, score_at in enumerate((seeds[0], partners[0])):
        # --force re-predicts once; the second pass reads the fresh cache.
        force = args.force and k == 0
        per_fold, resolutions = run_ensembles(names, columns, root, args.folds, force, score_at)
        if resolutions[score_at] in per_res:  # both configs at the same resolution: score once
            continue
        check_members(names, per_fold, resolutions, score_at)
        per_res[resolutions[score_at]] = per_fold

    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"base": args.base, "seeds": seeds, "partner": args.partner, "partners": partners,
         "per_resolution": {str(r): {str(f): v for f, v in pf.items()} for r, pf in per_res.items()}},
        indent=2) + "\n")
    write_reports(per_res, args, seeds, partners, columns, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
