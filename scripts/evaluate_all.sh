#!/bin/bash
set -e
set -u

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <framework> [<experiment>] [--env=local|cluster] [--force]"
    exit 1
fi

FRAMEWORK=$1
shift

EXPERIMENT=""

# Optional positional experiment filter (must come before any --flag)
if [[ $# -gt 0 && "$1" != --* ]]; then
    EXPERIMENT="$1"
    shift
fi

ENV="local"
FORCE=0

# Parse optional --env / --force flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --env=*)
            ENV="${1#*=}"
            shift
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done


FOLDS=(0 1 2 3 4)

if [[ -n "${EXPERIMENT}" ]]; then
    if [[ ! -d "results/${FRAMEWORK}" ]]; then
        echo "Experiment '${EXPERIMENT}' not found in results/${FRAMEWORK}"
        exit 1
    fi
    EXPERIMENTS=("${EXPERIMENT}")
else
    EXPERIMENTS=($(ls "results/${FRAMEWORK}"))
fi

echo "Found ${#EXPERIMENTS[@]} experiments for framework=${FRAMEWORK}: ${EXPERIMENTS[*]}"

# A fold is worth evaluating once it has predictions (.pred_complete) and its
# eval_results.json is missing or older than those predictions — re-predicting
# a run therefore re-evaluates it, instead of keeping results scored from
# predictions that no longer exist. --force re-evaluates every predicted fold
# (needed when the evaluator itself changes, e.g. a metric is added).
# Experiments with nothing to do are dropped — including results dirs that are
# not experiments at all, such as _old/.
TODO_EXPERIMENTS=()
TODO_PAIRS=()
skipped=0
for exp_name in "${EXPERIMENTS[@]}"; do
    exp_todo=0
    for fold in "${FOLDS[@]}"; do
        run_dir="results/${FRAMEWORK}/${exp_name}/fold_${fold}"
        if [[ ! -f "${run_dir}/.pred_complete" ]]; then
            continue
        fi
        if [[ ${FORCE} -eq 0 && "${run_dir}/eval_results.json" -nt "${run_dir}/.pred_complete" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
        TODO_PAIRS+=("${exp_name} ${fold}")
        exp_todo=1
    done
    if [[ ${exp_todo} -eq 1 ]]; then
        TODO_EXPERIMENTS+=("${exp_name}")
    fi
done

echo "${#TODO_PAIRS[@]} fold(s) to evaluate in ${#TODO_EXPERIMENTS[@]} experiment(s), ${skipped} already evaluated"
if [[ ${#TODO_PAIRS[@]} -eq 0 ]]; then
    echo "Nothing to evaluate."
    exit 0
fi

if [[ "${ENV}" == "cluster" ]]; then
    # All experiments/folds run sequentially inside a single sbatch job —
    # evaluation is cheap CPU-only scoring (no GPU inference), so submitting
    # one job per fold is pure SLURM scheduling/container-startup overhead.
    # The job evaluates every fold of the experiments passed to it, so an
    # experiment with one pending fold has its other folds re-scored too.
    job_name="${FRAMEWORK}-evaluate_all"
    echo "Submitting ${job_name} (${#TODO_EXPERIMENTS[@]} experiments as one job)"
    sbatch --job-name="${job_name}" \
        --output="slurm_logs/${job_name}/%j.out" \
        --error="slurm_logs/${job_name}/%j.err" \
        ./scripts/evaluate_all.sbatch "${FRAMEWORK}" "${TODO_EXPERIMENTS[*]}"
    exit 0
fi

FAILED=()

for pair in "${TODO_PAIRS[@]}"; do
    read -r exp_name fold <<< "${pair}"
    if ! ./scripts/evaluate.sh "${FRAMEWORK}" "${exp_name}" "${fold}" --env=${ENV}; then
        echo "FAILED: ${exp_name} fold=${fold}"
        FAILED+=("${exp_name} fold=${fold}")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "${#FAILED[@]} evaluation(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
