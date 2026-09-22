#!/bin/bash

# Check positional argument for framework
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <framework> <experiment> <fold> [--env=local|cluster] [-- <hydra overrides>...]"
    exit 1
fi

FRAMEWORK="$1"
shift

EXPERIMENT="$1"
shift

FOLD="$1"
shift

ENV="local"
# Extra Hydra overrides after `--`, e.g. `-- seed=3 experiment.name=exp_reference_3`
# to run a config under another name (see scripts/run_variance.py).
OVERRIDES=()

# Parse optional --env flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            OVERRIDES=("$@")
            break
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

# The run's own name: experiment.name if overridden, else the experiment config.
# Job and log names use it, so they match results/<framework>/<name>/.
RUN_NAME="${EXPERIMENT}"
for override in "${OVERRIDES[@]}"; do
    if [[ "${override}" == experiment.name=* ]]; then
        RUN_NAME="${override#experiment.name=}"
    fi
done

run_local() {
    echo "Running ${FRAMEWORK} experiment=${EXPERIMENT} fold=${FOLD} ${OVERRIDES[*]} locally"
    uv run "${FRAMEWORK}/run.py" experiment="${EXPERIMENT}" experiment.fold="${FOLD}" "${OVERRIDES[@]}"
}

run_cluster() {
    job_name="${FRAMEWORK}-${RUN_NAME}-fold_${FOLD}"
    echo "Submitting ${job_name}"
    sbatch --job-name="${job_name}" \
        --output="slurm_logs/${job_name}/%j.out" \
        --error="slurm_logs/${job_name}/%j.err" \
        ./scripts/run.sbatch "${FRAMEWORK}" "${EXPERIMENT}" "${FOLD}" --env=${ENV} -- "${OVERRIDES[@]}"
}

if [[ "${ENV}" == "local" ]]; then
    run_local
elif [[ "${ENV}" == "cluster" ]]; then
    run_cluster
else
    echo "Unknown environment: ${ENV}"
    exit 1
fi