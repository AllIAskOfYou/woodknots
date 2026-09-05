#!/bin/bash

# Check positional argument for framework
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <framework> <experiment> <fold> [--env=local|cluster]"
    exit 1
fi

FRAMEWORK="$1"
shift

EXPERIMENT="$1"
shift

FOLD="$1"
shift

ENV="local"

# Parse optional --env flag
while [[ $# -gt 0 ]]; do
    case "$1" in
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

run_local() {
    echo "Evaluating ${FRAMEWORK} experiment=${EXPERIMENT} fold=${FOLD} locally"
    uv run "${FRAMEWORK}/evaluate.py" experiment="${EXPERIMENT}" experiment.fold="${FOLD}"
}

run_cluster() {
    job_name="${FRAMEWORK}-${EXPERIMENT}-fold_${FOLD}"
    echo "Submitting ${job_name}"
    sbatch --job-name="${job_name}" \
        --output="slurm_logs/${job_name}/%j.out" \
        --error="slurm_logs/${job_name}/%j.err" \
        ./scripts/evaluate.sbatch "${FRAMEWORK}" "${EXPERIMENT}" "${FOLD}" --env=${ENV}
}

if [[ "${ENV}" == "local" ]]; then
    run_local
elif [[ "${ENV}" == "cluster" ]]; then
    run_cluster
else
    echo "Unknown environment: ${ENV}"
    exit 1
fi
