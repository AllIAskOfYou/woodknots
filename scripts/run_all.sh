#!/bin/bash
set -e
set -u

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <framework> [--env=local|cluster]"
    exit 1
fi

FRAMEWORK=$1
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


FOLDS=(0 1 2 3 4)

EXPERIMENTS=($(ls "${FRAMEWORK}/conf/experiment" | sed 's/\.yaml$//'))

echo "Found ${#EXPERIMENTS[@]} experiments for framework=${FRAMEWORK}: ${EXPERIMENTS[*]}"

for exp_name in "${EXPERIMENTS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        if [[ "${ENV}" == "local" ]]; then
            echo "Running ${FRAMEWORK} experiment=${exp_name} fold=${fold} locally"
            ./scripts/run.sh "${FRAMEWORK}" "${exp_name}" "${fold}" --env=${ENV}
            continue
        fi
        if [[ "${ENV}" == "cluster" ]]; then
            job_name="${FRAMEWORK}-${exp_name}-fold_${fold}"
            echo "Submitting ${job_name}"

            sbatch --job-name="${job_name}" \
                --output="slurm_logs/${job_name}_%j.out" \
                --error="slurm_logs/${job_name}_%j.err" \
                ./scripts/run.sh "${FRAMEWORK}" "${exp_name}" "${fold}" --env=${ENV}
            continue
        else
            echo "Unknown environment: ${ENV}"
            exit 1
        fi
    done
done