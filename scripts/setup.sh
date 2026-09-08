#!/bin/bash
set -e
set -u

# Check positional argument for framework
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <framework> [--env=local|cluster]"
    exit 1
fi

FRAMEWORK="$1"
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
    uv run "${FRAMEWORK}/setup.py"
}

run_cluster() {
    job_name="${FRAMEWORK}-setup"
    echo "Submitting ${job_name}"
    sbatch --job-name="${job_name}" \
        --output="slurm_logs/${job_name}/%j.out" \
        --error="slurm_logs/${job_name}/%j.err" \
        ./scripts/setup.sbatch "${FRAMEWORK}" --env=${ENV}
}

if [[ "$ENV" == "local" ]]; then
    run_local
else
    run_cluster
fi