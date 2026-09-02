#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -e
set -u

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
    uv run "${FRAMEWORK}/evaluate.py" experiment="${EXPERIMENT}" experiment.fold="${FOLD}"
}

run_cluster() {
    local inner_cmd="uv run ${FRAMEWORK}/evaluate.py experiment=${EXPERIMENT} experiment.fold=${FOLD}"
    srun \
        --container-image=./containers/base.sqsh \
        --container-mounts="${PWD}:/workspace" \
        --container-workdir=/workspace \
        bash -c "${inner_cmd}"
}

if [[ "$ENV" == "local" ]]; then
    run_local
else
    run_cluster
fi
