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
    local inner_cmd="uv run ${FRAMEWORK}/setup.py"
    srun \
        --gres=gpu --cpus-per-task=8 --mem=32G --time=06:00:00 \
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