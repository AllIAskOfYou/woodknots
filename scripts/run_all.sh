#!/bin/bash
set -e
set -u

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <framework> [<experiment>] [--env=local|cluster]"
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

if [[ -n "${EXPERIMENT}" ]]; then
    if [[ ! -f "${FRAMEWORK}/conf/experiment/${EXPERIMENT}.yaml" ]]; then
        echo "Experiment '${EXPERIMENT}' not found in ${FRAMEWORK}/conf/experiment/"
        exit 1
    fi
    EXPERIMENTS=("${EXPERIMENT}")
else
    EXPERIMENTS=($(ls "${FRAMEWORK}/conf/experiment" | sed 's/\.yaml$//'))
fi

echo "Found ${#EXPERIMENTS[@]} experiments for framework=${FRAMEWORK}: ${EXPERIMENTS[*]}"

for exp_name in "${EXPERIMENTS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        ./scripts/run.sh "${FRAMEWORK}" "${exp_name}" "${fold}" --env=${ENV}
    done
done