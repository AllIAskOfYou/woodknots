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

# A run is finished once predict() has touched .pred_complete (training
# alone leaves only .train_complete, and run.sh then just predicts).
# On the cluster, runs already queued or running are skipped too, so
# re-running this script never submits a run twice.
QUEUED=""
if [[ "${ENV}" == "cluster" ]]; then
    QUEUED="$(squeue -h -u "${USER}" -o "%j" 2>/dev/null || true)"
fi

launched=0
done_runs=0
queued_runs=0
for exp_name in "${EXPERIMENTS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        if [[ -f "results/${FRAMEWORK}/${exp_name}/fold_${fold}/.pred_complete" ]]; then
            done_runs=$((done_runs + 1))
            continue
        fi
        if grep -qx "${FRAMEWORK}-${exp_name}-fold_${fold}" <<< "${QUEUED}"; then
            queued_runs=$((queued_runs + 1))
            continue
        fi
        ./scripts/run.sh "${FRAMEWORK}" "${exp_name}" "${fold}" --env=${ENV}
        launched=$((launched + 1))
    done
done

echo "${launched} launched, ${done_runs} already complete, ${queued_runs} already queued"