#!/bin/bash
# drop SBATCH --cpus-per-task=4
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


run_cluster() {
    # Create a container and run the evaluation script inside it.
    local inner_cmd="export PYTHONUNBUFFERED=1; uv run ${FRAMEWORK}/evaluate.py experiment=${EXPERIMENT} experiment.fold=${FOLD}"
    local container_name="jobcontainer_nnunet_${SLURM_JOB_ID}_$(hostname)"

    if enroot list | grep -qx "${container_name}"; then
        echo "Found leftover container '${container_name}', removing it..."
        enroot remove -f "${container_name}"
    fi

    # Double quotes: container_name is `local`, so it must be expanded now,
    # while it is still in scope. With single quotes the trap body is evaluated
    # at script exit, after run_cluster has returned, and dies under `set -u`
    # with "container_name: unbound variable" -- leaving the container behind.
    trap "enroot remove -f '${container_name}' 2>/dev/null || true" EXIT

    enroot create --name "${container_name}" ./containers/base.sqsh
    enroot start \
        --mount "${PWD}:/workspace" \
        --rw \
        "${container_name}" \
        bash -c "cd /workspace && ${inner_cmd}"
}

# Print start time and node info
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================="

run_cluster
