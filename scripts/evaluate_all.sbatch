#!/bin/bash
# com SBATCH --gres=gpu:1
# com SBATCH --partition=frida
# com SBATCH --constraint=GPU_GEN:AMPERE|GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1


set -e
set -u

# Check positional arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <framework> <space-separated experiment names>"
    exit 1
fi

FRAMEWORK="$1"
shift

EXPERIMENTS="$1"
shift

run_cluster() {
    # Everything runs in ONE uv run invocation (${FRAMEWORK}/evaluate_all.py
    # loops all experiments/folds inside a single Python process), so the
    # per-process import/startup cost is paid once for the whole batch
    # instead of once per fold.
    # Thread pools default to the node's core count, not the CPUs this job
    # was granted; on a shared node that oversubscribes them many times over.
    local threads="${SLURM_CPUS_PER_TASK:-1}"
    local inner_cmd="export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}; uv run ${FRAMEWORK}/evaluate_all.py '${EXPERIMENTS}'"
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
