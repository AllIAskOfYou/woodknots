#!/bin/bash
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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
    # Create a container and run the training script inside it. The
    # distributed-training env vars are set as part of inner_cmd (executed
    # inside the container via enroot) rather than exported in this outer
    # shell, since enroot doesn't reliably pass the host shell's environment
    # through into the container.
    local inner_cmd="
        export RANK=-1
        export LOCAL_RANK=-1
        export WORLD_SIZE=1
        export PYTHONUNBUFFERED=1
        unset MASTER_ADDR MASTER_PORT
        uv run ${FRAMEWORK}/run.py experiment=${EXPERIMENT} experiment.fold=${FOLD}
    "
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

# Print GPU info
echo 'Basic GPU Information:'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo 'Full GPU Info:'
nvidia-smi

run_cluster