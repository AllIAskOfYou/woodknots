#!/bin/bash
#SBATCH --job-name=seg-ensemble_pairs
#SBATCH --output=slurm_logs/seg-ensemble_pairs-%j.out
#SBATCH --error=slurm_logs/seg-ensemble_pairs-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1

# Two-model ensembles of two configs, paired by seed index
# (seg/ensemble_pairs.py). Reuses the shared probability cache, so cached
# members need no inference.
#
#   sbatch seg/ensemble_pairs.sbatch                                  # exp_reference_i + exp_union_i, i = 0..4
#   sbatch seg/ensemble_pairs.sbatch --seeds 3                        # only seeds 0..2
#
# All arguments are passed through to ensemble_pairs.py.

set -e
set -u

# Quote each pass-through argument so it survives the `bash -c` inside enroot.
EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
    EXTRA_ARGS="$(printf '%q ' "$@")"
fi

run_cluster() {
    # Same container setup as scripts/run.sbatch. Paths are relative to the
    # project root, so submit from there.
    local inner_cmd="
        export RANK=-1
        export LOCAL_RANK=-1
        export WORLD_SIZE=1
        export PYTHONUNBUFFERED=1
                unset MASTER_ADDR MASTER_PORT
        uv run seg/ensemble_pairs.py ${EXTRA_ARGS}
    "
    local container_name="jobcontainer_seg_ensemble_pairs_${SLURM_JOB_ID}_$(hostname)"

    if enroot list | grep -qx "${container_name}"; then
        echo "Found leftover container '${container_name}', removing it..."
        enroot remove -f "${container_name}"
    fi

    # Double quotes: expand container_name now, while it is still in scope.
    trap "enroot remove -f '${container_name}' 2>/dev/null || true" EXIT

    enroot create --name "${container_name}" ./containers/base.sqsh
    enroot start \
        --mount "${PWD}:/workspace" \
        --rw \
        "${container_name}" \
        bash -c "cd /workspace && ${inner_cmd}"
}

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Extra args: ${EXTRA_ARGS}"
echo "========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_cluster
