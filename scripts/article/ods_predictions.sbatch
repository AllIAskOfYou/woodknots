#!/bin/bash
#SBATCH --job-name=seg-ods_predictions
#SBATCH --output=slurm_logs/seg-ods_predictions-%j.out
#SBATCH --error=slurm_logs/seg-ods_predictions-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1

# Qualitative out-of-distribution check of the seg pair ensemble on data/ods
# (scripts/article/ods_predictions.py): predicts the ODS images and draws the
# annotations and predictions. Submit from the project root.
#
#   sbatch scripts/article/ods_predictions.sbatch                      # run 0, split 0 (tests on partition 1)
#   sbatch scripts/article/ods_predictions.sbatch --run 2 --fold 3 --threshold 0.6
#
# All arguments are passed through to ods_predictions.py.

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
        uv run scripts/article/ods_predictions.py ${EXTRA_ARGS}
    "
    local container_name="jobcontainer_seg_ods_predictions_${SLURM_JOB_ID}_$(hostname)"

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
