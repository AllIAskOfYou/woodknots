#!/bin/bash
#SBATCH --job-name=seg-ensemble_scales
#SBATCH --output=slurm_logs/seg-ensemble_scales-%j.out
#SBATCH --error=slurm_logs/seg-ensemble_scales-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1

# Test-time ensemble of seg runs at different resolutions
# (seg/ensemble_scales.py). No training: each run re-predicts its test folds
# once (cached), then the averaged probabilities are scored.
#
#   sbatch seg/ensemble_scales.sbatch                                  # exp_reference + exp_res_1
#   sbatch seg/ensemble_scales.sbatch --experiments exp_reference exp_res_1 --weights 0.5 0.5
#
# All arguments are passed through to ensemble_scales.py.

set -e
set -u

# Quote each pass-through argument so it survives the `bash -c` inside enroot.
EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
    EXTRA_ARGS="$(printf '%q ' "$@")"
fi
if [[ " ${EXTRA_ARGS} " != *" -o "* && " ${EXTRA_ARGS} " != *" --output "* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} -o results/ensembles/seg/ensemble_scales.json"
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
        uv run seg/ensemble_scales.py ${EXTRA_ARGS}
    "
    local container_name="jobcontainer_seg_ensemble_scales_${SLURM_JOB_ID}_$(hostname)"

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
