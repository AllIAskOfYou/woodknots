#!/bin/bash
#SBATCH --job-name=yolo-last_vs_best
#SBATCH --output=slurm_logs/yolo-last_vs_best-%j.out
#SBATCH --error=slurm_logs/yolo-last_vs_best-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1

# Score a YOLO run's last-epoch weights against the fitness-selected ones
# (yolo/last_vs_best.py). No training: it re-predicts each fold's test split
# from the existing train/weights/last.pt, so one experiment is five short
# inference passes plus the scoring.
#
#   sbatch yolo/last_vs_best.sbatch                        # exp_reference
#   sbatch yolo/last_vs_best.sbatch exp_freeze_0
#   sbatch yolo/last_vs_best.sbatch exp_freeze_0 --force --folds 3
#
# Anything after the experiment is passed through to last_vs_best.py. Results
# are written to results/yolo/<experiment>/last_vs_best.json unless -o is given.

set -e
set -u

EXPERIMENT="${1:-exp_reference}"
shift || true

# Quote each pass-through argument so it survives the `bash -c` inside enroot.
EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
    EXTRA_ARGS="$(printf '%q ' "$@")"
fi
if [[ " ${EXTRA_ARGS} " != *" -o "* && " ${EXTRA_ARGS} " != *" --output "* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} -o results/yolo/${EXPERIMENT}/last_vs_best.json"
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
        uv run yolo/last_vs_best.py --experiment ${EXPERIMENT} ${EXTRA_ARGS}
    "
    local container_name="jobcontainer_last_vs_best_${SLURM_JOB_ID}_$(hostname)"

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
echo "Experiment: ${EXPERIMENT}"
echo "Extra args: ${EXTRA_ARGS}"
echo "========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_cluster
