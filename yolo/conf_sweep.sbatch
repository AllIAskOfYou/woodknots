#!/bin/bash
#SBATCH --job-name=yolo-conf_sweep
#SBATCH --output=slurm_logs/yolo-conf_sweep-%j.out
#SBATCH --error=slurm_logs/yolo-conf_sweep-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=ixb7
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:BLACKWELL|GPU_GEN:BLACKWELL_ULTRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1

# Sweep YOLO's detection-confidence threshold on already-trained runs
# (yolo/conf_sweep.py). No training: it only re-predicts each fold's test split
# from the existing best.pt, so it is short.
#
#   sbatch yolo/conf_sweep.sbatch                              # exp_reference_0, default confs
#   sbatch yolo/conf_sweep.sbatch exp_reference_0 --conf 0.05 0.1 0.25
#
# Anything after the experiment is passed through to conf_sweep.py. Results
# are written to results/yolo/<experiment>/conf_sweep.json unless -o is given.

set -e
set -u

EXPERIMENT="${1:-exp_reference_0}"
shift || true

# Quote each pass-through argument so it survives the `bash -c` inside enroot.
EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
    EXTRA_ARGS="$(printf '%q ' "$@")"
fi
if [[ " ${EXTRA_ARGS} " != *" -o "* && " ${EXTRA_ARGS} " != *" --output "* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} -o results/yolo/${EXPERIMENT}/conf_sweep.json"
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
        uv run yolo/conf_sweep.py --experiment ${EXPERIMENT} ${EXTRA_ARGS}
    "
    local container_name="jobcontainer_conf_sweep_${SLURM_JOB_ID}_$(hostname)"

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
